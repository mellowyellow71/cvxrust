//! LinOp operation implementations
//!
//! This module contains implementations for all 22 LinOp operation types.
//! Each operation transforms input tensors according to the operation semantics.

mod arithmetic;
mod leaf;
mod specialized;
mod structural;

use crate::linop::{LinOp, LinOpData, OpType};
use crate::tensor::SparseTensor;
use std::collections::HashMap;

/// Context for processing LinOp trees
#[derive(Clone)]
pub struct ProcessingContext {
    pub id_to_col: HashMap<i64, i64>,
    pub param_to_size: HashMap<i64, i64>,
    pub param_to_col: HashMap<i64, i64>,
    pub var_length: i64,
    pub param_size_plus_one: i64,
}

impl ProcessingContext {
    /// Get the column offset for a variable ID
    pub fn var_col(&self, var_id: i64) -> i64 {
        *self.id_to_col.get(&var_id).unwrap_or(&0)
    }

    /// Get the column offset for a parameter ID
    pub fn param_col(&self, param_id: i64) -> i64 {
        *self.param_to_col.get(&param_id).unwrap_or(&0)
    }

    /// Get the size of a parameter
    pub fn param_size(&self, param_id: i64) -> i64 {
        *self.param_to_size.get(&param_id).unwrap_or(&1)
    }

    /// Get the constant column offset (var_length for the 'b' column)
    pub fn const_col(&self) -> i64 {
        self.var_length
    }

    /// Get the constant parameter offset (last slice for non-parametric)
    pub fn const_param(&self) -> i64 {
        self.param_size_plus_one - 1
    }
}

/// Process a LinOp node and its children recursively
///
/// This is the main entry point for converting a LinOp tree to a SparseTensor.
pub fn process_linop(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    match lin_op.op_type {
        // Leaf nodes
        OpType::Variable => leaf::process_variable(lin_op, ctx),
        OpType::ScalarConst => leaf::process_scalar_const(lin_op, ctx),
        OpType::DenseConst => leaf::process_dense_const(lin_op, ctx),
        OpType::SparseConst => leaf::process_sparse_const(lin_op, ctx),
        OpType::Param => leaf::process_param(lin_op, ctx),

        // Trivial operations
        OpType::Sum => process_sum(lin_op, ctx),
        OpType::Neg => arithmetic::process_neg(lin_op, ctx),
        OpType::Reshape => process_reshape(lin_op, ctx),

        // Arithmetic operations
        OpType::Mul => arithmetic::process_mul(lin_op, ctx),
        OpType::Rmul => arithmetic::process_rmul(lin_op, ctx),
        OpType::MulElem => arithmetic::process_mul_elem(lin_op, ctx),
        OpType::Div => arithmetic::process_div(lin_op, ctx),

        // Structural operations
        OpType::Index => structural::process_index(lin_op, ctx),
        OpType::Transpose => structural::process_transpose(lin_op, ctx),
        OpType::Promote => structural::process_promote(lin_op, ctx),
        OpType::BroadcastTo => structural::process_broadcast_to(lin_op, ctx),
        OpType::Hstack => structural::process_hstack(lin_op, ctx),
        OpType::Vstack => structural::process_vstack(lin_op, ctx),
        OpType::Concatenate => structural::process_concatenate(lin_op, ctx),

        // Specialized operations
        OpType::SumEntries => specialized::process_sum_entries(lin_op, ctx),
        OpType::Trace => specialized::process_trace(lin_op, ctx),
        OpType::DiagVec => specialized::process_diag_vec(lin_op, ctx),
        OpType::DiagMat => specialized::process_diag_mat(lin_op, ctx),
        OpType::UpperTri => specialized::process_upper_tri(lin_op, ctx),
        OpType::Conv => specialized::process_conv(lin_op, ctx),
        OpType::KronR => specialized::process_kron_r(lin_op, ctx),
        OpType::KronL => specialized::process_kron_l(lin_op, ctx),

        // No-op
        OpType::NoOp => SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1)),
    }
}

/// Count the exact number of non-zeros a LinOp tree will produce.
///
/// This is the "structure pass" of the two-pass build: it walks the tree
/// doing the same dispatch as `process_linop` but only counts entries,
/// without allocating tensors or computing values.  The count is used to
/// pre-allocate the output buffer with exact capacity.
pub fn count_nnz(lin_op: &LinOp, _ctx: &ProcessingContext) -> usize {
    match lin_op.op_type {
        // — Leaf nodes: exact counts —
        OpType::Variable => lin_op.size(),
        OpType::ScalarConst => match &lin_op.data {
            LinOpData::Float(v) if *v == 0.0 => 0,
            LinOpData::Int(v) if *v == 0 => 0,
            _ => 1,
        },
        OpType::DenseConst => match &lin_op.data {
            LinOpData::DenseArray { data, .. } => data.iter().filter(|&&x| x != 0.0).count(),
            _ => lin_op.size(),
        },
        OpType::SparseConst => match &lin_op.data {
            LinOpData::SparseArray { data, .. } => data.iter().filter(|&&x| x != 0.0).count(),
            _ => lin_op.size(),
        },
        OpType::Param => lin_op.size(),

        // — Pass-through ops: output nnz == arg nnz —
        OpType::Neg | OpType::Reshape | OpType::Div | OpType::Transpose => {
            lin_op.args.first().map_or(0, |a| count_nnz(a, _ctx))
        }

        // — Aggregation ops: nnz preserved (rows remapped, not removed) —
        OpType::SumEntries | OpType::Trace | OpType::DiagVec
        | OpType::DiagMat | OpType::UpperTri => {
            lin_op.args.first().map_or(0, |a| count_nnz(a, _ctx))
        }

        // — Combining ops: sum of children —
        OpType::Sum | OpType::Hstack | OpType::Vstack | OpType::Concatenate => {
            lin_op.args.iter().map(|a| count_nnz(a, _ctx)).sum()
        }

        // — Index: upper bound = arg nnz (selecting rows can only reduce) —
        OpType::Index => lin_op.args.first().map_or(0, |a| count_nnz(a, _ctx)),

        // — Promote / BroadcastTo: nnz multiplied by broadcast factor —
        OpType::Promote | OpType::BroadcastTo => {
            let arg_nnz = lin_op.args.first().map_or(0, |a| count_nnz(a, _ctx));
            let arg_size = lin_op.args.first().map_or(1, |a| a.size().max(1));
            let output_size = lin_op.size();
            // broadcast factor = output_size / arg_size
            arg_nnz * (output_size / arg_size).max(1)
        }

        // — Mul / Rmul: upper bound based on data nnz × num_blocks —
        // NOTE: the Mul(Const, Variable) fast path in arithmetic.rs emits
        // exactly data_nnz * num_blocks entries — keep these in sync.
        OpType::Mul | OpType::Rmul => {
            let num_blocks = lin_op.args.first()
                .map_or(1, |a| a.shape.get(1).copied().unwrap_or(1));
            match &lin_op.data {
                // Scalar * tensor scales entries in place: nnz == arg nnz
                LinOpData::LinOpRef(ref inner) if inner.op_type == OpType::ScalarConst => {
                    lin_op.args.first().map_or(0, |a| count_nnz(a, _ctx))
                }
                LinOpData::LinOpRef(ref inner) => count_nnz(inner, _ctx) * num_blocks,
                _ => lin_op.size() * num_blocks,
            }
        }

        // — MulElem: upper bound = min(arg_nnz, data_nnz * broadcast) —
        OpType::MulElem => {
            let arg_nnz = lin_op.args.first().map_or(0, |a| count_nnz(a, _ctx));
            let data_nnz = match &lin_op.data {
                LinOpData::LinOpRef(ref inner) => count_nnz(inner, _ctx),
                _ => arg_nnz,
            };
            // Elementwise mul can't produce more entries than arg has
            arg_nnz.min(data_nnz * lin_op.size() / lin_op.args.first().map_or(1, |a| a.size().max(1)))
                .max(arg_nnz) // safety: never less than 0
        }

        // — Conv: product of data size and arg nnz —
        OpType::Conv => {
            let data_size = match &lin_op.data {
                LinOpData::LinOpRef(ref inner) => inner.size(),
                _ => 1,
            };
            let arg_nnz = lin_op.args.first().map_or(1, |a| count_nnz(a, _ctx));
            data_size * arg_nnz
        }

        // — Kron: product of data size and arg nnz —
        OpType::KronR | OpType::KronL => {
            let data_size = match &lin_op.data {
                LinOpData::LinOpRef(ref inner) => inner.size(),
                _ => 1,
            };
            let arg_nnz = lin_op.args.first().map_or(1, |a| count_nnz(a, _ctx));
            data_size * arg_nnz
        }

        OpType::NoOp => 0,
    }
}

/// Sum operation - accumulates results from all args (NOOP for single arg)
fn process_sum(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Process all arguments and combine
    let mut result = process_linop(&lin_op.args[0], ctx);
    for arg in &lin_op.args[1..] {
        let arg_tensor = process_linop(arg, ctx);
        result.extend(arg_tensor);
    }
    result
}

/// Reshape operation - just passes through since we use COO format
fn process_reshape(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Reshape is a NOOP in COO format - the row indices already encode position
    process_linop(&lin_op.args[0], ctx)
}
