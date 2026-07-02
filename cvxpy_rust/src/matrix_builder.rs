//! Main build_matrix algorithm
//!
//! This module contains the core algorithm for converting LinOp trees
//! to sparse coefficient matrices, with support for parallel processing.

use rayon::prelude::*;
use std::collections::HashMap;

use crate::linop::LinOp;
use crate::operations::{count_nnz, process_linop, ProcessingContext};
use crate::tensor::{BuildMatrixResult, SparseTensor, CONSTANT_ID};

/// Minimum number of constraints to consider parallel processing
const PARALLEL_MIN_CONSTRAINTS: usize = 4;

/// Minimum estimated work (total non-zeros) to trigger parallel processing
/// This prevents overhead from outweighing benefits for small problems
const PARALLEL_MIN_WORK: usize = 500;

/// Main entry point for building the coefficient matrix
///
/// Takes a list of LinOp trees and produces a sparse matrix in COO format.
/// The output can be directly used to construct a scipy.sparse.csc_array.
pub fn build_matrix_internal(
    lin_ops: &[LinOp],
    param_size_plus_one: i64,
    id_to_col: &HashMap<i64, i64>,
    param_to_size: &HashMap<i64, i64>,
    param_to_col: &HashMap<i64, i64>,
    var_length: i64,
) -> BuildMatrixResult {
    // Create processing context
    let mut full_id_to_col = id_to_col.clone();
    full_id_to_col.insert(CONSTANT_ID, var_length); // Constant column

    let mut full_param_to_col = param_to_col.clone();
    full_param_to_col.insert(CONSTANT_ID, param_size_plus_one - 1); // Constant slice

    let mut full_param_to_size = param_to_size.clone();
    full_param_to_size.insert(CONSTANT_ID, 1);

    let ctx = ProcessingContext {
        id_to_col: full_id_to_col,
        param_to_size: full_param_to_size,
        param_to_col: full_param_to_col,
        var_length,
        param_size_plus_one,
    };

    // Compute row offsets for each constraint
    let row_offsets: Vec<i64> = lin_ops
        .iter()
        .scan(0i64, |offset, lin_op| {
            let current = *offset;
            *offset += lin_op.size() as i64;
            Some(current)
        })
        .collect();

    let total_rows: usize = lin_ops.iter().map(|l| l.size()).sum();

    // Phase 1 (structure pass): count exact non-zeros per constraint
    // for precise pre-allocation and parallelization decisions.
    let per_constraint_nnz: Vec<usize> = lin_ops
        .iter()
        .map(|l| count_nnz(l, &ctx))
        .collect();
    let total_nnz: usize = per_constraint_nnz.iter().sum();

    // Process constraints (parallel or sequential based on count AND work)
    // Only parallelize if we have enough constraints AND enough work
    let should_parallelize =
        lin_ops.len() >= PARALLEL_MIN_CONSTRAINTS && total_nnz >= PARALLEL_MIN_WORK;

    let combined = if should_parallelize {
        process_constraints_parallel(
            lin_ops,
            &row_offsets,
            &ctx,
            total_nnz,
            total_rows,
        )
    } else {
        process_constraints_sequential(
            lin_ops,
            &row_offsets,
            &ctx,
            total_nnz,
            total_rows,
        )
    };

    BuildMatrixResult::from_tensor(combined, param_size_plus_one as usize)
}

/// Process constraints sequentially into a single pre-allocated output buffer.
///
/// Instead of creating N separate SparseTensors and combining them at the end,
/// this extends a single buffer directly, eliminating intermediate allocations
/// and the final combine copy.
fn process_constraints_sequential(
    lin_ops: &[LinOp],
    row_offsets: &[i64],
    ctx: &ProcessingContext,
    estimated_total_nnz: usize,
    total_rows: usize,
) -> SparseTensor {
    let mut combined = SparseTensor::with_capacity(
        (total_rows, ctx.var_length as usize + 1),
        estimated_total_nnz,
    );

    for (lin_op, &row_offset) in lin_ops.iter().zip(row_offsets.iter()) {
        let mut tensor = process_linop(lin_op, ctx);
        tensor.offset_rows_in_place(row_offset);
        combined.extend(tensor);
    }

    combined
}

/// Process constraints in parallel using rayon, then combine into pre-allocated buffer.
fn process_constraints_parallel(
    lin_ops: &[LinOp],
    row_offsets: &[i64],
    ctx: &ProcessingContext,
    estimated_total_nnz: usize,
    total_rows: usize,
) -> SparseTensor {
    // Process in parallel — each thread returns its own tensor
    let tensors: Vec<SparseTensor> = lin_ops
        .par_iter()
        .zip(row_offsets.par_iter())
        .map(|(lin_op, &row_offset)| {
            let mut tensor = process_linop(lin_op, ctx);
            tensor.offset_rows_in_place(row_offset);
            tensor
        })
        .collect();

    // Combine into pre-allocated buffer
    let mut combined = SparseTensor::with_capacity(
        (total_rows, ctx.var_length as usize + 1),
        estimated_total_nnz,
    );
    for tensor in tensors {
        combined.extend(tensor);
    }
    combined
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linop::{LinOpData, OpType};

    fn make_test_ctx() -> (HashMap<i64, i64>, HashMap<i64, i64>, HashMap<i64, i64>) {
        let mut id_to_col = HashMap::new();
        id_to_col.insert(0, 0); // Variable 0 at column 0
        id_to_col.insert(1, 3); // Variable 1 at column 3

        let mut param_to_col = HashMap::new();
        param_to_col.insert(0, 0); // Parameter 0 at slice 0

        let mut param_to_size = HashMap::new();
        param_to_size.insert(0, 2); // Parameter 0 has size 2

        (id_to_col, param_to_col, param_to_size)
    }

    #[test]
    fn test_single_variable() {
        let (id_to_col, param_to_col, param_to_size) = make_test_ctx();

        let lin_op = LinOp {
            op_type: OpType::Variable,
            shape: vec![3],
            args: vec![],
            data: LinOpData::Int(0),
        };

        let result = build_matrix_internal(
            &[lin_op],
            3, // param_size_plus_one
            &id_to_col,
            &param_to_size,
            &param_to_col,
            6, // var_length
        );

        // Should have 3 entries (identity matrix for variable)
        assert_eq!(result.data.len(), 3);
        assert_eq!(result.data, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_negation() {
        let (id_to_col, param_to_col, param_to_size) = make_test_ctx();

        let var_op = LinOp {
            op_type: OpType::Variable,
            shape: vec![2],
            args: vec![],
            data: LinOpData::Int(0),
        };

        let neg_op = LinOp {
            op_type: OpType::Neg,
            shape: vec![2],
            args: vec![var_op],
            data: LinOpData::None,
        };

        let result =
            build_matrix_internal(&[neg_op], 3, &id_to_col, &param_to_size, &param_to_col, 6);

        assert_eq!(result.data.len(), 2);
        assert_eq!(result.data, vec![-1.0, -1.0]);
    }

    #[test]
    fn test_multiple_constraints() {
        let (id_to_col, param_to_col, param_to_size) = make_test_ctx();

        let var_op1 = LinOp {
            op_type: OpType::Variable,
            shape: vec![2],
            args: vec![],
            data: LinOpData::Int(0),
        };

        let var_op2 = LinOp {
            op_type: OpType::Variable,
            shape: vec![3],
            args: vec![],
            data: LinOpData::Int(1),
        };

        let result = build_matrix_internal(
            &[var_op1, var_op2],
            3,
            &id_to_col,
            &param_to_size,
            &param_to_col,
            6,
        );

        // Total 5 entries: 2 for var0, 3 for var1
        assert_eq!(result.data.len(), 5);

        // Check shape: 5 rows (2+3), 7 cols (6 vars + 1 const)
        assert_eq!(result.shape, (5 * 7, 3));
    }

    #[test]
    fn test_scalar_constant() {
        let (id_to_col, param_to_col, param_to_size) = make_test_ctx();

        let const_op = LinOp {
            op_type: OpType::ScalarConst,
            shape: vec![],
            args: vec![],
            data: LinOpData::Float(5.0),
        };

        let result =
            build_matrix_internal(&[const_op], 3, &id_to_col, &param_to_size, &param_to_col, 6);

        assert_eq!(result.data.len(), 1);
        assert_eq!(result.data[0], 5.0);
    }
}
