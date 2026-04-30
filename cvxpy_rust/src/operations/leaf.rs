//! Leaf node operations: variable, const, param.
//!
//! Each handler exposes two entry points:
//!
//! * `process_*_block(...) -> NodeValue` — produces a typed Block representation
//!   when one fits naturally (Identity for Variable, Dense column-vector for
//!   DenseConst), and falls back to `Block::Coo` for leaves whose structure
//!   doesn't map cleanly onto a typed block (per-row parameter slots in Param,
//!   F-order flat-index encoding in SparseConst).
//! * `process_*(...) -> SparseTensor` — the legacy flat path, now implemented
//!   as `process_*_block(...).into_coo()`. Output is byte-identical to the
//!   pre-PR-2 version.

use std::sync::Arc;

use super::ProcessingContext;
use crate::block::{Block, BlockEntry, DenseF, NodeValue};
use crate::linop::{LinOp, LinOpData};
#[cfg(test)]
use crate::tensor::CONSTANT_ID;
use crate::tensor::SparseTensor;

// ---------------------------------------------------------------------------
// Variable
// ---------------------------------------------------------------------------

/// Process a variable node into an `Identity(n)` block placed at the
/// variable's column offset.
pub fn process_variable_block(lin_op: &LinOp, ctx: &ProcessingContext) -> NodeValue {
    let n = lin_op.size();

    let var_id = match &lin_op.data {
        LinOpData::Int(id) => *id,
        _ => panic!("Variable node must have integer data (variable ID)"),
    };

    let col_offset = ctx.var_col(var_id);
    let param_slot = ctx.const_param();
    let var_cols = ctx.var_length as usize + 1;

    NodeValue {
        out_rows: n,
        var_cols,
        blocks: vec![BlockEntry {
            param_slot,
            var_col_offset: col_offset,
            block: Block::Identity { n },
        }],
    }
}

pub fn process_variable(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    process_variable_block(lin_op, ctx).into_coo()
}

// ---------------------------------------------------------------------------
// ScalarConst
// ---------------------------------------------------------------------------

pub fn process_scalar_const_block(lin_op: &LinOp, ctx: &ProcessingContext) -> NodeValue {
    let value = match &lin_op.data {
        LinOpData::Float(v) => *v,
        LinOpData::Int(v) => *v as f64,
        _ => panic!("Scalar const node must have numeric data"),
    };

    let var_cols = ctx.var_length as usize + 1;

    if value == 0.0 {
        return NodeValue::empty(1, var_cols);
    }

    // Single entry at (row=0, col=const_col, param=const_param). A 1x1 Dense
    // matrix at var_col_offset = const_col flattens to exactly that.
    let dense = Arc::new(DenseF::from_col_major(1, 1, Arc::from(vec![value])));
    NodeValue {
        out_rows: 1,
        var_cols,
        blocks: vec![BlockEntry {
            param_slot: ctx.const_param(),
            var_col_offset: ctx.const_col(),
            block: Block::Dense(dense),
        }],
    }
}

pub fn process_scalar_const(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    process_scalar_const_block(lin_op, ctx).into_coo()
}

// ---------------------------------------------------------------------------
// DenseConst
// ---------------------------------------------------------------------------

/// Process a dense constant into a `Dense(n, 1)` column-vector block placed
/// at the constant column.
///
/// CVXPY passes the constant array already in F-order. Treating it as a
/// `DenseF` of shape `(n, 1)` reproduces the legacy single-column emission
/// (every entry at `col = const_col`, `row = flat F-order index`) without
/// allocation, and gives a future Mul handler a real Dense block to gemm
/// against.
pub fn process_dense_const_block(lin_op: &LinOp, ctx: &ProcessingContext) -> NodeValue {
    let data = match &lin_op.data {
        LinOpData::DenseArray { data, .. } => data,
        _ => panic!("Dense const node must have dense array data"),
    };

    let n = lin_op.size();
    let var_cols = ctx.var_length as usize + 1;

    // Empty constant (n == 0) — degenerate but legal.
    if n == 0 {
        return NodeValue::empty(0, var_cols);
    }

    debug_assert_eq!(data.len(), n);

    let dense = Arc::new(DenseF::from_col_major(n, 1, Arc::clone(data)));
    NodeValue {
        out_rows: n,
        var_cols,
        blocks: vec![BlockEntry {
            param_slot: ctx.const_param(),
            var_col_offset: ctx.const_col(),
            block: Block::Dense(dense),
        }],
    }
}

pub fn process_dense_const(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    process_dense_const_block(lin_op, ctx).into_coo()
}

// ---------------------------------------------------------------------------
// SparseConst
// ---------------------------------------------------------------------------

pub fn process_sparse_const_block(lin_op: &LinOp, ctx: &ProcessingContext) -> NodeValue {
    let (data, indices, indptr, shape) = match &lin_op.data {
        LinOpData::SparseArray {
            data,
            indices,
            indptr,
            shape,
        } => (data, indices, indptr, shape),
        _ => panic!("Sparse const node must have sparse array data"),
    };

    let n = lin_op.size();
    let var_cols = ctx.var_length as usize + 1;
    let col_offset = ctx.const_col();
    let param_offset = ctx.const_param();

    let mut tensor = SparseTensor::with_capacity((n, var_cols), data.len());

    let n_cols = indptr.len() - 1;
    let m = shape.0;

    for j in 0..n_cols {
        let start = indptr[j] as usize;
        let end = indptr[j + 1] as usize;
        for idx in start..end {
            let i = indices[idx] as usize;
            let value = data[idx];
            if value != 0.0 {
                let flat_idx = (j * m + i) as i64;
                tensor.push(value, flat_idx, col_offset, param_offset);
            }
        }
    }

    NodeValue::from_coo(tensor)
}

pub fn process_sparse_const(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    process_sparse_const_block(lin_op, ctx).into_coo()
}

// ---------------------------------------------------------------------------
// Param
// ---------------------------------------------------------------------------

pub fn process_param_block(lin_op: &LinOp, ctx: &ProcessingContext) -> NodeValue {
    let n = lin_op.size();
    let var_cols = ctx.var_length as usize + 1;

    let param_id = match &lin_op.data {
        LinOpData::Int(id) => *id,
        _ => panic!("Param node must have integer data (parameter ID)"),
    };

    let param_col_offset = ctx.param_col(param_id);
    let param_size = ctx.param_size(param_id) as usize;
    let col_offset = ctx.const_col();

    let mut tensor = SparseTensor::with_capacity((n, var_cols), n);

    for i in 0..n {
        let param_offset = if param_size == 0 {
            param_col_offset
        } else {
            param_col_offset + (i % param_size) as i64
        };
        tensor.push(1.0, i as i64, col_offset, param_offset);
    }

    NodeValue::from_coo(tensor)
}

pub fn process_param(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    process_param_block(lin_op, ctx).into_coo()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linop::OpType;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn make_ctx() -> ProcessingContext {
        let mut id_to_col = HashMap::new();
        id_to_col.insert(0, 0);
        id_to_col.insert(1, 5);

        let mut param_to_col = HashMap::new();
        param_to_col.insert(0, 0);
        param_to_col.insert(CONSTANT_ID, 2);

        let mut param_to_size = HashMap::new();
        param_to_size.insert(0, 2);
        param_to_size.insert(CONSTANT_ID, 1);

        ProcessingContext {
            id_to_col,
            param_to_col,
            param_to_size,
            var_length: 10,
            param_size_plus_one: 3,
        }
    }

    #[test]
    fn test_variable() {
        let ctx = make_ctx();
        let lin_op = LinOp {
            op_type: OpType::Variable,
            shape: vec![3],
            args: vec![],
            data: LinOpData::Int(0),
        };

        let tensor = process_variable(&lin_op, &ctx);

        assert_eq!(tensor.nnz(), 3);
        assert_eq!(tensor.data, vec![1.0, 1.0, 1.0]);
        assert_eq!(tensor.rows, vec![0, 1, 2]);
        assert_eq!(tensor.cols, vec![0, 1, 2]);
    }

    #[test]
    fn test_variable_block_returns_identity() {
        let ctx = make_ctx();
        let lin_op = LinOp {
            op_type: OpType::Variable,
            shape: vec![3],
            args: vec![],
            data: LinOpData::Int(1), // Variable 1 at column 5
        };

        let nv = process_variable_block(&lin_op, &ctx);
        assert_eq!(nv.out_rows, 3);
        assert_eq!(nv.var_cols, 11);
        assert_eq!(nv.blocks.len(), 1);
        let entry = &nv.blocks[0];
        assert_eq!(entry.var_col_offset, 5);
        assert_eq!(entry.param_slot, 2);
        assert!(matches!(entry.block, Block::Identity { n: 3 }));
    }

    #[test]
    fn test_scalar_const() {
        let ctx = make_ctx();
        let lin_op = LinOp {
            op_type: OpType::ScalarConst,
            shape: vec![],
            args: vec![],
            data: LinOpData::Float(3.14),
        };

        let tensor = process_scalar_const(&lin_op, &ctx);

        assert_eq!(tensor.nnz(), 1);
        assert_eq!(tensor.data, vec![3.14]);
        assert_eq!(tensor.cols[0], 10);
    }

    #[test]
    fn test_scalar_const_zero_is_empty() {
        let ctx = make_ctx();
        let lin_op = LinOp {
            op_type: OpType::ScalarConst,
            shape: vec![],
            args: vec![],
            data: LinOpData::Float(0.0),
        };
        let tensor = process_scalar_const(&lin_op, &ctx);
        assert_eq!(tensor.nnz(), 0);
    }

    #[test]
    fn test_dense_const() {
        let ctx = make_ctx();
        let lin_op = LinOp {
            op_type: OpType::DenseConst,
            shape: vec![3],
            args: vec![],
            data: LinOpData::DenseArray {
                data: Arc::from(vec![1.0, 0.0, 2.0]),
                shape: vec![3],
            },
        };

        let tensor = process_dense_const(&lin_op, &ctx);

        assert_eq!(tensor.nnz(), 2);
        assert_eq!(tensor.data, vec![1.0, 2.0]);
        assert_eq!(tensor.rows, vec![0, 2]);
    }

    #[test]
    fn test_dense_const_3d() {
        let ctx = make_ctx();

        let mut data = vec![0.0; 24];
        data[6] = 42.0;

        let lin_op = LinOp {
            op_type: OpType::DenseConst,
            shape: vec![2, 3, 4],
            args: vec![],
            data: LinOpData::DenseArray {
                data: Arc::from(data),
                shape: vec![2, 3, 4],
            },
        };

        let tensor = process_dense_const(&lin_op, &ctx);

        assert_eq!(tensor.nnz(), 1);
        assert_eq!(tensor.data, vec![42.0]);
        assert_eq!(tensor.rows, vec![6]);
    }

    #[test]
    fn test_dense_const_block_is_dense() {
        let ctx = make_ctx();
        let lin_op = LinOp {
            op_type: OpType::DenseConst,
            shape: vec![3],
            args: vec![],
            data: LinOpData::DenseArray {
                data: Arc::from(vec![1.0, 0.0, 2.0]),
                shape: vec![3],
            },
        };
        let nv = process_dense_const_block(&lin_op, &ctx);
        assert_eq!(nv.blocks.len(), 1);
        assert!(matches!(nv.blocks[0].block, Block::Dense(_)));
        assert_eq!(nv.blocks[0].var_col_offset, 10);
    }

    #[test]
    fn test_sparse_const_round_trips() {
        let ctx = make_ctx();
        // 2x2 CSC matrix:
        //   1 .
        //   . 3
        let lin_op = LinOp {
            op_type: OpType::SparseConst,
            shape: vec![4],
            args: vec![],
            data: LinOpData::SparseArray {
                data: Arc::from(vec![1.0, 3.0]),
                indices: Arc::from(vec![0_i64, 1]),
                indptr: Arc::from(vec![0_i64, 1, 2]),
                shape: (2, 2),
            },
        };
        let tensor = process_sparse_const(&lin_op, &ctx);
        // F-order flat indices: (i=0,j=0) -> 0, (i=1,j=1) -> 3.
        assert_eq!(tensor.nnz(), 2);
        assert_eq!(tensor.data, vec![1.0, 3.0]);
        assert_eq!(tensor.rows, vec![0, 3]);
        assert_eq!(tensor.cols, vec![10, 10]);
    }

    #[test]
    fn test_param_round_trips() {
        let ctx = make_ctx();
        let lin_op = LinOp {
            op_type: OpType::Param,
            shape: vec![4],
            args: vec![],
            data: LinOpData::Int(0), // size 2, col 0
        };
        let tensor = process_param(&lin_op, &ctx);
        assert_eq!(tensor.nnz(), 4);
        assert_eq!(tensor.data, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(tensor.rows, vec![0, 1, 2, 3]);
        // param_offset cycles: 0, 1, 0, 1.
        assert_eq!(tensor.param_offsets, vec![0, 1, 0, 1]);
    }
}
