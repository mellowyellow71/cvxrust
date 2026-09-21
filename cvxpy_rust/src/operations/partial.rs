//! Row-restricted lowering.
//!
//! `diag_mat`, `index` and `trace` read only a few rows of their argument.
//! Lowering the argument in full and then discarding most of it is what made
//! `diag(V @ G @ V.T)` cost a dense 10000 x 4950 intermediate for 100 rows of
//! output. `process_linop_rows` lowers a node restricted to a set of output
//! rows, pushing the restriction through the operations whose rows depend on
//! a computable subset of their argument's rows (elementwise ops, sums,
//! reshapes, permutations, the diagonal selectors themselves, and the
//! non-parametric block-diagonal mul/rmul kernels). Anything else is lowered
//! in full and filtered.

use super::{arithmetic, process_linop, specialized, structural, ProcessingContext};
use crate::linop::{LinOp, LinOpData, OpType};
use crate::tensor::SparseTensor;

/// Above this fraction of an argument's rows a restricted lowering is not
/// worth its bookkeeping; the argument is lowered in full instead.
const PARTIAL_MAX_FRACTION: f64 = 0.5;

/// Lower `arg` for a consumer that reads only `rows` of it (any order, repeats
/// allowed). The result has the argument's full shape with only those rows
/// populated, or is the complete lowering when most rows are read anyway.
pub(crate) fn process_arg_rows(arg: &LinOp, ctx: &ProcessingContext, rows: &[i64]) -> SparseTensor {
    let size = arg.size();
    let wanted = unique_sorted(rows, size);
    if wanted.len() as f64 > PARTIAL_MAX_FRACTION * size as f64 {
        return process_linop(arg, ctx);
    }
    process_linop_rows(arg, ctx, &wanted)
}

/// Lower `lin_op` restricted to `rows` (sorted, unique, within its row range).
fn process_linop_rows(lin_op: &LinOp, ctx: &ProcessingContext, rows: &[i64]) -> SparseTensor {
    let size = lin_op.size();
    let shape = (size, ctx.var_length as usize + 1);
    if rows.is_empty() {
        return SparseTensor::empty(shape);
    }
    if rows.len() >= size {
        return process_linop(lin_op, ctx);
    }
    if lin_op.args.is_empty() {
        return filter_rows(process_linop(lin_op, ctx), rows);
    }

    let arg = &lin_op.args[0];
    match lin_op.op_type {
        OpType::Neg => {
            let mut tensor = process_linop_rows(arg, ctx, rows);
            tensor.negate_in_place();
            tensor.shape = shape;
            tensor
        }
        OpType::Reshape => {
            let mut tensor = process_linop_rows(arg, ctx, rows);
            tensor.shape = shape;
            tensor
        }
        OpType::Sum => {
            let mut result = process_linop_rows(arg, ctx, rows);
            for other in &lin_op.args[1..] {
                result.extend(process_linop_rows(other, ctx, rows));
            }
            result.shape = shape;
            result
        }
        OpType::MulElem => {
            arithmetic::mul_elem_apply(lin_op, process_linop_rows(arg, ctx, rows), ctx)
        }
        OpType::Div => arithmetic::div_apply(lin_op, process_linop_rows(arg, ctx, rows), ctx),
        OpType::Index => {
            let map = structural::index_row_indices(lin_op);
            through_permutation(arg, ctx, rows, &map)
        }
        OpType::Transpose => {
            let map = structural::transpose_row_indices(lin_op);
            through_permutation(arg, ctx, rows, &map)
        }
        OpType::DiagMat => {
            let needed = specialized::diag_mat_arg_rows(lin_op, rows);
            let tensor = process_arg_rows(arg, ctx, &needed);
            filter_rows(specialized::diag_mat_apply(lin_op, tensor, ctx), rows)
        }
        OpType::Trace if arg.shape.len() >= 2 => {
            let needed = specialized::trace_arg_rows(lin_op, rows);
            let tensor = process_arg_rows(arg, ctx, &needed);
            filter_rows(specialized::trace_apply(lin_op, tensor, ctx), rows)
        }
        OpType::Mul => mul_rows(lin_op, ctx, rows),
        OpType::Rmul => rmul_rows(lin_op, ctx, rows),
        _ => filter_rows(process_linop(lin_op, ctx), rows),
    }
}

/// A node whose output row `r` is the argument's row `map[r]`.
fn through_permutation(
    arg: &LinOp,
    ctx: &ProcessingContext,
    rows: &[i64],
    map: &[i64],
) -> SparseTensor {
    let needed: Vec<i64> = rows.iter().map(|&r| map[r as usize]).collect();
    let tensor = process_arg_rows(arg, ctx, &needed);
    // Other output rows may read the same argument rows; keep only `rows`.
    filter_rows(tensor.select_rows(map), rows)
}

/// `kron(I_k, A) @ arg` restricted to output rows: each output block needs
/// its whole argument block.
fn mul_rows(lin_op: &LinOp, ctx: &ProcessingContext, rows: &[i64]) -> SparseTensor {
    let lhs_linop = match &lin_op.data {
        LinOpData::LinOpRef(inner) => inner.as_ref(),
        _ => panic!("Mul operation must have LinOp data"),
    };
    if arithmetic::is_parametric(lhs_linop) {
        return filter_rows(process_linop(lin_op, ctx), rows);
    }
    let lhs_data = arithmetic::get_constant_matrix_data(lhs_linop, Some(ctx));
    let arg = &lin_op.args[0];

    if let Some(var_id) = arithmetic::as_plain_variable(arg) {
        if let Some(result) =
            arithmetic::mul_const_by_variable(&lhs_data, var_id, arg.size(), lin_op, ctx)
        {
            return filter_rows(result, rows);
        }
    }

    let needed = match &lhs_data {
        arithmetic::ConstantMatrix::Scalar(_) => rows.to_vec(),
        arithmetic::ConstantMatrix::DenseColMajor {
            rows: a_rows,
            cols: a_cols,
            ..
        }
        | arithmetic::ConstantMatrix::DenseRowMajor {
            rows: a_rows,
            cols: a_cols,
            ..
        }
        | arithmetic::ConstantMatrix::Sparse {
            rows: a_rows,
            cols: a_cols,
            ..
        } => block_rows(rows, *a_rows, *a_cols),
    };
    let rhs = process_arg_rows(arg, ctx, &needed);
    arithmetic::multiply_block_diagonal(&lhs_data, &rhs, lin_op, ctx, Some(rows))
}

/// `vec(X @ A)` restricted to output rows: output entry (i, j) needs the
/// whole row i of X.
fn rmul_rows(lin_op: &LinOp, ctx: &ProcessingContext, rows: &[i64]) -> SparseTensor {
    let rhs_linop = match &lin_op.data {
        LinOpData::LinOpRef(inner) => inner.as_ref(),
        _ => panic!("Rmul operation must have LinOp data"),
    };
    if arithmetic::is_parametric(rhs_linop) {
        return filter_rows(process_linop(lin_op, ctx), rows);
    }
    let arg = &lin_op.args[0];
    let rhs_data = arithmetic::get_constant_matrix_data_for_rmul(rhs_linop, arg, ctx);

    let needed = match &rhs_data {
        arithmetic::ConstantMatrix::Scalar(_) => rows.to_vec(),
        arithmetic::ConstantMatrix::DenseColMajor { cols: p, .. }
        | arithmetic::ConstantMatrix::DenseRowMajor { cols: p, .. }
        | arithmetic::ConstantMatrix::Sparse { cols: p, .. } => {
            let k = lin_op.size() / p.max(&1);
            if k == 0 {
                return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
            }
            let n = arg.size() / k;
            // Distinct rows i of X, then all of X's columns l: flat i + k * l.
            let mut x_rows: Vec<usize> = rows.iter().map(|&r| r as usize % k).collect();
            x_rows.sort_unstable();
            x_rows.dedup();
            let mut needed = Vec::with_capacity(x_rows.len() * n);
            for l in 0..n {
                for &i in &x_rows {
                    needed.push((i + k * l) as i64);
                }
            }
            needed
        }
    };
    let lhs = process_arg_rows(arg, ctx, &needed);
    arithmetic::multiply_block_diagonal_right(&lhs, &rhs_data, lin_op, ctx, Some(rows))
}

/// Argument rows for the blocks that contain the given output rows, for a
/// block-diagonal operator with `a_rows` output and `a_cols` input rows per
/// block.
fn block_rows(rows: &[i64], a_rows: usize, a_cols: usize) -> Vec<i64> {
    if a_rows == 0 {
        return Vec::new();
    }
    let mut blocks: Vec<usize> = rows.iter().map(|&r| r as usize / a_rows).collect();
    blocks.dedup();
    let mut needed = Vec::with_capacity(blocks.len() * a_cols);
    for block in blocks {
        needed.extend((block * a_cols) as i64..((block + 1) * a_cols) as i64);
    }
    needed
}

/// Keep only the entries whose row is in `rows` (sorted, unique).
pub(crate) fn filter_rows(tensor: SparseTensor, rows: &[i64]) -> SparseTensor {
    let n = tensor.shape.0;
    let mut keep = vec![false; n];
    for &r in rows {
        if (r as usize) < n {
            keep[r as usize] = true;
        }
    }
    let mut result = SparseTensor::with_capacity(tensor.shape, rows.len().min(tensor.nnz()));
    for i in 0..tensor.nnz() {
        let r = tensor.rows[i];
        if (r as usize) < n && keep[r as usize] {
            result.push(tensor.data[i], r, tensor.cols[i], tensor.param_offsets[i]);
        }
    }
    result
}

/// Valid rows of `rows`, sorted and deduplicated.
fn unique_sorted(rows: &[i64], size: usize) -> Vec<i64> {
    let mut wanted: Vec<i64> = rows
        .iter()
        .copied()
        .filter(|&r| r >= 0 && (r as usize) < size)
        .collect();
    wanted.sort_unstable();
    wanted.dedup();
    wanted
}

#[cfg(test)]
mod tests {
    //! Restricted lowering must equal the full lowering filtered to the rows.
    use super::*;
    use crate::linop::SliceData;
    use crate::tensor::CONSTANT_ID;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    fn ctx() -> ProcessingContext {
        let mut id_to_col = HashMap::new();
        id_to_col.insert(1, 0);
        id_to_col.insert(CONSTANT_ID, 12);
        let mut param_to_col = HashMap::new();
        param_to_col.insert(CONSTANT_ID, 0);
        let mut param_to_size = HashMap::new();
        param_to_size.insert(CONSTANT_ID, 1);
        ProcessingContext {
            id_to_col,
            param_to_col,
            param_to_size,
            var_length: 12,
            param_size_plus_one: 1,
            shared: Default::default(),
        }
    }

    fn variable(shape: &[usize]) -> LinOp {
        LinOp {
            op_type: OpType::Variable,
            shape: shape.to_vec(),
            args: vec![],
            data: LinOpData::Int(1),
        }
    }

    fn dense(rows: usize, cols: usize, seed: u64) -> LinOp {
        let mut x = seed;
        let data: Vec<f64> = (0..rows * cols)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 33) % 5) as f64 - 2.0
            })
            .collect();
        LinOp {
            op_type: OpType::DenseConst,
            shape: vec![rows, cols],
            args: vec![],
            data: LinOpData::DenseArray {
                data: Arc::from(data),
                shape: vec![rows, cols],
            },
        }
    }

    fn node(op_type: OpType, shape: &[usize], args: Vec<LinOp>, data: LinOpData) -> LinOp {
        LinOp {
            op_type,
            shape: shape.to_vec(),
            args,
            data,
        }
    }

    fn entries(t: &SparseTensor) -> HashMap<(i64, i64, i64), f64> {
        let mut m = HashMap::new();
        for i in 0..t.nnz() {
            *m.entry((t.rows[i], t.cols[i], t.param_offsets[i]))
                .or_insert(0.0) += t.data[i];
        }
        m.retain(|_, v| *v != 0.0);
        m
    }

    fn check(lin_op: &LinOp, rows: &[i64]) {
        let c = ctx();
        let full = entries(&filter_rows(process_linop(lin_op, &c), rows));
        let restricted = process_linop_rows(lin_op, &c, rows);
        let rows_present: HashSet<i64> = restricted.rows.iter().copied().collect();
        assert!(
            rows_present.iter().all(|r| rows.contains(r)),
            "extra rows emitted"
        );
        let got = entries(&restricted);
        assert_eq!(got.len(), full.len(), "entry count for rows {:?}", rows);
        for (k, v) in &full {
            let g = got.get(k).copied().unwrap_or(f64::NAN);
            assert!((g - v).abs() < 1e-9, "key {:?}: got {} want {}", k, g, v);
        }
    }

    /// diag(V @ X @ W) with X a 3x4 variable, V 4x3, W 4x4: output (4,) via 4x4.
    fn diag_of_matmul() -> LinOp {
        let x = variable(&[3, 4]);
        let v = dense(4, 3, 1);
        let w = dense(4, 4, 2);
        let vx = node(
            OpType::Mul,
            &[4, 4],
            vec![x],
            LinOpData::LinOpRef(Box::new(v)),
        );
        let vxw = node(
            OpType::Rmul,
            &[4, 4],
            vec![vx],
            LinOpData::LinOpRef(Box::new(w)),
        );
        node(OpType::DiagMat, &[4, 1], vec![vxw], LinOpData::Int(0))
    }

    #[test]
    fn diag_of_dense_matmul() {
        let op = diag_of_matmul();
        check(&op, &[0, 2]);
        check(&op, &[3]);
        check(&op, &[0, 1, 2, 3]);
    }

    #[test]
    fn index_over_sum_neg_reshape_and_transpose() {
        let x = variable(&[3, 4]);
        let v = dense(4, 3, 3);
        let vx = node(
            OpType::Mul,
            &[4, 4],
            vec![x],
            LinOpData::LinOpRef(Box::new(v)),
        );
        let t = node(OpType::Transpose, &[4, 4], vec![vx], LinOpData::None);
        let neg = node(OpType::Neg, &[4, 4], vec![t], LinOpData::None);
        let other = node(OpType::Reshape, &[4, 4], vec![affine_16()], LinOpData::None);
        let sum = node(OpType::Sum, &[4, 4], vec![neg, other], LinOpData::None);
        let slices = LinOpData::Slices(vec![
            SliceData {
                start: 1,
                stop: 4,
                step: 2,
            },
            SliceData {
                start: 0,
                stop: 4,
                step: 3,
            },
        ]);
        let index = node(OpType::Index, &[2, 2], vec![sum], slices);
        check(&index, &[0, 3]);
        check(&index, &[1]);
    }

    /// A 16-row affine expression of the variable: W2 @ (V @ X) with W2 4x4.
    fn affine_16() -> LinOp {
        let x = variable(&[3, 4]);
        let v = dense(4, 3, 4);
        let vx = node(
            OpType::Mul,
            &[4, 4],
            vec![x],
            LinOpData::LinOpRef(Box::new(v)),
        );
        let w2 = dense(4, 4, 5);
        node(
            OpType::Mul,
            &[4, 4],
            vec![vx],
            LinOpData::LinOpRef(Box::new(w2)),
        )
    }

    #[test]
    fn trace_of_matmul_rows() {
        let x = variable(&[3, 4]);
        let v = dense(4, 3, 6);
        let w = dense(4, 4, 7);
        let vx = node(
            OpType::Mul,
            &[4, 4],
            vec![x],
            LinOpData::LinOpRef(Box::new(v)),
        );
        let vxw = node(
            OpType::Rmul,
            &[4, 4],
            vec![vx],
            LinOpData::LinOpRef(Box::new(w)),
        );
        let trace = node(OpType::Trace, &[1], vec![vxw], LinOpData::None);
        check(&trace, &[0]);
    }

    #[test]
    fn mul_elem_and_div_rows() {
        let x = variable(&[3, 4]);
        let v = dense(4, 3, 8);
        let vx = node(
            OpType::Mul,
            &[4, 4],
            vec![x],
            LinOpData::LinOpRef(Box::new(v)),
        );
        let m = dense(4, 4, 9);
        let me = node(
            OpType::MulElem,
            &[4, 4],
            vec![vx],
            LinOpData::LinOpRef(Box::new(m)),
        );
        let d = dense(4, 4, 10);
        let div = node(
            OpType::Div,
            &[4, 4],
            vec![me],
            LinOpData::LinOpRef(Box::new(d)),
        );
        check(&div, &[0, 5, 10, 15]);
        check(&div, &[7]);
    }
}
