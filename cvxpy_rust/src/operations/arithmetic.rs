//! Arithmetic operations: neg, mul, rmul, mul_elem, div
//!
//! These operations perform arithmetic transformations on tensors.

// Allow non-snake-case for matrix math variables like row_in_X
#![allow(non_snake_case)]

use super::{process_linop, process_linop_block, ProcessingContext};
use crate::block::{Block, NodeValue};
use crate::linop::{LinOp, LinOpData, OpType};
use crate::tensor::SparseTensor;
use std::sync::Arc;

/// Process negation operation
///
/// Negates all values in the tensor: (A, b) -> (-A, -b)
pub fn process_neg(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    let mut tensor = process_linop(&lin_op.args[0], ctx);
    tensor.negate_in_place();
    tensor
}

/// Process left multiplication: data @ arg
///
/// Multiplies the argument tensor from the left by a constant matrix.
/// The constant matrix is block-diagonalized according to the output shape.
/// When the constant data is parametric (contains parameters), preserves
/// the parametric structure by creating output entries with appropriate param_offsets.
pub fn process_mul(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Get the constant data (lhs)
    let lhs_linop = match &lin_op.data {
        LinOpData::LinOpRef(inner) => inner.as_ref(),
        _ => panic!("Mul operation must have LinOp data"),
    };

    // Fast path: Mul(ConstantMatrix, Variable) — the Jacobian of A @ x is just A.
    // Skip building the identity tensor and doing A @ I; directly remap A's column
    // indices to the variable's column offset. Zero multiplications required.
    if !is_parametric(lhs_linop) {
        if let Some(var_id) = as_plain_variable(&lin_op.args[0]) {
            let lhs_data = get_constant_matrix_data(lhs_linop, Some(ctx));
            return mul_const_by_variable(lhs_data, var_id, lin_op, ctx);
        }
    }

    // Generalised fast path via the Block IR: any RHS subtree that reduces to
    // a single non-parametric `Identity(n)` block at some `var_col_offset`
    // takes the same shortcut — emit A's entries directly with column indices
    // shifted by that offset. Catches `Mul(A, Index(Variable, [a..a+n]))`
    // (the LASSO canonicalisation) and any other Identity-reducing subtree
    // that the legacy `as_plain_variable` doesn't recognise.
    let rhs_block = if !is_parametric(lhs_linop) {
        let block = process_linop_block(&lin_op.args[0], ctx);
        if let Some((var_col_offset, n)) = as_typed_identity(&block, ctx) {
            let lhs_data = get_constant_matrix_data(lhs_linop, Some(ctx));
            return mul_const_by_identity_offset(lhs_data, var_col_offset, n, lin_op, ctx);
        }
        Some(block)
    } else {
        None
    };

    // Process the argument (rhs). Reuse the NodeValue computed above when
    // available — falling through from the typed probe is free.
    let rhs = match rhs_block {
        Some(b) => b.to_coo(),
        None => process_linop(&lin_op.args[0], ctx),
    };

    // Check if the lhs is parametric (contains Param nodes)
    if is_parametric(lhs_linop) {
        // Process the lhs as a tensor to preserve param_offsets
        let lhs_tensor = process_linop(lhs_linop, ctx);
        return multiply_parametric_left(&lhs_tensor, lhs_linop, &rhs, lin_op, ctx);
    }

    // Get constant data tensor
    let lhs_data = get_constant_matrix_data(lhs_linop, Some(ctx));

    // Perform block diagonal multiplication
    multiply_block_diagonal(&lhs_data, &rhs, lin_op, ctx, false)
}

/// Process right multiplication: arg @ data
///
/// Multiplies the argument tensor from the right by a constant matrix.
/// When the constant data is parametric (contains parameters), preserves
/// the parametric structure by creating output entries with appropriate param_offsets.
pub fn process_rmul(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Get the constant data (rhs in mathematical sense)
    let rhs_linop = match &lin_op.data {
        LinOpData::LinOpRef(inner) => inner.as_ref(),
        _ => panic!("Rmul operation must have LinOp data"),
    };

    // Process the argument (lhs in mathematical sense)
    let lhs = process_linop(&lin_op.args[0], ctx);

    // Check if the rhs is parametric (contains Param nodes)
    if is_parametric(rhs_linop) {
        // Process the rhs as a tensor to preserve param_offsets
        let rhs_tensor = process_linop(rhs_linop, ctx);
        return multiply_parametric_right(&lhs, &rhs_tensor, rhs_linop, lin_op, ctx);
    }

    // Get constant data tensor with special handling for RMul 1D arrays
    // For RMul, we need to check if the 1D constant should be treated as
    // a row or column vector based on the argument's shape
    let rhs_data = get_constant_matrix_data_for_rmul(rhs_linop, &lin_op.args[0], ctx);

    // Perform block diagonal multiplication from right
    multiply_block_diagonal_right(&lhs, &rhs_data, lin_op, ctx)
}

/// Get constant matrix data for RMul, with special handling for 1D arrays
///
/// For RMul (X @ A), a 1D array 'a' should be:
/// - Column vector (n, 1) if X has n columns (standard matrix-vector product)
/// - Row vector (1, n) if X has 1 column (broadcast-like behavior)
fn get_constant_matrix_data_for_rmul(
    lin_op: &LinOp,
    arg: &LinOp,
    ctx: &ProcessingContext,
) -> ConstantMatrix {
    let result = get_constant_matrix_data(lin_op, Some(ctx));

    // Only need special handling for 1D arrays
    if lin_op.shape.len() != 1 {
        return result;
    }

    // Get arg_cols: number of columns in the argument
    let arg_cols = if arg.shape.len() == 1 {
        arg.shape[0]
    } else if arg.shape.len() >= 2 {
        arg.shape[1]
    } else {
        1
    };

    // Check if we need to transpose
    // Python logic: if len(lin.data.shape) == 1 and arg_cols != lhs.shape[0]: lhs = lhs.T
    match result {
        ConstantMatrix::DenseRowMajor { data, rows, cols } => {
            // Currently rows=1, cols=n for 1D array treated as row vector
            // If arg_cols != rows (i.e., arg_cols != 1), we need to transpose to (n, 1)
            if arg_cols != rows {
                // Transpose: swap rows and cols, becomes column vector
                // Column vector with 1 column can use DenseColMajor (same memory layout)
                ConstantMatrix::DenseColMajor {
                    data,
                    rows: cols, // Transpose: swap rows and cols
                    cols: rows,
                }
            } else {
                ConstantMatrix::DenseRowMajor { data, rows, cols }
            }
        }
        ConstantMatrix::DenseColMajor { data, rows, cols } => {
            // Handle column-major case similarly
            if arg_cols != rows {
                ConstantMatrix::DenseColMajor {
                    data,
                    rows: cols,
                    cols: rows,
                }
            } else {
                ConstantMatrix::DenseColMajor { data, rows, cols }
            }
        }
        other => other,
    }
}

/// Process elementwise multiplication: arg * data
///
/// Multiplies the argument tensor elementwise by data (which may be parametric).
/// This is equivalent to left multiplication by a diagonal matrix.
/// For scalar constants, the scalar is broadcast to all elements.
///
/// When data contains parameters, we need to properly handle the parametric structure:
/// - Process data as a tensor (not just extract constant values)
/// - Multiply arg entries by data entries, preserving param_offsets from data
pub fn process_mul_elem(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Get the data LinOp
    let data_linop = match &lin_op.data {
        LinOpData::LinOpRef(inner) => inner.as_ref(),
        _ => panic!("MulElem operation must have LinOp data"),
    };

    // Process the argument tensor
    let arg_tensor = process_linop(&lin_op.args[0], ctx);

    // Process the data as a tensor to preserve parametric structure
    let data_tensor = process_linop(data_linop, ctx);

    // Check if data is scalar (single output)
    let is_scalar = data_linop.size() == 1;

    // Build result tensor by multiplying arg entries by corresponding data entries
    // For each arg entry at row r, we need to find all data entries at row r
    // (or at row 0 if scalar) and create result entries with data's param_offset

    // For efficiency, group data entries by row
    let data_size = data_linop.size();
    let mut data_by_row: Vec<Vec<(f64, i64)>> = vec![Vec::new(); data_size];
    for i in 0..data_tensor.nnz() {
        let row = data_tensor.rows[i] as usize;
        if row < data_size {
            // Store (value, param_offset) for each row
            data_by_row[row].push((data_tensor.data[i], data_tensor.param_offsets[i]));
        }
    }

    // Build result tensor
    let mut result = SparseTensor::with_capacity(
        (lin_op.size(), ctx.var_length as usize + 1),
        arg_tensor.nnz() * if is_scalar { data_tensor.nnz() } else { 1 },
    );

    for i in 0..arg_tensor.nnz() {
        let arg_row = arg_tensor.rows[i] as usize;
        let arg_col = arg_tensor.cols[i];
        let arg_val = arg_tensor.data[i];
        let arg_param = arg_tensor.param_offsets[i];

        // Get data entries for this row (or row 0 if scalar)
        let data_row = if is_scalar { 0 } else { arg_row };

        if data_row < data_by_row.len() && !data_by_row[data_row].is_empty() {
            for &(data_val, data_param) in &data_by_row[data_row] {
                // The result param_offset depends on both arg and data params
                // If arg has non-constant param, we need to handle param*param (not supported)
                // For now, assume arg params are constant and use data's param
                let result_param = if arg_param == ctx.const_param() {
                    data_param
                } else if data_param == ctx.const_param() {
                    arg_param
                } else {
                    // Both are parametric - this would need tensor product of params
                    // For now, just use data's param (may not be fully correct)
                    data_param
                };

                result.push(
                    arg_val * data_val,
                    arg_tensor.rows[i],
                    arg_col,
                    result_param,
                );
            }
        }
        // If no data entries for this row, the data is zero for that position
        // and we produce zero output (don't add any entry since 0*anything = 0)
    }

    result
}

/// Process division: arg / data
///
/// Divides the argument tensor elementwise by constant data.
/// This is equivalent to elementwise multiplication by reciprocal.
/// For scalar constants, the scalar is broadcast to all elements.
pub fn process_div(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Get the constant data
    let data_linop = match &lin_op.data {
        LinOpData::LinOpRef(inner) => inner.as_ref(),
        _ => panic!("Div operation must have LinOp data"),
    };

    // Process the argument
    let mut tensor = process_linop(&lin_op.args[0], ctx);

    // Get constant data as flat array (in column-major order)
    let data = get_constant_vector_data(data_linop, Some(ctx));

    // Divide each tensor entry by the corresponding data element
    // For scalars (len=1), broadcast to all elements
    let is_scalar = data.len() == 1;
    for i in 0..tensor.nnz() {
        let row = tensor.rows[i] as usize;
        let data_val = if is_scalar {
            data[0]
        } else if row < data.len() {
            data[row]
        } else {
            1.0 // Default to no-op for out-of-bounds (shouldn't happen)
        };
        if data_val != 0.0 {
            tensor.data[i] /= data_val;
        }
    }

    tensor
}

/// Helper: extract constant matrix data from a LinOp (2D case)
/// For left multiplication, 1D arrays are treated as row vectors (1, n)
/// ctx is optional but required for complex LinOp types (Hstack, Vstack, Transpose, etc.)
fn get_constant_matrix_data(lin_op: &LinOp, ctx: Option<&ProcessingContext>) -> ConstantMatrix {
    use crate::linop::OpType;

    // First check op_type to handle complex constant expressions
    match &lin_op.op_type {
        OpType::DenseConst | OpType::SparseConst | OpType::ScalarConst => {
            // Simple constant types - extract data directly from lin_op.data
            extract_matrix_from_data(lin_op)
        }
        OpType::Param => {
            // For parameters, we need to use the actual processing context
            // to properly extract the parameter values into a matrix
            if let Some(ctx) = ctx {
                extract_matrix_from_linop_with_ctx(lin_op, ctx)
            } else {
                panic!("Cannot extract constant matrix from Param without context")
            }
        }
        _ => {
            // Complex expression type (Hstack, Vstack, Transpose, etc.)
            // Process recursively using the canonicalization machinery
            if let Some(ctx) = ctx {
                extract_matrix_from_linop(lin_op, ctx)
            } else {
                // Fallback: try to extract from data directly
                extract_matrix_from_data(lin_op)
            }
        }
    }
}

/// Extract matrix data directly from LinOp.data field
/// OPTIMIZATION: Keep data in column-major format to avoid conversions
fn extract_matrix_from_data(lin_op: &LinOp) -> ConstantMatrix {
    match &lin_op.data {
        LinOpData::Float(v) => ConstantMatrix::Scalar(*v),
        LinOpData::Int(v) => ConstantMatrix::Scalar(*v as f64),
        LinOpData::DenseArray { data, shape } => {
            if shape.is_empty() || shape.iter().product::<usize>() == 1 {
                ConstantMatrix::Scalar(data[0])
            } else if shape.len() == 1 {
                // 1D array treated as row vector (1, n) for left multiplication
                // This matches Python: lin_op_shape = [1, lin_op.shape[0]]
                // Store as row-major since it's effectively 1D
                ConstantMatrix::DenseRowMajor {
                    data: Arc::clone(data),
                    rows: 1,
                    cols: shape[0],
                }
            } else {
                // 2D array: data is already in F-order (column-major) from extract_dense_array
                // OPTIMIZATION: Keep it in column-major format - no conversion needed!
                let nrows = shape[0];
                let ncols = shape[1];
                ConstantMatrix::DenseColMajor {
                    data: Arc::clone(data),
                    rows: nrows,
                    cols: ncols,
                }
            }
        }
        LinOpData::SparseArray {
            data,
            indices,
            indptr,
            shape,
        } => ConstantMatrix::Sparse {
            values: Arc::clone(data),
            row_indices: Arc::clone(indices),
            col_indptr: Arc::clone(indptr),
            rows: shape.0,
            cols: shape.1,
        },
        _ => panic!(
            "Cannot extract constant matrix from data: {:?}",
            lin_op.op_type
        ),
    }
}

/// Extract matrix data by processing a complex LinOp
/// This creates a minimal context for constant-only processing
/// OPTIMIZATION: Keep data in column-major format to avoid conversions
fn extract_matrix_from_linop(lin_op: &LinOp, _ctx: &ProcessingContext) -> ConstantMatrix {
    use crate::tensor::CONSTANT_ID;
    use std::collections::HashMap;

    // Create a minimal context for processing constant expressions
    // No variables, just the constant column
    let mut param_to_col = HashMap::new();
    param_to_col.insert(CONSTANT_ID, 0);
    let mut param_to_size = HashMap::new();
    param_to_size.insert(CONSTANT_ID, 1);

    let const_ctx = ProcessingContext {
        id_to_col: HashMap::new(),
        param_to_col,
        param_to_size,
        var_length: 0,
        param_size_plus_one: 1,
    };

    // Process the LinOp to get a tensor
    let tensor = process_linop(lin_op, &const_ctx);

    // Extract constant values from tensor
    // For constant expressions, entries are in the constant column (col 0)
    // Data is already in column-major order from tensor processing
    let size = lin_op.size();
    let mut dense_data = vec![0.0; size];

    for i in 0..tensor.nnz() {
        let row = tensor.rows[i] as usize;
        if row < size {
            dense_data[row] += tensor.data[i];
        }
    }

    // Determine matrix shape
    let (rows, cols) = if lin_op.shape.is_empty() {
        (1, 1)
    } else if lin_op.shape.len() == 1 {
        // 1D array treated as row vector for left multiplication
        (1, lin_op.shape[0])
    } else {
        (lin_op.shape[0], lin_op.shape[1])
    };

    // OPTIMIZATION: Keep data in column-major format - no conversion needed!
    if rows == 1 && cols == 1 {
        ConstantMatrix::Scalar(dense_data[0])
    } else if rows == 1 {
        // Row vector - store as row-major (same as column-major for 1 row)
        ConstantMatrix::DenseRowMajor {
            data: Arc::from(dense_data),
            rows,
            cols,
        }
    } else {
        // 2D matrix - keep in column-major format
        ConstantMatrix::DenseColMajor {
            data: Arc::from(dense_data),
            rows,
            cols,
        }
    }
}

/// Extract matrix data by processing a LinOp using the actual processing context
/// This is used for parameters and other cases where we need the full context
/// OPTIMIZATION: Keep data in column-major format to avoid conversions
fn extract_matrix_from_linop_with_ctx(lin_op: &LinOp, ctx: &ProcessingContext) -> ConstantMatrix {
    // Process the LinOp to get a tensor using the actual context
    let tensor = process_linop(lin_op, ctx);

    // Extract values from tensor
    // For parameters, each element maps to a different parameter slice
    // We need to collect values indexed by row
    // Data is already in column-major order from tensor processing
    let size = lin_op.size();
    let mut dense_data = vec![0.0; size];

    for i in 0..tensor.nnz() {
        let row = tensor.rows[i] as usize;
        if row < size {
            // For now, just collect the values
            // This works for parameters where the data is scalar 1.0 multiplied by param slice
            dense_data[row] += tensor.data[i];
        }
    }

    // Determine matrix shape
    let (rows, cols) = if lin_op.shape.is_empty() {
        (1, 1)
    } else if lin_op.shape.len() == 1 {
        // 1D array treated as row vector for left multiplication
        (1, lin_op.shape[0])
    } else {
        (lin_op.shape[0], lin_op.shape[1])
    };

    // OPTIMIZATION: Keep data in column-major format - no conversion needed!
    if rows == 1 && cols == 1 {
        ConstantMatrix::Scalar(dense_data[0])
    } else if rows == 1 {
        // Row vector - store as row-major (same as column-major for 1 row)
        ConstantMatrix::DenseRowMajor {
            data: Arc::from(dense_data),
            rows,
            cols,
        }
    } else {
        // 2D matrix - keep in column-major format
        ConstantMatrix::DenseColMajor {
            data: Arc::from(dense_data),
            rows,
            cols,
        }
    }
}

/// Helper: extract constant vector data from a LinOp (flattened in column-major order)
///
/// CVXPY uses column-major (Fortran) order for flattening tensors, but NumPy stores
/// arrays in row-major (C) order. This function converts row-major NumPy data to
/// column-major order for elementwise operations.
///
/// For complex constant expressions (like Mul), this function recursively processes
/// them to extract the constant values.
fn get_constant_vector_data(lin_op: &LinOp, ctx: Option<&ProcessingContext>) -> Vec<f64> {
    use crate::linop::OpType;

    // First check op_type to handle complex constant expressions
    match &lin_op.op_type {
        OpType::DenseConst | OpType::SparseConst | OpType::ScalarConst | OpType::Param => {
            // Simple constant types - extract data directly
            match &lin_op.data {
                LinOpData::Float(v) => vec![*v],
                LinOpData::Int(v) => vec![*v as f64],
                LinOpData::DenseArray { data, .. } => {
                    // Data is already stored in F-order (column-major) from extract_dense_array
                    // Return directly for elementwise operations
                    data.to_vec()
                }
                LinOpData::SparseArray {
                    data,
                    indices,
                    indptr,
                    shape,
                } => {
                    // Convert sparse to dense for elementwise ops
                    // CSC format naturally gives column-major order
                    let n = shape.0 * shape.1;
                    let mut dense = vec![0.0; n];
                    let n_cols = indptr.len() - 1;
                    for j in 0..n_cols {
                        let start = indptr[j] as usize;
                        let end = indptr[j + 1] as usize;
                        for idx in start..end {
                            let i = indices[idx] as usize;
                            let flat_idx = j * shape.0 + i; // Column-major indexing
                            dense[flat_idx] = data[idx];
                        }
                    }
                    dense
                }
                _ => {
                    // Scalar or unhandled data type
                    vec![1.0]
                }
            }
        }
        _ => {
            // Complex expression type (Mul, Neg, etc.) - recursively process
            if let Some(ctx) = ctx {
                let tensor = process_linop(lin_op, ctx);
                // Extract constant values from the tensor
                // For constant expressions, all entries should be in the constant column
                extract_constant_values_from_tensor(&tensor, lin_op.size())
            } else {
                panic!(
                    "Cannot extract constant vector from {:?} without context",
                    lin_op.op_type
                )
            }
        }
    }
}

/// Extract constant values from a tensor result
///
/// For constant expressions, all non-zero entries should map to a flat vector
/// indexed by row. We collect values by row index.
fn extract_constant_values_from_tensor(tensor: &SparseTensor, size: usize) -> Vec<f64> {
    use crate::tensor::CONSTANT_ID;

    let mut result = vec![0.0; size];

    // Iterate through all entries in the tensor
    for i in 0..tensor.nnz() {
        let row = tensor.rows[i] as usize;
        let col = tensor.cols[i];

        // For constant expressions, we expect entries in the constant column (col 0)
        // or in parameter columns. We sum all contributions to each row.
        if row < size {
            // The value contributes to the constant at this row position
            // For constant expressions, this should give us the evaluated constant
            if col == CONSTANT_ID {
                result[row] += tensor.data[i];
            }
        }
    }

    result
}

/// Representation of constant matrix data
#[derive(Debug, Clone)]
enum ConstantMatrix {
    Scalar(f64),
    /// Dense matrix stored in column-major (F-order) format
    /// This avoids costly conversions since CVXPY uses F-order internally
    DenseColMajor {
        data: Arc<[f64]>,
        rows: usize,
        cols: usize,
    },
    /// Dense matrix stored in row-major (C-order) format
    /// Used for 1D arrays which are treated as row vectors
    DenseRowMajor {
        data: Arc<[f64]>,
        rows: usize,
        cols: usize,
    },
    Sparse {
        values: Arc<[f64]>,
        row_indices: Arc<[i64]>,
        col_indptr: Arc<[i64]>,
        rows: usize,
        cols: usize,
    },
}

/// Returns Some(var_id) if this LinOp is a plain variable, possibly wrapped
/// in Reshape or single-arg Sum nodes (which are no-ops in COO format).
fn as_plain_variable(lin_op: &LinOp) -> Option<i64> {
    match lin_op.op_type {
        OpType::Variable => {
            if let LinOpData::Int(id) = lin_op.data {
                Some(id)
            } else {
                None
            }
        }
        // Reshape and single-arg Sum are no-ops in COO format — unwrap and recurse
        OpType::Reshape => {
            if lin_op.args.len() == 1 {
                as_plain_variable(&lin_op.args[0])
            } else {
                None
            }
        }
        OpType::Sum => {
            if lin_op.args.len() == 1 {
                as_plain_variable(&lin_op.args[0])
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Fast path for Mul(ConstantMatrix, Variable).
///
/// The Jacobian of f(x) = A @ x with respect to x is just A. Rather than
/// building an identity tensor for x and doing A @ I (n*m scalar multiplies),
/// we directly remap A's column indices to the variable's column offset.
///
/// Cost: O(nnz(A)) — a single pass over A's data with index arithmetic only.
fn mul_const_by_variable(
    lhs: ConstantMatrix,
    var_id: i64,
    lin_op: &LinOp,
    ctx: &ProcessingContext,
) -> SparseTensor {
    let var_col_offset = ctx.var_col(var_id);
    let n = lin_op.args[0].size();
    mul_const_by_identity_offset(lhs, var_col_offset, n, lin_op, ctx)
}

/// Generalised fast path: `Mul(ConstantMatrix, Identity(n)) at var_col_offset`.
///
/// Equivalent to `mul_const_by_variable` but takes the variable column offset
/// and identity width directly, so it works for any RHS subtree that reduces
/// to a typed `Identity` block — including `Index(Variable, contiguous_slice)`
/// which the LASSO canonicalisation produces.
fn mul_const_by_identity_offset(
    lhs: ConstantMatrix,
    var_col_offset: i64,
    n: usize,
    lin_op: &LinOp,
    ctx: &ProcessingContext,
) -> SparseTensor {
    let output_rows = lin_op.size();
    let param_offset = ctx.const_param();
    let out_shape = (output_rows, ctx.var_length as usize + 1);

    match lhs {
        ConstantMatrix::Scalar(s) => {
            // Scalar * Identity: scaled identity, one entry per variable column.
            let mut result = SparseTensor::with_capacity(out_shape, n);
            if s != 0.0 {
                for i in 0..n {
                    result.push(s, i as i64, var_col_offset + i as i64, param_offset);
                }
            }
            result
        }

        ConstantMatrix::DenseColMajor { data, rows: a_rows, cols: a_cols } => {
            // data is in column-major layout: data[col * a_rows + row] = A[row, col]
            // Map each nonzero A[row, col] → output (row, var_col_offset + col, val).
            //
            // Fully-dense fast path (the common LASSO case where A is a random
            // gaussian matrix): every entry is non-zero, so we can bulk-fill
            // each output array instead of doing per-element push calls.
            // `data` becomes a memcpy; `rows` is the pattern [0..a_rows]
            // repeated a_cols times; `cols` is each output column repeated
            // a_rows times; `param_offsets` is a single value × nnz.
            let total = a_rows * a_cols;
            let nnz_count = data.iter().filter(|&&v| v != 0.0).count();
            if nnz_count == total && total > 0 {
                let mut result_rows: Vec<i64> = Vec::with_capacity(total);
                let mut result_cols: Vec<i64> = Vec::with_capacity(total);
                // Build row index pattern [0, 1, .., a_rows-1] once and reuse.
                let row_pattern: Vec<i64> = (0..a_rows as i64).collect();
                for col in 0..a_cols {
                    result_rows.extend_from_slice(&row_pattern);
                    result_cols
                        .extend(std::iter::repeat(var_col_offset + col as i64).take(a_rows));
                }
                let result_data: Vec<f64> = data.iter().copied().collect();
                let result_params: Vec<i64> = vec![param_offset; total];
                return SparseTensor {
                    shape: out_shape,
                    data: result_data,
                    rows: result_rows,
                    cols: result_cols,
                    param_offsets: result_params,
                };
            }

            // Scalar path: matrix has zeros, skip them per-entry.
            let mut result = SparseTensor::with_capacity(out_shape, nnz_count);
            for col in 0..a_cols {
                let out_col = var_col_offset + col as i64;
                let col_start = col * a_rows;
                for row in 0..a_rows {
                    let val = data[col_start + row];
                    if val != 0.0 {
                        result.push(val, row as i64, out_col, param_offset);
                    }
                }
            }
            result
        }

        ConstantMatrix::DenseRowMajor { data, rows: a_rows, cols: a_cols } => {
            // data is in row-major layout: data[row * a_cols + col] = A[row, col]
            // Used for 1D arrays treated as row vectors (a_rows == 1 typically)
            let nnz_estimate = data.iter().filter(|&&v| v != 0.0).count();
            let mut result = SparseTensor::with_capacity(out_shape, nnz_estimate);
            for row in 0..a_rows {
                for col in 0..a_cols {
                    let val = data[row * a_cols + col];
                    if val != 0.0 {
                        let out_col = var_col_offset + col as i64;
                        result.push(val, row as i64, out_col, param_offset);
                    }
                }
            }
            result
        }

        ConstantMatrix::Sparse { values, row_indices, col_indptr, rows: _, cols: a_cols } => {
            // CSC sparse: iterate columns then nonzeros within each column
            let mut result = SparseTensor::with_capacity(out_shape, values.len());
            for col in 0..a_cols {
                let out_col = var_col_offset + col as i64;
                let start = col_indptr[col] as usize;
                let end = col_indptr[col + 1] as usize;
                for idx in start..end {
                    let val = values[idx];
                    if val != 0.0 {
                        result.push(val, row_indices[idx], out_col, param_offset);
                    }
                }
            }
            result
        }
    }
}

/// Returns `Some((var_col_offset, n))` if the NodeValue is a single
/// non-parametric `Identity(n)` block — the precondition for the diffengine
/// fast path. Defensive: requires `param_slot == const_param()` so we never
/// take the fast path when parametric structure has slipped through.
fn as_typed_identity(nv: &NodeValue, ctx: &ProcessingContext) -> Option<(i64, usize)> {
    if nv.blocks.len() != 1 {
        return None;
    }
    let entry = &nv.blocks[0];
    if entry.param_slot != ctx.const_param() {
        return None;
    }
    match &entry.block {
        Block::Identity { n } => Some((entry.var_col_offset, *n)),
        _ => None,
    }
}

/// Block diagonal multiplication from left: kron(I, A) @ tensor
fn multiply_block_diagonal(
    lhs: &ConstantMatrix,
    rhs: &SparseTensor,
    lin_op: &LinOp,
    ctx: &ProcessingContext,
    _transpose_lhs: bool,
) -> SparseTensor {
    let output_rows = lin_op.size();

    match lhs {
        ConstantMatrix::Scalar(s) => {
            // Scalar multiplication
            let mut result = rhs.clone();
            result.scale_in_place(*s);
            result.shape = (output_rows, ctx.var_length as usize + 1);
            result
        }
        ConstantMatrix::DenseColMajor { data, rows, cols } => {
            // Dense block diagonal multiplication with column-major data
            multiply_dense_block_diagonal_colmajor(data, *rows, *cols, rhs, output_rows, ctx)
        }
        ConstantMatrix::DenseRowMajor { data, rows, cols } => {
            // Dense block diagonal multiplication with row-major data
            multiply_dense_block_diagonal_rowmajor(data, *rows, *cols, rhs, output_rows, ctx)
        }
        ConstantMatrix::Sparse {
            values,
            row_indices,
            col_indptr,
            rows,
            cols,
        } => {
            // Sparse block diagonal multiplication
            multiply_sparse_block_diagonal(
                values,
                row_indices,
                col_indptr,
                *rows,
                *cols,
                rhs,
                output_rows,
                ctx,
            )
        }
    }
}

/// Dense block diagonal multiplication with column-major data: kron(I_k, A) @ tensor
/// Direct element-by-element processing with pre-allocated output
#[inline]
fn multiply_dense_block_diagonal_colmajor(
    data: &[f64],
    a_rows: usize,
    a_cols: usize,
    rhs: &SparseTensor,
    output_rows: usize,
    ctx: &ProcessingContext,
) -> SparseTensor {
    // Guard against edge cases
    if a_cols == 0 || a_rows == 0 || rhs.nnz() == 0 {
        return SparseTensor::empty((output_rows, ctx.var_length as usize + 1));
    }

    // Pre-allocate with exact capacity
    let exact_nnz = rhs.nnz() * a_rows;
    let mut rows = Vec::with_capacity(exact_nnz);
    let mut cols = Vec::with_capacity(exact_nnz);
    let mut vals = Vec::with_capacity(exact_nnz);
    let mut params = Vec::with_capacity(exact_nnz);

    // Direct processing with pre-computed column starts
    for idx in 0..rhs.nnz() {
        let rhs_row = rhs.rows[idx] as usize;
        let rhs_col = rhs.cols[idx];
        let rhs_val = rhs.data[idx];
        let rhs_param = rhs.param_offsets[idx];

        let block = rhs_row / a_cols;
        let col_in_block = rhs_row % a_cols;
        let row_offset = block * a_rows;
        let col_start = col_in_block * a_rows;

        // Emit scaled column of A - iterate using slice for better cache behavior
        let a_col = &data[col_start..col_start + a_rows];
        for (i, &a_val) in a_col.iter().enumerate() {
            if a_val != 0.0 {
                let new_row = row_offset + i;
                if new_row < output_rows {
                    rows.push(new_row as i64);
                    cols.push(rhs_col);
                    vals.push(a_val * rhs_val);
                    params.push(rhs_param);
                }
            }
        }
    }

    SparseTensor {
        shape: (output_rows, ctx.var_length as usize + 1),
        rows,
        cols,
        data: vals,
        param_offsets: params,
    }
}

/// Dense block diagonal multiplication with row-major data: kron(I_k, A) @ tensor
/// Used for 1D arrays (row vectors) - typically small, so keep simple implementation
#[inline]
fn multiply_dense_block_diagonal_rowmajor(
    data: &[f64],
    a_rows: usize,
    a_cols: usize,
    rhs: &SparseTensor,
    output_rows: usize,
    ctx: &ProcessingContext,
) -> SparseTensor {
    // Guard against edge cases
    if a_cols == 0 || a_rows == 0 {
        return SparseTensor::empty((output_rows, ctx.var_length as usize + 1));
    }

    // Row-major is only used for 1D arrays (row vectors), which are small
    // Keep simple implementation - not worth faer overhead for 1-row matrices
    let est_nnz = rhs.nnz() * a_rows;
    let mut result =
        SparseTensor::with_capacity((output_rows, ctx.var_length as usize + 1), est_nnz);

    for idx in 0..rhs.nnz() {
        let rhs_row = rhs.rows[idx] as usize;
        let rhs_col = rhs.cols[idx];
        let rhs_val = rhs.data[idx];
        let rhs_param = rhs.param_offsets[idx];

        let block = rhs_row / a_cols;
        let col_in_block = rhs_row % a_cols;

        // Row-major data: element (i, j) at index i * cols + j
        for i in 0..a_rows {
            let a_val = data[i * a_cols + col_in_block];
            if a_val != 0.0 {
                let new_row = (block * a_rows + i) as i64;
                result.push(a_val * rhs_val, new_row, rhs_col, rhs_param);
            }
        }
    }

    result
}

/// Sparse block diagonal multiplication: kron(I_k, A) @ tensor
fn multiply_sparse_block_diagonal(
    values: &[f64],
    row_indices: &[i64],
    col_indptr: &[i64],
    a_rows: usize,
    a_cols: usize,
    rhs: &SparseTensor,
    output_rows: usize,
    ctx: &ProcessingContext,
) -> SparseTensor {
    // Guard against division by zero
    if a_cols == 0 {
        return SparseTensor::empty((output_rows, ctx.var_length as usize + 1));
    }

    // Number of blocks (unused but documents the algorithm)
    let _k = rhs.shape.0 / a_cols;

    // Estimate output nnz
    let est_nnz = rhs.nnz() * values.len() / a_cols.max(1);
    let mut result =
        SparseTensor::with_capacity((output_rows, ctx.var_length as usize + 1), est_nnz);

    // For each entry in rhs, compute its contribution
    for idx in 0..rhs.nnz() {
        let rhs_row = rhs.rows[idx] as usize;
        let rhs_col = rhs.cols[idx];
        let rhs_val = rhs.data[idx];
        let rhs_param = rhs.param_offsets[idx];

        // Determine which block and column in block
        let block = rhs_row / a_cols;
        let col_in_block = rhs_row % a_cols;

        // Get non-zeros in column col_in_block of A
        if col_in_block < col_indptr.len() - 1 {
            let start = col_indptr[col_in_block] as usize;
            let end = col_indptr[col_in_block + 1] as usize;

            for nnz_idx in start..end {
                let a_row = row_indices[nnz_idx] as usize;
                let a_val = values[nnz_idx];
                if a_val != 0.0 {
                    let new_row = (block * a_rows + a_row) as i64;
                    result.push(a_val * rhs_val, new_row, rhs_col, rhs_param);
                }
            }
        }
    }

    result
}

/// Block diagonal multiplication from right: tensor @ kron(I_k, A^T)
fn multiply_block_diagonal_right(
    lhs: &SparseTensor,
    rhs: &ConstantMatrix,
    lin_op: &LinOp,
    ctx: &ProcessingContext,
) -> SparseTensor {
    let output_rows = lin_op.size();

    match rhs {
        ConstantMatrix::Scalar(s) => {
            // Scalar multiplication
            let mut result = lhs.clone();
            result.scale_in_place(*s);
            result.shape = (output_rows, ctx.var_length as usize + 1);
            result
        }
        ConstantMatrix::DenseColMajor { data, rows, cols } => {
            // Dense block diagonal right multiplication with column-major data
            multiply_dense_block_diagonal_right_colmajor(data, *rows, *cols, lhs, output_rows, ctx)
        }
        ConstantMatrix::DenseRowMajor { data, rows, cols } => {
            // Dense block diagonal right multiplication with row-major data
            multiply_dense_block_diagonal_right_rowmajor(data, *rows, *cols, lhs, output_rows, ctx)
        }
        ConstantMatrix::Sparse {
            values,
            row_indices,
            col_indptr,
            rows,
            cols,
        } => {
            // Sparse block diagonal right multiplication
            multiply_sparse_block_diagonal_right(
                values,
                row_indices,
                col_indptr,
                *rows,
                *cols,
                lhs,
                output_rows,
                ctx,
            )
        }
    }
}

/// Dense block diagonal right multiplication with column-major data: kron(A^T, I_k) @ tensor
/// For X @ A where X has shape (k, n) and A has shape (n, p):
/// - Input tensor represents vec(X) in column-major order
/// - Output is vec(X @ A) in column-major order
/// - The operation is kron(A^T, I_k) @ vec(X)
/// Direct element-by-element processing with pre-allocated output
#[inline]
fn multiply_dense_block_diagonal_right_colmajor(
    data: &[f64],
    a_rows: usize, // n (rows of A = cols of X)
    a_cols: usize, // p (cols of A = cols of output)
    lhs: &SparseTensor,
    output_rows: usize, // k * p (total elements in output)
    ctx: &ProcessingContext,
) -> SparseTensor {
    // Guard against edge cases
    if a_cols == 0 || a_rows == 0 || lhs.nnz() == 0 {
        return SparseTensor::empty((output_rows, ctx.var_length as usize + 1));
    }

    // k = number of rows in X (and in output)
    let k = output_rows / a_cols;
    if k == 0 {
        return SparseTensor::empty((output_rows, ctx.var_length as usize + 1));
    }

    // Pre-allocate with exact capacity
    let exact_nnz = lhs.nnz() * a_cols;
    let mut rows = Vec::with_capacity(exact_nnz);
    let mut cols = Vec::with_capacity(exact_nnz);
    let mut vals = Vec::with_capacity(exact_nnz);
    let mut params = Vec::with_capacity(exact_nnz);

    // Direct processing
    for idx in 0..lhs.nnz() {
        let lhs_row = lhs.rows[idx] as usize;
        let lhs_col = lhs.cols[idx];
        let lhs_val = lhs.data[idx];
        let lhs_param = lhs.param_offsets[idx];

        let row_in_X = lhs_row % k;
        let col_in_X = lhs_row / k;

        // For each column j of A, emit contribution
        // A is column-major: A[col_in_X, j] = data[j * a_rows + col_in_X]
        for j in 0..a_cols {
            let a_val = data[j * a_rows + col_in_X];
            if a_val != 0.0 {
                let out_row = j * k + row_in_X;
                if out_row < output_rows {
                    rows.push(out_row as i64);
                    cols.push(lhs_col);
                    vals.push(lhs_val * a_val);
                    params.push(lhs_param);
                }
            }
        }
    }

    SparseTensor {
        shape: (output_rows, ctx.var_length as usize + 1),
        rows,
        cols,
        data: vals,
        param_offsets: params,
    }
}

/// Dense block diagonal right multiplication with row-major data: kron(A^T, I_k) @ tensor
/// Used for 1D arrays (row vectors)
#[inline]
fn multiply_dense_block_diagonal_right_rowmajor(
    data: &[f64],
    _a_rows: usize, // n (rows of A = cols of X) - unused but documents interface
    a_cols: usize,  // p (cols of A = cols of output)
    lhs: &SparseTensor,
    output_rows: usize, // k * p (total elements in output)
    ctx: &ProcessingContext,
) -> SparseTensor {
    // Guard against division by zero
    if a_cols == 0 {
        return SparseTensor::empty((output_rows, ctx.var_length as usize + 1));
    }

    // k = number of rows in X (and in output)
    // output_rows = k * p, so k = output_rows / p
    let k = output_rows / a_cols;

    // Guard against division by zero for k
    if k == 0 {
        return SparseTensor::empty((output_rows, ctx.var_length as usize + 1));
    }

    let est_nnz = lhs.nnz() * a_cols;
    let mut result =
        SparseTensor::with_capacity((output_rows, ctx.var_length as usize + 1), est_nnz);

    // For each entry in lhs tensor (representing variable X)
    // Input uses column-major ordering: index v represents X[v % k, v / k]
    for idx in 0..lhs.nnz() {
        let lhs_row = lhs.rows[idx] as usize;
        let lhs_col = lhs.cols[idx];
        let lhs_val = lhs.data[idx];
        let lhs_param = lhs.param_offsets[idx];

        // Column-major decomposition of input index
        // lhs_row represents X[row_in_X, col_in_X]
        let row_in_X = lhs_row % k; // i = row in X
        let col_in_X = lhs_row / k; // l = column in X (also row in A)

        // For X @ A: (X @ A)[i, j] = sum_l X[i, l] * A[l, j]
        // This input element X[i, l] contributes to all output columns j
        // with coefficient A[l, j]
        // Row-major data: A[l, j] at index l * a_cols + j
        for j in 0..a_cols {
            let a_val = data[col_in_X * a_cols + j]; // A[col_in_X, j] = A[l, j] in row-major
            if a_val != 0.0 {
                // Output index in column-major: (X@A)[i, j] at index j * k + i
                let new_row = (j * k + row_in_X) as i64;
                result.push(a_val * lhs_val, new_row, lhs_col, lhs_param);
            }
        }
    }

    result
}

/// Sparse block diagonal right multiplication: kron(A^T, I_k) @ tensor
/// For X @ A where X has shape (k, n) and A has shape (n, p):
/// - Input tensor represents vec(X) in column-major order
/// - Output is vec(X @ A) in column-major order
/// - A is stored in CSC format
fn multiply_sparse_block_diagonal_right(
    values: &[f64],
    row_indices: &[i64],
    col_indptr: &[i64],
    a_rows: usize, // n (rows of A = cols of X)
    a_cols: usize, // p (cols of A = cols of output)
    lhs: &SparseTensor,
    output_rows: usize, // k * p
    ctx: &ProcessingContext,
) -> SparseTensor {
    // Guard against division by zero
    if a_cols == 0 {
        return SparseTensor::empty((output_rows, ctx.var_length as usize + 1));
    }

    // k = number of rows in X (and in output)
    let k = output_rows / a_cols;

    // Guard against division by zero for k
    if k == 0 {
        return SparseTensor::empty((output_rows, ctx.var_length as usize + 1));
    }

    let est_nnz = lhs.nnz() * values.len() / a_rows.max(1);
    let mut result =
        SparseTensor::with_capacity((output_rows, ctx.var_length as usize + 1), est_nnz);

    // For each entry in lhs tensor (representing variable X)
    for idx in 0..lhs.nnz() {
        let lhs_row = lhs.rows[idx] as usize;
        let lhs_col = lhs.cols[idx];
        let lhs_val = lhs.data[idx];
        let lhs_param = lhs.param_offsets[idx];

        // Column-major decomposition: lhs_row represents X[row_in_X, col_in_X]
        let row_in_X = lhs_row % k; // i = row in X
        let col_in_X = lhs_row / k; // l = column in X (also row in A)

        // For X @ A: (X @ A)[i, j] = sum_l X[i, l] * A[l, j]
        // Find all A[l, j] entries (row l = col_in_X, any column j)
        // CSC format: column j has entries at indices col_indptr[j] to col_indptr[j+1]
        for j in 0..a_cols {
            if j < col_indptr.len() - 1 {
                let start = col_indptr[j] as usize;
                let end = col_indptr[j + 1] as usize;

                // Search for row col_in_X in column j
                for nnz_idx in start..end {
                    if row_indices[nnz_idx] as usize == col_in_X {
                        let a_val = values[nnz_idx];
                        if a_val != 0.0 {
                            // Output index in column-major: (X@A)[i, j] at index j * k + i
                            let new_row = (j * k + row_in_X) as i64;
                            result.push(a_val * lhs_val, new_row, lhs_col, lhs_param);
                        }
                    }
                }
            }
        }
    }

    result
}

/// Check if a LinOp tree contains any parameter nodes
fn is_parametric(lin_op: &LinOp) -> bool {
    use crate::linop::OpType;

    match lin_op.op_type {
        OpType::Param => true,
        OpType::Variable | OpType::ScalarConst | OpType::DenseConst | OpType::SparseConst => false,
        _ => {
            // Check data if it's a LinOp
            if let LinOpData::LinOpRef(ref data_op) = lin_op.data {
                if is_parametric(data_op) {
                    return true;
                }
            }
            // Check all args
            lin_op.args.iter().any(is_parametric)
        }
    }
}

/// Parametric left multiplication: param_tensor @ variable_tensor
///
/// When multiplying a parametric matrix A by a variable x (A @ x):
/// - A has shape (m, n) and is stored as a tensor with different param_offsets per element
/// - x has shape (n,) and is stored as identity matrix in the variable columns
/// - Output has shape (m,) where each entry is sum_j(A[i,j] * x[j])
/// - The output preserves param_offsets from A
///
/// The key difference from constant multiplication is that each A[i,j] has its own
/// param_offset, so the output entries inherit those offsets.
fn multiply_parametric_left(
    lhs_tensor: &SparseTensor,
    lhs_linop: &LinOp,
    rhs_tensor: &SparseTensor,
    output_linop: &LinOp,
    ctx: &ProcessingContext,
) -> SparseTensor {
    let output_rows = output_linop.size();

    // Get matrix dimensions from lhs LinOp shape
    let (a_rows, a_cols) = if lhs_linop.shape.len() == 1 {
        // 1D array treated as row vector for left multiplication
        (1, lhs_linop.shape[0])
    } else if lhs_linop.shape.len() >= 2 {
        (lhs_linop.shape[0], lhs_linop.shape[1])
    } else {
        // Scalar
        (1, 1)
    };

    // Guard against empty
    if a_cols == 0 {
        return SparseTensor::empty((output_rows, ctx.var_length as usize + 1));
    }

    // Number of blocks (for block-diagonal structure, unused but documents algorithm)
    let _k = rhs_tensor.shape.0 / a_cols;

    // Build index of lhs_tensor entries by their row (which corresponds to A[row/a_cols, row%a_cols])
    // Actually, for a parameter tensor, row corresponds to flat index in column-major order
    // For shape (m, n), flat index i -> (i % m, i / m) in column-major
    let lhs_size = lhs_linop.size();
    let mut lhs_by_flat_idx: Vec<Vec<(f64, i64)>> = vec![Vec::new(); lhs_size];
    for i in 0..lhs_tensor.nnz() {
        let row = lhs_tensor.rows[i] as usize;
        if row < lhs_size {
            lhs_by_flat_idx[row].push((lhs_tensor.data[i], lhs_tensor.param_offsets[i]));
        }
    }

    // Estimate capacity
    let est_nnz = rhs_tensor.nnz() * a_rows;
    let mut result =
        SparseTensor::with_capacity((output_rows, ctx.var_length as usize + 1), est_nnz);

    // For each entry in rhs (the variable tensor)
    for rhs_idx in 0..rhs_tensor.nnz() {
        let rhs_row = rhs_tensor.rows[rhs_idx] as usize; // Index in variable
        let rhs_col = rhs_tensor.cols[rhs_idx]; // Variable column
        let rhs_val = rhs_tensor.data[rhs_idx];
        let rhs_param = rhs_tensor.param_offsets[rhs_idx];

        // Determine which block and which column within block
        let block = rhs_row / a_cols;
        let col_in_block = rhs_row % a_cols; // j in A[i, j]

        // For A @ x, entry (i, j) of A contributes to output row i
        // A[i, j] is at flat index j * a_rows + i (column-major)
        for i in 0..a_rows {
            let flat_idx = col_in_block * a_rows + i;

            if flat_idx < lhs_by_flat_idx.len() {
                for &(lhs_val, lhs_param) in &lhs_by_flat_idx[flat_idx] {
                    // Output row: block * a_rows + i
                    let new_row = (block * a_rows + i) as i64;

                    // Determine result param_offset
                    // If both have non-constant params, we'd need param*param (unsupported)
                    // Typically, variable is constant-param and A is parametric
                    let result_param = if rhs_param == ctx.const_param() {
                        lhs_param
                    } else if lhs_param == ctx.const_param() {
                        rhs_param
                    } else {
                        // Both parametric - use lhs param (A's param)
                        lhs_param
                    };

                    result.push(lhs_val * rhs_val, new_row, rhs_col, result_param);
                }
            }
        }
    }

    result
}

/// Parametric right multiplication: variable_tensor @ param_tensor
///
/// When multiplying a variable x by a parametric matrix A (x @ A):
/// - x has shape (k, n) and is stored as identity matrix in the variable columns
/// - A has shape (n, p) and is stored as a tensor with different param_offsets per element
/// - Output has shape (k, p) where each entry is sum_l(x[i,l] * A[l,j])
/// - The output preserves param_offsets from A
fn multiply_parametric_right(
    lhs_tensor: &SparseTensor,
    rhs_tensor: &SparseTensor,
    rhs_linop: &LinOp,
    output_linop: &LinOp,
    ctx: &ProcessingContext,
) -> SparseTensor {
    let output_rows = output_linop.size();

    // Get matrix dimensions from rhs LinOp shape
    let (a_rows, a_cols) = if rhs_linop.shape.len() == 1 {
        // 1D array - check context for proper handling
        let arg = &output_linop.args[0];
        let arg_cols = if arg.shape.len() == 1 {
            arg.shape[0]
        } else if arg.shape.len() >= 2 {
            arg.shape[1]
        } else {
            1
        };
        // If arg_cols != n, transpose interpretation
        if arg_cols != rhs_linop.shape[0] {
            (1, rhs_linop.shape[0]) // Row vector
        } else {
            (rhs_linop.shape[0], 1) // Column vector
        }
    } else if rhs_linop.shape.len() >= 2 {
        (rhs_linop.shape[0], rhs_linop.shape[1])
    } else {
        (1, 1)
    };

    // Guard against empty
    if a_cols == 0 {
        return SparseTensor::empty((output_rows, ctx.var_length as usize + 1));
    }

    // k = number of rows in output (and in X)
    let k = output_rows / a_cols;
    if k == 0 {
        return SparseTensor::empty((output_rows, ctx.var_length as usize + 1));
    }

    // Build index of rhs_tensor entries by their row (flat index in column-major)
    let rhs_size = rhs_linop.size();
    let mut rhs_by_flat_idx: Vec<Vec<(f64, i64)>> = vec![Vec::new(); rhs_size];
    for i in 0..rhs_tensor.nnz() {
        let row = rhs_tensor.rows[i] as usize;
        if row < rhs_size {
            rhs_by_flat_idx[row].push((rhs_tensor.data[i], rhs_tensor.param_offsets[i]));
        }
    }

    // Estimate capacity
    let est_nnz = lhs_tensor.nnz() * a_cols;
    let mut result =
        SparseTensor::with_capacity((output_rows, ctx.var_length as usize + 1), est_nnz);

    // For each entry in lhs (the variable tensor)
    for lhs_idx in 0..lhs_tensor.nnz() {
        let lhs_row = lhs_tensor.rows[lhs_idx] as usize; // Index in variable
        let lhs_col = lhs_tensor.cols[lhs_idx]; // Variable column
        let lhs_val = lhs_tensor.data[lhs_idx];
        let lhs_param = lhs_tensor.param_offsets[lhs_idx];

        // Column-major decomposition of input index
        // lhs_row represents X[row_in_X, col_in_X]
        let row_in_X = lhs_row % k; // i = row in X
        let col_in_X = lhs_row / k; // l = column in X (also row in A)

        // For X @ A: (X @ A)[i, j] = sum_l X[i, l] * A[l, j]
        // This X[i, l] entry contributes to all output columns j
        // A[l, j] is at flat index j * a_rows + l (column-major)
        for j in 0..a_cols {
            let flat_idx = j * a_rows + col_in_X;

            if flat_idx < rhs_by_flat_idx.len() {
                for &(rhs_val, rhs_param) in &rhs_by_flat_idx[flat_idx] {
                    // Output index in column-major: (X@A)[i, j] at index j * k + i
                    let new_row = (j * k + row_in_X) as i64;

                    // Determine result param_offset
                    let result_param = if lhs_param == ctx.const_param() {
                        rhs_param
                    } else if rhs_param == ctx.const_param() {
                        lhs_param
                    } else {
                        // Both parametric - use rhs param (A's param)
                        rhs_param
                    };

                    result.push(lhs_val * rhs_val, new_row, lhs_col, result_param);
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linop::OpType;
    use crate::operations::process_sum;
    use crate::tensor::CONSTANT_ID;
    use std::collections::HashMap;

    fn make_ctx() -> ProcessingContext {
        let mut id_to_col = HashMap::new();
        id_to_col.insert(1, 0); // Variable 1 at column 0

        let mut param_to_col = HashMap::new();
        param_to_col.insert(CONSTANT_ID, 0);

        let mut param_to_size = HashMap::new();
        param_to_size.insert(CONSTANT_ID, 1);

        ProcessingContext {
            id_to_col,
            param_to_col,
            param_to_size,
            var_length: 4,
            param_size_plus_one: 1,
        }
    }

    #[test]
    fn test_neg() {
        let ctx = make_ctx();

        // Create variable (2x2)
        let var_op = LinOp {
            op_type: OpType::Variable,
            shape: vec![2, 2],
            args: vec![],
            data: LinOpData::Int(1),
        };

        // Negate it
        let neg_op = LinOp {
            op_type: OpType::Neg,
            shape: vec![2, 2],
            args: vec![var_op],
            data: LinOpData::None,
        };

        let tensor = process_neg(&neg_op, &ctx);

        assert_eq!(tensor.nnz(), 4);
        // All values should be -1.0
        for &v in &tensor.data {
            assert_eq!(v, -1.0);
        }
    }

    #[test]
    fn test_sum_two_variables() {
        // Test summing two variables
        let mut id_to_col = HashMap::new();
        id_to_col.insert(1, 0); // Variable 1 at column 0
        id_to_col.insert(2, 4); // Variable 2 at column 4

        let mut param_to_col = HashMap::new();
        param_to_col.insert(CONSTANT_ID, 0);

        let mut param_to_size = HashMap::new();
        param_to_size.insert(CONSTANT_ID, 1);

        let ctx = ProcessingContext {
            id_to_col,
            param_to_col,
            param_to_size,
            var_length: 8,
            param_size_plus_one: 1,
        };

        let var1 = LinOp {
            op_type: OpType::Variable,
            shape: vec![2, 2],
            args: vec![],
            data: LinOpData::Int(1),
        };

        let var2 = LinOp {
            op_type: OpType::Variable,
            shape: vec![2, 2],
            args: vec![],
            data: LinOpData::Int(2),
        };

        let sum_op = LinOp {
            op_type: OpType::Sum,
            shape: vec![2, 2],
            args: vec![var1, var2],
            data: LinOpData::None,
        };

        let tensor = process_sum(&sum_op, &ctx);

        // Sum of two identity matrices should have 8 non-zeros
        assert_eq!(tensor.nnz(), 8);
    }

    #[test]
    fn test_mul_scalar() {
        let ctx = make_ctx();

        // Create variable (2x2)
        let var_op = LinOp {
            op_type: OpType::Variable,
            shape: vec![2, 2],
            args: vec![],
            data: LinOpData::Int(1),
        };

        // Create scalar constant (2.0)
        let const_op = LinOp {
            op_type: OpType::ScalarConst,
            shape: vec![],
            args: vec![],
            data: LinOpData::Float(2.0),
        };

        // Multiply: 2.0 * var
        let mul_op = LinOp {
            op_type: OpType::Mul,
            shape: vec![2, 2],
            args: vec![var_op],
            data: LinOpData::LinOpRef(Box::new(const_op)),
        };

        let tensor = process_mul(&mul_op, &ctx);

        assert_eq!(tensor.nnz(), 4);
        // All values should be 2.0
        for &v in &tensor.data {
            assert_eq!(v, 2.0);
        }
    }

    #[test]
    fn test_mul_dense_matrix_row_major() {
        // Test that row-major input data is handled correctly
        let ctx = make_ctx();

        // Create variable (2x2)
        let var_op = LinOp {
            op_type: OpType::Variable,
            shape: vec![2, 2],
            args: vec![],
            data: LinOpData::Int(1),
        };

        // Create 2x2 constant matrix [[1, 2], [3, 4]] in row-major order
        // Row-major: [1, 2, 3, 4]
        let const_op = LinOp {
            op_type: OpType::DenseConst,
            shape: vec![2, 2],
            args: vec![],
            data: LinOpData::DenseArray {
                data: Arc::from(vec![1.0, 2.0, 3.0, 4.0]),
                shape: vec![2, 2],
            },
        };

        // Multiply: const @ var
        let mul_op = LinOp {
            op_type: OpType::Mul,
            shape: vec![2, 2],
            args: vec![var_op],
            data: LinOpData::LinOpRef(Box::new(const_op)),
        };

        let tensor = process_mul(&mul_op, &ctx);

        // Should have non-zero entries
        assert!(tensor.nnz() > 0);
        // The multiplication should produce correct results
        // For identity variable, result should reflect the constant matrix
    }

    #[test]
    fn test_mul_elem() {
        let ctx = make_ctx();

        // Create variable (2x2)
        let var_op = LinOp {
            op_type: OpType::Variable,
            shape: vec![2, 2],
            args: vec![],
            data: LinOpData::Int(1),
        };

        // Create constant for elementwise multiply [2, 3, 4, 5]
        let const_op = LinOp {
            op_type: OpType::DenseConst,
            shape: vec![2, 2],
            args: vec![],
            data: LinOpData::DenseArray {
                data: Arc::from(vec![2.0, 3.0, 4.0, 5.0]),
                shape: vec![2, 2],
            },
        };

        let mul_elem_op = LinOp {
            op_type: OpType::MulElem,
            shape: vec![2, 2],
            args: vec![var_op],
            data: LinOpData::LinOpRef(Box::new(const_op)),
        };

        let tensor = process_mul_elem(&mul_elem_op, &ctx);

        assert_eq!(tensor.nnz(), 4);
    }

    #[test]
    fn test_div() {
        // Use smaller variable for div test
        let mut id_to_col = HashMap::new();
        id_to_col.insert(1, 0);

        let mut param_to_col = HashMap::new();
        param_to_col.insert(CONSTANT_ID, 0);

        let mut param_to_size = HashMap::new();
        param_to_size.insert(CONSTANT_ID, 1);

        let ctx = ProcessingContext {
            id_to_col,
            param_to_col,
            param_to_size,
            var_length: 2,
            param_size_plus_one: 1,
        };

        // Create variable (2,)
        let var_op = LinOp {
            op_type: OpType::Variable,
            shape: vec![2],
            args: vec![],
            data: LinOpData::Int(1),
        };

        // Create constant for division [2, 4]
        let const_op = LinOp {
            op_type: OpType::DenseConst,
            shape: vec![2],
            args: vec![],
            data: LinOpData::DenseArray {
                data: Arc::from(vec![2.0, 4.0]),
                shape: vec![2],
            },
        };

        let div_op = LinOp {
            op_type: OpType::Div,
            shape: vec![2],
            args: vec![var_op],
            data: LinOpData::LinOpRef(Box::new(const_op)),
        };

        let tensor = process_div(&div_op, &ctx);

        assert_eq!(tensor.nnz(), 2);
        // Values should be 1/2 and 1/4
        assert!((tensor.data[0] - 0.5).abs() < 1e-10);
        assert!((tensor.data[1] - 0.25).abs() < 1e-10);
    }

    /// PR 3 fast path: `Mul(DenseConst, Index(Variable, contiguous_slice))`
    /// (the LASSO canonicalisation pattern). The new path probes the RHS via
    /// `process_linop_block`, sees a typed `Identity` at a shifted offset,
    /// and dispatches to `mul_const_by_identity_offset`. This test verifies
    /// numerical equivalence to a manual hand-computation.
    #[test]
    fn test_mul_with_index_variable_lasso_pattern() {
        // Variable y of size 5 at column 0. We slice y[1..4] and multiply
        // by a 3x3 dense matrix:
        //   A = [[ 1, 2, 3 ]
        //        [ 4, 5, 6 ]
        //        [ 7, 8, 9 ]]
        //
        // Expected output: row i has A's row i emitted at columns
        // var_col_offset + col, where var_col_offset = ctx.var_col(1) + 1 = 1.
        let mut id_to_col = HashMap::new();
        id_to_col.insert(1, 0);

        let mut param_to_col = HashMap::new();
        param_to_col.insert(CONSTANT_ID, 0);

        let mut param_to_size = HashMap::new();
        param_to_size.insert(CONSTANT_ID, 1);

        let ctx = ProcessingContext {
            id_to_col,
            param_to_col,
            param_to_size,
            var_length: 5,
            param_size_plus_one: 1,
        };

        let var = LinOp {
            op_type: OpType::Variable,
            shape: vec![5],
            args: vec![],
            data: LinOpData::Int(1),
        };
        let index = LinOp {
            op_type: OpType::Index,
            shape: vec![3],
            args: vec![var],
            data: LinOpData::Slices(vec![crate::linop::SliceData {
                start: 1,
                stop: 4,
                step: 1,
            }]),
        };
        // Column-major: (col 0) 1 4 7  (col 1) 2 5 8  (col 2) 3 6 9
        let dense_a = LinOp {
            op_type: OpType::DenseConst,
            shape: vec![3, 3],
            args: vec![],
            data: LinOpData::DenseArray {
                data: Arc::from(vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]),
                shape: vec![3, 3],
            },
        };
        let mul_op = LinOp {
            op_type: OpType::Mul,
            shape: vec![3],
            args: vec![index],
            data: LinOpData::LinOpRef(Box::new(dense_a)),
        };

        let tensor = process_mul(&mul_op, &ctx);

        // Expect 9 entries, A's 9 nonzeros placed at cols {1, 2, 3}.
        // Emission order is col-outer, row-inner (F-order): same as DenseColMajor walk.
        assert_eq!(tensor.nnz(), 9);
        assert_eq!(tensor.data, vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
        assert_eq!(tensor.cols, vec![1, 1, 1, 2, 2, 2, 3, 3, 3]);
        assert_eq!(tensor.rows, vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
        for &p in &tensor.param_offsets {
            assert_eq!(p, 0);
        }
    }

    /// Sanity: the existing `as_plain_variable` fast path still fires
    /// byte-for-byte the same as before (Mul of constant against a plain
    /// Variable), even though the new probe path comes after.
    #[test]
    fn test_mul_plain_variable_fast_path_still_fires() {
        let ctx = make_ctx();

        let var_op = LinOp {
            op_type: OpType::Variable,
            shape: vec![4],
            args: vec![],
            data: LinOpData::Int(1),
        };
        let dense_a = LinOp {
            op_type: OpType::DenseConst,
            shape: vec![2, 4],
            args: vec![],
            // Column-major 2x4: identity-ish content for easy verification.
            data: LinOpData::DenseArray {
                data: Arc::from(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
                shape: vec![2, 4],
            },
        };
        let mul_op = LinOp {
            op_type: OpType::Mul,
            shape: vec![2],
            args: vec![var_op],
            data: LinOpData::LinOpRef(Box::new(dense_a)),
        };

        let tensor = process_mul(&mul_op, &ctx);

        // 4 nonzeros (the 1.0 entries). Variable starts at col 0, so out_col = col index.
        assert_eq!(tensor.nnz(), 4);
        assert_eq!(tensor.data, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(tensor.rows, vec![0, 1, 0, 1]);
        assert_eq!(tensor.cols, vec![0, 1, 2, 3]);
    }
}
