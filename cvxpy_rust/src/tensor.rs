//! Sparse tensor representation in COO format
//!
//! This module provides the SparseTensor type which stores 3D tensor data
//! in coordinate (COO) format, matching CVXPY's TensorRepresentation.

use rayon::prelude::*;
use std::collections::HashMap;

/// Constant ID used for non-parametric entries
pub const CONSTANT_ID: i64 = -1;

/// Sparse tensor in COO format
///
/// Represents a 3D tensor with dimensions (rows, cols, param_slices).
/// This matches Python's TensorRepresentation from canon_backend.py.
#[derive(Debug, Clone)]
pub struct SparseTensor {
    pub data: Vec<f64>,
    pub rows: Vec<i64>,
    pub cols: Vec<i64>,
    pub param_offsets: Vec<i64>,
    pub shape: (usize, usize),
}

impl SparseTensor {
    /// Create an empty tensor with given shape
    pub fn empty(shape: (usize, usize)) -> Self {
        SparseTensor {
            data: Vec::new(),
            rows: Vec::new(),
            cols: Vec::new(),
            param_offsets: Vec::new(),
            shape,
        }
    }

    /// Create an empty tensor with pre-allocated capacity
    pub fn with_capacity(shape: (usize, usize), capacity: usize) -> Self {
        SparseTensor {
            data: Vec::with_capacity(capacity),
            rows: Vec::with_capacity(capacity),
            cols: Vec::with_capacity(capacity),
            param_offsets: Vec::with_capacity(capacity),
            shape,
        }
    }

    /// Number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Add a single entry to the tensor
    #[inline]
    pub fn push(&mut self, value: f64, row: i64, col: i64, param_offset: i64) {
        self.data.push(value);
        self.rows.push(row);
        self.cols.push(col);
        self.param_offsets.push(param_offset);
    }

    /// Extend this tensor with entries from another tensor
    pub fn extend(&mut self, other: SparseTensor) {
        self.data.extend(other.data);
        self.rows.extend(other.rows);
        self.cols.extend(other.cols);
        self.param_offsets.extend(other.param_offsets);
    }

    /// Negate all data values in place
    pub fn negate_in_place(&mut self) {
        for d in &mut self.data {
            *d = -*d;
        }
    }

    /// Scale all data values in place
    pub fn scale_in_place(&mut self, factor: f64) {
        for d in &mut self.data {
            *d *= factor;
        }
    }

    /// Offset all row indices in place
    pub fn offset_rows_in_place(&mut self, offset: i64) {
        for r in &mut self.rows {
            *r += offset;
        }
    }

    /// Select rows by index array (creates new tensor)
    /// OPTIMIZATION: Uses fast paths for common patterns
    pub fn select_rows(&self, row_indices: &[i64]) -> SparseTensor {
        // Fast path 1: Empty input
        if row_indices.is_empty() {
            return SparseTensor::empty((0, self.shape.1));
        }

        // Fast path 2: Identity permutation (no change needed)
        if self.is_identity_permutation(row_indices) {
            return self.clone();
        }

        // Fast path 3: Simple offset (contiguous range starting from offset)
        if let Some(offset) = self.check_contiguous_with_offset(row_indices) {
            return self.select_contiguous_rows(offset, row_indices.len());
        }

        // Fast path 4: Reversed identity permutation
        if self.is_reversed_identity(row_indices) {
            return self.reverse_rows();
        }

        // General case: use HashMap
        self.select_rows_general(row_indices)
    }

    /// Check if row_indices is an identity permutation [0, 1, 2, ..., n-1]
    #[inline]
    fn is_identity_permutation(&self, row_indices: &[i64]) -> bool {
        if row_indices.len() != self.shape.0 {
            return false;
        }
        row_indices.iter().enumerate().all(|(i, &r)| r == i as i64)
    }

    /// Check if row_indices is a contiguous range with offset [offset, offset+1, ..., offset+n-1]
    /// Returns the offset if so
    #[inline]
    fn check_contiguous_with_offset(&self, row_indices: &[i64]) -> Option<i64> {
        if row_indices.is_empty() {
            return Some(0);
        }
        let offset = row_indices[0];
        if row_indices
            .iter()
            .enumerate()
            .all(|(i, &r)| r == offset + i as i64)
        {
            Some(offset)
        } else {
            None
        }
    }

    /// Check if row_indices is reversed identity [n-1, n-2, ..., 1, 0]
    #[inline]
    fn is_reversed_identity(&self, row_indices: &[i64]) -> bool {
        if row_indices.len() != self.shape.0 {
            return false;
        }
        let n = row_indices.len();
        row_indices
            .iter()
            .enumerate()
            .all(|(i, &r)| r == (n - 1 - i) as i64)
    }

    /// Select contiguous rows starting from offset
    fn select_contiguous_rows(&self, offset: i64, count: usize) -> SparseTensor {
        let end_row = offset + count as i64;

        // Count entries in range for capacity estimation
        let est_nnz = self
            .rows
            .iter()
            .filter(|&&r| r >= offset && r < end_row)
            .count();

        let mut result = SparseTensor::with_capacity((count, self.shape.1), est_nnz);

        for i in 0..self.nnz() {
            let row = self.rows[i];
            if row >= offset && row < end_row {
                result.push(
                    self.data[i],
                    row - offset, // Adjust row index
                    self.cols[i],
                    self.param_offsets[i],
                );
            }
        }

        result
    }

    /// Reverse all row indices
    fn reverse_rows(&self) -> SparseTensor {
        let n_rows = self.shape.0 as i64;
        let mut result = SparseTensor::with_capacity(self.shape, self.nnz());

        for i in 0..self.nnz() {
            result.push(
                self.data[i],
                n_rows - 1 - self.rows[i], // Reverse row index
                self.cols[i],
                self.param_offsets[i],
            );
        }

        result
    }

    /// General row selection using HashMap (fallback)
    fn select_rows_general(&self, row_indices: &[i64]) -> SparseTensor {
        // Build mapping from old row to new positions
        let mut row_map: HashMap<i64, Vec<usize>> = HashMap::with_capacity(row_indices.len());
        for (new_idx, &old_row) in row_indices.iter().enumerate() {
            row_map.entry(old_row).or_default().push(new_idx);
        }

        // Estimate capacity
        let mut result = SparseTensor::with_capacity(
            (row_indices.len(), self.shape.1),
            self.nnz() * row_indices.len() / self.shape.0.max(1),
        );

        // Select entries
        for i in 0..self.nnz() {
            if let Some(new_positions) = row_map.get(&self.rows[i]) {
                for &new_row in new_positions {
                    result.push(
                        self.data[i],
                        new_row as i64,
                        self.cols[i],
                        self.param_offsets[i],
                    );
                }
            }
        }

        result
    }

    /// Combine multiple tensors into one (concatenate all entries)
    pub fn combine(tensors: Vec<SparseTensor>) -> SparseTensor {
        if tensors.is_empty() {
            return SparseTensor::empty((0, 0));
        }

        let total_nnz: usize = tensors.iter().map(|t| t.nnz()).sum();
        let shape = tensors[0].shape;

        let mut result = SparseTensor::with_capacity(shape, total_nnz);
        for tensor in tensors {
            result.extend(tensor);
        }
        result
    }
}


/// Result structure returned to Python
#[derive(Debug)]
pub struct BuildMatrixResult {
    pub data: Vec<f64>,
    pub rows: Vec<i64>,
    pub cols: Vec<i64>,
    pub shape: (usize, usize),
}

/// Pre-computed reduction matching `canonInterface.reduce_problem_data_tensor`.
///
/// Walking the already-sorted COO output once produces every value the Python
/// helper would compute via `np.unique` + scipy COO→CSR conversion. Returning
/// it alongside the raw matrix lets `MatrixData.cache()` skip
/// `reduce_problem_data_tensor` entirely (~30% of warm wall-clock for LASSO
/// 200×500 per the cProfile run).
///
/// `reduced_*` describe a CSR sparse matrix of shape `reduced_shape`:
///  - `reduced_data[k]`, `reduced_col_indices[k]` give the value and column of
///    entry k.
///  - `reduced_indptr[r]` is the offset into `reduced_data` where row r starts.
///
/// `final_indices` / `final_indptr` / `final_shape` are the CSC components of
/// the eventual problem-data matrix that `MatrixData` stores in
/// `problem_data_index`.
#[derive(Debug)]
pub struct ReducedMatrix {
    pub reduced_data: Vec<f64>,
    pub reduced_col_indices: Vec<i64>,
    pub reduced_indptr: Vec<i64>,
    pub reduced_shape: (usize, usize),
    pub final_indices: Vec<i64>,
    pub final_indptr: Vec<i64>,
    pub final_shape: (usize, usize),
}

/// Compute the reduction that `reduce_problem_data_tensor` would compute
/// on `sp.csc_array((data, (rows, cols)), shape)`.
///
/// `var_length` is the number of solver variables; `quad_form` matches the
/// flag passed to `reduce_problem_data_tensor` (true for the quadratic form
/// matrix P, false for the constraint matrix A).
///
/// Assumes `rows` is non-decreasing — i.e., the `from_tensor` post-sort has
/// already run. That invariant lets us walk the COO once to find unique row
/// values; both np.unique calls in the Python helper collapse to O(nnz)
/// linear scans here.
///
/// Takes borrowed slices so the Python-side caller can pass numpy buffers
/// directly without an upfront memcpy.
pub fn compute_reduction_from_slices(
    data: &[f64],
    rows: &[i64],
    cols: &[i64],
    shape: (usize, usize),
    var_length: usize,
    quad_form: bool,
) -> ReducedMatrix {
    let nnz = data.len();
    let big_m = shape.0;
    let num_param_slices = shape.1;

        // Pass 1: count non-zero entries (mimics A.eliminate_zeros() in the
        // Python path) and unique row values.
        let mut nnz_after_drop: usize = 0;
        let mut unique_count: usize = 0;
        let mut last: i64 = i64::MIN;
        let mut have_seen_any = false;
        for i in 0..nnz {
            if data[i] == 0.0 {
                continue;
            }
            nnz_after_drop += 1;
            if !have_seen_any || rows[i] != last {
                unique_count += 1;
                last = rows[i];
                have_seen_any = true;
            }
        }

        // Pass 2: build reduced CSR by walking the same sorted entries.
        let mut reduced_data: Vec<f64> = Vec::with_capacity(nnz_after_drop);
        let mut reduced_col_indices: Vec<i64> = Vec::with_capacity(nnz_after_drop);
        let mut unique_rows: Vec<i64> = Vec::with_capacity(unique_count);
        // indptr is filled as a running counter — entries[r+1] gets +1 for
        // each entry we emit in row r. Final cumsum runs at the end.
        let mut reduced_indptr: Vec<i64> = vec![0; unique_count + 1];

        last = i64::MIN;
        have_seen_any = false;
        let mut current_reduced_row: i64 = -1;
        for i in 0..nnz {
            if data[i] == 0.0 {
                continue;
            }
            if !have_seen_any || rows[i] != last {
                current_reduced_row += 1;
                unique_rows.push(rows[i]);
                last = rows[i];
                have_seen_any = true;
            }
            reduced_data.push(data[i]);
            reduced_col_indices.push(cols[i]);
            reduced_indptr[current_reduced_row as usize + 1] += 1;
        }

        // Cumsum to turn per-row counts into row-start offsets.
        for r in 1..reduced_indptr.len() {
            reduced_indptr[r] += reduced_indptr[r - 1];
        }
        debug_assert_eq!(*reduced_indptr.last().unwrap() as usize, nnz_after_drop);

        // CSR validity: scipy expects column indices sorted within each row.
        // The Python path goes through `tocsr()` which sorts; mirror that.
        // Per row we typically have very few entries (one per param slot),
        // so an in-place stable sort per row is fine.
        for r in 0..unique_count {
            let start = reduced_indptr[r] as usize;
            let end = reduced_indptr[r + 1] as usize;
            if end - start > 1 {
                let cols_slice = &mut reduced_col_indices[start..end];
                let data_slice = &mut reduced_data[start..end];
                // Sort via permutation index since two slices need the same order.
                let mut perm: Vec<usize> = (0..end - start).collect();
                perm.sort_unstable_by_key(|&p| cols_slice[p]);
                let cols_copy: Vec<i64> = perm.iter().map(|&p| cols_slice[p]).collect();
                let data_copy: Vec<f64> = perm.iter().map(|&p| data_slice[p]).collect();
                cols_slice.copy_from_slice(&cols_copy);
                data_slice.copy_from_slice(&data_copy);
            }
        }

        let reduced_shape = (unique_count, num_param_slices);

        // Build final problem_data_index (indices, indptr, shape).
        let n_cols = if quad_form { var_length } else { var_length + 1 };
        let n_constr = big_m / n_cols;
        let final_shape = (n_constr, n_cols);
        let n_constr_i64 = n_constr as i64;

        let mut final_indices: Vec<i64> = Vec::with_capacity(unique_count);
        // For indptr, count how many unique rows fall into each "col" group.
        // col = unique_row // n_constr.
        let mut final_indptr: Vec<i64> = vec![0; n_cols + 1];
        for &nr in &unique_rows {
            let col = (nr / n_constr_i64) as usize;
            let idx = nr % n_constr_i64;
            final_indices.push(idx);
            final_indptr[col + 1] += 1;
        }
        for r in 1..final_indptr.len() {
            final_indptr[r] += final_indptr[r - 1];
        }

    ReducedMatrix {
        reduced_data,
        reduced_col_indices,
        reduced_indptr,
        reduced_shape,
        final_indices,
        final_indptr,
        final_shape,
    }
}

impl BuildMatrixResult {
    /// Create from a SparseTensor by flattening the 3D structure to 2D.
    ///
    /// The output matrix has shape (total_rows * (var_length + 1), param_size_plus_one)
    /// where the tensor is flattened in column-major (Fortran) order.
    ///
    /// Entries are sorted by flat_row before returning so that numpy.unique
    /// in reduce_problem_data_tensor gets a pre-sorted array and runs in O(n)
    /// (timsort on sorted input) rather than O(n log n).
    pub fn from_tensor(tensor: SparseTensor, num_param_slices: usize) -> Self {
        let (n_rows, n_cols) = tensor.shape;
        let output_rows = n_rows * n_cols;
        let output_cols = num_param_slices;

        let nnz = tensor.data.len();

        // Compute flat_row = col * n_rows + row for each entry
        let flat_rows: Vec<i64> = tensor
            .rows
            .iter()
            .zip(tensor.cols.iter())
            .map(|(&r, &c)| c * (n_rows as i64) + r)
            .collect();

        // Sort by flat_row so the downstream `np.unique` runs O(n) timsort.
        //
        // For small/medium problems (cold-start LASSO 200x500 at ~100k nnz),
        // serial sort beats parallel because rayon initialises its global
        // thread pool lazily on first call and the per-process startup cost
        // (~hundreds of microseconds, plus first-time TLB / cache warm) is not
        // amortised by the parallel speedup at this size. Above the threshold,
        // par_sort wins. The cutoff (1M) was chosen so rustybench-class
        // problems (5M nnz) still take the parallel path.
        const PAR_SORT_MIN_NNZ: usize = 1_000_000;

        let mut order: Vec<usize> = (0..nnz).collect();
        if nnz >= PAR_SORT_MIN_NNZ {
            order.par_sort_unstable_by_key(|&i| flat_rows[i]);
        } else {
            order.sort_unstable_by_key(|&i| flat_rows[i]);
        }

        // Apply the permutation to all arrays
        let sorted_rows: Vec<i64> = order.iter().map(|&i| flat_rows[i]).collect();
        let sorted_data: Vec<f64> = order.iter().map(|&i| tensor.data[i]).collect();
        let sorted_cols: Vec<i64> = order.iter().map(|&i| tensor.param_offsets[i]).collect();

        BuildMatrixResult {
            data: sorted_data,
            rows: sorted_rows,
            cols: sorted_cols,
            shape: (output_rows, output_cols),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_tensor_basic() {
        let mut tensor = SparseTensor::empty((3, 4));
        tensor.push(1.0, 0, 0, 0);
        tensor.push(2.0, 1, 1, 0);
        tensor.push(3.0, 2, 2, 0);

        assert_eq!(tensor.nnz(), 3);
        assert_eq!(tensor.shape, (3, 4));
    }

    #[test]
    fn test_sparse_tensor_negate() {
        let mut tensor = SparseTensor::empty((2, 2));
        tensor.push(1.0, 0, 0, 0);
        tensor.push(-2.0, 1, 1, 0);

        tensor.negate_in_place();

        assert_eq!(tensor.data, vec![-1.0, 2.0]);
    }

    /// Reduction sanity test. Mirrors the simplest case
    /// `reduce_problem_data_tensor` would handle: a small csc-shaped tensor
    /// with a few duplicate rows and parameters.
    ///
    /// Setup:
    ///   * The conceptual problem-data matrix has shape (n_constr=3, n_cols=4),
    ///     so the flattened tensor has big_M = 12 rows.
    ///   * 5 nonzero entries at flat_rows [0, 0, 5, 5, 11], sorted.
    ///   * 2 param slices.
    /// Expected reduced output:
    ///   * unique_rows = [0, 5, 11] (3 unique rows out of 12)
    ///   * reduced_indptr = [0, 2, 4, 5]
    ///   * reduced_col_indices = [0, 1, 0, 1, 0]
    ///   * final_shape = (3, 4) (quad_form=true uses var_length=4 directly)
    ///   * final_indices = [0, 5%3=2, 11%3=2]  i.e. [0, 2, 2]
    ///   * final_indptr per col group: cols = [0, 1, 3] -> col 0:1, col 1:1, col 3:1
    #[test]
    fn test_compute_reduction_basic() {
        let result = BuildMatrixResult {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            rows: vec![0, 0, 5, 5, 11],
            cols: vec![0, 1, 0, 1, 0],
            shape: (12, 2),
        };
        let red = compute_reduction_from_slices(
            &result.data, &result.rows, &result.cols, result.shape, 4, true);

        assert_eq!(red.reduced_data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(red.reduced_col_indices, vec![0, 1, 0, 1, 0]);
        assert_eq!(red.reduced_indptr, vec![0, 2, 4, 5]);
        assert_eq!(red.reduced_shape, (3, 2));

        // n_cols = 4 (quad_form=true), n_constr = 12 / 4 = 3.
        assert_eq!(red.final_shape, (3, 4));
        assert_eq!(red.final_indices, vec![0, 2, 2]);
        // unique_rows / n_constr = [0, 1, 3]; one per group.
        // indptr is cumsum of [_, 1, 1, 0, 1] = [0, 1, 2, 2, 3].
        assert_eq!(red.final_indptr, vec![0, 1, 2, 2, 3]);
    }

    /// `quad_form=false` adds 1 to n_cols. Verifies the n_constr arithmetic.
    #[test]
    fn test_compute_reduction_non_quad_form() {
        // big_M = 6, var_length=2, quad_form=false -> n_cols = 3, n_constr = 2.
        let result = BuildMatrixResult {
            data: vec![1.0, 2.0, 3.0],
            rows: vec![0, 2, 5],
            cols: vec![0, 0, 0],
            shape: (6, 1),
        };
        let red = compute_reduction_from_slices(
            &result.data, &result.rows, &result.cols, result.shape, 2, false);
        assert_eq!(red.final_shape, (2, 3));
        // unique_rows = [0, 2, 5]
        // final_indices = [0%2, 2%2, 5%2] = [0, 0, 1]
        assert_eq!(red.final_indices, vec![0, 0, 1]);
        // cols = [0/2, 2/2, 5/2] = [0, 1, 2]
        // Counts per col: 1, 1, 1 in cols 0,1,2
        // indptr cumsum on [_, 1, 1, 1] = [0, 1, 2, 3]
        assert_eq!(red.final_indptr, vec![0, 1, 2, 3]);
    }

    /// Zero entries must be eliminated, exactly like `A.eliminate_zeros()`.
    #[test]
    fn test_compute_reduction_drops_zeros() {
        let result = BuildMatrixResult {
            data: vec![1.0, 0.0, 2.0, 0.0, 3.0],
            rows: vec![0, 0, 5, 5, 11],
            cols: vec![0, 1, 0, 1, 0],
            shape: (12, 2),
        };
        let red = compute_reduction_from_slices(
            &result.data, &result.rows, &result.cols, result.shape, 4, true);
        // Three non-zero entries at rows {0, 5, 11} -> 3 unique rows still,
        // but reduced_data has only 3 entries.
        assert_eq!(red.reduced_data, vec![1.0, 2.0, 3.0]);
        assert_eq!(red.reduced_col_indices, vec![0, 0, 0]);
        // Each unique row contributes 1 nonzero -> indptr = [0, 1, 2, 3].
        assert_eq!(red.reduced_indptr, vec![0, 1, 2, 3]);
    }

    /// CSR-style within-row column sort: entries within a row are emitted
    /// sorted by column index.
    #[test]
    fn test_compute_reduction_sorts_within_row() {
        // Single row 0 with entries at cols 5, 1, 3 (out of order).
        // big_M = 1, var_length = 1, quad_form = true -> n_cols = 1, n_constr = 1.
        let result = BuildMatrixResult {
            data: vec![10.0, 20.0, 30.0],
            rows: vec![0, 0, 0],
            cols: vec![5, 1, 3],
            shape: (1, 6),
        };
        let red = compute_reduction_from_slices(
            &result.data, &result.rows, &result.cols, result.shape, 1, true);
        assert_eq!(red.reduced_col_indices, vec![1, 3, 5]);
        // Data values must be permuted along with the cols.
        assert_eq!(red.reduced_data, vec![20.0, 30.0, 10.0]);
    }
}
