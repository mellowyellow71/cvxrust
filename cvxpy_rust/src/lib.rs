//! cvxpy_rust - Rust canonicalization backend for CVXPY
//!
//! This crate provides a high-performance replacement for the C++ cvxcore backend.
//! It converts LinOp trees into sparse matrices for optimization solvers.

// Allow some clippy lints that are too noisy for this codebase
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::useless_conversion)] // False positives from PyO3 macro expansion

mod linop;
mod matrix_builder;
mod operations;
mod tensor;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::collections::HashMap;

use crate::linop::{DeserializationContext, LinOp};
use crate::matrix_builder::build_matrix_internal;
use crate::operations::ProcessingContext;
use crate::tensor::{BuildMatrixResult, SparseTensor, CONSTANT_ID};

/// Build the coefficient matrix from LinOp trees.
///
/// This is the main entry point called from Python's RustCanonBackend.
///
/// # Arguments
/// * `lin_ops` - List of LinOp trees representing constraints/objective
/// * `param_size_plus_one` - Number of parameter slices plus one for constants
/// * `id_to_col` - Maps variable IDs to column offsets
/// * `param_to_size` - Maps parameter IDs to their sizes
/// * `param_to_col` - Maps parameter IDs to column offsets in param vector
/// * `var_length` - Total number of variables
///
/// # Returns
/// Tuple of (data, (row, col), shape) in COO format for scipy.sparse.csc_array
#[pyfunction]
fn build_matrix<'py>(
    py: Python<'py>,
    lin_ops: Vec<Bound<'py, PyAny>>,
    param_size_plus_one: i64,
    id_to_col: HashMap<i64, i64>,
    param_to_size: HashMap<i64, i64>,
    param_to_col: HashMap<i64, i64>,
    var_length: i64,
) -> PyResult<(
    Py<PyArray1<f64>>,
    (Py<PyArray1<i64>>, Py<PyArray1<i64>>),
    (i64, i64),
)> {
    // Extract LinOp trees from Python objects
    let rust_lin_ops: Vec<LinOp> = lin_ops
        .iter()
        .map(|obj| LinOp::from_python(obj))
        .collect::<PyResult<Vec<_>>>()?;

    // Build the matrix (release GIL during computation)
    let result = py.detach(|| {
        build_matrix_internal(
            &rust_lin_ops,
            param_size_plus_one,
            &id_to_col,
            &param_to_size,
            &param_to_col,
            var_length,
        )
    });

    // Convert to numpy arrays
    // into_pyarray moves the Vec into numpy (no copy), unlike to_pyarray
    let data = result.data.into_pyarray(py).into();
    let rows = result.rows.into_pyarray(py).into();
    let cols = result.cols.into_pyarray(py).into();
    let shape = (result.shape.0 as i64, result.shape.1 as i64);

    Ok((data, (rows, cols), shape))
}

/// Build the coefficient matrix from pre-serialized LinOp data.
///
/// This avoids per-node Python attribute access by accepting pre-flattened data
/// from Python's serialize_linop_trees(). The bulk array data (float_data, int_data)
/// is passed as NumPy arrays for zero-copy access.
///
/// # Arguments
/// * `nodes` - Pre-order list of node tuples: (op_type_int, shape, num_args, data_tag, payload, has_data_linop)
/// * `float_data` - Contiguous buffer of all float data (dense arrays, sparse values)
/// * `int_data` - Contiguous buffer of all int data (sparse indices, indptr)
/// * Other args same as build_matrix
#[pyfunction]
fn build_matrix_serialized<'py>(
    py: Python<'py>,
    nodes: Vec<Bound<'py, PyTuple>>,
    float_data: PyReadonlyArray1<f64>,
    int_data: PyReadonlyArray1<i64>,
    param_size_plus_one: i64,
    id_to_col: HashMap<i64, i64>,
    param_to_size: HashMap<i64, i64>,
    param_to_col: HashMap<i64, i64>,
    var_length: i64,
) -> PyResult<(
    Py<PyArray1<f64>>,
    (Py<PyArray1<i64>>, Py<PyArray1<i64>>),
    (i64, i64),
)> {
    // Get numpy array slices (zero-copy view into Python memory)
    let float_slice = float_data.as_slice()?;
    let int_slice = int_data.as_slice()?;

    // Deserialize LinOp trees from the pre-order stream
    let mut deser_ctx = DeserializationContext::new(&nodes, float_slice, int_slice);
    let mut rust_lin_ops = Vec::new();
    while deser_ctx.cursor < nodes.len() {
        rust_lin_ops.push(deser_ctx.read_linop()?);
    }

    // Build the matrix (release GIL during computation)
    let result = py.detach(|| {
        build_matrix_internal(
            &rust_lin_ops,
            param_size_plus_one,
            &id_to_col,
            &param_to_size,
            &param_to_col,
            var_length,
        )
    });

    // Convert to numpy arrays
    // into_pyarray moves the Vec into numpy (no copy), unlike to_pyarray
    let data = result.data.into_pyarray(py).into();
    let rows = result.rows.into_pyarray(py).into();
    let cols = result.cols.into_pyarray(py).into();
    let shape = (result.shape.0 as i64, result.shape.1 as i64);

    Ok((data, (rows, cols), shape))
}

/// Cached LinOp graph for repeated solves.
///
/// Stores the Rust-side LinOp trees and processing context so that subsequent
/// calls skip Python extraction entirely.  For constraints whose subtrees are
/// parameter-free, the computed SparseTensor is cached and reused verbatim.
#[pyclass]
struct CachedLinOpGraph {
    lin_ops: Vec<LinOp>,
    ctx: ProcessingContext,
    param_size_plus_one: i64,
    /// Per-constraint flag: true if the subtree contains any Param nodes.
    param_dependent: Vec<bool>,
    /// Cached SparseTensor (with row offsets applied) for non-parametric constraints.
    constraint_cache: Vec<Option<SparseTensor>>,
    /// Row offsets for each constraint in the output tensor.
    row_offsets: Vec<i64>,
    /// Total rows across all constraints.
    total_rows: usize,
}

impl CachedLinOpGraph {
    fn build(&mut self) -> BuildMatrixResult {
        use crate::operations::process_linop;

        let total_nnz: usize = self.lin_ops.iter()
            .map(|l| crate::operations::count_nnz(l, &self.ctx))
            .sum();

        let mut combined = SparseTensor::with_capacity(
            (self.total_rows, self.ctx.var_length as usize + 1),
            total_nnz,
        );

        for (i, (lin_op, &row_offset)) in
            self.lin_ops.iter().zip(self.row_offsets.iter()).enumerate()
        {
            if !self.param_dependent[i] {
                // Check if we already have a cached result
                if let Some(ref cached) = self.constraint_cache[i] {
                    combined.extend(cached.clone());
                    continue;
                }
            }

            // Process and cache
            let mut tensor = process_linop(lin_op, &self.ctx);
            tensor.offset_rows_in_place(row_offset);

            if !self.param_dependent[i] {
                self.constraint_cache[i] = Some(tensor.clone());
            }

            combined.extend(tensor);
        }

        BuildMatrixResult::from_tensor(combined, self.param_size_plus_one as usize)
    }
}

/// Build the coefficient matrix AND return a cached graph for future re-use.
///
/// Cold path: extracts LinOp trees from serialized data, builds the matrix,
/// and returns both the result and a CachedLinOpGraph that can be passed to
/// `build_matrix_cached` for subsequent evaluations without Python extraction.
#[pyfunction]
fn build_matrix_and_cache<'py>(
    py: Python<'py>,
    nodes: Vec<Bound<'py, PyTuple>>,
    float_data: PyReadonlyArray1<f64>,
    int_data: PyReadonlyArray1<i64>,
    param_size_plus_one: i64,
    id_to_col: HashMap<i64, i64>,
    param_to_size: HashMap<i64, i64>,
    param_to_col: HashMap<i64, i64>,
    var_length: i64,
) -> PyResult<(
    Py<PyArray1<f64>>,
    (Py<PyArray1<i64>>, Py<PyArray1<i64>>),
    (i64, i64),
    Py<CachedLinOpGraph>,
)> {
    let float_slice = float_data.as_slice()?;
    let int_slice = int_data.as_slice()?;

    let mut deser_ctx = DeserializationContext::new(&nodes, float_slice, int_slice);
    let mut rust_lin_ops = Vec::new();
    while deser_ctx.cursor < nodes.len() {
        rust_lin_ops.push(deser_ctx.read_linop()?);
    }

    // Build processing context
    let mut full_id_to_col = id_to_col;
    full_id_to_col.insert(CONSTANT_ID, var_length);
    let mut full_param_to_col = param_to_col;
    full_param_to_col.insert(CONSTANT_ID, param_size_plus_one - 1);
    let mut full_param_to_size = param_to_size;
    full_param_to_size.insert(CONSTANT_ID, 1);

    let ctx = ProcessingContext {
        id_to_col: full_id_to_col,
        param_to_size: full_param_to_size,
        param_to_col: full_param_to_col,
        var_length,
        param_size_plus_one,
    };

    // Compute per-constraint metadata
    let param_dependent: Vec<bool> = rust_lin_ops.iter().map(|l| l.has_param()).collect();
    let row_offsets: Vec<i64> = rust_lin_ops
        .iter()
        .scan(0i64, |offset, lin_op| {
            let current = *offset;
            *offset += lin_op.size() as i64;
            Some(current)
        })
        .collect();
    let total_rows: usize = rust_lin_ops.iter().map(|l| l.size()).sum();
    let n_constraints = rust_lin_ops.len();

    // Build the graph object
    let mut graph = CachedLinOpGraph {
        lin_ops: rust_lin_ops,
        ctx,
        param_size_plus_one,
        param_dependent,
        constraint_cache: vec![None; n_constraints],
        row_offsets,
        total_rows,
    };

    // Build the matrix (also populates cache for non-parametric constraints)
    let result = graph.build();

    // into_pyarray moves the Vec into numpy (no copy), unlike to_pyarray
    let data = result.data.into_pyarray(py).into();
    let rows = result.rows.into_pyarray(py).into();
    let cols = result.cols.into_pyarray(py).into();
    let shape = (result.shape.0 as i64, result.shape.1 as i64);

    let graph_py = Py::new(py, graph)?;

    Ok((data, (rows, cols), shape, graph_py))
}

/// Hot path: rebuild the coefficient matrix from a cached graph.
///
/// Reuses the Rust-side LinOp trees (no Python extraction) and returns
/// cached results for parameter-free constraints.  Only parametric
/// subtrees are re-processed.
#[pyfunction]
fn build_matrix_cached<'py>(
    py: Python<'py>,
    graph: &Bound<'py, CachedLinOpGraph>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    (Py<PyArray1<i64>>, Py<PyArray1<i64>>),
    (i64, i64),
)> {
    // Build while holding the borrow; no GIL release needed since the
    // graph contains Python-tied data (Arc from PyO3 extraction).
    let mut graph = graph.borrow_mut();
    let result = graph.build();

    // into_pyarray moves the Vec into numpy (no copy), unlike to_pyarray
    let data = result.data.into_pyarray(py).into();
    let rows = result.rows.into_pyarray(py).into();
    let cols = result.cols.into_pyarray(py).into();
    let shape = (result.shape.0 as i64, result.shape.1 as i64);

    Ok((data, (rows, cols), shape))
}

/// Test function for debugging module loading
#[pyfunction]
fn test_function() -> String {
    "cvxpy_rust module loaded successfully".to_string()
}

/// Python module definition
#[pymodule]
fn cvxpy_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(build_matrix_serialized, m)?)?;
    m.add_function(wrap_pyfunction!(build_matrix_and_cache, m)?)?;
    m.add_function(wrap_pyfunction!(build_matrix_cached, m)?)?;
    m.add_function(wrap_pyfunction!(test_function, m)?)?;
    m.add_class::<CachedLinOpGraph>()?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
