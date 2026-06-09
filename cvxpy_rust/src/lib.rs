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
use std::collections::HashMap;

use crate::linop::{DeserializationContext, LinOp};
use crate::matrix_builder::build_matrix_internal;

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
/// * `node_meta` - Flat i64 metadata stream, pre-order per node (see
///   serialize_linop_trees in canon_backend.py for the layout)
/// * `float_data` - Contiguous buffer of all float data (dense arrays, sparse values)
/// * `int_data` - Contiguous buffer of all int data (sparse indices, indptr)
/// * Other args same as build_matrix
#[pyfunction]
fn build_matrix_serialized<'py>(
    py: Python<'py>,
    node_meta: PyReadonlyArray1<i64>,
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
    // Phase timing for FFI-overhead analysis (e.g. the arena-allocator
    // question): how much of a call is spent deserializing nodes vs building.
    // See rust_benchmarks/FFI_OVERHEAD_ANALYSIS.md for measured results.
    let profile = std::env::var("CVXPY_RUST_FFI_PROFILE").as_deref() == Ok("1");
    let t0 = if profile { Some(std::time::Instant::now()) } else { None };

    // Get numpy array slices (zero-copy view into Python memory)
    let meta_slice = node_meta.as_slice()?;
    let float_slice = float_data.as_slice()?;
    let int_slice = int_data.as_slice()?;

    // Deserialize LinOp trees from the pre-order stream
    let mut deser_ctx = DeserializationContext::new(meta_slice, float_slice, int_slice);
    let mut rust_lin_ops = Vec::new();
    while !deser_ctx.done() {
        rust_lin_ops.push(deser_ctx.read_linop()?);
    }

    let deser_ms = t0.map(|t| t.elapsed().as_secs_f64() * 1000.0);
    let t1 = if profile { Some(std::time::Instant::now()) } else { None };

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

    if let (Some(deser), Some(t)) = (deser_ms, t1) {
        let build = t.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[cvxpy_rust] build_matrix_serialized: trees={}, meta_len={}, deser={:.3}ms, build={:.3}ms, deser_share={:.1}%",
            rust_lin_ops.len(),
            meta_slice.len(),
            deser,
            build,
            100.0 * deser / (deser + build).max(1e-9),
        );
    }

    // Convert to numpy arrays
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
    m.add_function(wrap_pyfunction!(test_function, m)?)?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
