//! LinOp representation and Python extraction
//!
//! This module defines the Rust representation of CVXPY's LinOp nodes
//! and provides extraction from Python objects via PyO3.

use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySequence, PyTuple};
use std::fmt;
use std::sync::Arc;

/// Helper to get item from either a list or tuple
fn get_sequence_item<'py>(obj: &Bound<'py, PyAny>, index: usize) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(list) = obj.cast::<PyList>() {
        list.get_item(index)
    } else if let Ok(tuple) = obj.cast::<PyTuple>() {
        tuple.get_item(index)
    } else if let Ok(seq) = obj.cast::<PySequence>() {
        seq.get_item(index)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Expected list or tuple, got {:?}",
            obj.get_type().name()
        )))
    }
}

/// Helper to get length of list or tuple
fn get_sequence_len(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(list) = obj.cast::<PyList>() {
        Ok(list.len())
    } else if let Ok(tuple) = obj.cast::<PyTuple>() {
        Ok(tuple.len())
    } else if let Ok(seq) = obj.cast::<PySequence>() {
        Ok(seq.len()?)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Expected list or tuple, got {:?}",
            obj.get_type().name()
        )))
    }
}

/// Operation types matching CVXPY's lin_op.py
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
    // Leaf nodes
    Variable,
    ScalarConst,
    DenseConst,
    SparseConst,
    Param,

    // Trivial operations
    Sum,
    Neg,
    Reshape,

    // Arithmetic operations
    Mul,
    Rmul,
    MulElem,
    Div,

    // Structural operations
    Index,
    Transpose,
    Promote,
    BroadcastTo,
    Hstack,
    Vstack,
    Concatenate,

    // Specialized operations
    SumEntries,
    Trace,
    DiagVec,
    DiagMat,
    UpperTri,
    Conv,
    KronR,
    KronL,

    // No-op
    NoOp,
}

impl OpType {
    /// Parse operation type from Python string
    pub fn from_str(s: &str) -> PyResult<Self> {
        match s {
            "variable" => Ok(OpType::Variable),
            "scalar_const" => Ok(OpType::ScalarConst),
            "dense_const" => Ok(OpType::DenseConst),
            "sparse_const" => Ok(OpType::SparseConst),
            "param" => Ok(OpType::Param),
            "sum" => Ok(OpType::Sum),
            "neg" => Ok(OpType::Neg),
            "reshape" => Ok(OpType::Reshape),
            "mul" => Ok(OpType::Mul),
            "rmul" => Ok(OpType::Rmul),
            "mul_elem" => Ok(OpType::MulElem),
            "div" => Ok(OpType::Div),
            "index" => Ok(OpType::Index),
            "transpose" => Ok(OpType::Transpose),
            "promote" => Ok(OpType::Promote),
            "broadcast_to" => Ok(OpType::BroadcastTo),
            "hstack" => Ok(OpType::Hstack),
            "vstack" => Ok(OpType::Vstack),
            "concatenate" => Ok(OpType::Concatenate),
            "sum_entries" => Ok(OpType::SumEntries),
            "trace" => Ok(OpType::Trace),
            "diag_vec" => Ok(OpType::DiagVec),
            "diag_mat" => Ok(OpType::DiagMat),
            "upper_tri" => Ok(OpType::UpperTri),
            "conv" => Ok(OpType::Conv),
            "kron_r" => Ok(OpType::KronR),
            "kron_l" => Ok(OpType::KronL),
            "no_op" => Ok(OpType::NoOp),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown operation type: {}",
                s
            ))),
        }
    }

    /// Check if this is a leaf node type. Used to skip the `args` PyO3
    /// fetch in `LinOp::from_python`.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        matches!(
            self,
            OpType::Variable
                | OpType::ScalarConst
                | OpType::DenseConst
                | OpType::SparseConst
                | OpType::Param
        )
    }
}

/// Data associated with a LinOp node
#[derive(Debug, Clone)]
pub enum LinOpData {
    None,
    Int(i64),
    Float(f64),
    DenseArray {
        data: Arc<[f64]>,
        shape: Vec<usize>,
    },
    SparseArray {
        data: Arc<[f64]>,
        indices: Arc<[i64]>,
        indptr: Arc<[i64]>,
        shape: (usize, usize),
    },
    Slices(Vec<SliceData>),
    LinOpRef(Box<LinOp>),
    /// For sum_entries and transpose: (axis, keepdims) or axis tuple
    AxisData {
        axis: Option<AxisSpec>,
        keepdims: bool,
    },
    /// For concatenate: axis
    ConcatAxis(Option<i64>),
}

/// Slice specification (start, stop, step)
#[derive(Debug, Clone)]
pub struct SliceData {
    pub start: i64,
    pub stop: i64,
    pub step: i64,
}

/// Axis specification - can be single axis or tuple of axes
#[derive(Debug, Clone)]
pub enum AxisSpec {
    Single(i64),
    Multiple(Vec<i64>),
}

/// A node in the LinOp expression tree
#[derive(Debug, Clone)]
pub struct LinOp {
    pub op_type: OpType,
    pub shape: Vec<usize>,
    pub args: Vec<LinOp>,
    pub data: LinOpData,
}

impl LinOp {
    /// Extract a LinOp tree from a Python object
    pub fn from_python(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Extract op_type via a borrowed PyString — avoids the per-node
        // String allocation that `extract::<String>()` does.
        let type_attr = obj.getattr("type")?;
        let type_pystr = type_attr.cast::<pyo3::types::PyString>()?;
        let op_type = OpType::from_str(type_pystr.to_str()?)?;

        // Extract shape
        let shape: Vec<usize> = obj.getattr("shape")?.extract()?;

        // Skip the `args` getattr entirely for leaf nodes (Variable, *Const,
        // Param, NoOp). On a typical LASSO LinOp tree leaves are roughly a
        // third of all nodes, and each saved getattr is two PyO3 calls.
        let args = if op_type.is_leaf() {
            Vec::new()
        } else {
            let args_list = obj.getattr("args")?;
            let args_list = args_list.cast::<PyList>()?;
            args_list
                .iter()
                .map(|arg| LinOp::from_python(&arg))
                .collect::<PyResult<Vec<_>>>()?
        };

        // Extract data based on operation type
        let data = Self::extract_data(obj, op_type)?;

        Ok(LinOp {
            op_type,
            shape,
            args,
            data,
        })
    }

    /// Extract data field based on operation type
    fn extract_data(obj: &Bound<'_, PyAny>, op_type: OpType) -> PyResult<LinOpData> {
        let data_attr = obj.getattr("data")?;

        if data_attr.is_none() {
            return Ok(LinOpData::None);
        }

        match op_type {
            OpType::Variable | OpType::Param => {
                // data is int (id)
                let id: i64 = data_attr.extract()?;
                Ok(LinOpData::Int(id))
            }

            OpType::ScalarConst => {
                // data is a scalar value
                let value: f64 = data_attr.extract()?;
                Ok(LinOpData::Float(value))
            }

            OpType::DenseConst => {
                // data is a numpy array
                Self::extract_dense_array(&data_attr)
            }

            OpType::SparseConst => {
                // data is a scipy sparse matrix
                Self::extract_sparse_array(&data_attr)
            }

            OpType::Index => {
                // data is list of slices
                Self::extract_slices(&data_attr)
            }

            OpType::Mul
            | OpType::Rmul
            | OpType::MulElem
            | OpType::Div
            | OpType::Conv
            | OpType::KronL
            | OpType::KronR => {
                // data is another LinOp tree (the constant operand)
                let inner = LinOp::from_python(&data_attr)?;
                Ok(LinOpData::LinOpRef(Box::new(inner)))
            }

            OpType::DiagVec | OpType::DiagMat => {
                // data is int (diagonal offset k)
                let k: i64 = data_attr.extract()?;
                Ok(LinOpData::Int(k))
            }

            OpType::Transpose => {
                // data is (axes,) tuple or list - extract axes permutation
                let len = get_sequence_len(&data_attr)?;
                if len > 0 {
                    let axes = get_sequence_item(&data_attr, 0)?;
                    if axes.is_none() {
                        Ok(LinOpData::None)
                    } else {
                        let axes_vec: Vec<i64> = axes.extract()?;
                        Ok(LinOpData::AxisData {
                            axis: Some(AxisSpec::Multiple(axes_vec)),
                            keepdims: false,
                        })
                    }
                } else {
                    Ok(LinOpData::None)
                }
            }

            OpType::SumEntries => {
                // data is [axis, keepdims] list or tuple
                let axis = get_sequence_item(&data_attr, 0)?;
                let keepdims: bool = get_sequence_item(&data_attr, 1)?.extract().unwrap_or(false);

                let axis_spec = if axis.is_none() {
                    None
                } else if let Ok(single) = axis.extract::<i64>() {
                    Some(AxisSpec::Single(single))
                } else if let Ok(multi) = axis.extract::<Vec<i64>>() {
                    Some(AxisSpec::Multiple(multi))
                } else {
                    None
                };

                Ok(LinOpData::AxisData {
                    axis: axis_spec,
                    keepdims,
                })
            }

            OpType::Concatenate => {
                // data is [axis] list or tuple
                let axis = get_sequence_item(&data_attr, 0)?;
                if axis.is_none() {
                    Ok(LinOpData::ConcatAxis(None))
                } else {
                    let axis_val: i64 = axis.extract()?;
                    Ok(LinOpData::ConcatAxis(Some(axis_val)))
                }
            }

            _ => Ok(LinOpData::None),
        }
    }

    /// Extract dense numpy array data
    fn extract_dense_array(data_attr: &Bound<'_, PyAny>) -> PyResult<LinOpData> {
        let arr = data_attr.cast::<PyArrayDyn<f64>>()?;
        let shape: Vec<usize> = arr.shape().to_vec();

        // Fast path: cvxpy stores its dense constants F-contiguous, so the
        // underlying buffer is already in the layout we want. Read it
        // directly via the numpy crate without bouncing through a Python
        // `ravel("F")` call. This skips a Python attribute lookup + method
        // dispatch + temporary numpy view per dense constant — small but
        // it's per-LinOp.
        if arr.is_fortran_contiguous() {
            let readonly = arr.readonly();
            let slice = readonly.as_slice()?;
            return Ok(LinOpData::DenseArray {
                data: Arc::from(slice.to_vec()),
                shape,
            });
        }

        // Fallback for non-contiguous numpy views / slices.
        let flat_arr = data_attr.call_method1("ravel", ("F",))?;
        let data: Vec<f64> = flat_arr.extract()?;
        Ok(LinOpData::DenseArray {
            data: Arc::from(data),
            shape,
        })
    }

    /// Extract sparse scipy matrix data (assumes CSC format)
    fn extract_sparse_array(data_attr: &Bound<'_, PyAny>) -> PyResult<LinOpData> {
        // Convert to CSC if needed
        let csc = data_attr.call_method0("tocsc")?;

        let data: Vec<f64> = csc.getattr("data")?.extract()?;
        let indices: Vec<i64> = csc.getattr("indices")?.extract()?;
        let indptr: Vec<i64> = csc.getattr("indptr")?.extract()?;
        let shape: (usize, usize) = csc.getattr("shape")?.extract()?;

        Ok(LinOpData::SparseArray {
            data: Arc::from(data),
            indices: Arc::from(indices),
            indptr: Arc::from(indptr),
            shape,
        })
    }

    /// Extract slice data for index operation
    fn extract_slices(data_attr: &Bound<'_, PyAny>) -> PyResult<LinOpData> {
        let len = get_sequence_len(data_attr)?;
        let mut slices = Vec::with_capacity(len);

        for i in 0..len {
            let item = get_sequence_item(data_attr, i)?;
            // Each item is a slice object
            let start: i64 = item.getattr("start")?.extract()?;
            let stop: i64 = item.getattr("stop")?.extract()?;
            let step: i64 = item.getattr("step")?.extract().unwrap_or(1);
            slices.push(SliceData { start, stop, step });
        }

        Ok(LinOpData::Slices(slices))
    }

    /// Get the total number of elements in the output
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Estimate the number of non-zeros for pre-allocation
    pub fn estimate_nnz(&self) -> usize {
        match self.op_type {
            OpType::Variable => self.size(),
            OpType::ScalarConst => 1,
            OpType::DenseConst => self.size(),
            OpType::SparseConst => {
                if let LinOpData::SparseArray { ref data, .. } = self.data {
                    data.len()
                } else {
                    self.size()
                }
            }
            OpType::Param => self.size(),
            OpType::Neg | OpType::Reshape => self.args.first().map_or(0, |a| a.estimate_nnz()),
            OpType::Mul | OpType::Rmul => {
                // Estimate based on data nnz and number of blocks
                let data_nnz = match &self.data {
                    LinOpData::LinOpRef(ref inner) => inner.estimate_nnz(),
                    _ => self.size(),
                };
                let num_blocks = self
                    .args
                    .first()
                    .map_or(1, |a| a.shape.get(1).copied().unwrap_or(1));
                data_nnz * num_blocks
            }
            OpType::Hstack | OpType::Vstack | OpType::Concatenate => {
                self.args.iter().map(|a| a.estimate_nnz()).sum()
            }
            OpType::KronR | OpType::KronL => {
                let data_size = match &self.data {
                    LinOpData::LinOpRef(ref inner) => inner.size(),
                    _ => 1,
                };
                let arg_nnz = self.args.first().map_or(1, |a| a.estimate_nnz());
                data_size * arg_nnz
            }
            _ => self
                .args
                .iter()
                .map(|a| a.estimate_nnz())
                .sum::<usize>()
                .max(self.size()),
        }
    }
}

impl fmt::Display for LinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LinOp({:?}, shape={:?}, {} args)",
            self.op_type,
            self.shape,
            self.args.len()
        )
    }
}
