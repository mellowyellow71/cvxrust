//! LinOp representation and Python extraction
//!
//! This module defines the Rust representation of CVXPY's LinOp nodes
//! and provides extraction from Python objects via PyO3.

use numpy::{PyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySequence, PyTuple};
use std::fmt;
use std::sync::Arc;

/// Helper to get item from either a list or tuple
fn get_sequence_item<'py>(obj: &Bound<'py, PyAny>, index: usize) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(list) = obj.downcast::<PyList>() {
        list.get_item(index)
    } else if let Ok(tuple) = obj.downcast::<PyTuple>() {
        tuple.get_item(index)
    } else if let Ok(seq) = obj.downcast::<PySequence>() {
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
    if let Ok(list) = obj.downcast::<PyList>() {
        Ok(list.len())
    } else if let Ok(tuple) = obj.downcast::<PyTuple>() {
        Ok(tuple.len())
    } else if let Ok(seq) = obj.downcast::<PySequence>() {
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
    /// Parse operation type from integer (used by serialized path)
    pub fn from_int(i: u8) -> PyResult<Self> {
        match i {
            0 => Ok(OpType::Variable),
            1 => Ok(OpType::ScalarConst),
            2 => Ok(OpType::DenseConst),
            3 => Ok(OpType::SparseConst),
            4 => Ok(OpType::Param),
            5 => Ok(OpType::Sum),
            6 => Ok(OpType::Neg),
            7 => Ok(OpType::Reshape),
            8 => Ok(OpType::Mul),
            9 => Ok(OpType::Rmul),
            10 => Ok(OpType::MulElem),
            11 => Ok(OpType::Div),
            12 => Ok(OpType::Index),
            13 => Ok(OpType::Transpose),
            14 => Ok(OpType::Promote),
            15 => Ok(OpType::BroadcastTo),
            16 => Ok(OpType::Hstack),
            17 => Ok(OpType::Vstack),
            18 => Ok(OpType::Concatenate),
            19 => Ok(OpType::SumEntries),
            20 => Ok(OpType::Trace),
            21 => Ok(OpType::DiagVec),
            22 => Ok(OpType::DiagMat),
            23 => Ok(OpType::UpperTri),
            24 => Ok(OpType::Conv),
            25 => Ok(OpType::KronR),
            26 => Ok(OpType::KronL),
            27 => Ok(OpType::NoOp),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown op type int: {}",
                i
            ))),
        }
    }

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

    /// Check if this is a leaf node type
    #[allow(dead_code)]
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
        // Extract operation type
        let type_str: String = obj.getattr("type")?.extract()?;
        let op_type = OpType::from_str(&type_str)?;

        // Extract shape
        let shape: Vec<usize> = obj.getattr("shape")?.extract()?;

        // Extract args recursively
        let args_list = obj.getattr("args")?;
        let args_list = args_list.downcast::<PyList>()?;
        let args: Vec<LinOp> = args_list
            .iter()
            .map(|arg| LinOp::from_python(&arg))
            .collect::<PyResult<Vec<_>>>()?;

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
        let arr = data_attr.downcast::<PyArrayDyn<f64>>()?;
        let shape: Vec<usize> = arr.shape().to_vec();
        // CVXPY stores constants in F-order (column-major), so we need to read in F-order.
        // Call numpy's ravel with order='F' to get flattened data in column-major order.
        // This handles non-contiguous arrays (views, slices) correctly.
        let flat_arr = data_attr.call_method1("ravel", ("F",))?;
        let data: Vec<f64> = flat_arr.extract()?;
        Ok(LinOpData::DenseArray { data: Arc::from(data), shape })
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

/// Context for deserializing LinOp trees from pre-serialized Python data.
///
/// The Python side serializes the tree in pre-order traversal into a flat list
/// of node tuples, plus contiguous float/int buffers for array data.
/// This avoids per-node Python attribute access (the main FFI bottleneck).
pub struct DeserializationContext<'a, 'py> {
    nodes: &'a [Bound<'py, PyTuple>],
    float_data: &'a [f64],
    int_data: &'a [i64],
    pub cursor: usize,
}

impl<'a, 'py> DeserializationContext<'a, 'py> {
    pub fn new(
        nodes: &'a [Bound<'py, PyTuple>],
        float_data: &'a [f64],
        int_data: &'a [i64],
    ) -> Self {
        DeserializationContext {
            nodes,
            float_data,
            int_data,
            cursor: 0,
        }
    }

    /// Read the next LinOp from the stream (recursive, pre-order)
    pub fn read_linop(&mut self) -> PyResult<LinOp> {
        if self.cursor >= self.nodes.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Unexpected end of serialized node stream",
            ));
        }

        let node = &self.nodes[self.cursor];
        self.cursor += 1;

        let op_type_int: u8 = node.get_item(0)?.extract()?;
        let op_type = OpType::from_int(op_type_int)?;

        let shape: Vec<usize> = node.get_item(1)?.extract()?;
        let num_args: usize = node.get_item(2)?.extract()?;
        let data_tag: u8 = node.get_item(3)?.extract()?;
        let payload = node.get_item(4)?;
        let has_data_linop: u8 = node.get_item(5)?.extract()?;

        // Extract non-LinOpRef data
        let data = self.extract_data(data_tag, &payload)?;

        // If data is a LinOpRef, read the inline data LinOp (serialized before args)
        let data = if has_data_linop == 1 {
            let data_linop = self.read_linop()?;
            LinOpData::LinOpRef(Box::new(data_linop))
        } else {
            data
        };

        // Read args in order
        let mut args = Vec::with_capacity(num_args);
        for _ in 0..num_args {
            args.push(self.read_linop()?);
        }

        Ok(LinOp {
            op_type,
            shape,
            args,
            data,
        })
    }

    fn extract_data(
        &self,
        data_tag: u8,
        payload: &Bound<'py, PyAny>,
    ) -> PyResult<LinOpData> {
        match data_tag {
            0 => Ok(LinOpData::None),

            1 => {
                // Int
                let v: i64 = payload.extract()?;
                Ok(LinOpData::Int(v))
            }

            2 => {
                // Float
                let v: f64 = payload.extract()?;
                Ok(LinOpData::Float(v))
            }

            3 => {
                // DenseArray: (float_offset, float_len, shape_tuple)
                let tup: &Bound<'py, PyTuple> = payload.downcast()?;
                let offset: usize = tup.get_item(0)?.extract()?;
                let len: usize = tup.get_item(1)?.extract()?;
                let shape: Vec<usize> = tup.get_item(2)?.extract()?;
                let data = Arc::from(&self.float_data[offset..offset + len]);
                Ok(LinOpData::DenseArray { data, shape })
            }

            4 => {
                // SparseArray: (f_off, f_len, i_off_idx, i_len_idx, i_off_ptr, i_len_ptr, nrows, ncols)
                let tup: &Bound<'py, PyTuple> = payload.downcast()?;
                let f_off: usize = tup.get_item(0)?.extract()?;
                let f_len: usize = tup.get_item(1)?.extract()?;
                let i_off_idx: usize = tup.get_item(2)?.extract()?;
                let i_len_idx: usize = tup.get_item(3)?.extract()?;
                let i_off_ptr: usize = tup.get_item(4)?.extract()?;
                let i_len_ptr: usize = tup.get_item(5)?.extract()?;
                let nrows: usize = tup.get_item(6)?.extract()?;
                let ncols: usize = tup.get_item(7)?.extract()?;

                let data = Arc::from(&self.float_data[f_off..f_off + f_len]);
                let indices = Arc::from(&self.int_data[i_off_idx..i_off_idx + i_len_idx]);
                let indptr = Arc::from(&self.int_data[i_off_ptr..i_off_ptr + i_len_ptr]);

                Ok(LinOpData::SparseArray {
                    data,
                    indices,
                    indptr,
                    shape: (nrows, ncols),
                })
            }

            5 => {
                // Slices: list of (start, stop, step) tuples
                let list: Vec<(i64, i64, i64)> = payload.extract()?;
                let slices = list
                    .into_iter()
                    .map(|(start, stop, step)| SliceData { start, stop, step })
                    .collect();
                Ok(LinOpData::Slices(slices))
            }

            6 => {
                // LinOpRef placeholder — handled by caller
                Ok(LinOpData::None)
            }

            7 => {
                // AxisData: (axis_spec, keepdims)
                let tup: &Bound<'py, PyTuple> = payload.downcast()?;
                let axis_obj = tup.get_item(0)?;
                let keepdims: bool = tup.get_item(1)?.extract()?;
                let axis = if axis_obj.is_none() {
                    None
                } else if let Ok(single) = axis_obj.extract::<i64>() {
                    Some(AxisSpec::Single(single))
                } else if let Ok(multi) = axis_obj.extract::<Vec<i64>>() {
                    Some(AxisSpec::Multiple(multi))
                } else {
                    None
                };
                Ok(LinOpData::AxisData { axis, keepdims })
            }

            8 => {
                // ConcatAxis
                if payload.is_none() {
                    Ok(LinOpData::ConcatAxis(None))
                } else {
                    let v: i64 = payload.extract()?;
                    Ok(LinOpData::ConcatAxis(Some(v)))
                }
            }

            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown data tag: {}",
                data_tag
            ))),
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
