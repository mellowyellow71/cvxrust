# Optimization 1: Batch LinOp Tree Serialization — Detailed Code Plan

## Problem Statement

`LinOp::from_python()` (`linop.rs:194-218`) extracts the LinOp tree from Python one node
at a time. For each node, it makes 4+ PyO3 calls:

```rust
let type_str: String = obj.getattr("type")?.extract()?;     // 1. getattr + extract
let shape: Vec<usize> = obj.getattr("shape")?.extract()?;   // 2. getattr + extract
let args_list = obj.getattr("args")?;                        // 3. getattr
// ... then recursively for each arg
let data = Self::extract_data(obj, op_type)?;                // 4+ more getattr calls
```

For data extraction, additional calls are needed:
- `DenseConst`: `obj.getattr("data")` + `data.call_method1("ravel", ("F",))` + extract
- `SparseConst`: `obj.getattr("data")` + `.call_method0("tocsc")` + 3x getattr (data, indices, indptr, shape)
- `LinOpRef` (Mul/Rmul/etc.): **recursive** `LinOp::from_python()` on the data field

For a tree with N nodes, this is O(N) Python API round-trips, each with GIL acquisition
overhead, type checking, and reference counting.

## Solution: Python-Side Serialization

Walk the tree once in Python (cheap — Python accessing Python objects), pack everything
into NumPy arrays and a metadata list, then pass it all to Rust in a single FFI call.

---

## Part 1: Python Serializer (`canon_backend.py`)

Add a function that walks the LinOp tree and produces:
- `nodes: list[tuple]` — flat list of node metadata, in pre-order traversal
- `float_data: np.ndarray` — contiguous buffer of all float data (dense arrays, sparse values)
- `int_data: np.ndarray` — contiguous buffer of all int data (sparse indices, indptr, shapes)

### Node Metadata Tuple Format

Each node is a tuple:
```python
(op_type_int, shape_tuple, num_args, data_tag, data_payload)
```

Where:
- `op_type_int`: int mapping of op type (0=variable, 1=scalar_const, ..., 27=no_op)
- `shape_tuple`: tuple of ints
- `num_args`: int — how many children follow in the pre-order sequence
- `data_tag`: int indicating what `data_payload` contains:
  - 0 = None
  - 1 = Int (payload is the int value)
  - 2 = Float (payload is the float value)
  - 3 = DenseArray (payload is (float_offset, float_len, shape_tuple))
  - 4 = SparseArray (payload is (float_offset, float_len, int_offset_indices, int_len_indices, int_offset_indptr, int_len_indptr, nrows, ncols))
  - 5 = Slices (payload is list of (start, stop, step) tuples)
  - 6 = LinOpRef (payload is None — the data linop is serialized inline as the next node in the stream, before the args)
  - 7 = AxisData (payload is (axis_spec, keepdims))
  - 8 = ConcatAxis (payload is axis_int_or_none)
- `data_payload`: varies by tag

### Serialization Function

```python
# In canon_backend.py

# Op type string -> int mapping
_OP_TYPE_MAP = {
    "variable": 0, "scalar_const": 1, "dense_const": 2, "sparse_const": 3,
    "param": 4, "sum": 5, "neg": 6, "reshape": 7, "mul": 8, "rmul": 9,
    "mul_elem": 10, "div": 11, "index": 12, "transpose": 13, "promote": 14,
    "broadcast_to": 15, "hstack": 16, "vstack": 17, "concatenate": 18,
    "sum_entries": 19, "trace": 20, "diag_vec": 21, "diag_mat": 22,
    "upper_tri": 23, "conv": 24, "kron_r": 25, "kron_l": 26, "no_op": 27,
}

# Op types whose data field is a LinOp (LinOpRef)
_LINOP_DATA_OPS = {"mul", "rmul", "mul_elem", "div", "conv", "kron_l", "kron_r"}


def serialize_linop_trees(lin_ops: list[LinOp]) -> tuple:
    """Serialize a list of LinOp trees into flat buffers for Rust.

    Returns:
        (nodes, float_data, int_data) where:
        - nodes: list of tuples (op_type, shape, num_args, data_tag, data_payload)
        - float_data: np.ndarray[f64] — all dense/sparse float data concatenated
        - int_data: np.ndarray[i64] — all sparse int data concatenated
    """
    nodes = []
    float_chunks = []
    int_chunks = []
    float_offset = 0
    int_offset = 0

    def _serialize_node(lin_op):
        nonlocal float_offset, int_offset

        op_type_int = _OP_TYPE_MAP[lin_op.type]
        shape = tuple(lin_op.shape)
        num_args = len(lin_op.args)
        has_data_linop = lin_op.type in _LINOP_DATA_OPS and lin_op.data is not None

        # Determine data tag and payload
        if lin_op.data is None:
            data_tag, payload = 0, None

        elif lin_op.type in ("variable", "param"):
            data_tag, payload = 1, int(lin_op.data)

        elif lin_op.type == "scalar_const":
            data_tag, payload = 2, float(lin_op.data)

        elif lin_op.type == "dense_const":
            arr = np.asarray(lin_op.data, dtype=np.float64)
            flat = arr.ravel(order='F')
            float_chunks.append(flat)
            payload = (float_offset, len(flat), tuple(arr.shape))
            float_offset += len(flat)
            data_tag = 3

        elif lin_op.type == "sparse_const":
            csc = sp.csc_array(lin_op.data)
            vals = np.asarray(csc.data, dtype=np.float64)
            indices = np.asarray(csc.indices, dtype=np.int64)
            indptr = np.asarray(csc.indptr, dtype=np.int64)
            float_chunks.append(vals)
            int_chunks.append(indices)
            int_chunks.append(indptr)
            payload = (
                float_offset, len(vals),
                int_offset, len(indices),
                int_offset + len(indices), len(indptr),
                csc.shape[0], csc.shape[1],
            )
            float_offset += len(vals)
            int_offset += len(indices) + len(indptr)
            data_tag = 4

        elif lin_op.type == "index":
            slices = [(s.start, s.stop, s.step) for s in lin_op.data]
            data_tag, payload = 5, slices

        elif lin_op.type in _LINOP_DATA_OPS:
            # Data is a LinOp — serialize inline, tag=6
            data_tag, payload = 6, None

        elif lin_op.type in ("diag_vec", "diag_mat"):
            data_tag, payload = 1, int(lin_op.data)

        elif lin_op.type == "sum_entries":
            axis, keepdims = lin_op.data[0], lin_op.data[1]
            data_tag, payload = 7, (axis, keepdims)

        elif lin_op.type == "transpose":
            if lin_op.data is not None and len(lin_op.data) > 0:
                axes = lin_op.data[0]
                data_tag, payload = 7, (axes, False)
            else:
                data_tag, payload = 0, None

        elif lin_op.type == "concatenate":
            axis = lin_op.data[0] if lin_op.data else None
            data_tag, payload = 8, axis

        else:
            data_tag, payload = 0, None

        nodes.append((op_type_int, shape, num_args, data_tag, payload,
                       1 if has_data_linop else 0))

        # If data is a LinOp, serialize it BEFORE args (so Rust reads it first)
        if has_data_linop:
            _serialize_node(lin_op.data)

        # Serialize args in order
        for arg in lin_op.args:
            _serialize_node(arg)

    for lin_op in lin_ops:
        _serialize_node(lin_op)

    # Concatenate float and int buffers
    if float_chunks:
        float_data = np.concatenate(float_chunks)
    else:
        float_data = np.empty(0, dtype=np.float64)

    if int_chunks:
        int_data = np.concatenate(int_chunks)
    else:
        int_data = np.empty(0, dtype=np.int64)

    return nodes, float_data, int_data
```

### Update `RustCanonBackend.build_matrix()`

```python
class RustCanonBackend(CanonBackend):
    def build_matrix(self, lin_ops: list[LinOp]) -> sp.csc_array:
        import cvxpy_rust
        self.id_to_col[-1] = self.var_length

        # Serialize on Python side (fast — Python-to-Python)
        nodes, float_data, int_data = serialize_linop_trees(lin_ops)

        # Single FFI call to Rust
        (data, (row, col), shape) = cvxpy_rust.build_matrix_serialized(
            nodes, float_data, int_data,
            self.param_size_plus_one,
            self.id_to_col,
            self.param_to_size,
            self.param_to_col,
            self.var_length,
        )

        self.id_to_col.pop(-1)
        return sp.csc_array((data, (row, col)), shape)
```

---

## Part 2: Rust Deserializer (`linop.rs`)

Add a `from_serialized()` method that reads from the flat buffers.

### Op Type Integer Mapping

```rust
// In linop.rs
impl OpType {
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
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown op type int: {}", i)
            )),
        }
    }
}
```

### Deserialization Context

```rust
// In linop.rs

/// Context for deserializing a stream of serialized nodes
pub struct DeserializationContext<'a> {
    nodes: &'a [&'a PyTuple],  // Pre-order list of node tuples
    float_data: &'a [f64],     // Shared float buffer
    int_data: &'a [i64],       // Shared int buffer
    cursor: usize,             // Current position in nodes list
}

impl<'a> DeserializationContext<'a> {
    pub fn new(
        nodes: &'a [&'a PyTuple],
        float_data: &'a [f64],
        int_data: &'a [i64],
    ) -> Self {
        DeserializationContext {
            nodes, float_data, int_data, cursor: 0,
        }
    }

    /// Read the next LinOp from the stream (recursive)
    pub fn read_linop(&mut self) -> PyResult<LinOp> {
        let node = self.nodes[self.cursor];
        self.cursor += 1;

        let op_type_int: u8 = node.get_item(0)?.extract()?;
        let op_type = OpType::from_int(op_type_int)?;

        let shape: Vec<usize> = node.get_item(1)?.extract()?;
        let num_args: usize = node.get_item(2)?.extract()?;
        let data_tag: u8 = node.get_item(3)?.extract()?;
        let payload = node.get_item(4)?;
        let has_data_linop: u8 = node.get_item(5)?.extract()?;

        // Extract data
        let data = self.extract_data(op_type, data_tag, &payload)?;

        // If data is a LinOpRef, read the inline data LinOp
        let data = if has_data_linop == 1 {
            let data_linop = self.read_linop()?;
            LinOpData::LinOpRef(Box::new(data_linop))
        } else {
            data
        };

        // Read args
        let mut args = Vec::with_capacity(num_args);
        for _ in 0..num_args {
            args.push(self.read_linop()?);
        }

        Ok(LinOp { op_type, shape, args, data })
    }

    fn extract_data(
        &self,
        op_type: OpType,
        data_tag: u8,
        payload: &Bound<'_, PyAny>,
    ) -> PyResult<LinOpData> {
        match data_tag {
            0 => Ok(LinOpData::None),

            1 => { // Int
                let v: i64 = payload.extract()?;
                Ok(LinOpData::Int(v))
            }

            2 => { // Float
                let v: f64 = payload.extract()?;
                Ok(LinOpData::Float(v))
            }

            3 => { // DenseArray — read from float_data buffer
                let tup: &PyTuple = payload.downcast()?;
                let offset: usize = tup.get_item(0)?.extract()?;
                let len: usize = tup.get_item(1)?.extract()?;
                let shape: Vec<usize> = tup.get_item(2)?.extract()?;
                let data = Arc::from(&self.float_data[offset..offset + len]);
                Ok(LinOpData::DenseArray { data, shape })
            }

            4 => { // SparseArray — read from float_data + int_data buffers
                let tup: &PyTuple = payload.downcast()?;
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
                    data, indices, indptr, shape: (nrows, ncols),
                })
            }

            5 => { // Slices
                let list: Vec<(i64, i64, i64)> = payload.extract()?;
                let slices = list.into_iter()
                    .map(|(start, stop, step)| SliceData { start, stop, step })
                    .collect();
                Ok(LinOpData::Slices(slices))
            }

            6 => { // LinOpRef placeholder — handled by caller
                Ok(LinOpData::None)
            }

            7 => { // AxisData
                let tup: &PyTuple = payload.downcast()?;
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

            8 => { // ConcatAxis
                if payload.is_none() {
                    Ok(LinOpData::ConcatAxis(None))
                } else {
                    let v: i64 = payload.extract()?;
                    Ok(LinOpData::ConcatAxis(Some(v)))
                }
            }

            _ => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown data tag: {}", data_tag)
            )),
        }
    }
}
```

### Key Design Decision: `Arc<[f64]>` Slicing

The `DenseArray` and `SparseArray` variants use `Arc<[f64]>` for zero-copy sharing.
In the serialized path, we create slices of the shared float/int buffers:

```rust
// Option A: Copy slices (simpler, current approach works unchanged)
let data = Arc::from(&self.float_data[offset..offset + len]);

// Option B: Share the whole buffer via Arc (zero-copy, but needs refactor)
// Would require changing LinOpData to use (Arc<[f64]>, Range<usize>) pairs
// Skip this for now — the copy is fast and keeps the existing code working.
```

**Decision: Use Option A (copy slices).** The copy is O(n) memcpy which is fast, and it
avoids changing the `LinOpData` struct. We can optimize to zero-copy later if profiling
shows this matters.

---

## Part 3: New PyO3 Entry Point (`lib.rs`)

```rust
// In lib.rs

/// Build matrix from pre-serialized LinOp data.
///
/// This avoids per-node Python attribute access by accepting pre-flattened data.
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
    // Get numpy array views (zero-copy)
    let float_slice = float_data.as_slice()?;
    let int_slice = int_data.as_slice()?;

    // Convert Bound<PyTuple> references for the deserializer
    let node_refs: Vec<&PyTuple> = nodes.iter().map(|n| n.as_ref()).collect();

    // Deserialize LinOp trees
    let mut deser_ctx = DeserializationContext::new(&node_refs, float_slice, int_slice);

    // Read all top-level LinOps
    // We need to know how many there are — count by reading until cursor exhausted
    let mut rust_lin_ops = Vec::new();
    while deser_ctx.cursor < deser_ctx.nodes.len() {
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
    let data = result.data.to_pyarray(py).into();
    let rows = result.rows.to_pyarray(py).into();
    let cols = result.cols.to_pyarray(py).into();
    let shape = (result.shape.0 as i64, result.shape.1 as i64);

    Ok((data, (rows, cols), shape))
}

// Register in module
#[pymodule]
fn cvxpy_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(build_matrix_serialized, m)?)?;
    m.add_function(wrap_pyfunction!(test_function, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
```

**Important:** The deserialization still touches Python objects (the `nodes` list of
tuples), but the key improvement is:
1. **No recursive Python attribute access** — no `getattr("type")`, `getattr("shape")`, etc.
2. **Bulk data in NumPy arrays** — `float_data` and `int_data` are zero-copy via PyReadonlyArray
3. **No Python method calls** — no `.ravel("F")`, no `.tocsc()`, no `.downcast::<PyList>()`
4. **Pre-computed sparse conversions** — CSC conversion happens once in Python, not per-node

---

## Part 4: Rust Dependencies

Add `numpy` readonly array support to `Cargo.toml` if not already present:
```toml
[dependencies]
numpy = { version = "0.24", features = ["nalgebra"] }  # or whatever version is used
```

Check existing Cargo.toml for the current numpy dependency and ensure `PyReadonlyArray1`
is available.

---

## Part 5: Keeping the Old Path

**Do NOT remove `build_matrix()`.** Keep both entry points:
- `build_matrix()` — old path, works with raw Python LinOp objects
- `build_matrix_serialized()` — new optimized path

This allows:
- Gradual rollout (toggle between paths)
- Fallback if serialization has bugs
- Benchmarking old vs new

---

## Part 6: Further Optimization — All-NumPy Serialization (Optional)

If profiling shows the `nodes` list-of-tuples is still slow (Python tuple creation +
extraction overhead), we can go fully array-based:

```python
# Pack ALL metadata into a single int64 array:
# [op_type, ndim, shape..., num_args, data_tag, payload_ints...]
# Fixed-width per node with a length prefix
node_metadata = np.array([...], dtype=np.int64)
```

This would make the entire FFI boundary just 3 NumPy arrays (metadata, floats, ints) —
zero Python object overhead. But this is more complex to implement and may not be needed
if the tuple approach already eliminates the bottleneck.

**Decision: Start with the tuple approach. Optimize to all-NumPy only if profiling shows
tuple extraction is still significant.**

---

## Implementation Steps (Ordered)

1. **Add `OpType::from_int()`** to `linop.rs` — pure Rust, no Python changes
2. **Add `DeserializationContext`** to `linop.rs` — the `read_linop()` method
3. **Add `build_matrix_serialized()`** to `lib.rs` — new PyO3 entry point
4. **Build and test Rust side** — `maturin develop`, verify it compiles
5. **Add `serialize_linop_trees()`** to `canon_backend.py`
6. **Update `RustCanonBackend.build_matrix()`** to use the serialized path
7. **Run `rustybench.py`** — verify correctness (output matches SCIPY)
8. **Run `profile_rust_backend.py`** — measure performance improvement
9. **Run CVXPY tests** — `pytest cvxpy/tests/ -x -q` with RUST backend

## Expected Outcome

The Python serialization loop is O(N) pure Python (dict lookups, tuple creation, numpy
concat) — fast. The Rust deserialization reads tuples and memcpys slices — fast. The
bottleneck shifts from "thousands of PyO3 getattr calls" to "one bulk data transfer +
fast deserialization."

Target: **2-5x speedup** on the n=1000 least squares benchmark, bringing Rust from
~2x slower than SciPy to matching or beating it.
