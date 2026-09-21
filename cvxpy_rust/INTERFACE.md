# cvxpy_rust: Python-Rust Interface

## Overview

The `cvxpy_rust` backend accelerates CVXPY's canonicalization step by replacing the C++ matrix builder with a Rust implementation. It uses **PyO3** for Python bindings and **maturin** as the build system, compiled as a `cdylib` via `setuptools-rust` in `setup.py`.

## Entry Points

Two `#[pyfunction]`s are exposed in `cvxpy_rust/src/lib.rs`:

### `build_matrix()` (legacy)

Takes raw Python `LinOp` objects and extracts attributes through the GIL. Slower due to per-node Python attribute access overhead.

```python
build_matrix(
    lin_ops: list[PyObject],
    param_size_plus_one: int,
    id_to_col: dict[int, int],
    param_to_size: dict[int, int],
    param_to_col: dict[int, int],
    var_length: int,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[int, int]]
```

### `build_matrix_serialized()` (production path)

Takes pre-serialized data with zero-copy NumPy buffer access. ~3x faster than the C++ backend.

```python
build_matrix_serialized(
    nodes: list[tuple],           # pre-order serialized node tuples
    float_data: np.ndarray[f64],  # dense/sparse values buffer
    int_data: np.ndarray[i64],    # sparse indices/indptr buffer
    param_size_plus_one: int,
    id_to_col: dict[int, int],
    param_to_size: dict[int, int],
    param_to_col: dict[int, int],
    var_length: int,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[int, int]]
#    ^ data      ^ (rows, cols)                        ^ shape
```

Both return COO-format sparse matrix components that the Python side converts to `scipy.sparse.csc_array`.

## Data Flow

```
Python LinOp trees
       |
       v
serialize_linop_trees()              [Python, canon_backend.py]
  - Pre-order tree walk
  - Packs nodes as tuples
  - Flattens array data into two contiguous NumPy buffers
       |
       v
build_matrix_serialized()            [Rust, lib.rs]
  - Zero-copy access to NumPy buffers
  - Releases GIL with py.detach()
  - Deserializes into Rust LinOp trees
  - Processes constraints (parallel via rayon when beneficial)
  - Returns COO data
       |
       v
scipy.sparse.csc_array              [Python, canon_backend.py]
```

## Serialization Format

`serialize_linop_trees()` (canon_backend.py:695-813) walks each LinOp tree in pre-order and produces three outputs:

### Node Tuples

Each node is serialized as:

```python
(
    op_type: int,             # 0-27, maps to OpType enum
    shape: tuple[int, ...],   # output shape
    num_args: int,            # number of child nodes
    data_tag: int,            # type of data payload (0-8)
    payload: Any,             # data, format depends on tag
    has_data_linop: int,      # 1 if data is itself a LinOp subtree
)
```

### Data Tags

| Tag | Meaning      | Payload                                                                 |
|-----|--------------|-------------------------------------------------------------------------|
| 0   | None         | `None`                                                                  |
| 1   | Integer      | `int` (variable ID, param ID, diagonal offset)                          |
| 2   | Float        | `float` (scalar constant)                                               |
| 3   | DenseArray   | `(float_offset, float_len, shape_tuple)`                                |
| 4   | SparseArray  | `(f_off, f_len, i_off_idx, i_len_idx, i_off_ptr, i_len_ptr, nrows, ncols)` |
| 5   | Slices       | `list[(start, stop, step)]`                                             |
| 6   | LinOpRef     | `None` (actual data serialized inline as a subtree)                     |
| 7   | AxisData     | `(axis_spec, keepdims)`                                                 |
| 8   | ConcatAxis   | `Optional[int]`                                                         |

### Buffers

- **`float_data`**: contiguous `np.ndarray[f64]` — all dense array values (column-major / F-order) and sparse matrix values.
- **`int_data`**: contiguous `np.ndarray[i64]` — all sparse indices and indptr arrays.

Offsets in node payloads (tags 3 and 4) index into these buffers.

### OpType Mapping

```
 0: variable        7: reshape        14: promote        21: diag_vec
 1: scalar_const    8: mul            15: broadcast_to   22: diag_mat
 2: dense_const     9: rmul           16: hstack         23: upper_tri
 3: sparse_const   10: mul_elem       17: vstack         24: conv
 4: param          11: div            18: concatenate    25: kron_r
 5: sum            12: index          19: sum_entries    26: kron_l
 6: neg            13: transpose      20: trace          27: no_op
```

## Rust Data Structures

### `LinOp` (linop.rs)

```rust
pub struct LinOp {
    pub op_type: OpType,
    pub shape: Vec<usize>,
    pub args: Vec<LinOp>,
    pub data: LinOpData,
}
```

### `LinOpData` (linop.rs)

```rust
pub enum LinOpData {
    None,
    Int(i64),
    Float(f64),
    DenseArray { data: Arc<[f64]>, shape: Vec<usize> },
    SparseArray { data: Arc<[f64]>, indices: Arc<[i64]>, indptr: Arc<[i64]>, shape: (usize, usize) },
    Slices(Vec<SliceData>),
    LinOpRef(Box<LinOp>),
    AxisData { axis: Option<AxisSpec>, keepdims: bool },
    ConcatAxis(Option<i64>),
}
```

### `SparseTensor` (tensor.rs)

Internal 3D COO representation used during processing:

```rust
pub struct SparseTensor {
    pub data: Vec<f64>,
    pub rows: Vec<i64>,
    pub cols: Vec<i64>,
    pub param_offsets: Vec<i64>,
    pub shape: (usize, usize),
}
```

Flattened to 2D on output: `output_row = col * n_rows + row`, `output_col = param_offset`.

### `DeserializationContext` (linop.rs)

```rust
pub struct DeserializationContext<'a, 'py> {
    nodes: &'a [Bound<'py, PyTuple>],
    float_data: &'a [f64],   // zero-copy slice into NumPy buffer
    int_data: &'a [i64],
    cursor: usize,            // current position in pre-order traversal
}
```

## Operation Dispatch

`process_linop()` in `matrix_builder.rs` dispatches on `OpType` to handlers in `operations/`:

| Category     | Operations                                                        |
|--------------|-------------------------------------------------------------------|
| Leaf         | `variable`, `scalar_const`, `dense_const`, `sparse_const`, `param` |
| Trivial      | `sum`, `neg`, `reshape`                                           |
| Arithmetic   | `mul`, `rmul`, `mul_elem`, `div`                                  |
| Structural   | `index`, `transpose`, `promote`, `broadcast_to`, `hstack`, `vstack`, `concatenate` |
| Specialized  | `sum_entries`, `trace`, `diag_vec`, `diag_mat`, `upper_tri`, `conv`, `kron_r`, `kron_l` |
| No-op        | `no_op`                                                           |

## Performance Techniques

| Technique              | Location                  | Effect                                                    |
|------------------------|---------------------------|-----------------------------------------------------------|
| Batch serialization    | Python `serialize_linop_trees()` | One tree walk packs everything into flat buffers, minimizing per-node GIL overhead |
| Zero-copy NumPy access | Rust `build_matrix_serialized()` | `float_data`/`int_data` read directly, no allocation      |
| GIL release            | Rust `py.detach()`        | Rust runs without holding the Python lock                  |
| Rayon parallelism      | Rust `matrix_builder.rs`  | Constraints processed in parallel when ≥4 constraints and ≥500 estimated NNZ |
| `Arc<[T]>`             | Rust `LinOpData`          | Shared ownership of array data avoids cloning              |

## File Map

| File | Purpose |
|------|---------|
| `cvxpy_rust/Cargo.toml` | Crate config and dependencies (pyo3, numpy, ndarray, sprs, rayon) |
| `cvxpy_rust/src/lib.rs` | PyO3 module definition, `build_matrix` and `build_matrix_serialized` |
| `cvxpy_rust/src/linop.rs` | `LinOp`, `OpType`, `LinOpData` structs and deserialization |
| `cvxpy_rust/src/tensor.rs` | `SparseTensor` COO format and flatten logic |
| `cvxpy_rust/src/matrix_builder.rs` | Main algorithm, parallelization decisions |
| `cvxpy_rust/src/operations/*.rs` | Per-operation handlers |
| `cvxpy/lin_ops/canon_backend.py` | `RustCanonBackend`, `serialize_linop_trees()`, OpType mapping |
| `setup.py` | `setuptools-rust` integration |
