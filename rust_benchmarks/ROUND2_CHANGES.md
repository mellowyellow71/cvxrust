# Round 2: Rust Backend Optimizations — Changes & Results

Branch `alan/arena-allocator`, June 2026. Commits `87a6798eb..30c9ce328`.
Successor to `CODE_CHANGES.md` (which describes the previous round:
serialization, two-pass NNZ build, cached graph — note its "Optimization 3"
was removed this round, see §6).

Summary: **the Rust backend now beats SciPy on 40/40 suite problems
(geomean ~4.4x) and the C++ backend on 39/40 (geomean ~2.0x)**, the test
suite is fully green for the first time (132/132, was 4 failures), and the
maintainer's arena-allocator question is answered with measurements.

---

## 1. Sort COO output by flat row inside Rust (`87a6798eb`)

**File:** `cvxpy_rust/src/tensor.rs` (`BuildMatrixResult::from_tensor`)

Rust used to return COO entries in processing order; Python then paid an
O(n log n) sort inside scipy's COO→CSC constructor, and `np.unique` in
`reduce_problem_data_tensor` sorted again. `from_tensor` now sorts entries
by flattened row index before returning, so both downstream steps hit their
linear sorted fast paths (scipy's CSC counting sort is stable, so a
flat-row-only key keeps rows sorted within every param column).

Two details that matter:

- **Serial sort below 1M nnz, rayon `par_sort` above.** Rayon's global
  thread pool initializes lazily on first use; that one-time cost dominates
  small problems. (This fix was identified on the `ray/vibe-rust` branch,
  where it took small LASSO cold-starts from 0.50x to 1.21x vs SciPy.)
- **Already-sorted early-out:** one cheap pass skips the permutation
  entirely when entries arrived sorted (common for small problems).

**Before** (`from_tensor` flattened and returned unsorted):

```rust
// Convert 3D COO to 2D COO in-place
// Output row = col * n_rows + row (column-major flattening)
for i in 0..tensor.rows.len() {
    tensor.rows[i] = tensor.cols[i] * n_rows_i64 + tensor.rows[i];
}

BuildMatrixResult {
    data: tensor.data,
    rows: tensor.rows,
    cols: tensor.param_offsets,
    shape: (output_rows, output_cols),
}
```

**After** (sort by flat row, serial below the rayon threshold):

```rust
const PAR_SORT_MIN_NNZ: usize = 1_000_000;
...
for i in 0..tensor.rows.len() {
    tensor.rows[i] = tensor.cols[i] * n_rows_i64 + tensor.rows[i];
}

let nnz = tensor.rows.len();
let already_sorted = tensor.rows.windows(2).all(|w| w[0] <= w[1]);
if !already_sorted {
    let mut order: Vec<usize> = (0..nnz).collect();
    if nnz < PAR_SORT_MIN_NNZ {
        order.sort_unstable_by_key(|&i| tensor.rows[i]);
    } else {
        order.par_sort_unstable_by_key(|&i| tensor.rows[i]);
    }
    tensor.data = order.iter().map(|&i| tensor.data[i]).collect();
    tensor.param_offsets = order.iter().map(|&i| tensor.param_offsets[i]).collect();
    tensor.rows = order.iter().map(|&i| tensor.rows[i]).collect();
}

BuildMatrixResult { data: tensor.data, rows: tensor.rows,
                    cols: tensor.param_offsets, shape: (output_rows, output_cols) }
```

Measured: lasso n=1000 RUST build_matrix 114.6 → 56.5ms; many_constraints
m=5000 34.9 → 24.1ms; geomean vs SciPy 4.26x → 4.67x.

**Same commit:** all `build_matrix*` return sites use `into_pyarray`
(moves the Vec into numpy) instead of `to_pyarray` (copies) —
`cvxpy_rust/src/lib.rs`:

```rust
// Before: copies each Vec into a fresh numpy array
let data = result.data.to_pyarray(py).into();
// After: moves the Vec's buffer into numpy, no copy
let data = result.data.into_pyarray(py).into();
```

## 2. Mul(Const, Variable) fast path (`92e087d83`)

**File:** `cvxpy_rust/src/operations/arithmetic.rs`

The Jacobian of `A @ x` w.r.t. `x` is `A` itself — but `process_mul`
materialized an identity tensor for `x` and multiplied `A` through it
entry-by-entry. The fast path detects a plain-variable RHS
(`as_plain_variable`, unwrapping no-op Reshape / single-arg Sum only) and
emits `A`'s entries directly at the variable's column offset:
O(nnz(A) · num_blocks) index arithmetic, zero multiplications. Fully-dense
constants bulk-fill via `extend_from_slice`.

**Multi-block correctness fix:** for a matrix variable `X(n,k)` the result
is `kron(I_k, A)` — entry `A[r,c]` of block `b` lands at
`(b·a_rows + r, var_col + b·a_cols + c)`. The equivalent fast path on
`ray/vibe-rust` emits only `nnz(A)` entries with no block loop (a latent
bug for matrix variables); ours loops blocks and falls back to the general
path when the constant doesn't tile the variable evenly. Parametric
constants are excluded and use the parametric path unchanged.
`count_nnz`'s Mul arm was updated to stay exact (scalar-const data counts
arg nnz, matching scale-in-place).

**Before** (`process_mul` always materialized the rhs tensor and ran the
general block-diagonal multiply; for a plain variable the rhs is just an
identity, so the inner loop multiplied every A entry by 1.0):

```rust
// process_mul
let rhs = process_linop(&lin_op.args[0], ctx);     // identity tensor for a variable
...
let lhs_data = get_constant_matrix_data(lhs_linop, Some(ctx));
let result = multiply_block_diagonal(&lhs_data, &rhs, lin_op, ctx, false);

// multiply_dense_block_diagonal_colmajor — the hot loop it lands in
for idx in 0..rhs.nnz() {
    let rhs_val = rhs.data[idx];                   // 1.0 for every variable entry
    ...
    let a_col = &data[col_start..col_start + a_rows];
    for (i, &a_val) in a_col.iter().enumerate() {
        if a_val != 0.0 {
            rows.push(new_row as i64);
            cols.push(rhs_col);
            vals.push(a_val * rhs_val);            // multiply by 1.0
            params.push(rhs_param);
        }
    }
}
```

**After** (`process_mul` detects the pattern and emits A directly; no rhs
tensor, no multiplications):

```rust
// process_mul — fast path, tried before processing the rhs
if let Some(var_id) = as_plain_variable(&lin_op.args[0]) {
    let arg_size = lin_op.args[0].size();
    if let Some(result) = mul_const_by_variable(&lhs_data, var_id, arg_size, lin_op, ctx) {
        return result;
    }
}

// as_plain_variable: Variable, unwrapping no-op Reshape / single-arg Sum only
fn as_plain_variable(lin_op: &LinOp) -> Option<i64> {
    match lin_op.op_type {
        OpType::Variable => match &lin_op.data {
            LinOpData::Int(id) => Some(*id),
            _ => None,
        },
        OpType::Reshape | OpType::Sum if lin_op.args.len() == 1 => {
            as_plain_variable(&lin_op.args[0])
        }
        _ => None,
    }
}

// mul_const_by_variable, dense column-major arm (block loop is the
// multi-block fix; ray's equivalent emits only block 0):
let num_blocks = checked_num_blocks(arg_size, *a_cols)?;   // None -> slow path
for b in 0..num_blocks {
    let row_base = (b * a_rows) as i64;
    let col_base = var_col + (b * a_cols) as i64;
    for c in 0..*a_cols {
        let out_col = col_base + c as i64;
        let col_start = c * a_rows;
        for (r, &val) in data[col_start..col_start + a_rows].iter().enumerate() {
            if val != 0.0 {
                result.push(val, row_base + r as i64, out_col, param_offset);
            }
        }
    }
}
```

Measured: geomean 4.67x → 4.84x; broad −10..−24% on small dense/indexing
problems. New Rust unit test compares fast vs slow path on a 2×3 const ×
3×2 matrix variable.

## 3. Fix transpose panic on default `.T` (`968c22935`)

**File:** `cvxpy_rust/src/operations/structural.rs`

CVXPY emits transpose LinOps with `data=[None]` for a plain `.T`, which
serializes to `AxisData(axis=None)`. `process_transpose` accepted
`Some(Multiple(..))` and bare `None` data but **panicked** on
`AxisData{axis: None}` — which by numpy semantics means "reverse all
axes".

**Before:**

```rust
let axes = match &lin_op.data {
    LinOpData::AxisData { axis: Some(AxisSpec::Multiple(axes)), .. } => axes.clone(),
    LinOpData::None => {
        // Default transpose: reverse all axes
        let n_dims = lin_op.args[0].shape.len();
        (0..n_dims).rev().map(|i| i as i64).collect()
    }
    _ => panic!("Transpose operation must have axis data"),   // <- hit by plain .T
};
```

**After:**

```rust
let axes = match &lin_op.data {
    LinOpData::AxisData { axis: Some(AxisSpec::Multiple(axes)), .. } => axes.clone(),
    // None data, or AxisData with axis=None (a plain `.T` — CVXPY sends
    // data=[None]): numpy semantics, reverse all axes
    LinOpData::None | LinOpData::AxisData { axis: None, .. } => {
        let n_dims = lin_op.args[0].shape.len();
        (0..n_dims).rev().map(|i| i as i64).collect()
    }
    _ => panic!("Transpose operation must have axis data"),
};
```

One match arm fixed all five known failures, including a panic on a
real solver path:

- `test_rust_backend.py::test_transpose_2d` (3 parametrizations)
- `test_rust_backend.py::test_mul_with_transpose_data`
- `test_conic_solvers.py::TestClarabel::test_clarabel_socp_3`

After: `test_rust_backend.py` **132/132 — first fully passing run.**

## 4. O(1) lookups in trace / diag_mat / upper_tri (`9a7e6a57b`)

**File:** `cvxpy_rust/src/operations/specialized.rs`

- `process_trace`: `diag_indices.contains(&row)` (O(n) per entry) → the
  arithmetic test `row % (n+1) == 0` (Fortran-order diagonal flat indices
  are exactly the multiples of n+1 below n²).

  ```rust
  // Before: build the index list, scan it per entry
  let diag_indices: Vec<i64> = (0..n).map(|i| (i * (n + 1)) as i64).collect();
  for i in 0..tensor.nnz() {
      if diag_indices.contains(&tensor.rows[i]) { ... }
  }

  // After: O(1) arithmetic test
  let step = (n + 1) as i64;
  for i in 0..tensor.nnz() {
      if tensor.rows[i] % step == 0 { ... }
  }
  ```

- `process_diag_mat`: per-entry `.position()` scan → closed-form inverse
  of the affine diagonal map (`i·(orig_rows+1) + offset`).

  ```rust
  // Before: linear search for each entry's output row
  if let Some(new_row) = diag_indices.iter().position(|&d| d == row) { ... }

  // After: invert the map directly
  let step = (orig_rows + 1) as i64;
  let offset: i64 = if k > 0 { k * orig_rows as i64 } else { -k };
  let t = tensor.rows[i] - offset;
  if t >= 0 && t % step == 0 {
      let new_row = t / step;
      if (new_row as usize) < rows { ... }
  }
  ```

- `process_upper_tri`: per-entry `.position()` over n(n−1)/2 indices
  (O(nnz·n²) total) → precomputed n² lookup table preserving the
  `np.triu_indices_from` numbering.

  ```rust
  // Before: linear search per entry over the upper-tri index list
  if let Some(new_row) = upper_indices.iter().position(|&u| u == row) { ... }

  // After: O(1) table lookup (same i-outer/j-inner numbering)
  let mut new_row_of = vec![-1i64; n * n];
  let mut counter = 0i64;
  for i in 0..n.saturating_sub(1) {
      for j in (i + 1)..n {
          new_row_of[i + j * n] = counter;
          counter += 1;
      }
  }
  ...
  let row = tensor.rows[i] as usize;
  if row < new_row_of.len() && new_row_of[row] >= 0 { ... }
  ```

Output entries and ordering unchanged; only lookup mechanics.

## 5. Flat i64 metadata-stream serialization (`b8a4544a3`)

**Files:** `cvxpy/lin_ops/canon_backend.py` (`serialize_linop_trees`),
`cvxpy_rust/src/linop.rs` (`DeserializationContext`), `cvxpy_rust/src/lib.rs`

Phase-timing instrumentation (run anything with `CVXPY_RUST_FFI_PROFILE=1`)
showed Rust-side deserialization of the tuple-based node encoding was
**~35% of build_matrix time** on many-constraint problems (4.9ms of ~14ms
in Rust at m=5000 / 30k nodes): each node cost ~8–12 PyO3
`get_item`/`extract` calls.

Node metadata is now packed into **one flat `np.int64` array**
(`[op_type, ndim, *shape, num_args, data_tag, *payload]` per node; f64
scalars as bit patterns) and Rust deserializes by walking a borrowed slice
— zero per-node Python access. Measured at m=5000: deser 4.9 → 1.4ms
(3.4x), Python-side serialize +1.5ms, **net ~2ms faster**; neutral
elsewhere.

**Before** (Python built one tuple per node; Rust unpacked each field with
PyO3 calls):

```python
# canon_backend.py — per node
nodes.append((op_type_int, shape, num_args, data_tag, payload,
              1 if has_data_linop else 0))
```

```rust
// linop.rs — per node, each line is one or more Python C-API round-trips
let node = &self.nodes[self.cursor];
let op_type_int: u8 = node.get_item(0)?.extract()?;
let shape: Vec<usize> = node.get_item(1)?.extract()?;
let num_args: usize = node.get_item(2)?.extract()?;
let data_tag: u8 = node.get_item(3)?.extract()?;
let payload = node.get_item(4)?;
let has_data_linop: u8 = node.get_item(5)?.extract()?;
```

**After** (Python extends one int list per node; Rust reads plain i64s
from a slice cursor):

```python
# canon_backend.py — per node (variable shown; one extend per node)
extend((op_map[t], len(shape), *shape, nargs, 1, int(data)))
...
node_meta = np.array(meta, dtype=np.int64)
```

```rust
// linop.rs — per node, no Python objects involved
let op_type = OpType::from_int(self.next()? as u8)?;
let ndim = self.next()? as usize;
let mut shape = Vec::with_capacity(ndim);
for _ in 0..ndim {
    shape.push(self.next()? as usize);
}
let num_args = self.next()? as usize;
let data_tag = self.next()?;

#[inline]
fn next(&mut self) -> PyResult<i64> {       // the whole "FFI": a slice read
    match self.meta.get(self.cursor) {
        Some(&v) => { self.cursor += 1; Ok(v) }
        None => Err(...),
    }
}
```

### The arena-allocator question (answered)

This change came out of evaluating the maintainer's suggestion to use an
arena allocator for Rust-side node handling. Full writeup:
`rust_benchmarks/FFI_OVERHEAD_ANALYSIS.md`. Conclusion: arena and
serialization are complementary, but measurement showed the dominant deser
cost was per-node PyO3 access, not allocation. With that eliminated
(~48ns/node left), an arena's upper bound is **~0.7ms ≈ 3%** on the
workload it helps most, against a ~25-function refactor — deferred. The
largest remaining FFI cost is Python-side serialization itself (~13.7ms at
m=5000); the next lever there is caching serialized buffers across
`get_problem_data` calls, which the flat-buffer format makes natural.

## 6. Remove unreachable CachedLinOpGraph; prune deps (`0a70a0346`)

**Files:** `cvxpy_rust/src/lib.rs`, `canon_backend.py`, `Cargo.toml`

The previous round's "Optimization 3" (cached Rust-side graph for
parameterized re-solves) was verified unreachable: `get_problem_matrix` is
only called at compile time (`coeff_extractor.py`, `affine_atom.py`), and
parameterized re-solves go through `ParamConeProg.apply_parameters`, which
multiplies the *cached tensor* by the parameter vector — `build_matrix` is
never re-invoked on parameter changes. The `param_prog`/`ReducedMat`
caching already covers that use case at a higher level. Deleted
`CachedLinOpGraph`, both entry points, the uncalled Python wrappers
(~257 lines), plus never-imported dependencies: `faer`, `sprs`, `ndarray`,
`thiserror`.

**Removed API** (nothing replaces it — it was unreachable):

```rust
#[pyclass] struct CachedLinOpGraph { lin_ops: Vec<LinOp>, ctx: ProcessingContext, ... }
#[pyfunction] fn build_matrix_and_cache(...) -> (..., Py<CachedLinOpGraph>)
#[pyfunction] fn build_matrix_cached(graph: &Bound<CachedLinOpGraph>) -> ...
// plus LinOp::has_param() and SparseTensor::combine(), only used by the above
```

```python
# canon_backend.py — uncalled static wrappers, removed
RustCanonBackend.build_matrix_and_cache(...)
RustCanonBackend.build_matrix_cached(cached_graph)
```

```toml
# Cargo.toml dependencies, before -> after
pyo3, numpy, ndarray, sprs, rayon, thiserror, faer   ->   pyo3, numpy, rayon
```

## 7. CPP backend support in the benchmark suite (`30c9ce328`)

**File:** `rust_benchmarks/benchmark_suite.py`

The build_matrix layer timed backends via `CanonBackend.get_backend`,
which doesn't register CPP (it dispatches inside
`canonInterface.get_problem_matrix`). `time_build_matrix` now routes CPP
to `cppbackend.build_matrix` directly, and captured calls store
`constr_length`. Usage: `--backends RUST SCIPY CPP`.

```python
# Before: every backend went through the registry — KeyError for CPP
def time_build_matrix(captured_call, backend_name):
    backend = CanonBackend.get_backend(backend_name, ...)
    return time_single(lambda: backend.build_matrix(c["linOps"]))

# After: CPP times its dedicated entry point, same thin layer
def time_build_matrix(captured_call, backend_name):
    if backend_name == "CPP":
        from cvxpy.cvxcore.python.cppbackend import build_matrix
        return time_single(lambda: build_matrix(
            dict(c["id_to_col"]), dict(c["param_to_size"]), dict(c["param_to_col"]),
            c["var_length"], c["constr_length"], c["linOps"]))
    backend = CanonBackend.get_backend(backend_name, ...)
    return time_single(lambda: backend.build_matrix(c["linOps"]))
```

---

## Benchmark results

Warm `build_matrix`-layer timings, Apple M5 (Rosetta x86_64 env
`cvxpy-py313`), threads pinned by the suite. "Pre" = baseline before this
round (`baseline_full.json`); "Post"/SciPy/C++ from a single three-backend
run (`tri_backend.json`). All ms (mean).

| problem | pre RUST | post RUST | SciPy | C++ | SciPy/Rust | C++/Rust |
|---|---|---|---|---|---|---|
| dense_matmul (n=50) | 0.09 | 0.11 | 0.18 | 0.13 | 1.62x | 1.14x |
| dense_matmul (n=200) | 0.08 | 0.10 | 0.18 | 0.11 | 1.76x | 1.10x |
| dense_matmul (n=500) | 0.08 | 0.08 | 0.17 | 0.13 | 2.20x | 1.67x |
| dense_matmul (n=1000) | 0.08 | 0.09 | 0.16 | 0.12 | 1.74x | 1.29x |
| sparse_matmul (n=100) | 0.07 | 0.09 | 0.16 | 0.11 | 1.86x | 1.26x |
| sparse_matmul (n=500) | 0.07 | 0.08 | 0.15 | 0.10 | 1.92x | 1.27x |
| sparse_matmul (n=2000) | 0.07 | 0.09 | 0.16 | 0.12 | 1.74x | 1.29x |
| dense_qp (n=50) | 0.09 | 0.11 | 0.43 | 0.15 | 3.95x | 1.38x |
| dense_qp (n=200) | 0.09 | 0.11 | 0.41 | 0.16 | 3.87x | 1.52x |
| dense_qp (n=500) | 0.09 | 0.10 | 0.49 | 0.19 | 4.95x | 1.87x |
| many_constraints (m=10) | 0.25 | 0.26 | 2.51 | 0.38 | 9.62x | 1.46x |
| many_constraints (m=50) | 0.49 | 0.44 | 11.92 | 1.37 | 27.1x | 3.12x |
| many_constraints (m=100) | 0.78 | 0.66 | 23.74 | 2.73 | 36.0x | 4.15x |
| many_constraints (m=500) | 3.19 | 2.44 | 116.9 | 17.00 | 47.9x | 6.96x |
| many_constraints (m=1000) | 6.32 | 4.71 | 235.7 | 43.19 | 50.1x | 9.17x |
| many_constraints (m=5000) | 34.94 | **24.24** | 1198.6 | 599.7 | 49.5x | 24.8x |
| box_constraints (n=50) | 0.12 | 0.13 | 0.36 | 0.17 | 2.81x | 1.28x |
| box_constraints (n=200) | 0.13 | 0.12 | 0.38 | 0.18 | 3.23x | 1.51x |
| box_constraints (n=1000) | 0.26 | 0.16 | 0.54 | 0.38 | 3.36x | 2.31x |
| matrix_indexing (n=20) | 0.07 | 0.08 | 0.14 | 0.10 | 1.72x | 1.23x |
| matrix_indexing (n=50) | 0.08 | 0.08 | 0.15 | 0.11 | 1.87x | 1.33x |
| matrix_indexing (n=100) | 0.09 | 0.07 | 0.15 | 0.10 | 1.96x | 1.34x |
| hstack (width=10) | 0.11 | 0.13 | 2.34 | 0.26 | 18.4x | 2.03x |
| hstack (width=50) | 0.22 | 0.23 | 10.82 | 0.92 | 47.7x | 4.06x |
| hstack (width=200) | 0.57 | 0.59 | 43.29 | 4.86 | 73.0x | 8.20x |
| portfolio (n=50) | 0.09 | 0.09 | 0.37 | 0.15 | 3.98x | 1.58x |
| portfolio (n=200) | 0.16 | 0.23 | 0.61 | 0.17 | 2.67x | 0.73x |
| portfolio (n=500) | 0.27 | 0.23 | 0.50 | 0.26 | 2.20x | 1.15x |
| lasso (n=50) | 0.29 | 0.22 | 0.92 | 0.51 | 4.19x | 2.32x |
| lasso (n=200) | 3.10 | **1.79** | 6.65 | 5.39 | 3.71x | 3.01x |
| lasso (n=500) | 22.45 | **13.32** | 43.36 | 34.86 | 3.26x | 2.62x |
| lasso (n=1000) | 114.58 | **70.70** | 185.6 | 164.5 | 2.63x | 2.33x |
| svm (m=100) | 0.27 | 0.26 | 0.93 | 0.56 | 3.63x | 2.19x |
| svm (m=500) | 0.77 | 0.80 | 2.49 | 2.10 | 3.14x | 2.65x |
| convolution (len=100) | 0.10 | 0.09 | 0.15 | 0.11 | 1.81x | 1.26x |
| convolution (len=500) | 0.09 | 0.09 | 0.15 | 0.11 | 1.71x | 1.23x |
| nested_affine (depth=3) | 0.08 | 0.08 | 0.15 | 0.11 | 1.79x | 1.33x |
| nested_affine (depth=5) | 0.09 | 0.09 | 0.17 | 0.13 | 1.94x | 1.48x |
| nested_affine (depth=10) | 0.07 | 0.10 | 0.16 | 0.10 | 1.62x | 1.03x |
| nested_affine (depth=20) | 0.10 | 0.08 | 0.14 | 0.10 | 1.68x | 1.18x |

**Aggregates:** Rust vs SciPy geomean **4.43x**, wins **40/40**. Rust vs
C++ geomean **1.96x**, wins **39/40** (the lone C++ "win", portfolio
n=200, is a sub-0.3ms row inside the noise band). Biggest absolute wins
this round: the lasso family (−40..−55% Rust time; previously Rust *lost*
to both SciPy and C++ on large dense problems) and many-constraint
problems (−17..−29%).

**Cold start** (fresh subprocess, end-to-end `get_problem_data`,
`current_coldstart.json`): Rust ahead on all four problems — lasso n=200
1.37x, dense_qp n=200 1.37x, many_constraints m=500 3.38x, sparse_matmul
n=500 1.22x vs SciPy — with no small-problem rayon-init penalty.

### Reading the numbers

- Rows under ~0.2ms are dominated by fixed per-call overhead and swing
  ±30–80% between runs on this machine; treat pre→post movement there as
  noise. Trust the >1ms rows and the `CVXPY_RUST_FFI_PROFILE=1` component
  timers.
- The suite's `dense_matmul`/`convolution` bm-layer rows time a trivial
  captured call (largest-linop-count heuristic picks the wrong one); the
  lasso family is the honest dense-problem signal.

## Test status

`pytest cvxpy/tests/test_rust_backend.py`: **132/132** (baseline had 4
failures, all rooted in the transpose bug of §3). `cargo test`: 31/31.
CLARABEL solver subset: 21 passed including the previously-failing
`test_clarabel_socp_3`.

## Not done / future work

- **Serialization caching** across `get_problem_data` calls — the largest
  remaining FFI cost (~13.7ms Python-side at m=5000) and the natural next
  step now that the serialized form is three flat arrays.
- **Arena allocator** — deferred with data (≤3% upper bound); see
  `FFI_OVERHEAD_ANALYSIS.md`.
- **faer/BLAS** — documented dead end (prior attempt was slower; rayon
  cold-start, not BLAS, was the dense-problem overhead).
- **Block IR / Index(Variable) fast-path generalization** from ray's later
  branches — worth revisiting after this round's results are upstreamed.
