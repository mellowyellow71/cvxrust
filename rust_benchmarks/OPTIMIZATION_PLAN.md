# Optimization Plan: DiffEngine-Inspired Improvements to the Rust Canonicalizer

## Context

The Rust canonicalizer (`cvxpy_rust`) is currently **~2x slower than SciPy** and **~2.8x
slower than C++** on least-squares problems (see `FIRST_RUN_ANALYSIS.md`). Profiling
shows the bottleneck is NOT Rust computation speed — it's the Python-to-Rust data
transfer overhead. Every node in the LinOp tree requires a PyO3 boundary crossing in
`LinOp::from_python()` (`linop.rs:194`), and for large problems (n=1000, m=5000) there
are thousands of nodes.

William's DiffEngine work (cvxgrp/DNLP PR #180) demonstrates a better architecture:
**build the computation graph once, separate structure from values, and reuse across
evaluations.** This plan adapts those principles to the Rust canonicalizer.

## The Three Optimizations (ordered by impact and feasibility)

---

### Optimization 1: Batch LinOp Tree Serialization (HIGH IMPACT, MEDIUM EFFORT)

**Problem:** `LinOp::from_python()` recursively calls into Python for every node —
`obj.getattr("type")`, `obj.getattr("shape")`, `obj.getattr("args")`,
`obj.getattr("data")` — each is a PyO3 boundary crossing. For a tree with N nodes,
this is O(N) Python API calls with significant per-call overhead.

**Solution:** Serialize the entire LinOp tree on the Python side into a flat buffer
(e.g., a single NumPy array or bytes object), then pass it to Rust in one shot. Rust
deserializes from the buffer without any Python API calls.

**Implementation:**

1. **Python side** — add a `serialize_linop_tree()` function in `canon_backend.py`:
   - Walk the LinOp tree in Python (which is cheap — Python accessing Python objects)
   - Encode each node into a flat format:
     - `op_type` (u8), `shape` (u32 × ndim), `num_args` (u16), `data_type` (u8)
     - Data payload: int/float inline, dense/sparse arrays as offsets into a separate
       data buffer
   - Output: `(metadata_bytes, data_arrays)` where `data_arrays` is a list of NumPy
     arrays referenced by offset

2. **Rust side** — add `build_matrix_from_serialized()` in `lib.rs`:
   - Accept `(bytes, list[ndarray])` instead of `list[LinOp]`
   - Deserialize into Rust `LinOp` structs by reading the byte buffer sequentially
   - Zero-copy for NumPy arrays via PyO3's array views

3. **Python entry point** — update `RustCanonBackend.build_matrix()` to call the
   serialized path.

**Files to modify:**
- `cvxpy/lin_ops/canon_backend.py` — add serialization, update `RustCanonBackend`
- `cvxpy_rust/src/lib.rs` — add `build_matrix_from_serialized` PyO3 function
- `cvxpy_rust/src/linop.rs` — add `LinOp::from_bytes()` deserialization

**Expected impact:** Eliminate ~90% of PyO3 overhead. The per-node cost drops from
multiple Python API calls to a single memcpy-like buffer read.

---

### Optimization 2: Sparsity-Structure Pre-analysis (MEDIUM IMPACT, MEDIUM EFFORT)

**Problem:** The current canonicalizer discovers sparsity on-the-fly. Each operation
handler allocates `Vec`s dynamically, grows them as it encounters nonzeros, and the
final `SparseTensor::combine()` concatenates everything. This means:
- Unpredictable memory allocation (many small `Vec` growths)
- No ability to pre-allocate the exact output size
- The `estimate_nnz()` in `linop.rs` is a rough heuristic, not exact

**Solution:** Inspired by DiffEngine's `init_jacobian_coo()` which computes the sparsity
pattern BEFORE evaluating values, do a lightweight first pass to determine the exact
output size, then a second pass to fill values into pre-allocated buffers.

**Implementation:**

1. **Phase 1 — Structure pass** (`operations/mod.rs`):
   - Add `count_nnz(lin_op, ctx) -> usize` that walks the tree but only counts
     nonzeros without allocating tensors or computing values
   - For leaf nodes: exact count is trivial (variable = size, sparse_const = nnz, etc.)
   - For operations: apply the same logic as the full handler but only track counts
   - This is cheap because it's just integer arithmetic, no f64 arrays

2. **Phase 2 — Value pass** (`matrix_builder.rs`):
   - Pre-allocate `SparseTensor::with_capacity(exact_nnz)` using the count from phase 1
   - Run the existing `process_linop()` but with guaranteed no reallocation

3. **Bonus — parallel pre-allocation** (`matrix_builder.rs`):
   - Each constraint's nnz is known upfront, so each rayon task can get a pre-allocated
     slice of the output buffer (no lock contention on Vec growth)

**Files to modify:**
- `cvxpy_rust/src/operations/mod.rs` — add `count_nnz()` dispatch
- `cvxpy_rust/src/operations/*.rs` — add nnz counting for each operation
- `cvxpy_rust/src/matrix_builder.rs` — two-pass build with pre-allocation

**Expected impact:** 10-30% speedup on large problems by eliminating dynamic allocation
overhead and improving cache locality (single contiguous buffer vs. many small Vecs).

---

### Optimization 3: Cached Graph for Parameterized Problems (HIGH IMPACT, HIGH EFFORT)

**Problem:** When a CVXPY problem has `Parameter` objects and is solved repeatedly with
different values, `get_problem_data()` currently has a "fast path" (line ~783 in
`problem.py`) that caches the `param_prog`. But within `build_matrix()`, the Rust
backend still:
1. Re-converts the entire LinOp tree from Python to Rust (full PyO3 overhead again)
2. Re-walks the tree and re-computes all structural operations (kronecker products,
   index selections, etc.) even though only the parameter VALUES changed

**Solution:** Cache the Rust-side LinOp graph and the structural computation. On
subsequent evaluations, only recompute the entries that depend on parameters.

**Implementation:**

1. **Persistent Rust graph** — store the converted `Vec<LinOp>` in a Python-accessible
   Rust object (PyO3 `#[pyclass]`):
   ```rust
   #[pyclass]
   struct CachedLinOpGraph {
       lin_ops: Vec<LinOp>,
       ctx: ProcessingContext,
       // Pre-computed: which entries in the output depend on parameters
       param_dependent_mask: Vec<bool>,
   }
   ```

2. **Two entry points**:
   - `build_matrix(lin_ops, ...)` — current behavior (cold path)
   - `build_matrix_cached(graph, param_values)` — hot path that:
     a. Reuses the cached `lin_ops` structure (no Python extraction)
     b. Only updates `Param` leaf data with new values
     c. Only re-processes subtrees that contain `Param` nodes
     d. Reuses the coefficient entries for parameter-free subtrees

3. **Python integration** — `RustCanonBackend` stores the `CachedLinOpGraph` and
   decides which path to use:
   ```python
   def build_matrix(self, lin_ops):
       if self._cached_graph is not None:
           return cvxpy_rust.build_matrix_cached(self._cached_graph, param_values)
       else:
           graph, result = cvxpy_rust.build_matrix_and_cache(lin_ops, ...)
           self._cached_graph = graph
           return result
   ```

**Files to modify:**
- `cvxpy_rust/src/lib.rs` — add `CachedLinOpGraph` pyclass and new entry points
- `cvxpy_rust/src/linop.rs` — add `has_param()` tree analysis
- `cvxpy_rust/src/matrix_builder.rs` — add cached evaluation path
- `cvxpy/lin_ops/canon_backend.py` — add caching logic to `RustCanonBackend`

**Expected impact:** For parameterized problems solved in a loop, 5-10x speedup on
re-solves after the first compilation. First solve is same speed.

---

## Implementation Order

```
Phase A (week 1):  Optimization 1 — Batch serialization
                   This is the highest-ROI change. It attacks the #1 bottleneck
                   (PyO3 overhead) and benefits ALL problems, not just parameterized ones.

Phase B (week 2):  Optimization 2 — Sparsity pre-analysis
                   Build on Phase A's serialized graph. The structure pass is cheap
                   to add once you already have a Rust-native LinOp tree.

Phase C (week 3):  Optimization 3 — Cached graph
                   This is the most complex change but has the biggest payoff for
                   the parameterized problem use case (which is common in practice).
```

## Verification

For each optimization, validate with:

1. **Correctness** — run `rustybench.py` and compare output matrices (data, row, col)
   between RUST and SCIPY backends. They must be identical (or equivalent up to
   entry ordering in COO format).

2. **Performance** — run `rustybench.py` and `profile_rust_backend.py` before and after.
   Target:
   - After Phase A: Rust should match or beat SciPy on the n=1000 least squares problem
   - After Phase B: Rust should beat SciPy by 1.5-2x
   - After Phase C: Repeated solves of parameterized problems should be 5-10x faster

3. **Existing tests** — run CVXPY's test suite with `CANON_BACKEND=RUST` to catch
   regressions:
   ```bash
   cd /Users/alanxiao/Code/cvxpy_hackathon/cvxpy
   python -m pytest cvxpy/tests/ -x -q --canon-backend=RUST
   ```

## Key Files Reference

| File | Role |
|------|------|
| `cvxpy/lin_ops/canon_backend.py:681-713` | `RustCanonBackend` — Python entry point |
| `cvxpy_rust/src/lib.rs` | PyO3 module, `build_matrix()` function |
| `cvxpy_rust/src/linop.rs:194-218` | `LinOp::from_python()` — the bottleneck |
| `cvxpy_rust/src/matrix_builder.rs` | Core algorithm, parallel dispatch |
| `cvxpy_rust/src/operations/mod.rs` | `process_linop()` — tree traversal dispatch |
| `cvxpy_rust/src/tensor.rs` | `SparseTensor` — COO storage |
| `cvxpy/problems/problem.py:783-835` | Parameterized problem cache (fast path) |
| `rust_benchmarks/rustybench.py` | Primary benchmark |
| `rust_benchmarks/profile_rust_backend.py` | Detailed profiling script |
