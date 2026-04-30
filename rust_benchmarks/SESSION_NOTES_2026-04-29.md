# Cold-start Overhaul Session — 2026-04-29

A detailed write-up of one extended session attempting to close the cold-start
performance gap between the cvxpy Rust backend and the SciPy baseline. The
goals were ambitious — a Block-IR architectural shift inspired by Peter Wang's
DNLP/SparseDiffEngine prototype — but most of the actual wall-clock win came
from a single one-line conceptual fix that the prior analysis doc had
mis-diagnosed. This document records the full reasoning trail so future
sessions don't waste cycles on the same dead ends.

Read this in conjunction with:
- `RUST_BACKEND_PERFORMANCE_ANALYSIS.md` — current bench numbers and headline
  state of the backend
- `FAST_PATH_IMPLEMENTATION.md` — the April 3 changes that this session built
  on top of
- `4-3-26-changes.md` — the session that produced those April 3 changes
- `context.md` — Peter Wang's diffengine-backend Discord context

---

## 1. Background going in

### 1.1 The repo

`/home/ray/cvxrust/cvxpy` is a fork of cvxpy with a Rust canonicalization
backend at `cvxpy_rust/`. The Rust crate is built via `maturin` and exposes
PyO3 entry points called from `cvxpy/lin_ops/canon_backend.py::RustCanonBackend`.
The fork tracks `origin = cvxpy/cvxpy.git` (no push access) and has two
working remotes:

- `mellowyellow71` — `https://github.com/mellowyellow71/cvxpy.git` (note: GitHub
  redirects this to `mellowyellow71/cvxrust.git`). This is the active tracking
  remote.
- `fork` — `https://github.com/switchpiggy/cvxpy.git`. Mirror push, no
  tracking. Set up to migrate over but not switched.

User instruction: keep tracking on `mellowyellow71`, push to both.

### 1.2 The user's framing

> "It is not faster than the cpp or scipy backend which is bad. One of the
> ideas that we wanted to implement is based on some of the implementation
> ideas of DNLP, the cvxpy team has realized that parsing through the linear
> operator tree is equivalent to a forward pass auto diff since each linear op
> is represented as a matrix and derivative of A^T x = A, we just use chain
> rule to get same effort of linop tree."

The user supplied a Discord chat log (March 2024 → April 2026) covering the
project's history. Key themes from that log:

- **Peter Wang's DNLP/SparseDiffEngine prototype** (Mar 2026): hits 3.7s on a
  least-squares problem where CPP is 9.7s, SCIPY is 11.8s, COO is 14.8s. Uses
  forward-mode autodiff against a sparse-diff library, with a dense-matrix
  fast path (PR #49) and a "BLAS copies for sparse rows with one entry"
  fast path (PR #51) that recognises the `mul(A, x)` Jacobian is just `A`.
- **pigs4410** (Apr 2026): proposed serialising the LinOp tree to a numpy
  buffer on the Python side so the Rust backend doesn't pay per-node PyO3
  costs. Reported a "good speedup" but the work didn't make it onto the
  branch we were starting from.
- **Parth** (Apr 2026): mentioned wanting to switch to an arena allocator
  for the SparseTensor allocations.
- **Peter** (Mar 2026): noted that `np.unique` in `reduce_problem_data_tensor`
  was a major bottleneck — 575ms out of ~10s on the original LS benchmark.

### 1.3 What the analysis doc claimed

`RUST_BACKEND_PERFORMANCE_ANALYSIS.md` (pre-session) said:

- LASSO 200×500: RUST 60ms vs SCIPY 40ms = **0.66× SciPy** (the regression)
- Diagnosis: "Pure Rust scalar operations, even with good memory access
  patterns, can't match BLAS performance."
- Recommended fixes: link to BLAS via faer, custom SIMD kernels, cache
  constant matrix extractions.

This diagnosis turned out to be **wrong**. The actual cause was rayon thread
pool initialisation cost on cold start. See §3.4 below.

### 1.4 The plan we approved

The user picked, from a multi-question briefing:
- "Docs are roughly right but I want bigger wins" (the DNLP target)
- "Architectural: redesign around explicit forward-mode AD/dense propagation"

So the original plan was a Block IR migration:
1. PR 1: scaffolding for Block IR types
2. PR 2: leaves migration to typed Block output
3. PR 3a: parallel API
4. PR 3b: lazy `Index(Identity)` → `ColPerm`/sub-Identity
5. PR 3c: Mul Block-aware dispatch matrix
6. PR 3d: faer gemm for Dense × Dense

Plan file: `/home/ray/.claude/plans/iterative-meandering-naur.md`.

---

## 2. Branches, commits, and what each one does

### 2.1 `ray/vibe-rust` (base — pre-session)

Already had the April 3 wins committed: `mul_const_by_variable` fast path for
`Mul(Const, Variable)`, `as_plain_variable` unwrap, `from_tensor` parallel
COO sort to fix the `np.unique` bottleneck. State at `6e3e627ee`.

### 2.2 `ray/vibe-rust` (this session — `2e64aa44b`)

Branched directly off the prior tip. One commit:

```
2e64aa44b  block IR scaffolding: types, leaf migration, parallel API
```

Adds `cvxpy_rust/src/block.rs` with the `Block` enum (`Zero`, `Identity`,
`ScaledIdentity`, `ColPerm`, `Dense`, `SparseCsc`, `Coo`), `DenseF` strided
F-order view, `SparseCsc` Arc-slice CSC, `BlockEntry`, `NodeValue`, plus
`to_coo()` / `from_coo()` round-trip. Migrates leaf handlers in
`operations/leaf.rs` to expose `process_*_block` variants returning typed
`NodeValue`; legacy `process_*` are now thin wrappers calling `.to_coo()`.
Adds `process_linop_block` parallel API in `operations/mod.rs`.

Pure scaffolding. Existing 30 cargo tests pass plus 14 new ones. End-to-end
benchmark within ±2% of baseline.

**Why this commit on `ray/vibe-rust` rather than the dispatch branch:** user
wanted a clean checkpoint of the IR foundation before any handler-touching
PRs.

### 2.3 `ray/block-ir-dispatch` (off `2e64aa44b`)

Where the riskier handler-touching work lived. 5 commits on top:

```
fd800457e  small LinOp::from_python wins + clear pyo3 0.27 deprecations
0f67b68c9  update perf analysis doc with cold-start numbers and corrected diagnosis
593a963c4  serial sort below 1M nnz to skip rayon thread-pool cold-start cost  ← biggest win
e730d6bde  bulk emit for fully-dense Mul fast path
19147a0ad  generalise diffengine fast path: Index of Variable -> Identity at offset
```

In commit order:

**`19147a0ad`** generalises the existing April-3 `as_plain_variable` fast path.
Before: only `Mul(ConstantMatrix, plain Variable / Reshape / single-arg Sum)`
took the diffengine shortcut. After: any RHS subtree that reduces to a single
non-parametric `Identity` block does. The route:
- `process_index_block` in `structural.rs` recognises `Index(Identity)` and
  returns `Identity` at a shifted `var_col_offset` (contiguous slice) or
  `ColPerm` (general slice). Other inputs fall back to the legacy
  `select_rows` COO path.
- `process_mul` probes RHS via `process_linop_block`; if the result is a
  single non-parametric Identity, dispatches to a new
  `mul_const_by_identity_offset` (shared body with `mul_const_by_variable`,
  parameterised on the offset). Otherwise falls through to
  `multiply_block_diagonal`.

This catches `Mul(A, Index(Variable, contiguous_slice))` — the LASSO
canonicalisation pattern where `x` is sliced out of the canonical variable
`y = [t; x]`. Functionally correct (numerical equivalence verified), but
wall-clock impact within noise on rustybench (~1.70s) and LASSO 200×500
(~1.08× SciPy vs 1.12× baseline) — see §4 for why.

**`e730d6bde`** bulk-emit for the fully-dense Mul fast path. The previous
`mul_const_by_identity_offset` body did per-entry `SparseTensor::push` calls
(four `Vec::push` each with capacity check). When the constant matrix has no
zeros (the typical LASSO case where `A` is random gaussian), pre-counts
non-zeros, then bulk-fills the four output arrays via `extend_from_slice` of
a precomputed row pattern, `std::iter::repeat(...).take(a_rows)` for cols,
`vec![p; nnz]` for param_offsets, and a straight `iter().copied().collect()`
for data. Falls back to the per-element zero-skip path when the matrix has
zeros. Modest 1–2% bench win.

**`593a963c4`** — the **single biggest win of the session**. The April 3
commit added `par_sort_unstable_by_key` in `from_tensor` to make the
downstream Python `np.unique` run on pre-sorted input. But rayon's global
thread pool initialises lazily on first parallel call, and that init cost
(plus first-time TLB / cache warm) isn't amortised by the parallel speedup
on small/medium nnz. For cold-start LASSO 50×100 (~8.5K nnz) it's the
dominant cost; even for LASSO 200×500 (~100K nnz) the parallel sort barely
breaks even.

The fix: serial `sort_unstable_by_key` below a 1M nnz threshold. rustybench
(5M nnz LS) keeps the parallel path. The threshold was picked by inspecting
problem sizes in the test suite.

Cold-start `quick_benchmark.py` impact:
- LASSO (n=50, m=100): 0.50× → 1.21× (2.5× speedup)
- LASSO (n=200, m=500): 0.65× → 0.95× (1.45× speedup)
- Many constraints (n=50, m=500): 2.80× → 3.14×

Avg RUST/SCIPY across 6 problems: 1.36× → 1.76×, wins 4/6 → 5/6.

This commit alone delivers more cold-start improvement than every other
commit on the branch combined. The previous diagnosis ("LASSO is slow because
no BLAS") was wrong. The fix is a one-line conceptual change. Lesson
re-learned: **profile first, design second**.

**`0f67b68c9`** rewrites `RUST_BACKEND_PERFORMANCE_ANALYSIS.md` with the new
cold-start numbers, the corrected diagnosis, and a complete inventory of
what's still attackable. Replaces the speculative "BLAS would help LASSO"
section with a profile-driven one.

**`fd800457e`** small mechanical wins on the LinOp extraction path (op_type as
borrowed `&str` instead of `String`, skip `args` getattr for leaf ops, replace
deprecated `.downcast` with `.cast`, skip Python `ravel("F")` round-trip when
the numpy array is already F-contiguous). Within bench noise but clears
deprecation warnings and reduces per-node allocation.

### 2.4 `ray/cold-start-overhaul` (off `fd800457e`)

The branch where the user explicitly asked us to attempt all three of
LinOp serialisation, arena allocator, and np.unique elimination. 3 commits:

```
1598c4be9  add Rust compute_reduction (parked) — scipy transforms defeat the wiring
5f1e579f3  add LinOp tree numpy-buffer serialisation entry point
fd800457e  (← branch base)
```

**`5f1e579f3`** implements pigs4410's idea. Adds a parallel PyO3 entry
`build_matrix_from_buffer` taking a flat little-endian byte buffer that
encodes the LinOp tree (depth-first, pre-order) plus a list of "heavy"
attachments (numpy arrays, scipy sparse matrices) referenced by index from
the buffer. Python serialiser `_serialize_linop_tree` lives in
`canon_backend.py`. Rust deserialiser `LinOp::from_buffer` /
`list_from_buffer` lives in `linop.rs`. The wire format is documented inline
above `BufferReader`. Op-type bytes are part of the wire protocol — never
reorder, only append.

**Bench result was disappointing.** I'd estimated ~80 nodes per LASSO LinOp
tree based on intuition; the actual buffer turned out to be 352 bytes ≈
~10–20 nodes. The PyO3 saving was bounded to ~600 µs — within cold-start
subprocess noise. The 3 ms `extract_linop` time we'd seen in the trace was
dominated by the **800 KB memcpy** of the dense `A` matrix (still happens
in Rust on the buffer path), not per-node Python crossings. See §4.4.

**`1598c4be9`** implements `compute_reduction` in Rust — the moral equivalent
of `canonInterface.reduce_problem_data_tensor`. Walks the already-sorted COO
once, finds unique rows, builds a CSR for the reduced matrix, and computes
the final indices/indptr/shape. PyO3 entry `compute_reduction` in `lib.rs`,
internal `compute_reduction_from_slices` in `tensor.rs` (taking borrowed
slices to avoid an upfront memcpy), four unit tests covering basic reduction,
quad_form vs constraint shape, eliminate_zeros behaviour, and within-row
column sort.

**The wiring is parked.** Tracing `ReducedMat.__init__` and `cache()`
revealed that the cached `ReducedMat`s operate on **transformed copies** of
`build_matrix`'s output. Specifically: `ParamConeProg.__init__` materialises
`2 * params_to_P`, which scipy returns as a fresh `csc_array` without the
`_rust_raw_coo` attribute we'd attached, and with `rows[]` no longer sorted.
Both possible recovery vectors (attribute stash, sorted-input fast path)
fail.

The Rust function is left in the build because it's correct and tested, and
any future deeper integration (subclassing `csc_array` to survive scipy
transforms, or hooking earlier in the pipeline before `2 * P` lands) can
call it directly.

---

## 3. Cold-start vs warm-start: the methodology pivot

### 3.1 The two harnesses

- `rustybench.py` — calls `Problem.get_problem_data` once per backend in a
  process that already imported cvxpy. Useful for relative deltas of
  build_matrix-internal changes. Single call → high noise.
- `quick_benchmark.py` — runs each (problem, backend) cell in 10 fresh
  Python subprocesses, with `np.random.seed(42)` reset each time. This is
  the **user-facing measurement**: a typical cvxpy user invokes one Python
  script per problem and exits.

### 3.2 They tell different stories

For LASSO 200×500:

| Metric        | rustybench (warm in-process)  | quick_benchmark (cold subprocess) |
|---------------|--------------------------------|------------------------------------|
| Pre-PR-3      | RUST 18.8ms, ~1.12× SciPy     | RUST 30.3ms, **0.65× SciPy**      |
| Post-PR-3     | unchanged                     | unchanged                         |
| Post-rayon-fix| RUST 18.5ms, ~1.08× SciPy     | RUST 20.9ms, **0.95× SciPy**      |

The doc's "0.66× SciPy" figure was always cold-start. That detail wasn't
documented when the original analysis was written, and it took a fresh
profile to realise that warm and cold answer different questions.

**Always quote cold-start numbers as the headline.** Anything else doesn't
match what users see.

### 3.3 The cProfile trace (warm, 20 iters of LASSO 200×500)

```
Component                                  Time    Share
cvxpy_rust.build_matrix (Rust FFI)         128ms    30%
np.unique in reduce_problem_data_tensor    148ms    34%
numpy.argsort (canonInterface)              76ms    18%
numpy.sort (canonInterface)                 40ms     9%
Other (scipy CSC construction, misc)        40ms     9%
Total                                       433ms
```

Per problem: ~21 ms warm. Rust FFI is **30%** of warm wall-clock. The biggest
single chunk is actually `np.unique` in the reduce step.

### 3.4 The CVXRUST_TRACE-gated breakdown of the Rust call

For one cold-start LASSO 200×500 invocation, with Instant probes inserted
in `lib.rs`, `tensor.rs`, `matrix_builder.rs`:

```
extract_linop:                           2.7-3.0ms  (38% of Rust call)
process_constraints (3 constraints):     1.2-1.3ms  (17%)
combine (nnz=101800):                    0.85-0.99ms (12%)
from_tensor:
  flat_rows:                             ~0.03ms
  sort:                                  1.4-1.6ms  (20%)
  permute:                               0.8-0.9ms  (11%)
  total:                                 2.3-2.6ms
to_pyarray:                              ~0.05ms
WHOLE Rust call:                         7.1-8.1ms / 20.8ms total wall
```

The trace lives behind `CVXRUST_TRACE=1`. It was added during the session,
used to guide the np.unique attempt, and removed before final commit because
it added complexity to lean code. Re-add temporarily by reapplying the
pattern in `lib.rs:38-76`, `tensor.rs::from_tensor`, and
`matrix_builder.rs::build_matrix_internal` if needed for future
investigation.

### 3.5 What changed once we had the trace

The trace flipped two priors:

- The plan said PR 3d (faer gemm) would deliver the LASSO win. The trace
  said gemm would target ≤10% of wall-clock at best (the scalar-emit
  portion of `process_constraints`), and the actual LASSO regression was
  rayon-init in `from_tensor::sort`. **Deprioritised PR 3d.**
- I'd assumed `extract_linop` was 2.7 ms because of ~80 PyO3 calls × 30 µs
  each. The buffer experiment showed the actual tree is ~10-20 nodes (352
  bytes). The 2.7 ms is dominated by the 800 KB memcpy of the dense `A`
  matrix into a Rust `Vec<f64>` (still happens on the buffer path). **Bounds
  the LinOp serialisation win to ~600 µs.**

---

## 4. Findings & dead-end reasons

### 4.1 Rayon thread-pool init was the LASSO bottleneck (not BLAS)

`par_sort_unstable_by_key` in `from_tensor` triggers rayon's lazy global
thread pool initialisation on first parallel call. The pool init has a
fixed cost (~hundreds of µs, plus first-time TLB / cache warm) that the
parallel speedup can't amortise on small/medium nnz problems. For cold
LASSO 50×100 (~8.5 K nnz), this fixed cost was bigger than the entire
serial sort. Switching below a 1M-nnz threshold to serial sort was a
4-line conceptual change with a 2.5× cold-start speedup on small LASSO.

The doc's "LASSO is slow because no BLAS" diagnosis sent me toward faer
gemm, custom SIMD kernels, etc. — none of which would have helped.
Profile first.

### 4.2 LinOp tree is much smaller than I estimated

Tracing the buffer size: 352 bytes for LASSO 200×500's main constraint
matrix call, 2 attachments. That's ~10-20 nodes. Per-node PyO3 overhead at
~30 µs each totals ~600 µs — within cold-start noise after rayon was fixed.

The 3 ms `extract_linop` time we saw is mostly:
1. memcpy of the 800 KB `A` buffer when `extract_dense_array` does
   `slice.to_vec()` (~80–100 µs at memory bandwidth, plus Vec realloc
   amortised cost = a couple hundred µs).
2. The dense numpy buffer path opening (`arr.cast::<PyArrayDyn<f64>>()`,
   `arr.is_fortran_contiguous()`, `arr.readonly()`), tens of µs each, 1× per
   tree.
3. PyO3 boundary crossings at ~30 µs × ~15 nodes ≈ 450 µs.
4. First-call Rust startup costs (PyO3 init, allocator warm) — free of
   problem size, paid once per process.

Nothing in (1)–(4) is reachable by changing tree-walk strategy. (1) needs
zero-copy via PyReadonlyArray-backed Arc; (4) needs binary size reduction
or AOT loading. Both are substantial standalone refactors.

### 4.3 np.unique elimination is blocked by scipy transforms

The plan was: precompute the reduction in Rust, attach via attribute on the
returned `csc_array`, have `ReducedMat.cache()` use the precomputed
reduction.

What actually happens (verified by tracing `ReducedMat.__init__` and
`cache()` calls in `/tmp/debug_attr.py`):

```
[build_matrix] returned id=...872272 hasattr=True
[build_matrix] returned id=...872400 hasattr=True
[ReducedMat.__init__] id=...872272 hasattr=True   shape=(810900, 1)
[ReducedMat.__init__] id=...872000 hasattr=False  shape=(810000, 1)
[ReducedMat.__init__] id=...775248 hasattr=False  shape=(810900, 1)
[ReducedMat.__init__] id=...872000 hasattr=False  shape=(810000, 1)
[ReducedMat.cache]    id=...775248 hasattr=False
[ReducedMat.cache]    id=...872000 hasattr=False
```

Two observations:
1. The first ReducedMat (the one with our attribute, shape=(810900, 1)) is
   constructed but its `cache()` is never called. The cached ReducedMats
   are constructed later from different objects.
2. The LASSO 200×500 constraint matrix arriving at `cache()` has
   `sorted=False` and `unique_rows=nnz=101800` (every row is already
   unique — the reduction is essentially a no-op).

What's happening: `ParamConeProg.__init__` (in
`cvxpy/reductions/dcp2cone/cone_matrix_stuffing.py:174`) constructs
ReducedMat with the raw matrix. But the conic solver / downstream pipeline
later constructs **another** ParamConeProg using transformed matrices
(e.g. `2 * params_to_P` for the quadratic block, plus other operations
that scipy returns as fresh `csc_array`s without our attribute). The new
ReducedMats are the ones whose `cache()` actually runs.

The sorted-input fast path inside the existing
`reduce_problem_data_tensor` is also blocked: scipy CSC↔COO round-trips
re-order the rows, so by the time `np.unique` runs on `A_coo.row` the
data isn't sorted any more.

Recovery vectors that *might* work but weren't pursued:
- Subclass `csc_array` to preserve attributes through arithmetic. Fragile —
  any new scipy method that goes through C-level paths would lose them.
- Hook the reduction earlier in the pipeline, before
  `2 * params_to_P` lands. Requires understanding the conic solver
  transformation chain in detail.
- Wrap the matrix in a non-csc_array proxy that exposes the same interface.
  Significant breakage risk.

For now: Rust `compute_reduction` lives in the build (correct, tested) but
unused. See `cvxpy_rust/src/lib.rs::compute_reduction` and
`cvxpy_rust/src/tensor.rs::compute_reduction_from_slices`.

### 4.4 LinOp serialisation: small win, foundation laid

Net cold-start delta for the buffer path: within noise. The estimated
~600 µs PyO3 saving is offset by the Python serialisation walk (~11 µs
measured) and the buffer construction overhead. The serialisation logic
is exercised in production via `RustCanonBackend.build_matrix`'s
`hasattr(cvxpy_rust, "build_matrix_from_buffer")` check.

The wire format is documented above `BufferReader` in `linop.rs`. Op-type
bytes are part of the protocol — never reorder, only append. Future
extensions:
- Zero-copy data extraction via `PyReadonlyArray`-backed `Arc<[f64]>` for
  dense constants. Saves the ~100 µs memcpy of the 800 KB `A` buffer for
  LASSO. Risk: Rust holding the slice across `py.detach()` (GIL release)
  needs care — the underlying numpy buffer is stable but normal scipy
  refcount changes need GIL.

### 4.5 Arena allocator: skipped after measurement

LinOp tree is ~10-20 nodes × ~3 small Vec allocs each ≈ 30–60 small
allocations per problem ≈ 5 µs total at modern allocator speed. SparseTensor
allocations across `build_matrix` are bigger but still sub-millisecond
total. Bumpalo would shave ~300 µs cold-start. The refactor (LinOp lifetime
parameter `LinOp<'arena>` threaded through every handler) is large.
ROI didn't pencil out.

### 4.6 faer gemm: deprioritised, not deleted

Plan PR 3d called for faer-backed Dense × Dense gemm. The trace shows
`process_constraints` (where any gemm path would live) is ~17% of cold-start
Rust time. Even halving it saves ~0.6 ms cold-start. faer is in
`Cargo.toml` but unused. The Block IR provides clean dispatch points
(see `arithmetic.rs::process_mul`'s typed-Identity probe) so this remains
a low-effort unlock if it becomes useful — but it's not the LASSO win
the doc claimed.

---

## 5. Architectural design: the Block IR

Even though the perf payoff was modest, the Block IR has long-term value.
Documenting the design here.

### 5.1 The `Block` enum (`cvxpy_rust/src/block.rs`)

```rust
pub enum Block {
    Zero { rows, cols },
    Identity { n },
    ScaledIdentity { alpha: f64, n: usize },
    ColPerm { perm: Arc<[i64]>, ncols: usize },  // row i has 1 at perm[i]
    Dense(Arc<DenseF>),
    SparseCsc(Arc<SparseCsc>),
    Coo(SparseTensor),                            // 2D escape hatch
}
```

Variants are ordered by structural cheapness. `Zero` and the Identity
family encode their value in O(1) words; `Dense` / `SparseCsc` own real
numerical data; `Coo` is the legacy escape hatch for any handler that
hasn't been migrated yet.

`DenseF` is a strided F-order view (rows, cols, data: Arc<[f64]>,
row_stride, col_stride, row_offset). Allows zero-copy Reshape / Transpose
/ Index by manipulating strides.

`SparseCsc` holds `Arc<[i64]>` indptr / indices and `Arc<[f64]>` data.
Mirrors the layout cvxpy passes from Python.

### 5.2 `NodeValue` and `BlockEntry`

```rust
pub struct NodeValue {
    pub out_rows: usize,
    pub var_cols: usize,
    pub blocks: Vec<BlockEntry>,
}

pub struct BlockEntry {
    pub param_slot: i64,
    pub var_col_offset: i64,
    pub block: Block,
}
```

A `NodeValue` represents one LinOp subtree's Jacobian. The `blocks` list
is conceptually a sum of contributions, each placed at
`(param_slot, var_col_offset)`. Non-parametric subtrees have a single
entry with `param_slot == ctx.const_param()` — the common case.

For typed blocks, `param_slot` and `var_col_offset` describe the
placement. For `Block::Coo`, both are unused (the COO entries carry their
own row/col/param indices). `NodeValue::from_coo` documents this — both
fields are set to 0 as placeholders.

### 5.3 What's migrated and what isn't

Migrated to the Block IR (return `NodeValue` directly):
- All five leaves: Variable → `Identity(n)`, ScalarConst → `Dense(1, 1)`,
  DenseConst → `Dense(n, 1)`, SparseConst / Param → wrapped Coo (the
  flat-F-order encoding for the column they live in doesn't fit a typed
  matrix block cleanly).
- `Index`: returns `Identity` at shifted offset for contiguous slice over
  Identity input, `ColPerm` for general slice over Identity. Other inputs
  fall through to legacy `select_rows`.
- `Mul`: probes RHS via `process_linop_block` after the legacy
  `as_plain_variable` check; if the result is a single non-parametric
  Identity, dispatches to `mul_const_by_identity_offset`. Otherwise
  falls through to `multiply_block_diagonal`.

NOT migrated (still return `SparseTensor` and get wrapped as
`Block::Coo` by `process_linop_block`):
- `Sum`, `Neg`, `Reshape` (trivial; could migrate easily)
- `Rmul`, `MulElem`, `Div` (would benefit from migration when paired with
  a Dense×Dense gemm path)
- `Transpose`, `Promote`, `BroadcastTo`, `Hstack`, `Vstack`, `Concatenate`
- All specialised: `SumEntries`, `Trace`, `DiagVec`, `DiagMat`,
  `UpperTri`, `Conv`, `KronR`, `KronL`

The migration is incremental by design. Each handler can be moved
independently; the legacy `to_coo` wrapper at the boundary preserves
behavioural equivalence.

### 5.4 The dispatch matrix (Mul)

| `M \ V`              | Identity(n)              | ScaledId(β,n)        | Dense                   | SparseCsc               |
|---|---|---|---|---|
| Scalar(α)            | ScaledIdentity(α,n)      | ScaledId(αβ,n)       | Dense (scale)           | SparseCsc (scale)       |
| Dense(m×n)           | **Dense (Arc-clone M)**  | Dense (αM)           | **faer gemm → Dense**   | spmm → Dense or CSC     |
| SparseCsc(m×n)       | **SparseCsc (Arc-clone)**| SparseCsc (αM)       | spmm → Dense            | csc·csc → SparseCsc     |
| ColPerm              | ColPerm (compose)        | scaled ColPerm       | row-permuted Dense view | row-permuted CSC        |

Bolded cells are the diffengine fast path: `M × Identity → Arc-clone of M
with column-shifted output`. The current Mul handler implements only this
case; the rest are future work. The benefit: `Mul(A, anything-that-reduces-
to-Identity)` skips constructing the identity tensor and doing scalar
multiply-by-1.

---

## 6. How to navigate the codebase

### 6.1 Rust side (`cvxpy_rust/`)

```
src/
├── lib.rs                  PyO3 module entry. build_matrix, build_matrix_from_buffer,
│                            compute_reduction (parked), test_function
├── linop.rs                LinOp / OpType definitions, from_python (legacy) and
│                            from_buffer / list_from_buffer (PR A)
├── matrix_builder.rs       build_matrix_internal, parallel/sequential constraint
│                            processing, rayon thresholds (PARALLEL_MIN_CONSTRAINTS=4,
│                            PARALLEL_MIN_WORK=500)
├── tensor.rs               SparseTensor (3D COO), SparseTensorBuilder,
│                            BuildMatrixResult::from_tensor (April 3 sort + PAR_SORT
│                            threshold), ReducedMatrix + compute_reduction_from_slices
├── block.rs                Block IR types: Block, NodeValue, BlockEntry, DenseF,
│                            SparseCsc, to_coo, from_coo
└── operations/
    ├── mod.rs              ProcessingContext, process_linop dispatcher,
    │                        process_linop_block parallel API
    ├── leaf.rs             5 leaf handlers (Variable, ScalarConst, DenseConst,
    │                        SparseConst, Param) with both legacy and _block variants
    ├── arithmetic.rs       Mul (with typed-Identity fast-path probe), Rmul, MulElem,
    │                        Div, Neg, get_constant_matrix_data, ConstantMatrix enum,
    │                        mul_const_by_variable / mul_const_by_identity_offset
    ├── structural.rs       Index (with process_index_block), Transpose, Promote,
    │                        BroadcastTo, Hstack, Vstack, Concatenate
    └── specialized.rs      SumEntries, Trace, DiagVec, DiagMat, UpperTri, Conv,
                             KronR, KronL
```

`Cargo.toml` deps: `pyo3 0.27.1`, `numpy 0.27.0`, `ndarray 0.16`,
`sprs 0.11`, `rayon 1.10`, `thiserror 2.0`, `faer 0.20`. **`sprs` and `faer`
are present but currently unused** — keep them; they're allocated for
the next round of gemm/CSC work.

### 6.2 Python side

- `cvxpy/lin_ops/canon_backend.py::RustCanonBackend.build_matrix` — entry
  from cvxpy. Calls `cvxpy_rust.build_matrix_from_buffer` if present
  (uses `_serialize_linop_tree` to walk the tree once); falls back to
  `cvxpy_rust.build_matrix` on older builds.
- `cvxpy/lin_ops/canon_backend.py::_serialize_linop_tree` — Python
  serialiser writing the LinOp tree to a flat byte buffer. Format is
  little-endian, depth-first. **Op-type byte mapping (`_OP_TYPE_TO_BYTE`)
  must match `OpType::to_byte` / `from_byte` in `linop.rs`.**
- `cvxpy/cvxcore/python/canonInterface.py::reduce_problem_data_tensor` —
  the np.unique-heavy reducer. Eliminating its calls from the Rust path
  is parked (see §4.3).
- `cvxpy/reductions/utilities.py::ReducedMat` — caches the reduced matrix
  per problem. Construction site for the matrices that `build_matrix`
  output flows into.

### 6.3 Benchmarks

- `rust_benchmarks/quick_benchmark.py` — **the authoritative cold-start
  measurement**. 10 fresh subprocesses per (problem, backend). Six
  problem types. Use this when reporting numbers.
- `rust_benchmarks/rustybench.py` — single-process LS at n=1000, m=5000.
  Useful for warm relative deltas of build_matrix-internal changes. Don't
  use for headline numbers.
- `rust_benchmarks/benchmark_suite.py` — comprehensive statistical
  benchmark with environment fingerprinting, warmup, GC isolation, JSON
  output. Underused in practice.
- `rust_benchmarks/profile_rust_backend.py` — cProfile bottleneck analysis.

### 6.4 Build commands (always release)

```bash
# Rust unit tests
cd /home/ray/cvxrust/cvxpy/cvxpy_rust
cargo test --release

# Rebuild the Python extension after Rust changes
cd /home/ray/cvxrust/cvxpy/cvxpy_rust
maturin develop --release   # use VIRTUAL_ENV=... PATH=... if not in venv

# Cold-start benchmark
/home/ray/cvxrust/.venv/bin/python rust_benchmarks/quick_benchmark.py

# Numerical equivalence sweep (use this before any commit)
/home/ray/cvxrust/.venv/bin/python /tmp/equiv_check.py   # see §7.1

# Trace LASSO 200x500
CVXRUST_TRACE=1 /home/ray/cvxrust/.venv/bin/python /tmp/trace_one.py
```

**Always use `--release`. Debug builds are 2–3× slower.** If the maturin
output says `[unoptimized + debuginfo]`, the build is wrong. Look for
`[optimized]`.

---

## 7. Verification protocol

### 7.1 Equivalence sweep

Before any commit that changes canonicalisation behaviour, run a sweep
that compares RUST and SCIPY backend outputs problem-by-problem. The
script (`/tmp/equiv_check.py`) constructs five canonical problem types
(LS, LASSO, many-constraint LP, strided-index, promote/broadcast),
canonicalises each with both backends, and checks all keys in the
returned `data` dict for byte-equivalence up to 1e-10. Failure means
the change has changed semantics.

The five problems aren't exhaustive but cover every Block-IR-touched
code path the session introduced. Recreate from this template if needed:

```python
import numpy as np, cvxpy as cp
np.random.seed(0)

def check(name, problem):
    data_rust, _, _ = problem.get_problem_data(cp.CLARABEL, canon_backend="RUST")
    data_scipy, _, _ = problem.get_problem_data(cp.CLARABEL, canon_backend="SCIPY")
    # ... compare data_rust and data_scipy element-wise to 1e-10 ...

n, m = 200, 500; A = np.random.randn(m, n); b = np.random.randn(m)
x = cp.Variable(n)
check("LS n=200,m=500", cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b))))
check("LASSO n=200,m=500", cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.norm1(x))))
# ... etc ...
```

### 7.2 Cold-start sanity

After equivalence, run `quick_benchmark.py`. Compare each problem's
`RUST` column against the previous run. A 5%+ regression on any cell
needs investigation before commit.

### 7.3 Rust unit tests

`cargo test --release` runs ~54 tests covering `block.rs` (9 tests),
`leaf.rs` (5 base + 5 _block variants), `arithmetic.rs` (mul fast path,
including LASSO pattern), `structural.rs` (Index lazy paths),
`specialized.rs`, `tensor.rs` (5 SparseTensor + 4 ReducedMatrix tests),
`matrix_builder.rs` (4 end-to-end). The reduction tests
(`test_compute_reduction_*`) pin down the parked compute_reduction
function's semantics — keep these green even if the function isn't
wired up.

---

## 8. Bench scoreboard

### 8.1 Final cold-start numbers (after this session)

```
Problem                              RUST      SCIPY        CPP   RUST/SCIPY    CPP/SCIPY
LASSO (n=50, m=100)                 3.4ms      4.2ms      4.3ms       1.21x       0.97x
LASSO (n=200, m=500)               20.9ms     19.8ms     21.7ms       0.95x       0.91x
Dense QP (n=50)                     2.4ms      3.0ms      3.2ms       1.24x       0.95x
Dense QP (n=200)                    2.8ms      3.4ms      3.7ms       1.23x       0.92x
Many constraints (n=50, m=100)     11.4ms     31.6ms     12.8ms       2.78x       2.47x
Many constraints (n=50, m=500)     47.6ms    149.5ms     58.2ms       3.14x       2.57x

RUST vs SCIPY: avg 1.78x, min 0.95x, max 3.14x, wins 5/6
CPP vs SCIPY:  avg 1.47x, min 0.92x, max 2.57x, wins 2/6
```

Numbers vary cell-to-cell by 3-5% across runs. The 1.78x avg has hovered
between 1.74x and 1.79x across the session's later commits — call it
~1.76x with noise.

### 8.2 Pre-session vs post-session

| Problem                       | Pre-session | Post-session | Δ      |
|-------------------------------|-------------|--------------|--------|
| LASSO (n=50, m=100)           | 0.50× SciPy | **1.21×**    | +2.4×  |
| LASSO (n=200, m=500)          | 0.65×       | **0.95×**    | +1.45× |
| Dense QP (n=50)               | 1.18×       | 1.24×        | +5%    |
| Dense QP (n=200)              | 1.16×       | 1.23×        | +6%    |
| Many constraints (n=50, m=100)| 1.88×       | 2.78×        | +48%   |
| Many constraints (n=50, m=500)| 2.80×       | 3.14×        | +12%   |

Wins: 4/6 → 5/6. Avg: 1.36× → 1.78×. Min: 0.50× → 0.95×.

Of that delta, **the rayon threshold commit (`593a963c4`) is responsible
for almost all of it**. Everything else combined moved within noise.

---

## 9. What's left to attack (and how)

### 9.1 LASSO 200×500 still ~1.1 ms behind SciPy

Cold-start: 20.9 ms RUST vs 19.8 ms SCIPY. Of that 20.9 ms, only ~7 ms is
in the Rust call (per the warm trace; cold-start adds first-call costs
on top). Even free Rust would only save ~1.4 ms — the rest is process
startup, .so loading, scipy transformations.

Reachable optimisations, ordered by expected ROI:

1. **np.unique elimination via subclassed csc_array** (~1–3 ms cold-start
   savings if reachable). Subclass scipy's csc_array so attribute
   assignment survives `2 * matrix` etc. Risk: any scipy method that
   doesn't go through `__array_wrap__` would lose the subclass. Need to
   audit transformations in the conic-solver pipeline. The Rust
   `compute_reduction` function is already in place — wire is the work.
2. **Zero-copy dense data extraction** (~80–100 µs cold-start). `Arc<[f64]>`
   backed by `PyReadonlyArray<f64>`. Holds the GIL guard alongside the
   Arc; needs care that `py.detach()` doesn't drop the guard. Saves the
   memcpy of the dense `A` matrix.
3. **faer gemm for Dense × Dense Mul** (~0.5 ms cold-start). Fills in
   the dispatch matrix cells `Dense × Dense` and `Sparse × Dense`. Most
   useful for problems where `Mul(A, B@x)` doesn't fold into a single
   constant matrix on the cvxpy side. Rare in practice for typical
   cvxpy expressions.

### 9.2 Bigger structural changes

1. **Migrate remaining handlers to Block IR**. The dispatch matrix only
   covers `Mul × Identity` today. Adding `Mul × Dense` (gemm), `Mul × Sparse`
   (spmm), and lazy `Hstack` / `Vstack` / `Promote` / `Transpose` over
   typed blocks would cover more problem patterns. Each is a 100–200 LOC
   change with its own equivalence test.
2. **Pre-uniquify on the Rust side and skip the csc_array round-trip
   entirely**. Today: Rust returns `(data, rows, cols)`, Python wraps in
   `csc_array`, scipy converts to COO/CSR/CSC multiple times in the
   reduction pipeline. If Rust returns the reduced CSR triple directly
   and Python skips the canon_backend.py csc_array build, several
   redundant scipy conversions go away. Substantial cross-language
   refactor.
3. **AOT or smaller binary** for cold-start startup. The Rust .so
   takes time to load. `lto = true, codegen-units = 1, opt-level = 3`
   are already set. Could explore `panic = "abort"` and feature gating
   to shrink binary further. Sub-millisecond gains, large effort.

### 9.3 Problems where we still lose on bench

- **LASSO 200×500: 0.95× SciPy.** ~1.1 ms gap. Bounded as above.

That's the only loss in the suite.

---

## 10. Failed approaches — don't repeat

1. **faer gemm as the LASSO fix.** The pre-session doc claimed this. Profile
   said no. Don't reach for BLAS until you've measured the rayon-init and
   memcpy costs.
2. **Attribute-stash on csc_array for cross-call data passing.** Survives
   simple operations but `2 * matrix` creates a fresh array; any
   conic-solver transformation drops the attribute. Subclass or side-channel
   if you need this.
3. **Estimating tree size by intuition.** I assumed ~80 LinOp nodes per
   LASSO problem; actual is ~10–20. Always measure.
4. **Trusting the existing analysis doc.** `RUST_BACKEND_PERFORMANCE_ANALYSIS.md`
   pre-session was Claude-generated and had a confident incorrect diagnosis.
   Verify before building on a doc's recommendation.
5. **Splitting PRs into too-small pieces.** PR 1 (scaffolding only),
   PR 2 (leaves only), PR 3a (parallel API only) were each behaviour-neutral
   and individually un-testable for performance. They built up a foundation
   with no commit-by-commit validation — until PR 3 (Index + Mul), there
   was nothing measurable. Bundle the value-delivery commit with at least
   a minimal user.

---

## 11. Lessons learned (general)

1. **Profile first, design second.** The doc's BLAS theory was wrong. The
   trace took 30 minutes and pointed straight at the rayon issue.
2. **Cold-start is the user-facing measurement.** Warm in-process timings
   are useful for relative deltas of internal changes; users see
   cold-start. Always re-bench cold after a "warm" win to make sure it
   transfers.
3. **The Rust call is only ~30% of cold-start wall-clock for problems of
   this size.** Optimising scalar emit is bounded; bigger wins are in the
   Python pipeline (or in not paying the pipeline cost at all).
4. **Cross-language interface changes are expensive and brittle.** The
   np.unique elimination wiring failed because scipy's transformations
   defeat attribute-based stashing. Plan for this at design time, not at
   debug time.
5. **Block IR scaffolding without immediate handler migration is mostly
   value-free for performance.** It's a foundation; commit it, but don't
   expect it to move bench numbers on its own.

---

## 12. Files changed in this session

```
NEW   cvxpy_rust/src/block.rs
NEW   rust_benchmarks/SESSION_NOTES_2026-04-29.md  (this file)
EDIT  cvxpy_rust/src/lib.rs
EDIT  cvxpy_rust/src/linop.rs
EDIT  cvxpy_rust/src/matrix_builder.rs            (only via PAR_SORT threshold work)
EDIT  cvxpy_rust/src/tensor.rs
EDIT  cvxpy_rust/src/operations/mod.rs
EDIT  cvxpy_rust/src/operations/leaf.rs
EDIT  cvxpy_rust/src/operations/arithmetic.rs
EDIT  cvxpy_rust/src/operations/structural.rs
EDIT  cvxpy/lin_ops/canon_backend.py
EDIT  rust_benchmarks/RUST_BACKEND_PERFORMANCE_ANALYSIS.md
```

`cvxpy/reductions/utilities.py` was edited and reverted in the np.unique
attempt; final state matches the pre-session content.

---

## 13. Reconvening

State at end of session:
- Two branches with all wins committed and pushed (mellowyellow + fork).
- Working tree clean.
- Equivalence sweep passes; 54 cargo tests green.
- Cold-start avg 1.78× SciPy, 5/6 wins.
- Rust `compute_reduction` parked but tested.
- `faer` and `sprs` deps in place but unused, awaiting future gemm work.

Possible next moves to discuss:
- Subclass-csc_array experiment to unblock the np.unique elimination
- Zero-copy dense data extraction for the buffer path
- Migrate more handlers to Block IR (probably Sum/Neg/Reshape first as
  cheapest wins)
- Investigate the multi-stage scipy transformation pipeline to find
  where to hook the reduction earlier
- Rerun the original DNLP benchmark on Peter's problem size to confirm
  whether we close the gap to his 3.7s number
