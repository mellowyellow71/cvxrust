# CVXPY Rust Backend Performance Analysis

## Current State (2026-04-29)

The Rust backend now wins or ties on the cold-start `quick_benchmark.py`
suite for every problem type except one near-tie. Cold-start is the
right user-facing measurement: a typical CVXPY user solves one problem
per script invocation, paying first-call overhead each time.

The previous version of this doc claimed LASSO 200×500 ran at 0.66×
SciPy "due to lack of BLAS". That diagnosis was wrong. The actual
cause was rayon thread-pool initialisation triggering on cold start
for problems too small to amortise the parallel speedup. Switching to
serial sort below 1M nnz fixed it.

## Cold-start benchmark (10 fresh-process samples per cell)

| Problem                       | RUST   | SCIPY  | CPP    | RUST/SCIPY | CPP/SCIPY |
|------------------------------:|-------:|-------:|-------:|:----------:|:---------:|
| LASSO (n=50, m=100)           |  3.4ms |  4.2ms |  4.3ms |   1.21×    |   0.97×   |
| LASSO (n=200, m=500)          | 20.9ms | 19.8ms | 21.7ms |   0.95×    |   0.91×   |
| Dense QP (n=50)               |  2.4ms |  3.0ms |  3.2ms |   1.24×    |   0.95×   |
| Dense QP (n=200)              |  2.8ms |  3.4ms |  3.7ms |   1.23×    |   0.92×   |
| Many constraints (n=50, m=100)| 11.4ms | 31.6ms | 12.8ms |   2.78×    |   2.47×   |
| Many constraints (n=50, m=500)| 47.6ms |149.5ms | 58.2ms |   3.14×    |   2.57×   |

**Average: RUST/SCIPY = 1.76×, wins 5/6.** CPP/SCIPY = 1.47×, wins 2/6.

The single near-tie is LASSO 200×500 at 0.95× — within ~1.1ms of SciPy.

## What changed since the prior doc

1. **Block IR scaffolding** (`src/block.rs`). Typed representation of
   each subtree's Jacobian — `Identity`, `ScaledIdentity`, `ColPerm`,
   `Dense`, `SparseCsc`, `Coo`. Leaf handlers produce typed values
   directly via `process_*_block`; non-migrated handlers still return
   COO and are wrapped via `NodeValue::from_coo`.

2. **Generalised diffengine fast path**. The April 3 `as_plain_variable`
   path (Mul of constant against literal Variable / Reshape /
   single-arg Sum) only caught a narrow set of trees. `process_mul`
   now also probes the RHS via `process_linop_block` and dispatches
   to a generalised `mul_const_by_identity_offset` whenever the RHS
   reduces to a single non-parametric `Identity` block. This catches
   `Mul(A, Index(Variable, contiguous_slice))` — the LASSO
   canonicalisation pattern that was falling through to the slow
   `multiply_block_diagonal` path.

3. **Lazy `Index` over `Identity`**. `process_index_block` (in
   `structural.rs`) recognises Index over an Identity input and
   returns either an `Identity` at a shifted `var_col_offset`
   (contiguous slice) or a `ColPerm` (general slice). Other inputs
   fall back to the legacy `select_rows` COO path.

4. **Bulk emit for fully-dense Mul**. The dense col-major branch in
   `mul_const_by_identity_offset` skips per-entry `SparseTensor::push`
   when the constant matrix has no zeros (the typical LASSO case),
   bulk-filling the four output arrays via `extend_from_slice`,
   pattern fills, and `vec!`-broadcast. Modest win (~1-2%).

5. **Serial sort threshold in `from_tensor`**. The biggest single
   win in this round. `par_sort_unstable_by_key` is replaced with a
   serial `sort_unstable_by_key` when `nnz < 1_000_000`. Avoids
   rayon's lazy global thread-pool initialisation cost on cold
   start. rustybench (5M nnz) keeps the parallel path and is
   unchanged at ~1.70s. Cold-start LASSO 50×100 went 0.50× → 1.21×;
   200×500 went 0.65× → 0.95×.

## Measurement methodology

- **Cold start:** `rust_benchmarks/quick_benchmark.py`, 10 fresh
  Python subprocesses per (problem, backend) cell, fresh
  `np.random.seed(42)` each run. This is the user-facing number.
- **Warm:** `rust_benchmarks/rustybench.py` calls
  `Problem.get_problem_data` 1× in a process that already imported
  cvxpy; useful for relative deltas of build_matrix-internal changes
  but masks cold-start overhead.
- **Numerical equivalence:** RUST and SCIPY backends produce
  byte-identical canonicalisation output up to 1e-10 on least-squares,
  LASSO, many-constraint LP, strided-index, and promote/broadcast
  problems (`/tmp/equiv_check.py` style sweeps).
- **Build mode:** `maturin develop --release` always. Debug builds
  are 2–3× slower; if the maturin output says
  `[unoptimized + debuginfo]`, the build is wrong.

## Where the remaining cold-start wall-clock goes

`cProfile` of LASSO 200×500 (warm, 20 iters, 433ms total):

| Component                                  | Time   | Share |
|--------------------------------------------|-------:|------:|
| `cvxpy_rust.build_matrix` (Rust FFI)       | 128ms  | 30%   |
| `np.unique` in `reduce_problem_data_tensor`| 148ms  | 34%   |
| `numpy.argsort` (canonInterface)           |  76ms  | 18%   |
| `numpy.sort` (canonInterface)              |  40ms  |  9%   |
| Other (scipy CSC construction, misc)       |  40ms  |  9%   |

Rust is ~30% of total. The single biggest remaining lever is
`np.unique` in `canonInterface.py::reduce_problem_data_tensor`. Rust
already pre-sorts the COO output (so `np.unique` runs in O(n) timsort
rather than O(n log n)), but it's still called 6× per LASSO problem.
Eliminating it would require the Rust crate to also pre-uniquify and
return reduced indices, plus a path in `canonInterface` that detects
"already reduced" output and skips `reduce_problem_data_tensor`. This
is a cross-language interface change spanning `lib.rs`, `tensor.rs`,
`canon_backend.py`, and `canonInterface.py`; deferred.

## Potential future optimisations

1. **`np.unique` elimination** — biggest single lever (~34% of cold
   wall-clock). Cross-language interface change. Substantial blast
   radius.
2. **LinOp tree extraction overhead** — Discord-mentioned numpy-buffer
   serialisation of the LinOp tree for batch transfer (avoids per-node
   PyO3 calls). Benefits cold start where Rust FFI overhead is fixed.
3. **`faer` gemm for Dense × Dense** — handles nested
   `Mul(Const, Mul(Const, x))` chains and other Dense-Dense cases that
   the Identity fast path doesn't cover. Note: rare in practice
   because cvxpy's expression simplifier folds constant matrix
   products. *Not* the LASSO win — that turned out to be the rayon
   threshold.
4. **`cargo +nightly --release-with-debug` binary size / load time**.
   Would help cold start by reducing the time to load the .so.

## Files changed in the current cycle

- `cvxpy_rust/src/block.rs` (new) — Block IR types and conversions
- `cvxpy_rust/src/operations/leaf.rs` — `_block` variants for all 5
  leaf handlers
- `cvxpy_rust/src/operations/mod.rs` — `process_linop_block` parallel
  entry point
- `cvxpy_rust/src/operations/structural.rs` — `process_index_block`
  with lazy Identity → Identity/ColPerm
- `cvxpy_rust/src/operations/arithmetic.rs` — typed Identity probe in
  `process_mul`, generalised `mul_const_by_identity_offset`,
  bulk-emit dense path
- `cvxpy_rust/src/tensor.rs` — serial sort threshold in `from_tensor`
- `cvxpy_rust/src/lib.rs` — `mod block;`
