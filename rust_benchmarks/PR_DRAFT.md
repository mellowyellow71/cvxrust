# DRAFT — cvxpy/benchmarks PR (do not send until reviewed)

Status: **awaiting your review and edits.** Nothing has been pushed anywhere.
The payload lives on the local branch `backend-canonicalization-benchmarks` in
`rust_benchmarks/cvxpy-benchmarks/` (2 files: `benchmark/canonicalization_backends.py`,
`README.md`). To send after review: fork `cvxpy/benchmarks` on GitHub, push the
branch to your fork, open the PR against `cvxpy/benchmarks` `main`.

---

## Proposed title

Add canonicalization backend comparison benchmarks

## Proposed body

CVXPY now has several canonicalization backends — `SCIPY`, the new `COO`
backend (cvxpy#3031), the `CPP` (cvxcore) backend, and the experimental `RUST`
backend (cvxpy#3018, whose stated blocker is "more complete benchmarking").
The existing benchmarks in this repo time whole problems on the default
backend; nothing compares the backends head-to-head on the expression shapes
they process.

This PR adds `benchmark/canonicalization_backends.py` with four benchmark
classes, all parameterized over `["SCIPY", "COO", "CPP", "RUST"]`:

- **`BackendCompileCanonicalization`** — end-to-end `get_problem_data` for 16
  cases covering the affine atom families (matmul/multiply/divide, rmul +
  promote, hstack/vstack, concatenate, diag/trace/kron with both `kron_r` and
  `kron_l`, convolve, indexing/transpose/reshape), ND arrays and broadcasting,
  `einsum`, deep and wide expression trees, a parameterized LP (the COO
  backend's native workload), a cone-heavy composite (norm1 + huber +
  quad_form), and murray-type dense constants in two density regimes (below
  and above `SPARSE_DENSITY_THRESHOLD`, so sparsification heuristics can be
  observed separately from the dense path).
- **`BackendBuildMatrixCanonicalization`** — the same cases, but timing only
  the backend's `build_matrix` call on captured LinOp trees, isolating backend
  cost from the rest of the compilation chain.
- **`DeepExpressionTreeScaling`** / **`WideExpressionTreeScaling`** —
  `get_problem_data` as tree depth (`-(-(...-(x)))`, depth 4→256) and sum
  width (n matmul terms, 8→256) grow.

Design notes:

- Backends that are unavailable in the installed CVXPY raise
  `NotImplementedError` in `setup()`, so asv reports **n/a instead of
  failing**: `RUST` is gated on `import cvxpy_rust`, `COO` on the
  `COO_CANON_BACKEND` constant, and `CPP` cells are skipped for expressions
  the C++ core does not support (checked via the same contract
  `get_problem_data` enforces). The suite therefore runs against PyPI cvxpy
  on CI today, and RUST cells light up automatically wherever the extension
  is installed.
- Each timed compile builds a **fresh `Problem`** (the solving chain caches
  `get_problem_data` results per instance), while expensive problem
  construction stays in `setup()` outside the timed region.
- A module-level **`LINOP_COVERAGE`** dict documents which cases exercise
  each LinOp node type the backends process — every type is covered except
  the unreachable `no_op`.
- The deep-tree benchmark raises the recursion limit in `setup()` (CVXPY's
  tree walks recurse per nesting level; the default 1000-frame limit is too
  tight at depth 256 for some backends).

The README gains a section documenting the new benchmarks, including
`asv run --python=same` for benchmarking a locally built CVXPY (required for
`RUST`, whose extension is not on PyPI).

Validation: `asv check` passes; ruff (`E,F,I`, line-length 100) clean; on a
local CVXPY master build with all four backends available, every case×backend
cell runs, and all four backends produce numerically identical stuffed
tensors on every case (max abs diff 0.0, parameter slices included).

Follows the precedent of `backends/dpp_canonicalization.py` for backend
comparison benchmarks, but inside `benchmark_dir` so asv discovers it.

---

## Reviewer notes for you (not part of the PR body)

- The commit message on the local branch mirrors the summary above; amend
  freely (`git commit --amend` in the clone).
- The validation claims are backed by: `asv check --config asv_local.conf.json`
  (No problems found), `ruff check` (clean), the runnability gate (0 failures,
  8 contract-based n/a cells), and `rust_benchmarks/verify_backends.py`
  (ALL MATCH, 18 cases × 3 backends vs SCIPY).
- `asv_local.conf.json` is intentionally NOT in the commit (local-only helper;
  upstream's `asv_pr.conf.json` expects the benchmarks clone to sit directly
  inside the cvxpy repo root).
- If maintainers ask why einsum/ND/concatenate CPP cells are n/a: those
  expressions are rejected by `Problem._supports_cpp()`; the suite honors the
  same contract `get_problem_data` enforces.
