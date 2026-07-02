# FFI Overhead Analysis: the Arena-Allocator Question

**Question (from the project maintainer):** should the Rust backend use an
arena allocator for LinOp node handling, instead of / in addition to the
Python-side serialization of the LinOp tree?

**Short answer:** the two are complementary, not alternatives — but neither
the arena nor any further format engineering is the right next move. The
measurements below show where the FFI time actually goes, what we changed
based on them, and why the arena is deferred.

All measurements: Apple M5 (Rosetta x86_64 conda env `cvxpy-py313`),
`many_constraints (m=5000)` from `benchmark_suite.py` — 5,000 constraint
trees, 30,000 LinOp nodes — the workload class with the most per-node FFI
work. Enable the instrumentation with `CVXPY_RUST_FFI_PROFILE=1` (prints
deser/build phase times from `build_matrix_serialized`).

## How the cost decomposes (m=5000, ~24ms total)

| phase | v1 (tuple nodes) | v2 (i64 meta stream, current) |
|---|---|---|
| Python `serialize_linop_trees` | ~12.2ms | ~13.7ms |
| Rust deserialization | ~4.9ms (35% of Rust call) | **~1.4ms** (14%) |
| Rust build (two-pass + sort) | ~9ms | ~9ms |

- **v1** encoded each node as a Python tuple; Rust paid ~8-12 PyO3
  `get_item`/`extract` calls per node (~160ns/node total deser).
- **v2 (current)** packs all node metadata into one flat `np.int64` array
  (layout documented in `serialize_linop_trees` and
  `linop.rs::DeserializationContext`). Rust deser is a pure slice walk:
  zero per-node Python access, ~48ns/node. 3.4x faster deser, ~1.5ms more
  Python-side cost, net ~2ms better on many-constraint problems and neutral
  elsewhere (dense-problem deser is dominated by the one `Arc::from` memcpy
  of constant data, ~0.4ms for a 2M-element matrix — unavoidable while the
  build runs with the GIL released).

## What an arena would and wouldn't buy now

The remaining ~48ns/node of deser is mostly the 2 heap allocations per node
(`shape: Vec<usize>`, `args: Vec<LinOp>`) plus stream parsing. An
index-based arena (`Vec<LinOpNode>` + u32 child ranges — bumpalo's
lifetimes fight PyO3, so indices are the right shape) could roughly halve
it: **upper bound ~0.7ms on a ~24ms call (~3%)**, at the cost of threading
arena+index through ~25 `process_*` signatures in 4 operation files.

When the maintainer suggested the arena, the context was his original
implementation where Rust extracted nodes from Python one `getattr` at a
time — there, per-node costs dominated and an arena looked attractive.
Python-side serialization (v1) removed the bulk of that, and the meta
stream (v2) removed what was left of the Rust-side share. The arena's
target has shrunk to low-single-digit percent.

## Where the FFI time actually is now

**Python serialization is now the largest FFI cost by far (~13.7ms vs
1.4ms).** It is per-node *interpreter* work — tuple building, branching,
attribute access — and resists micro-optimization: we measured variants
(int-list appends 18.7ms → hoisted locals + struct-based f64 bit-casting +
single-extend-per-node 13.2ms → explicit-stack traversal 13.9ms — the
recursion is not the bottleneck on CPython 3.13; `array('q')` appends and
`np.fromiter` are both ~2x slower than list+`np.array`).

The way to eliminate it is not a better format but **not re-serializing at
all**: cache `(node_meta, float_data, int_data)` keyed by problem structure
and patch only constant values on re-compile. The flat-buffer format makes
that cache natural (three arrays, no object graphs) — which is the lasting
benefit of v2 beyond the ~2ms.

## Decision

- **Arena: deferred.** Upper bound ~3% on the workload it helps most, large
  refactor surface. Revisit only if a workload appears where deser_share
  (via `CVXPY_RUST_FFI_PROFILE=1`) exceeds ~15-20% again.
- **Kept:** the i64 meta-stream serialization (v2) and the phase-timing
  instrumentation.
- **Next FFI-related lever, if needed:** serialization caching across
  `get_problem_data` calls, not allocation tuning.
