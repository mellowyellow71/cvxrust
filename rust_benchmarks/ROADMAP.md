# Rust Backend for CVXPY — Full Briefing & Roadmap

## What Is the Rust Backend?

CVXPY solves optimization problems in two phases:

1. **Canonicalization** — transforms your high-level problem (e.g., `minimize ||Ax - b||²`)
   into a standard matrix form (`P, q, A, b`) that a solver understands.
2. **Solving** — a solver (CLARABEL, OSQP, etc.) takes those matrices and finds the answer.

The Rust backend is a **drop-in replacement for Phase 1 only**. It replaces the Python/C++
canonicalization code with a Rust implementation compiled via PyO3. It does NOT replace or
modify any solver.

There are three backends:

| Backend | Language | How it works |
|---------|----------|--------------|
| **RUST** | Rust via PyO3 | Tree traversal + parallel tensor ops (rayon) |
| **CPP** | C++ via Cython | Legacy, doesn't support n-dim expressions |
| **SCIPY** | Pure Python + SciPy | Uses scipy.sparse heavily, calls into C/BLAS |

The Rust backend is selected by default when `cvxpy_rust` is importable.

---

## How Canonicalization Works (The Pipeline)

When you call `problem.solve()` or `problem.get_problem_data()`:

```
Step 1: Problem Definition
   minimize ||Ax - b||²

Step 2: DCP Reductions (Python)
   The problem is rewritten into standard conic form.
   For least-squares, this introduces auxiliary variable t:
     minimize  t^T @ t
     subject to  t = Ax - b
   
   Combined variable y = [t; x] of dimension m + n.

Step 3: Expression Tree → LinOp Tree (Python)
   Each expression/constraint becomes a tree of "LinOp" nodes:
     - Leaf nodes: Variable, DenseConst, SparseConst, Param
     - Arithmetic: Mul, Rmul, MulElem, Div, Neg
     - Structural: Index, Transpose, Hstack, Vstack, Reshape
     - Specialized: SumEntries, Trace, DiagVec, Kron, Conv

Step 4: LinOp Trees → Sparse Coefficient Matrix  ← THIS IS WHAT THE BACKEND DOES
   The backend traverses each LinOp tree and produces a sparse COO matrix
   that encodes ALL constraints in a single matrix.
   
   Entry point: CanonBackend.build_matrix()
     → Rust: cvxpy_rust.build_matrix() (FFI call)
     → SciPy: PythonCanonBackend.build_matrix()
   
   Output: COO data → scipy.sparse.csc_array

Step 5: Matrix Stuffing (Python)
   The coefficient matrix is used to construct:
     P  — quadratic objective (m+n × m+n)
     q  — linear objective
     AF — constraint matrix (equalities + inequalities stacked)
     bg — constraint bounds

Step 6: Solver Interface (Python)
   AF is split into A (equalities) and F (inequalities).
   Matrices are passed to the solver.
```

### The Specific Math for Least Squares

For `minimize ||Ax - b||²` with `A` being `m×n`:

The canonical variable is `y = [t; x]`, dimension `m + n`.

**P matrix** (quadratic objective): `(m+n) × (m+n)`
```
P = [ 2*I_m   0 ]
    [   0     0 ]
```
2×Identity on the `m` auxiliary vars, zeros on the `n` original vars. Encodes `Σ tᵢ²`.

**AF matrix** (equality constraint): `m × (m+n)`
```
AF = [ -I_m | A ]
```
Negative identity on auxiliary block, then the dense A on the original block.
The equality `AF @ y + bg == 0` gives `-t + Ax = b`, i.e., `t = Ax - b`.

**This is why the dense matrix `A` (5000×1000 = 5M entries) appears in the hot path.**

---

## The Rust Backend Architecture

### Source Files

```
cvxpy_rust/
├── Cargo.toml                    # Dependencies: pyo3, rayon, faer, ndarray, sprs
├── src/
│   ├── lib.rs                    # PyO3 entry point: build_matrix() FFI function
│   ├── linop.rs                  # LinOp struct + Python extraction
│   ├── matrix_builder.rs         # Core algorithm: parallel/sequential dispatch
│   ├── tensor.rs                 # SparseTensor (COO format) + BuildMatrixResult
│   └── operations/
│       ├── mod.rs                # process_linop() — dispatcher for 22 op types
│       ├── leaf.rs               # Variable, constants, parameters
│       ├── arithmetic.rs         # Mul, Rmul, MulElem, Div, Neg  ← HOT PATH
│       ├── structural.rs         # Index, Transpose, Hstack, Vstack, etc.
│       └── specialized.rs        # SumEntries, Trace, Diag, Kron, Conv
```

### Python-Side Integration

```
cvxpy/
├── lin_ops/canon_backend.py      # Backend ABC + RustCanonBackend (line ~681)
├── settings.py                   # Backend selection priority: RUST > CPP > SCIPY
├── reductions/solvers/
│   ├── solving_chain_utils.py    # Backend dispatch logic
│   └── qp_solvers/
│       ├── qp_solver.py          # AF splitting (lines 90-101) ← BOTTLENECK
│       └── osqp_qpif.py          # OSQP: vstacks A+F back together (line 91)
```

### Data Flow Through Rust

```
Python LinOp tree (nested Python objects)
    ↓  [PyO3 extraction — lib.rs]
Rust LinOp tree (native Rust structs)
    ↓  [matrix_builder.rs — parallel or sequential]
Per-constraint SparseTensors
    ↓  [tensor.rs — combine()]
Single combined SparseTensor
    ↓  [BuildMatrixResult::from_tensor]
COO arrays (data, row, col) returned to Python
    ↓  [Python — canon_backend.py]
scipy.sparse.csc_array
```

---

## Why Rust Is Slower on the Least-Squares Benchmark

### The Benchmark

```python
# rustybench.py — n=1000, m=5000, dense A matrix
x = cp.Variable(n)
minimize(sum_squares(A @ x - b))
```

### Root Cause: Dense Matrix Multiplication Without BLAS

The critical hot path is in `arithmetic.rs`, function `multiply_dense_block_diagonal_colmajor`
(line 664). This function computes `kron(I_k, A) @ tensor` — which for the least-squares
case means multiplying the dense 5000×1000 matrix `A` against the identity tensor for
variable `x`.

**What the Rust code does** (lines 685-709):
```rust
for idx in 0..rhs.nnz() {           // For each nonzero in the RHS tensor
    let col_in_block = rhs_row % a_cols;
    let a_col = &data[col_start..col_start + a_rows];  // Get column of A
    for (i, &a_val) in a_col.iter().enumerate() {       // SCALAR LOOP
        if a_val != 0.0 {
            rows.push(new_row as i64);
            vals.push(a_val * rhs_val);                 // Element-by-element
        }
    }
}
```

This is a **scalar loop** over 5M entries. It pushes results one-by-one into Vecs.

**What SciPy does**: Calls into C/Cython sparse matrix routines which ultimately use
**BLAS** (Basic Linear Algebra Subroutines) — hand-tuned assembly with SIMD (AVX/SSE),
cache-optimal blocking, and decades of optimization for exactly this kind of operation.

**The gap**: For `n=1000, m=5000`, the Rust backend generates ~5M output entries via scalar
loops. SciPy generates the same entries via BLAS. BLAS wins by ~1.7x per call, which
compounds to the overall slowdown you see.

### Why Rust Wins on Many-Constraint Problems

When you have 1000 small constraints (each a few variables), the bottleneck is Python loop
overhead — iterating over constraints in Python, calling scipy functions per constraint, etc.
Rust eliminates this overhead entirely with native iteration + rayon parallelism.

| Problem Type | Bottleneck | Winner |
|-------------|-----------|--------|
| Many small constraints | Python loop overhead | **Rust (3-4x)** |
| Few large dense constraints | Dense matrix arithmetic | **SciPy (1.5x)** |
| Mixed/typical problems | Both matter | **Rust (~1.2x)** |

---

## What the Researchers Found (context.md Explained)

### SparseDiffEngine Approach

A researcher achieved **3.7s** compile time (vs 9.7s CPP, 11.8s SciPy) by using a
fundamentally different algorithm — **reusing derivative computations** from a "diffengine"
instead of building LinOp trees and traversing them.

Key insight: for `mul(A, x)`, the current system computes the Jacobian as `A @ I_x`
(A times identity of dimension x). This is silly — the answer is just `A` itself. The
diffengine approach recognizes this and short-circuits.

### PR #3240: Post-Canonicalization Bottlenecks

After `build_matrix` returns, there are additional bottlenecks that affect ALL backends:

**Problem 1: Row slicing on CSC matrices is slow**
```python
# qp_solver.py lines 91-98
A = AF[:len_eq, :]      # Row-slice a CSC matrix — requires scanning all columns
F = -AF[len_eq:, :]     # Another row-slice + negation
```
CSC (Compressed Sparse Column) stores data column-by-column. Extracting rows requires
traversing every column's index array. For 80M+ nonzeros, this takes significant time.

**Problem 2: OSQP vstacks A and F back together**
```python
# osqp_qpif.py line 91
A = sp.vstack([data[s.A], data[s.F]]).tocsc()  # Undo the split we just did!
```
This is a pointless round-trip: split AF → A, F → vstack(A, F) → AF again.

**Problem 3: Unnecessary format conversions**
Multiple `.tocsc()`, `.tocsr()` calls throughout the pipeline, each copying data.

**The Fix (partially implemented):**
Pass `AF` directly to the solver. Instead of negating F rows, flip the constraint bounds:
```
Old: A @ x = b,  F @ x <= g    (requires split)
New: AF @ x + bg,  bounds = [b, -inf..g]  (no split needed)
```
This is mathematically equivalent and avoids all three bottlenecks. Currently only OSQP
has a partial version of this. It needs to be extended to CLARABEL and other solvers.

### Dense Matrix Class (SparseDiffEngine PR #49)

SparseDiffEngine added a dedicated dense matrix class so that the dense `A` matrix can
be passed directly without converting to sparse format first. The current Rust backend
does keep dense matrices in dense format (good), but doesn't use BLAS for the actual
multiplication (the gap).

### BLAS Copy Fast Path (SparseDiffEngine PR #51)

When evaluating the Jacobian of `A @ x`, the diffengine recognizes that sparse row
vectors with a single entry can use BLAS copies instead of full sparse operations.
The Rust backend has no equivalent optimization.

---

## Current Optimizations Already Implemented

| Optimization | File | Status |
|-------------|------|--------|
| Column-major (F-order) matrix storage | `arithmetic.rs` | Done |
| Fast paths for `select_rows` (identity, contiguous, reversed) | `tensor.rs` | Done |
| Work-based parallel threshold (rayon) | `matrix_builder.rs` | Done |
| Pre-allocated output vectors | `arithmetic.rs` | Done |
| `faer` crate in Cargo.toml (unused) | `Cargo.toml` | Available |

---

## Roadmap: What To Work On

### Phase 1: Close the BLAS Gap (Highest Impact)

**Goal**: Make the least-squares benchmark faster than SciPy.

**Task 1.1: Use `faer` for dense matrix multiplication**

The `faer` crate is already in `Cargo.toml` but unused. Replace the scalar loop in
`multiply_dense_block_diagonal_colmajor` (arithmetic.rs:664) with a `faer`-backed
dense matmul.

The current code does:
```rust
// For each nonzero in RHS tensor, iterate over A's column elements one-by-one
for (i, &a_val) in a_col.iter().enumerate() {
    vals.push(a_val * rhs_val);
}
```

The replacement should:
1. Detect when the RHS tensor represents an identity-like pattern (variable node)
2. Use `faer::Mat` to perform the dense multiplication as a single BLAS call
3. Convert the dense result back to COO entries

For the least-squares case, this turns 5M scalar operations into a single optimized
matrix multiply.

**Task 1.2: Short-circuit `A @ x` Jacobian**

When the expression is `Mul(DenseConst(A), Variable(x))`, the Jacobian is just `A` itself.
The current code computes `A @ I_x` through the full tensor machinery.

Add a fast path in `process_mul`:
```rust
// If lhs is a dense constant and rhs is a plain variable (identity tensor),
// the result is just lhs with appropriate column mapping
if rhs_is_identity_variable(rhs, lin_op, ctx) {
    return dense_const_to_tensor(lhs_data, lin_op, ctx);
}
```

This would skip the multiplication entirely for the most common expression pattern.

**Expected impact**: 2-4x speedup on the least-squares benchmark. Should make Rust
faster than SciPy for this problem.

### Phase 2: Fix Post-Canonicalization Bottlenecks (All Backends Benefit)

**Goal**: Eliminate the AF split/rejoin overhead.

**Task 2.1: Extend AF pass-through to CLARABEL**

Currently only OSQP has a partial AF fast path. Implement the same pattern for CLARABEL
(`cvxpy/reductions/solvers/conic_solvers/clarabel_conif.py`):
- Accept `AF` directly instead of separate `A` and `F`
- Use bound manipulation instead of matrix negation

**Task 2.2: Avoid row-slicing CSC matrices**

In `qp_solver.py` lines 91-98, add a fast path:
- If the solver supports AF pass-through, skip the split entirely
- If splitting is necessary, convert to CSR first (row operations are O(1) in CSR)

**Task 2.3: Remove redundant format conversions**

Audit the pipeline for unnecessary `.tocsc()` / `.tocsr()` calls. Track the format
through the pipeline and only convert when the target format actually differs.

**Expected impact**: Measurable speedup for large problems (80M+ nonzeros) across
all three backends.

### Phase 3: Algorithmic Improvements

**Goal**: Reduce work done, not just do the same work faster.

**Task 3.1: Cache constant expression evaluations**

When the same constant sub-expression appears in multiple constraints, it gets
re-evaluated each time. Add memoization keyed on LinOp identity.

**Task 3.2: Batch FFI extraction**

Currently LinOp trees are extracted from Python one node at a time via PyO3.
Batch the extraction: serialize the entire tree on the Python side, deserialize
once in Rust.

**Task 3.3: Investigate SparseDiffEngine integration**

The diffengine approach (reusing derivative computations) achieved significant speedups.
This is a larger architectural change — study the DNLP PR (#180) and SparseDiffEngine
PRs to understand if/how this approach could be integrated into the Rust backend.

### Phase 4: Advanced Optimizations

**Task 4.1: SIMD for hot loops**

For operations that can't use BLAS (sparse-dense mixed operations), use explicit SIMD
intrinsics (`std::arch` or `packed_simd`) for the inner loops.

**Task 4.2: Better parallelism for the combine step**

After parallel constraint processing, `SparseTensor::combine()` concatenates results
sequentially. Use a parallel merge strategy (e.g., tree reduction with rayon).

**Task 4.3: Return CSC directly from Rust**

Currently Rust returns COO which Python converts to CSC. Building CSC directly in Rust
would avoid the conversion overhead. This is tricky because CSC requires sorted
column-major order.

---

## How To Approach This Work

### Getting Started

```bash
# Run the benchmark to establish your baseline
cd /home/ray/cvxrust/cvxpy
python rust_benchmarks/rustybench.py

# Run the full benchmark suite
python rust_benchmarks/benchmark_rust_backend.py

# Profile the Rust backend
python rust_benchmarks/profile_rust_backend.py
```

### Development Workflow

1. **Make a change** in `cvxpy_rust/src/`
2. **Rebuild**: `pip install -e .` (or `maturin develop --release` if using maturin)
3. **Benchmark**: `python rust_benchmarks/rustybench.py`
4. **Test**: `python -m pytest cvxpy/tests/test_rust_backend.py -x`

### Key Files to Edit (By Priority)

1. `cvxpy_rust/src/operations/arithmetic.rs` — BLAS integration, fast paths
2. `cvxpy/reductions/solvers/qp_solvers/qp_solver.py` — AF splitting optimization
3. `cvxpy_rust/src/matrix_builder.rs` — parallelism improvements
4. `cvxpy_rust/src/lib.rs` — FFI optimizations

### Testing Strategy

- `test_rust_backend.py` — 24 LinOp type tests, correctness against SciPy
- `test_python_backends.py` — Backend comparison tests
- `rustybench.py` — Quick performance check
- `benchmark_rust_backend.py` — Full benchmark suite with multiple problem types

Always run correctness tests after performance changes. It's easy to break edge cases
when optimizing hot paths.

---

## Summary: The Three Layers of Performance Work

```
Layer 1: BLAS Gap (Rust-specific)
  ├── Use faer for dense matmul         → fixes least-squares benchmark
  └── Short-circuit A @ x Jacobian      → eliminates unnecessary work

Layer 2: Pipeline Overhead (All backends)
  ├── AF pass-through to all solvers    → eliminates split/rejoin
  ├── CSR for row operations            → faster slicing
  └── Remove format conversions         → less copying

Layer 3: Algorithmic (Future)
  ├── Constant memoization              → less redundant work
  ├── Batch FFI extraction              → less Python↔Rust overhead
  └── SparseDiffEngine ideas            → fundamentally less computation
```

The Rust backend is architecturally sound — it correctly implements all 22 LinOp types,
has parallel processing, and is the default backend. The performance gap on dense problems
is a known, fixable issue (BLAS integration), not a fundamental design flaw.
