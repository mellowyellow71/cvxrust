# DiffEngine Insights for Rust Canonicalizer Optimization

## Background: What the Canonicalizer Actually Does

The CVXPY canonicalizer converts DCP problems into SOCP standard form using the procedure
from Boyd's "Code Generation for Second Order Cone Programs." It walks a tree of linear
operators (linops) and produces a sparse coefficient matrix `A` such that the problem is
in the form `min c^T x  s.t.  Ax + b in K`.

**Key insight: this process is mathematically equivalent to computing the Jacobian of the
affine expression tree via the chain rule.** For an affine expression `f(x) = Ax + b`,
the derivative `df/dx = A` IS the coefficient matrix. Each linop operation handler
(mul, index, transpose, etc.) is implicitly an adjoint rule in reverse-mode automatic
differentiation.

## The Current Rust Backend (`cvxpy_rust`)

**Location:** `cvxpy_rust/src/`

The Rust canonicalizer:
1. Receives a list of LinOp trees from Python via PyO3 (`lib.rs`)
2. Converts Python objects to Rust `LinOp` structs (`linop.rs`)
3. Walks each tree recursively via `process_linop()` (`operations/mod.rs`)
4. Each operation handler propagates sparse coefficient tensors (`operations/*.rs`)
5. Combines results into a single COO matrix (`matrix_builder.rs`)
6. Returns `(data, (row, col), shape)` back to Python

**What it does NOT do:** Any explicit derivative/gradient/Hessian computation. The
Jacobian computation is implicit in the tree walk — it just calls it "building a
coefficient matrix."

## The DiffEngine Approach (from cvxgrp/DNLP PR #180)

William's branch introduces a `_diffengine` C extension that takes a fundamentally
different approach:

### Architecture

```
CVXPY Problem
    |
    v
converters.py: convert_expr()    -- Recursively converts CVXPY atoms to C graph nodes
    |
    v
C_problem (c_problem.py)         -- Python wrapper around C differentiation engine
    |
    +--> init_jacobian_coo()      -- Analyze sparsity pattern ONCE (structural nonzeros)
    +--> init_hessian_coo_lower_tri()
    |
    +--> objective_forward(u)     -- Evaluate at a point
    +--> constraint_forward(u)
    |
    +--> gradient()               -- Compute derivatives (reuses forward pass state)
    +--> eval_jacobian_vals()     -- Evaluate ONLY the nonzero entries
    +--> eval_hessian_vals_coo_lower_tri(obj_factor, lagrange)
```

### Key Design Principles

1. **Separate structure from values.** The sparsity pattern is determined once via
   `init_jacobian_coo()` / `init_hessian_coo_lower_tri()`. Subsequent evaluations
   via `eval_jacobian_vals()` only compute the nonzero values. The current Rust
   canonicalizer interleaves sparsity discovery and value computation in a single pass.

2. **Build the graph once, evaluate many times.** `convert_expr()` builds a C-level
   computational graph once. This graph can then be evaluated repeatedly at different
   points (for NLP solvers) or with different parameter values (for parameterized
   CVXPY problems). The Rust canonicalizer currently re-walks the entire linop tree
   from scratch each time.

3. **Reuse forward pass state.** `gradient()` and `eval_jacobian_vals()` reuse the
   internal state set by `objective_forward()` / `constraint_forward()`. No redundant
   recomputation.

4. **Constants handled efficiently.** Converters in `_CONVERTERS_HANDLING_CONSTANTS`
   (MulExpression, multiply, QuadForm, SymbolicQuadForm) read constant values directly
   from expression arguments, bypassing expensive `make_constant()` copies.

### Converter Design (`converters.py`)

- `convert_expr(expr, var_dict, n_vars)` — main recursive converter, dispatches atoms
  through an `ATOM_CONVERTERS` dictionary (~40+ atom types)
- `build_variable_dict(variables)` — maps CVXPY variable IDs to C variables
- Specialized converters for matmul (sparse/dense detection), vstack (via hstack +
  permutation), indexing (Fortran-order flat indices), quad forms (size-based dispatch)

## Connection to Canonicalization Performance

### Why the Rust backend is slow for large problems

The current bottleneck is NOT Rust computation speed. It's:

1. **Per-node PyO3 overhead:** Each linop tree node requires a Python-to-Rust object
   conversion. For large problems (e.g., 1000-variable least squares), the tree has
   thousands of nodes, and the cumulative FFI overhead dominates.

2. **No computation reuse:** The entire linop tree is re-traversed for every call to
   `get_problem_data()`. For parameterized problems solved repeatedly, this is wasteful.

3. **Interleaved sparsity + values:** The canonicalizer discovers which matrix entries
   are nonzero while simultaneously computing their values. Separating these phases
   could enable better memory allocation and parallelization.

### Optimizations Inspired by DiffEngine

1. **Serialize the entire linop tree in one FFI call.** Instead of converting nodes
   one-by-one across the PyO3 boundary, serialize the full tree structure (e.g., as a
   flat buffer) on the Python side and pass it to Rust in a single call. This is the
   "batch the FFI calls" pattern.

2. **Cache the computation graph for parameterized problems.** Build the Rust-side
   representation of the linop tree once, cache it, and on subsequent solves with
   different parameter values, only recompute the values (not the structure). This
   mirrors DiffEngine's "build once, evaluate many times" pattern.

3. **Separate sparsity analysis from value computation.** Analyze the tree structure
   to determine the output sparsity pattern first (which entries in the coefficient
   matrix will be nonzero), pre-allocate the exact output buffers, then fill in values.
   This avoids dynamic Vec growth and enables better parallelization.

4. **Common subexpression elimination.** If the same sub-tree appears in multiple
   constraints (common after DCP expansion), compute its coefficients once and reuse.
   This is standard in sparse AD engines (JAX, CasADi, etc.).

5. **Exploit known sparsity patterns.** Operations like kronecker products with identity
   matrices have known output sparsity. Skip explicit computation of zero blocks rather
   than discovering zeros on-the-fly.

## Relevant Code Locations

### Rust Backend
- Entry point: `cvxpy_rust/src/lib.rs` (PyO3 `build_matrix` function)
- LinOp conversion: `cvxpy_rust/src/linop.rs` (Python object -> Rust struct)
- Core algorithm: `cvxpy_rust/src/matrix_builder.rs` (`build_matrix_internal`)
- Tree traversal: `cvxpy_rust/src/operations/mod.rs` (`process_linop`)
- Operations: `cvxpy_rust/src/operations/{arithmetic,leaf,structural,specialized}.rs`
- Sparse tensor: `cvxpy_rust/src/tensor.rs`

### Python Canon Backend
- Backend ABC + implementations: `cvxpy/lin_ops/canon_backend.py`
  - `RustCanonBackend` (line ~681): calls `cvxpy_rust.build_matrix()`
  - `PythonCanonBackend.process_constraint()`: recursive linop tree walk
  - `PythonCanonBackend.build_matrix()`: main entry point
- LinOp definition: `cvxpy/lin_ops/lin_op.py`

### Derivative/Sensitivity Code (existing in CVXPY)
- `problem.backward()` / `problem.derivative()`: `cvxpy/problems/problem.py` (~line 1249, 1384)
- Parameter Jacobian: `cvxpy/reductions/dcp2cone/cone_matrix_stuffing.py` (`ParamConeProg.apply_param_jac`, ~line 235)
- DIFFCP solver: `cvxpy/reductions/solvers/conic_solvers/diffcp_conif.py`

### DiffEngine (cvxgrp/DNLP PR #180)
- Converters: `cvxpy/reductions/solvers/nlp_solvers/diff_engine/converters.py`
- C problem wrapper: `cvxpy/reductions/solvers/nlp_solvers/diff_engine/c_problem.py`

## The Parameterized Tensor Connection

The canonicalizer builds a 3D tensor `T` (parameterized by parameter slices) where:
```
T @ param_vec = [A; b; c]
```

The derivative code in `ParamConeProg.apply_param_jac()` multiplies by `T^T` to
propagate gradients back to parameters. **Both operations use the same tensor `T`.**
If the Rust backend caches this tensor, both canonicalization and sensitivity analysis
benefit — no redundant computation.
