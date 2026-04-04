# First Run Analysis: CVXPY Rust Canonicalization Backend

## Benchmark Setup

- **Python**: 3.13 (via `/opt/anaconda3/envs/cvxpy-py313`)
- **Rust extension**: pre-built `cvxpy_rust.cpython-313-darwin.so`
- **Note**: Cold-start benchmark script (`quick_benchmark.py`) fails under default Python 3.9 — `typing.Self` requires Python ≥ 3.11. Ran in-process benchmark instead.

## Results: Least Squares — `minimize 0.5 * ||Ax - b||²`


| Problem       | RUST     | SCIPY    | CPP      | RUST/SCIPY   |
| ------------- | -------- | -------- | -------- | ------------ |
| n=50, m=100   | 4.38ms   | 3.71ms   | 3.15ms   | 1.18x slower |
| n=200, m=500  | 50.36ms  | 23.66ms  | 18.11ms  | 2.13x slower |
| n=500, m=2000 | 483.77ms | 230.37ms | 171.38ms | 2.10x slower |


**The Rust backend is currently ~2x slower than SciPy and ~2.8x slower than C++.**

---

## How CVXPY Canonicalization Works

### Overview

CVXPY implements *Disciplined Convex Programming* (DCP). When you call
`prob.get_problem_data(solver)`, it transforms the user's problem into a
**standard conic form** that a solver like CLARABEL or SCS can consume:

```
minimize    c'x
subject to  A x + b ∈ K
```

where K is a product of cones (zero cone for equality, non-negative orthant,
second-order cone, PSD cone, etc.). This transformation is called
**canonicalization**, and it is the step being benchmarked.

### Step 1: Atom Canonicalization (DCP → Conic Form)

Each CVXPY atom has a *graph implementation* that rewrites it into affine
constraints plus auxiliary variables. For example:

```
minimize 0.5 * sum_squares(A @ x - b)
```

gets rewritten (approximately) as:

```
minimize  t
subject to  ||A @ x - b||₂² ≤ t
```

which is further cast as a second-order cone constraint:

```
(t, Ax - b) ∈ SOC
```

This follows the approach in Boyd et al., *"Code Generation for Second-Order
Cone Programs"* (CVXGEN paper). Every atom is reduced to affine maps plus
membership in a primitive cone, and auxiliary variables fill in the slack.

### Step 2: LinOp Tree Construction

After atom canonicalization, every affine expression is represented as a tree
of **linear operations** (`LinOp` nodes). For `A @ x - b` the tree is:

```
Sum
├── Mul(data=DenseConst(A))   ← left-multiply x by A
│   └── Variable(x)
└── Neg
    └── DenseConst(b)
```

Each `LinOp` node has:

- `type` — operation kind (`mul`, `neg`, `sum`, `variable`, `dense_const`, ...)
- `shape` — output dimensions
- `args` — child nodes (inputs to the operation)
- `data` — constant operand (for `mul`, `dense_const`, `index`, etc.)

### Step 3: Matrix Building (the Benchmarked Step)

The matrix builder does a **post-order traversal** of every LinOp tree,
accumulating a sparse coefficient tensor that maps variable values to
expression values.

Because CVXPY supports **parametric problems** — where constants can be
updated without re-canonicalizing — the coefficient representation is a 3D
tensor `T[row, col, param_offset]`:

- Axis 0 (`row`): index into the constraint/objective vector
- Axis 1 (`col`): index into the variable vector
- Axis 2 (`param_offset`): which parameter this coefficient belongs to;
`-1` (or `Constant.ID`) means a literal numeric constant

This 3D tensor is stored in COO-like form as `TensorRepresentation(data, row, col, parameter_offset)` in `canon_backend.py`.

Each op type implements a specific linear algebra transformation:


| Op             | Action                                         |
| -------------- | ---------------------------------------------- |
| `variable`     | Identity block at the variable's column offset |
| `dense_const`  | Literal matrix; contributes to constant column |
| `sparse_const` | Same, but sparse                               |
| `mul`          | Left-multiply child tensor by constant matrix  |
| `rmul`         | Right-multiply child tensor                    |
| `neg`          | Negate all values                              |
| `sum`          | Sum entries along an axis                      |
| `index`        | Row-select / slice                             |
| `transpose`    | Permute axes                                   |
| `reshape`      | No-op on the tensor (shape metadata only)      |


### Step 4: Output

The accumulated tensor slices are stacked into a final sparse matrix (CSC
format) and a constant vector `b`. These, plus the objective vector `c`, are
what get handed to the solver.

---

## Why the Rust Backend Is Slower

### 1. `dbg!()` macro left in production code (`lib.rs:52`)

```rust
let rust_lin_ops: Vec<LinOp> = dbg!(lin_ops
    .iter()
    .map(|obj| LinOp::from_python(obj))
    .collect::<PyResult<Vec<_>>>()?);
```

`dbg!()` formats and prints the entire LinOp tree to stderr on every call.
For a problem with a large dense matrix this is a significant string
allocation and I/O hit. This should be removed.

### 2. Dense intermediate tensors for `mul`

For `A @ x` where A is an m×n dense matrix, the Rust backend builds a
dense intermediate `SparseTensor` before finalizing. The SciPy backend
delegates directly to `scipy.sparse` routines (written in C/Fortran) that
are highly optimized for exactly this pattern. The performance gap widens
with problem size (2.1x at n=200, still 2.1x at n=500), suggesting the
bottleneck is in the matrix arithmetic itself, not PyO3 overhead.

### 3. Data copying through PyO3 boundary

`linop.rs:extract_dense_array` calls Python's `numpy.ravel('F')` and then
extracts the result as a `Vec<f64>`, making a full copy of the matrix data.
This happens before any Rust computation starts.

---

## Next Steps

1. **Remove `dbg!()` from `lib.rs:52`** — immediate, zero-cost fix.
2. **Profile `process_mul`** — this is almost certainly where the time goes
  for dense least squares; consider leveraging BLAS via `faer` or `ndarray`
   for the dense matrix multiply instead of the manual sparse accumulation.
3. **Avoid `ravel('F')` copy** — use numpy's buffer protocol directly via
  `PyReadonlyArrayDyn` to get a zero-copy view of the array data.
4. **Fix cold-start benchmark** — `quick_benchmark.py` uses `typing.Self`
  which requires Python ≥ 3.11; the script spawns subprocesses with the
   default `python` (3.9 on this machine).

