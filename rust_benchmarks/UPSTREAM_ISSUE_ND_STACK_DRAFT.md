# Draft upstream issue (Ray to review and post to cvxpy/cvxpy)

Status 2026-09-16: verified on upstream `773163ebd` with scipy 1.18.1, in the
`rust-backend-pr-20260915` worktree. Not posted.

## Title

N-D `hstack` and `concatenate(axis=None)` canonicalize differently from their own `.value`

## Body

The stuffed affine map for `cp.hstack` on N-D arguments and for
`cp.concatenate(..., axis=None)` (any ndim) disagrees with the atom's numeric value.
Both Python backends (SCIPY, COO) produce the same wrong map, so a backend-vs-backend
comparison cannot see it; a direct numeric oracle does.

Repro (6 lines):

```python
import numpy as np, cvxpy as cp
x = cp.Variable((3, 2, 2)); x0 = np.arange(12.0).reshape(3, 2, 2)
expr = cp.hstack([x, x])                       # same with cp.concatenate([x, x], axis=None)
data, _, _ = cp.Problem(cp.Minimize(0), [expr == 0]).get_problem_data(cp.CLARABEL)
x.value = x0
print(np.allclose((data["A"] @ x0.ravel(order="F")).reshape(expr.shape, order="F"), expr.value))  # False
```

Observed (first 8 entries of the mapped vector vs `expr.value`, F-order):

| case | canonicalized | `.value` (numpy) | SCIPY | COO |
| --- | --- | --- | --- | --- |
| `hstack` of two (3,2,2) | `[0 0 2 2 1 1 3 3 ...]` | `[0 1 2 3 0 1 2 3 ...]` | wrong | wrong |
| `concatenate(axis=None)` of two (2,3) | `[0 3 1 4 2 5 0 3 ...]` | `[0 1 2 3 4 5 0 1 ...]` | wrong | wrong |
| `concatenate(axis=None)` of two (3,2,2) | `[0 4 8 2 6 10 1 5 ...]` | `[0 1 2 3 4 5 6 7 ...]` | wrong | wrong |
| `concatenate(axis=1 or 2)` of two (3,2,2) | matches | | ok | ok |
| `vstack` of two (3,2,2) | matches | | ok | ok |
| 2-D `hstack` | matches | | ok | ok |

Cause: `PythonCanonBackend.hstack` (`cvxpy/lin_ops/backends/base.py`) stacks the
arguments' F-order flattenings end to end. For 2-D arguments that equals concatenation
along axis 1, but for N-D arguments numpy's `hstack` concatenates along axis 1 of the
N-D array, which is a different row order. `concatenate(axis=None)` reuses that flat
stacking (`order = np.arange(...)`), while numpy flattens each argument in C order before
concatenating. A least-squares recovery of `x0` from the stuffed map fails accordingly.

Suggested fix: route N-D `hstack` through the `concatenate(axis=1)` index permutation, and
build the `axis=None` permutation from `np.arange(size).reshape(arg.shape, order="F").ravel(order="C")`
per argument.

Related, COO-only: `cp.multiply(sparse_3d_constant, x)` raises
`ValueError: inconsistent shapes (8, 8) and (4, 1)` in the COO backend, and
`test_atoms.py::TestAtoms::test_lambda_sum_largest_nd_solve` fails with
`CVXPY_DEFAULT_CANON_BACKEND=COO` for the same reason (`inconsistent shapes (8, 2) and (4, 1)`).
Happy to split that into its own issue.

A cross-backend test that checks `A @ vec_F(x0) + b == expr.value` for a handful of N-D
atoms (with these two cases marked xfail) is included in the Rust backend PR branch; it can
be lifted out separately if useful.
