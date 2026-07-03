#!/usr/bin/env python
"""
Cross-backend canonicalization correctness gate.

For every case in the ASV backend suite (cvxpy-benchmarks/benchmark/
canonicalization_backends.py), capture the LinOp trees once, build the
stuffed [A b] tensor with the SCIPY reference backend, and assert that
RUST, CPP, and COO produce numerically identical tensors (including all
parameter slices).

Run this before every timing campaign and after any Rust-side change:

    python verify_backends.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import cvxpy as cp

BENCH_DIR = Path(__file__).resolve().parent / "cvxpy-benchmarks" / "benchmark"
sys.path.insert(0, str(BENCH_DIR))

import canonicalization_backends as cb  # noqa: E402

ATOL = 1e-10
CANDIDATES = ["RUST", "CPP", "COO"]
REFERENCE = "SCIPY"


def _as_csc(matrix) -> sp.csc_matrix:
    return sp.csc_matrix(matrix)


def _max_abs_diff(a, b) -> float:
    d = _as_csc(a) - _as_csc(b)
    return 0.0 if d.nnz == 0 else float(abs(d).max())


def _verify_case(name: str, problem: cp.Problem, solver: str,
                 supports_cpp: bool) -> list[str]:
    failures = []
    captured = cb._capture_build_matrix_call(problem, solver)
    reference = cb._build_matrix(captured, REFERENCE)
    for backend in CANDIDATES:
        if not cb._available_backend(backend):
            print(f"  n/a   {name} x {backend}: backend unavailable")
            continue
        if backend == "CPP":
            checker = getattr(problem, "_supports_cpp", None)
            if not supports_cpp or (checker is not None and not checker()):
                print(f"  n/a   {name} x {backend}: unsupported expressions")
                continue
        try:
            result = cb._build_matrix(captured, backend)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {name} x {backend}: raised {type(exc).__name__}: {exc}")
            failures.append(f"{name} x {backend} (raised)")
            continue
        if _as_csc(result).shape != _as_csc(reference).shape:
            print(f"  FAIL  {name} x {backend}: shape "
                  f"{_as_csc(result).shape} != {_as_csc(reference).shape}")
            failures.append(f"{name} x {backend} (shape)")
            continue
        diff = _max_abs_diff(result, reference)
        if diff > ATOL:
            print(f"  FAIL  {name} x {backend}: max|diff| = {diff:.3e}")
            failures.append(f"{name} x {backend} (values)")
        else:
            print(f"  ok    {name} x {backend}: max|diff| = {diff:.1e}")
    return failures


def main() -> int:
    all_failures: list[str] = []

    print(f"== ASV cases ({len(cb.CASES)}) vs {REFERENCE} ==")
    for case in cb.CASES:
        problem, solver = case.factory()
        all_failures += _verify_case(case.name, problem, solver, case.supports_cpp)

    print("== tree-scaling shapes ==")
    for cls, n in [(cb.DeepExpressionTreeScaling, 256), (cb.WideExpressionTreeScaling, 256)]:
        inst = cls()
        inst.setup(n, REFERENCE)
        problem = cp.Problem(inst.objective)
        all_failures += _verify_case(f"{cls.__name__}[{n}]", problem, cp.CLARABEL, True)

    print(f"\n{'FAILURES: ' + str(len(all_failures)) if all_failures else 'ALL MATCH'}")
    for f in all_failures:
        print(f"  {f}")
    return 1 if all_failures else 0


if __name__ == "__main__":
    np.random.seed(0)
    sys.exit(main())
