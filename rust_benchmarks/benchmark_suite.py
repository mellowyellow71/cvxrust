#!/usr/bin/env python
"""
Comprehensive benchmark suite for CVXPY canonicalization backends.

Produces reliable, cross-platform-comparable results by:
- Capturing full environment fingerprint (BLAS, CPU, versions)
- Controlling thread counts to isolate algorithm performance
- Measuring at two isolation layers (end-to-end and build_matrix only)
- Using proper statistical methodology (warmup, GC control, confidence intervals)
- Reporting speedup ratios (stable across machines) not just absolute times
- Supporting scaling analysis across constraint count, variable size, and density

Usage:
    python benchmark_suite.py                          # Default run
    python benchmark_suite.py --quick                  # Fast run, fewer iterations
    python benchmark_suite.py --thorough               # More iterations for precision
    python benchmark_suite.py --single-thread           # Disable all parallelism
    python benchmark_suite.py --layers bm              # build_matrix only
    python benchmark_suite.py --json results.json      # Save structured results
    python benchmark_suite.py --compare a.json b.json  # Compare two machines
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import platform
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.stats import t as t_dist

import cvxpy as cp
from cvxpy.cvxcore.python import canonInterface
from cvxpy.lin_ops.canon_backend import CanonBackend

# ---------------------------------------------------------------------------
# 1. Environment fingerprinting
# ---------------------------------------------------------------------------

def _get_cpu_brand() -> str:
    """Get CPU brand string."""
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return "unknown"


def _get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _detect_blas() -> str:
    """Detect which BLAS library numpy/scipy are using."""
    try:
        cfg = np.show_config(mode="dicts")
        blas_info = cfg.get("Build Dependencies", {}).get("blas", {})
        if isinstance(blas_info, dict):
            name = blas_info.get("name", "unknown")
            version = blas_info.get("version", "")
            return f"{name} {version}".strip()
    except Exception:
        pass
    # Fallback: check linked libraries
    try:
        cfg_str = str(np.__config__.blas_opt_info)
        for blas in ["mkl", "openblas", "accelerate", "blis"]:
            if blas in cfg_str.lower():
                return blas
    except Exception:
        pass
    return "unknown"


def capture_environment() -> dict:
    """Capture full machine environment for reproducibility."""
    env = {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "cvxpy_version": cp.__version__,
        "blas": _detect_blas(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cpu_brand": _get_cpu_brand(),
        "os": platform.system(),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS", "auto"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "auto"),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_hash": _get_git_hash(),
    }
    try:
        import cvxpy_rust
        env["cvxpy_rust_available"] = True
    except ImportError:
        env["cvxpy_rust_available"] = False
    return env


def print_environment(env: dict):
    """Print environment summary."""
    print(f"Machine: {env['os']} {env['platform']} / {env['architecture']} / "
          f"{env['cpu_count']} cores")
    print(f"CPU: {env['cpu_brand']}")
    print(f"BLAS: {env['blas']}")
    print(f"Python {env['python_version']} / NumPy {env['numpy_version']} / "
          f"SciPy {env['scipy_version']} / CVXPY {env['cvxpy_version']}")
    print(f"Rust backend: {'available' if env.get('cvxpy_rust_available') else 'NOT available'}")
    print(f"Threads: RAYON={env['rayon_num_threads']}, OMP={env['omp_num_threads']}")
    print(f"Git: {env['git_hash']}")


# ---------------------------------------------------------------------------
# 2. Measurement engine
# ---------------------------------------------------------------------------

@dataclass
class TimingResult:
    """Statistical timing result for one (problem, backend, layer) combination."""
    backend: str
    problem_name: str
    layer: str  # "end_to_end" or "build_matrix"
    raw_times_ms: list[float] = field(default_factory=list)
    warmup_times_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.raw_times_ms)) if self.raw_times_ms else 0.0

    @property
    def median_ms(self) -> float:
        return float(np.median(self.raw_times_ms)) if self.raw_times_ms else 0.0

    @property
    def std_ms(self) -> float:
        return float(np.std(self.raw_times_ms, ddof=1)) if len(self.raw_times_ms) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        return float(np.min(self.raw_times_ms)) if self.raw_times_ms else 0.0

    @property
    def max_ms(self) -> float:
        return float(np.max(self.raw_times_ms)) if self.raw_times_ms else 0.0

    @property
    def ci_95(self) -> tuple[float, float]:
        """95% confidence interval using t-distribution."""
        n = len(self.raw_times_ms)
        if n < 2:
            return (self.mean_ms, self.mean_ms)
        se = self.std_ms / np.sqrt(n)
        h = se * t_dist.ppf(0.975, n - 1)
        return (self.mean_ms - h, self.mean_ms + h)

    @property
    def cv(self) -> float:
        """Coefficient of variation. Flag if > 0.10."""
        return self.std_ms / self.mean_ms if self.mean_ms > 0 else float("inf")

    def to_dict(self) -> dict:
        return {
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "ci_95": [round(x, 3) for x in self.ci_95],
            "cv": round(self.cv, 4),
            "n_samples": len(self.raw_times_ms),
            "raw_ms": [round(x, 3) for x in self.raw_times_ms],
        }


def determine_iterations(estimated_time_ms: float, mode: str = "default") -> tuple[int, int]:
    """Return (warmup, iterations) based on estimated single-run time."""
    if mode == "quick":
        return (1, 5)
    if mode == "thorough":
        mult = 2
    else:
        mult = 1

    if estimated_time_ms < 10:
        return (5, 30 * mult)
    elif estimated_time_ms < 100:
        return (3, 15 * mult)
    elif estimated_time_ms < 1000:
        return (2, 7 * mult)
    else:
        return (1, 5 * mult)


def time_single(fn: Callable) -> float:
    """Time a single invocation in ms with GC disabled."""
    gc.collect()
    gc.disable()
    try:
        start = time.perf_counter_ns()
        fn()
        elapsed_ns = time.perf_counter_ns() - start
    finally:
        gc.enable()
    return elapsed_ns / 1e6


# ---------------------------------------------------------------------------
# 3. build_matrix isolation via monkey-patching
# ---------------------------------------------------------------------------

def extract_build_matrix_inputs(
    problem_factory: Callable[[], cp.Problem],
    solver=cp.CLARABEL,
) -> list[dict]:
    """
    Run the reduction chain once to capture the arguments that
    canonInterface.get_problem_matrix passes to build_matrix.

    Returns a list of captured call dicts (one per get_problem_matrix invocation).
    Each dict contains: linOps, var_length, id_to_col, param_to_size,
    param_to_col, param_size_plus_one.
    """
    captured: list[dict] = []
    original_fn = canonInterface.get_problem_matrix

    def capturing_fn(linOps, var_length, id_to_col, param_to_size,
                     param_to_col, constr_length, canon_backend=None):
        param_size_plus_one = sum(param_to_size.values())
        captured.append({
            "linOps": linOps,
            "var_length": var_length,
            "id_to_col": dict(id_to_col),
            "param_to_size": dict(param_to_size),
            "param_to_col": dict(param_to_col),
            "param_size_plus_one": param_size_plus_one,
        })
        # Call original with SCIPY to complete the chain
        return original_fn(linOps, var_length, id_to_col, param_to_size,
                           param_to_col, constr_length, canon_backend="SCIPY")

    canonInterface.get_problem_matrix = capturing_fn
    try:
        prob = problem_factory()
        prob.get_problem_data(solver, canon_backend="SCIPY")
    finally:
        canonInterface.get_problem_matrix = original_fn

    return captured


def time_build_matrix(captured_call: dict, backend_name: str) -> float:
    """Time a single build_matrix call for a given backend. Returns ms."""
    c = captured_call
    # Fresh copy of id_to_col because build_matrix mutates it
    backend = CanonBackend.get_backend(
        backend_name,
        dict(c["id_to_col"]),
        dict(c["param_to_size"]),
        dict(c["param_to_col"]),
        c["param_size_plus_one"],
        c["var_length"],
    )
    return time_single(lambda: backend.build_matrix(c["linOps"]))


# ---------------------------------------------------------------------------
# 4. Problem suite
# ---------------------------------------------------------------------------

@dataclass
class ProblemSpec:
    name: str
    category: str
    factory: Callable[[], cp.Problem]
    size_label: str  # "small", "medium", "large"
    dominant_ops: list[str] = field(default_factory=list)


def _seed(seed=42):
    np.random.seed(seed)


def make_problems(sizes: str = "all") -> list[ProblemSpec]:
    """Generate the full problem suite. Filter by sizes if requested."""
    specs: list[ProblemSpec] = []

    # ---- Category A: Arithmetic-Heavy ----

    for n, label in [(50, "small"), (200, "medium"), (500, "large"), (1000, "large")]:
        def factory(n=n):
            _seed()
            A = np.random.randn(2 * n, n)
            b = np.random.randn(2 * n)
            x = cp.Variable(n)
            return cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))
        specs.append(ProblemSpec(
            f"dense_matmul (n={n})", "A: Arithmetic", factory, label,
            ["dense_const", "mul", "sum_entries"],
        ))

    for n, label in [(100, "small"), (500, "medium"), (2000, "large")]:
        def factory(n=n):
            _seed()
            A = sp.random(2 * n, n, density=0.05, format="csc", random_state=42)
            b = np.random.randn(2 * n)
            x = cp.Variable(n)
            return cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))
        specs.append(ProblemSpec(
            f"sparse_matmul (n={n})", "A: Arithmetic", factory, label,
            ["sparse_const", "mul", "sum_entries"],
        ))

    for n, label in [(50, "small"), (200, "medium"), (500, "large")]:
        def factory(n=n):
            _seed()
            Q = np.eye(n)
            c = np.random.randn(n)
            A = np.random.randn(n, n)
            b = np.random.randn(n)
            x = cp.Variable(n)
            return cp.Problem(
                cp.Minimize(0.5 * cp.quad_form(x, Q) + c @ x),
                [A @ x <= b],
            )
        specs.append(ProblemSpec(
            f"dense_qp (n={n})", "A: Arithmetic", factory, label,
            ["mul", "rmul", "quad_form"],
        ))

    # ---- Category B: Constraint-Heavy ----

    for m, label in [(10, "small"), (50, "small"), (100, "medium"),
                     (500, "medium"), (1000, "large"), (5000, "large")]:
        def factory(m=m):
            _seed()
            n = 50
            x = cp.Variable(n)
            constraints = [np.random.randn(n) @ x <= np.random.randn() for _ in range(m)]
            return cp.Problem(cp.Minimize(cp.sum(x)), constraints)
        specs.append(ProblemSpec(
            f"many_constraints (m={m})", "B: Constraints", factory, label,
            ["mul", "constraint_stacking"],
        ))

    for n, label in [(50, "small"), (200, "medium"), (1000, "large")]:
        def factory(n=n):
            _seed()
            x = cp.Variable(n)
            return cp.Problem(cp.Minimize(cp.sum(x)), [x >= -1, x <= 1])
        specs.append(ProblemSpec(
            f"box_constraints (n={n})", "B: Constraints", factory, label,
            ["index", "variable"],
        ))

    # ---- Category C: Structural ----

    for n, label in [(20, "small"), (50, "medium"), (100, "large")]:
        def factory(n=n):
            _seed()
            X = cp.Variable((n, n))
            half = n // 2
            return cp.Problem(cp.Minimize(cp.sum_squares(X[: half, : half])))
        specs.append(ProblemSpec(
            f"matrix_indexing (n={n})", "C: Structural", factory, label,
            ["index", "reshape"],
        ))

    for width, label in [(10, "small"), (50, "medium"), (200, "large")]:
        def factory(width=width):
            _seed()
            x = cp.Variable(10)
            exprs = [np.random.randn(10) @ x for _ in range(width)]
            return cp.Problem(cp.Minimize(cp.sum(cp.hstack(exprs))))
        specs.append(ProblemSpec(
            f"hstack (width={width})", "C: Structural", factory, label,
            ["hstack", "mul"],
        ))

    for n, label in [(50, "small"), (200, "medium"), (500, "large")]:
        def factory(n=n):
            _seed()
            mu = np.random.randn(n)
            F = np.random.randn(n, n)
            Sigma = F.T @ F / n + 0.1 * np.eye(n)  # well-conditioned PSD
            w = cp.Variable(n)
            gamma = 1.0
            return cp.Problem(
                cp.Maximize(mu @ w - gamma * cp.quad_form(w, cp.psd_wrap(Sigma))),
                [cp.sum(w) == 1, w >= 0],
            )
        specs.append(ProblemSpec(
            f"portfolio (n={n})", "C: Structural", factory, label,
            ["mul", "sum", "quad_form"],
        ))

    # ---- Category D: Specialized ----

    for n, label in [(50, "small"), (200, "medium"), (500, "large"), (1000, "large")]:
        def factory(n=n):
            _seed()
            m = 2 * n
            A = np.random.randn(m, n)
            b = np.random.randn(m)
            x = cp.Variable(n)
            lam = 0.1
            return cp.Problem(
                cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + lam * cp.norm(x, 1))
            )
        specs.append(ProblemSpec(
            f"lasso (n={n})", "D: Specialized", factory, label,
            ["sum_squares", "norm1", "mul"],
        ))

    for m, label in [(100, "small"), (500, "medium")]:
        def factory(m=m):
            _seed()
            n = 50
            A = np.random.randn(m, n)
            b = 2.0 * (np.random.randn(m) > 0) - 1.0  # +/- 1 labels
            x = cp.Variable(n)
            slack = cp.Variable(m)
            return cp.Problem(
                cp.Minimize(cp.sum_squares(x) + cp.sum(slack)),
                [cp.multiply(b, A @ x) >= 1 - slack, slack >= 0],
            )
        specs.append(ProblemSpec(
            f"svm (m={m})", "D: Specialized", factory, label,
            ["mul", "mul_elem"],
        ))

    for sig_len, label in [(100, "small"), (500, "medium")]:
        def factory(sig_len=sig_len):
            _seed()
            kernel_len = 20
            c = np.random.randn(kernel_len)
            b = np.random.randn(sig_len + kernel_len - 1)
            x = cp.Variable(sig_len)
            return cp.Problem(cp.Minimize(cp.sum_squares(cp.conv(c, x) - b)))
        specs.append(ProblemSpec(
            f"convolution (len={sig_len})", "D: Specialized", factory, label,
            ["conv", "sum_squares"],
        ))

    # ---- Category E: Expression Depth ----

    for depth, label in [(3, "small"), (5, "small"), (10, "medium"), (20, "large")]:
        def factory(depth=depth):
            _seed()
            n = 20
            x = cp.Variable(n)
            expr = x
            for _ in range(depth):
                A = np.random.randn(n, n) / np.sqrt(n)
                expr = A @ expr
            return cp.Problem(cp.Minimize(cp.sum_squares(expr)))
        specs.append(ProblemSpec(
            f"nested_affine (depth={depth})", "E: Depth", factory, label,
            ["mul_chain"],
        ))

    # Filter by size
    if sizes != "all":
        allowed = set(sizes.split(","))
        specs = [s for s in specs if s.size_label in allowed]

    return specs


# ---------------------------------------------------------------------------
# 5. Scaling analysis
# ---------------------------------------------------------------------------

def run_scaling_analysis(
    backends: list[str], mode: str, seed: int = 42,
) -> dict:
    """Run constraint count, variable size, and density scaling sweeps."""
    results = {}

    # -- Constraint count sweep --
    constraint_counts = [4, 10, 50, 100, 500, 1000, 2000, 5000]
    n_vars = 50
    results["constraint_count"] = _run_sweep(
        "Constraint Count Scaling",
        "n_constraints",
        constraint_counts,
        lambda m: _make_constraint_problem(n_vars, m, seed),
        backends, mode,
    )

    # -- Variable size sweep --
    var_sizes = [10, 50, 100, 500, 1000, 2000, 5000]
    n_constraints = 10
    results["variable_size"] = _run_sweep(
        "Variable Size Scaling",
        "n_vars",
        var_sizes,
        lambda n: _make_varsize_problem(n, n_constraints, seed),
        backends, mode,
    )

    # -- Density sweep --
    densities = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    results["density"] = _run_sweep(
        "Matrix Density Scaling",
        "density",
        densities,
        lambda d: _make_density_problem(500, 200, d, seed),
        backends, mode,
    )

    return results


def _make_constraint_problem(n: int, m: int, seed: int) -> cp.Problem:
    np.random.seed(seed)
    x = cp.Variable(n)
    constraints = [np.random.randn(n) @ x <= np.random.randn() for _ in range(m)]
    return cp.Problem(cp.Minimize(cp.sum(x)), constraints)


def _make_varsize_problem(n: int, m: int, seed: int) -> cp.Problem:
    np.random.seed(seed)
    x = cp.Variable(n)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    return cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x <= b])


def _make_density_problem(n: int, m: int, density: float, seed: int) -> cp.Problem:
    np.random.seed(seed)
    if density >= 1.0:
        A = np.random.randn(m, n)
    else:
        A = sp.random(m, n, density=density, format="csc", random_state=seed)
    c = np.random.randn(n)
    b = np.random.randn(m)
    x = cp.Variable(n)
    return cp.Problem(cp.Minimize(c @ x), [A @ x <= b])


def _run_sweep(
    title: str,
    axis_name: str,
    axis_values: list,
    problem_fn: Callable,
    backends: list[str],
    mode: str,
) -> dict:
    """Run a single scaling sweep at the build_matrix layer."""
    print(f"\n{'=' * 60}")
    print(f"SCALING: {title}")
    print(f"{'=' * 60}")

    header = f"{'Value':<12}"
    for b in backends:
        header += f" {b:>10}"
    if len(backends) >= 2:
        header += f" {'Speedup':>10}"
    print(header)
    print("-" * len(header))

    sweep_data: dict[str, list[float]] = {b: [] for b in backends}
    speedups: list[float] = []

    for val in axis_values:
        factory = lambda v=val: problem_fn(v)
        try:
            captured = extract_build_matrix_inputs(factory)
        except Exception as e:
            print(f"  {val:<12} ERROR: {e}")
            for b in backends:
                sweep_data[b].append(float("nan"))
            speedups.append(float("nan"))
            continue

        if not captured:
            print(f"  {val:<12} NO CAPTURED CALLS")
            continue

        # Use the largest captured call (usually constraints)
        call = max(captured, key=lambda c: len(c["linOps"]))

        row = f"  {str(val):<10}"
        times: dict[str, float] = {}
        n_warmup, n_iter = determine_iterations(5.0, mode)  # conservative estimate

        for backend in backends:
            try:
                # Warmup
                for _ in range(n_warmup):
                    time_build_matrix(call, backend)
                # Timed
                t_list = [time_build_matrix(call, backend) for _ in range(n_iter)]
                mean_t = float(np.mean(t_list))
                times[backend] = mean_t
                sweep_data[backend].append(mean_t)
                row += f" {mean_t:>8.2f}ms"
            except Exception as e:
                row += f" {'ERR':>10}"
                sweep_data[backend].append(float("nan"))

        if len(backends) >= 2 and all(b in times for b in backends[:2]):
            spd = times[backends[1]] / times[backends[0]] if times[backends[0]] > 0 else 0
            speedups.append(spd)
            row += f" {spd:>8.2f}x"

        print(row)

    return {
        "title": title,
        "axis_name": axis_name,
        "axis_values": [str(v) for v in axis_values],
        "build_matrix_ms": {b: vals for b, vals in sweep_data.items()},
        "speedups": speedups,
    }


# ---------------------------------------------------------------------------
# 6. Main benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    spec: ProblemSpec
    end_to_end: dict[str, TimingResult] = field(default_factory=dict)
    build_matrix: dict[str, TimingResult] = field(default_factory=dict)


def run_benchmarks(
    backends: list[str],
    problems: list[ProblemSpec],
    layers: str,
    mode: str,
) -> list[BenchmarkResult]:
    """Run the benchmark suite."""
    results: list[BenchmarkResult] = []

    current_category = ""
    for spec in problems:
        if spec.category != current_category:
            current_category = spec.category
            print(f"\n--- {current_category} ---")

        result = BenchmarkResult(spec=spec)

        # Layer A: End-to-End
        if layers in ("all", "e2e"):
            result.end_to_end = _measure_end_to_end(spec, backends, mode)

        # Layer B: build_matrix only
        if layers in ("all", "bm"):
            result.build_matrix = _measure_build_matrix(spec, backends, mode)

        # Print results
        _print_problem_result(spec.name, result, backends, layers)
        results.append(result)

    return results


def _measure_end_to_end(
    spec: ProblemSpec, backends: list[str], mode: str,
) -> dict[str, TimingResult]:
    """Measure get_problem_data end-to-end."""
    results = {}

    # Estimate time with a single run
    prob = spec.factory()
    est_ms = time_single(lambda: prob.get_problem_data(cp.CLARABEL, canon_backend=backends[0]))
    n_warmup, n_iter = determine_iterations(est_ms, mode)

    for backend in backends:
        tr = TimingResult(backend, spec.name, "end_to_end")
        try:
            # Warmup
            for _ in range(n_warmup):
                prob = spec.factory()
                time_single(lambda: prob.get_problem_data(cp.CLARABEL, canon_backend=backend))
                tr.warmup_times_ms.append(0)  # not tracking warmup times

            # Timed iterations
            for _ in range(n_iter):
                prob = spec.factory()
                t = time_single(lambda: prob.get_problem_data(cp.CLARABEL, canon_backend=backend))
                tr.raw_times_ms.append(t)
        except Exception as e:
            print(f"    ERROR ({backend} e2e): {e}")
        results[backend] = tr

    return results


def _measure_build_matrix(
    spec: ProblemSpec, backends: list[str], mode: str,
) -> dict[str, TimingResult]:
    """Measure just the build_matrix call."""
    results = {}

    # Extract inputs once
    try:
        captured = extract_build_matrix_inputs(spec.factory)
    except Exception as e:
        print(f"    ERROR extracting build_matrix inputs: {e}")
        return results

    if not captured:
        return results

    # Use the largest captured call
    call = max(captured, key=lambda c: len(c["linOps"]))

    # Estimate time
    est_ms = time_build_matrix(call, backends[0])
    n_warmup, n_iter = determine_iterations(est_ms, mode)

    for backend in backends:
        tr = TimingResult(backend, spec.name, "build_matrix")
        try:
            for _ in range(n_warmup):
                time_build_matrix(call, backend)
            for _ in range(n_iter):
                t = time_build_matrix(call, backend)
                tr.raw_times_ms.append(t)
        except Exception as e:
            print(f"    ERROR ({backend} bm): {e}")
        results[backend] = tr

    return results


def _print_problem_result(
    name: str, result: BenchmarkResult, backends: list[str], layers: str,
):
    """Print a single problem's result inline."""
    parts = [f"  {name:<35}"]

    primary = result.build_matrix if layers in ("all", "bm") else result.end_to_end
    if not primary:
        primary = result.end_to_end

    for backend in backends:
        if backend in primary:
            tr = primary[backend]
            flag = "!" if tr.cv > 0.10 else " "
            parts.append(f"{tr.mean_ms:>8.2f}ms{flag}")
        else:
            parts.append(f"{'N/A':>10}")

    # Speedup vs first backend
    if len(backends) >= 2:
        first = backends[0]
        for other in backends[1:]:
            if first in primary and other in primary:
                if primary[first].mean_ms > 0:
                    spd = primary[other].mean_ms / primary[first].mean_ms
                    parts.append(f"{spd:>7.2f}x")
                else:
                    parts.append(f"{'N/A':>8}")

    print(" ".join(parts))


# ---------------------------------------------------------------------------
# 7. Summary and output
# ---------------------------------------------------------------------------

def print_summary(
    results: list[BenchmarkResult], backends: list[str], layers: str,
):
    """Print summary statistics."""
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    for layer_name in (["build_matrix", "end_to_end"] if layers == "all"
                       else ["build_matrix" if layers == "bm" else "end_to_end"]):
        speedups_by_pair: dict[str, list[float]] = {}

        for r in results:
            data = r.build_matrix if layer_name == "build_matrix" else r.end_to_end
            if not data:
                continue

            first = backends[0]
            if first not in data:
                continue

            for other in backends[1:]:
                if other not in data:
                    continue
                pair = f"{first}_vs_{other}"
                if data[first].mean_ms > 0 and data[other].mean_ms > 0:
                    spd = data[other].mean_ms / data[first].mean_ms
                    speedups_by_pair.setdefault(pair, []).append(spd)

        if not speedups_by_pair:
            continue

        print(f"\nLayer: {layer_name}")
        for pair, spds in speedups_by_pair.items():
            arr = np.array(spds)
            geo_mean = float(np.exp(np.mean(np.log(arr))))
            wins = sum(1 for s in spds if s > 1.0)
            first, other = pair.split("_vs_")
            print(f"  {other}/{first}: geomean {geo_mean:.2f}x, "
                  f"arith mean {np.mean(arr):.2f}x, "
                  f"range [{np.min(arr):.2f}x, {np.max(arr):.2f}x], "
                  f"{first} wins {wins}/{len(spds)}")

        # Flag high-variance results
        high_cv = []
        for r in results:
            data = r.build_matrix if layer_name == "build_matrix" else r.end_to_end
            if not data:
                continue
            for b, tr in data.items():
                if tr.cv > 0.10:
                    high_cv.append(f"{r.spec.name}/{b} (CV={tr.cv:.0%})")
        if high_cv:
            print(f"  High variance (CV>10%): {', '.join(high_cv[:5])}")
            if len(high_cv) > 5:
                print(f"    ... and {len(high_cv) - 5} more")


def write_json(
    filepath: str,
    env: dict,
    results: list[BenchmarkResult],
    scaling: dict | None,
    backends: list[str],
    config: dict,
):
    """Write structured JSON output."""
    out: dict[str, Any] = {
        "metadata": {
            "suite_version": "1.0.0",
            "environment": env,
            "config": config,
        },
        "results": [],
        "summary": {},
    }

    all_speedups: list[float] = []

    for r in results:
        entry: dict[str, Any] = {
            "problem": r.spec.name,
            "category": r.spec.category,
            "size_label": r.spec.size_label,
            "dominant_ops": r.spec.dominant_ops,
            "measurements": {},
        }
        for layer_name, data in [("end_to_end", r.end_to_end),
                                  ("build_matrix", r.build_matrix)]:
            if data:
                entry["measurements"][layer_name] = {
                    b: tr.to_dict() for b, tr in data.items()
                }
                # Compute speedups
                if len(backends) >= 2 and backends[0] in data and backends[1] in data:
                    first_mean = data[backends[0]].mean_ms
                    other_mean = data[backends[1]].mean_ms
                    if first_mean > 0:
                        spd = other_mean / first_mean
                        entry.setdefault("speedups", {})[layer_name] = {
                            f"{backends[0]}_vs_{backends[1]}": round(spd, 3)
                        }
                        if layer_name == "build_matrix":
                            all_speedups.append(spd)

        out["results"].append(entry)

    if all_speedups:
        arr = np.array(all_speedups)
        out["summary"] = {
            "total_problems": len(all_speedups),
            "geometric_mean_speedup": round(float(np.exp(np.mean(np.log(arr)))), 3),
            "arithmetic_mean_speedup": round(float(np.mean(arr)), 3),
            "min_speedup": round(float(np.min(arr)), 3),
            "max_speedup": round(float(np.max(arr)), 3),
            f"{backends[0]}_wins": int(sum(1 for s in all_speedups if s > 1.0)),
        }

    if scaling:
        out["scaling_analysis"] = scaling

    with open(filepath, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults written to {filepath}")


# ---------------------------------------------------------------------------
# 8. Compare mode
# ---------------------------------------------------------------------------

def compare_results(file_a: str, file_b: str):
    """Compare benchmark results from two different machines."""
    with open(file_a) as f:
        a = json.load(f)
    with open(file_b) as f:
        b = json.load(f)

    env_a = a["metadata"]["environment"]
    env_b = b["metadata"]["environment"]

    print(f"{'=' * 70}")
    print("CROSS-MACHINE COMPARISON")
    print(f"{'=' * 70}")
    print(f"Machine A: {env_a.get('cpu_brand', '?')} / {env_a.get('os', '?')} / BLAS: {env_a.get('blas', '?')}")
    print(f"Machine B: {env_b.get('cpu_brand', '?')} / {env_b.get('os', '?')} / BLAS: {env_b.get('blas', '?')}")

    # Index results by problem name
    a_results = {r["problem"]: r for r in a.get("results", [])}
    b_results = {r["problem"]: r for r in b.get("results", [])}

    common = sorted(set(a_results.keys()) & set(b_results.keys()))
    if not common:
        print("\nNo common problems found!")
        return

    print(f"\n{'Problem':<35} {'A ratio':>10} {'B ratio':>10} {'Agree?':>8}")
    print("-" * 70)

    for name in common:
        ar = a_results[name]
        br = b_results[name]

        a_spd = _get_speedup(ar)
        b_spd = _get_speedup(br)

        if a_spd is not None and b_spd is not None:
            # "Agree" if both say the same backend is faster
            agree = "yes" if (a_spd > 1) == (b_spd > 1) else "DIFFER"
            print(f"  {name:<33} {a_spd:>8.2f}x {b_spd:>8.2f}x {agree:>8}")
        else:
            print(f"  {name:<33} {'N/A':>10} {'N/A':>10}")

    # Summary
    a_summary = a.get("summary", {})
    b_summary = b.get("summary", {})
    print(f"\nGeometric mean speedup:")
    print(f"  Machine A: {a_summary.get('geometric_mean_speedup', 'N/A')}")
    print(f"  Machine B: {b_summary.get('geometric_mean_speedup', 'N/A')}")


def _get_speedup(result: dict) -> float | None:
    """Extract the build_matrix speedup from a result entry."""
    spds = result.get("speedups", {}).get("build_matrix", {})
    if spds:
        return list(spds.values())[0]
    spds = result.get("speedups", {}).get("end_to_end", {})
    if spds:
        return list(spds.values())[0]
    return None


# ---------------------------------------------------------------------------
# 9. Cold-start mode
# ---------------------------------------------------------------------------

def run_cold_start(backends: list[str], seed: int = 42, samples: int = 10):
    """Run a subset of problems in fresh subprocesses."""
    print(f"\n{'=' * 60}")
    print("COLD START BENCHMARK (subprocess isolation)")
    print(f"{'=' * 60}")
    print(f"Samples per measurement: {samples}")

    problems = {
        "lasso (n=200)": f"""
np.random.seed({seed})
A = np.random.randn(400, 200)
b = np.random.randn(400)
x = cp.Variable(200)
prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1)))
""",
        "dense_qp (n=200)": f"""
np.random.seed({seed})
Q = np.eye(200)
c = np.random.randn(200)
A = np.random.randn(200, 200)
b = np.random.randn(200)
x = cp.Variable(200)
prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, Q) + c @ x), [A @ x <= b])
""",
        "many_constraints (m=500)": f"""
np.random.seed({seed})
x = cp.Variable(50)
constraints = [np.random.randn(50) @ x <= np.random.randn() for _ in range(500)]
prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)
""",
        "sparse_matmul (n=500)": f"""
import scipy.sparse as sps
np.random.seed({seed})
A = sps.random(1000, 500, density=0.05, format='csc', random_state={seed})
b = np.random.randn(1000)
x = cp.Variable(500)
prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))
""",
    }

    header = f"  {'Problem':<30}"
    for b in backends:
        header += f" {b:>10}"
    if len(backends) >= 2:
        header += f" {'Speedup':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, code in problems.items():
        row = f"  {name:<30}"
        times_by_backend: dict[str, list[float]] = {}

        for backend in backends:
            try:
                t = _cold_start_measure(code, backend, samples)
                times_by_backend[backend] = t
                row += f" {np.mean(t):>7.1f}ms"
            except Exception as e:
                row += f" {'ERR':>10}"

        if len(backends) >= 2:
            b0, b1 = backends[0], backends[1]
            if b0 in times_by_backend and b1 in times_by_backend:
                spd = np.mean(times_by_backend[b1]) / np.mean(times_by_backend[b0])
                row += f" {spd:>8.2f}x"

        print(row)


def _cold_start_measure(problem_code: str, backend: str, samples: int) -> list[float]:
    """Run benchmark in fresh Python processes."""
    code = textwrap.dedent(f"""
import time, gc, numpy as np, cvxpy as cp
{textwrap.dedent(problem_code).strip()}
gc.collect()
start = time.perf_counter()
prob.get_problem_data(cp.CLARABEL, canon_backend='{backend}')
print((time.perf_counter() - start) * 1000)
""").strip()

    times = []
    for _ in range(samples):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Subprocess failed: {result.stderr[:200]}")
        times.append(float(result.stdout.strip()))
    return times


# ---------------------------------------------------------------------------
# 10. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CVXPY Canonicalization Backend Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Subcommands")

    # -- run subcommand (default) --
    run_parser = sub.add_parser("run", help="Run benchmarks")
    _add_run_args(run_parser)

    # -- compare subcommand --
    cmp_parser = sub.add_parser("compare", help="Compare two JSON result files")
    cmp_parser.add_argument("file_a", help="First JSON results file")
    cmp_parser.add_argument("file_b", help="Second JSON results file")

    # Also support running without subcommand (default = run)
    _add_run_args(parser)

    args = parser.parse_args()

    if args.command == "compare":
        compare_results(args.file_a, args.file_b)
        return

    # Default: run benchmarks
    _run_main(args)


def _add_run_args(parser):
    parser.add_argument("--backends", nargs="+", default=["RUST", "SCIPY"],
                        help="Backends to benchmark (default: RUST SCIPY)")
    parser.add_argument("--quick", action="store_const", dest="mode", const="quick",
                        help="Fewer iterations for faster results")
    parser.add_argument("--thorough", action="store_const", dest="mode", const="thorough",
                        help="More iterations for precision")
    parser.add_argument("--single-thread", action="store_true",
                        help="Set all thread counts to 1")
    parser.add_argument("--sizes", default="all",
                        help="Filter: small,medium,large,all (default: all)")
    parser.add_argument("--layers", choices=["all", "e2e", "bm"], default="bm",
                        help="Isolation layers to run (default: bm)")
    parser.add_argument("--scaling", action="store_true", default=True,
                        help="Include scaling analysis (default)")
    parser.add_argument("--no-scaling", action="store_false", dest="scaling")
    parser.add_argument("--json", metavar="FILE", help="Write JSON results to file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--cold-start", action="store_true",
                        help="Also run cold-start (subprocess) benchmarks")
    parser.set_defaults(mode="default")


def _run_main(args):
    # Thread control (must happen before numpy/scipy operations)
    if args.single_thread:
        os.environ["RAYON_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
    else:
        # Disable BLAS threading by default so we measure algorithm, not threads
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    # Detect available backends
    available = []
    for b in args.backends:
        if b == "RUST":
            try:
                import cvxpy_rust  # noqa: F401
                available.append(b)
            except ImportError:
                print(f"WARNING: {b} backend not available, skipping")
        elif b == "CPP":
            try:
                from cvxpy.cvxcore.python.cppbackend import build_matrix  # noqa: F401
                available.append(b)
            except ImportError:
                print(f"WARNING: {b} backend not available, skipping")
        else:
            available.append(b)

    if len(available) < 1:
        print("ERROR: No backends available!")
        sys.exit(1)

    backends = available

    # Header
    env = capture_environment()
    print(f"{'=' * 60}")
    print("CVXPY Canonicalization Benchmark Suite v1.0.0")
    print(f"{'=' * 60}")
    print_environment(env)
    layer_desc = {"all": "end-to-end + build_matrix", "e2e": "end-to-end", "bm": "build_matrix"}
    print(f"Mode: {args.mode} | Layer: {layer_desc[args.layers]} | "
          f"Backends: {', '.join(backends)} | Seed: {args.seed}")
    if args.single_thread:
        print("Threading: ALL DISABLED (single-thread mode)")
    print(f"{'=' * 60}")

    # Generate problems
    problems = make_problems(args.sizes)
    print(f"\nRunning {len(problems)} problems across {len(backends)} backends...")

    # Print header
    header = f"  {'Problem':<35}"
    for b in backends:
        header += f" {b:>10}"
    if len(backends) >= 2:
        header += f" {backends[1]+'/'+backends[0]:>12}"
    print(header)

    # Run benchmarks
    results = run_benchmarks(backends, problems, args.layers, args.mode)

    # Summary
    print_summary(results, backends, args.layers)

    # Scaling analysis
    scaling = None
    if args.scaling:
        scaling = run_scaling_analysis(backends, args.mode, args.seed)

    # Cold start
    if args.cold_start:
        run_cold_start(backends, args.seed)

    # JSON output
    if args.json:
        config = {
            "mode": args.mode,
            "layers": args.layers,
            "backends": backends,
            "seed": args.seed,
            "single_thread": args.single_thread,
            "sizes": args.sizes,
        }
        write_json(args.json, env, results, scaling, backends, config)


if __name__ == "__main__":
    main()
