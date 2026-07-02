#!/usr/bin/env python
"""Run ONE benchmark class across one or more canon backends.

Usage:
    run_one.py <module.py> <ClassName> <BACKENDS> <warmups> <reps> <timeout_s>

  <module.py>  path to a file in the cvxpy/benchmarks `benchmark/` package
  <BACKENDS>   comma list, e.g. RUST,SCIPY,CPP
  <warmups>    untimed warm-up reps per backend (e.g. 1)
  <reps>       timed reps per backend (e.g. 2); median reported
  <timeout_s>  per-measurement SIGALRM watchdog cap (macOS has no `timeout`)

Why this and not `asv run`:
  * asv builds cvxpy hermetically from a git commit and would NOT pick up the
    locally-built `cvxpy_rust` extension. We want to measure THIS env's backend.
  * get_problem_data caches the solving chain on first call, so a second call
    with a different canon_backend returns stale data. => fresh Problem per rep.
  * Each class runs in its own subprocess (this script) for crash isolation.

Emits one JSON line per (class, backend) to stdout, prefixed with `RESULT `.
"""
import importlib.util
import json
import os
import signal
import statistics
import sys
import time

import cvxpy as cp
from cvxpy.problems.problem import Problem


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def load_class(module_path, class_name):
    # Import the benchmark module as part of the `benchmark` package so its
    # siblings resolve. Add the repo root (parent of `benchmark/`) to sys.path.
    bench_dir = os.path.dirname(os.path.abspath(module_path))
    repo_root = os.path.dirname(bench_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    mod_name = "benchmark." + os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def time_once(Cls, backend, timeout_s):
    """Build a FRESH problem, run its time_compile_problem with canon_backend
    forced to `backend` by transiently wrapping Problem.get_problem_data."""
    inst = Cls()
    inst.setup()

    orig = Problem.get_problem_data

    def wrapped(self, *args, **kwargs):
        kwargs["canon_backend"] = backend
        return orig(self, *args, **kwargs)

    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    Problem.get_problem_data = wrapped
    try:
        t0 = time.perf_counter()
        inst.time_compile_problem()
        dt = time.perf_counter() - t0
    finally:
        Problem.get_problem_data = orig
        signal.setitimer(signal.ITIMER_REAL, 0)
    return dt


def main():
    module_path, class_name, backends_csv, warmups, reps, timeout_s = (
        sys.argv[1], sys.argv[2], sys.argv[3],
        int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]),
    )
    backends = backends_csv.split(",")
    Cls = load_class(module_path, class_name)

    for backend in backends:
        rec = {"class": class_name, "module": os.path.basename(module_path),
               "backend": backend}
        try:
            for _ in range(warmups):
                time_once(Cls, backend, timeout_s)
            samples = [time_once(Cls, backend, timeout_s) for _ in range(reps)]
            rec["median_ms"] = round(statistics.median(samples) * 1000, 3)
            rec["samples_ms"] = [round(s * 1000, 3) for s in samples]
            rec["status"] = "ok"
        except _Timeout:
            rec["status"] = "timeout"
        except Exception as e:  # noqa: BLE001
            rec["status"] = "error"
            rec["error"] = f"{type(e).__name__}: {e}"
        print("RESULT " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
