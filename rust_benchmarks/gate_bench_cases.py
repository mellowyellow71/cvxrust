"""Runnability gate: run every ASV benchmark cell once, report ok / n/a / ERROR.

Run from rust_benchmarks/cvxpy-benchmarks/:
    ../../.venv/bin/python ../gate_bench_cases.py
Exit code 1 on any failure. Expected n/a pattern: CPP x {concatenate, nd_*, einsum}.
"""
import sys
import time
import traceback

sys.path.insert(0, "benchmark")
import canonicalization_backends as m  # noqa: E402

failures = []


def run_cell(cls, label, *params):
    inst = cls()
    try:
        inst.setup(*params)
    except NotImplementedError as exc:
        print(f"  n/a   {label}: {exc}")
        return
    except Exception:
        print(f"  ERROR {label} (setup)")
        traceback.print_exc()
        failures.append(label)
        return
    timed = getattr(inst, [a for a in dir(inst) if a.startswith("time_")][0])
    try:
        t0 = time.perf_counter()
        timed(*params)
        dt = time.perf_counter() - t0
        print(f"  ok    {label}: {dt*1000:.1f} ms")
    except Exception:
        print(f"  ERROR {label} (timed)")
        traceback.print_exc()
        failures.append(label)


for cls in (m.BackendCompileCanonicalization, m.BackendBuildMatrixCanonicalization):
    print(f"== {cls.__name__} ==")
    for case in m.CASE_NAMES:
        for backend in m.BACKENDS:
            run_cell(cls, f"{case} x {backend}", case, backend)

for cls in (m.DeepExpressionTreeScaling, m.WideExpressionTreeScaling,
            m.WideConstraintScaling):
    print(f"== {cls.__name__} ==")
    for n in cls.params[0]:
        for backend in cls.params[1]:
            run_cell(cls, f"n={n} x {backend}", n, backend)

print(f"\nFAILURES: {len(failures)}")
for f in failures:
    print(f"  {f}")
sys.exit(1 if failures else 0)
