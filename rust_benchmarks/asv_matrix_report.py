#!/usr/bin/env python
"""
Pivot backend benchmark results into pairwise comparison tables (markdown).

Inputs:
  --jsonl results.jsonl            sweep.sh/run_one.py output (RESULT lines)
  --baseline results_prerebase.jsonl   optional older run for a delta column
  --asv results/<machine>/<f>.json     asv --python=same results file

Ratios are reported as OTHER/RUST (>1 => Rust faster), matching
CVXPY_BENCHMARKS_RESULTS.md.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_jsonl(path: str) -> dict[tuple[str, str], dict]:
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line.startswith("RESULT "):
            continue
        rec = json.loads(line[len("RESULT "):])
        out[(rec["class"], rec["backend"])] = rec
    return out


def load_asv(path: str) -> dict[tuple[str, str, str], float]:
    """Return {(benchmark, case, backend): median_seconds} from an asv result file."""
    raw = json.loads(Path(path).read_text())
    out = {}
    for bench_name, entry in raw.get("results", {}).items():
        # asv 0.5 schema: [values, stats, samples, profile, param_names?...] —
        # entry[0] is the flat list of results, entry[-1]/params in
        # raw["benchmarks"]? params are stored per-benchmark in the companion
        # benchmarks.json; but each result entry also carries its params as
        # entry[1] in some versions. Support the common layout: entry is a
        # dict {"result": [...], "params": [[...], [...]]} or a list.
        if isinstance(entry, dict):
            values = entry.get("result")
            params = entry.get("params")
        else:
            values, params = entry[0], entry[1] if len(entry) > 1 else None
        if values is None or params is None:
            continue
        cases, backends = params[0], params[1]
        idx = 0
        for case in cases:
            for backend in backends:
                if idx < len(values) and values[idx] is not None:
                    out[(bench_name, case.strip("'\""), backend.strip("'\""))] = values[idx]
                idx += 1
    return out


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float("nan")


def pairwise_tables(results: dict, baseline: dict | None, ref: str = "RUST") -> str:
    classes = sorted({cls for cls, _ in results})
    others = sorted({b for _, b in results if b != ref})
    lines = []
    for other in others:
        rows = []
        for cls in classes:
            r, o = results.get((cls, ref)), results.get((cls, other))
            if not r or not o:
                continue
            if r.get("status") != "ok" or o.get("status") != "ok":
                rows.append((cls, r, o, None, None))
                continue
            ratio = o["median_ms"] / r["median_ms"]
            delta = None
            if baseline:
                br, bo = baseline.get((cls, ref)), baseline.get((cls, other))
                if br and bo and br.get("status") == "ok" and bo.get("status") == "ok":
                    delta = ratio / (bo["median_ms"] / br["median_ms"])
            rows.append((cls, r, o, ratio, delta))
        rows.sort(key=lambda t: -(t[3] or 0))

        header = f"\n## RUST vs {other}\n\n"
        cols = f"| Benchmark | Rust (ms) | {other} (ms) | {other}/Rust |"
        sep = "| --- | --- | --- | --- |"
        if baseline:
            cols += " Δ vs pre-rebase |"
            sep += " --- |"
        lines.append(header + cols + "\n" + sep)
        ratios = []
        for cls, r, o, ratio, delta in rows:
            if ratio is None:
                status = r.get("status") if r.get("status") != "ok" else o.get("status")
                row = f"| {cls} | — | — | {status} |"
                if baseline:
                    row += " |"
                lines.append(row)
                continue
            ratios.append(ratio)
            bold = f"**{ratio:.2f}×**" if ratio >= 1.05 else f"{ratio:.2f}×"
            row = f"| {cls} | {r['median_ms']:.0f} | {o['median_ms']:.0f} | {bold} |"
            if baseline:
                row += f" {f'{delta:.2f}×' if delta else '—'} |"
            lines.append(row)
        wins = sum(1 for x in ratios if x > 1)
        lines.append(
            f"\n**geomean {geomean(ratios):.2f}× | {wins}/{len(ratios)} wins "
            f"(ratio >1 ⇒ Rust faster)**"
        )
    return "\n".join(lines)


def asv_tables(asv: dict, ref: str = "RUST") -> str:
    lines = []
    benches = sorted({b for b, _, _ in asv})
    for bench in benches:
        cases = sorted({c for b, c, _ in asv if b == bench})
        backends = sorted({bk for b, _, bk in asv if b == bench})
        others = [b for b in backends if b != ref]
        if ref not in backends:
            continue
        lines.append(f"\n## {bench}\n")
        lines.append("| case | " + " | ".join(f"{b} (ms)" for b in backends)
                     + " | " + " | ".join(f"{o}/RUST" for o in others) + " |")
        lines.append("|" + " --- |" * (1 + len(backends) + len(others)))
        ratio_acc = {o: [] for o in others}
        for case in cases:
            vals = {b: asv.get((bench, case, b)) for b in backends}
            cells = [f"{v * 1000:.2f}" if v else "n/a" for v in (vals[b] for b in backends)]
            ratio_cells = []
            for o in others:
                if vals.get(o) and vals.get(ref):
                    ratio = vals[o] / vals[ref]
                    ratio_acc[o].append(ratio)
                    ratio_cells.append(f"{ratio:.2f}×")
                else:
                    ratio_cells.append("—")
            lines.append(f"| {case} | " + " | ".join(cells) + " | "
                         + " | ".join(ratio_cells) + " |")
        summary = " | ".join(
            f"{o}: geomean {geomean(ratio_acc[o]):.2f}×" for o in others if ratio_acc[o]
        )
        lines.append(f"\n**{summary}**")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl")
    ap.add_argument("--baseline")
    ap.add_argument("--asv")
    args = ap.parse_args()
    if args.jsonl:
        results = load_jsonl(args.jsonl)
        baseline = load_jsonl(args.baseline) if args.baseline else None
        print(pairwise_tables(results, baseline))
    if args.asv:
        print(asv_tables(load_asv(args.asv)))


if __name__ == "__main__":
    main()
