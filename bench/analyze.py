"""Aggregate a matrix run into a digest table + per-cell bottleneck call.

Usage:
    python bench/analyze.py bench/results/<timestamp>

For each cell directory it reads:
  - k6_summary.json    QPS, latency percentiles, error rate, thresholds
  - docker_stats.jsonl per-container CPU/memory during the cell window

Writes alongside the cells:
  - RESULTS.md         matrix table + per-cell container utilization
  - summary.json       machine-readable, one entry per cell

Bottleneck call: the container whose mean CPU% was highest relative to
its configured `cpus` budget. >= 80% is flagged as saturated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# --- expected cpu budgets per service (matches bench/docker-compose.bench.yml) ---

CPU_BUDGET = {
    "rosalinddb-bench-cp-1": 1.0,
    "rosalinddb-bench-query_dp-1": 1.0,
    "rosalinddb-bench-validator-1": 0.5,
    "rosalinddb-bench-index_builder-1": 0.5,
    "rosalinddb-bench-ephemeral_runner-1": 0.5,
    "rosalinddb-bench-postgres-1": 0.5,
    "rosalinddb-bench-redis-1": 0.25,
    "rosalinddb-bench-minio-1": 0.5,
    "rosalinddb-bench-k6": 1.0,
}


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_k6(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {"error": "no k6_summary.json"}
    with summary_path.open() as f:
        data = json.load(f)
    m = data.get("metrics", {})

    def trend(name):
        return m.get(name, {}).get("values", {}) if name in m else {}

    ql = trend("rb_query_latency")
    fql = trend("rb_filtered_query_latency")
    errs = m.get("rb_query_errors", {}).get("values", {})
    cnt = m.get("rb_queries_run", {}).get("values", {})
    httpf = m.get("http_req_failed", {}).get("values", {})

    # Threshold outcomes
    thresholds = {}
    for tname, tdata in m.items():
        if "thresholds" in tdata:
            for expr, outcome in tdata["thresholds"].items():
                thresholds[f"{tname}: {expr}"] = bool(outcome.get("ok"))

    return {
        "queries": cnt.get("count"),
        "qps": cnt.get("rate"),
        "p50_ms": ql.get("med"),
        "p95_ms": ql.get("p(95)"),
        "p99_ms": ql.get("p(99)"),
        "filtered_p95_ms": fql.get("p(95)"),
        "error_rate_pct": (errs.get("rate") or 0) * 100,
        "http_req_failed_pct": (httpf.get("rate") or 0) * 100,
        "thresholds": thresholds,
    }


def parse_docker_stats(stats_path: Path) -> dict[str, dict[str, float]]:
    """Return {container_name: {mean_cpu_pct, p95_cpu_pct, max_cpu_pct, mean_mem_mib}}.

    docker stats emits CPU as a percentage of one CPU (so a container with
    cpus=1.0 saturating yields CPUPerc=100.00 %; a container with cpus=2.0
    saturating BOTH cores yields 200.00 %). We normalize by the budget.
    """
    if not stats_path.exists():
        return {}
    per_container: dict[str, list[dict]] = {}
    with stats_path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("Error response"):
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            name = row.get("Name") or row.get("Container") or "?"
            per_container.setdefault(name, []).append(row)

    out: dict[str, dict[str, float]] = {}
    for name, rows in per_container.items():
        cpus = []
        mems = []
        for r in rows:
            cpu_s = (r.get("CPUPerc") or "0%").rstrip("%")
            try:
                cpus.append(float(cpu_s))
            except ValueError:
                pass
            mem_s = (r.get("MemUsage") or "0MiB / 0MiB").split("/")[0].strip()
            # MemUsage is like "123.4MiB" or "1.234GiB"
            mib = _to_mib(mem_s)
            if mib is not None:
                mems.append(mib)
        if not cpus:
            continue
        cpus_sorted = sorted(cpus)
        p95_idx = max(0, int(len(cpus_sorted) * 0.95) - 1)
        out[name] = {
            "samples": len(cpus),
            "mean_cpu_pct": sum(cpus) / len(cpus),
            "p95_cpu_pct": cpus_sorted[p95_idx],
            "max_cpu_pct": max(cpus),
            "mean_mem_mib": sum(mems) / len(mems) if mems else 0.0,
        }
    return out


def _to_mib(s: str) -> float | None:
    """Parse '123.4MiB' / '1.5GiB' / '500KiB' to MiB."""
    s = s.strip()
    try:
        if s.endswith("GiB"):
            return float(s[:-3]) * 1024
        if s.endswith("MiB"):
            return float(s[:-3])
        if s.endswith("KiB"):
            return float(s[:-3]) / 1024
        return float(s)
    except ValueError:
        return None


def call_bottleneck(stats: dict[str, dict[str, float]]) -> str:
    """Return the container most saturated relative to its CPU budget."""
    worst_name = None
    worst_ratio = 0.0
    for name, s in stats.items():
        budget = CPU_BUDGET.get(name)
        if budget is None or budget <= 0:
            continue
        ratio = (s["mean_cpu_pct"] / 100.0) / budget
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_name = name
    if worst_name is None:
        return "unknown"
    if worst_ratio >= 0.80:
        return f"{worst_name} saturated ({worst_ratio*100:.0f}% of budget)"
    return f"{worst_name} hottest ({worst_ratio*100:.0f}% of budget)"


def collect(run_dir: Path) -> list[dict[str, Any]]:
    cells = []
    for dim_dir in sorted(run_dir.glob("dim-*")):
        try:
            dim = int(dim_dir.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        for vu_dir in sorted(dim_dir.glob("vus-*"), key=lambda p: int(p.name.split("-")[1])):
            try:
                vus = int(vu_dir.name.split("-")[1])
            except (IndexError, ValueError):
                continue
            k6 = parse_k6(vu_dir / "k6_summary.json")
            stats = parse_docker_stats(vu_dir / "docker_stats.jsonl")
            cells.append({
                "dim": dim,
                "vus": vus,
                "k6": k6,
                "stats": stats,
                "bottleneck": call_bottleneck(stats),
            })
    return cells


def render_md(cells: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Bench results")
    lines.append("")
    lines.append("## Matrix")
    lines.append("")
    lines.append("| Dim | VUs | QPS | p50 ms | p95 ms | p99 ms | filtered p95 ms | err % | thresholds |")
    lines.append("| --- | --: | --: | --: | --: | --: | --: | --: | --- |")
    for c in cells:
        k = c["k6"]
        thr_pass = sum(1 for v in (k.get("thresholds") or {}).values() if v)
        thr_total = len(k.get("thresholds") or {})
        thr_summary = f"{thr_pass}/{thr_total}"
        def fmt(v):
            return "-" if v is None else f"{v:.0f}" if isinstance(v, (int, float)) and v >= 1 else f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
        lines.append(
            f"| {c['dim']} | {c['vus']} "
            f"| {fmt(k.get('qps'))} | {fmt(k.get('p50_ms'))} | {fmt(k.get('p95_ms'))} | {fmt(k.get('p99_ms'))} "
            f"| {fmt(k.get('filtered_p95_ms'))} | {fmt(k.get('error_rate_pct'))} | {thr_summary} |"
        )

    lines.append("")
    lines.append("## Per-cell container utilization")
    lines.append("")
    for c in cells:
        lines.append(f"### dim={c['dim']} VUs={c['vus']}")
        lines.append(f"**Bottleneck call:** {c['bottleneck']}")
        lines.append("")
        lines.append("| Container | CPU budget | mean CPU% | p95 CPU% | max CPU% | mean mem MiB | % of budget (mean) |")
        lines.append("| --- | --: | --: | --: | --: | --: | --: |")
        rows = sorted(c["stats"].items(), key=lambda kv: -kv[1]["mean_cpu_pct"])
        for name, s in rows:
            budget = CPU_BUDGET.get(name, 0)
            pct_of_budget = (s["mean_cpu_pct"] / 100.0) / budget * 100 if budget > 0 else 0
            lines.append(
                f"| {name} | {budget:.2f} | {s['mean_cpu_pct']:.0f} | "
                f"{s['p95_cpu_pct']:.0f} | {s['max_cpu_pct']:.0f} | {s['mean_mem_mib']:.0f} | "
                f"{pct_of_budget:.0f}% |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2:
        print("usage: analyze.py <results-dir>", file=sys.stderr)
        sys.exit(2)
    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"no such dir: {run_dir}", file=sys.stderr)
        sys.exit(2)

    cells = collect(run_dir)
    md = render_md(cells)

    (run_dir / "RESULTS.md").write_text(md)
    (run_dir / "summary.json").write_text(json.dumps(cells, indent=2, default=str))

    print(md)
    print(f"\nWrote {run_dir/'RESULTS.md'} and {run_dir/'summary.json'}")


if __name__ == "__main__":
    main()
