"""Aggregate a multi-agent memory bench run into a digest + comparison table.

Usage:
    python bench/analyze_agents.py bench/results/agents-<timestamp>

The run directory holds one cell per (mode x agent-count):
    <run>/<mode>/agents-<N>/{k6_summary.json,docker_stats.jsonl,...}

For each cell it reads:
  - k6_summary.json    write throughput, write/search latency percentiles,
                       read-your-writes hit-rate + lag, error rates
  - docker_stats.jsonl per-container CPU/memory during the cell window

Writes alongside the cells:
  - RESULTS.md   per-mode matrix + a per-agent-vs-shared comparison table
                 across agent counts, plus per-cell container utilization
  - summary.json machine-readable, one entry per cell

Bottleneck call: the container whose mean CPU% was highest relative to its
configured `cpus` budget (matches bench/docker-compose.recall-bench.yml). The
pgvector recall instance is the one to watch in shared mode.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# --- expected cpu budgets per service (matches docker-compose.recall-bench.yml) ---
# Keyed by the container name docker stats emits (project = rosalinddb-recall-bench).

CPU_BUDGET = {
    "rosalinddb-recall-bench-cp-1": 1.0,
    "rosalinddb-recall-bench-query_dp-1": 1.0,
    "rosalinddb-recall-bench-validator-1": 0.5,
    "rosalinddb-recall-bench-index_builder-1": 0.5,
    "rosalinddb-recall-bench-ephemeral_runner-1": 0.5,
    "rosalinddb-recall-bench-postgres-1": 0.5,
    "rosalinddb-recall-bench-pgvector-1": 1.0,
    "rosalinddb-recall-bench-redis-1": 0.25,
    "rosalinddb-recall-bench-minio-1": 0.5,
    "rosalinddb-recall-bench-k6": 1.0,
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

    def vals(name):
        return m.get(name, {}).get("values", {}) if name in m else {}

    wl = vals("rb_write_latency")
    sl = vals("rb_search_latency")
    w = vals("rb_writes")
    ryw = vals("rb_ryw_hit")
    lag = vals("rb_ryw_lag_ms")
    errs = vals("rb_errors")
    httpf = vals("http_req_failed")

    thresholds = {}
    for tname, tdata in m.items():
        if "thresholds" in tdata:
            for expr, outcome in tdata["thresholds"].items():
                thresholds[f"{tname}: {expr}"] = bool(outcome.get("ok"))

    return {
        "writes": w.get("count"),
        "write_throughput_ops": w.get("rate"),
        "write_p50_ms": wl.get("med"),
        "write_p95_ms": wl.get("p(95)"),
        "write_p99_ms": wl.get("p(99)"),
        "search_p50_ms": sl.get("med"),
        "search_p95_ms": sl.get("p(95)"),
        "search_p99_ms": sl.get("p(99)"),
        "ryw_hit_rate_pct": (ryw.get("rate") or 0) * 100,
        "ryw_lag_p50_ms": lag.get("med"),
        "ryw_lag_p95_ms": lag.get("p(95)"),
        "error_rate_pct": (errs.get("rate") or 0) * 100,
        "http_req_failed_pct": (httpf.get("rate") or 0) * 100,
        "thresholds": thresholds,
    }


def _to_mib(s: str) -> float | None:
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


def parse_docker_stats(stats_path: Path) -> dict[str, dict[str, float]]:
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


def call_bottleneck(stats: dict[str, dict[str, float]]) -> str:
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
    cells: list[dict[str, Any]] = []
    for mode_dir in sorted(run_dir.iterdir()):
        if not mode_dir.is_dir():
            continue
        # mode dirs are 'per-agent' / 'shared'; skip anything without agents-* kids.
        agent_dirs = sorted(
            mode_dir.glob("agents-*"),
            key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0,
        )
        if not agent_dirs:
            continue
        mode = mode_dir.name
        for ad in agent_dirs:
            try:
                agents = int(ad.name.split("-")[1])
            except (IndexError, ValueError):
                continue
            k6 = parse_k6(ad / "k6_summary.json")
            stats = parse_docker_stats(ad / "docker_stats.jsonl")
            cells.append({
                "mode": mode,
                "agents": agents,
                "k6": k6,
                "stats": stats,
                "bottleneck": call_bottleneck(stats),
            })
    return cells


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}"
    return str(v)


def render_md(cells: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Multi-agent memory bench results")
    lines.append("")
    lines.append(
        "Concurrent agents bombarding the recall (read-your-writes) path. "
        "Each agent loops: write a memory batch (sync 200) -> search -> a "
        "read-your-writes sentinel probe. `per-agent` = each agent owns its own "
        "dataset; `shared` = all agents share one dataset and filter by "
        "`agent_id` (exhaustive server-side scan)."
    )
    lines.append("")

    modes = sorted({c["mode"] for c in cells})

    # --- per-mode matrix ----------------------------------------------------
    for mode in modes:
        lines.append(f"## Mode: {mode}")
        lines.append("")
        lines.append(
            "| Agents | write tput (ops/s) | write p50/p95/p99 ms | "
            "search p50/p95/p99 ms | RYW hit % | RYW lag p95 ms | err % | thresholds |"
        )
        lines.append("| --: | --: | --- | --- | --: | --: | --: | --- |")
        for c in [c for c in cells if c["mode"] == mode]:
            k = c["k6"]
            thr_pass = sum(1 for v in (k.get("thresholds") or {}).values() if v)
            thr_total = len(k.get("thresholds") or {})
            wlat = f"{_fmt(k.get('write_p50_ms'))}/{_fmt(k.get('write_p95_ms'))}/{_fmt(k.get('write_p99_ms'))}"
            slat = f"{_fmt(k.get('search_p50_ms'))}/{_fmt(k.get('search_p95_ms'))}/{_fmt(k.get('search_p99_ms'))}"
            lines.append(
                f"| {c['agents']} | {_fmt(k.get('write_throughput_ops'))} | {wlat} | {slat} "
                f"| {_fmt(k.get('ryw_hit_rate_pct'))} | {_fmt(k.get('ryw_lag_p95_ms'))} "
                f"| {_fmt(k.get('error_rate_pct'))} | {thr_pass}/{thr_total} |"
            )
        lines.append("")

    # --- per-agent vs shared comparison across agent counts -----------------
    if "per-agent" in modes and "shared" in modes:
        agent_counts = sorted({c["agents"] for c in cells})
        by = {(c["mode"], c["agents"]): c["k6"] for c in cells}
        lines.append("## per-agent vs shared")
        lines.append("")
        lines.append(
            "Write throughput (ops/s), search p95 (ms), and read-your-writes "
            "hit-rate (%), side by side. The `shared` column is where the "
            "scaling cliff shows up — one dataset, every query exhaustively "
            "scans the whole agent_id-filtered partition."
        )
        lines.append("")
        lines.append(
            "| Agents | tput per-agent | tput shared | search p95 per-agent | "
            "search p95 shared | RYW per-agent % | RYW shared % |"
        )
        lines.append("| --: | --: | --: | --: | --: | --: | --: |")
        for n in agent_counts:
            pa = by.get(("per-agent", n), {})
            sh = by.get(("shared", n), {})
            lines.append(
                f"| {n} "
                f"| {_fmt(pa.get('write_throughput_ops'))} | {_fmt(sh.get('write_throughput_ops'))} "
                f"| {_fmt(pa.get('search_p95_ms'))} | {_fmt(sh.get('search_p95_ms'))} "
                f"| {_fmt(pa.get('ryw_hit_rate_pct'))} | {_fmt(sh.get('ryw_hit_rate_pct'))} |"
            )
        lines.append("")

    # --- per-cell container utilization -------------------------------------
    lines.append("## Per-cell container utilization")
    lines.append("")
    for c in cells:
        lines.append(f"### mode={c['mode']} agents={c['agents']}")
        lines.append(f"**Bottleneck call:** {c['bottleneck']}")
        lines.append("")
        lines.append(
            "| Container | CPU budget | mean CPU% | p95 CPU% | max CPU% | "
            "mean mem MiB | % of budget (mean) |"
        )
        lines.append("| --- | --: | --: | --: | --: | --: | --: |")
        rows = sorted(c["stats"].items(), key=lambda kv: -kv[1]["mean_cpu_pct"])
        for name, s in rows:
            budget = CPU_BUDGET.get(name, 0)
            pct = (s["mean_cpu_pct"] / 100.0) / budget * 100 if budget > 0 else 0
            lines.append(
                f"| {name} | {budget:.2f} | {s['mean_cpu_pct']:.0f} | "
                f"{s['p95_cpu_pct']:.0f} | {s['max_cpu_pct']:.0f} | "
                f"{s['mean_mem_mib']:.0f} | {pct:.0f}% |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2:
        print("usage: analyze_agents.py <results-dir>", file=sys.stderr)
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
