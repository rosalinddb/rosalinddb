"""Aggregate an ssd-cache 3-cell run into RESULTS.md + summary.json.

Modelled on `bench/analyze.py`, with two differences:

  1. The cells are NAMED (tier-off, tier-on-cold, tier-on-warm), not laid
     out as a (dim, vus) grid. Each cell is a single directory under the
     run root containing `k6_summary.json` + `docker_stats.jsonl` + a
     post-cell `query_results.json` written by capture_query_results.py.

  2. **Cell-agreement check** instead of recall@10 against brute force.
     The SSD-cache claim is "the tier does not change query semantics" —
     every cell must return the same matches for the same query. The
     baseline is tier-off; tier-on-cold and tier-on-warm are scored
     against it. Agreement = avg over queries of
     |returned_ids_cell ∩ returned_ids_baseline| / top_k.

     Why not recall@10 against brute force: IVFFlat is an approximate
     index. On a corpus without natural cluster structure (synthetic
     uniform-random vectors) IVF recall against brute force is near-zero
     by design — this is documented behaviour, not a bug in our system.
     The bench's job is to verify the SSD-cache LAYER, not the engine's
     absolute recall on synthetic data. Cell agreement IS the right test.

Floor enforcement: if cell-agreement drops below 0.95 the script exits
with rc=3 AFTER writing RESULTS.md (so partial output is preserved for
debugging). Below 0.95 means "tier-on returns different IDs than tier-off
for the same query" — that's the wiring bug the bench was designed to
catch (routing returning the wrong shard, invalidation racing, etc.).

Usage:
    python bench/analyze_ssd_cache.py bench/results/<timestamp>-ssd-cache
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

# Canonical cell order. Matches the sequence run_ssd_cache.sh produces.
# The baseline (tier-off) MUST be first; the agreement check compares the
# other cells against it.
CELL_ORDER = ["tier-off", "tier-on-cold", "tier-on-warm"]
BASELINE_CELL = "tier-off"

AGREEMENT_FLOOR = 0.95


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_k6(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {"error": f"no k6_summary.json at {summary_path}"}
    with summary_path.open() as f:
        data = json.load(f)
    m = data.get("metrics", {})

    def trend(name):
        return m.get(name, {}).get("values", {}) if name in m else {}

    ql = trend("rb_query_latency")
    errs = m.get("rb_query_errors", {}).get("values", {})
    cnt = m.get("rb_queries_run", {}).get("values", {})
    httpf = m.get("http_req_failed", {}).get("values", {})

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
        "error_rate_pct": (errs.get("rate") or 0) * 100,
        "http_req_failed_pct": (httpf.get("rate") or 0) * 100,
        "thresholds": thresholds,
    }


def parse_query_results(path: Path) -> dict[str, list[str]] | None:
    """Return {qid: [returned_id, ...]} or None if the file is absent.

    Written by `bench/capture_query_results.py` at the END of each cell
    (a quiescent backend, not under k6 load) so the captured IDs reflect
    "what would the tier return for this query if asked once" — exactly
    the semantics the cell-agreement check needs.
    """
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    return data.get("results")


def compute_agreement(
    baseline: dict[str, list[str]],
    cell: dict[str, list[str]],
) -> tuple[float, int, int]:
    """Mean overlap fraction between `cell` and `baseline` over shared qids.

    Returns (mean_agreement, queries_compared, queries_with_zero_overlap).
    Skips qids missing from either side (those are failures already
    accounted for in capture_query_results.py's `failures` field).
    """
    common = set(baseline.keys()) & set(cell.keys())
    if not common:
        return 0.0, 0, 0
    total = 0.0
    zeros = 0
    for qid in common:
        b = baseline[qid]
        c = cell[qid]
        if not b:
            continue
        overlap = len(set(b) & set(c)) / len(b)
        total += overlap
        if overlap == 0.0:
            zeros += 1
    return total / len(common), len(common), zeros


def parse_docker_stats(stats_path: Path) -> dict[str, dict[str, float]]:
    """Return {container_name: {mean_cpu_pct, p95_cpu_pct, max_cpu_pct, mean_mem_mib}}.

    docker stats emits CPU as a percentage of one CPU; we normalize by the
    container's `cpus:` budget in render_md.
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


def collect(run_dir: Path) -> list[dict[str, Any]]:
    """Find the named cell dirs in CELL_ORDER. Skip any that aren't there."""
    cells = []
    for name in CELL_ORDER:
        cell_dir = run_dir / name
        if not cell_dir.is_dir():
            continue
        k6 = parse_k6(cell_dir / "k6_summary.json")
        stats = parse_docker_stats(cell_dir / "docker_stats.jsonl")
        results = parse_query_results(cell_dir / "query_results.json")
        cells.append(
            {
                "cell": name,
                "dir": str(cell_dir.relative_to(run_dir)),
                "k6": k6,
                "stats": stats,
                "query_results": results,
            }
        )
    return cells


def attach_agreement(cells: list[dict[str, Any]]) -> list[str]:
    """Compute cell-agreement vs the baseline cell. Returns violation list.

    The baseline (tier-off) gets agreement=1.0 by definition (it agrees
    with itself). Other cells are scored against it.
    """
    baseline_results = None
    for c in cells:
        if c["cell"] == BASELINE_CELL:
            baseline_results = c["query_results"]
            c["agreement"] = {
                "vs_baseline": 1.0,
                "queries_compared": (
                    len(c["query_results"]) if c["query_results"] else 0
                ),
                "queries_with_zero_overlap": 0,
                "note": "self (baseline)",
            }
            break

    violations: list[str] = []

    if baseline_results is None:
        # No baseline = no agreement check possible.
        for c in cells:
            c.setdefault("agreement", {
                "vs_baseline": None,
                "note": f"no {BASELINE_CELL} cell or no query_results.json in it",
            })
        violations.append(
            f"no {BASELINE_CELL} query_results.json — cell agreement "
            "cannot be computed"
        )
        return violations

    for c in cells:
        if c["cell"] == BASELINE_CELL:
            continue
        if not c["query_results"]:
            c["agreement"] = {
                "vs_baseline": None,
                "note": "no query_results.json captured for this cell",
            }
            violations.append(
                f"{c['cell']}: no query_results.json — agreement undefined"
            )
            continue
        mean, n_cmp, n_zero = compute_agreement(
            baseline_results, c["query_results"]
        )
        c["agreement"] = {
            "vs_baseline": mean,
            "queries_compared": n_cmp,
            "queries_with_zero_overlap": n_zero,
        }
        if mean < AGREEMENT_FLOOR:
            violations.append(
                f"{c['cell']}: agreement vs {BASELINE_CELL} = {mean:.3f} "
                f"(< {AGREEMENT_FLOOR})"
            )
    return violations


def render_md(cells: list[dict[str, Any]], violations: list[str]) -> str:
    lines: list[str] = []
    lines.append("# SSD-cache bench results")
    lines.append("")
    lines.append(
        "Three cells: tier-off (baseline, no SSD tier), tier-on-cold "
        "(SSD tier enabled, fresh DP state), tier-on-warm (SSD tier "
        "warm from prior cell — same DP process)."
    )
    lines.append("")
    lines.append(
        "Pass criteria:"
    )
    lines.append(
        "  1. **Cell agreement** — tier-on-cold and tier-on-warm return "
        "the same top-K IDs as tier-off for the same queries "
        f"(agreement >= {AGREEMENT_FLOOR}). Catches routing-returns-wrong-shard "
        "and invalidation-race bugs."
    )
    lines.append(
        "  2. **No regression** — tier-off and tier-on-cold are within "
        "margin of error on QPS / latency (the cold-tier path still pays "
        "the cold GET; the tier helps subsequent queries)."
    )
    lines.append(
        "  3. **Tier benefit** — tier-on-warm measurably faster than "
        "tier-on-cold IF the corpus is large enough that the cold GET is "
        "expensive. On a small corpus (e.g. the default 50k vectors at "
        "dim=128), the benefit is invisible because the cold GET is "
        "already fast; use a larger corpus to make the warm-tier delta "
        "visible."
    )
    lines.append("")
    lines.append("## Matrix")
    lines.append("")
    lines.append(
        "| Cell | QPS | p50 ms | p95 ms | p99 ms | err % "
        "| agree vs baseline | queries cmp | zero-overlap |"
    )
    lines.append(
        "| --- | --: | --: | --: | --: | --: | --: | --: | --: |"
    )
    for c in cells:
        k = c["k6"]
        a = c.get("agreement") or {}

        def fmt_ms(v):
            return "-" if v is None else f"{v:.0f}"

        def fmt_pct(v):
            return "-" if v is None else f"{v:.2f}"

        def fmt_agr(v):
            if v is None:
                return "-"
            return f"{v:.3f}"

        def fmt_int(v):
            return "-" if v is None else f"{int(v)}"

        lines.append(
            f"| {c['cell']} "
            f"| {fmt_ms(k.get('qps'))} "
            f"| {fmt_ms(k.get('p50_ms'))} "
            f"| {fmt_ms(k.get('p95_ms'))} "
            f"| {fmt_ms(k.get('p99_ms'))} "
            f"| {fmt_pct(k.get('error_rate_pct'))} "
            f"| {fmt_agr(a.get('vs_baseline'))} "
            f"| {fmt_int(a.get('queries_compared'))} "
            f"| {fmt_int(a.get('queries_with_zero_overlap'))} |"
        )

    if violations:
        lines.append("")
        lines.append("## CELL-AGREEMENT VIOLATION")
        lines.append("")
        lines.append(
            f"Agreement floor is {AGREEMENT_FLOOR}. The following cells did not meet it:"
        )
        for v in violations:
            lines.append(f"  - {v}")
        lines.append("")
        lines.append(
            "This is the bench's primary correctness signal: a violation "
            "means the SSD-cache layer returned different bytes than the "
            "tier-off baseline for the same query. Likely causes: routing "
            "returning another tenant's shard, invalidation racing with a "
            "write, the wrong-version shard bytes being served by an "
            "evict-during-fetch race. Re-run with verbose tier logs."
        )

    lines.append("")
    lines.append("## Per-cell container utilization")
    lines.append("")
    for c in cells:
        lines.append(f"### {c['cell']}")
        lines.append("")
        if not c["stats"]:
            lines.append("_(no docker_stats.jsonl captured)_")
            lines.append("")
            continue
        lines.append(
            "| Container | CPU budget | mean CPU% | p95 CPU% | max CPU% "
            "| mean mem MiB | % of budget (mean) |"
        )
        lines.append(
            "| --- | --: | --: | --: | --: | --: | --: |"
        )
        rows = sorted(c["stats"].items(), key=lambda kv: -kv[1]["mean_cpu_pct"])
        for name, s in rows:
            budget = CPU_BUDGET.get(name, 0)
            pct = (s["mean_cpu_pct"] / 100.0) / budget * 100 if budget > 0 else 0
            lines.append(
                f"| {name} | {budget:.2f} | {s['mean_cpu_pct']:.0f} "
                f"| {s['p95_cpu_pct']:.0f} | {s['max_cpu_pct']:.0f} "
                f"| {s['mean_mem_mib']:.0f} | {pct:.0f}% |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2:
        print("usage: analyze_ssd_cache.py <results-dir>", file=sys.stderr)
        sys.exit(2)
    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"no such dir: {run_dir}", file=sys.stderr)
        sys.exit(2)

    cells = collect(run_dir)
    if not cells:
        print(
            f"no cells found under {run_dir} (looked for: {', '.join(CELL_ORDER)})",
            file=sys.stderr,
        )
        sys.exit(2)

    violations = attach_agreement(cells)
    md = render_md(cells, violations)
    (run_dir / "RESULTS.md").write_text(md)

    # Slim the cells before serialising: the raw query_results dict can
    # be tens of thousands of entries (one per query) and dominates the
    # summary.json size. Keep the agreement summary; drop the per-query
    # ID lists.
    slim_cells = []
    for c in cells:
        c_slim = {k: v for k, v in c.items() if k != "query_results"}
        slim_cells.append(c_slim)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {"cells": slim_cells, "agreement_violations": violations},
            indent=2,
            default=str,
        )
    )

    print(md)
    print(f"\nWrote {run_dir/'RESULTS.md'} and {run_dir/'summary.json'}")

    if violations:
        print("\nCELL-AGREEMENT VIOLATION:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
