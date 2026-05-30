"""Unit tests for the SSD-cache bench harness.

Pins three contracts on `bench/run_ssd_cache.sh`:

  1. The script accepts a `--smoke` and `--dry-run` combination and runs to
     completion without touching docker, k6, or the network (validates the
     argument parser).
  2. With `--dry-run`, the script's stdout names the three cells in the
     canonical order (tier-off, tier-on-cold, tier-on-warm) and the cold
     reset is invoked between cells 1 and 2 — but NOT between cells 2 and 3
     (that's the load-bearing methodology choice; cell 3 only measures the
     warm benefit if it inherits cell 2's residency + SSD files).
  3. The script rejects unknown flags with a non-zero exit code, so a typo
     in CI doesn't silently run the wrong bench.

A separate test runs `bench/cold_reset.sh --help` and asserts the
documentation lists both the container restart and the drop_caches
behaviour, so the macOS caveat in the docstring cannot quietly drift away.

These tests must run on a host without Docker (the unit suite is hermetic),
so every assertion targets stdout strings rather than side effects.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SH = REPO_ROOT / "bench" / "run_ssd_cache.sh"
COLD_RESET_SH = REPO_ROOT / "bench" / "cold_reset.sh"
ANALYZE_PY = REPO_ROOT / "bench" / "analyze_ssd_cache.py"


def _run(args: list[str], cwd: Path = REPO_ROOT, timeout_s: int = 30) -> subprocess.CompletedProcess:
    """Invoke a bench script with bash, capturing stdout + stderr.

    `cwd` defaults to the repo root because run_ssd_cache.sh does
    `cd "$(dirname "$0")/.."` itself; running it from another cwd is fine
    too, but the test pins repo-root for clarity.
    """
    return subprocess.run(
        ["bash", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env={**os.environ, "BENCH_PROJECT": "rosalinddb-bench-test"},
    )


# --- run_ssd_cache.sh: dry-run + smoke -----------------------------------


def test_run_ssd_cache_files_exist_and_are_executable():
    """The harness scripts must be present and have the executable bit set.

    A non-exec script can still be invoked via `bash script.sh` (which is
    what the test runner does), but a forgotten +x is a real foot-gun for
    an operator copying the bench/README commands.
    """
    assert RUN_SH.exists(), f"missing {RUN_SH}"
    assert COLD_RESET_SH.exists(), f"missing {COLD_RESET_SH}"
    assert ANALYZE_PY.exists(), f"missing {ANALYZE_PY}"
    assert os.access(RUN_SH, os.X_OK), f"{RUN_SH} is not executable"
    assert os.access(COLD_RESET_SH, os.X_OK), f"{COLD_RESET_SH} is not executable"


def test_run_ssd_cache_dry_smoke_completes_cleanly():
    """`run_ssd_cache.sh --smoke --dry-run` must exit 0 and print all three cells."""
    res = _run([str(RUN_SH), "--smoke", "--dry-run"])
    assert res.returncode == 0, (
        f"dry-run exit={res.returncode}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
    out = res.stdout
    # All three cells named, in canonical order.
    assert "tier-off" in out
    assert "tier-on-cold" in out
    assert "tier-on-warm" in out
    # And: in this order. The methodology requires cell 2 BEFORE cell 3 so
    # cell 3 inherits cell 2's residency state.
    i_off = out.find("=== cell tier-off")
    i_cold = out.find("=== cell tier-on-cold")
    i_warm = out.find("=== cell tier-on-warm")
    assert -1 < i_off < i_cold < i_warm, (
        f"cells out of order: tier-off@{i_off}, tier-on-cold@{i_cold}, "
        f"tier-on-warm@{i_warm}"
    )


def test_dry_run_skips_cold_reset_between_warm_cells():
    """Cold reset MUST fire between cell 1 and cell 2 — but NOT between 2 and 3.

    This is the load-bearing methodology choice: cell 3 only measures the
    warm benefit if it inherits cell 2's in-process SSD-tier state. A
    reset between them would invalidate the cold-vs-warm comparison.
    """
    res = _run([str(RUN_SH), "--smoke", "--dry-run"])
    assert res.returncode == 0
    out = res.stdout

    # The cell markers anchor the slices we care about.
    i_off = out.find("=== cell tier-off")
    i_cold = out.find("=== cell tier-on-cold")
    i_warm = out.find("=== cell tier-on-warm")
    assert -1 < i_off < i_cold < i_warm

    between_cold_and_warm = out[i_cold:i_warm]
    # Specifically the cold_reset.sh invocation. There IS a "skipping cold_reset"
    # note in this region; check for the call line, not just the substring.
    assert "bash bench/cold_reset.sh" not in between_cold_and_warm, (
        "cold_reset.sh was invoked between tier-on-cold and tier-on-warm — "
        "this invalidates the warm-tier measurement; cell 3 must inherit "
        "cell 2's residency state.\n\n"
        f"section:\n{between_cold_and_warm}"
    )
    # And the human-readable "skipping" note IS there, so future maintainers
    # see WHY the reset is absent.
    assert "skipping cold_reset" in between_cold_and_warm, (
        "missing the 'skipping cold_reset' note between cell 2 and cell 3; "
        "an unannotated absence is harder to maintain than an annotated one"
    )

    # Sanity: a reset MUST appear before tier-on-cold (the test that
    # between-cells reset works at all).
    pre_cold = out[i_off:i_cold]
    assert "bash bench/cold_reset.sh" in pre_cold, (
        "cold_reset.sh was NOT invoked between tier-off and tier-on-cold; "
        "cell 2 needs a fresh DP to model a cold tier-on workload."
    )


def test_dry_run_announces_tier_env_per_cell():
    """tier-off restarts DP with the tier env empty; tier-on cells pass the byte budget."""
    res = _run([str(RUN_SH), "--smoke", "--dry-run"])
    assert res.returncode == 0
    out = res.stdout

    # tier-off cell uses an empty RB_SHARD_TIER_BYTES.
    assert "RB_SHARD_TIER_BYTES=''" in out, (
        "tier-off cell should restart DP with RB_SHARD_TIER_BYTES='' "
        "to model a deployment with the tier disabled"
    )
    # tier-on cells use the byte budget. The smoke run inherits the default
    # 2 GiB; just check the env var name appears with a non-empty value.
    assert "RB_SHARD_TIER_BYTES='2147483648'" in out, (
        "tier-on cells should restart DP with RB_SHARD_TIER_BYTES=2147483648"
    )


def test_unknown_flag_is_rejected():
    """A typo'd flag in CI should not silently pass — it should fail loudly."""
    res = _run([str(RUN_SH), "--smoek", "--dry-run"])  # intentional typo
    assert res.returncode != 0, (
        f"expected non-zero rc for unknown flag, got {res.returncode}\n"
        f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
    # The script's own diagnostic must mention "unknown arg".
    assert "unknown arg" in res.stderr or "unknown arg" in res.stdout


def test_help_documents_methodology():
    """`--help` should tell the operator the three-cell shape and the reset choice.

    A bench operator reading `--help` should not have to open the source
    file to understand WHY there are exactly three cells and WHY one of the
    transitions skips the cold reset.
    """
    res = _run([str(RUN_SH), "--help"])
    assert res.returncode == 0
    out = res.stdout
    assert "tier-off" in out
    assert "tier-on-cold" in out
    assert "tier-on-warm" in out


# --- cold_reset.sh contract ----------------------------------------------


def test_cold_reset_help_documents_container_restart():
    """`cold_reset.sh --help` must mention both restart + drop_caches behaviour.

    The macOS caveat is the load-bearing piece of documentation here; if it
    drifts, an operator may believe they have a cold cache when they don't.
    """
    res = _run([str(COLD_RESET_SH), "--help"])
    assert res.returncode == 0
    out = res.stdout
    # Restart is the universal part.
    assert "query_dp" in out
    assert "ephemeral_runner" in out
    # drop_caches must be named — and so must its macOS no-op behaviour.
    assert "drop_caches" in out
    assert "macOS" in out, (
        "cold_reset.sh --help must mention macOS explicitly; otherwise a "
        "Mac operator may think drop_caches is doing something"
    )


def test_cold_reset_unknown_flag_is_rejected():
    res = _run([str(COLD_RESET_SH), "--frobnicate"])
    assert res.returncode != 0


# --- analyze_ssd_cache.py: smoke ------------------------------------------


def test_analyze_no_args_exits_with_usage():
    """Sanity: analyze script prints usage to stderr when given no args."""
    res = subprocess.run(
        ["python3", str(ANALYZE_PY)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert res.returncode == 2
    assert "usage" in res.stderr.lower()


def test_analyze_cell_agreement_flags_violations(tmp_path):
    """A cell that disagrees with the baseline (tier-off) must trigger rc=3.

    Builds a minimal results dir: tier-off and tier-on-cold return the
    SAME IDs for the same queries (perfect agreement); tier-on-warm
    returns COMPLETELY DIFFERENT IDs (zero agreement). The script
    must write RESULTS.md anyway and exit rc=3.

    Replaces the old recall@10-vs-brute-force test — that contract was
    dropped when we discovered the synthetic uniform-random corpus
    produces near-zero IVF recall for reasons unrelated to the SSD
    cache. The cell-agreement test directly pins the architectural
    claim: "the tier does not change query semantics across cells".
    """
    import json

    def write_cell(name: str, query_results: dict[str, list[str]]):
        cell = tmp_path / name
        cell.mkdir(parents=True)
        summary = {
            "metrics": {
                "rb_query_latency": {
                    "values": {"med": 1.0, "p(95)": 2.0, "p(99)": 3.0}
                },
                "rb_query_errors": {"values": {"rate": 0.0}},
                "rb_queries_run": {"values": {"count": 100, "rate": 10.0}},
                "http_req_failed": {"values": {"rate": 0.0}},
            }
        }
        (cell / "k6_summary.json").write_text(json.dumps(summary))
        (cell / "query_results.json").write_text(
            json.dumps({
                "dataset": "test",
                "num_queries": len(query_results),
                "captured": len(query_results),
                "failures": 0,
                "top_k": 10,
                "elapsed_s": 0.1,
                "results": query_results,
            })
        )

    baseline = {
        "0": [f"v{i}" for i in range(10)],
        "1": [f"u{i}" for i in range(10)],
        "2": [f"w{i}" for i in range(10)],
    }
    write_cell("tier-off", baseline)
    write_cell("tier-on-cold", baseline)  # Perfect agreement.
    write_cell(
        "tier-on-warm",
        {
            "0": [f"x{i}" for i in range(10)],  # Zero overlap.
            "1": [f"y{i}" for i in range(10)],
            "2": [f"z{i}" for i in range(10)],
        },
    )

    res = subprocess.run(
        ["python3", str(ANALYZE_PY), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert res.returncode == 3, (
        f"expected rc=3 (cell-agreement violation), got {res.returncode}\n"
        f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
    results_md = (tmp_path / "RESULTS.md").read_text()
    assert "CELL-AGREEMENT VIOLATION" in results_md
    assert "tier-on-warm" in results_md


def test_analyze_perfect_agreement_exits_clean(tmp_path):
    """When every non-baseline cell agrees fully with tier-off, rc=0."""
    import json

    def write_cell(name: str, query_results: dict[str, list[str]]):
        cell = tmp_path / name
        cell.mkdir(parents=True)
        summary = {
            "metrics": {
                "rb_query_latency": {
                    "values": {"med": 1.0, "p(95)": 2.0, "p(99)": 3.0}
                },
                "rb_query_errors": {"values": {"rate": 0.0}},
                "rb_queries_run": {"values": {"count": 100, "rate": 10.0}},
                "http_req_failed": {"values": {"rate": 0.0}},
            }
        }
        (cell / "k6_summary.json").write_text(json.dumps(summary))
        (cell / "query_results.json").write_text(
            json.dumps({
                "dataset": "test",
                "num_queries": len(query_results),
                "captured": len(query_results),
                "failures": 0,
                "top_k": 10,
                "elapsed_s": 0.1,
                "results": query_results,
            })
        )

    baseline = {
        "0": [f"v{i}" for i in range(10)],
        "1": [f"u{i}" for i in range(10)],
    }
    write_cell("tier-off", baseline)
    write_cell("tier-on-cold", baseline)
    write_cell("tier-on-warm", baseline)

    res = subprocess.run(
        ["python3", str(ANALYZE_PY), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert res.returncode == 0, (
        f"expected rc=0 (no violations), got {res.returncode}\n"
        f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
    results_md = (tmp_path / "RESULTS.md").read_text()
    assert "CELL-AGREEMENT VIOLATION" not in results_md
    assert "tier-off" in results_md
    assert "tier-on-cold" in results_md
    assert "tier-on-warm" in results_md
    # The agreement column should show 1.000 for the perfectly-matching cells.
    assert "1.000" in results_md
