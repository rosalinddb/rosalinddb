"""`services/_common/residency_writer.py` — shard residency registry writer.

The residency writer is a long-lived daemon thread that periodically
reconciles `shard_tier.residency()` against the residency registry rows
written by `state.register_dp_shard_warm` / `state.unregister_dp_shard_warm`.

Periodic-reconcile (not event-driven write-on-admit) is the chosen
contract — see the module docstring for why (decoupling: the storage tier
stays pure and does not import state). The staleness window is bounded by
`RB_DP_RESIDENCY_SYNC_S` (default 5 s).

Tests here pin:

  - Lifecycle: `start_if_needed()` is idempotent; `stop()` joins quickly.
  - Diff semantics: new entries -> register; dropped entries -> unregister;
    unchanged entries -> register (UPSERT refreshes `last_query_at`).
  - Resilience: a DB write failure does NOT kill the worker; the next
    cycle re-tries with bounded backoff.
  - Default-off rollback: when `RB_DP_RESIDENCY_REGISTRY` is unset, no
    thread spawns and no register/unregister fires.
"""
from __future__ import annotations

import importlib
import threading
import time
from typing import List, Tuple

import pytest


pytestmark = pytest.mark.unit


# --- helpers --------------------------------------------------------------


def _wait_until(predicate, timeout_s: float = 2.0, interval_s: float = 0.01) -> bool:
    """Poll `predicate` until True or `timeout_s` elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


class _FakeResidencyEntry:
    """Stand-in for `shard_tier.ResidencyEntry` (only the fields we read)."""

    def __init__(self, shard_uri: str, nbytes: int = 100):
        self.shard_uri = shard_uri
        self.local_path = f"/tmp/{shard_uri.replace('/', '_')}"
        self.nbytes = nbytes
        self.last_admit_at = 0.0
        self.last_query_at = 0.0


# --- fixtures -------------------------------------------------------------


@pytest.fixture
def writer_mod(monkeypatch, tmp_path):
    """Reload `services._common.residency_writer` with a clean per-test env.

    The fast sync interval (50 ms) keeps the diff-loop assertions snappy
    without the test having to sleep for the production 5 s default.
    """
    monkeypatch.setenv("RB_DP_RESIDENCY_REGISTRY", "true")
    monkeypatch.setenv("RB_DP_RESIDENCY_SYNC_S", "0.05")
    monkeypatch.setenv("RB_DP_ID", "test-dp-1")
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("RB_SHARD_TIER_DIR", str(tmp_path / "tier"))

    # Reload dp_id first so the writer's import sees the env-pinned id.
    import services._common.dp_id as dp_id_mod
    importlib.reload(dp_id_mod)
    import services._common.residency_writer as mod

    importlib.reload(mod)
    yield mod
    try:
        mod.stop(join_timeout_s=1.0)
    except Exception:  # noqa: BLE001 - test cleanup
        pass


def _install_fake_residency(monkeypatch, mod, snapshots: List[List[str]]):
    """Replace `shard_tier.residency()` with a generator over `snapshots`.

    Each entry in `snapshots` is the list of shard_uris the tier reports
    in that diff cycle. After the list is exhausted the fake returns the
    last snapshot forever (so the assertion phase can run without the
    fake racing the worker into IndexError).
    """
    cursor = {"i": 0}
    lock = threading.Lock()

    def _fake_residency():
        with lock:
            i = min(cursor["i"], len(snapshots) - 1)
            cursor["i"] += 1
        return [_FakeResidencyEntry(u) for u in snapshots[i]]

    monkeypatch.setattr(mod.shard_tier, "residency", _fake_residency)


def _install_recorder(monkeypatch, mod):
    """Replace `state.register_dp_shard_warm` / `unregister_dp_shard_warm`
    with call recorders so the test can read the diff dispatched per cycle.

    Returns the two lists the recorders append into.
    """
    registered: List[Tuple[str, str, float, float]] = []
    unregistered: List[Tuple[str, str]] = []
    lock = threading.Lock()

    def _fake_register(dp_id, shard_uri, warm_since, last_query_at):
        with lock:
            registered.append((dp_id, shard_uri, warm_since, last_query_at))

    def _fake_unregister(dp_id, shard_uri):
        with lock:
            unregistered.append((dp_id, shard_uri))

    monkeypatch.setattr(mod.state, "register_dp_shard_warm", _fake_register)
    monkeypatch.setattr(mod.state, "unregister_dp_shard_warm", _fake_unregister)
    return registered, unregistered


# --- tests ----------------------------------------------------------------


def test_start_is_idempotent(writer_mod, monkeypatch):
    """`start_if_needed` twice spawns exactly one worker thread."""
    _install_fake_residency(monkeypatch, writer_mod, [[]])
    _install_recorder(monkeypatch, writer_mod)

    writer_mod.start_if_needed()
    assert _wait_until(writer_mod._is_running)
    writer_mod.start_if_needed()  # idempotent
    assert writer_mod._is_running()


def test_new_entries_dispatch_register(writer_mod, monkeypatch):
    """A residency entry not seen last cycle triggers `register_dp_shard_warm`.

    The empty -> [a] transition is the "first cycle after a cold start"
    shape — every URI in the tier gets a registry write.
    """
    snapshots = [["memory://b/s1.bin"]]
    _install_fake_residency(monkeypatch, writer_mod, snapshots)
    registered, _ = _install_recorder(monkeypatch, writer_mod)

    writer_mod.start_if_needed()
    assert _wait_until(lambda: any(r[1] == "memory://b/s1.bin" for r in registered))
    # The registered tuple is `(dp_id, shard_uri, warm_since, last_query_at)`.
    dp_id, shard_uri, warm_since, last_query_at = registered[0]
    assert dp_id == "test-dp-1"
    assert shard_uri == "memory://b/s1.bin"
    assert warm_since > 0  # wall-clock seconds
    assert last_query_at >= warm_since


def test_dropped_entries_dispatch_unregister(writer_mod, monkeypatch):
    """A URI seen last cycle but missing this cycle triggers `unregister`.

    The [a] -> [] transition is the "evicted between cycles" shape. The
    writer must drop the registry row so routing does not steer at a DP
    that no longer holds the shard.
    """
    snapshots = [
        ["memory://b/s1.bin"],
        [],  # s1 evicted
    ]
    _install_fake_residency(monkeypatch, writer_mod, snapshots)
    _, unregistered = _install_recorder(monkeypatch, writer_mod)

    writer_mod.start_if_needed()
    assert _wait_until(
        lambda: ("test-dp-1", "memory://b/s1.bin") in unregistered,
        timeout_s=3.0,
    )


def test_unchanged_entries_dispatch_register_to_refresh_last_query_at(
    writer_mod, monkeypatch,
):
    """A URI still resident triggers a re-register so the UPSERT refreshes the row.

    `register_dp_shard_warm` is the UPSERT primitive; calling it on an
    unchanged URI is how the writer keeps `last_query_at` recent. A CP
    reading the registry with a freshness filter would otherwise time
    out a still-warm shard.
    """
    snapshots = [
        ["memory://b/s1.bin"],
        ["memory://b/s1.bin"],
        ["memory://b/s1.bin"],
    ]
    _install_fake_residency(monkeypatch, writer_mod, snapshots)
    registered, unregistered = _install_recorder(monkeypatch, writer_mod)

    writer_mod.start_if_needed()
    # Wait for at least 2 register calls for s1 (cycle 1 + cycle 2 refresh).
    assert _wait_until(
        lambda: sum(1 for r in registered if r[1] == "memory://b/s1.bin") >= 2,
        timeout_s=3.0,
    )
    # And no unregister for the still-resident shard.
    assert ("test-dp-1", "memory://b/s1.bin") not in unregistered


def test_register_failure_does_not_kill_worker(writer_mod, monkeypatch):
    """A `register_dp_shard_warm` raising does NOT crash the daemon thread.

    Simulates a DB outage: the writer logs the failure, sleeps for the
    bounded backoff window, and retries on the next cycle. A worker that
    died on first DB error would silently stop syncing — exactly the
    failure mode the catalog_listener / prewarm_consumer survive.
    """
    snapshots = [["memory://b/s1.bin"], ["memory://b/s1.bin"]]
    _install_fake_residency(monkeypatch, writer_mod, snapshots)

    call_count = {"n": 0}
    successful_register: List[Tuple[str, str, float, float]] = []

    def _flaky_register(dp_id, shard_uri, warm_since, last_query_at):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated DB outage")
        successful_register.append((dp_id, shard_uri, warm_since, last_query_at))

    monkeypatch.setattr(writer_mod.state, "register_dp_shard_warm", _flaky_register)
    monkeypatch.setattr(writer_mod.state, "unregister_dp_shard_warm", lambda *a, **k: None)

    writer_mod.start_if_needed()
    assert _wait_until(lambda: len(successful_register) >= 1, timeout_s=3.0)
    assert writer_mod._is_running()


def test_stop_joins_worker_quickly(writer_mod, monkeypatch):
    """`stop()` is idempotent and the worker exits within the join timeout."""
    _install_fake_residency(monkeypatch, writer_mod, [[]])
    _install_recorder(monkeypatch, writer_mod)

    writer_mod.start_if_needed()
    assert _wait_until(writer_mod._is_running)

    writer_mod.stop(join_timeout_s=2.0)
    assert _wait_until(lambda: not writer_mod._is_running())
    # Idempotent — a second stop on a dead worker is a no-op.
    writer_mod.stop(join_timeout_s=0.1)


def test_gate_off_does_not_spawn_thread(monkeypatch, tmp_path):
    """`RB_DP_RESIDENCY_REGISTRY` unset -> `start_if_needed` is a no-op.

    Default-off rollback contract: every new env flag preserves current
    behaviour when unset. A DP that has not opted in must NOT spawn the
    writer thread and must NOT call into the state adapter.
    """
    monkeypatch.delenv("RB_DP_RESIDENCY_REGISTRY", raising=False)
    monkeypatch.setenv("RB_DP_ID", "test-dp-gateoff")
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("RB_SHARD_TIER_DIR", str(tmp_path / "tier"))
    import services._common.dp_id as dp_id_mod
    importlib.reload(dp_id_mod)
    import services._common.residency_writer as mod
    importlib.reload(mod)

    registered, unregistered = _install_recorder(monkeypatch, mod)
    _install_fake_residency(monkeypatch, mod, [["memory://b/s1.bin"]])

    mod.start_if_needed()
    # Brief sleep so a (buggy) early-spawned thread would have time to
    # land at least one cycle. After the sleep, neither thread nor
    # registry-write should have happened.
    time.sleep(0.2)
    assert not mod._is_running(), "writer must not run with the gate off"
    assert registered == [], (
        f"writer must not write with the gate off; calls={registered}"
    )
    assert unregistered == []
