from __future__ import annotations

"""Periodic DP residency reconciler.

A long-lived daemon thread, modelled on `services/_common/catalog_listener.py`
and `services/_common/prewarm_consumer.py`. Every `RB_DP_RESIDENCY_SYNC_S`
seconds (default 5), the worker takes a snapshot of `shard_tier.residency()`,
diffs it against the previous snapshot, and writes the diff to the
`dp_shard_residency` table via the state adapter:

  - URIs newly present: `state.register_dp_shard_warm(...)` — first INSERT.
  - URIs newly absent: `state.unregister_dp_shard_warm(...)` — DELETE.
  - URIs still present: `state.register_dp_shard_warm(...)` — UPSERT
    refreshes `last_query_at`, keeping the row "fresh" for the CP-side
    freshness filter (residency-aware routing is not yet wired).

Why periodic-reconcile instead of write-on-admit/evict:

  The SSD tier (`adapters/storage/shard_tier.py`) is a low-level adapter
  that owns its own files and admission logic. Importing the state
  adapter into the tier to push residency rows from `_admit_locked` /
  `_evict_locked` would couple two layers awkwardly — the tier would
  then transitively depend on Postgres, blow up its test surface, and
  make a "tier without registry" deployment impossible.

  Periodic-reconcile keeps the tier pure. The cost is a bounded
  staleness window of `RB_DP_RESIDENCY_SYNC_S` between an
  admit/evict and the registry row reflecting it. The registry is
  ADVISORY; a stale row only causes a cache miss on the wrong DP, never
  a correctness break.

Activation gate (default-off rollback):

  - `RB_DP_RESIDENCY_REGISTRY` unset or false: `start_if_needed()` is a
    no-op, no thread spawns, no DB writes happen. The DP behaves
    exactly as before.
  - `RB_DP_RESIDENCY_REGISTRY=true`: the writer thread spawns and the
    diff/UPSERT loop runs at the configured cadence.

Resilience: a DB write failure (Postgres outage, pool exhausted, etc.) is
caught and the previous-snapshot is NOT advanced — so the next cycle
re-emits the same diff. The bounded backoff prevents a tight retry loop
against a hard-down database. The daemon-thread pattern is the substrate
for resilience: a worker exception is logged-and-suppressed; one bad cycle
must not stop the writer.
"""

import logging
import threading
import time
from typing import Dict, Optional, Set, Tuple

from adapters import config
from adapters.state import state
from adapters.storage import shard_tier
from services._common.dp_id import dp_id


_LOG = logging.getLogger(__name__)

# Default sync interval. 5 s is the advisory staleness window — registry
# rows reflect admits/evicts within 5 s, comfortable for the advisory
# contract and not so tight that the writer is a noisy producer. An
# operator can shorten this via `RB_DP_RESIDENCY_SYNC_S` for testing or
# lengthen it to reduce write pressure on a multi-tenant Postgres.
_DEFAULT_SYNC_INTERVAL_S = 5.0

# Backoff bounds for a DB-write failure. The writer logs and backs off
# rather than killing the worker; on a hard-down database the worker
# sleeps up to the cap between retries. Module-scope so tests can
# monkeypatch them to sub-second values.
_BACKOFF_INITIAL_S = 0.5
_BACKOFF_CAP_S = 30.0


# --- gate -----------------------------------------------------------------


def _gate_enabled() -> bool:
    """Read the activation gate at call time so a flip-flopping env across
    pod restarts cleanly toggles the writer. Matches the rest of the
    codebase's env-flag parsing — `true` / `1` / `yes`.
    """
    return config.dp_residency_registry()


def _sync_interval_s() -> float:
    """Sync interval from env, with a sane positive floor.

    Reading at call time (not at import) lets tests pin a fast cadence
    via `monkeypatch.setenv` without a module reload between the env set
    and the first cycle. The 0.001-second floor prevents an operator
    accidentally pinning the worker into a busy loop.
    """
    raw = config.dp_residency_sync_s()
    if raw is None or raw.strip() == "":
        return _DEFAULT_SYNC_INTERVAL_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        _LOG.warning(
            "residency_writer: invalid RB_DP_RESIDENCY_SYNC_S=%r; using default %s",
            raw, _DEFAULT_SYNC_INTERVAL_S,
        )
        return _DEFAULT_SYNC_INTERVAL_S
    return max(0.001, value)


# --- internal state -------------------------------------------------------


_THREAD_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None
_STOP_EVENT = threading.Event()


# --- public API -----------------------------------------------------------


def start_if_needed() -> None:
    """Spawn the writer thread when the gate is enabled. Idempotent.

    A call with the gate disabled is a no-op — that is the rollback
    contract: a deployment that never sets `RB_DP_RESIDENCY_REGISTRY`
    runs this file as dead code (the thread never starts, no DB writes
    happen). A second call observes the existing thread and returns; a
    call after `stop()` clears the prior stop signal and spawns a fresh
    thread (so tests with module reloads round-trip cleanly).
    """
    if not _gate_enabled():
        return
    global _THREAD
    with _THREAD_LOCK:
        if _THREAD is not None and _THREAD.is_alive():
            return
        _STOP_EVENT.clear()
        _THREAD = threading.Thread(
            target=_run, name="residency-writer", daemon=True,
        )
        _THREAD.start()


def stop(join_timeout_s: float = 2.0) -> None:
    """Signal the writer thread to exit; join with a short timeout.

    Idempotent: calling stop() with no live thread is a no-op. The join
    timeout prevents a stuck worker from blocking shutdown indefinitely;
    a daemon thread is left to die with the process if the join times out.
    """
    _STOP_EVENT.set()
    with _THREAD_LOCK:
        t = _THREAD
    if t is not None and t.is_alive():
        t.join(timeout=join_timeout_s)


def _is_running() -> bool:
    """Test helper — returns whether the writer thread is alive."""
    with _THREAD_LOCK:
        t = _THREAD
    return bool(t and t.is_alive())


# --- internals ------------------------------------------------------------


def _reconcile_once(
    previous: Set[str],
    warm_since_by_uri: Dict[str, float],
) -> Tuple[Set[str], Dict[str, float]]:
    """Diff one cycle and dispatch the register/unregister calls.

    Returns the updated `(previous_uris, warm_since_by_uri)` tuple so the
    caller threads the snapshot into the next cycle. A DB-write failure
    raises out of this function so the caller can apply backoff and NOT
    advance `previous` — the next cycle then re-emits the same diff
    rather than silently losing a row.

    `warm_since_by_uri` accumulates the FIRST `time.time()` at which we
    observed each URI. The UPSERT contract is "first-write wins for
    warm_since"; we mirror that on the caller side by remembering the
    earliest observation, so the value sent into `register_dp_shard_warm`
    is stable across cycles for a still-resident URI. (The state
    adapter's memory branch protects this server-side too — both layers
    converge on the same first-write-wins semantics.)
    """
    snapshot = shard_tier.residency()
    current_uris: Set[str] = {entry.shard_uri for entry in snapshot}
    now = time.time()

    # `time.time()` is wall-clock seconds (epoch). The DB stores epoch
    # seconds (DOUBLE PRECISION column), so `time.time()` is the right
    # source — NOT `time.monotonic()` (which is opaque to other
    # processes). A wall-clock step between cycles could make
    # `last_query_at` non-monotonic on one row; the freshness filter
    # tolerates that (it cares about "is this row recent", not "is this
    # row strictly increasing").
    new_uris = current_uris - previous
    dropped_uris = previous - current_uris

    # Maintain warm_since_by_uri across cycles: new URIs get a fresh
    # warm_since; dropped URIs are removed; unchanged URIs keep their
    # original. The dict is the writer's source of truth for the first-
    # observed timestamp of each URI.
    next_warm_since = {
        uri: warm_since_by_uri.get(uri, now) for uri in current_uris
    }

    me = dp_id()

    # Dispatch order is intentionally:
    #   1. unregister dropped URIs first (clear out stale rows so a CP
    #      read between unregister and register sees the truth);
    #   2. register all current URIs (new + unchanged) so the UPSERT
    #      refreshes last_query_at uniformly.
    # Failures bubble up; the caller does not advance `previous` so the
    # same diff re-runs on the next cycle.
    for uri in dropped_uris:
        state.unregister_dp_shard_warm(me, uri)
    for uri in current_uris:
        warm_since = next_warm_since[uri]
        state.register_dp_shard_warm(me, uri, warm_since, now)

    return current_uris, next_warm_since


def _run() -> None:
    """Worker loop: reconcile / sleep / repeat, with bounded backoff on error.

    Each successful cycle resets the backoff so a transient failure that
    cleared is not penalised on the next genuine error. The `_STOP_EVENT.
    wait(...)` returns True if stop() fires during the sleep, so shutdown
    is prompt even at the longest backoff.
    """
    previous: Set[str] = set()
    warm_since_by_uri: Dict[str, float] = {}
    backoff = _BACKOFF_INITIAL_S
    while not _STOP_EVENT.is_set():
        try:
            previous, warm_since_by_uri = _reconcile_once(
                previous, warm_since_by_uri,
            )
        except Exception:  # noqa: BLE001 - one bad cycle must not kill the worker
            _LOG.exception(
                "residency_writer: reconcile failed; backoff %.1fs", backoff,
            )
            if _STOP_EVENT.wait(timeout=backoff):
                return
            backoff = min(backoff * 2.0, _BACKOFF_CAP_S)
            continue
        # Success — reset backoff and sleep the normal interval. A stop
        # during the sleep returns True from `wait()` so shutdown is
        # observed within one cycle rather than the worst-case backoff.
        backoff = _BACKOFF_INITIAL_S
        if _STOP_EVENT.wait(timeout=_sync_interval_s()):
            return
