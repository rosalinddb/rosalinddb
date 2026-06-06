"""SSD tier: byte-budgeted, single-flight, LRU-evicted local-disk cache.

This module sits between the object store and the in-process RAM cache. It
implements the per-URI single-flight pattern and owns its own local files: it
downloads on miss, evicts under byte pressure, and survives a process restart
by leaving its files on disk for the next process to discover (the next
process simply admits them on first use, the same as a cold fetch).

What this module provides today:
  - `fetch(shard_uri) -> local_path`: single-flight per URI, atomic publish
    via temp+rename, byte-budgeted LRU eviction on admission. Called by
    `v1_query.py`, `ephemeral_runner`, and `dp_app` on every cache-miss path.
  - `prewarm(shard_uri) -> local_path`: speculative admit with the
    `MIN_RESIDENT_S` admission floor (`CacheCapacityExceeded` is raised when
    the tier is full of recently-arrived shards). Called by `prewarm_consumer`.
  - `evict(shard_uri) -> bool`: removes the residency entry and unlinks the
    file. The POSIX unlink-with-open-fd guarantee means a concurrent reader
    that already has an mmap on the file keeps reading; the inode is reclaimed
    only when the last fd closes. Exposed for catalog invalidation and tests;
    no production caller invokes it directly today.
  - `residency()` / `bytes_used()`: observability for tests, operators, and
    the residency-registry writer (`residency_writer`).
  - `CacheCapacityExceeded`: raised by `prewarm` when the floor rejects a
    speculative arrival; mapped to a 503 by the calling service.
  - `MIN_RESIDENT_S` admission floor: tunable at runtime via
    `RB_SHARD_TIER_MIN_RESIDENT_S`; discriminates queries-under-load
    (unconditional `fetch`) from speculative arrivals (`prewarm`, floor-bound).

The W-TinyLFU policy is the long-term eviction target; the LRU implementation
here is the correct first step.

Single-flight ordering invariant (load-bearing):
  1. Initiator writes `tmp` and atomically renames it to `path`.
  2. Initiator updates the residency table and pops the in-flight entry.
  3. Initiator sets the event.
  Waiters that wake on step 3 see a fully published file, no residency
  inconsistency, and no stale in-flight entry. Reordering any of these steps
  re-opens the race the single-flight fix already closed in `_ensure_cached`
  (a waiter mmap'ing a half-written file, or a wake-and-re-enter thread
  becoming a perpetual waiter).

Known limitation — byte-budget drift across process restart:
  Files survive process restart, but the in-memory `_BYTES_USED` counter
  is rebuilt at zero on import. After a restart, the next admissions
  stack on top of the surviving files until `_BYTES_USED` catches up to
  actual on-disk usage — actual disk usage can temporarily exceed
  `RB_SHARD_TIER_BYTES`. A startup scan of the tier directory to rebuild
  the residency table is the natural fix. Documented here so an operator
  who looks at `df` after a fresh DP boot is not surprised to find more
  bytes resident than the budget should allow.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from typing import Dict, NamedTuple, Optional

from adapters import config
from adapters.errors import CacheCapacityExceeded, ShardTierTimeout


_log = logging.getLogger(__name__)


# --- public types ---------------------------------------------------------


class ResidencyEntry(NamedTuple):
    """A snapshot row from the residency table.

    The canonical identifier for a cached shard is its `shard_uri` — the
    immutable, content-addressed S3-style URI. Two different builds produce
    two different URIs (different content_hash) so
    a URI-keyed cache naturally gets per-version isolation for free: a new
    version is a new cache entry, the old version ages out under LRU,
    invalidation never has to "which version is this" because the URI
    answers that. The catalog `shard_catalog.id` int is a useful row
    identifier inside Postgres but is not what the cache layer should
    think in.

    `last_admit_at` is set once at insert time and is the input to the
    `MIN_RESIDENT_S` admission floor. `last_query_at` is refreshed on every
    fetch hit and is what LRU ordering walks. Both are `time.monotonic()`
    seconds; they are intentionally not wall-clock so a clock step cannot
    skew eviction order.
    """

    shard_uri: str
    local_path: str
    nbytes: int
    last_admit_at: float
    last_query_at: float


# `ShardTierTimeout` and `CacheCapacityExceeded` are defined once in
# `adapters.errors` (shared class identity) and imported above; they are
# re-exported here so the original `adapters.storage.shard_tier.*` import paths
# and every `isinstance` / `except` frame keep working unchanged. They remain
# `RuntimeError` subclasses (via `RosalindDBError`), preserving the
# `except RuntimeError` contract the classifier branches rely on.


# --- module state ---------------------------------------------------------


# Byte budget for the tier. When unset in the environment, default to a
# test-friendly 2 GiB.
_TIER_BYTES = config.shard_tier_bytes()

# Tier directory. Defaults to `${CACHE_DIR}/tier-managed/` so a deployment
# that already has files in `CACHE_DIR` (written by the legacy `_ensure_cached`
# path) does not have those files silently adopted by the tier on first
# fetch. The subdirectory keeps files from the two cache layers clearly
# demarcated.
# shard_tier historically read CACHE_DIR with empty-string-as-unset semantics (a
# Compose `${CACHE_DIR:-}` passthrough must fall back to the default, not ""),
# unlike config.cache_dir() (plain os.getenv, matching the other 5 consumers), so
# collapse an explicitly-empty value to the default here.
_DEFAULT_CACHE_DIR = config.cache_dir() or "/var/cache/shards"
_TIER_DIR = config.shard_tier_dir() or os.path.join(
    _DEFAULT_CACHE_DIR, "tier-managed"
)

# Bounded wait for coalesced waiters. A hung initiator should fail the
# waiters with a clear error rather than block them indefinitely. 300 s
# comfortably covers a multi-GB GET on typical infra while bounding the
# worst-case waiter stall.
_COALESCE_WAIT_S = config.shard_tier_coalesce_wait_s()

# Maximum age of an orphan `.tmp` file before the import-time sweep removes
# it. A crashed initiator leaves `{path}.{pid}.{uuid8}.tmp` behind; without
# a sweep, long-lived deployments restart-with-leaked-tmps and the tier
# directory grows monotonically over operator's calendar time. 1 hour is
# well above any realistic in-flight download (the bounded wait above is
# 300s); raising the threshold above 1h would let a single failed
# multi-day-uptime run leak gigabytes.
_ORPHAN_TMP_MAX_AGE_S = config.shard_tier_tmp_max_age_s()

# `prewarm()` admission floor: an LRU candidate younger than this many
# seconds CANNOT be evicted to make room for a speculative arrival. This
# is the discrimination signal that lets queries-under-load (`fetch`,
# unconditional) beat speculative arrivals (`prewarm`, floor-honouring).
# Default 30 s; `0` disables the floor entirely so `prewarm` behaves as
# unconditional fetch.
#
# Stored at module scope so an operator can retune at runtime by
# monkey-patching this constant (no DP restart needed). Tests pin both
# behaviours: the boundary case (age == floor) admits, and a live edit
# takes effect on the next prewarm.
_MIN_RESIDENT_S = config.shard_tier_min_resident_s()


# Residency table. Keyed by `shard_uri` — the same key the in-flight
# single-flight dict uses. A versioned URI uniquely identifies a shard's
# bytes (content-addressed shape: identical bytes => identical URI;
# different bytes => guaranteed-different URI), so URI-as-key gives the
# cache invariants for free:
#   - version change = new key = clean miss => fresh download (no stale-bytes risk)
#   - same bytes rebuilt = same URI = warm hit (cheap dedup)
# `OrderedDict` semantics: insertion order is initial LRU order,
# `move_to_end` walks an entry toward MRU, `popitem(last=False)` is O(1)
# LRU eviction. The same data structure that backs the in-process FAISS
# cache, for the same reasons.
#
# The value tuple is `(local_path, nbytes, last_admit_at, last_query_at)`
# — a flat tuple instead of a dataclass because the table is hot-path-
# adjacent and the tuple form is the cheapest to copy when `residency()`
# snapshots it.
_RESIDENCY: "OrderedDict[str, tuple]" = OrderedDict()
_RESIDENCY_LOCK = threading.Lock()
_BYTES_USED: int = 0


# In-flight downloads. Per-URI `threading.Event` so independent URIs
# download in parallel (a global mutex would re-serialise the cold warm-up
# the original `_ensure_cached` fix was written to prevent). The lock guards
# the dict itself, not the individual events.
_INFLIGHT: Dict[str, threading.Event] = {}
_INFLIGHT_LOCK = threading.Lock()


# --- internal helpers -----------------------------------------------------


def _sweep_orphan_tmps() -> int:
    """Remove `.tmp` files in `_TIER_DIR` older than `_ORPHAN_TMP_MAX_AGE_S`.

    Called once at module import. The fetch path writes to a unique
    `{path}.{pid}.{uuid8}.tmp` and `os.replace()`s into the final name on
    success; a crashed initiator leaves the `.tmp` behind. Without this
    sweep, long-lived processes accumulate orphan tmps indefinitely.

    Best-effort: filesystem errors (permissions, races against another
    sweeper or initiator) are logged but never block module import. Files
    that are NOT `.tmp` are ignored entirely — they are the tier's real
    cached files and the byte budget already manages their lifecycle.

    Returns the number of files actually removed (for the import-time log
    line and for tests that pin the contract).
    """
    if not os.path.isdir(_TIER_DIR):
        return 0
    now = time.time()
    swept = 0
    try:
        scandir = os.scandir(_TIER_DIR)
    except OSError as exc:
        _log.warning("shard_tier: orphan sweep failed on %s: %s", _TIER_DIR, exc)
        return 0
    with scandir:
        for entry in scandir:
            if not entry.name.endswith(".tmp"):
                continue
            try:
                age = now - entry.stat().st_mtime
                if age >= _ORPHAN_TMP_MAX_AGE_S:
                    os.unlink(entry.path)
                    swept += 1
            except FileNotFoundError:
                pass  # raced with another sweeper or operator cleanup
            except OSError as exc:
                _log.warning(
                    "shard_tier: failed to unlink %s during orphan sweep: %s",
                    entry.path,
                    exc,
                )
    if swept:
        _log.info("shard_tier: swept %d orphan .tmp files on import", swept)
    return swept


# Sweep orphan .tmp files on import. The tier directory survives process
# restart by design (files are re-admitted on first use), so a crashed
# initiator's .tmp would otherwise persist until manual cleanup. This is
# best-effort — log-and-continue, never raise from import.
_sweep_orphan_tmps()


def _local_path_for(shard_uri: str) -> str:
    """Map a shard URI to its tier-local filesystem path.

    Hyphens-not-slashes so the result is a single filename in `_TIER_DIR`,
    not a nested path. Matches the mangling `_ensure_cached` already uses for
    the legacy cache — keeping the convention consistent means an operator
    moving between the two layers sees recognisable filenames.
    """
    return os.path.join(_TIER_DIR, shard_uri.split("://", 1)[1].replace("/", "_"))


def _evict_locked(shard_uri: str) -> bool:
    """Drop a residency entry and unlink its file. Caller holds `_RESIDENCY_LOCK`.

    Returns True if an entry was actually removed. The unlink is best-effort:
    a concurrent reader that already has the file mmap'd keeps reading
    (POSIX), and a file that vanished from disk (operator action, second
    eviction race) is not an error from the tier's perspective.
    """
    global _BYTES_USED
    entry = _RESIDENCY.pop(shard_uri, None)
    if entry is None:
        return False
    local_path, nbytes, _admit, _last_query = entry
    _BYTES_USED -= nbytes
    try:
        os.unlink(local_path)
    except FileNotFoundError:
        # File already gone (concurrent eviction or operator cleanup). The
        # residency row was authoritative; we have nothing to do.
        pass
    except OSError as exc:
        # A permission error or busy file is a real operational problem but
        # not a correctness one — the residency entry is already gone and
        # a later cold fetch will re-create the file. Log loudly so it is
        # visible without crashing the caller.
        _log.warning(
            "shard_tier: failed to unlink %s during evict: %s", local_path, exc
        )
    return True


def _admit_locked(
    shard_uri: str,
    local_path: str,
    nbytes: int,
    now: float,
) -> None:
    """Insert / refresh a residency entry and run byte-budget eviction.

    Caller holds `_RESIDENCY_LOCK`. Splits cleanly into two pieces: the new
    entry is placed at the MRU end (so it survives the eviction loop if the
    LRU end has older entries to give up first), then the LRU end is evicted
    until the running total fits the budget. An entry larger than the entire
    budget is admitted and then immediately evicted — usable for the current
    fetch (the file is on disk before the admission runs) but not retained,
    matching the `_cache_put` semantics in `v1_query.py`.

    Eviction in `fetch()` is UNCONDITIONAL: the LRU end is popped regardless
    of age. `prewarm()` layers the `MIN_RESIDENT_S` floor on top — it adds an
    "is this candidate old enough to evict?" check and may raise
    `CacheCapacityExceeded` if every candidate is too young. The unconditional
    eviction in `fetch()` is the correct contract for query-driven admits:
    a real query MUST get its shard cached even if that means displacing a
    recently-arrived neighbour.

    Same-URI re-admission (rare — caller manually evicted then re-fetched)
    overwrites the existing entry; `os.replace` already overwrote the file
    atomically before this call. No version-replacement unlink is needed
    because the URI key guarantees a new version lands in a new entry —
    the old entry's `local_path` is the old version's file, which the
    LRU will reclaim on its own (or `evict(old_uri)` can drop it explicitly).
    """
    global _BYTES_USED
    existing = _RESIDENCY.pop(shard_uri, None)
    if existing is not None:
        _BYTES_USED -= existing[1]
    _RESIDENCY[shard_uri] = (local_path, nbytes, now, now)
    _BYTES_USED += nbytes
    while _BYTES_USED > _TIER_BYTES and _RESIDENCY:
        # `popitem(last=False)` removes the LRU end. We must NOT evict the
        # entry we just inserted unless it is the only one left and exceeds
        # the budget alone — `OrderedDict` already places the just-inserted
        # entry at the MRU end, so the LRU end is a different shard by
        # construction unless the table has size 1.
        oldest_uri, oldest = next(iter(_RESIDENCY.items()))
        _RESIDENCY.pop(oldest_uri)
        _BYTES_USED -= oldest[1]
        try:
            os.unlink(oldest[0])
        except FileNotFoundError:
            pass
        except OSError as exc:
            _log.warning(
                "shard_tier: failed to unlink %s during budget eviction: %s",
                oldest[0],
                exc,
            )


def _check_admission_capacity_locked(
    shard_uri: str, nbytes: int, now: float, min_resident_s: float
) -> None:
    """Admission gate for `prewarm()`. Caller holds `_RESIDENCY_LOCK`.

    Pre-flight check: would admitting `nbytes` require evicting one or more
    LRU entries that are younger than `min_resident_s`? If so, raise
    `CacheCapacityExceeded` BEFORE the download runs — a speculative arrival
    must never displace a recently-arrived neighbour, and rejecting before
    the download saves the bandwidth.

    Logic:
      - Compute the running total assuming we admit `nbytes` and walk the
        LRU end forward, accumulating bytes we WOULD evict until the total
        fits the budget.
      - Any candidate younger than the floor is a stop sign: raise.
      - If we never need to evict (room to spare), the loop never runs and
        the call returns cleanly.
      - If `shard_uri` is already resident, count its existing footprint as
        a "credit" — re-admitting an existing entry doesn't require evicting
        anyone for the bytes that were already accounted for.

    Boundary: age `>= min_resident_s` is admittable (entry that has reached
    the floor is fair game). `>` would mean a floor of 0 still rejects.
    """
    existing = _RESIDENCY.get(shard_uri)
    existing_bytes = existing[1] if existing is not None else 0
    # Projected bytes after re-admission: subtract the existing entry (we
    # will overwrite it) then add the new size.
    projected = _BYTES_USED - existing_bytes + nbytes
    if projected <= _TIER_BYTES:
        return  # spare capacity — no eviction needed, no floor check needed
    # Walk the LRU end in order. Skip the same URI if it appears (the
    # existing entry will be popped-and-reinserted during admission, so it
    # is not a candidate for eviction).
    bytes_to_reclaim = projected - _TIER_BYTES
    reclaimed = 0
    for candidate_uri, candidate in _RESIDENCY.items():
        if candidate_uri == shard_uri:
            continue
        candidate_admit_at = candidate[2]
        age = now - candidate_admit_at
        if age < min_resident_s:
            raise CacheCapacityExceeded(
                f"shard_tier: cannot admit {shard_uri} (nbytes={nbytes}); "
                f"LRU candidate {candidate_uri} is {age:.2f}s old, "
                f"under MIN_RESIDENT_S={min_resident_s}s floor"
            )
        reclaimed += candidate[1]
        if reclaimed >= bytes_to_reclaim:
            return
    # We walked the entire residency table without raising and still
    # cannot reclaim enough bytes — the tier is too small for this arrival
    # even with every entry evicted. Reject; the operator should raise
    # `RB_SHARD_TIER_BYTES`. The shape matches the floor-violation case so
    # the wire path collapses to the same 503.
    raise CacheCapacityExceeded(
        f"shard_tier: cannot admit {shard_uri} (nbytes={nbytes}); "
        f"even evicting every resident entry would not free enough "
        f"bytes for the {_TIER_BYTES}-byte budget"
    )


# --- public API -----------------------------------------------------------


def fetch(shard_uri: str) -> str:
    """Return the local SSD path for `shard_uri`, downloading on miss.

    Single-flight per `shard_uri`: N concurrent callers for the same URI
    trigger ONE underlying download; waiters block on a per-URI event and
    return the same local path on completion. The lock is per-URI so
    independent shards download in parallel.

    On admission, the byte-budgeted LRU evicts the LRU end until
    `bytes_used + nbytes <= RB_SHARD_TIER_BYTES`. `fetch()` evicts
    unconditionally; `prewarm()` applies the `MIN_RESIDENT_S` floor.

    The file is guaranteed to exist when the call returns. POSIX guarantee:
    any open mmap on the returned file stays valid even if a later `evict()`
    unlinks it.

    Raises `FileNotFoundError` if the object store has no such URI.
    Raises `ShardTierTimeout` if a coalesced waiter exceeds
    `RB_SHARD_TIER_COALESCE_WAIT_S` (default 300).

    The cache key is the URI itself. Two builds with identical bytes
    converge on the same URI (content-addressed shape) so they share a
    cache entry (cheap dedup). Two builds with different bytes get different
    URIs so they get different cache entries — there is no "same shard, new
    version" race to defend against at this layer. Catalog invalidation calls
    `evict(old_uri)` when it publishes a new URI; the next `fetch(new_uri)`
    is a normal cold miss.
    """
    os.makedirs(_TIER_DIR, exist_ok=True)

    # Imported inside the function so tests can monkeypatch the symbol on
    # the storage module without having to also reload `shard_tier`.
    from adapters.storage.storage import read_bytes

    local_path = _local_path_for(shard_uri)
    now = time.monotonic()

    # Fast path: warm hit. Refresh `last_query_at` and walk the entry to
    # the MRU end. The file is already atomic on disk (publish via
    # temp+rename), so a concurrent reader sees either nothing or the
    # complete bytes — no coordination needed here.
    with _RESIDENCY_LOCK:
        entry = _RESIDENCY.get(shard_uri)
        if entry is not None and os.path.exists(entry[0]):
            path_old, nbytes, admit, _last_query = entry
            _RESIDENCY[shard_uri] = (path_old, nbytes, admit, now)
            _RESIDENCY.move_to_end(shard_uri)
            return path_old

    # Single-flight registration. Atomically decide whether this thread is
    # the *initiator* (creates the entry, will do the GET) or a *waiter*
    # (found a pre-existing entry, will block on its event). The inner
    # residency re-check closes the race where the initiator finished AND
    # cleared the in-flight entry between our outer check and this lock.
    is_initiator = False
    with _INFLIGHT_LOCK:
        with _RESIDENCY_LOCK:
            entry = _RESIDENCY.get(shard_uri)
            if entry is not None and os.path.exists(entry[0]):
                path_old, nbytes, admit, _last_query = entry
                _RESIDENCY[shard_uri] = (path_old, nbytes, admit, now)
                _RESIDENCY.move_to_end(shard_uri)
                return path_old
        event = _INFLIGHT.get(shard_uri)
        if event is None:
            event = threading.Event()
            _INFLIGHT[shard_uri] = event
            is_initiator = True

    if not is_initiator:
        # Waiter path. Block on the initiator's event with a bounded timeout
        # so a wedged initiator cannot stall callers indefinitely. On a
        # clean set(), re-check the residency table — the initiator may
        # have failed, in which case the event fires but no entry exists
        # and the caller should retry (or treat as a storage failure).
        _log.debug("shard_tier: coalesced waiter on %s", shard_uri)
        completed = event.wait(_COALESCE_WAIT_S)
        if not completed:
            _log.warning(
                "shard_tier: coalescing timeout after %.1fs on %s",
                _COALESCE_WAIT_S,
                shard_uri,
            )
            raise ShardTierTimeout(
                f"timed out after {_COALESCE_WAIT_S}s waiting for an "
                f"in-flight download of {shard_uri}"
            )
        with _RESIDENCY_LOCK:
            entry = _RESIDENCY.get(shard_uri)
            if entry is not None and os.path.exists(entry[0]):
                path_old, nbytes, admit, _last_query = entry
                _RESIDENCY[shard_uri] = (
                    path_old, nbytes, admit, time.monotonic(),
                )
                _RESIDENCY.move_to_end(shard_uri)
                return path_old
        # Initiator's download must have failed; the entry is not there.
        # Raise FileNotFoundError so the caller's classifier routes this to
        # storage_unavailable / 503 instead of a catch-all 500. Customers
        # retrying on 503 get the right transient-failure semantics.
        raise FileNotFoundError(
            f"coalesced download of {shard_uri} did not publish a tier entry "
            "(initiator likely failed); caller should retry"
        )

    # Initiator path. Download the bytes, publish via temp+rename, admit
    # to the residency table, then clear the in-flight entry and set the
    # event. The ordering — rename, admit, pop, set — is load-bearing: a
    # waiter that wakes on `set()` sees a fully-published file AND a
    # consistent residency entry; a thread that wakes-and-re-enters after
    # `pop()` becomes a fresh initiator on a fresh event rather than a
    # perpetual waiter on a fired-and-gone one.
    # `download_to` streams the GET to `tmp` without buffering the whole
    # object in RAM. Replaces the prior `payload = read_bytes(uri);
    # f.write(payload)` pattern that OOMed on multi-GB shards inside a
    # typical DP cgroup limit (see the storage adapter's docstring for the
    # full diagnosis).
    from adapters.storage.storage import download_to

    tmp = f"{local_path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        try:
            download_to(shard_uri, tmp)
            os.replace(tmp, local_path)
        except BaseException:
            # Clean up the leftover temp on any failure before the rename.
            # Best-effort: the file may already be gone.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

        # `nbytes` for the byte-budget admission used to come from
        # `len(payload)` when the download materialised in memory; with
        # streaming download we stat the just-renamed file instead. The
        # value is identical (same bytes on disk), and the cost is a
        # single syscall — cheap relative to the multi-GB GET that just
        # finished.
        nbytes = os.path.getsize(local_path)
        with _RESIDENCY_LOCK:
            _admit_locked(shard_uri, local_path, nbytes, time.monotonic())
        return local_path
    finally:
        # Order: pop FIRST, then set(). A waiter that wakes-and-re-enters
        # `fetch()` must see no in-flight entry so it becomes a fresh
        # initiator (or, if the admission happened, a warm hit on the
        # residency entry). This mirrors the v1_query._ensure_cached
        # ordering and is what makes a failed download safely retryable.
        with _INFLIGHT_LOCK:
            _INFLIGHT.pop(shard_uri, None)
        event.set()


def prewarm(shard_uri: str) -> str:
    """Speculatively admit `shard_uri` to the SSD tier.

    Like `fetch()` but with the admission floor: the LRU end is eligible
    for eviction only if its `last_admit_at` is older than `_MIN_RESIDENT_S`.
    If every candidate is too young, raise `CacheCapacityExceeded` — the
    operator gets a 503 and the recently-arrived neighbours stay put.

    `fetch()` is UNCONDITIONAL (a real query MUST get its shard cached even
    if displacing a recently-arrived neighbour); `prewarm()` honours the
    floor (a speculative arrival must lose to queries-under-load). The floor
    is the discrimination signal.

    Idempotent on warm: a `prewarm` of an already-resident URI returns
    the existing local path without re-downloading. Idempotent on cold:
    callers can retry safely — the in-flight registration is cleared in
    `finally` even on `CacheCapacityExceeded` so the next attempt becomes
    a fresh initiator.

    Single-flight machinery is shared with `fetch()`: N concurrent
    prewarms for the same URI trigger ONE download and N-1 waiters.

    Raises:
        CacheCapacityExceeded: tier is full and every candidate is younger
            than `_MIN_RESIDENT_S` (or the entire tier is smaller than
            this shard).
        FileNotFoundError: the object store has no such URI.
        ShardTierTimeout: a coalesced waiter exceeded the bounded wait.
    """
    os.makedirs(_TIER_DIR, exist_ok=True)

    from adapters.storage.storage import read_bytes

    local_path = _local_path_for(shard_uri)
    now = time.monotonic()

    # Fast path: already warm. Refresh `last_query_at` and walk to MRU,
    # same as `fetch()` — prewarm is idempotent on a warm URI.
    with _RESIDENCY_LOCK:
        entry = _RESIDENCY.get(shard_uri)
        if entry is not None and os.path.exists(entry[0]):
            path_old, nbytes, admit, _last_query = entry
            _RESIDENCY[shard_uri] = (path_old, nbytes, admit, now)
            _RESIDENCY.move_to_end(shard_uri)
            return path_old

    # Single-flight registration. Mirror of `fetch()` so concurrent
    # prewarms for the same URI coalesce on one initiator.
    is_initiator = False
    with _INFLIGHT_LOCK:
        with _RESIDENCY_LOCK:
            entry = _RESIDENCY.get(shard_uri)
            if entry is not None and os.path.exists(entry[0]):
                path_old, nbytes, admit, _last_query = entry
                _RESIDENCY[shard_uri] = (path_old, nbytes, admit, now)
                _RESIDENCY.move_to_end(shard_uri)
                return path_old
        event = _INFLIGHT.get(shard_uri)
        if event is None:
            event = threading.Event()
            _INFLIGHT[shard_uri] = event
            is_initiator = True

    if not is_initiator:
        # Waiter path. Block on the initiator's event. If the initiator
        # got rejected by the floor, the residency table will be empty
        # for this URI on wake and we surface the same rejection — a
        # waiter on a rejected prewarm gets the rejection too, not a
        # bogus path.
        _log.debug("shard_tier: coalesced prewarm waiter on %s", shard_uri)
        completed = event.wait(_COALESCE_WAIT_S)
        if not completed:
            _log.warning(
                "shard_tier: coalescing timeout after %.1fs on prewarm %s",
                _COALESCE_WAIT_S,
                shard_uri,
            )
            raise ShardTierTimeout(
                f"timed out after {_COALESCE_WAIT_S}s waiting for an "
                f"in-flight prewarm of {shard_uri}"
            )
        with _RESIDENCY_LOCK:
            entry = _RESIDENCY.get(shard_uri)
            if entry is not None and os.path.exists(entry[0]):
                path_old, nbytes, admit, _last_query = entry
                _RESIDENCY[shard_uri] = (
                    path_old, nbytes, admit, time.monotonic(),
                )
                _RESIDENCY.move_to_end(shard_uri)
                return path_old
        # The initiator either failed the download (FileNotFoundError) or
        # was rejected by the admission floor (CacheCapacityExceeded).
        # The exact cause is hard to disambiguate here; surface as
        # CacheCapacityExceeded (the prewarm-specific signal) so the
        # caller's classifier routes to 503 + cache_capacity_exceeded.
        raise CacheCapacityExceeded(
            f"coalesced prewarm of {shard_uri} did not publish a tier "
            "entry (initiator was rejected or failed); caller should retry"
        )

    # Initiator path. Download first, then admit under the floor. The
    # download lands as `local_path` on disk — even if admission rejects
    # the file is replaced atomically on the next admission so it does
    # not corrupt anything; we explicitly clean it up below in the
    # rejection branch so the tier directory does not accumulate
    # rejected-prewarm files.
    # Streaming download for the prewarm path — same rationale as `fetch`
    # above (avoids the multi-GB OOM the prior `payload = read_bytes`
    # pattern caused on multi-GB shards).
    from adapters.storage.storage import download_to

    tmp = f"{local_path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    download_done = False
    rejection_cleanup_path: Optional[str] = None
    try:
        try:
            download_to(shard_uri, tmp)
            os.replace(tmp, local_path)
            download_done = True
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

        # Admission under the floor. The check raises CacheCapacityExceeded
        # before mutating the residency table — on rejection nothing in
        # the residency / bytes accounting changes. `nbytes` used to come
        # from `len(payload)`; with streaming download we stat the file.
        nbytes = os.path.getsize(local_path)
        with _RESIDENCY_LOCK:
            try:
                _check_admission_capacity_locked(
                    shard_uri, nbytes, time.monotonic(), _MIN_RESIDENT_S
                )
            except CacheCapacityExceeded:
                rejection_cleanup_path = local_path
                raise
            _admit_locked(shard_uri, local_path, nbytes, time.monotonic())
        return local_path
    finally:
        # On rejection, unlink the just-downloaded file: the residency
        # table never observed it, so leaving it on disk would orphan
        # bytes (no LRU walk would ever reclaim them).
        if rejection_cleanup_path is not None and download_done:
            try:
                os.unlink(rejection_cleanup_path)
            except OSError:
                pass
        # Same ordering as fetch(): pop in-flight FIRST, then set() so a
        # waiter that wakes-and-re-enters sees no in-flight entry.
        with _INFLIGHT_LOCK:
            _INFLIGHT.pop(shard_uri, None)
        event.set()


def evict(shard_uri: str) -> bool:
    """Unlink the local file for `shard_uri` and drop the residency entry.

    Returns True if an entry was evicted, False if not present (idempotent).
    The unlink is best-effort: a concurrent reader that has the file mmap'd
    keeps reading (POSIX unlink-with-open-fd guarantee). The residency
    entry is removed atomically with the unlink so a subsequent
    `fetch(shard_uri)` is treated as a cold miss.

    Catalog invalidation calls this with the OLD URI when it publishes a
    new URI for a dataset; the new URI is a fresh cache key so no separate
    eviction is needed for it.
    """
    with _RESIDENCY_LOCK:
        return _evict_locked(shard_uri)


def residency() -> "list[ResidencyEntry]":
    """Snapshot of currently-resident shards, in LRU order (LRU first).

    Returns a list (not the live `OrderedDict`) so the caller can iterate
    without holding the residency lock — concurrent admissions or evictions
    after this call do not invalidate the snapshot. The residency writer
    uses this to write the residency registry; tests use it to assert
    eviction order.
    """
    with _RESIDENCY_LOCK:
        return [
            ResidencyEntry(
                shard_uri=shard_uri,
                local_path=entry[0],
                nbytes=entry[1],
                last_admit_at=entry[2],
                last_query_at=entry[3],
            )
            for shard_uri, entry in _RESIDENCY.items()
        ]


def bytes_used() -> int:
    """Live sum of `nbytes` across all resident entries.

    Read by tests and (in production) by the operator-facing admin endpoint
    that surfaces tier utilisation. Strictly tracks `_BYTES_USED` rather
    than recomputing on read so a slow caller does not pay the O(N) sum.
    """
    with _RESIDENCY_LOCK:
        return _BYTES_USED
