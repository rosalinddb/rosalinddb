from __future__ import annotations

"""Customer-facing `/v1/query` router.

This module hosts the v1 vector-search surface as a self-contained
`fastapi.APIRouter` so it can be mounted on more than one app:

  - `services/query_api/main.py` includes it directly (the combined
    legacy query service).
  - `services/query_api/dp_app.py` mounts it for the CP/DP-split deploy
    (private Query Data Plane).

The endpoints implement `POST /v1/query` and `GET /v1/query/status/{job_id}`
per `docs/api/v1.md`. They reuse the hot-path / ephemeral-fallback
strategy from the legacy `/query` route but reshape the response into the
v1 contract: `{matches: [{id, score, metadata}], latency_ms, mode}`.

The id/metadata bridge: FAISS `IndexIDMap2` only stores SHA1-derived int64
hashes, so a search yields int64s, not the customer's string ids. The index
builder writes a `{shard_uri}.meta.json` sidecar mapping each int64 back to
`{id, metadata}`; we load it here and translate every hit.
"""

import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import faiss  # type: ignore
import numpy as np
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from opentelemetry import context as otel_context

from adapters.landing.parquet_reader import read_shard_sidecar
from adapters.metrics.metrics import counter, timer
from adapters.observability import metrics as obs_metrics
from adapters.observability.tracing import (
    faiss_load_index_span,
    faiss_search_span,
    hot_search_span,
    list_shards_span,
    shard_download_span,
)
from adapters.queue.queue import consume, publish, ack, nack
from adapters.queue.shutdown import should_stop
from adapters.state.state import (
    RecallUnavailable,
    get_dataset,
    list_shards,
    recall_enabled,
    recall_search,
    try_consume_query,
)
# SSD-tier import. The import-time side effect is the bounded `.tmp` orphan
# sweep in `adapters/storage/shard_tier.py` (one `scandir`, log-and-continue
# on errors), cheap enough to run unconditionally. The actual `fetch` call at
# the `_ensure_cached` entry is gated behind `RB_SHARD_TIER_BYTES` so a
# deployment with the tier off does not touch any tier state at query time.
from adapters.storage import shard_tier
from services.auth.jwt_utils import current_tenant_id
from services.auth.quota import query_quota_429, quotas_enabled, rate_limit
from services.query_api import result_store

CACHE_DIR = os.getenv("CACHE_DIR", "/var/cache/shards")
DEFAULT_TOP_K = 10
MAX_TOP_K = 1000

# --- IVF nprobe -----------------------------------------------------------
# An IVF index partitions the space into `nlist` cells; `nprobe` controls how
# many cells a query searches. FAISS defaults `nprobe` to 1 — a query then
# inspects one of (typically) ~4,096 cells and misses any true neighbour that
# landed in an adjacent cell, which is why the benchmark measured recall@10 of
# only ~0.22. `nprobe` is a *query-time* knob: raising it lifts recall at a
# modest latency cost and does NOT require rebuilding the index.
#
# The default below was chosen from an nprobe sweep against SIFT 1M. Recall
# climbs steeply with nprobe then flattens — nprobe 64 reaches ~0.99 on
# IVFFlat, and query latency is effectively flat across the whole sweep (the
# FAISS search is sub-ms; HTTP/GIL dominate), so a higher nprobe costs
# nothing measurable. 64 is therefore the default: essentially full achievable
# recall at no latency penalty. Override per-deployment with `RB_QUERY_NPROBE`,
# or per query with the request body's `nprobe` — both are query-time, no
# index rebuild needed.
DEFAULT_NPROBE = 64

# `nprobe` is bounded above by `MAX_NPROBE` (mirrors the `MAX_TOP_K` pattern):
# an unbounded per-query override could ask FAISS to scan every cell, turning
# a cheap ANN search into a full scan and a latency-amplification vector.
# Above this ceiling the request is rejected (server default is clamped).
MAX_NPROBE = 1024


def query_nprobe() -> int:
    """Return the IVF `nprobe` to use for searches.

    Reads `RB_QUERY_NPROBE` live so a test or an operator can retune it
    without re-importing the module. Floored at 1, clamped at `MAX_NPROBE`.
    """
    raw = max(1, int(os.getenv("RB_QUERY_NPROBE", str(DEFAULT_NPROBE))))
    return min(raw, MAX_NPROBE)


def _ivf_search_params(index, override: Optional[int] = None, full_coverage: bool = False):
    """Return `(search_params, nprobe)` for an IVF search, or `(None, nprobe)`.

    `override` (a per-request `nprobe`, if supplied) wins over the server
    default `query_nprobe()`; the effective value is clamped to `MAX_NPROBE`.

    `full_coverage`, when True, ignores `override`/the server default and
    sets `nprobe = nlist` — every IVF cell is scanned, turning the IVF search
    into an exhaustive (exact) one. This is what the filtered hot path needs:
    a metadata-matching vector sitting in an unprobed cell would otherwise be
    invisible no matter how large `fetch_k` is. `nlist` is read from the IVF
    index itself; the resolved value is NOT clamped to `MAX_NPROBE` because a
    full scan is an explicit, internal decision (not an attacker-controllable
    per-query knob).

    Crucially, the resolved `nprobe` is delivered via a *per-search*
    `faiss.SearchParametersIVF` object — passed to `index.search(...)` for
    this call only — instead of mutating the shared cached index's
    `ivf.nprobe`. Mutating the shared object races: FAISS releases the GIL
    inside `search()`, so a concurrent query could overwrite `nprobe` between
    one query setting it and its own search starting (recall jitter). Per-call
    `SearchParameters` isolates each query's `nprobe` to that call.

    A flat (non-IVF) index has no `nprobe` — `faiss.extract_index_ivf` raises
    for it, so `search_params` is `None` (a no-op) and the resolved value is
    still returned for the span attribute. A flat index is already exact, so
    `full_coverage` is a no-op for it; in that case the resolved value is the
    sentinel `0` so the `rosalinddb.nprobe` span attribute does not
    misrepresent an inherently-exhaustive flat scan as a bounded IVF probe.
    """
    raw = max(1, int(override)) if override is not None else query_nprobe()
    nprobe = min(raw, MAX_NPROBE)
    try:
        ivf = faiss.extract_index_ivf(index)
    except Exception:  # noqa: BLE001 - flat index / not an IVF index
        # A flat index has no `nprobe` concept and is exhaustive by nature;
        # under `full_coverage` report `0` rather than an IVF-shaped value.
        return None, (0 if full_coverage else nprobe)
    if full_coverage:
        # Scan every cell — `nprobe = nlist` makes the IVF search exhaustive.
        nprobe = int(ivf.nlist)
    return faiss.SearchParametersIVF(nprobe=nprobe), nprobe

# Ephemeral result store, keyed by correlation_id. The `RESULT_READY` consumer
# thread writes results here; `GET /v1/query/status/{job_id}` reads them.
#
# Multi-worker safety: the store lives in `result_store`, which uses a SHARED
# Redis store (`query_result:{job_id}` with a TTL) when `REDIS_URL` is set so a
# result written by the consumer in one `query_api` replica is visible to a
# status poll that lands on any other replica. With no `REDIS_URL` (unit tests
# / single process) it falls back to an in-process dict.
#
# `_RESULTS` / `_RESULTS_LOCK` are re-exported as aliases of the `result_store`
# in-process fallback so existing test fixtures that reset the store via
# `v1_query._RESULTS.clear()` keep working unchanged.
_RESULTS: Dict[str, dict] = result_store._RESULTS
_RESULTS_LOCK = result_store._RESULTS_LOCK

# --- In-memory index cache ------------------------------------------------
# `_hot_search` used to call `faiss.read_index()` (deserialise a ~13 MB index)
# and `read_shard_sidecar()` (parse a multi-MB JSON) on EVERY query — the
# `mode:"hot"` label was cosmetic, nothing was actually cached.
#
# `_SHARD_CACHE` is a genuine in-memory cache keyed by `shard_id`, holding the
# *deserialised* FAISS index and the *parsed* sidecar. The first query for a
# shard does a real cold load; every subsequent query reuses the in-memory
# objects, so the query collapses to just the FAISS search.
#
#   - Byte-budgeted LRU: an entry's footprint varies ~100x across datasets
#     (a 1k-vector shard's index+sidecar vs a 1M-vector shard's ~430 MB), so
#     a count cap (the old `RB_SHARD_CACHE_SIZE`) cannot bound memory — four
#     large shards co-resident would OOM the node. Instead each entry's
#     approximate footprint (index + parsed sidecar) is measured once at
#     insert time and a running total is kept; on insert the LRU end is
#     evicted until `total <= RB_SHARD_CACHE_BYTES`. An `OrderedDict` is the
#     LRU — a hit moves the key to the end.
#     `RB_SHARD_CACHE_SIZE` remains an optional *secondary* safety cap.
#   - Oversized-shard bypass (the 1M-scale correctness fix): a single shard
#     whose measured footprint alone EXCEEDS the whole budget can never
#     coexist with anything else. Inserting it as MRU and then running the
#     eviction loop would evict every warm neighbour first and finally the
#     oversized entry too — wiping the cache on *every* query for that shard
#     (pathological evict-then-reinsert thrash). So `_cache_put` detects the
#     oversize case up front and BYPASSES the cache entirely: the entry is
#     never inserted, no neighbour is evicted, and the current query still
#     searches the in-hand index/sidecar held in the call site's locals. The
#     bypass emits `record_shard_cache("oversize")` + a one-time-per-shard
#     WARNING so an under-provisioned operator can see they need a bigger
#     `RB_SHARD_CACHE_BYTES`.
#   - Evicted in step with the shard sweep: when the rough-edges sweeper
#     deletes a superseded shard, `evict_shard()` drops its cache entry so a
#     stale index is never served. Resolving the *newest* shard per query
#     means a freshly-built shard is a natural cache miss → cold load.
#   - Thread-guarded: `_hot_search` runs concurrently under the GIL; every
#     read/mutate of the cache holds `_SHARD_CACHE_LOCK`.
# `RB_SHARD_CACHE_BYTES` — the in-memory shard-cache byte budget; the primary
# operator knob for this cache. Default 1 GiB.
#
# Default rationale: the cache must be able to hold at least ONE large shard,
# otherwise that shard is bypassed on every query (a permanent cold load) and
# the cache provides zero value at the scale where it matters most. A 1M-vector
# flat-float32 shard is dim x 4B x 1e6: ~512 MB at 128 dims, ~3 GB at 768 dims.
# The previous 512 MB default could not hold even a single 128-dim 1M shard
# once the sidecar is accounted for — so at 1M scale it degenerated into the
# oversize-bypass path for the common case. 1 GiB comfortably holds a 128-dim
# 1M shard (the most common large config) with headroom for the sidecar and a
# warm neighbour or two, while staying small enough not to dominate RAM on a
# modest node (a 4 GB pod still leaves ~3 GB for everything else). Deployers
# running 768-dim-at-1M (or wanting more warm shards) raise this — the private
# scale bench, for instance, overrides it to 2 GB. Deployers on tiny nodes can
# lower it; an oversized shard is then served via the graceful bypass rather
# than thrashing the cache.
RB_SHARD_CACHE_BYTES = max(1, int(os.getenv("RB_SHARD_CACHE_BYTES", str(1024 * 1024 * 1024))))
# Optional secondary count cap. 0 (or unset → 0) disables it; the byte budget
# is the primary bound.
RB_SHARD_CACHE_SIZE = max(0, int(os.getenv("RB_SHARD_CACHE_SIZE", "0")))

# Shard ids already warned about for exceeding the cache budget. Keeps the
# oversize WARNING "one-time-ish" — logged once per shard id per process so an
# under-provisioned operator gets a loud signal without a per-query log storm.
# Kept strictly bounded: `evict_shard()` discards a swept shard's id and
# `cache_clear()` empties the set, so it never accumulates ids for shards that
# are gone and a re-added shard can warn again.
_OVERSIZE_WARNED: set = set()


def _truthy(value: Optional[str]) -> bool:
    """Mirror `adapters.observability.otel._truthy` (kept local to avoid a
    cross-package import for one line of env parsing — the duplicate is
    tiny, stable, and keeps this module's import graph slim).
    """
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _read_major_faults(stat_path: str = "/proc/self/stat") -> Optional[int]:
    """Return the process's cumulative major-fault count, or None if unavailable.

    Field 12 in `/proc/self/stat` is `majflt` per `man 5 proc`. The format
    treats fields as whitespace-separated tokens, but `comm` (field 2) can
    contain spaces AND a literal `)`, so the safe parse is "find the LAST
    `)` and split from there." After dropping pid (field 1) and the
    parenthesised comm (field 2), `majflt` is the 10th token (index 9) of
    the trailing slice — fields 3..12 inclusive.

    This is the input to the page-fault sampler that wraps `faiss.search` in
    `_hot_search`. Best-effort: returns `None`
    on any error (file missing on macOS dev, format surprise, non-int at the
    expected offset) so the page-fault counter never raises into the query
    path. On the macOS dev cycle this would otherwise raise FileNotFoundError
    twice per query; `_STAT_PATH_AVAILABLE` short-circuits on the first miss
    so the failure mode is a single failed open per process, not per query.
    """
    global _STAT_PATH_AVAILABLE
    if _STAT_PATH_AVAILABLE is False and stat_path == "/proc/self/stat":
        return None
    try:
        with open(stat_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except (FileNotFoundError, PermissionError, OSError):
        if stat_path == "/proc/self/stat":
            _STAT_PATH_AVAILABLE = False
        return None
    try:
        # `rpartition(")")` skips `pid + comm` together; comm may itself
        # contain `)` characters so split from the LAST occurrence.
        head, _, tail = raw.rpartition(")")
        if not head:
            return None
        after = tail.split()
        # After pid + comm, majflt is the 10th token (index 9).
        return int(after[9])
    except (IndexError, ValueError):
        return None


# Tri-state: None (unprobed), True (open succeeded once), False (open failed
# once on the default path; future calls short-circuit). Per-process sentinel
# so the macOS dev cycle doesn't pay FileNotFoundError churn on every query.
# The test-supplied `stat_path` override bypasses the sentinel.
_STAT_PATH_AVAILABLE: Optional[bool] = None


# `RB_FAISS_MMAP` — mmap toggle. When True, the hot path opens a shard's
# FAISS index with `IO_FLAG_MMAP | IO_FLAG_READ_ONLY` instead of deserialising
# it fully into RSS. The byte-budgeted cache then accounts for an mmap'd entry
# with `_MMAP_INDEX_ESTIMATE_BYTES` (see `_index_nbytes`) because file-backed
# pages are NOT resident RSS — counting them against the budget would force
# premature evictions of genuinely-resident neighbours.
#
# Captured ONCE at module import on purpose: a mid-run flip would require every
# call site to re-read the flag and would force tests into intrusive monkeypatch
# dances. Tests that need to flip the flag use `importlib.reload(v1_query)`
# (see `tests/unit/test_mmap_flag.py` and `tests/unit/test_dp_io_offload.py`
# for the pattern).
_MMAP_ENABLED = _truthy(os.getenv("RB_FAISS_MMAP"))

# Fixed estimate charged to the byte-budgeted cache for an mmap'd FAISS entry.
# The mmap'd index's working-set is governed by the OS page cache, not the
# Python process's RSS, so `serialize_index(...).nbytes` (which would deserialise
# the entire index to measure it — defeating the point of mmap) is the wrong
# accounting. 32 MiB is large enough that an unbounded number of mmap'd entries
# still pressures the cache toward eviction; small enough that the default
# `RB_SHARD_CACHE_BYTES` budget holds many warm shards. Documented in
# `docs/architecture/mmap.md`.
_MMAP_INDEX_ESTIMATE_BYTES = 32 * 1024 * 1024


# Filesystem types that are unsafe to mmap a FAISS index from. mmap on a FUSE-
# backed object-store mount (s3fs, goofys, mountpoint-s3, …) is a foot-gun:
# each page fault becomes a synchronous range-GET against S3, with no kernel
# read-ahead, and a sufficiently large shard turns a single query into a long
# tail of object-store round trips. The guard is advisory — we log a WARNING
# and let the operator decide — because forcing a hard refusal would block
# legitimate testing setups (e.g. a tmpfs-backed FUSE for fault injection).
_UNSAFE_MMAP_FSTYPES = frozenset(
    {
        "fuse",
        "fuse3",
        "fuse.s3fs",
        "fuse.goofys",
        "fuse.mountpoint-s3",
        "fuseblk",
        "s3fs",
    }
)


def _is_fuse_mount(path: str, mountinfo_path: str = "/proc/self/mountinfo") -> bool:
    """Return True when `path` resolves onto a FUSE-style filesystem.

    The decision uses `/proc/self/mountinfo` directly so the helper has no
    `subprocess` / external-tool dependency and can be unit-tested with a fake
    mountinfo file. Strategy: parse every entry, pick the one whose
    `mount_point` is the longest prefix of `path` (the kernel's own resolution
    rule), and check whether that entry's filesystem type is in
    `_UNSAFE_MMAP_FSTYPES`. Returns False (silently) when `mountinfo_path`
    does not exist — that is the macOS-dev path and must not crash import.

    `mountinfo` line shape (man 5 proc):
      mount_id parent_id major:minor root mount_point opts - fs_type src super
    Fields are space-separated; the literal `-` marks the boundary between the
    optional-fields list and the (fs_type, source, super_opts) triple.
    """
    try:
        with open(mountinfo_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (FileNotFoundError, PermissionError, OSError):
        # macOS / restricted container — guardrail is Linux-only.
        return False

    best_mount = ""
    best_fs = ""
    # Normalise the candidate so a trailing slash does not break prefix matching.
    candidate = os.path.abspath(path)
    for raw in lines:
        parts = raw.split()
        try:
            sep = parts.index("-")
        except ValueError:
            continue
        # mountinfo lines have at least 6 fixed fields before the `-` (man 5
        # proc); shorter is malformed.
        if sep < 6 or len(parts) <= sep + 1:
            continue
        mount_point = parts[4]
        fs_type = parts[sep + 1]
        # The kernel matches the LONGEST mount point that is a path prefix of
        # the candidate; mimic that so a nested FUSE mount under an ext4 root
        # is correctly attributed to the FUSE entry. The `/`-mount prefix is
        # already covered by the startswith check (every absolute path starts
        # with "/").
        if candidate == mount_point or candidate.startswith(
            mount_point.rstrip("/") + "/"
        ):
            if len(mount_point) >= len(best_mount):
                best_mount = mount_point
                best_fs = fs_type

    return best_fs in _UNSAFE_MMAP_FSTYPES


def _maybe_warn_about_fuse_cache_dir(
    cache_dir: str, mmap_enabled: bool = _MMAP_ENABLED
) -> bool:
    """Emit the FUSE-cache-dir warning if mmap is on and the cache is FUSE.

    Returns True iff a warning was emitted (so tests can assert against the
    log line without round-tripping through `importlib.reload`). Extracted
    from the import-time guard purely for testability — the production
    behaviour is unchanged.
    """
    if mmap_enabled and _is_fuse_mount(cache_dir):
        logging.getLogger(__name__).warning(
            "RB_FAISS_MMAP is enabled but CACHE_DIR=%s is on a FUSE filesystem; "
            "mmap'd FAISS reads will translate to synchronous object-store GETs "
            "per page fault. Consider a local SSD/ext4/tmpfs cache dir.",
            cache_dir,
        )
        return True
    return False


# Import-time advisory check. The warning fires once per process start so an
# operator who runs `RB_FAISS_MMAP=true` against an s3fs-backed `CACHE_DIR`
# sees a loud log line in the startup banner — the mmap path itself is still
# attempted, but the operator is forewarned about the fault-storm risk.
_maybe_warn_about_fuse_cache_dir(CACHE_DIR)

# --- Per-`(tenant, dataset)` catalog cache -----------------------------------
#
# `list_shards(tenant, dataset)` is on the hot path: every query path resolves
# "newest shard" by calling it. In Postgres mode that is a SELECT round-trip
# per query — fine at low QPS, a per-query cost we can avoid at high QPS by
# remembering the answer for `RB_CATALOG_FRESHNESS_S` seconds and refreshing
# it on a NOTIFY from `add_shard`.
#
# Two activation gates, both default-off so the rollback is identical to
# today's behaviour:
#
#   - `RB_SHARD_TIER_BYTES` (the SSD tier toggle) gates the cache itself.
#     Without the tier the legacy single-flight download already caches the
#     bytes on disk, and a fresh PG lookup per query is cheap relative to
#     the work `_hot_search` already does — the cache adds no value and
#     risks staleness if the deployment isn't ready for the eventual
#     consistency window. With the tier on, the cache is a meaningful
#     latency win and the freshness contract is documented.
#   - `RB_CATALOG_FRESHNESS_S` (TTL in seconds, default 5). Setting this to
#     `0` disables the cache entirely — operator emergency knob.
#
# The cache is keyed by `(tenant, dataset)` and bounded by a fixed-size LRU
# (`_CATALOG_CACHE_MAX_ENTRIES`) to cap memory growth in a process serving
# many tenants. Entries are tiny (a list of dicts from the catalog row), so
# the cap is high enough that hot tenants will never evict each other.
#
# Invalidation has two channels:
#   - NOTIFY push via `services._common.catalog_listener` when
#     `RB_CATALOG_LISTEN=true`. On a notify, `_on_catalog_notify` evicts the
#     affected entry.
#   - TTL pull: every read past `RB_CATALOG_FRESHNESS_S` re-queries
#     `list_shards` and refreshes the entry.
# Without LISTEN the TTL is the only channel; with LISTEN the TTL is the
# safety net for missed notifies (the LISTEN protocol is best-effort).

# Hardcoded LRU bound. `_CATALOG_CACHE_MAX_ENTRIES` datasets is generous —
# even a self-host with 10k distinct tenants under one dataset each fits.
# This can be promoted to an env knob if a deployment grows past it;
# today the env surface is intentionally minimal.
_CATALOG_CACHE_MAX_ENTRIES = 10_000

_CATALOG_CACHE: "OrderedDict[tuple[str, str], tuple[float, list]]" = OrderedDict()
_CATALOG_CACHE_LOCK = threading.Lock()

# Per-key generation counter. Bumped on every invalidation; the cache
# writer compares pre-fetch and post-fetch generations under the lock and
# refuses to install rows older than a concurrent invalidate.
#
# Closes the otherwise-real "race against concurrent invalidate" window:
# a reader sees a miss, drops the lock, fetches `list_shards`, and is
# about to install — meanwhile a NOTIFY-driven invalidate fires. Without
# the generation check the just-fetched-but-already-stale rows would be
# installed and live until the next TTL expiry; with it, the install is
# silently skipped and the next caller re-fetches. The staleness window
# was already bounded by `RB_CATALOG_FRESHNESS_S` (so this is a
# tightening, not a correctness fix), but the generation check is cheap
# and removes the bound entirely for invalidate-driven flows.
_CATALOG_CACHE_GEN: "Dict[tuple[str, str], int]" = {}


def _now() -> float:
    """Monotonic time source for the cache TTL.

    Indirected so tests can advance virtual time with a single monkeypatch
    instead of patching `time.monotonic` globally (which other modules
    also call). Use `monotonic`, not `time.time`, so a clock step (NTP
    correction) does not retroactively expire or extend entries.
    """
    return time.monotonic()


def _catalog_freshness_s() -> float:
    """Read `RB_CATALOG_FRESHNESS_S` live so an operator can retune at runtime.

    Default 5 s — short enough that a missed NOTIFY heals quickly, long
    enough that bursty same-dataset traffic shares cache hits. `0`
    disables the cache.
    """
    try:
        return max(0.0, float(os.getenv("RB_CATALOG_FRESHNESS_S", "5")))
    except ValueError:
        return 5.0


def _catalog_cache_active() -> bool:
    """Cache is active iff the SSD tier is on AND TTL is positive.

    Both gates default-off-equivalent: with the tier off OR TTL=0 the
    wrapper must defer every call to `list_shards`.
    """
    return bool(os.getenv("RB_SHARD_TIER_BYTES")) and _catalog_freshness_s() > 0


def _cached_list_shards(tenant: str, dataset: str) -> list:
    """Return `list_shards(tenant, dataset)`, cached for `RB_CATALOG_FRESHNESS_S`.

    When the cache is inactive (see `_catalog_cache_active`), defers
    every call to the source. When active, returns a fresh-enough cached
    list or re-fetches on expiry. The cached list is returned BY
    REFERENCE — callers must not mutate it (the catalog rows are dicts
    that the existing code already treats as read-only inside the
    query path).

    Single-flight is NOT a contract: N concurrent callers on a cold
    miss can each call the source. `list_shards` is cheap and TTL is
    short, so the worst case (a thundering herd at startup) is bounded;
    if it ever shows up in traces, a per-key in-flight set is the fix.
    """
    if not _catalog_cache_active():
        return list_shards(tenant, dataset)
    key = (tenant, dataset)
    ttl = _catalog_freshness_s()
    now = _now()
    with _CATALOG_CACHE_LOCK:
        entry = _CATALOG_CACHE.get(key)
        if entry is not None and (now - entry[0]) < ttl:
            # LRU bump — move the warm entry to the most-recent end so
            # the bounded-cap eviction prefers cold entries.
            _CATALOG_CACHE.move_to_end(key)
            return entry[1]
        # Capture generation under the lock so a concurrent invalidate
        # AFTER this snapshot bumps the counter and our install loses.
        gen_pre = _CATALOG_CACHE_GEN.get(key, 0)
    # Miss / expired — fetch outside the lock so a slow Postgres does
    # not serialise other readers.
    rows = list_shards(tenant, dataset)
    with _CATALOG_CACHE_LOCK:
        # Generation check: if an invalidate fired during the fetch, the
        # counter has moved past our snapshot. Skip the install — the
        # rows we just fetched are no fresher than the invalidate, and a
        # future caller will re-fetch.
        if _CATALOG_CACHE_GEN.get(key, 0) != gen_pre:
            return rows
        _CATALOG_CACHE[key] = (now, rows)
        _CATALOG_CACHE.move_to_end(key)
        # Bounded-cap eviction: drop the oldest entries until we are
        # under the cap. The just-inserted entry is the MRU so it is
        # never the one evicted in the same call.
        while len(_CATALOG_CACHE) > _CATALOG_CACHE_MAX_ENTRIES:
            _CATALOG_CACHE.popitem(last=False)
    return rows


def _invalidate_catalog_cache(tenant: str, dataset: str) -> bool:
    """Drop the cached entry for `(tenant, dataset)`. Idempotent.

    Bumps the per-key generation counter so a `_cached_list_shards`
    call mid-flight (already past the miss-check, fetching rows from
    the source) refuses to install the now-stale rows.

    Called by the NOTIFY handler and exposed for test/operator use.
    Returns True iff an entry was actually removed.
    """
    key = (tenant, dataset)
    with _CATALOG_CACHE_LOCK:
        _CATALOG_CACHE_GEN[key] = _CATALOG_CACHE_GEN.get(key, 0) + 1
        return _CATALOG_CACHE.pop(key, None) is not None


def _on_catalog_notify(payload: dict) -> None:
    """Catalog listener subscriber — evict the affected cache entry.

    Defensive: the payload SHAPE is the contract with `state.add_shard`,
    not a deeper schema check. A missing/empty tenant or dataset is
    treated as "skip" (we cannot route the invalidation) rather than
    flushing the whole cache.
    """
    tenant = payload.get("tenant") or ""
    dataset = payload.get("dataset") or ""
    if tenant and dataset:
        _invalidate_catalog_cache(tenant, dataset)


def _catalog_cache_clear() -> None:
    """Test helper — drop every cached `(tenant, dataset)` entry."""
    with _CATALOG_CACHE_LOCK:
        _CATALOG_CACHE.clear()
        _CATALOG_CACHE_GEN.clear()


# Activate the LISTEN-driven push channel when the env opts in. The
# subscribe call spawns the listener thread (idempotent if another
# subscriber already started it). Default-off: with `RB_CATALOG_LISTEN`
# unset, no thread starts and the TTL pull is the only invalidation
# channel. The import is local so a process that never opts in does not
# even import psycopg2's LISTEN scaffolding.
if _truthy(os.getenv("RB_CATALOG_LISTEN")):
    from services._common import catalog_listener as _catalog_listener

    _catalog_listener.subscribe(_on_catalog_notify)

# Entry value is `(index, sidecar, nbytes)`.
_SHARD_CACHE: "OrderedDict[Any, tuple]" = OrderedDict()
_SHARD_CACHE_LOCK = threading.RLock()
_SHARD_CACHE_BYTES_USED = 0

# Per-URI single-flight download coordination. When N queries simultaneously
# request an uncached shard, only one thread initiates the GET against the
# object store; the others block on the per-URI Event and return the same
# local path once the download completes. Without this, a concurrent burst
# against a multi-GB shard fans out into N parallel GETs of the same object
# which the object store throttles (MinIO returns TooManyRequests; S3 throttles
# similarly), and the system fails the cold-warmup entirely — observed in a
# concurrent-VU bench: 10 VUs against a 6 GB shard produced 10 parallel GETs,
# MinIO threw `TooManyRequests`, and the DP surfaced `storage_unavailable`
# (504) on every request. The cache never warmed.
#
# The lock is intentionally per-URI: independent shards must download in
# parallel, otherwise warming a cold cache after a deploy serialises every
# distinct shard behind one global mutex.
#
# `_INFLIGHT_DOWNLOADS` size is bounded in practice by
# `(concurrent requests) x (distinct uncached URIs)`. Entries are popped in
# the initiator's finally block, so the dict never grows past the in-flight
# set — there is no accumulating leak. A pathological burst (e.g. thousands
# of distinct uncached URIs requested simultaneously) would transiently grow
# the dict to that burst's cardinality; each entry is a short string key plus
# a `threading.Event` (small), so the bound is real-time memory, not a slow
# leak, and the entries drain as their downloads complete.
#
# NOTE: `services/ephemeral_runner/run.py` has a duplicate `_ensure_cached`
# WITHOUT this coalescing — that runner deliberately avoids importing from
# the query_api package (circular import) and the same classifier-duplication
# rationale applies. The ephemeral path runs at much lower concurrency so the
# stampede has not been observed there; a future refactor may want to port the
# fix or extract a shared `adapters/storage/shard_fetch.py` module.
_INFLIGHT_DOWNLOADS: Dict[str, threading.Event] = {}
_INFLIGHT_DOWNLOADS_LOCK = threading.Lock()

# Bounded wait — a hung initiator should fail the waiters with a clear error
# rather than block them indefinitely. Tunable via env for the unusual case
# where a single shard download legitimately takes longer than the default
# (e.g. a multi-GB shard on a slow link). Default 300 s comfortably covers a
# multi-GB GET on typical infra while bounding the worst-case waiter stall.
_DOWNLOAD_COALESCE_WAIT_S = float(os.getenv("RB_DOWNLOAD_COALESCE_WAIT_S", "300"))


class DownloadCoalescingTimeout(RuntimeError):
    """A coalesced waiter exceeded its deadline on someone else's download.

    Raised by `_ensure_cached` when the per-URI in-flight event was not set
    within `_DOWNLOAD_COALESCE_WAIT_S` seconds. Surfaced as a distinct
    exception (rather than a generic `TimeoutError`) so the caller can map it
    to a specific error code / observability signal — a waiter timing out is
    a different operational condition from the initiator's download itself
    failing.
    """


def _sidecar_nbytes(sidecar) -> int:
    """Approximate the in-memory footprint of a parsed sidecar.

    The sidecar is a JSON-derived dict; its serialised length is a cheap,
    stable proxy for memory footprint (the parsed dict is larger, but the
    ratio is roughly constant, so it is fine as a relative budget unit).
    Falls back to `sys.getsizeof` if the object is not JSON-serialisable.
    """
    try:
        return len(json.dumps(sidecar))
    except Exception:  # noqa: BLE001 - non-serialisable / exotic value
        return sys.getsizeof(sidecar)


def _index_nbytes(index) -> int:
    """Approximate the in-memory footprint of a FAISS index in bytes.

    When `_MMAP_ENABLED` is True the index is file-backed: its pages live in
    the page cache, not RSS, and the kernel evicts them under memory pressure
    without our cache having to do anything. Charging the full serialised
    size against `RB_SHARD_CACHE_BYTES` would then double-count and force
    premature eviction of genuinely-resident neighbours. We return the
    `_MMAP_INDEX_ESTIMATE_BYTES` constant so an mmap'd entry still counts for
    *something* — covering the FAISS metadata, IVF coarse quantiser, sidecar
    references, etc. that ARE resident — without dominating the budget.
    """
    if _MMAP_ENABLED:
        return _MMAP_INDEX_ESTIMATE_BYTES
    try:
        return int(faiss.serialize_index(index).nbytes)
    except Exception:  # noqa: BLE001 - serialise unsupported for this type
        ntotal = int(getattr(index, "ntotal", 0))
        dim = int(getattr(index, "d", 0))
        # 4 bytes per code is the flat-float32 worst case; a coarse estimate.
        return ntotal * dim * 4


def _entry_nbytes(index, sidecar) -> int:
    """Approximate the total footprint of a cache entry (index + sidecar)."""
    return _index_nbytes(index) + _sidecar_nbytes(sidecar)


def _cache_get(shard_id) -> Optional[tuple]:
    """Return the cached `(faiss_index, sidecar)` for `shard_id`, or None.

    A hit refreshes LRU recency (moves the key to the most-recent end).
    """
    with _SHARD_CACHE_LOCK:
        entry = _SHARD_CACHE.get(shard_id)
        if entry is None:
            return None
        _SHARD_CACHE.move_to_end(shard_id)
        index, sidecar, _nbytes = entry
        return index, sidecar


def _cache_put(shard_id, index, sidecar) -> None:
    """Insert `(index, sidecar)` for `shard_id`, byte-budgeting the cache.

    The entry's footprint is measured once here. On insert, LRU entries are
    evicted until the running total fits `RB_SHARD_CACHE_BYTES` (and, if
    enabled, the secondary count cap `RB_SHARD_CACHE_SIZE`).

    Oversized-shard bypass: a shard whose footprint alone exceeds the whole
    budget can never coexist with any other entry. Inserting it would force the
    eviction loop to drain every warm neighbour before discarding the oversized
    entry itself — thrashing the cache on every query for that shard. Instead
    we detect that case up front and return WITHOUT touching the cache: the
    entry is never inserted, no neighbour is evicted, and the caller's in-hand
    `index`/`sidecar` still serve the current query. The bypass is recorded as
    `record_shard_cache("oversize")` and warned about once per shard id so an
    under-provisioned operator sees they should raise `RB_SHARD_CACHE_BYTES`.
    """
    global _SHARD_CACHE_BYTES_USED
    nbytes = _entry_nbytes(index, sidecar)

    # Oversized-shard graceful bypass — never insert, never evict neighbours.
    if nbytes > RB_SHARD_CACHE_BYTES:
        with _SHARD_CACHE_LOCK:
            # Drop any prior (smaller) entry for this id so a stale index is
            # never served, but do NOT evict anyone else.
            old = _SHARD_CACHE.pop(shard_id, None)
            if old is not None:
                _SHARD_CACHE_BYTES_USED -= old[2]
            first_time = shard_id not in _OVERSIZE_WARNED
            if first_time:
                _OVERSIZE_WARNED.add(shard_id)
        # Signal + warn outside the cache lock (logging/metrics must not run
        # under the hot-path lock).
        try:
            obs_metrics.record_shard_cache("oversize")
        except Exception:  # noqa: BLE001 - metrics must never break the query path
            pass
        if first_time:
            logging.getLogger(__name__).warning(
                "shard %s footprint (~%d bytes) exceeds the whole shard-cache "
                "budget RB_SHARD_CACHE_BYTES=%d; serving it BYPASSED (not "
                "cached) to avoid evicting the warm cache. Raise "
                "RB_SHARD_CACHE_BYTES to cache shards this large.",
                shard_id,
                nbytes,
                RB_SHARD_CACHE_BYTES,
            )
        return

    with _SHARD_CACHE_LOCK:
        # Replacing an existing key: drop its old footprint first.
        old = _SHARD_CACHE.pop(shard_id, None)
        if old is not None:
            _SHARD_CACHE_BYTES_USED -= old[2]
        _SHARD_CACHE[shard_id] = (index, sidecar, nbytes)
        _SHARD_CACHE.move_to_end(shard_id)
        _SHARD_CACHE_BYTES_USED += nbytes
        # Evict LRU entries until within the byte budget. The just-inserted
        # entry is the MRU, so it is evicted last; the oversize case is handled
        # above, so this loop only ever trims genuinely-evictable neighbours.
        while _SHARD_CACHE_BYTES_USED > RB_SHARD_CACHE_BYTES and _SHARD_CACHE:
            _evicted_id, evicted = _SHARD_CACHE.popitem(last=False)
            _SHARD_CACHE_BYTES_USED -= evicted[2]
        # Optional secondary count cap (0 = disabled).
        if RB_SHARD_CACHE_SIZE > 0:
            while len(_SHARD_CACHE) > RB_SHARD_CACHE_SIZE:
                _evicted_id, evicted = _SHARD_CACHE.popitem(last=False)
                _SHARD_CACHE_BYTES_USED -= evicted[2]


def evict_shard(shard_id) -> bool:
    """Drop a shard's cached index/sidecar — called when a shard is swept.

    Returns True if an entry was actually removed. Idempotent: evicting an
    uncached shard is a no-op. The index builder's superseded-shard sweeper
    calls this so a stale index can never be served after its shard is gone.
    """
    global _SHARD_CACHE_BYTES_USED
    with _SHARD_CACHE_LOCK:
        entry = _SHARD_CACHE.pop(shard_id, None)
        # Drop any oversize-warned mark so a re-added shard can warn again and
        # the warned set cannot grow without bound as shards come and go.
        _OVERSIZE_WARNED.discard(shard_id)
        if entry is None:
            return False
        _SHARD_CACHE_BYTES_USED -= entry[2]
        return True


def cache_clear() -> None:
    """Drop the entire shard cache (test helper / process reset)."""
    global _SHARD_CACHE_BYTES_USED
    with _SHARD_CACHE_LOCK:
        _SHARD_CACHE.clear()
        _SHARD_CACHE_BYTES_USED = 0
        # Keep the oversize-warned set bounded: a full reset clears it too.
        _OVERSIZE_WARNED.clear()


router = APIRouter()


def _err(status_code: int, code: str, message: str, details: Optional[dict] = None) -> JSONResponse:
    """Build a v1 error envelope response."""
    body: dict = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=body)


def _ensure_cached(shard_uri: str) -> str:
    """Ensure a shard is present in the local cache and return its path.

    FAISS's `read_index` only accepts a filesystem path, so an object-store
    shard (`s3://` or `memory://`) is fetched once into `CACHE_DIR`. There is
    no `file://` branch: RosalindDB is object-storage-first and `memory://`
    is the unit-test backend.

    Concurrent callers for the same URI are coalesced into a single download
    via `_INFLIGHT_DOWNLOADS` — see the module-level comment on that dict
    for the why. Briefly: without coalescing, N concurrent queries against a
    cold shard issue N parallel GETs, the object store throttles, and the
    cold warm-up fails for everyone. With coalescing, exactly one thread
    downloads and the rest block on a per-URI `Event` until the rename
    publishes the file.

    SSD-tier activation gate. When `RB_SHARD_TIER_BYTES` is set in the
    environment, delegation runs through
    `adapters.storage.shard_tier.fetch(shard_uri)` — the same single-flight
    contract, but the local file is owned by the tier (which runs its own
    byte-budgeted LRU eviction). The env check happens here rather than at
    import time so a flip-flop env var across pod restarts cleanly toggles
    the path without needing a code redeploy. When the env is unset, the
    legacy single-flight body below runs unchanged — that is the rollback
    contract.
    """
    if os.getenv("RB_SHARD_TIER_BYTES"):
        # Tier handles its own directory creation, single-flight, and
        # eviction. `ShardTierTimeout` and `FileNotFoundError` are the two
        # raise paths the caller's classifier already maps to 503; let them
        # propagate untouched.
        return shard_tier.fetch(shard_uri)

    os.makedirs(CACHE_DIR, exist_ok=True)
    if not (shard_uri.startswith("s3://") or shard_uri.startswith("memory://")):
        raise ValueError("Unsupported shard uri")

    from adapters.storage.storage import read_bytes

    cache_key = shard_uri.split("://", 1)[1].replace("/", "_")
    path = os.path.join(CACHE_DIR, cache_key)

    # Fast path: file already on disk. No coordination needed — the file is
    # already atomic (written via temp + rename) so a concurrent reader sees
    # either nothing or the complete bytes. Skip the lock/event dance
    # entirely so the warm-cache case stays byte-for-byte unchanged.
    if os.path.exists(path):
        return path

    # Single-flight registration. Atomically decide whether this thread is
    # the *initiator* (creates the entry, will do the GET) or a *waiter*
    # (found a pre-existing entry, will block on its Event).
    is_initiator = False
    with _INFLIGHT_DOWNLOADS_LOCK:
        # A second `exists` check inside the lock closes a race where the
        # initiator finished the download AND cleared the in-flight entry
        # between our outer `exists` check and acquiring the lock. Without
        # this, we'd needlessly become a fresh initiator for an already-
        # cached shard.
        if os.path.exists(path):
            return path
        event = _INFLIGHT_DOWNLOADS.get(shard_uri)
        if event is None:
            event = threading.Event()
            _INFLIGHT_DOWNLOADS[shard_uri] = event
            is_initiator = True

    if not is_initiator:
        # Waiter path. Block on the initiator's event with a bounded timeout
        # so a wedged initiator cannot stall callers indefinitely. On a clean
        # set(), re-check that the file actually materialised — the initiator
        # may have failed, in which case the event fires but the file is
        # absent and we surface that to the caller via the classifier as
        # storage_unavailable (503), the right retry hint for "another caller
        # tried, you should try again."
        logging.getLogger(__name__).debug(
            "coalesced waiter on %s", shard_uri,
        )
        completed = event.wait(_DOWNLOAD_COALESCE_WAIT_S)
        if not completed:
            logging.getLogger(__name__).warning(
                "download coalescing timeout after %.1fs on %s",
                _DOWNLOAD_COALESCE_WAIT_S, shard_uri,
            )
            raise DownloadCoalescingTimeout(
                f"timed out after {_DOWNLOAD_COALESCE_WAIT_S}s waiting for "
                f"an in-flight download of {shard_uri}"
            )
        if not os.path.exists(path):
            # Initiator's download must have failed; the file is not there.
            # Raise FileNotFoundError (not bare RuntimeError) so the existing
            # classifier branch routes this to storage_unavailable / 503
            # instead of the catch-all ephemeral_error / 500. Customers
            # retrying on 503 get the right transient-failure semantics.
            raise FileNotFoundError(
                f"coalesced download of {shard_uri} did not produce a local file "
                "(initiator likely failed); caller should retry"
            )
        return path

    # Initiator path. Do the actual download, then publish the file via an
    # atomic rename. Cleanup of the in-flight entry runs in `finally` so a
    # failure does NOT leave a stale entry that wedges every future caller
    # as a perpetual waiter — the next caller becomes a fresh initiator on
    # a fresh event and can retry the download.
    #
    # `download_to` streams the GET to `tmp` without buffering the whole
    # object in RAM. The previous `f.write(read_bytes(shard_uri))` pattern
    # OOMed the DP container on a multi-GB shard (a 6 GB Python bytes
    # object inside a 2 GB cgroup limit).
    from adapters.storage.storage import download_to

    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        try:
            download_to(shard_uri, tmp)
            os.replace(tmp, path)
        except BaseException:
            # On any failure before the rename, remove the leftover temp
            # file so a crash mid-write does not leak `.tmp` files into
            # CACHE_DIR. Best-effort: the file may already be gone.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path
    finally:
        # Order matters: clear the registry FIRST so a thread that wakes from
        # `event.wait()` and re-enters this function sees no in-flight entry
        # and becomes a fresh initiator (if the file is still missing).
        # Then set the event so existing waiters unblock and re-check the
        # filesystem. Both steps run on both success and failure paths.
        with _INFLIGHT_DOWNLOADS_LOCK:
            _INFLIGHT_DOWNLOADS.pop(shard_uri, None)
        event.set()


def map_hits_to_matches(
    ids_row: np.ndarray,
    distances_row: np.ndarray,
    sidecar: Dict[str, dict],
    top_k: int,
) -> List[dict]:
    """Translate one FAISS result row into the v1 `matches` shape.

    `ids_row`/`distances_row` are the per-query rows returned by
    `index.search`. FAISS pads short result sets with `-1` ids; those are
    skipped. Each surviving int64 id is looked up in `sidecar` to recover the
    original string id and metadata. A hit missing from the sidecar (should
    not happen — defensive only) falls back to the stringified int64 id with
    empty metadata.

    `score` is the raw FAISS L2 distance (lower == closer); we do not
    normalise it. Documented in `docs/api/query.md`.
    """
    matches: List[dict] = []
    for i in range(min(top_k, len(ids_row))):
        raw_id = int(ids_row[i])
        if raw_id == -1:
            continue
        entry = sidecar.get(str(raw_id))
        if entry is not None:
            matches.append(
                {
                    "id": entry.get("id", str(raw_id)),
                    "score": float(distances_row[i]),
                    "metadata": entry.get("metadata") or {},
                }
            )
        else:
            matches.append({"id": str(raw_id), "score": float(distances_row[i]), "metadata": {}})
    return matches


def metadata_matches_filter(metadata: dict, flt: Dict[str, Any]) -> bool:
    """AND-of-equals predicate for a single record's metadata.

    A record matches only if, for EVERY key in `flt`, `metadata` contains
    that key with an *exactly-equal* value. Semantics:

      - missing key  -> no match (the record is excluded)
      - type mismatch -> no match: filter value `"2024"` (str) does NOT
        match metadata `2024` (int). We do not coerce. Python's `==` treats
        `1 == 1.0` as true and `True == 1` as true, so we additionally
        require identical `type()` to keep "string compares to string,
        number to number" strict and predictable for v1.
      - a `null` filter value (`{"k": null}`) -> no match, always. `null`
        is accepted by request validation but is not a meaningful equality
        target in v1, so any record carrying that key is simply excluded.
        Documented in `docs/api/v1.md`.

    An empty `flt` matches everything (callers short-circuit before calling
    this, but it stays correct if invoked directly).
    """
    for key, want in flt.items():
        if want is None:
            # A `null` filter value never matches — see docstring / v1.md.
            return False
        if key not in metadata:
            return False
        got = metadata[key]
        if type(got) is not type(want):
            return False
        if got != want:
            return False
    return True


# --- Consolidated (FAISS) search: resolution and search, split -----------------
#
# `_hot_search` (the old name) bundled TWO independent phases into one call:
#
#   1. SHARD RESOLUTION — the cheap Postgres catalog lookup that picks the
#      newest shard and yields the recall watermark (its `consolidated_lsn`).
#   2. The FAISS SEARCH — the ~33 ms (at 1M) vector search on the resolved
#      shard's deserialised index.
#
# The recall scan only needs phase 1's WATERMARK, not phase 2's FAISS result,
# so bundling them forced `run_query` to wait for the whole consolidated search
# (resolution + FAISS) before even STARTING the recall round-trip — making the
# recall cost strictly additive (the measured ~15 ms recall tax, #31). Splitting
# the two phases lets `run_query` resolve ONCE, then run the FAISS search and the
# recall scan CONCURRENTLY (recall only needs the watermark).
#
# The `_hot_search` name is retired (the "hot" overload was always misleading —
# it returned a hot|consolidated cache-state, not a "hot tier"). The OTel span
# name `query.hot_search` is DELIBERATELY KEPT for now: bench dashboards and
# latency attribution key on it; a span rename is tracked separately as #33.


def _resolve_shard(
    tenant: str,
    dataset: str,
    resolved: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve the newest shard for `(tenant, dataset)` — phase 1 only (no FAISS).

    Returns the resolved-shard info dict (with the newest catalog row under the
    key `"shard"`), or `None` when the dataset has no shard yet (the SAME
    no-shard signal `_hot_search` returned, so the ephemeral fallback still
    triggers). The returned dict is exactly what `_watermark_for_shard(...)`
    consumes via `.get("shard")` — resolve ONCE here and pair the watermark to
    THIS shard (invariant I3); never read a watermark independently.

    `resolved`, when supplied, is the SAME out-dict the caller passes through to
    `_search_consolidated_shard` so the FAISS search reads exactly the shard this
    resolution picked. When `resolved` is None a fresh dict is allocated. On a
    no-shard dataset the dict is left without a `"shard"` key (watermark 0).

    Runs UNDER the caller's open `query.hot_search` span (the caller opens it so
    the consolidated FAISS spans nest under the SAME parent). The
    `state.list_shards` span is opened here so the catalog lookup stays an
    attributable child of `query.hot_search`.
    """
    if resolved is None:
        resolved = {}
    # The Postgres shard-catalog lookup — its own span (it used to run
    # before any span opened and was therefore invisible in traces).
    with list_shards_span(tenant=tenant, dataset=dataset):
        # Per-`(tenant, dataset)` catalog cache wrapper. With the SSD
        # tier off OR `RB_CATALOG_FRESHNESS_S=0` this is a passthrough
        # to `list_shards`; with both active, repeated lookups for the
        # same dataset within the TTL skip the Postgres round-trip.
        shards = _cached_list_shards(tenant, dataset)
    if not shards:
        # No shard for this dataset yet — the same signal `_hot_search`
        # returned. `resolved` carries no `"shard"` key so the watermark is 0
        # and (with recall on) all recall rows qualify; with recall off the
        # caller falls through to the ephemeral path.
        return None
    latest = shards[0]
    # I3 watermark pairing: record WHICH shard the consolidated search will read,
    # so the recall-tier union filters with this exact shard's consolidated_lsn
    # (never a watermark resolved independently). `latest` is the head of the
    # newest-first catalog list — the same row `_search_consolidated_shard` loads.
    resolved["shard"] = latest
    return resolved


def _search_consolidated_shard(
    tenant: str,
    dataset: str,
    vector: List[float],
    top_k: int,
    flt: Optional[Dict[str, Any]] = None,
    nprobe: Optional[int] = None,
    resolved: Optional[Dict[str, Any]] = None,
):
    """Run the FAISS search on the already-resolved shard — phase 2 only.

    Returns `(matches, mode)` where `mode` is the cache-state `"hot"` (cache
    hit) or `"cold"` (a shard that had to be faulted into the local cache for
    the first time) — the SAME strings `_hot_search` returned — or `None` if
    `resolved` carries no shard (the no-shard signal — so the caller still
    falls through to the ephemeral path exactly as before).

    `resolved` is the out-dict produced by `_resolve_shard(...)`: this function
    reads `resolved["shard"]` and never re-resolves, so the FAISS search reads
    exactly the shard the recall watermark was paired to (invariant I3).

    `nprobe`, when supplied, overrides the server-default IVF `nprobe` for
    this query only — used by the recall benchmark to sweep `nprobe` across
    one index build, and a per-query tuning knob for callers.

    When `flt` is a non-empty dict the filtered path is *exhaustive*: FAISS
    is asked for the whole shard (`fetch_k = index.ntotal`) and, on an IVF
    index, every cell is scanned (`nprobe = nlist`). The AND-of-equals
    predicate is then applied to each candidate's sidecar metadata and the
    survivors — already in ascending-distance order from FAISS — are
    truncated to `top_k`. The search is exhaustive (every cell, every
    vector) and therefore exact for IVFFlat and flat indexes; a legacy
    IVF+PQ shard's per-vector distances stay PQ-approximate (full cell
    coverage fixes which cells are scanned, not PQ's lossy quantization).
    A filtered query returns exactly `min(top_k, total_matching)`; the
    result count never depends on `top_k` for a dataset with at least
    `top_k` matches. A
    query can still legitimately return fewer than `top_k` results when the
    dataset genuinely contains fewer than `top_k` matching records — that is
    an exact answer, not an approximation or an error.

    Runs UNDER the caller's open `query.hot_search` span so the `shard.download`,
    `faiss.load_index` and `faiss.search` children stay attributable to it.
    """
    latest = resolved.get("shard") if resolved else None
    if latest is None:
        return None
    has_filter = bool(flt)
    shard_id = latest.get("id")

    # In-memory cache lookup. A hit reuses the already-deserialised FAISS
    # index + parsed sidecar (the query is then just the search); a miss
    # is a real cold load — the `shard.download` and `faiss.load_index`
    # spans make a cold query obviously distinguishable from a warm one
    # in a trace.
    cached = _cache_get(shard_id)
    if cached is not None:
        index, sidecar = cached
        is_cold = False
        obs_metrics.record_shard_cache("hit")
    else:
        with shard_download_span(uri=latest["shard_uri"]):
            # `_ensure_cached` internally checks `RB_SHARD_TIER_BYTES`
            # to decide between the SSD tier and the legacy in-process
            # single-flight. The URI is the cache key in either path —
            # no separate shard id needed at this layer.
            local_path = _ensure_cached(latest["shard_uri"])
        # `mmap=_MMAP_ENABLED` stamps `rosalinddb.mmap` on the span so a
        # trace makes the cold-load strategy (mmap vs full deserialise)
        # obvious without an out-of-band lookup of the deployment's env.
        with faiss_load_index_span(uri=latest["shard_uri"], mmap=_MMAP_ENABLED):
            # mmap path: FAISS keeps the index file open and serves reads
            # from the page cache instead of copying the whole serialised
            # blob into RSS. `IO_FLAG_READ_ONLY` pairs with the mmap flag
            # — the cached entry is shared across queries, so a write
            # would race with concurrent searches in flight.
            if _MMAP_ENABLED:
                index = faiss.read_index(
                    local_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
                )
            else:
                index = faiss.read_index(local_path)
        sidecar = read_shard_sidecar(latest["shard_uri"])
        _cache_put(shard_id, index, sidecar)
        is_cold = True
        obs_metrics.record_shard_cache("miss")
    x = np.array([vector], dtype=np.float32)

    # `faiss.search` span — now the actual vector search ALONE, so the
    # name finally means what it says. High-cardinality tenant/dataset
    # attributes are correct on a span (not a metric).
    #
    # Page-fault sampler: read /proc/self/stat's `majflt` before and
    # after the search and record the delta to `rosalinddb.shard.page_faults`.
    # On the mmap path this is the operator-facing signal that the shard's
    # pages were cold in the page cache (synchronous disk reads happened
    # during the search). With mmap off the delta stays at zero — the
    # whole index is already resident in RSS — so the metric is a clean
    # "mmap is on AND the cache cooled" indicator. Best-effort: the
    # sampler returns None on macOS dev / format surprise; we then skip
    # the record.
    maj_before = _read_major_faults()
    with faiss_search_span(tenant=tenant, dataset=dataset, top_k=top_k) as sp:
        matches = _run_faiss_search(
            index, sidecar, x, top_k, flt, nprobe, has_filter, sp
        )
    maj_after = _read_major_faults()
    if maj_before is not None and maj_after is not None:
        obs_metrics.record_shard_page_faults(max(0, maj_after - maj_before))
    return matches, ("cold" if is_cold else "hot")


def _consolidated_search(
    tenant: str,
    dataset: str,
    vector: List[float],
    top_k: int,
    flt: Optional[Dict[str, Any]] = None,
    nprobe: Optional[int] = None,
    resolved: Optional[Dict[str, Any]] = None,
):
    """Resolve the newest shard, then FAISS-search it — the serial composition.

    Equivalent to the retired `_hot_search`: opens the `query.hot_search` span
    and runs `_resolve_shard(...)` + `_search_consolidated_shard(...)` in
    sequence under it. Returns `(matches, mode)` (cache-state `"hot"`/`"cold"`)
    or `None` when the dataset has no shard yet.

    This is the CONSOLIDATED-ONLY path (recall off) and the entry point for
    callers that just want a complete consolidated search without orchestrating
    the resolve/search split themselves (tests, the incremental-indexing/mmap
    integration checks). The recall-union path in `run_query` does NOT call this
    — it opens its own `query.hot_search` span and runs the two phases with the
    FAISS search OVERLAPPING the recall scan.
    """
    # `query.hot_search` is the parent span; the catalog lookup, shard
    # download, index deserialize and vector search nest under it as separate
    # children so a query trace decomposes into attributable pieces. The span
    # literal stays `query.hot_search` (bench/attribution key on it; rename #33).
    with hot_search_span(tenant=tenant, dataset=dataset):
        resolved = _resolve_shard(tenant, dataset, resolved)
        if resolved is None:
            return None
        return _search_consolidated_shard(
            tenant, dataset, vector, top_k, flt, nprobe, resolved
        )


def _run_faiss_search(index, sidecar, x, top_k, flt, nprobe, has_filter, sp):
    """Run the vector search and map hits — the body of the `faiss.search` span.

    Extracted so the `faiss.search` span wraps only the search itself (no
    download/deserialize). `sp` is that span, used to record the
    `nprobe`/`fetch_k` attributes.
    """
    if has_filter:
        # Exhaustive-when-filtered: FAISS cannot filter by metadata, so we
        # post-filter. A fixed over-fetch is unsafe — it ignores filter
        # selectivity, so the `fetch_k` nearest vectors are not guaranteed
        # to contain `top_k` survivors and the result count then depends
        # on `top_k`. The fix is an exact search: fetch EVERY vector
        # (`fetch_k = ntotal`) and, on an IVF index, scan EVERY cell
        # (`nprobe = nlist`) — a filter-match in an unprobed cell would
        # otherwise be invisible no matter how large `fetch_k` is.
        #
        # IDSelector-based pre-filtering — so FAISS skips distance math on
        # non-matching vectors — is the intended optimization for large
        # shards and is deliberately deferred: the full scan is correct
        # and fast at MVP scale (datasets are capped at 100k vectors).
        # A per-request `nprobe` override is intentionally NOT honored for
        # a filtered query: the search must be exhaustive for correctness,
        # so `full_coverage=True` forces `nprobe = nlist` regardless.
        search_params, applied_nprobe = _ivf_search_params(
            index, full_coverage=True
        )
        sp.set_attribute("rosalinddb.nprobe", applied_nprobe)
        search_kwargs = {"params": search_params} if search_params is not None else {}
        fetch_k = int(getattr(index, "ntotal", 0)) or top_k
        # Record the *actual* candidate count searched, not the requested
        # `top_k` — a filtered query fetches the whole shard, and a span
        # tagged only with `top_k` would mislead latency triage.
        sp.set_attribute("rosalinddb.fetch_k", fetch_k)
        distances, ids = index.search(x, fetch_k, **search_kwargs)
        # Map ALL candidates (mapped list is already nearest-first), then
        # filter by the predicate and truncate to top_k.
        candidates = map_hits_to_matches(ids[0], distances[0], sidecar, fetch_k)
        return [
            m for m in candidates if metadata_matches_filter(m["metadata"], flt)
        ][:top_k]
    # Unfiltered: resolve `nprobe` into per-search params (no shared-index
    # mutation, no cross-query race; no-op on a flat index). A per-request
    # `nprobe` overrides the server default and is clamped to MAX_NPROBE.
    # FAISS searches for exactly `top_k`.
    search_params, applied_nprobe = _ivf_search_params(index, nprobe)
    sp.set_attribute("rosalinddb.nprobe", applied_nprobe)
    search_kwargs = {"params": search_params} if search_params is not None else {}
    sp.set_attribute("rosalinddb.fetch_k", top_k)
    distances, ids = index.search(x, top_k, **search_kwargs)
    return map_hits_to_matches(ids[0], distances[0], sidecar, top_k)


# --- Auth/quota-free query core -----------------------------------------------
#
# The `POST /v1/query` work splits into three layers so it can be mounted on
# both the legacy authenticated route and the DP-trust route:
#
#   - `validate_query_body(body, tenant_id)` — pure body validation. Parses and
#     range-checks `dataset` / `vector` / `top_k` / `nprobe` / `filter` and
#     resolves the dataset via `get_dataset(tenant_id, ...)`. Returns either a
#     v1 error `JSONResponse` or a `_ParsedQuery`. No auth, no quota, no I/O
#     beyond the catalog lookup.
#   - `run_query(tenant_id, parsed)` — runs the hot/ephemeral search for an
#     already-validated request and returns the v1 response dict. No auth, no
#     quota.
#   - `execute_v1_query(tenant_id, body, *, consume_quota)` — the glue: validate
#     → (optionally) consume quota → search. `consume_quota`, when supplied, is
#     invoked AFTER body validation and BEFORE the search, exactly matching the
#     legacy quota timing. The DP path passes `consume_quota=None` to skip it.
#
# The authenticated `POST /v1/query` route below is a thin wrapper:
# `current_tenant_id` + `rate_limit` dependencies + a `try_consume_query`
# callback + `execute_v1_query`. Its observable behaviour is unchanged.


class _ParsedQuery:
    """A validated `/v1/query` request, ready for `run_query`.

    Produced by `validate_query_body`; carries the post-validation values so
    `run_query` does no re-parsing. `vector` is already coerced to `float`s.
    """

    __slots__ = ("dataset_name", "vector", "top_k", "nprobe", "filter")

    def __init__(
        self,
        dataset_name: str,
        vector: List[float],
        top_k: int,
        nprobe: Optional[int],
        flt: Dict[str, Any],
    ) -> None:
        self.dataset_name = dataset_name
        self.vector = vector
        self.top_k = top_k
        self.nprobe = nprobe
        self.filter = flt


def validate_query_body(
    body: Any, tenant_id: str
) -> Union[JSONResponse, _ParsedQuery]:
    """Validate a `/v1/query` request body for `tenant_id`.

    Auth-free and quota-free: this performs ONLY the body validation the v1
    contract specifies (dataset existence/ownership, vector shape, dimension,
    `top_k`, `nprobe`, `filter`) plus the dataset catalog lookup. Returns a v1
    error `JSONResponse` on the first failure, or a `_ParsedQuery` on success.

    The dataset is resolved with `get_dataset(tenant_id, name)`, so a
    cross-tenant or missing dataset collapses to `404 dataset_not_found` —
    existence is never leaked. Identical validation order/codes to the legacy
    inline path.
    """
    if not isinstance(body, dict):
        return _err(400, "invalid_request", "Request body must be a JSON object")

    dataset_name = body.get("dataset")
    if not isinstance(dataset_name, str) or not dataset_name:
        return _err(404, "dataset_not_found", "dataset is required")

    # Cross-tenant / missing both collapse to 404 — never leak existence.
    dataset = get_dataset(tenant_id, dataset_name)
    if dataset is None:
        return _err(404, "dataset_not_found", f"Dataset '{dataset_name}' not found")

    vector = body.get("vector")
    if not isinstance(vector, list) or not all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in vector
    ):
        return _err(400, "invalid_request", "vector must be an array of numbers")

    expected_dim = int(dataset["dimension"])
    if len(vector) != expected_dim:
        return _err(
            400,
            "dimension_mismatch",
            f"query vector length {len(vector)} != dataset dimension {expected_dim}",
            details={"expected": expected_dim, "got": len(vector)},
        )

    top_k = body.get("top_k", DEFAULT_TOP_K)
    if top_k is None:
        top_k = DEFAULT_TOP_K
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1 or top_k > MAX_TOP_K:
        return _err(
            400,
            "top_k_out_of_range",
            f"top_k must be between 1 and {MAX_TOP_K}",
        )

    # Optional per-query IVF `nprobe` override. Absent / null → the
    # server default (`RB_QUERY_NPROBE`). A positive int overrides it for this
    # query only — used by `make bench-recall` to sweep `nprobe` over a single
    # index build, and a per-query recall/latency knob for callers.
    nprobe_override = body.get("nprobe")
    if nprobe_override is not None:
        if (
            not isinstance(nprobe_override, int)
            or isinstance(nprobe_override, bool)
            or nprobe_override < 1
        ):
            return _err(400, "invalid_request", "nprobe must be a positive integer")
        if nprobe_override > MAX_NPROBE:
            return _err(
                400,
                "nprobe_out_of_range",
                f"nprobe must be between 1 and {MAX_NPROBE}",
            )

    # `filter`: optional flat object of field->value, AND-of-equals.
    # Absent / null / `{}` -> no filtering (behaviour unchanged). A non-flat
    # filter (nested object / array value) is rejected — v1 has no ranges,
    # OR, or nesting. A `null` *value* (`{"k": null}`) is accepted by this
    # validation but never matches any record (see `metadata_matches_filter`
    # and `docs/api/v1.md`).
    flt = body.get("filter")
    if flt is None:
        flt = {}
    if not isinstance(flt, dict):
        return _err(400, "invalid_request", "filter must be a JSON object")
    for k, v in flt.items():
        if isinstance(v, (dict, list)):
            return _err(
                400,
                "invalid_request",
                "filter values must be scalars (no nesting, ranges, or OR in v1)",
            )
    vector_f = [float(v) for v in vector]

    return _ParsedQuery(dataset_name, vector_f, top_k, nprobe_override, flt)


def _classify_hot_path_error(exc: BaseException) -> Tuple[str, str]:
    """Map a hot-path exception to a v1 `(error_code, safe_message)` tuple.

    Mirrors `services/ephemeral_runner/run.py:_classify_error` so the hot
    path and the ephemeral path surface the same error codes for the same
    failure shapes (a `PermissionError` on the cache fs is `cache_unavailable`
    in both worlds; an S3 outage is `storage_unavailable` in both). The
    duplicate exists rather than a shared helper because the ephemeral runner
    deliberately avoids importing from the query_api package — a circular
    import — so the classification table is held twice. Keep them in sync.

    `safe_message` is built from the exception CLASS NAME only; `str(exc)` is
    never surfaced — a botocore `ClientError` carries an endpoint URL and
    sometimes signed-URL params that must not leak to the customer.
    """
    if isinstance(exc, RecallUnavailable):
        # The recall (pgvector) tier is unreachable for this query — a typed
        # boundary error raised ONLY by the recall search path (`recall_search`
        # wraps a recall-store connection failure / sustained recall-pool
        # exhaustion in `RecallUnavailable`). Distinct, retryable 503: NOT the
        # generic `ephemeral_error` 500 (benchmark finding C2: an unclassified
        # psycopg2 OperationalError from the recall path used to hard-500), NOT
        # the write-side `recall_write_failed`. The query path must NOT silently
        # serve consolidated-only results — a recall outage means recent,
        # unconsolidated writes are unreadable, so a silent consolidated-only 200 would
        # break read-your-writes without signal. A 503 tells the client to retry.
        # Scoped by TYPE, not by `isinstance(exc, OperationalError)`, so an
        # identical psycopg2 error from the control-plane/consolidated path is NOT
        # misclassified as recall_unavailable.
        return "recall_unavailable", "Recall tier is temporarily unavailable"
    if isinstance(exc, PermissionError):
        return "cache_unavailable", "Shard cache is unreadable or unwritable"
    if isinstance(exc, shard_tier.CacheCapacityExceeded):
        # SSD-tier admission floor (`MIN_RESIDENT_S`) rejected a speculative
        # arrival because every eviction candidate is too young. Distinct from
        # `storage_unavailable` so an operator dashboard can tell capacity
        # pressure from a storage outage; both map to 503 but the dedicated
        # code is the upsize signal for `RB_SHARD_TIER_BYTES`.
        return "cache_capacity_exceeded", "SSD cache tier is at capacity"
    if isinstance(exc, (DownloadCoalescingTimeout, shard_tier.ShardTierTimeout)):
        # Bounded-wait coalescing timeout — the in-flight initiator hasn't
        # released the event within the configured deadline. Transient by
        # design; the next request becomes a fresh initiator on a fresh event.
        # `DownloadCoalescingTimeout` is the legacy `_ensure_cached` path;
        # `ShardTierTimeout` is the SSD-tier path — both describe the same
        # operational condition from the customer's perspective and collapse
        # to the same 503 so a client-side retry policy does not
        # need to distinguish which layer timed out.
        return "storage_unavailable", "Shard storage is temporarily unavailable"
    if isinstance(exc, FileNotFoundError):
        return "storage_unavailable", "Shard storage is temporarily unavailable"
    # Optional botocore import — boto3 may not be installed in a memory-only
    # test environment. A late import keeps this module's import graph slim.
    try:
        from botocore.exceptions import ClientError as _BotoClientError  # type: ignore

        if isinstance(exc, _BotoClientError):
            return (
                "storage_unavailable",
                "Shard storage is temporarily unavailable",
            )
    except Exception:  # noqa: BLE001 - boto3 missing in this env
        pass
    if isinstance(exc, OSError):
        return "cache_unavailable", "Shard cache I/O error"
    return "ephemeral_error", f"Query failed: {type(exc).__name__}"


# --- Recall + Consolidated union (RB_RECALL) ------------------------------
#
# When the recall tier is on, `POST /v1/query` searches BOTH tiers and merges
# (docs/architecture/recall-consolidate.md, "Read path — the union"):
#
#   - Consolidated: the `_search_consolidated_shard` FAISS path, returning
#     matches with FAISS **L2² distances** and the cache-state `mode`
#     (hot|cold). UNCHANGED.
#   - Recall: a brute-force exact scan over `recall_vectors` above the resolved
#     shard's watermark, returning rows with the SAME metric (pgvector `<->`
#     squared → L2²) plus tombstones.
#
# The two are unioned by `_merge_recall_and_consolidated`: recall is AUTHORITATIVE for any
# id above the watermark, so every recall id (live, tombstoned, or filtered-out)
# SUPPRESSES the stale consolidated copy of that id; only filter-passing live recall rows
# contribute an actual match. The result is sorted ascending by L2² and truncated
# to `top_k` (invariant I1 guarantees the two tiers partition the universe, so the
# union is complete and non-double-counting).
#
# OVERLAP (#31): the consolidated FAISS search and the recall scan are
# INDEPENDENT once the shard is resolved (recall needs only the watermark, not
# the FAISS result). `run_query` therefore resolves the shard ONCE, then runs
# the FAISS search inline while the recall scan runs on a worker thread, so the
# union's wall-time is ~max(consolidated, recall) instead of their sum — erasing
# the measured ~15 ms additive recall tax. Both genuinely overlap under the GIL:
# FAISS releases it during its C++ search and psycopg2 releases it during the
# network round-trip.
#
# `_RECALL_EXECUTOR`: a module-level, BOUNDED ThreadPoolExecutor for the recall
# half of the overlap. A shared pool (not a per-call thread) avoids unbounded
# thread creation under query_dp concurrency; the bound is sized to track that
# concurrency so it does not serialize requests by starving on workers. Each
# query submits exactly ONE recall task and immediately runs FAISS inline, so a
# pool of N workers supports N concurrent overlapping queries; beyond that a
# query's recall task simply queues behind the FAISS work it would have waited on
# anyway (no correctness impact, graceful degradation under saturation). The
# default tracks a modest data-plane fan-out and is overridable via
# `RB_RECALL_OVERLAP_WORKERS` for high-concurrency deployments.
_RECALL_OVERLAP_WORKERS = max(1, int(os.getenv("RB_RECALL_OVERLAP_WORKERS", "32")))
_RECALL_EXECUTOR = ThreadPoolExecutor(
    max_workers=_RECALL_OVERLAP_WORKERS,
    thread_name_prefix="recall-overlap",
)


def _watermark_for_shard(shard: Optional[Dict[str, Any]]) -> int:
    """Resolve the recall watermark from the consolidated-search's resolved shard (I3).

    The watermark is the `consolidated_lsn` of the shard the consolidated search ACTUALLY
    resolved — every recall row with `lsn > watermark` is unconsolidated and must
    be unioned in. If no shard exists yet (`shard is None`), the watermark is `0`
    so ALL recall rows qualify (a brand-new dataset's writes live only in recall).
    A shard row predating migration 008, or the memory-mode shard row that has no
    `consolidated_lsn` field, defaults to `0` — backward-compatible with the
    `NOT NULL DEFAULT 0` column.
    """
    if not shard:
        return 0
    raw = shard.get("consolidated_lsn", 0)
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _merge_recall_and_consolidated(
    recall_suppress_ids: set,
    recall_matches: List[dict],
    consolidated_matches: List[dict],
    top_k: int,
) -> List[dict]:
    """Union recall + consolidated matches: recall-authoritative suppression, then top_k.

    Rules (docs/architecture/recall-consolidate.md, "Read path — the union",
    "Dedup"):
      - **Recall is authoritative for any id above the watermark.** Recall
        suppresses the stale consolidated copy of EVERY id it has a row for —
        `recall_suppress_ids` is the FULL set of recall ids above the watermark
        (live, tombstoned, filtered-out, and ranked-past-top_k alike). A consolidated
        match survives only if recall has NO row for its id. This closes the leak
        where a live re-upsert that fails the filter (or ranks past `top_k`) let a
        stale, filter-matching consolidated copy surface.
      - **Only filter-passing live recall rows are MATCHES.** `recall_matches`
        are exactly those rows; a tombstone or a filtered-out live row
        contributes NO match (it only suppresses, via `recall_suppress_ids`).
      - **Sort + truncate.** The surviving matches (recall matches + the
        un-suppressed consolidated matches) are sorted ascending by L2² `score` and
        truncated to `top_k`.

    Both inputs already carry FAISS-aligned L2² `score`s (the recall scan squares
    pgvector's `<->`), so a single ascending sort over the union ranks correctly.
    """
    # Recall matches always survive; consolidated matches survive only if recall has NO
    # row for that id (suppression keys on the FULL recall id-set, not just the
    # ids that became matches).
    merged: List[dict] = [
        {"id": r["id"], "score": r["score"], "metadata": r["metadata"]}
        for r in recall_matches
    ]
    merged.extend(m for m in consolidated_matches if m["id"] not in recall_suppress_ids)
    # Stable ascending sort by L2² distance; truncate to top_k.
    merged.sort(key=lambda m: m["score"])
    return merged[:top_k]


def run_query(tenant_id: str, parsed: _ParsedQuery) -> Union[JSONResponse, dict]:
    """Run the hot/ephemeral search for an already-validated query.

    Auth-free and quota-free: `parsed` has already cleared `validate_query_body`
    and no quota is consumed here. Returns either the customer-facing v1
    response dict — `{matches, latency_ms, mode}` for the hot/cold path, or the
    `{matches: [], mode: "ephemeral", job_id}` enqueue shape when the dataset
    has no shard yet — OR a v1 error `JSONResponse` (typically 503) when the
    hot path raises an unrecoverable error (cache fs unwritable, S3 outage,
    FAISS load failure, etc.). Callers wrap a dict in a `JSONResponse` or
    return it directly; a `JSONResponse` is returned unchanged.

    A hot-path exception used to collapse to `hot = None` and silently fall
    through to the ephemeral fallback (which then returned `{matches:[],
    mode:"ephemeral"}` with HTTP 200). An exception now classifies into a v1
    error envelope and returns the 503 (or class-specific code) directly —
    the ephemeral fallback is reserved for its actual semantics: the dataset
    legitimately has no shard yet. The consolidated search returns `None` ONLY
    in that case.

    OVERLAP (#31): with the recall union on, the consolidated FAISS search and
    the recall scan are INDEPENDENT given the resolved shard's watermark (recall
    needs only the watermark, not the FAISS result). So this resolves the shard
    ONCE, submits `recall_search` to a worker thread, runs the FAISS search
    INLINE, and joins the recall future — the union's wall-time is
    ~max(consolidated, recall) instead of their sum (erasing the ~15 ms recall
    tax). With recall off the consolidated-only path runs sequentially, exactly
    as before.
    """
    start = time.time()
    dataset_name = parsed.dataset_name
    vector_f = parsed.vector
    top_k = parsed.top_k
    flt = parsed.filter
    nprobe_override = parsed.nprobe

    # Recall-tier union gate. DEFAULT-OFF: with `RB_RECALL` off (or no
    # `RB_RECALL_DSN`) this is False and the consolidated-only branch below NEVER
    # opens a recall connection, never resolves a watermark, and is byte-identical
    # to today. Read once per query so the rest of the function branches on a
    # single stable value.
    union_on = recall_enabled()

    if not union_on:
        # --- Consolidated-only path (recall off) --------------------------
        # Sequential `_resolve_shard` + `_search_consolidated_shard` via the
        # `_consolidated_search` composition — byte-identical to the old
        # `_hot_search` path: no recall, no thread. A raised exception means the
        # search could not run (cache fs unwritable, S3 fetch failed, FAISS index
        # unreadable, etc.) — NOT "no shard exists yet" — and must NOT silently
        # fall through to the ephemeral path; classify it into a v1 error envelope.
        try:
            hot = _consolidated_search(
                tenant_id, dataset_name, vector_f, top_k, flt, nprobe_override
            )
        except Exception as exc:  # noqa: BLE001
            code, safe_message = _classify_hot_path_error(exc)
            print(
                "v1_query: hot path failed: "
                f"tenant={tenant_id} dataset={dataset_name} "
                f"exc_class={type(exc).__name__} exc={exc!r} code={code}"
            )
            # 503 for storage/cache transient errors (retry-safe); 500 for the
            # generic catch-all (an unexpected exception is a server bug, not a
            # transient resource failure).
            status = 500 if code == "ephemeral_error" else 503
            return _err(status, code, safe_message)

        if hot is not None:
            matches, mode = hot
            counter("query_reads", 1)
            counter("cache_hit", 1)
            latency_ms = int((time.time() - start) * 1000.0)
            timer("latency_ms", latency_ms)
            # rosalinddb.queries{mode} + rosalinddb.query.duration{mode}.
            # `mode` (hot|cold) is the only label — no tenant/dataset.
            obs_metrics.record_query(mode)
            obs_metrics.record_query_duration(latency_ms, mode)
            # Count filtered queries and record the post-filter result count so
            # a highly selective filter (few survivors) is observable.
            if flt:
                obs_metrics.record_filtered_query()
                obs_metrics.record_filtered_result_count(len(matches))
            return {"matches": matches, "latency_ms": latency_ms, "mode": mode}

        # No shard yet → fall through to the ephemeral runner below.
        return _enqueue_ephemeral(tenant_id, dataset_name, vector_f, top_k, flt, start)

    # --- Recall-tier union (RB_RECALL on): OVERLAP consolidated + recall ---
    #
    # Resolve the shard FIRST (cheap catalog lookup), then run the FAISS search
    # and the recall scan CONCURRENTLY: recall needs only the watermark from the
    # resolution, not the FAISS result. The recall scan is scoped to rows ABOVE
    # the resolved shard's watermark (I3); when no shard exists yet the watermark
    # is 0 so ALL recall rows qualify — this is what lets a brand-new dataset's
    # just-written vectors be answered SYNCHRONOUSLY from recall instead of forced
    # down the ephemeral path (docs/architecture/recall-consolidate.md, "Read path
    # — the union"). A recall-store failure maps to the same v1 503 envelope as a
    # consolidated-path storage failure — the union must not 500.
    #
    # Both spans (`recall.search` and the consolidated children) must nest under
    # the request span, so the whole union runs inside ONE `query.hot_search`
    # span; the worker thread re-attaches the request's OTel context so the
    # `recall.search` span it opens stays a CHILD of the request span instead of
    # becoming an orphaned trace root (OTel current-context does NOT auto-propagate
    # across threads). The span literal stays `query.hot_search` (bench/attribution
    # key on it; rename tracked as #33).
    with hot_search_span(tenant=tenant_id, dataset=dataset_name):
        # I3 watermark pairing: resolve WHICH shard the consolidated search will
        # read so the recall scan filters with that exact shard's
        # `consolidated_lsn` — never a watermark resolved independently. Resolution
        # is the cheap catalog lookup; a raised exception here is a consolidated-path
        # failure and maps to the consolidated error envelope.
        resolved: Optional[Dict[str, Any]] = {}
        try:
            resolved = _resolve_shard(tenant_id, dataset_name, resolved)
        except Exception as exc:  # noqa: BLE001
            return _consolidated_error_response(tenant_id, dataset_name, exc)

        # `resolved` is None when no shard exists yet; `.get("shard")` is then
        # never reached. A no-shard query gets watermark 0 → all recall rows qualify.
        watermark = _watermark_for_shard(resolved.get("shard") if resolved else None)

        # Capture the CURRENT OTel context so the recall worker thread can attach
        # it: OpenTelemetry's current-context is thread-local and does NOT
        # propagate to a freshly-scheduled worker, so without this the
        # `recall.search` span opened inside `recall_search` would become an
        # orphaned root instead of a child of `query.hot_search`.
        parent_ctx = otel_context.get_current()

        def _recall_worker():
            # Re-attach the request context inside the worker so spans opened here
            # parent correctly; detach in a finally so the worker thread's context
            # is left clean for its next pool task.
            token = otel_context.attach(parent_ctx)
            try:
                # `recall_search` returns (suppress_ids, matches): the FULL set of
                # recall ids above the watermark (for authoritative suppression of
                # the stale consolidated copy) AND only the filter-passing live rows.
                return recall_search(
                    tenant_id, dataset_name, vector_f, top_k, watermark, flt
                )
            finally:
                otel_context.detach(token)

        # Submit recall to the worker, run the FAISS search INLINE — they overlap.
        recall_future = _RECALL_EXECUTOR.submit(_recall_worker)

        # Run the consolidated FAISS search inline on the already-resolved shard.
        # Its exception is captured (not raised here) so the recall future is
        # ALWAYS joined — never leaked — before any error is returned.
        consolidated_exc: Optional[BaseException] = None
        hot = None
        try:
            hot = _search_consolidated_shard(
                tenant_id, dataset_name, vector_f, top_k, flt, nprobe_override,
                resolved,
            )
        except Exception as exc:  # noqa: BLE001
            consolidated_exc = exc

        # Join the recall future — surfaces any worker exception here.
        recall_exc: Optional[BaseException] = None
        recall_suppress_ids: set = set()
        recall_matches: List[dict] = []
        try:
            recall_suppress_ids, recall_matches = recall_future.result()
        except Exception as exc:  # noqa: BLE001
            recall_exc = exc

    # BOTH-ERROR PRECEDENCE (documented, deterministic): if both branches failed,
    # report the CONSOLIDATED error first. This is the pre-overlap behaviour's
    # ordering — the consolidated search ran (and could fail) before recall in the
    # old serial code — so it preserves which envelope a double failure surfaced.
    # Either way each branch's exception maps to the SAME envelope it mapped to
    # serially: a consolidated/FAISS error → `_classify_hot_path_error` → 503
    # (storage/transient) or 500 (ephemeral_error); a recall error (incl. the
    # typed `RecallUnavailable`) → its 503 `recall_unavailable`. A recall failure
    # is NEVER masked, and a consolidated failure is NEVER reported as a recall
    # error (or vice-versa).
    if consolidated_exc is not None:
        return _consolidated_error_response(tenant_id, dataset_name, consolidated_exc)
    if recall_exc is not None:
        code, safe_message = _classify_hot_path_error(recall_exc)
        print(
            "v1_query: recall search failed: "
            f"tenant={tenant_id} dataset={dataset_name} "
            f"exc_class={type(recall_exc).__name__} exc={recall_exc!r} code={code}"
        )
        status = 500 if code == "ephemeral_error" else 503
        return _err(status, code, safe_message)

    # `cold_mode` reflects the consolidated shard's cache state (hot|cold); when no
    # consolidated shard exists it is None and the response `mode` reports `recall`
    # — the consolidated tier contributed nothing, recall answered. See the docs
    # note on `mode` semantics in the no-consolidated-shard case.
    if hot is not None:
        consolidated_matches, cold_mode = hot
    else:
        consolidated_matches, cold_mode = [], None

    # If there is neither a consolidated shard NOR any recall row, fall through to
    # the ephemeral path exactly as the consolidated-only path would (the dataset is
    # genuinely empty for this query). No recall id above the watermark AND
    # hot being None means nothing can answer synchronously. (`suppress_ids`
    # is non-empty iff there is ANY recall row, including tombstones — a
    # tombstone-only recall set with no consolidated shard still has nothing to
    # return, but suppression-only is harmless and the merge yields [].)
    if not (hot is None and not recall_suppress_ids):
        matches = _merge_recall_and_consolidated(
            recall_suppress_ids, recall_matches, consolidated_matches, top_k
        )
        # `mode`: the consolidated-shard cache state when a shard was read; `recall`
        # when only recall could answer (no consolidated shard). The recall tier
        # contributed regardless — documented in docs/api/query.md.
        mode = cold_mode if cold_mode is not None else "recall"
        counter("query_reads", 1)
        counter("cache_hit", 1)
        latency_ms = int((time.time() - start) * 1000.0)
        timer("latency_ms", latency_ms)
        obs_metrics.record_query(mode)
        obs_metrics.record_query_duration(latency_ms, mode)
        if flt:
            obs_metrics.record_filtered_query()
            obs_metrics.record_filtered_result_count(len(matches))
        return {"matches": matches, "latency_ms": latency_ms, "mode": mode}

    # No shard AND no recall row → ephemeral enqueue (shared with the off path).
    return _enqueue_ephemeral(tenant_id, dataset_name, vector_f, top_k, flt, start)


def _consolidated_error_response(
    tenant_id: str, dataset_name: str, exc: BaseException
) -> JSONResponse:
    """Map a consolidated/FAISS-path exception to its v1 error envelope.

    Identical classification + status mapping the serial `_hot_search` path used:
    `_classify_hot_path_error(exc)` → 503 for storage/cache transient errors
    (retry-safe) or 500 for the generic `ephemeral_error` catch-all. Extracted so
    both the recall-on overlap path (resolution failure and the inline FAISS
    failure) and a single call site share ONE consolidated-error mapping — a
    consolidated failure is NEVER reported as a recall error.
    """
    code, safe_message = _classify_hot_path_error(exc)
    print(
        "v1_query: hot path failed: "
        f"tenant={tenant_id} dataset={dataset_name} "
        f"exc_class={type(exc).__name__} exc={exc!r} code={code}"
    )
    status = 500 if code == "ephemeral_error" else 503
    return _err(status, code, safe_message)


def _enqueue_ephemeral(
    tenant_id: str,
    dataset_name: str,
    vector_f: List[float],
    top_k: int,
    flt: Dict[str, Any],
    start: float,
) -> dict:
    """Enqueue an ephemeral query and return the `{matches:[], mode:ephemeral}` shape.

    The dataset has no shard yet (and, with recall on, no recall row either), so
    nothing can answer synchronously. The ephemeral runner does its own FAISS
    search + sidecar lookup and publishes RESULT_READY; the caller polls
    `GET /v1/query/status/{job_id}`. Shared by the recall-off and recall-on paths
    so the enqueue shape, metrics, and `job_id` are byte-identical in both.
    """
    correlation_id = "job_" + uuid.uuid4().hex
    publish(
        "RUN_EPHEMERAL_QUERY",
        {
            "dataset": dataset_name,
            "tenant": tenant_id,
            "vector": vector_f,
            "top_k": top_k,
            # Forward the AND-of-equals filter so the ephemeral runner applies
            # it too — a filtered query against a not-yet-indexed dataset must
            # not return unfiltered results.
            "filter": flt,
            "correlation_id": correlation_id,
            "reply_to": os.getenv("RESULT_TOPIC", "RESULT_READY"),
        },
    )
    counter("ephemeral_queries", 1)
    latency_ms = int((time.time() - start) * 1000.0)
    # rosalinddb.queries{mode=ephemeral} + duration. The ephemeral runner does
    # the actual search async; this records the enqueue path's `mode`.
    obs_metrics.record_query("ephemeral")
    obs_metrics.record_query_duration(latency_ms, "ephemeral")
    return {
        "matches": [],
        "latency_ms": latency_ms,
        "mode": "ephemeral",
        "job_id": correlation_id,
    }


def execute_v1_query(
    tenant_id: str,
    body: Any,
    *,
    consume_quota: Optional[Callable[[], Optional[JSONResponse]]] = None,
) -> Union[JSONResponse, dict]:
    """Validate, (optionally) consume quota, and run a `/v1/query` request.

    `tenant_id` is the resolved tenant — the authenticated route resolves it
    from `Authorization`, the DP route from the trusted `X-RB-Tenant-Id`
    header. `body` is the raw parsed JSON request body.

    `consume_quota`, when supplied, is invoked AFTER body validation succeeds
    and BEFORE the search runs — exactly the legacy quota timing, so a request
    that fails validation never burns quota. It returns `None` to allow the
    query, or a `JSONResponse` (the `429 query_quota_exceeded` envelope) to
    reject it. The DP path passes `consume_quota=None` to skip quota entirely
    (quota is consumed by the CP proxy before the request reaches the DP).

    Returns a v1 error `JSONResponse` on validation/quota failure, or the
    customer-facing response dict on success.
    """
    parsed = validate_query_body(body, tenant_id)
    if isinstance(parsed, JSONResponse):
        return parsed
    if consume_quota is not None:
        rejection = consume_quota()
        if rejection is not None:
            return rejection
    return run_query(tenant_id, parsed)


def _consume_query_quota(tenant_id: str) -> Optional[JSONResponse]:
    """Consume one unit of daily query quota; return a 429 on exhaustion.

    Extracted as a callback for `execute_v1_query`. On the authenticated route
    this runs AFTER body validation and BEFORE the search, so a query that
    fails validation never burns quota — only a request we are about to
    actually serve counts. Returns `None` when quota was consumed, or the
    `429 query_quota_exceeded` envelope when the daily cap is hit.

    OSS opt-in: when `RB_ENABLE_QUOTAS` is unset/false (the self-host default)
    this short-circuits before touching state — no DB write, no counter bump.
    """
    if not quotas_enabled():
        return None
    ok, usage = try_consume_query(tenant_id)
    if not ok:
        # rosalinddb.quota.rejections{kind=query}.
        obs_metrics.record_quota_rejection("query")
        return query_quota_429(usage)
    return None


@router.post("/v1/query")
async def v1_query(
    request: Request,
    tenant_id: str = Depends(current_tenant_id),
    _rl: None = Depends(rate_limit),
):
    """Vector similarity search over a tenant's dataset (authenticated route).

    Hot path: load the newest shard for `(tenant, dataset)`, run a FAISS
    top-K search, and map each hit back to `{id, score, metadata}` via the
    shard sidecar. If the dataset has no shard yet, the query is enqueued on
    the ephemeral runner and an empty result is returned immediately with a
    `job_id` to poll via `GET /v1/query/status/{job_id}`.

    This is a thin wrapper: `current_tenant_id` + `rate_limit` are the
    auth/edge concerns; the validate → quota → search core lives in
    `execute_v1_query`. Query quota is consumed AFTER body validation and
    BEFORE the search — the externally observable behaviour (status codes,
    bodies, quota timing) is unchanged.
    """
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _err(400, "invalid_request", "Request body must be JSON")

    return execute_v1_query(
        tenant_id,
        body,
        consume_quota=lambda: _consume_query_quota(tenant_id),
    )


@router.get("/v1/query/status/{job_id}")
def v1_query_status(job_id: str):
    """Return the result of a previously enqueued ephemeral query.

    `{ready: false}` while the runner is still computing (also covers an
    unknown job_id — we do not distinguish, matching the v1 contract's
    polling shape). `{ready: true, matches, latency_ms, mode}` once done.

    The runner publishes a STRUCTURED ERROR ENVELOPE
    (`{ok: false, error: {code, message}}`) on RESULT_READY when its search
    fails (storage outage, cache fs unwritable, etc.). When the status poll
    finds such an envelope it surfaces it as **HTTP 503** with the v1 error
    body — a successful 200 from this endpoint strictly implies the `matches`
    array is a real top-K answer, never "we could not compute anything but
    here is an empty list". The legitimate empty-result case (filter matches
    nothing / empty dataset) still returns 200 with `matches: []` — it
    carries `ok: true` (or no `ok` field on a legacy stored result, which is
    treated as success).

    The result is read from the shared `result_store` (Redis-backed when
    `REDIS_URL` is set), so a status poll that lands on a different `query_api`
    replica than the one whose `RESULT_READY` consumer stashed the result still
    finds it.
    """
    res = result_store.get_result(job_id)
    if not res:
        return {"ready": False}
    # An error envelope from the runner takes priority — surface it as a v1
    # 503 with the canonical `{error: {code, message}}` body so the client
    # distinguishes "search failed" from "search returned no matches".
    if res.get("ok") is False:
        err = res.get("error") or {}
        code = err.get("code") or "ephemeral_error"
        message = err.get("message") or "Ephemeral query failed"
        return _err(503, code, message)
    return {
        "ready": True,
        "matches": res.get("matches", []),
        "latency_ms": res.get("latency_ms"),
        "mode": "ephemeral",
    }


def _result_consumer_loop() -> None:
    """Consume RESULT_READY messages and stash them keyed by correlation_id.

    Reliable-queue contract: a RESULT_READY message is `ack`-ed only after it
    has been stashed in the shared `result_store`, so a query_api restart
    mid-consume does not drop an ephemeral result — it stays reclaimable. A
    message with no `correlation_id` is unaddressable; it is acked (dropping
    it) rather than redelivered forever. The loop exits cleanly on `SIGTERM`.
    """
    while not should_stop():
        msg = consume("RESULT_READY", block=True, timeout=1.0)
        if not msg:
            continue
        try:
            cid = msg.get("correlation_id")
            if cid:
                result_store.store_result(cid, dict(msg))
        except Exception as exc:  # noqa: BLE001
            print(f"query_api: RESULT_READY consume error, nacking: {exc}")
            nack(msg, requeue=True)
            continue
        ack(msg)


def start_result_consumer() -> threading.Thread:
    """Start the RESULT_READY consumer as a daemon thread.

    Idempotent-ish: each call spawns a thread; callers should invoke once on
    app startup. Exposed so both the standalone `query_api` process and a
    single-process dev/test harness can wire it.
    """
    t = threading.Thread(target=_result_consumer_loop, name="result-consumer", daemon=True)
    t.start()
    return t


def drain_result_queue_once() -> None:
    """Drain any pending RESULT_READY messages synchronously (test helper).

    The HTTP-level tests run the ephemeral runner inline rather than as a
    background thread, so there is no consumer loop. This pulls everything
    currently on the queue into the shared `result_store` so a subsequent
    status poll sees the result.
    """
    while True:
        msg = consume("RESULT_READY", block=False)
        if not msg:
            break
        cid = msg.get("correlation_id")
        if cid:
            result_store.store_result(cid, dict(msg))
        ack(msg)
