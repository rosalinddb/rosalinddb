from __future__ import annotations

"""Ephemeral Runner service.

Consumes `RUN_EPHEMERAL_QUERY` messages, downloads the latest index shard for a
dataset, and performs a FAISS top-K search. In the minimal implementation,
results are computed and discarded; production would return via callback.
"""

import json
import os
import time
import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler

import faiss  # type: ignore
import numpy as np

from adapters.observability import init_observability
from adapters.observability import metrics as obs_metrics
from adapters.observability.tracing import (
    ephemeral_query_span,
    faiss_load_index_span,
    faiss_search_span,
    shard_download_span,
)
from adapters.queue.queue import consume, publish, ack, nack
from adapters.queue.shutdown import install_signal_handlers, should_stop
from adapters.state.state import list_shards, migrate
from adapters.landing.parquet_reader import read_shard_sidecar
from adapters.metrics.metrics import snapshot
# SSD-tier import. The import-time `.tmp` orphan sweep is bounded and
# log-and-continue, so importing unconditionally is cheap; the activation
# gate inside `_ensure_cached` makes sure the tier's `fetch` only runs when
# the deployment has opted in via `RB_SHARD_TIER_BYTES`.
from adapters.storage import shard_tier

# Classify an arbitrary exception into a v1 error envelope. botocore is an
# optional import — only S3 deployments need it — so a missing botocore must
# not break this module on a local memory:// run.
try:
    from botocore.exceptions import ClientError as _BotoClientError  # type: ignore
except Exception:  # noqa: BLE001 - boto3 not installed (memory-only test env)
    _BotoClientError = None  # type: ignore[assignment]

# Reuse the hot-path AND-of-equals predicate so the two query paths cannot
# drift. The ephemeral runner does a brute-force search and already has every
# candidate's metadata in memory (from the shard sidecar), so it simply
# excludes non-matching hits — no over-fetch is needed.
from services.query_api.v1_query import (
    _MMAP_ENABLED,
    _ivf_search_params,
    _read_major_faults,
    _truthy,
    metadata_matches_filter,
)

# NOTE: this module has its own local `_ensure_cached` (below) that is
# deliberately NOT the coalescing version in `services/query_api/v1_query.py`.
# Factoring the helper into a shared module would invert the existing one-way
# import (this runner -> v1_query) or break a circular. The ephemeral path
# runs at much lower concurrency than the hot path, where the download stampede
# was observed and fixed; if ephemeral concurrency grows, port the coalescing
# logic across or extract a shared `adapters/storage/shard_fetch.py` module
# that both call.

# Observability bootstrap at import — idempotent; see validator_worker/run.py.
init_observability("rosalinddb-ephemeral-runner")

CACHE_DIR = os.getenv("CACHE_DIR", "/var/cache/shards")
METRICS_PORT = int(os.getenv("METRICS_PORT", "9102"))


# --- Per-`(tenant, dataset)` catalog cache (mirror of v1_query) --------------
#
# Mirrored from `services.query_api.v1_query` for the same circular-import
# reason that keeps `_ensure_cached` and `_classify_error` duplicated here.
# The two copies MUST keep their public shape identical:
#   - `_cached_list_shards(tenant, dataset) -> list`
#   - `_invalidate_catalog_cache(tenant, dataset) -> bool`
#   - `_on_catalog_notify(payload: dict) -> None`
#   - `_now()` indirected for testability
#   - `_catalog_cache_clear()` test helper
# See the v1_query.py comment block for the full rationale; this comment
# is intentionally short to avoid letting the two drift apart.

_CATALOG_CACHE_MAX_ENTRIES = 10_000
_CATALOG_CACHE: "OrderedDict[tuple[str, str], tuple[float, list]]" = OrderedDict()
_CATALOG_CACHE_LOCK = threading.Lock()
# Per-key generation counter — see v1_query.py mirror for the race the
# generation check closes (concurrent invalidate during in-flight fetch).
_CATALOG_CACHE_GEN: "Dict[tuple[str, str], int]" = {}


def _now() -> float:
    """Monotonic clock source for cache TTL (patchable for tests)."""
    return time.monotonic()


def _catalog_freshness_s() -> float:
    """`RB_CATALOG_FRESHNESS_S` in seconds (default 5; 0 disables cache)."""
    try:
        return max(0.0, float(os.getenv("RB_CATALOG_FRESHNESS_S", "5")))
    except ValueError:
        return 5.0


def _catalog_cache_active() -> bool:
    """Active iff SSD tier is on AND TTL is positive (default-off rollback)."""
    return bool(os.getenv("RB_SHARD_TIER_BYTES")) and _catalog_freshness_s() > 0


def _cached_list_shards(tenant: str, dataset: str) -> list:
    """TTL-cached wrapper around `list_shards`. See v1_query.py mirror.

    Single-flight is NOT a contract: N concurrent callers on a cold miss
    can each call the source. `list_shards` is cheap relative to the
    search itself; the bounded worst case is a startup thundering herd.
    """
    if not _catalog_cache_active():
        return list_shards(tenant, dataset)
    key = (tenant, dataset)
    ttl = _catalog_freshness_s()
    now = _now()
    with _CATALOG_CACHE_LOCK:
        entry = _CATALOG_CACHE.get(key)
        if entry is not None and (now - entry[0]) < ttl:
            _CATALOG_CACHE.move_to_end(key)
            return entry[1]
        # Snapshot generation so a concurrent invalidate after this
        # point causes our install to lose. See v1_query.py mirror.
        gen_pre = _CATALOG_CACHE_GEN.get(key, 0)
    rows = list_shards(tenant, dataset)
    with _CATALOG_CACHE_LOCK:
        if _CATALOG_CACHE_GEN.get(key, 0) != gen_pre:
            return rows
        _CATALOG_CACHE[key] = (now, rows)
        _CATALOG_CACHE.move_to_end(key)
        while len(_CATALOG_CACHE) > _CATALOG_CACHE_MAX_ENTRIES:
            _CATALOG_CACHE.popitem(last=False)
    return rows


def _invalidate_catalog_cache(tenant: str, dataset: str) -> bool:
    """Drop the cached `(tenant, dataset)` entry. Idempotent.

    Bumps the per-key generation counter so an in-flight fetch refuses
    to install the now-stale rows.
    """
    key = (tenant, dataset)
    with _CATALOG_CACHE_LOCK:
        _CATALOG_CACHE_GEN[key] = _CATALOG_CACHE_GEN.get(key, 0) + 1
        return _CATALOG_CACHE.pop(key, None) is not None


def _on_catalog_notify(payload: dict) -> None:
    """Listener subscriber — evict the affected entry on a notify payload."""
    tenant = payload.get("tenant") or ""
    dataset = payload.get("dataset") or ""
    if tenant and dataset:
        _invalidate_catalog_cache(tenant, dataset)


def _catalog_cache_clear() -> None:
    """Test helper — flush every cached entry."""
    with _CATALOG_CACHE_LOCK:
        _CATALOG_CACHE.clear()
        _CATALOG_CACHE_GEN.clear()


# Match the v1_query opt-in: subscribe the cache invalidator to the
# LISTEN consumer when `RB_CATALOG_LISTEN=true`. Default-off; with the
# env unset, no thread is spawned and the TTL pull is the only channel.
# Reuse v1_query's `_truthy` (already imported above) to keep the truthy
# semantics provably identical between the two mirrors.
if _truthy(os.getenv("RB_CATALOG_LISTEN")):
    from services._common import catalog_listener as _catalog_listener

    _catalog_listener.subscribe(_on_catalog_notify)


# Classify a worker exception into a v1 error envelope before it is published
# on RESULT_READY. The classification answers two questions for a caller: *what
# kind of failure* (cache fs broken, S3 outage, anything else) and a HUMAN-SAFE
# message that does not leak internal paths, credentials, or the full traceback.
#
# Order matters — `PermissionError` is a subclass of `OSError`, so it must be
# matched first; `FileNotFoundError` is also an `OSError` so it sits ahead of
# the generic OSError branch as well. The unknown bucket is the catch-all
# `ephemeral_error` so an unexpected exception still propagates as a
# structured envelope (never `{matches:[]}`).
def _classify_error(exc: BaseException) -> Tuple[str, str]:
    """Map an exception to `(error_code, safe_message)` for the v1 envelope.

    Codes are drawn from / extended on the v1 error catalog (`docs/api/v1.md`):

      - `cache_unavailable` — the local shard cache directory is unreadable
        or unwritable. Self-host hint: check `CACHE_DIR` mount permissions.
      - `storage_unavailable` — the object-store fetch failed. The query is
        safe to retry once storage recovers.
      - `ephemeral_error` — generic; the exact cause is not safe to surface.

    The returned `safe_message` carries only the EXCEPTION CLASS NAME — never
    `str(exc)` — so a botocore `ClientError` cannot leak an S3 endpoint URL,
    a bucket name, or signed-URL parameters into the customer-visible error.
    """
    if isinstance(exc, PermissionError):
        return "cache_unavailable", "Shard cache is unreadable or unwritable"
    if isinstance(exc, shard_tier.CacheCapacityExceeded):
        # SSD-tier admission floor rejected a speculative arrival. Mirror of
        # the `_classify_hot_path_error` branch in v1_query.py; the
        # duplication is intentional to keep this module's import graph free
        # of services.query_api.
        return "cache_capacity_exceeded", "SSD cache tier is at capacity"
    if isinstance(exc, shard_tier.ShardTierTimeout):
        # SSD-tier coalescing timeout — the in-flight initiator on the tier
        # did not release the per-URI event within
        # `RB_SHARD_TIER_COALESCE_WAIT_S`. Same customer-visible semantics as
        # a botocore transient failure: retry-safe, surfaces as 503. Kept
        # symmetric with `_classify_hot_path_error` in v1_query.py; the two
        # classifiers are intentionally duplicated to keep the ephemeral
        # runner's import graph free of services.query_api.
        return "storage_unavailable", "Shard storage is temporarily unavailable"
    if isinstance(exc, FileNotFoundError):
        # A missing cache directory looks like FileNotFoundError on the cache
        # write path; a missing S3 key also surfaces as FileNotFoundError via
        # `_s3_get_object`. Both collapse to `storage_unavailable` — the shard
        # the catalog points at is not retrievable. The runner cannot tell
        # them apart at this layer and the caller's remediation is the same
        # (check storage health), so this is the intended bucketing.
        return "storage_unavailable", "Shard storage is temporarily unavailable"
    if _BotoClientError is not None and isinstance(exc, _BotoClientError):
        return "storage_unavailable", "Shard storage is temporarily unavailable"
    if isinstance(exc, OSError):
        # A generic OSError on the cache path (disk full, EIO, etc.) — bucket
        # with `cache_unavailable` so self-hosters get a specific signal.
        return "cache_unavailable", "Shard cache I/O error"
    return "ephemeral_error", f"Cold-shard query failed: {type(exc).__name__}"


def _ensure_cached(shard_uri: str) -> str:
    """Ensure a shard is present in local cache and return its path.

    FAISS's `read_index` needs a filesystem path, so an object-store shard
    (`s3://` or `memory://`) is fetched once into `CACHE_DIR`. There is no
    `file://` branch — RosalindDB is object-storage-first.

    The fetch is written **atomically** — to a unique temp file, then
    `os.replace`-renamed into place. When several RosalindDB processes share
    one `CACHE_DIR` (a multi-worker / multi-service deployment), a plain
    `open(path,"wb").write(...)` races: one process creates the file (so a
    concurrent `os.path.exists` sees it) while another reads it mid-write and
    `faiss.read_index` fails on the truncated bytes. An atomic rename means a
    reader sees either no file or the fully-written file — never a partial one.

    SSD-tier activation gate (mirror of the hot-path wiring in
    `services/query_api/v1_query.py`). When `RB_SHARD_TIER_BYTES` is set,
    delegation runs through `shard_tier.fetch(shard_uri)`. When the env is
    unset the legacy body below runs unchanged so a deployment that has not
    opted in sees unchanged behaviour.
    """
    if os.getenv("RB_SHARD_TIER_BYTES"):
        # Tier owns its own directory, single-flight, and eviction. Its
        # `ShardTierTimeout` and `FileNotFoundError` both classify to 503
        # via `_classify_error`.
        return shard_tier.fetch(shard_uri)

    os.makedirs(CACHE_DIR, exist_ok=True)
    if shard_uri.startswith("s3://") or shard_uri.startswith("memory://"):
        # `download_to` streams the GET to disk without buffering the whole
        # object in RAM (matches the hot path's `_ensure_cached`; both use
        # this pattern to avoid the multi-GB-shard OOM of the prior
        # `f.write(read_bytes(shard_uri))` approach).
        from adapters.storage.storage import download_to

        cache_key = shard_uri.split("://", 1)[1].replace("/", "_")
        path = os.path.join(CACHE_DIR, cache_key)
        if not os.path.exists(path):
            tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
            try:
                download_to(shard_uri, tmp)
                # Atomic publish — concurrent readers never see a partial file.
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
    raise ValueError("Unsupported shard uri")


def handle(
    tenant: str,
    dataset: str,
    vector: list[float],
    top_k: int,
    flt: Optional[Dict[str, Any]] = None,
):
    """Execute a top-K FAISS search against the newest shard for a dataset.

    Results carry the customer's original string `id` and `metadata`, not the
    raw FAISS int64 hash. FAISS `IndexIDMap2` only stores the SHA1-derived hash
    of each id, so we load the `{shard_uri}.meta.json` sidecar written by the
    index builder and map every hit back. This keeps the ephemeral
    `RESULT_READY` payload identical in shape to the hot path. A hit missing
    from the sidecar (defensive) falls back to the stringified hash with empty
    metadata. `score` is the raw FAISS L2 distance.

    Metadata filtering: when `flt` is a non-empty dict, the AND-of-equals
    predicate (`metadata_matches_filter`, shared with the hot path) is applied
    to each candidate. The filtered path is *exhaustive*: FAISS is asked for
    the full shard (`fetch_k = ntotal`) and, on an IVF index, every cell is
    scanned (`nprobe = nlist`) — a filter-match in an unprobed cell would
    otherwise be invisible. The predicate then filters the hits and the
    survivors — already nearest-first — are truncated to `top_k`. The search
    is exhaustive (every cell, every vector) and therefore exact for IVFFlat
    and flat indexes; a legacy IVF+PQ shard's per-vector distances stay
    PQ-approximate (full cell coverage fixes which cells are scanned, not
    PQ's lossy quantization). A filtered query returns exactly `min(top_k,
    total_matching)`. A query can still legitimately return fewer than `top_k`
    results (or zero) when the dataset genuinely contains fewer than `top_k`
    matching records — an exact answer, not an approximation or an error
    (identical to the hot path).
    """
    # Catalog cache wrapper. With the SSD tier off OR `RB_CATALOG_FRESHNESS_S=0`
    # this is a passthrough to `list_shards`; with both active, repeated
    # lookups for the same dataset within the TTL skip the Postgres round-trip.
    # Invalidated on notify when `RB_CATALOG_LISTEN=true` (the LISTEN consumer
    # is the latency optimisation; the TTL is the correctness mechanism).
    shards = _cached_list_shards(tenant, dataset)
    if not shards:
        return []
    latest = shards[0]
    has_filter = bool(flt)
    # Cold load decomposed into separate spans — `shard.download` and
    # `faiss.load_index` — so the ephemeral path is attributable the same way
    # the query hot path is; `faiss.search` then times the vector search
    # alone. All nest under the `ephemeral_query` span (and the upload/query
    # trace, via queue propagation).
    with shard_download_span(uri=latest["shard_uri"]):
        # `_ensure_cached` internally checks `RB_SHARD_TIER_BYTES` to
        # decide between the SSD tier and the legacy in-process body.
        # The URI is the cache key in either path.
        local = _ensure_cached(latest["shard_uri"])
    with faiss_load_index_span(uri=latest["shard_uri"], mmap=_MMAP_ENABLED):
        # Mmap parity with the hot path: when `RB_FAISS_MMAP=true` the cold
        # query in this runner must also avoid slurping a multi-GB shard into
        # RSS. The flag is captured at v1_query import time and re-exported so
        # the two services cannot drift apart on a deploy that flips the env.
        if _MMAP_ENABLED:
            index = faiss.read_index(
                local, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
            )
        else:
            index = faiss.read_index(local)
    sidecar = read_shard_sidecar(latest["shard_uri"])
    x = np.array([vector], dtype=np.float32)
    # `faiss.search` span — high-cardinality tenant/dataset attrs are correct
    # on a span. Now wraps the vector search alone.
    #
    # Page-fault sampler mirrors the hot path's accounting (see
    # `services/query_api/v1_query.py` around the matching block): read
    # `/proc/self/stat`'s `majflt` before and after the search and record the
    # delta to `rosalinddb.shard.page_faults`. On the mmap path this is the
    # operator-facing signal that an ephemeral cold-search had to fault pages
    # in from disk; with mmap off the delta stays at zero. Without this the
    # counter undercounted whenever an ephemeral cell ran.
    maj_before = _read_major_faults()
    with faiss_search_span(tenant=tenant, dataset=dataset, top_k=top_k):
        # With a filter, search the entire shard so no matching record can be
        # missed; without one, top_k is sufficient.
        if has_filter:
            # Exhaustive-when-filtered: fetch every vector AND, on an IVF
            # index, scan every cell (`nprobe = nlist`). Without full cell
            # coverage a filter-match in an unprobed cell stays invisible no
            # matter how large `fetch_k` is. A flat index is already exact, so
            # `_ivf_search_params` returns no params for it.
            #
            # IDSelector-based pre-filtering is the intended optimization for
            # large shards and is deliberately deferred: the full scan is
            # correct and fast at MVP scale (datasets capped at 100k vectors).
            fetch_k = int(getattr(index, "ntotal", 0)) or top_k
            search_params, _nprobe = _ivf_search_params(index, full_coverage=True)
            search_kwargs = (
                {"params": search_params} if search_params is not None else {}
            )
        else:
            fetch_k = top_k
            search_kwargs = {}
        distances, ids = index.search(x, fetch_k, **search_kwargs)
    maj_after = _read_major_faults()
    if maj_before is not None and maj_after is not None:
        obs_metrics.record_shard_page_faults(max(0, maj_after - maj_before))
    matches: list[dict] = []
    for i in range(min(fetch_k, len(ids[0]))):
        raw_id = int(ids[0][i])
        if raw_id == -1:
            continue
        entry = sidecar.get(str(raw_id))
        if entry is not None:
            match = {
                "id": entry.get("id", str(raw_id)),
                "score": float(distances[0][i]),
                "metadata": entry.get("metadata") or {},
            }
        else:
            match = {"id": str(raw_id), "score": float(distances[0][i]), "metadata": {}}
        # AND-of-equals filtering: drop candidates that violate the filter.
        if has_filter and not metadata_matches_filter(match["metadata"], flt):
            continue
        matches.append(match)
        # Hits arrive nearest-first; once we have top_k survivors, stop.
        if len(matches) >= top_k:
            break
    if has_filter:
        # Count the filtered query and its post-filter result size, mirroring
        # the hot path (`v1_query.v1_query`).
        obs_metrics.record_filtered_query()
        obs_metrics.record_filtered_result_count(len(matches))
    return matches


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for metrics endpoints."""

    def do_GET(self):
        """Handle GET requests for metrics."""
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(snapshot()).encode())
        elif self.path == "/prometheus":
            self._serve_prometheus()
        elif self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok", "service": "ephemeral_runner"}')
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_prometheus(self):
        """Serve Prometheus format metrics."""
        try:
            from prometheus_client import CollectorRegistry, generate_latest, Gauge
        except ImportError:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"prometheus-client not installed")
            return

        reg = CollectorRegistry()
        snap = snapshot()
        counters = snap.get("counters", {})
        gauges = snap.get("gauges", {})
        timers = snap.get("timers", {})

        # Export counters as gauges
        for name, value in counters.items():
            g = Gauge(f"ephemeral_{name}", f"ephemeral counter {name}", registry=reg)
            g.set(float(value))

        # Export gauges
        for name, value in gauges.items():
            g = Gauge(f"ephemeral_{name}", f"ephemeral gauge {name}", registry=reg)
            g.set(float(value))

        # Export timer stats
        for name, values in timers.items():
            if values:
                count = len(values)
                avg_ms = (sum(values) / count) * 1000.0
                Gauge(f"ephemeral_{name}_count", f"ephemeral timer {name} count", registry=reg).set(float(count))
                Gauge(f"ephemeral_{name}_avg_ms", f"ephemeral timer {name} avg ms", registry=reg).set(float(avg_ms))

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.end_headers()
        self.wfile.write(generate_latest(reg))

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


def start_metrics_server():
    """Start the metrics HTTP server in a background thread."""
    def run_server():
        server = HTTPServer(("0.0.0.0", METRICS_PORT), MetricsHandler)
        server.serve_forever()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    print(f"Metrics server started on port {METRICS_PORT}")


def main_loop():
    """Blocking loop that consumes and handles ephemeral query requests.

    Reliable-queue contract: a RUN_EPHEMERAL_QUERY message is `ack`-ed once the
    search completes and the RESULT_READY reply (if any) has been published; an
    UNHANDLED crash `nack`s it for redelivery (then dead-lettering past
    `QUEUE_MAX_ATTEMPTS`). On `SIGTERM` the loop stops pulling new messages and
    exits cleanly.

    A search that raises (storage outage, cache fs unwritable, FAISS load
    failure, …) now publishes a STRUCTURED ERROR ENVELOPE to `reply_to` before
    re-raising so the status poll surfaces a real failure (HTTP 503) rather than
    an empty 200. The NACK + DLQ behaviour is preserved: every delivery attempt
    still publishes its own envelope, and a message exhausting
    `QUEUE_MAX_ATTEMPTS` is dead-lettered exactly as before.
    """
    migrate()
    install_signal_handlers()
    start_metrics_server()
    while not should_stop():
        msg = consume("RUN_EPHEMERAL_QUERY", block=True, timeout=1.0)
        if not msg:
            continue
        try:
            _handle_ephemeral(msg)
        except Exception as exc:  # noqa: BLE001
            # The error envelope was already published inside `_handle_ephemeral`
            # (so the status poll surfaces a 503, not a silent empty 200);
            # NACK lets the reliable queue retry up to `QUEUE_MAX_ATTEMPTS`
            # before dead-lettering. Logging carries the exception class + a
            # safe message so an operator can correlate the DLQ growth with
            # the failure shape without leaking paths/credentials.
            print(
                "ephemeral: handler error, nacking message: "
                f"exc_class={type(exc).__name__} msg={exc!s}"
            )
            nack(msg, requeue=True)
            continue
        ack(msg)
    print("ephemeral: shutdown signal received — exiting consume loop")


def _publish_error_envelope(
    reply_to: str,
    correlation_id: Optional[str],
    dataset: str,
    code: str,
    message: str,
    latency_ms: int,
) -> None:
    """Publish a structured failure envelope on the reply queue.

    The `RESULT_READY` consumer stashes the envelope under `correlation_id` in
    the shared result store, so `GET /v1/query/status/{job_id}` finds an
    `ok: false` payload instead of a missing key and surfaces it as HTTP 503.

    An unaddressable message (`correlation_id` absent) cannot be routed back
    to any client — we still record the failure in the trace/log but skip the
    publish (a message on `RESULT_READY` with no correlation id is unreachable
    by any status poll).
    """
    if not correlation_id:
        return
    publish(
        reply_to,
        {
            "correlation_id": correlation_id,
            "dataset": dataset,
            "ok": False,
            "error": {"code": code, "message": message},
            "latency_ms": latency_ms,
        },
    )


def _handle_ephemeral(msg) -> None:
    """Run one RUN_EPHEMERAL_QUERY search and publish its RESULT_READY reply.

    On a handler exception the worker publishes an EXPLICIT ERROR ENVELOPE
    (`{ok: false, error: {code, message}}`) to `reply_to` before re-raising,
    instead of silently NACKing and leaving the caller's status poll to see
    `{matches: []}` forever. The exception is then re-raised so `main_loop`
    still NACKs the queue message — both behaviours are needed: the envelope
    unblocks the caller, the NACK lets the reliable queue retry up to
    `QUEUE_MAX_ATTEMPTS` before dead-lettering.
    """
    dataset = msg["dataset"]
    tenant = msg.get("tenant", "default")
    vector = msg["vector"]
    top_k = int(msg.get("top_k", 10))
    # The filter rides on the RUN_EPHEMERAL_QUERY message; absent / null
    # collapses to no filtering.
    flt = msg.get("filter") or None
    correlation_id = msg.get("correlation_id")
    reply_to = msg.get("reply_to", "RESULT_READY")
    start = time.time()
    # `ephemeral_query` span — child of the originating `POST /v1/query`
    # trace (the queue adapter attached the producer context on consume()).
    # The `span(...)` helper records the exception and sets ERROR status on
    # the span automatically, so the trace already shows the failure.
    try:
        with ephemeral_query_span(tenant=tenant, dataset=dataset):
            matches = handle(tenant, dataset, vector, top_k, flt)
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.time() - start) * 1000.0)
        code, safe_message = _classify_error(exc)
        # Structured log: dataset, tenant, exception class, raw message.
        # `repr(exc)` instead of `str(exc)` so a bare exception (e.g.
        # `PermissionError()` with no message) is still distinguishable in
        # logs from one with a path attached.
        print(
            "ephemeral: search failed: "
            f"tenant={tenant} dataset={dataset} "
            f"exc_class={type(exc).__name__} exc={exc!r} code={code}"
        )
        _publish_error_envelope(
            reply_to, correlation_id, dataset, code, safe_message, latency_ms
        )
        # Re-raise so `main_loop` runs its NACK + retry logic. The envelope
        # is already on the wire so the caller is unblocked regardless.
        raise
    latency_ms = int((time.time() - start) * 1000.0)
    if correlation_id:
        publish(
            reply_to,
            {
                "correlation_id": correlation_id,
                "dataset": dataset,
                "ok": True,
                "matches": matches,
                "latency_ms": latency_ms,
            },
        )


if __name__ == "__main__":
    main_loop()

