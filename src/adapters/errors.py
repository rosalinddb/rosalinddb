"""Canonical home for RosalindDB's custom exception hierarchy.

This is a LEAF module: it imports only the standard library, so it can be
imported from anywhere (adapters, services, tests) without risking an import
cycle. Every custom exception that used to be defined ad-hoc in a feature
module is defined HERE exactly once, then re-exported from its original
definition site for backward compatibility.

Back-compat & class identity
----------------------------
Each original definition site (e.g. ``adapters.state.state``,
``adapters.storage.shard_tier``, ``services.query_api.v1_query``,
``services.auth.quota``) now imports its class from this module and re-exports
it. Because there is a SINGLE definition object that every site shares,
``isinstance`` checks and ``except`` clauses keep working identically whether a
caller reached the class via the old path or the new one.

Hierarchy notes
---------------
``RosalindDBError`` is a ``RuntimeError`` subclass. All of the operational /
classifier exceptions were ``RuntimeError`` subclasses before this refactor, and
that is RELIED UPON: several frames catch a bare ``except RuntimeError`` and
must keep catching these. Folding them under ``RosalindDBError(RuntimeError)``
preserves that exactly.

``RateLimited`` is the one exception that was NOT a ``RuntimeError`` (it derives
from ``Exception`` and carries a pre-built ``JSONResponse``). Making it a
``RuntimeError`` here would be a behavior change — a ``except RuntimeError``
frame would start catching it. To stay strictly behavior-preserving it remains
a plain ``Exception`` subclass; it still gains a stable ``.error_code`` so the
single source of truth for error codes covers it too.

``.error_code``
---------------
Every class carries a stable ``error_code`` class attribute. This lets the two
(deliberately duplicated, to avoid an import cycle) classifier tables in
``v1_query._classify_hot_path_error`` and ``ephemeral_runner._classify_error``
collapse onto a single authoritative mapping over time. The values below match
the wire codes those tables emit today.
"""

from __future__ import annotations

from typing import Optional, Tuple


def error_envelope(status_code: int, code: str, message: str, details: Optional[dict] = None):
    """Build the canonical v1 error-envelope `JSONResponse`.

    Single source of truth for the ``{"error": {"code", "message"[, "details"]}}``
    body that the service routers return. The four service-layer ``_err`` copies
    (``services.query_api.v1_query``, ``services.auth.auth``,
    ``services.source_registry.main``, ``services.query_api.query_proxy``) all
    delegate here so the wire shape cannot drift.

    The body is byte-identical to every previous copy: ``details`` is only added
    to the envelope when it is not ``None`` (so the ``query_proxy`` call site,
    which never passes ``details``, still emits exactly ``{"error": {"code",
    "message"}}``).

    ``fastapi`` is imported lazily INSIDE this function on purpose: this module
    is a stdlib-only LEAF that low-level adapters (``adapters.state.state``,
    ``adapters.storage.shard_tier``) import at module load. Importing
    ``fastapi`` at module top would drag the web framework into those adapters'
    import graph — a behavior change. The lazy import keeps the leaf property
    while still letting the service layer share this helper.
    """
    from fastapi.responses import JSONResponse

    body: dict = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=body)


class RosalindDBError(RuntimeError):
    """Base class for RosalindDB's operational exceptions.

    A ``RuntimeError`` subclass so that existing ``except RuntimeError`` frames
    continue to catch every error derived from it (this is load-bearing —
    several call sites and the shard-tier docstring rely on it).

    Subclasses set a stable ``error_code`` class attribute that maps to the
    customer-facing wire code / observability signal.
    """

    #: Stable wire/observability code. Overridden by each subclass.
    error_code: str = "rosalinddb_error"


class PoolCheckoutTimeout(RosalindDBError):
    """The pool stayed exhausted past the block-with-timeout deadline.

    Raised by `pooled_conn()` when every retry of `getconn()` hit a
    `PoolError` and the total checkout deadline elapsed. It signals a genuine
    *sustained* overload — the ASGI apps map it to HTTP 503 (service
    unavailable), never a 500. A transient exhaustion that clears within the
    deadline is invisible: the checkout simply blocks then succeeds.
    """

    error_code = "service_unavailable"


class RecallUnavailable(RosalindDBError):
    """The recall (pgvector) tier could not be reached for this operation.

    A TYPED boundary error raised by the recall READ path (`recall_search`)
    when the recall store is unreachable — a connection drop / refused socket /
    TLS reset (`psycopg2.OperationalError`/`InterfaceError`) or a SUSTAINED
    recall-pool exhaustion (`PoolCheckoutTimeout`). It wraps the underlying
    cause as `__cause__` so logs keep the full detail, but the customer-facing
    classifier surfaces only the typed name.

    Why a typed wrapper rather than classifying `psycopg2.OperationalError`
    directly: the SAME psycopg2 exception type is raised by the control-plane /
    cold path (catalog reads, quota), and a blanket "OperationalError ->
    recall_unavailable" rule would misattribute a control-plane outage to the
    recall tier. Raising this type ONLY at the recall boundary lets the query
    path classify it precisely as a retryable 503 `recall_unavailable` (a
    transient recall outage the client should retry), distinct from both the
    generic `ephemeral_error` 500 and the write-side `recall_write_failed`.
    The query path must NOT silently fall back to consolidated-only results: a
    recall outage means recent (unconsolidated) writes are unreadable, so a
    silent cold-only answer would drop read-your-writes without signal. A 503
    tells the client to retry; a silent degrade lies.
    """

    error_code = "recall_unavailable"


class DownloadCoalescingTimeout(RosalindDBError):
    """A coalesced waiter exceeded its deadline on someone else's download.

    Raised by `_ensure_cached` when the per-URI in-flight event was not set
    within `_DOWNLOAD_COALESCE_WAIT_S` seconds. Surfaced as a distinct
    exception (rather than a generic `TimeoutError`) so the caller can map it
    to a specific error code / observability signal — a waiter timing out is
    a different operational condition from the initiator's download itself
    failing.
    """

    error_code = "storage_unavailable"


class ShardTierTimeout(RosalindDBError):
    """A coalesced waiter exceeded `RB_SHARD_TIER_COALESCE_WAIT_S`.

    Raised by `fetch()` when the per-URI in-flight event was not set within
    the bounded wait. Distinct from a generic `TimeoutError` so the caller
    can map it to a specific error code / observability signal — a waiter
    timing out is a different operational condition from the initiator's
    download itself failing.
    """

    error_code = "storage_unavailable"


class CacheCapacityExceeded(RosalindDBError):
    """`prewarm()` could not admit a speculative shard.

    Raised when the tier is at its byte budget and every candidate for
    eviction is younger than `_MIN_RESIDENT_S` (the admission floor). The
    intent is to keep recently-arrived shards stable under a
    write storm: a prewarm that lands while the tier is full of fresh
    arrivals is rejected, the operator / dashboard sees a
    `cache_capacity_exceeded` 503, and the floor's discrimination signal
    (queries-under-load beat speculative arrivals) holds.

    `RuntimeError` subclass so callers that catch `RuntimeError` still
    catch this, and so the classifier branches in `_classify_hot_path_error`
    / `_classify_error` can sit alongside the existing `RuntimeError`-catching
    frames without re-ordering risk.
    """

    error_code = "cache_capacity_exceeded"


# Pluggable factory for the `RateLimited` response body. `services.auth.quota`
# registers `_rate_limited_response` here at import time (via
# `set_rate_limited_response_factory`). Keeping the factory injectable means the
# response-building concern (FastAPI `JSONResponse` + the rate-limit config it
# reads) stays in `quota` while the CLASS lives here once — so `quota.RateLimited`
# and `errors.RateLimited` are the SAME object and `isinstance` / the registered
# exception handler keep working. `quota` is always imported before a
# `RateLimited()` is ever raised (it is only raised inside `quota`), so the
# factory is registered by the time the constructor needs it.
_rate_limited_response_factory = None


def set_rate_limited_response_factory(factory) -> None:
    """Register the callable that builds a `RateLimited.response` (429 body).

    Called once by `services.auth.quota` at import. Idempotent.
    """
    global _rate_limited_response_factory
    _rate_limited_response_factory = factory


class RateLimited(Exception):
    """Raised by the `rate_limit` dependency when a bucket is exhausted.

    Carries a pre-built JSONResponse; the app-level handler installed by
    `install_rate_limit_handler` returns it verbatim. We use an exception
    rather than returning a response directly because FastAPI dependencies
    cannot short-circuit a request with a response object.

    NOTE: this is intentionally NOT a `RosalindDBError`/`RuntimeError` subclass
    — it was a plain `Exception` before the errors module existed, and an
    `except RuntimeError` frame must not start catching it. It still exposes a
    stable `error_code` so this module remains the single source of truth for
    wire codes.
    """

    error_code = "rate_limited"

    def __init__(self) -> None:
        # Build the 429 body via the factory `quota` registered. The factory is
        # the original `_rate_limited_response()`; this keeps the response shape
        # byte-for-byte identical to the pre-refactor inline construction.
        if _rate_limited_response_factory is None:  # pragma: no cover - defensive
            raise RuntimeError(
                "RateLimited response factory is not registered; "
                "import services.auth.quota before raising RateLimited"
            )
        self.response = _rate_limited_response_factory()
        super().__init__("rate_limited")


def classify_query_error(
    exc: BaseException,
    *,
    default_message_prefix: str = "Query failed",
) -> Tuple[str, str]:
    """Map a query-path exception to a v1 ``(error_code, safe_message)`` tuple.

    Single source of truth for the classification table that used to live —
    byte-identical except for the fallback message — in both
    ``services.query_api.v1_query._classify_hot_path_error`` (the synchronous
    hot path) and ``services.ephemeral_runner.run._classify_error`` (the
    cold-shard runner). Both now delegate here so the two query paths surface
    the SAME error codes for the SAME failure shapes and cannot drift again.

    The ONLY behavior that genuinely differed between the two call sites was the
    generic catch-all message (``"Query failed: <Cls>"`` on the hot path vs
    ``"Cold-shard query failed: <Cls>"`` in the runner). That difference is
    preserved verbatim via ``default_message_prefix`` — each caller passes its
    own prefix, so the customer-visible message is unchanged at both sites.

    Lives in ``adapters.errors`` (this module) because it depends ONLY on the
    exception hierarchy defined here plus an optional ``botocore`` import — all
    adapter-layer. Keeping it here satisfies the one-way import rule (``adapters``
    never imports from ``services``), so both services can import it.

    Branch ORDER is load-bearing and matches the prior copies exactly:

      - ``RecallUnavailable`` first — a typed boundary error from the recall
        (pgvector) read path. Distinct, retryable ``recall_unavailable`` 503,
        never the generic ``ephemeral_error`` 500 nor the write-side
        ``recall_write_failed``.
      - ``PermissionError`` before the generic ``OSError`` (it IS an ``OSError``)
        so a cache fs permission failure classifies as ``cache_unavailable``.
      - ``CacheCapacityExceeded`` — SSD-tier admission floor rejected a
        speculative arrival; distinct ``cache_capacity_exceeded`` 503 so an
        operator can tell capacity pressure from a storage outage.
      - ``DownloadCoalescingTimeout`` / ``ShardTierTimeout`` — bounded-wait
        coalescing timeouts; both collapse to ``storage_unavailable`` 503 so a
        client retry policy need not distinguish which layer timed out. The
        runner never raises ``DownloadCoalescingTimeout`` (it has no coalescing
        path), so listing it here is a no-op there and changes nothing.
      - ``FileNotFoundError`` before generic ``OSError`` (it IS an ``OSError``)
        → ``storage_unavailable`` (missing cache dir / missing S3 key).
      - botocore ``ClientError`` (optional import) → ``storage_unavailable``.
      - generic ``OSError`` → ``cache_unavailable`` (disk full, EIO, ...).
      - everything else → ``ephemeral_error`` with the per-caller prefix.

    ``safe_message`` is built from the exception CLASS NAME only; ``str(exc)``
    is never surfaced — a botocore ``ClientError`` carries an endpoint URL and
    sometimes signed-URL params that must not leak to the customer.
    """
    if isinstance(exc, RecallUnavailable):
        return "recall_unavailable", "Recall tier is temporarily unavailable"
    if isinstance(exc, PermissionError):
        return "cache_unavailable", "Shard cache is unreadable or unwritable"
    if isinstance(exc, CacheCapacityExceeded):
        return "cache_capacity_exceeded", "SSD cache tier is at capacity"
    if isinstance(exc, (DownloadCoalescingTimeout, ShardTierTimeout)):
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
    return "ephemeral_error", f"{default_message_prefix}: {type(exc).__name__}"
