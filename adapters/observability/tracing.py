from __future__ import annotations

"""Manual span helpers for the RosalindDB pipeline.

The contract pins these span names: `validate_dataset`, `build_index`,
`faiss.search`, `ephemeral_query`, `landing.write`, `landing.read`.

The query hot path additionally decomposes into `query.hot_search` (the
parent) with `state.list_shards`, `shard.download` and `faiss.load_index`
children, so `faiss.search` finally times the vector search alone — see
`services/query_api/v1_query.py`. `state.connect` marks each fresh,
unpooled Postgres connection so a repeated-connect pattern is visible.

Unlike metrics, spans MAY carry high-cardinality attributes (tenant id,
dataset name) — that is exactly what traces are for. Use `span(...)` as a
context manager and pass per-entity detail freely.

When the SDK is disabled / not initialised, `get_tracer` returns a no-op
tracer; `span()` still works (it yields a non-recording span) so call sites
need no guards.
"""

from contextlib import contextmanager

from opentelemetry import trace as _trace_api

_TRACER_NAME = "rosalinddb.pipeline"


def _tracer():
    return _trace_api.get_tracer(_TRACER_NAME)


@contextmanager
def span(name: str, attributes: dict | None = None):
    """Start a manual span as a context manager.

    `attributes` may include high-cardinality detail (tenant id, dataset
    name) — that is correct for traces. The span records an exception and
    sets an ERROR status if the body raises, then re-raises.
    """
    with _tracer().start_as_current_span(name) as sp:
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    sp.set_attribute(key, value)
        try:
            yield sp
        except Exception as exc:  # noqa: BLE001
            sp.record_exception(exc)
            sp.set_status(_trace_api.Status(_trace_api.StatusCode.ERROR, str(exc)))
            raise


# --- named convenience wrappers (contract span names) --------------------


def validate_dataset_span(tenant: str | None = None, dataset: str | None = None):
    """`validate_dataset` span — validator worker."""
    return span("validate_dataset", {"rosalinddb.tenant_id": tenant, "rosalinddb.dataset": dataset})


def build_index_span(tenant: str | None = None, dataset: str | None = None):
    """`build_index` span — index builder."""
    return span("build_index", {"rosalinddb.tenant_id": tenant, "rosalinddb.dataset": dataset})


def faiss_search_span(
    tenant: str | None = None,
    dataset: str | None = None,
    top_k: int | None = None,
    fetch_k: int | None = None,
):
    """`faiss.search` span — hot query path and ephemeral runner.

    `top_k` is the caller's requested result count. `fetch_k` is the number of
    candidates actually pulled from FAISS — equal to `top_k` for an unfiltered
    query, but far larger when a metadata filter forces over-fetching.
    Recording the real `fetch_k` keeps latency triage honest: a slow span with
    `top_k=10` is misleading if FAISS actually searched for 1000 candidates.
    """
    attrs = {"rosalinddb.tenant_id": tenant, "rosalinddb.dataset": dataset}
    if top_k is not None:
        attrs["rosalinddb.top_k"] = top_k
    if fetch_k is not None:
        attrs["rosalinddb.fetch_k"] = fetch_k
    return span("faiss.search", attrs)


def ephemeral_query_span(tenant: str | None = None, dataset: str | None = None):
    """`ephemeral_query` span — ephemeral runner job handler."""
    return span("ephemeral_query", {"rosalinddb.tenant_id": tenant, "rosalinddb.dataset": dataset})


# --- query hot-path decomposition ----------------------------------------
#
# A `POST /v1/query` trace used to show one opaque `faiss.search` span fusing
# four operations (catalog lookup, shard download from object storage, index
# deserialize, the actual search). These helpers split it so latency is
# attributable.


def hot_search_span(tenant: str | None = None, dataset: str | None = None):
    """`query.hot_search` span — parent of the whole query hot path.

    The catalog lookup, shard download, index deserialize and vector search
    all nest under this so a query trace decomposes cleanly.
    """
    return span("query.hot_search", {"rosalinddb.tenant_id": tenant, "rosalinddb.dataset": dataset})


def list_shards_span(tenant: str | None = None, dataset: str | None = None):
    """`state.list_shards` span — the Postgres shard-catalog lookup.

    Previously invisible: it ran before the `faiss.search` span opened.
    """
    return span("state.list_shards", {"rosalinddb.tenant_id": tenant, "rosalinddb.dataset": dataset})


def state_connect_span(reused: bool | None = None):
    """`state.connect` span — one Postgres connection checkout.

    `_conn()` / `pooled_conn()` now hand out connections from an application-
    side pool. A checkout is still traced, but the span's cost now reflects
    reality: ~0ms when a live pooled connection is reused, and only slow when
    the pool genuinely opens a new backend (TCP + TLS + auth handshake).

    `reused` annotates which case this checkout was so a trace distinguishes a
    free reuse from a real connect: it sets `rosalinddb.connection.reused`
    (True/False) and, when known, names the span `state.connect.reused` vs
    `state.connect.open`. A `state.connect` with no annotation is the legacy
    unpooled/dedicated path (`dataset_build_lock`).
    """
    if reused is None:
        return span("state.connect")
    name = "state.connect.reused" if reused else "state.connect.open"
    return span(name, {"rosalinddb.connection.reused": reused})


def shard_download_span(uri: str | None = None):
    """`shard.download` span — faulting a shard from object storage into the local cache."""
    return span("shard.download", {"rosalinddb.uri": uri})


def faiss_load_index_span(uri: str | None = None, mmap: bool | None = None):
    """`faiss.load_index` span — deserialising a FAISS index from disk.

    `mmap=True` indicates the load went through `IO_FLAG_MMAP | IO_FLAG_READ_ONLY`;
    `mmap=False` indicates the legacy `read_index(path)` deserialise. When
    `None` (default) no `rosalinddb.mmap` attribute is stamped — preserves
    backward compat for callers that have not been updated to pass the kwarg.
    """
    attrs: dict = {"rosalinddb.uri": uri}
    if mmap is not None:
        attrs["rosalinddb.mmap"] = mmap
    return span("faiss.load_index", attrs)


def landing_write_span(uri: str | None = None):
    """`landing.write` span — storage adapter write_bytes."""
    return span("landing.write", {"rosalinddb.uri": uri})


def landing_read_span(uri: str | None = None):
    """`landing.read` span — storage adapter read/open_reader."""
    return span("landing.read", {"rosalinddb.uri": uri})
