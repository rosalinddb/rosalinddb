"""Unit coverage for query hot-path tracing decomposition (obs/query-path-spans).

Hermetic — no Docker, no network. Builds a real FAISS shard in `memory://`
storage, runs `_consolidated_search` directly, and asserts the trace decomposes into
attributable child spans instead of one opaque `faiss.search` span.

The suite conftest sets `OTEL_SDK_DISABLED=true`, which only short-circuits
`init_observability` — the OTel *API* still honours whatever `TracerProvider`
is installed. So each test installs an isolated SDK `TracerProvider` with an
`InMemorySpanExporter`, captures the spans `_consolidated_search` emits, and restores
the previous provider afterwards.
"""
from __future__ import annotations

import json

import faiss  # type: ignore
import numpy as np
import pytest
from opentelemetry import trace as _trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import services.query_api.v1_query as v1q
from adapters.state import state as state_mod
from adapters.storage import storage as storage_mod


@pytest.fixture
def captured_spans(monkeypatch):
    """Install an isolated in-memory TracerProvider; yield the exporter.

    OTel forbids re-setting the global provider, so we keep the existing one,
    swap our SDK provider into the API's private slot for the test, capture
    spans, then restore. The suite sets `OTEL_SDK_DISABLED=true` (the SDK
    `TracerProvider` reads it at construction), so it is cleared here just
    while this isolated provider is built.
    """
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(_trace_api, "_TRACER_PROVIDER", provider, raising=False)
    yield exporter
    exporter.clear()


def _build_shard(tenant: str, dataset: str, dim: int = 8, n: int = 64):
    """Write a real FAISS shard + sidecar to memory:// and register it.

    Returns the shard id. Mirrors what the index builder produces: an
    `IndexIDMap2` `.bin` plus a `{shard_uri}.meta.json` sidecar.
    """
    rng = np.random.default_rng(7)
    vecs = rng.random((n, dim), dtype=np.float32)
    ids = np.arange(1, n + 1, dtype=np.int64)
    inner = faiss.IndexFlatL2(dim)
    index = faiss.IndexIDMap2(inner)
    index.add_with_ids(vecs, ids)

    shard_uri = f"memory://shards/{tenant}/{dataset}/shard.bin"
    storage_mod.write_bytes(shard_uri, faiss.serialize_index(index).tobytes())
    sidecar = {str(int(i)): {"id": f"r{int(i)}", "metadata": {}} for i in ids}
    storage_mod.write_bytes(
        f"{shard_uri}.meta.json", json.dumps(sidecar).encode("utf-8")
    )
    return state_mod.add_shard(
        tenant, dataset, shard_uri, "chk", n, "flat", "full", []
    )


@pytest.fixture
def shard_env(tmp_path, monkeypatch):
    """Fresh storage + state + shard cache + isolated CACHE_DIR per test."""
    storage_mod.memory_reset()
    state_mod._MEM_SHARDS.clear()
    v1q.cache_clear()
    monkeypatch.setattr(v1q, "CACHE_DIR", str(tmp_path / "shards"))
    yield


def _names(exporter):
    return [s.name for s in exporter.get_finished_spans()]


def _by_name(exporter, name):
    for s in exporter.get_finished_spans():
        if s.name == name:
            return s
    return None


def test_cold_query_emits_decomposed_child_spans(captured_spans, shard_env):
    """A cold `_consolidated_search` emits a `query.hot_search` parent with the R2
    download, FAISS deserialize and pure vector search as separate children."""
    _build_shard("t1", "ds1")
    out = v1q._consolidated_search("t1", "ds1", [0.1] * 8, top_k=5)
    assert out is not None
    _matches, mode = out
    assert mode == "cold"

    names = _names(captured_spans)
    for expected in (
        "query.hot_search",
        "state.list_shards",
        "shard.download",
        "faiss.load_index",
        "faiss.search",
    ):
        assert expected in names, f"missing span {expected!r} in {names}"


def test_child_spans_nest_under_hot_search_parent(captured_spans, shard_env):
    """Every query child span is parented to the `query.hot_search` span."""
    _build_shard("t2", "ds2")
    v1q._consolidated_search("t2", "ds2", [0.1] * 8, top_k=5)

    parent = _by_name(captured_spans, "query.hot_search")
    assert parent is not None
    parent_span_id = parent.context.span_id
    for child in ("state.list_shards", "shard.download", "faiss.load_index", "faiss.search"):
        sp = _by_name(captured_spans, child)
        assert sp is not None, f"{child} not emitted"
        assert sp.parent is not None and sp.parent.span_id == parent_span_id, (
            f"{child} is not a child of query.hot_search"
        )


def test_faiss_search_span_covers_only_the_search(captured_spans, shard_env):
    """`faiss.search` must NOT also contain the download/deserialize — those
    are now their own spans, so `faiss.search` starts after `faiss.load_index`."""
    _build_shard("t3", "ds3")
    v1q._consolidated_search("t3", "ds3", [0.1] * 8, top_k=5)

    search = _by_name(captured_spans, "faiss.search")
    load = _by_name(captured_spans, "faiss.load_index")
    assert search is not None and load is not None
    # The pure search starts only after the index is deserialized.
    assert search.start_time >= load.end_time


def test_warm_query_skips_cold_load_spans(captured_spans, shard_env):
    """A cache-hit query emits `faiss.search` but NOT the cold-load spans —
    the warm path is obviously distinguishable in a trace."""
    _build_shard("t4", "ds4")
    v1q._consolidated_search("t4", "ds4", [0.1] * 8, top_k=5)  # cold: populate cache
    captured_spans.clear()

    _matches, mode = v1q._consolidated_search("t4", "ds4", [0.1] * 8, top_k=5)
    assert mode == "hot"
    names = _names(captured_spans)
    assert "query.hot_search" in names
    assert "faiss.search" in names
    assert "shard.download" not in names
    assert "faiss.load_index" not in names


def test_search_span_keeps_existing_attributes(captured_spans, shard_env):
    """The decomposition keeps tenant/dataset/top_k/fetch_k attributes."""
    _build_shard("t5", "ds5")
    v1q._consolidated_search("t5", "ds5", [0.1] * 8, top_k=5)

    search = _by_name(captured_spans, "faiss.search")
    assert search is not None
    attrs = dict(search.attributes or {})
    assert attrs.get("rosalinddb.tenant_id") == "t5"
    assert attrs.get("rosalinddb.dataset") == "ds5"
    assert attrs.get("rosalinddb.top_k") == 5
    assert attrs.get("rosalinddb.fetch_k") == 5


# --- overlap (#31): recall span is a SIBLING of query.hot_search ----------
#
# The recall scan runs on a worker thread for the consolidated/recall overlap.
# `recall.search` is an INDEPENDENT, parallel operation — a SIBLING of
# `query.hot_search` under the REQUEST span, NOT a child of `query.hot_search`
# (the consolidated search). `query.hot_search` must time the consolidated work
# (resolve + inline FAISS) ALONE; the recall wait (the join) is moved OUTSIDE that
# span so it never inflates the consolidated-search latency bench/attribution read
# off `query.hot_search`.
#
# OTel's current-context is thread-local and does NOT auto-propagate to a new
# thread, so without explicit context propagation the `recall.search` span (opened
# INSIDE `recall_search`) would become an ORPHANED trace root. `run_query` captures
# the context from BEFORE entering `hot_search_span` (the request-span context) and
# re-attaches it inside the worker, so `recall.search` parents to the REQUEST span
# as a sibling of `query.hot_search`. These tests open a request span around
# `run_query` and assert the resulting parentage against a real in-memory SDK.


def _recall_search_with_span(tenant, dataset, vec, top_k, watermark, flt):
    """Stand-in for `recall_search` that opens the real `recall.search` span.

    Imitates the production `recall_search` (which opens `recall.search` via the
    tracing helper) without touching pgvector, so the span-parentage assertion is
    exercised purely through `run_query`'s thread/context plumbing. Runs in the
    worker thread, so the span it opens is the canary for context propagation.
    """
    from adapters.observability.tracing import recall_search_span

    with recall_search_span(tenant=tenant, dataset=dataset, top_k=top_k, watermark=watermark):
        return ({"r1"}, [{"id": "r1", "score": 0.0, "metadata": {}, "deleted": False}])


def _request_span():
    """Open a real `request` span so `run_query`'s children have a parent.

    In production `run_query` runs inside the ASGI request span; the unit test
    supplies an equivalent enclosing span so `query.hot_search` and `recall.search`
    have a common parent to be siblings UNDER.
    """
    from adapters.observability.tracing import span

    return span("request", {})


def test_recall_span_is_sibling_of_hot_search_under_request_span(
    captured_spans, shard_env, monkeypatch
):
    """With the union on, `recall.search` is a CHILD of the REQUEST span and a
    SIBLING of `query.hot_search` — NOT a child of `query.hot_search`. Recall is an
    independent, parallel operation, not part of the consolidated search; and the
    OTel context still propagated across the worker thread (no orphaned root)."""
    _build_shard("tr", "dsr")
    monkeypatch.setattr(v1q, "recall_enabled", lambda: True)
    monkeypatch.setattr(v1q, "recall_search", _recall_search_with_span)

    with _request_span():
        out = v1q.run_query("tr", v1q._ParsedQuery("dsr", [0.1] * 8, 5, None, {}))
    assert isinstance(out, dict) and "matches" in out

    request = _by_name(captured_spans, "request")
    hot = _by_name(captured_spans, "query.hot_search")
    recall = _by_name(captured_spans, "recall.search")
    assert request is not None, "request span missing"
    assert hot is not None, "query.hot_search span missing"
    assert recall is not None, "recall.search span missing"
    # recall.search must NOT be an orphaned root — context propagated across the
    # worker thread.
    assert recall.parent is not None, "recall.search became an orphaned trace root"
    # recall.search is a CHILD of the REQUEST span (sibling of query.hot_search),
    # NOT a child of query.hot_search.
    assert recall.parent.span_id == request.context.span_id, (
        "recall.search must parent to the REQUEST span (sibling of query.hot_search)"
    )
    assert recall.parent.span_id != hot.context.span_id, (
        "recall.search must NOT be a child of query.hot_search — recall is a "
        "parallel, independent operation, not part of the consolidated search"
    )
    # query.hot_search is itself a child of the request span — so recall and
    # hot_search are siblings.
    assert hot.parent is not None and hot.parent.span_id == request.context.span_id
    # All three share one trace.
    assert recall.context.trace_id == request.context.trace_id
    assert hot.context.trace_id == request.context.trace_id


def test_hot_search_duration_excludes_the_recall_wait(
    captured_spans, shard_env, monkeypatch
):
    """`query.hot_search` measures the CONSOLIDATED search ONLY, not the recall wait.

    The recall fake sleeps far longer than the (fast) inline FAISS search. Because
    the recall-future join is moved OUTSIDE `query.hot_search`, the span must END
    before the slow recall future resolves — so its duration is ~FAISS-only and
    excludes the recall wait. Proven by: `query.hot_search.end_time` is well below
    the slow recall span's `end_time`, AND `recall.search` is NOT a descendant of
    `query.hot_search`.
    """
    import time as _time

    _build_shard("th", "dsh")
    RECALL_SLEEP_S = 0.20  # recall is the slow leg; FAISS on a tiny shard is ~ms

    def _slow_recall(tenant, dataset, vec, top_k, watermark, flt):
        from adapters.observability.tracing import recall_search_span

        with recall_search_span(tenant=tenant, dataset=dataset, top_k=top_k, watermark=watermark):
            _time.sleep(RECALL_SLEEP_S)
            return (set(), [])

    monkeypatch.setattr(v1q, "recall_enabled", lambda: True)
    monkeypatch.setattr(v1q, "recall_search", _slow_recall)

    with _request_span():
        v1q.run_query("th", v1q._ParsedQuery("dsh", [0.1] * 8, 5, None, {}))

    hot = _by_name(captured_spans, "query.hot_search")
    recall = _by_name(captured_spans, "recall.search")
    assert hot is not None and recall is not None
    # The consolidated span closed BEFORE the slow recall future resolved — the
    # recall wait is NOT inside query.hot_search.
    assert hot.end_time < recall.end_time, (
        "query.hot_search ended after recall.search — the recall wait leaked into "
        "the consolidated-search span"
    )
    # And the consolidated span's own duration is far below the recall sleep —
    # i.e. it is FAISS-only, not max(FAISS, recall).
    hot_duration_s = (hot.end_time - hot.start_time) / 1e9
    assert hot_duration_s < RECALL_SLEEP_S, (
        f"query.hot_search duration {hot_duration_s:.3f}s ~ recall wait "
        f"({RECALL_SLEEP_S:.2f}s) — it must measure the consolidated search alone"
    )
    # recall.search is not nested under query.hot_search.
    assert recall.parent is not None
    assert recall.parent.span_id != hot.context.span_id


def test_overlap_consolidated_spans_still_child_of_hot_search(
    captured_spans, shard_env, monkeypatch
):
    """The inline consolidated FAISS children (`faiss.search`, etc.) stay children
    of the SAME `query.hot_search` span on the union/overlap path — the inline
    branch runs under `query.hot_search`."""
    _build_shard("tc", "dsc")
    monkeypatch.setattr(v1q, "recall_enabled", lambda: True)
    monkeypatch.setattr(v1q, "recall_search", _recall_search_with_span)

    with _request_span():
        v1q.run_query("tc", v1q._ParsedQuery("dsc", [0.1] * 8, 5, None, {}))

    parent = _by_name(captured_spans, "query.hot_search")
    assert parent is not None
    parent_span_id = parent.context.span_id
    for child in ("state.list_shards", "faiss.search"):
        sp = _by_name(captured_spans, child)
        assert sp is not None, f"{child} not emitted on the overlap path"
        assert sp.parent is not None and sp.parent.span_id == parent_span_id, (
            f"{child} is not a child of query.hot_search on the overlap path"
        )
