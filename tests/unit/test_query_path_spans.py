"""Unit coverage for query hot-path tracing decomposition (obs/query-path-spans).

Hermetic — no Docker, no network. Builds a real FAISS shard in `memory://`
storage, runs `_hot_search` directly, and asserts the trace decomposes into
attributable child spans instead of one opaque `faiss.search` span.

The suite conftest sets `OTEL_SDK_DISABLED=true`, which only short-circuits
`init_observability` — the OTel *API* still honours whatever `TracerProvider`
is installed. So each test installs an isolated SDK `TracerProvider` with an
`InMemorySpanExporter`, captures the spans `_hot_search` emits, and restores
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
    """A cold `_hot_search` emits a `query.hot_search` parent with the R2
    download, FAISS deserialize and pure vector search as separate children."""
    _build_shard("t1", "ds1")
    out = v1q._hot_search("t1", "ds1", [0.1] * 8, top_k=5)
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
    v1q._hot_search("t2", "ds2", [0.1] * 8, top_k=5)

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
    v1q._hot_search("t3", "ds3", [0.1] * 8, top_k=5)

    search = _by_name(captured_spans, "faiss.search")
    load = _by_name(captured_spans, "faiss.load_index")
    assert search is not None and load is not None
    # The pure search starts only after the index is deserialized.
    assert search.start_time >= load.end_time


def test_warm_query_skips_cold_load_spans(captured_spans, shard_env):
    """A cache-hit query emits `faiss.search` but NOT the cold-load spans —
    the warm path is obviously distinguishable in a trace."""
    _build_shard("t4", "ds4")
    v1q._hot_search("t4", "ds4", [0.1] * 8, top_k=5)  # cold: populate cache
    captured_spans.clear()

    _matches, mode = v1q._hot_search("t4", "ds4", [0.1] * 8, top_k=5)
    assert mode == "hot"
    names = _names(captured_spans)
    assert "query.hot_search" in names
    assert "faiss.search" in names
    assert "shard.download" not in names
    assert "faiss.load_index" not in names


def test_search_span_keeps_existing_attributes(captured_spans, shard_env):
    """The decomposition keeps tenant/dataset/top_k/fetch_k attributes."""
    _build_shard("t5", "ds5")
    v1q._hot_search("t5", "ds5", [0.1] * 8, top_k=5)

    search = _by_name(captured_spans, "faiss.search")
    assert search is not None
    attrs = dict(search.attributes or {})
    assert attrs.get("rosalinddb.tenant_id") == "t5"
    assert attrs.get("rosalinddb.dataset") == "ds5"
    assert attrs.get("rosalinddb.top_k") == 5
    assert attrs.get("rosalinddb.fetch_k") == 5
