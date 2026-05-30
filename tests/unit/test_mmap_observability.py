"""Observability contract for the mmap path.

Pins the public surface the operator-facing trace + metric pipelines depend on:

  - `record_shard_page_faults(n)` increments the `rosalinddb.shard.page_faults`
    counter with NO high-cardinality attributes (cardinality budget).
  - `faiss_load_index_span(uri=..., mmap=...)` stamps the `rosalinddb.mmap`
    attribute when `mmap` is supplied, and stays back-compat (no attribute)
    when callers omit it.
  - `_read_major_faults` parses field 12 of `/proc/self/stat` (the `majflt`
    field per `man 5 proc`), tolerating a `comm` value that contains spaces
    or `)`, and silently returns `None` on platforms where the file does not
    exist (macOS dev).

Hermetic — no Docker, no network. The suite conftest sets `OTEL_SDK_DISABLED=true`
which only short-circuits `init_observability`; the OTel *API* still honours
whatever provider is installed. Tests that need to observe an instrument or a
span install an isolated SDK provider, capture, and restore.
"""
from __future__ import annotations

import importlib

import pytest
from opentelemetry import metrics as _metrics_api
from opentelemetry import trace as _trace_api
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


pytestmark = pytest.mark.unit


# --- fixtures -------------------------------------------------------------


@pytest.fixture
def captured_spans(monkeypatch):
    """Install an isolated in-memory TracerProvider; yield the exporter.

    Mirrors the pattern in `tests/unit/test_query_path_spans.py`.
    """
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(_trace_api, "_TRACER_PROVIDER", provider, raising=False)
    yield exporter
    exporter.clear()


@pytest.fixture
def captured_metrics(monkeypatch):
    """Install an isolated SDK MeterProvider; yield (reader, metrics_module).

    OTel's `set_meter_provider` is set-once, so we patch the private slot
    directly (matches the tracer fixture pattern). The `metrics_module` is
    `adapters.observability.metrics`; its lazy instrument cache is reset so
    the next `_get_instruments()` call binds against THIS provider.
    """
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    # `set_meter_provider` is set-once; patch the private global directly so
    # tests are isolated and re-runnable.
    from opentelemetry.metrics import _internal as _m_internal

    monkeypatch.setattr(_m_internal, "_METER_PROVIDER", provider, raising=False)

    from adapters.observability import metrics as obs_metrics

    monkeypatch.setattr(obs_metrics, "_instruments", None, raising=False)
    yield reader, obs_metrics
    monkeypatch.setattr(obs_metrics, "_instruments", None, raising=False)


def _names(exporter):
    return [s.name for s in exporter.get_finished_spans()]


def _by_name(exporter, name):
    for s in exporter.get_finished_spans():
        if s.name == name:
            return s
    return None


def _metric_points(reader, metric_name):
    """Return the list of data points emitted for `metric_name`, or []."""
    data = reader.get_metrics_data()
    if data is None:
        return []
    points = []
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == metric_name:
                    points.extend(metric.data.data_points)
    return points


# --- record_shard_page_faults counter -------------------------------------


def test_record_shard_page_faults_counter_exists(captured_metrics):
    """Calling `record_shard_page_faults(42)` increments the counter.

    The counter must be `rosalinddb.shard.page_faults` and must carry NO
    per-tenant / per-dataset attributes (cardinality budget — pinned by
    `adapters/observability/metrics.py`'s module docstring).
    """
    reader, obs_metrics = captured_metrics
    obs_metrics.record_shard_page_faults(42)

    points = _metric_points(reader, "rosalinddb.shard.page_faults")
    assert points, "expected at least one data point for the counter"
    total = sum(int(p.value) for p in points)
    assert total == 42
    for p in points:
        attrs = dict(p.attributes or {})
        # No tenant/dataset/api_key/email or any other high-card label.
        assert attrs == {}, (
            f"page-faults counter must carry no attributes; got {attrs!r}"
        )


def test_record_shard_page_faults_ignores_zero_and_negative(captured_metrics):
    """`record_shard_page_faults(0)` (and negatives) must not emit a point.

    `/proc/self/stat`'s `majflt` is monotonic, so the delta computed in the
    query path is `max(0, after - before)`. A zero-delta call must not pad
    the metric with noise.
    """
    reader, obs_metrics = captured_metrics
    obs_metrics.record_shard_page_faults(0)
    obs_metrics.record_shard_page_faults(-3)
    points = _metric_points(reader, "rosalinddb.shard.page_faults")
    total = sum(int(p.value) for p in points)
    assert total == 0


# --- faiss_load_index_span carries the mmap attribute --------------------


def test_faiss_load_index_span_carries_mmap_true(captured_spans):
    """`mmap=True` stamps `rosalinddb.mmap = True` on the span."""
    from adapters.observability.tracing import faiss_load_index_span

    with faiss_load_index_span(uri="memory://t/d/s.bin", mmap=True):
        pass

    sp = _by_name(captured_spans, "faiss.load_index")
    assert sp is not None
    attrs = dict(sp.attributes or {})
    assert attrs.get("rosalinddb.mmap") is True


def test_faiss_load_index_span_carries_mmap_false(captured_spans):
    """`mmap=False` stamps `rosalinddb.mmap = False` (legacy deserialise)."""
    from adapters.observability.tracing import faiss_load_index_span

    with faiss_load_index_span(uri="memory://t/d/s.bin", mmap=False):
        pass

    sp = _by_name(captured_spans, "faiss.load_index")
    assert sp is not None
    attrs = dict(sp.attributes or {})
    assert attrs.get("rosalinddb.mmap") is False


def test_faiss_load_index_span_omits_mmap_when_not_passed(captured_spans):
    """No `mmap` kwarg — no `rosalinddb.mmap` attribute. Back-compat.

    Existing callers (and the docs agent / external consumers) must keep
    working unchanged when they construct the span without the new kwarg.
    """
    from adapters.observability.tracing import faiss_load_index_span

    with faiss_load_index_span(uri="memory://t/d/s.bin"):
        pass

    sp = _by_name(captured_spans, "faiss.load_index")
    assert sp is not None
    attrs = dict(sp.attributes or {})
    assert "rosalinddb.mmap" not in attrs


# --- _read_major_faults helper -------------------------------------------


# A realistic /proc/self/stat sample: pid, comm in parens (may contain spaces
# and a literal `)`), then 40+ whitespace-separated fields. `majflt` is field
# 12 per `man 5 proc` (1-indexed). After `rpartition(")")` drops pid + comm,
# the tail starts at field 3 (state); `majflt` is the 10th tail token (index 9).
#
# Tail field map (0-indexed):
#   tail[0]=state(3) tail[1]=ppid(4) tail[2]=pgrp(5) tail[3]=session(6)
#   tail[4]=tty_nr(7) tail[5]=tpgid(8) tail[6]=flags(9) tail[7]=minflt(10)
#   tail[8]=cminflt(11) tail[9]=majflt(12)  <-- the value the sampler returns
#
# The `comm` deliberately contains a `)` so the sampler must `rpartition(")")`
# (split on the LAST one) instead of the first.
_FAKE_STAT_LINE = (
    # pid + (comm with a `)` inside) + state ppid pgrp session tty_nr tpgid
    # flags minflt cminflt MAJFLT  cmajflt utime stime cutime cstime priority
    # nice ... (40+ trailing zeros). MAJFLT=1234 sits at tail index 9.
    "12345 (python (test)) S 1 12345 12345 0 -1 4194304 9876 0 "
    "1234 0 0 0 0 0 20 0 1 0 100 0 0 0 0 0 0 0 0 0 0 0 0 0 0 17 0 0 0 0 0 0\n"
)


def test_page_fault_sampler_reads_proc_self_stat(monkeypatch, tmp_path):
    """The sampler returns the `majflt` value parsed from /proc/self/stat.

    Writes a fake stat file with a known majflt at the documented offset and
    asserts the helper returns that integer.
    """
    from services.query_api import v1_query as v1q

    stat = tmp_path / "stat"
    stat.write_text(_FAKE_STAT_LINE)
    value = v1q._read_major_faults(stat_path=str(stat))
    assert value == 1234


def test_page_fault_sampler_silent_on_missing_proc(tmp_path):
    """Missing /proc/self/stat (macOS dev) returns None and does not raise."""
    from services.query_api import v1_query as v1q

    missing = str(tmp_path / "no-such-stat")
    assert v1q._read_major_faults(stat_path=missing) is None


def test_page_fault_sampler_silent_on_malformed_line(tmp_path):
    """A surprising format returns None instead of raising.

    Best-effort contract: the query path never sees an exception from the
    sampler. A line with no `)` (legitimately impossible on Linux but a
    defensive guard) or a non-int at the expected field returns None.
    """
    from services.query_api import v1_query as v1q

    bad = tmp_path / "stat"
    bad.write_text("garbage no parens here\n")
    assert v1q._read_major_faults(stat_path=str(bad)) is None

    short = tmp_path / "short"
    short.write_text("1 (comm) S 1 2\n")  # too few fields after `)`
    assert v1q._read_major_faults(stat_path=str(short)) is None

    nonint = tmp_path / "nonint"
    # Same shape as `_FAKE_STAT_LINE` but with a non-int at the majflt slot
    # (tail index 9).
    nonint.write_text(
        "1 (comm) S 1 1 1 0 -1 0 0 0 NOT_AN_INT 0 0 0 20 0 1 0 100\n"
    )
    assert v1q._read_major_faults(stat_path=str(nonint)) is None


# --- end-to-end: page-fault delta is recorded around the search ----------


def test_page_fault_delta_is_recorded_around_search(
    captured_metrics, monkeypatch, tmp_path
):
    """A `_hot_search` call records `after - before` page faults via the helper.

    Drives a real cold search against a `memory://` shard with the page-fault
    sampler monkeypatched to return increasing values (before, after). Asserts
    the page-faults counter rose by exactly the delta.
    """
    import json

    import faiss  # type: ignore
    import numpy as np

    from adapters.state import state as state_mod
    from adapters.storage import storage as storage_mod
    from services.query_api import v1_query as v1q

    reader, obs_metrics = captured_metrics

    # Re-bind the v1_query module's obs_metrics reference to the same one the
    # fixture reset — the module imported it at the top, so the binding is
    # shared by name.
    monkeypatch.setattr(v1q, "obs_metrics", obs_metrics, raising=False)

    # Fresh storage + state + cache.
    storage_mod.memory_reset()
    state_mod._MEM_SHARDS.clear()
    v1q.cache_clear()
    monkeypatch.setattr(v1q, "CACHE_DIR", str(tmp_path / "shards"))

    # Build a tiny shard the hot path can find.
    rng = np.random.default_rng(11)
    dim, n = 8, 32
    vecs = rng.random((n, dim), dtype=np.float32)
    ids = np.arange(1, n + 1, dtype=np.int64)
    inner = faiss.IndexFlatL2(dim)
    index = faiss.IndexIDMap2(inner)
    index.add_with_ids(vecs, ids)
    shard_uri = "memory://shards/tenant/ds/shard.bin"
    storage_mod.write_bytes(shard_uri, faiss.serialize_index(index).tobytes())
    sidecar = {str(int(i)): {"id": f"r{int(i)}", "metadata": {}} for i in ids}
    storage_mod.write_bytes(
        f"{shard_uri}.meta.json", json.dumps(sidecar).encode("utf-8")
    )
    state_mod.add_shard("tenant", "ds", shard_uri, "chk", n, "flat", "full", [])

    # Two successive reads: before=100, after=107  ->  delta = 7.
    samples = iter([100, 107])
    monkeypatch.setattr(v1q, "_read_major_faults", lambda *a, **k: next(samples))

    out = v1q._hot_search("tenant", "ds", [0.1] * dim, top_k=3)
    assert out is not None

    points = _metric_points(reader, "rosalinddb.shard.page_faults")
    total = sum(int(p.value) for p in points)
    assert total == 7, f"expected page-fault delta of 7, got {total}"
