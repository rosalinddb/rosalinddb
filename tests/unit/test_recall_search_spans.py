"""Unit coverage for recall-search observability (task #20).

The recall half of the query union (`recall_search`) previously had NO trace
span and NO duration metric (Finding #0), so recall latency was invisible even
though the query critical path waits on it. This instruments it to mirror the
cold path's `query.hot_search` span + `rosalinddb.query.duration` histogram:

  - a `recall.search` span with tenant/dataset/top_k/watermark stamped at open
    and `rows_scanned` (the single scan's row count) + `match_count` stamped once
    the scan completes;
  - a `rosalinddb.recall_search.duration` histogram sample (ms) per call,
    exported through the OTel→Prometheus pipeline (namespace `rb`) as
    `rb_rosalinddb_recall_search_duration_milliseconds`.

Hermetic — no Docker, no pgvector. Mirrors `tests/unit/test_query_path_spans.py`
and `tests/unit/test_mmap_observability.py`: an isolated in-memory TracerProvider
captures spans and an isolated SDK MeterProvider captures metric points, while a
fake recall pool drives `recall_search` so no real database is touched.
"""
from __future__ import annotations

import importlib

import pytest
from opentelemetry import metrics as _metrics_api  # noqa: F401 (kept for parity)
from opentelemetry import trace as _trace_api
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def state(monkeypatch):
    """Fresh state module with the recall tier ON (flag + DSN set)."""
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    monkeypatch.setenv("RB_RECALL_DSN", "postgresql://u:p@recall:5432/recall")
    monkeypatch.setenv("RB_RECALL", "true")
    import adapters.state.state as state_mod

    importlib.reload(state_mod)
    yield state_mod
    monkeypatch.delenv("RB_RECALL", raising=False)
    monkeypatch.delenv("RB_RECALL_DSN", raising=False)
    importlib.reload(state_mod)


@pytest.fixture
def captured_spans(monkeypatch):
    """Install an isolated in-memory TracerProvider; yield the exporter."""
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(_trace_api, "_TRACER_PROVIDER", provider, raising=False)
    yield exporter
    exporter.clear()


@pytest.fixture
def captured_metrics(monkeypatch):
    """Install an isolated SDK MeterProvider; yield (reader, metrics_module)."""
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    from opentelemetry.metrics import _internal as _m_internal

    monkeypatch.setattr(_m_internal, "_METER_PROVIDER", provider, raising=False)

    from adapters.observability import metrics as obs_metrics

    monkeypatch.setattr(obs_metrics, "_instruments", None, raising=False)
    yield reader, obs_metrics
    monkeypatch.setattr(obs_metrics, "_instruments", None, raising=False)


# --- fake recall pool ------------------------------------------------------


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def commit(self):
        pass

    def rollback(self):
        pass


class _FakeRecallPool:
    def __init__(self, conn):
        self._conn = conn
        self._pool = [object()]

    def getconn(self):
        return self._conn

    def putconn(self, conn):
        pass


def _wire(state, monkeypatch, rows):
    cur = _FakeCursor(rows)
    pool = _FakeRecallPool(_FakeConn(cur))
    monkeypatch.setattr(state, "_RECALL_POOL", pool)
    monkeypatch.setattr(state, "_RECALL_POOL_DSN", state._recall_dsn())


def _by_name(exporter, name):
    for s in exporter.get_finished_spans():
        if s.name == name:
            return s
    return None


def _metric_points(reader, metric_name):
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


# --- span -----------------------------------------------------------------


def test_recall_search_emits_recall_search_span(state, captured_spans, monkeypatch):
    """`recall_search` emits a `recall.search` span (the recall-half timer)."""
    _wire(state, monkeypatch, rows=[("a", False, 0.1, {})])
    state.recall_search("t1", "ds1", [0.0, 0.0], top_k=5, watermark=7)
    sp = _by_name(captured_spans, "recall.search")
    assert sp is not None, "recall_search must emit a `recall.search` span"


def test_recall_search_span_carries_expected_attributes(
    state, captured_spans, monkeypatch
):
    """The span stamps tenant/dataset/top_k/watermark + rows_scanned/match_count.

    Three rows scanned (one match, one tombstone, one filter-failing live) with a
    filter that passes only the first -> rows_scanned=3, match_count=1.
    """
    rows = [
        ("keep", False, 0.05, {"k": 1}),   # live, passes filter -> match
        ("gone", True, None, {}),           # tombstone -> suppress only
        ("drop", False, 0.02, {"k": 9}),    # live, fails filter -> suppress only
    ]
    _wire(state, monkeypatch, rows)
    state.recall_search(
        "tenantX", "datasetY", [1.0, 2.0], top_k=10, watermark=42, flt={"k": 1}
    )
    sp = _by_name(captured_spans, "recall.search")
    assert sp is not None
    attrs = dict(sp.attributes or {})
    assert attrs.get("rosalinddb.tenant_id") == "tenantX"
    assert attrs.get("rosalinddb.dataset") == "datasetY"
    assert attrs.get("rosalinddb.top_k") == 10
    assert attrs.get("rosalinddb.watermark") == 42
    assert attrs.get("rosalinddb.rows_scanned") == 3, "rows_scanned = single-scan count"
    assert attrs.get("rosalinddb.match_count") == 1, "match_count = live filter-passing"


# --- duration metric ------------------------------------------------------


def test_recall_search_records_duration_metric(state, captured_metrics, monkeypatch):
    """Each `recall_search` call records the recall query-duration histogram.

    The metric is `rosalinddb.recall_search.duration` (ms) — the recall mirror of
    `rosalinddb.query.duration`. It carries a low-cardinality `mode` label
    (default `recall`) and NO tenant/dataset label (cardinality budget).
    """
    reader, _obs_metrics = captured_metrics
    _wire(state, monkeypatch, rows=[("a", False, 0.1, {})])
    state.recall_search("t1", "ds1", [0.0, 0.0], top_k=5, watermark=0)

    points = _metric_points(reader, "rosalinddb.recall_search.duration")
    assert points, "expected a recall_search.duration histogram sample"
    # One observation, non-negative duration, with the `mode=recall` label only.
    total_count = sum(p.count for p in points)
    assert total_count == 1
    for p in points:
        assert p.sum >= 0.0
        attrs = dict(p.attributes or {})
        assert attrs == {"mode": "recall"}, (
            f"recall duration must carry only a low-cardinality mode label; "
            f"got {attrs!r}"
        )


def test_recall_search_duration_records_even_when_no_rows(
    state, captured_metrics, monkeypatch
):
    """An empty recall set still records a duration sample (latency is real)."""
    reader, _obs_metrics = captured_metrics
    _wire(state, monkeypatch, rows=[])
    state.recall_search("t1", "ds1", [0.0, 0.0], top_k=5, watermark=0)
    points = _metric_points(reader, "rosalinddb.recall_search.duration")
    assert sum(p.count for p in points) == 1
