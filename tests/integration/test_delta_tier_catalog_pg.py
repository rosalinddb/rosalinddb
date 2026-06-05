"""Real-Postgres coverage for the delta-tier catalog columns (Phase 1, PR-A).

The unit tests (`tests/unit/test_delta_tier_catalog.py`) run on the `memory://`
adapter; this exercises the SAME contract against a REAL control-plane Postgres
so migration 009, the `BIGINT[]` tombstone array, and the nullable
`parent_shard_id` round-trip through actual SQL — and so the generation-aware
`superseded_shards`/`grace_watermark` are proven on the DB path, not just the
in-memory mirror.
"""
from __future__ import annotations

import importlib

import pytest

try:
    from testcontainers.postgres import PostgresContainer
except ImportError as exc:  # pragma: no cover
    PostgresContainer = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture(scope="module")
def pg_url():
    if PostgresContainer is None:  # pragma: no cover
        pytest.fail(
            "testcontainers is required for the delta-tier catalog suite. "
            f"Import error: {_IMPORT_ERROR}"
        )
    with PostgresContainer("postgres:15-alpine", driver=None) as pg:
        yield pg.get_connection_url()


@pytest.fixture
def state(monkeypatch, pg_url):
    monkeypatch.setenv("DATABASE_URL", pg_url)
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    assert not state_mod._MEMORY_MODE, "test must run against real Postgres"
    state_mod.migrate()
    # shard_catalog FKs to dataset_catalog, which FKs to tenants. The container
    # is module-scoped (persists across tests), so create idempotently.
    try:
        state_mod.create_tenant("t", "t@example.com", "x")
    except ValueError:
        pass  # duplicate_email — tenant already created by a prior test
    yield state_mod
    monkeypatch.delenv("DATABASE_URL", raising=False)
    importlib.reload(state_mod)


def _fresh_dataset(state, name):
    """A dataset unique to one test (the PG container is shared across tests)."""
    try:
        state.create_dataset("t", name, dimension=8)
    except ValueError:
        pass  # dataset_exists — re-run on the persistent container
    return name


def test_migration_009_columns_round_trip_through_pg(state):
    ds = _fresh_dataset(state, "ds_roundtrip")
    bid = state.add_shard(
        "t", ds, "s3://idx/ds/base.bin", checksum="b", vector_count=100,
        index_type="ivfflat", build_type="consolidate", consolidated_lsn=100,
        quantizer_version=2, parent_shard_id=None, level=0,
        covered_lsn_lo=0, covered_lsn_hi=100,
    )
    did = state.add_shard(
        "t", ds, "s3://idx/ds/delta.bin", checksum="d", vector_count=5,
        index_type="ivfflat", build_type="consolidate-delta", consolidated_lsn=140,
        quantizer_version=2, parent_shard_id=bid, level=1,
        covered_lsn_lo=101, covered_lsn_hi=140, tombstone_int_ids=[7, 8, 9],
    )
    rows = {s["id"]: s for s in state.list_shards("t", ds)}
    b, d = rows[bid], rows[did]
    assert b["parent_shard_id"] is None and b["level"] == 0
    assert b["quantizer_version"] == 2 and b["covered_lsn_hi"] == 100
    assert d["parent_shard_id"] == bid and d["level"] == 1
    assert (d["covered_lsn_lo"], d["covered_lsn_hi"]) == (101, 140)
    assert list(d["tombstone_int_ids"]) == [7, 8, 9]
    assert d["build_type"] == "consolidate-delta"


def test_generation_helpers_through_pg(state):
    ds = _fresh_dataset(state, "ds_gen")
    bid = state.add_shard(
        "t", ds, "s3://idx/ds/base.bin", checksum="b", vector_count=100,
        index_type="ivfflat", build_type="consolidate", consolidated_lsn=100,
        quantizer_version=1, level=0, covered_lsn_lo=0, covered_lsn_hi=100,
    )
    d1 = state.add_shard(
        "t", ds, "s3://idx/ds/d1.bin", checksum="d1", vector_count=5,
        index_type="ivfflat", build_type="consolidate-delta", consolidated_lsn=120,
        quantizer_version=1, parent_shard_id=bid, level=1,
        covered_lsn_lo=101, covered_lsn_hi=120,
    )
    d2 = state.add_shard(
        "t", ds, "s3://idx/ds/d2.bin", checksum="d2", vector_count=5,
        index_type="ivfflat", build_type="consolidate-delta", consolidated_lsn=140,
        quantizer_version=1, parent_shard_id=bid, level=1,
        covered_lsn_lo=121, covered_lsn_hi=140,
    )
    gen = state.live_generation("t", ds)
    assert gen["base"]["id"] == bid
    assert [d["id"] for d in gen["deltas"]] == [d1, d2]
    assert state.dataset_watermark("t", ds) == 140
    # 1 base + 2 deltas, one generation -> nothing superseded, nothing aged out
    assert state.superseded_shards("t", ds) == []
    assert state.grace_watermark("t", ds) == 0
