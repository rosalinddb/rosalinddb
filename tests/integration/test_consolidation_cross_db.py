"""Cross-DB consolidation: REAL control-plane Postgres + REAL recall pgvector + MinIO.

The other consolidation suite (`test_consolidation.py`) runs the control plane on
`memory://` (the recall path is gated on `RB_RECALL_DSN`, not the control-plane
DSN, so memory mode exercises the recall round-trips faithfully). This suite goes
one step further to prove the **cross-database seam** with TWO real Postgres
instances:

  - the watermark (`shard_catalog.consolidated_lsn`) lives in a real
    control-plane Postgres (`DATABASE_URL`);
  - the recall rows + LSN sequence live in a SEPARATE real pgvector instance
    (`RB_RECALL_DSN`);
  - the FAISS shard bytes live in the session MinIO.

It proves: a consolidation commits the watermark to the control-plane DB and
trims the recall rows in the recall DB (no distributed transaction); and the
crash-between-commit-and-trim failure mode is safe across the two real DBs.
"""
from __future__ import annotations

import importlib

import psycopg2
import pytest

try:
    from testcontainers.postgres import PostgresContainer
except ImportError as exc:  # pragma: no cover
    PostgresContainer = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture(scope="module")
def cp_url():
    """Control-plane Postgres (the catalog / watermark DB)."""
    if PostgresContainer is None:  # pragma: no cover
        pytest.fail(f"testcontainers required. Import error: {_IMPORT_ERROR}")
    with PostgresContainer("postgres:15-alpine", driver=None) as pg:
        yield pg.get_connection_url()


@pytest.fixture(scope="module")
def recall_url():
    """Separate recall pgvector instance (the recall rows / LSN DB)."""
    if PostgresContainer is None:  # pragma: no cover
        pytest.fail(f"testcontainers required. Import error: {_IMPORT_ERROR}")
    with PostgresContainer("pgvector/pgvector:pg15", driver=None) as pg:
        yield pg.get_connection_url()


def _truncate_recall(dsn):
    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass('public.recall_vectors'), "
                "to_regclass('public.recall_lsn_seq')"
            )
            hv, seq = cur.fetchone()
            if hv is not None:
                cur.execute("TRUNCATE recall_vectors")
            if seq is not None:
                cur.execute("TRUNCATE recall_lsn_seq")
    finally:
        conn.close()


def _recall_count(dsn, tenant, dataset):
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM recall_vectors WHERE tenant_id=%s AND dataset=%s",
                (tenant, dataset),
            )
            return int(cur.fetchone()[0])
    finally:
        conn.close()


def _cp_watermark(dsn, tenant, dataset):
    """Read the newest shard's consolidated_lsn straight from the control-plane DB."""
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT consolidated_lsn, build_type FROM shard_catalog "
                "WHERE tenant_id=%s AND dataset_name=%s ORDER BY created_at DESC LIMIT 1",
                (tenant, dataset),
            )
            return cur.fetchone()
    finally:
        conn.close()


@pytest.fixture
def env(monkeypatch, cp_url, recall_url, s3_landing_prefix, s3_indexes_prefix, tmp_path):
    """Both real Postgres instances + MinIO; recall tier ON."""
    monkeypatch.setenv("DATABASE_URL", cp_url)
    monkeypatch.setenv("RB_RECALL_DSN", recall_url)
    monkeypatch.setenv("RB_RECALL", "true")
    monkeypatch.setenv("LANDING_PREFIX", s3_landing_prefix)
    monkeypatch.setenv("INDEXES_PREFIX", s3_indexes_prefix)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TENANT_PREFIX", "true")
    monkeypatch.setenv("INDEX_TYPE", "flat")

    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    assert not state_mod._MEMORY_MODE, "must run against real control-plane Postgres"
    state_mod.migrate()
    state_mod._RECALL_MIGRATED = False
    state_mod.migrate_recall(force=True)
    _truncate_recall(recall_url)
    if state_mod.get_tenant_by_id("ten_x") is None:
        state_mod.create_tenant("ten_x", "x@example.com", "x")

    import services.index_builder.run as builder
    importlib.reload(builder)
    yield state_mod, builder, cp_url, recall_url

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("RB_RECALL_DSN", raising=False)
    monkeypatch.delenv("RB_RECALL", raising=False)
    importlib.reload(state_mod)


def _seed_recall(state_mod, tenant, dataset, records):
    state_mod.recall_upsert_vectors(tenant, dataset, records)


def test_cross_db_consolidation_commits_watermark_and_trims(env):
    """Watermark committed to the CP DB; recall rows trimmed in the recall DB.

    Two consolidations: after the second, the CP DB holds the advanced watermark
    + build_type='consolidate', and the recall DB has been grace-trimmed.
    """
    state_mod, builder, cp_url, recall_url = env
    state_mod.create_dataset("ten_x", "cd", 4)

    _seed_recall(state_mod, "ten_x", "cd", [
        {"id": "a", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
        {"id": "b", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {}},
    ])
    assert builder.run_consolidate_once("cd", "ten_x") == 2
    wm, bt = _cp_watermark(cp_url, "ten_x", "cd")
    assert wm == 2 and bt == "consolidate", (wm, bt)
    # Only one shard so far → grace trim is a no-op; both rows still in recall.
    assert _recall_count(recall_url, "ten_x", "cd") == 2

    _seed_recall(state_mod, "ten_x", "cd", [
        {"id": "c", "values": [0.0, 0.0, 1.0, 0.0], "metadata": {}},
    ])
    assert builder.run_consolidate_once("cd", "ten_x") == 3  # a, b re-folded + c
    wm2, bt2 = _cp_watermark(cp_url, "ten_x", "cd")
    assert wm2 == 3 and bt2 == "consolidate", (wm2, bt2)
    # Now 2 shards: the grace trim deletes up to the 2nd-newest watermark (2),
    # leaving the row at lsn 3 (covered only by the newest shard).
    assert _recall_count(recall_url, "ten_x", "cd") == 1


def test_cross_db_crash_between_commit_and_trim(env, monkeypatch):
    """A trim crash leaves the CP watermark committed + recall rows intact (no loss).

    Across two REAL databases: the control-plane commit succeeds, the recall trim
    raises — the watermark is advanced and the recall rows survive (the union
    excludes them, so no duplicate). The next consolidation GCs the orphans.
    """
    state_mod, builder, cp_url, recall_url = env
    state_mod.create_dataset("ten_x", "cr", 4)
    _seed_recall(state_mod, "ten_x", "cr", [
        {"id": "a", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
        {"id": "b", "values": [0.0, 1.0, 0.0, 0.0], "metadata": {}},
    ])

    real_trim = builder.recall_trim

    def _boom(t, d, g):
        raise RuntimeError("simulated crash after CP commit, before recall trim")

    builder.recall_trim = _boom
    try:
        assert builder.run_consolidate_once("cr", "ten_x") == 2
    finally:
        builder.recall_trim = real_trim

    # CP watermark IS committed despite the trim crash.
    wm, bt = _cp_watermark(cp_url, "ten_x", "cr")
    assert wm == 2 and bt == "consolidate", (wm, bt)
    # Recall rows survived (no loss) — the union excludes them by the watermark.
    assert _recall_count(recall_url, "ten_x", "cr") == 2

    # Next consolidation (trim restored) GCs the orphaned rows.
    _seed_recall(state_mod, "ten_x", "cr", [
        {"id": "c", "values": [0.0, 0.0, 1.0, 0.0], "metadata": {}},
    ])
    builder.run_consolidate_once("cr", "ten_x")
    # lsn 1,2 (the orphans) are now <= the 2nd-newest watermark (2) → trimmed.
    conn = psycopg2.connect(recall_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT lsn FROM recall_vectors WHERE tenant_id=%s AND dataset=%s "
                "ORDER BY lsn",
                ("ten_x", "cr"),
            )
            remaining = [int(r[0]) for r in cur.fetchall()]
    finally:
        conn.close()
    assert 1 not in remaining and 2 not in remaining, (
        f"orphaned recall rows must be GC'd next consolidation: {remaining}"
    )
