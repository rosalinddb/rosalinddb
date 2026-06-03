"""Integration coverage for the hot-tier (delta tier) schema migration.

The hot tier is a SEPARATE data-plane pgvector instance addressed via
`RB_HOT_DSN`, never the control-plane Postgres (see
docs/architecture/delta-tier.md, "Blast radius & control/data-plane isolation").
Its schema is applied by `migrate_hot()` / `_apply_hot_migrations()`, independent
of the control-plane `migrate()`.

These tests run against a REAL pgvector container (testcontainers) because the
extension + the `vector` column type only exist there — the in-memory state
adapter has no hot schema. They prove:

  - `migrate_hot()` against a pgvector DSN creates the `vector` extension, the
    `hot_vectors` table, and the `hot_lsn_seq` table;
  - the per-(tenant, dataset) LSN sequence is strictly monotonic;
  - `hot_vectors.embedding` is an UNPARAMETERISED `vector` so a single table
    stores different per-dataset dimensions;
  - a re-run of `_apply_hot_migrations()` is a clean, non-destructive no-op;
  - DEFAULT-OFF: with `RB_HOT_DSN` unset, `migrate_hot()` is a pure no-op and
    opens no connection.

The control-plane `DATABASE_URL` is irrelevant to the hot path — the gate is
`RB_HOT_DSN` — so these tests leave the default `memory://` control plane in
place and only stand up the pgvector instance.
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
def hot_url():
    """Start one pgvector container for this module; yield a psycopg2 DSN.

    `pgvector/pgvector:pg15` is the same image the compose `pgvector` service
    uses — it ships the `vector` extension that `CREATE EXTENSION` needs.
    """
    if PostgresContainer is None:  # pragma: no cover
        pytest.fail(
            "testcontainers is required for the hot-migration suite. "
            f"Import error: {_IMPORT_ERROR}"
        )
    with PostgresContainer("pgvector/pgvector:pg15", driver=None) as pg:
        yield pg.get_connection_url()


@pytest.fixture
def state(monkeypatch, hot_url):
    """Reload the state adapter with `RB_HOT_DSN` pointed at the container.

    Leaves `DATABASE_URL` at its default `memory://` — the hot path is gated on
    `RB_HOT_DSN`, not the control-plane DSN. Teardown clears the env and resets
    the process-local hot-migrate flag so other tests start clean.
    """
    monkeypatch.setenv("RB_HOT_DSN", hot_url)
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    state_mod._HOT_MIGRATED = False
    yield state_mod
    monkeypatch.delenv("RB_HOT_DSN", raising=False)
    importlib.reload(state_mod)


def _hot_tables(dsn: str) -> set:
    """Return the set of `public` table names in the hot instance."""
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def test_migrate_hot_creates_extension_and_tables(state, hot_url):
    """`migrate_hot()` creates the vector extension + hot_vectors + hot_lsn_seq."""
    applied = state.migrate_hot(force=True)
    assert applied is True, "migrate_hot reported off despite RB_HOT_DSN set"

    tables = _hot_tables(hot_url)
    for expected in ("hot_vectors", "hot_lsn_seq", "hot_schema_migrations"):
        assert expected in tables, f"missing hot table {expected}; got {tables}"

    # The vector extension is installed.
    conn = psycopg2.connect(hot_url)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            assert cur.fetchone() is not None, "vector extension not created"
    finally:
        conn.close()


def test_hot_vectors_embedding_is_unparameterised_vector(state, hot_url):
    """`hot_vectors.embedding` is plain `vector` (no fixed dim) — per-dataset dims.

    A single table must accept vectors of different dimensions across datasets,
    so the column carries no typmod. Proven by inserting a 3-dim and a 5-dim
    vector into the same table without error.
    """
    state.migrate_hot(force=True)
    conn = psycopg2.connect(hot_url)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            # The declared column type is the unparameterised `vector`.
            cur.execute(
                "SELECT format_type(atttypid, atttypmod) "
                "FROM pg_attribute "
                "WHERE attrelid = 'hot_vectors'::regclass AND attname = 'embedding'"
            )
            assert cur.fetchone()[0] == "vector", "embedding should be plain vector"

            # Different dimensions coexist in the one table.
            cur.execute(
                "INSERT INTO hot_vectors (tenant_id, dataset, id, embedding, lsn) "
                "VALUES ('t', 'd3', 'a', %s, 1)",
                ("[1,2,3]",),
            )
            cur.execute(
                "INSERT INTO hot_vectors (tenant_id, dataset, id, embedding, lsn) "
                "VALUES ('t', 'd5', 'b', %s, 1)",
                ("[1,2,3,4,5]",),
            )
            cur.execute("SELECT count(*) FROM hot_vectors")
            assert cur.fetchone()[0] == 2
    finally:
        conn.close()


def _next_lsn(dsn: str, tenant: str, dataset: str) -> int:
    """Allocate the next LSN for (tenant, dataset) via the upsert-increment."""
    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
INSERT INTO hot_lsn_seq (tenant_id, dataset, last_lsn)
VALUES (%s, %s, 1)
ON CONFLICT (tenant_id, dataset)
DO UPDATE SET last_lsn = hot_lsn_seq.last_lsn + 1
RETURNING last_lsn
                """,
                (tenant, dataset),
            )
            return cur.fetchone()[0]
    finally:
        conn.close()


def test_lsn_sequence_is_monotonic_per_dataset(state, hot_url):
    """The hot_lsn_seq upsert yields a strictly increasing LSN per (tenant, ds).

    Each (tenant, dataset) has its own counter; allocations are gap-free and
    monotonic, and two datasets advance independently.
    """
    state.migrate_hot(force=True)

    a = [_next_lsn(hot_url, "t1", "ds") for _ in range(5)]
    assert a == [1, 2, 3, 4, 5], f"non-monotonic per-dataset LSN: {a}"

    # A different dataset has its own independent counter.
    b = [_next_lsn(hot_url, "t1", "other") for _ in range(3)]
    assert b == [1, 2, 3], f"per-dataset isolation broken: {b}"

    # A different tenant, same dataset name, is also independent.
    c = _next_lsn(hot_url, "t2", "ds")
    assert c == 1, f"per-tenant isolation broken: {c}"


def test_migrate_hot_rerun_is_noop(state, hot_url):
    """A re-run of `_apply_hot_migrations()` is a clean, non-destructive no-op.

    All hot migrations are `IF NOT EXISTS`, and the ledger skips applied
    versions, so seeded data survives a second pass — stands in for a second
    process booting against the hot instance.
    """
    state.migrate_hot(force=True)
    # Seed a row; a naive re-run that re-created tables would drop it.
    conn = psycopg2.connect(hot_url)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO hot_vectors (tenant_id, dataset, id, embedding, lsn) "
                "VALUES ('t', 'd', 'keep', %s, 7) "
                "ON CONFLICT (tenant_id, dataset, id) DO NOTHING",
                ("[1,2,3]",),
            )
    finally:
        conn.close()

    # Re-apply directly (bypass the per-process flag) — a second booter.
    state._apply_hot_migrations(hot_url)

    conn = psycopg2.connect(hot_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT lsn FROM hot_vectors "
                "WHERE tenant_id='t' AND dataset='d' AND id='keep'"
            )
            row = cur.fetchone()
    finally:
        conn.close()
    assert row is not None and row[0] == 7, "re-run destroyed hot data"


def test_migrate_hot_is_noop_when_dsn_unset(monkeypatch):
    """DEFAULT-OFF: `migrate_hot()` is a pure no-op with `RB_HOT_DSN` unset.

    It returns False and opens NO connection, so a flag-off deploy behaves
    byte-identically to today. Reloaded with the env cleared so the module reads
    `RB_HOT_DSN` as unset.
    """
    monkeypatch.delenv("RB_HOT_DSN", raising=False)
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    try:
        assert state_mod._hot_dsn() is None
        assert state_mod.migrate_hot(force=True) is False
        # A blank value is also treated as off (cannot accidentally enable it).
        monkeypatch.setenv("RB_HOT_DSN", "   ")
        importlib.reload(state_mod)
        assert state_mod.migrate_hot(force=True) is False
    finally:
        monkeypatch.delenv("RB_HOT_DSN", raising=False)
        importlib.reload(state_mod)
