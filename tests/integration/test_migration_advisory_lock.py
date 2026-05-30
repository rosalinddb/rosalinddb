"""Integration coverage for concurrent schema migration (Change 2).

Multi-worker safety. When `query_api` runs multiple workers and the pipeline
services run as multiple replicas, every process calls `migrate()` at boot.
The idempotent DDL still takes brief `AccessExclusiveLock`s on Postgres, so
concurrent `CREATE TABLE` / `DROP TABLE` from separate processes can deadlock
against each other.

The fix wraps the migration transaction in a Postgres transaction advisory
lock (`pg_advisory_xact_lock`): one process at a time applies the schema, the
rest block then run the idempotent DDL as a no-op.

These tests run against a REAL Postgres (testcontainers) — the in-memory state
adapter has no migrations, so the advisory-lock path can only be exercised
here. They prove:

  - many `_apply_migrations()` calls firing concurrently from separate
    connections all complete with no deadlock and no error;
  - the schema ends up correctly applied (the catalog tables exist);
  - a plain idempotent re-run of `migrate()` is a clean no-op.

`_apply_migrations()` is called directly (bypassing the per-process
`_MIGRATE_LOCK` / `_MIGRATED` flag) so the threads genuinely race in Postgres,
standing in for N separate processes.
"""
from __future__ import annotations

import importlib
import threading

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
    """Start one Postgres container for this module; yield a psycopg2 DSN."""
    if PostgresContainer is None:  # pragma: no cover
        pytest.fail(
            "testcontainers is required for the migration suite. "
            f"Import error: {_IMPORT_ERROR}"
        )
    with PostgresContainer("postgres:15-alpine", driver=None) as pg:
        yield pg.get_connection_url()


@pytest.fixture
def state(monkeypatch, pg_url):
    """Reload the state adapter bound to the container Postgres.

    Teardown restores the default `memory://` adapter so the rest of the
    session (which expects in-memory state) is unaffected.
    """
    monkeypatch.setenv("DATABASE_URL", pg_url)
    import adapters.state.state as state_mod
    importlib.reload(state_mod)
    assert not state_mod._MEMORY_MODE, "test must run against real Postgres"
    yield state_mod
    monkeypatch.delenv("DATABASE_URL", raising=False)
    importlib.reload(state_mod)


def test_concurrent_apply_migrations_does_not_deadlock(state):
    """Many concurrent `_apply_migrations()` calls all succeed — no deadlock.

    Each thread opens its own connection and races to apply the schema,
    standing in for N separate processes booting at once. Without the
    `pg_advisory_xact_lock` the concurrent `CREATE TABLE` / `DROP TABLE` DDL
    deadlocks; with it they serialise and every call returns cleanly.
    """
    errors: list[Exception] = []
    errors_lock = threading.Lock()
    barrier = threading.Barrier(8)

    def migrate_worker() -> None:
        barrier.wait()  # all threads start applying at the same instant
        try:
            state._apply_migrations()
        except Exception as exc:  # noqa: BLE001
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=migrate_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent migration raised: {errors!r}"

    # The schema is correctly applied: the catalog tables exist.
    with state._conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
        tables = {row[0] for row in cur.fetchall()}
    for expected in ("tenants", "api_keys", "dataset_catalog",
                     "shard_catalog", "import_jobs"):
        assert expected in tables, f"missing table {expected}; got {tables}"


def test_migrate_rerun_preserves_existing_data(state):
    """A re-run of `_apply_migrations()` must NOT destroy committed data.

    Regression guard: a re-run of the migration step must not destroy committed
    data. Migrations 002 and 003 do `DROP TABLE ... CASCADE` + unconditional
    `CREATE TABLE` on `dataset_catalog` / `shard_catalog`. Without migration
    version tracking the second pass re-runs them and the seeded dataset row
    vanishes (and `import_jobs` / `shard_catalog` cascade away). With version
    tracking the apply loop skips every already-recorded migration, so the
    data survives.

    A real dataset row is inserted between the two migration passes — the
    failure scenario: rows written after migrator A commits are silently
    destroyed when migrator B re-runs the destructive 002/003.
    """
    # First run applies the schema (migrator A).
    state.migrate()
    # Seed a tenant + dataset — the dataset lives in `dataset_catalog`, which
    # 002/003 DROP CASCADE. Stand-in for rows written between A committing and
    # B re-running `_apply_migrations()`.
    if state.get_tenant_by_id("ten_keep") is None:
        state.create_tenant("ten_keep", "keep@example.com", "x")
    state.create_dataset("ten_keep", "survivor", 4)
    assert state.get_dataset("ten_keep", "survivor") is not None

    # Migrator B re-runs the apply loop directly (bypassing the per-process
    # `_MIGRATED` flag) — this is what a second process booting does.
    state._apply_migrations()

    # The dataset row MUST still exist: a re-run touches no data. Against the
    # pre-fix branch the re-run of 002/003 drops `dataset_catalog` and this is
    # None — the row was silently destroyed.
    row = state.get_dataset("ten_keep", "survivor")
    assert row is not None, "re-running migrations destroyed committed data"
    assert row["dimension"] == 4


def test_migrate_bootstraps_already_migrated_db_without_data_loss(state):
    """An existing dev DB with 001-005 applied but no `schema_migrations`.

    The old code applied 001-005 with no version tracking. When the new
    `_apply_migrations()` first runs against such a DB it creates
    `schema_migrations` — and must detect the schema is already current and
    backfill every version as applied, NOT re-run (and drop) the catalog
    tables. This simulates that: seed a tenant, then drop `schema_migrations`
    to mimic a pre-version-tracking database, then run `_apply_migrations()`.
    """
    state.migrate()
    if state.get_tenant_by_id("ten_boot") is None:
        state.create_tenant("ten_boot", "boot@example.com", "x")
    state.create_dataset("ten_boot", "legacy", 4)

    # Mimic a legacy DB: schema is fully applied, but there is no version
    # ledger (the old `_apply_migrations()` never created one).
    with state._conn() as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS schema_migrations")

    # First run of the version-tracking `_apply_migrations()` on a legacy DB.
    state._apply_migrations()

    # The legacy data survived the bootstrap — the catalogs were not dropped
    # and recreated (which a naive re-run of 002/003 would have done).
    assert state.get_tenant_by_id("ten_boot") is not None, (
        "bootstrap on a legacy DB destroyed committed data"
    )
    row = state.get_dataset("ten_boot", "legacy")
    assert row is not None, "bootstrap on a legacy DB dropped the catalog tables"
    # And the ledger is now populated so future re-runs are clean no-ops.
    with state._conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM schema_migrations")
        assert cur.fetchone()[0] >= 5, "bootstrap did not backfill the ledger"


def _all_tables(state) -> set:
    """Return the set of `public` table names in the bound Postgres."""
    with state._conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
        return {row[0] for row in cur.fetchall()}


def test_force_migrate_applies_schema_despite_skip_migrate(state, monkeypatch):
    """`migrate(force=True)` applies the schema even with `RB_SKIP_MIGRATE=1`.

    The deploy's migration step must apply the schema even when the process env
    sets `RB_SKIP_MIGRATE=1`. Long-running process groups set that flag so they
    skip migrating on boot, but the dedicated migration entrypoint
    (`python -m scripts.migrate`) must NOT honour it — its entire job is to
    apply the schema.

    `migrate(force=True)` is the entrypoint's call: it bypasses the
    `RB_SKIP_MIGRATE` early-return. Without the `force` parameter, `migrate()`
    short-circuits and applies nothing — the catalog tables are never created
    and a fresh deploy comes up with no schema.
    """
    monkeypatch.setenv("RB_SKIP_MIGRATE", "1")

    state.migrate(force=True)

    tables = _all_tables(state)
    for expected in ("tenants", "api_keys", "dataset_catalog",
                     "shard_catalog", "import_jobs"):
        assert expected in tables, (
            f"migrate(force=True) skipped the schema under RB_SKIP_MIGRATE=1; "
            f"missing {expected}, got {tables}"
        )


def test_plain_migrate_still_skips_under_skip_migrate(state, monkeypatch):
    """Plain `migrate()` still honours `RB_SKIP_MIGRATE=1` — guard intact.

    The `force` parameter must not weaken the existing guard: the service
    process groups call `migrate()` with no args and MUST still skip on boot
    (otherwise four process groups race Postgres DDL locks). Proven on a fresh
    DB: with the env var set, `migrate()` applies nothing — no catalog tables.
    """
    monkeypatch.setenv("RB_SKIP_MIGRATE", "1")

    # Fresh DB for this test — wipe anything a prior test in the module left.
    with state._conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DROP TABLE IF EXISTS shard_catalog, import_jobs, dataset_catalog, "
            "api_keys, tenants, schema_migrations CASCADE"
        )

    state.migrate()

    tables = _all_tables(state)
    assert "tenants" not in tables, (
        "plain migrate() must skip when RB_SKIP_MIGRATE=1; it applied the "
        f"schema anyway, got {tables}"
    )


# --- Migration 006: tenants.dp_pool column --------------------------------


def _tenants_columns(state) -> set:
    """Return the set of column names on the `tenants` table."""
    with state._conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'tenants'"
        )
        return {row[0] for row in cur.fetchall()}


def test_migration_006_adds_dp_pool_column(state):
    """Migration 006 adds `tenants.dp_pool` with a `'shared'` default.

    A tenant created after the migration lands on `'shared'` with no backfill,
    and `get_tenant_dp_pool` reads the column back through real Postgres.
    """
    state.migrate()
    assert "dp_pool" in _tenants_columns(state), (
        "migration 006 did not add the dp_pool column"
    )

    if state.get_tenant_by_id("ten_dp") is None:
        state.create_tenant("ten_dp", "dp@example.com", "x")
    # The column default applies — a new tenant routes to the shared pool.
    assert state.get_tenant_dp_pool("ten_dp") == "shared"

    # An explicit dedicated pool reads back through the column.
    with state._conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE tenants SET dp_pool = %s WHERE id = %s",
            ("dedicated-ten_dp", "ten_dp"),
        )
        conn.commit()
    assert state.get_tenant_dp_pool("ten_dp") == "dedicated-ten_dp"

    # An unknown tenant defaults to shared — routing never fails open.
    assert state.get_tenant_dp_pool("no_such_tenant") == "shared"


def test_migration_006_is_idempotent(state):
    """Re-running migration 006 is a clean no-op — additive, non-destructive.

    `_apply_migrations()` skips an already-recorded version; the 006 SQL itself
    is also a bare `ADD COLUMN IF NOT EXISTS`, so even a direct re-execute is
    safe. A dp_pool value set between passes survives the re-run.
    """
    state.migrate()
    if state.get_tenant_by_id("ten_idem") is None:
        state.create_tenant("ten_idem", "idem@example.com", "x")
    with state._conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE tenants SET dp_pool = %s WHERE id = %s",
            ("dedicated-ten_idem", "ten_idem"),
        )
        conn.commit()

    # Re-run the apply loop directly — stands in for a second process booting.
    state._apply_migrations()

    # The column still exists and the value was not clobbered.
    assert "dp_pool" in _tenants_columns(state)
    assert state.get_tenant_dp_pool("ten_idem") == "dedicated-ten_idem"

    # A direct re-execute of the 006 SQL file is also a no-op.
    from pathlib import Path

    sql = (
        Path(state.__file__).parent / "migrations" / "006_tenants_dp_pool.sql"
    ).read_text(encoding="utf-8")
    with state._conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()
    assert state.get_tenant_dp_pool("ten_idem") == "dedicated-ten_idem"
