from __future__ import annotations

"""Schema-migration runners + version ledgers for the state adapter.

Extracted from `adapters.state.state` (behaviour-preserving). This module holds
BOTH migration runners and BOTH version ledgers:

  * control-plane: `migrate()` (the public runner) + `_apply_migrations()` (the
    `schema_migrations` ledger, advisory-locked, version-tracked) + the OSS
    bootstrap seed (`_bootstrap_default_tenant_memory`, plus the in-SQL `default`
    tenant INSERT inside `_apply_migrations`);
  * recall-tier: `migrate_recall()` (the public runner) + `_apply_recall_migrations()`
    (the separate `recall_schema_migrations` ledger against `RB_RECALL_DSN`).

The `.sql` files stay under `adapters/state/migrations/` (package-data already
ships them); `_apply_migrations` resolves them via the state package directory.

Mutable process-wide state — `_MEMORY_MODE`, the `_MIGRATED` / `_RECALL_MIGRATED`
run-once flags, the in-memory `_MEM_TENANTS` store the OSS bootstrap writes, and
the recall connection seam (`_recall_dsn`, `_recall_connect`,
`_RECALL_PREPARE_THRESHOLD`) — is OWNED by `adapters.state.state` and reached here
through `_state.X` at CALL time (never at import time), so `importlib.reload(state)`
and `monkeypatch.setattr(state, …)` are both honoured (the test suite relies on
this; see `pooling.py` for the same seam rationale).
"""

import contextlib
import datetime as _dt
import os
import threading
from pathlib import Path

# The state module owns the mutable process-wide globals + the recall connection
# seam. Reference them through `_state.X` at call time. Imported here, but every
# access is deferred to call time (no import-time use), so the partial-init of
# `state` during its own import of this module is safe.
import adapters.state.state as _state


# `migrate()` is idempotent, but its idempotent DDL still takes brief
# `AccessExclusiveLock`s on Postgres. When several services run as threads in
# one process (a single-process dev/test harness) their `migrate()` calls would
# race and deadlock on those locks. This lock + run-once flag serialise
# migration within a process: the first caller applies the schema, the rest
# return immediately. Harmless in production where each service process calls
# `migrate()` once.
#
# NOTE: `_MIGRATE_LOCK` is process-local serialisation and can safely live here;
# the run-once boolean `_MIGRATED` is OWNED by `state` (so `importlib.reload(state)`
# resets it) and is read/written via `_state._MIGRATED`.
_MIGRATE_LOCK = threading.Lock()


# Constant 64-bit key for the schema-migration advisory lock. Any process
# applying migrations takes `pg_advisory_xact_lock(_MIGRATE_LOCK_KEY)` first,
# so concurrent callers serialise in Postgres instead of deadlocking on the
# `AccessExclusiveLock`s the idempotent DDL takes. The value is arbitrary but
# fixed and namespaced to this purpose; it must never change once deployed.
_MIGRATE_LOCK_KEY = 0x726F73616C696E64  # ASCII "rosalind", a stable constant


# Ordered list of migration versions. The version string is the filename
# stem; this list IS the canonical apply order. Appending a new migration
# means adding its stem here — never reorder or remove an existing entry, the
# `schema_migrations` ledger keys on these exact strings.
_MIGRATION_VERSIONS = (
    "001_tenants_and_keys",
    "002_datasets_tenant_isolation",
    "003_shard_incremental_indexing",
    "004_import_jobs",
    "005_dataset_status_updated_at",
    "006_tenants_dp_pool",
    "007_dp_shard_residency",
    "008_shard_consolidated_lsn",
    "009_delta_tier",
)


# Ordered list of RECALL-instance migration versions. These run against the
# SEPARATE data-plane pgvector instance (RB_RECALL_DSN), NOT the control-plane
# Postgres above (see docs/architecture/recall-consolidate.md, "Blast radius").
# They live under `migrations/recall/` and are applied by `migrate_recall()`
# with the same advisory-lock + version-ledger discipline as the control-plane
# set. The two ledgers live in different databases by design and never share a
# connection.
_RECALL_MIGRATION_VERSIONS = ("001_recall_vectors",)


# Distinct advisory-lock key for the recall ledger. It lives in a different
# database from `_MIGRATE_LOCK_KEY`, so collision is impossible either way, but
# a separate constant keeps the intent clear. Fixed forever once deployed.
_RECALL_MIGRATE_LOCK_KEY = 0x726F73685F686F74  # ASCII "rosh_hot", a stable constant

# Process-local serialisation for recall migrations, mirroring the control-plane
# `_MIGRATE_LOCK`. The run-once boolean `_RECALL_MIGRATED` is OWNED by `state`
# (reset on `importlib.reload(state)`) and accessed via `_state._RECALL_MIGRATED`.
_RECALL_MIGRATE_LOCK = threading.Lock()


def migrate(force: bool = False):
    """Apply control-plane schema migrations in order: 001 → … → 008.

    001 (tenants/keys), 002 (datasets tenant isolation), 003 (shard catalog
    incremental-indexing columns), 004 (import_jobs for async bulk ingest),
    005 (dataset `status_updated_at` for the reconciliation reaper), 006
    (`tenants.dp_pool` routing), 007 (`dp_shard_residency` SSD-cache registry),
    008 (`shard_catalog.consolidated_lsn`, the recall-tier watermark — default 0,
    so it is a no-op for behaviour until the recall tier is enabled).
    002 references the `tenants` table for FKs so it must run after 001; 003
    recreates `shard_catalog` so it must run after 002; 004 FKs to
    `dataset_catalog` so it must run after 002. All files live under
    `adapters/state/migrations/`. Memory mode is a no-op.

    This migrates ONLY the control-plane Postgres. The recall-tier (pgvector)
    instance has its own DSN (`RB_RECALL_DSN`) and a separate runner,
    `migrate_recall()` — see that function and `scripts/migrate.py`.

    `_MIGRATE_LOCK` is a `threading.Lock` — it serialises migration *within* a
    single process only. Cross-process safety is provided by a Postgres
    advisory lock taken inside `_apply_migrations()` (`pg_advisory_xact_lock`):
    several *separate processes* all calling `migrate()` at once (a
    multi-process deployment, or `scripts/multiworker_all.py` starting N
    uvicorn workers + the pipeline workers together) serialise in the database
    instead of deadlocking on the `AccessExclusiveLock`s the migration DDL
    takes. The first process to grab the advisory lock applies the schema; the
    rest block, then re-enter `_apply_migrations()`. That re-run is a genuine
    no-op — `_apply_migrations()` tracks applied migration versions in a
    `schema_migrations` ledger and skips every version already recorded, so a
    loser never re-runs the destructive `DROP TABLE` in 002/003 and never
    touches data. (The advisory lock alone would only serialise data loss, not
    prevent it — see `_apply_migrations()`.)

    When migrations run as a separate one-shot step (e.g. an init container
    or k8s job), services can be started with `RB_SKIP_MIGRATE=1` so they do
    NOT migrate on boot — the schema is applied out-of-band. Unset (the
    default), behaviour for a single process is unchanged: the first caller
    applies the schema.

    `force=True` bypasses the `RB_SKIP_MIGRATE` early-return: this is the one
    caller (`scripts/migrate.py`, the dedicated migration entrypoint) whose
    entire job IS to apply the schema. The migration runner typically inherits
    the same env as the services (so it too sees `RB_SKIP_MIGRATE=1`) —
    without `force` it would short-circuit and apply nothing, and the workers
    skip migration too, so a fresh deploy would come up with no schema.
    `force` only skips the env guard; everything else (the `threading.Lock`,
    `_apply_migrations()` with its `pg_advisory_xact_lock` + `schema_migrations`
    ledger) is unchanged.
    """
    if _state._MEMORY_MODE:
        # Memory mode has no schema to apply, but the OSS bootstrap "default"
        # tenant must still exist so that — when `RB_REQUIRE_AUTH` is off —
        # `current_tenant_id` resolving to "default" can write
        # tenant-keyed rows that look up successfully against the in-memory
        # tenants store (see _bootstrap_default_tenant_memory).
        _bootstrap_default_tenant_memory()
        return
    if not force and os.getenv("RB_SKIP_MIGRATE", "").lower() in ("1", "true", "yes"):
        # Schema applied out-of-band (see docstring) — skip to avoid the
        # cross-process AccessExclusiveLock deadlock. `force=True` (the release
        # command) bypasses this: it must apply the schema regardless.
        return
    with _MIGRATE_LOCK:
        if _state._MIGRATED:
            return
        _apply_migrations()
        _state._MIGRATED = True


def _bootstrap_default_tenant_memory() -> None:
    """Idempotently seed the bootstrap "default" tenant in the memory store.

    The OSS opt-in path (`RB_REQUIRE_AUTH` unset/false) collapses every
    request's tenant_id to the literal string "default" via
    `current_tenant_id`'s short-circuit. Postgres mode seeds the row in
    `_apply_migrations()` (see the INSERT ... ON CONFLICT DO NOTHING at the
    end of that function); memory mode mirrors that here so the two adapters
    behave the same.

    Postgres mode is a guarded no-op — the in-memory dicts are not the
    authoritative store there and the migration runner has already taken
    care of the row. Memory mode (and only memory mode) seeds the dict.

    `password_hash='!disabled!'` is a non-bcrypt sentinel — bcrypt.checkpw
    will never match — so even if a self-hoster later flips
    `RB_REQUIRE_AUTH=true`, no one can `/auth/login` as this tenant.
    """
    if not _state._MEMORY_MODE:
        return
    if "default" in _state._MEM_TENANTS:
        return
    vector_quota, query_quota = _state._quota_defaults()
    row = {
        "id": "default",
        "email": "self-host@localhost",
        "password_hash": "!disabled!",
        "plan": "oss",
        "vector_quota": vector_quota,
        "daily_query_quota": query_quota,
        "vectors_used": 0,
        "queries_today": 0,
        "queries_reset_at": _dt.date.today().isoformat(),
        "created_at": _state._now_iso(),
    }
    _state._MEM_TENANTS["default"] = row
    _state._MEM_TENANTS_BY_EMAIL["self-host@localhost"] = "default"


def _apply_migrations() -> None:
    """Apply the ordered migration files exactly once each (version-tracked).

    Cross-process safety: the transaction first takes a transaction advisory
    lock (`pg_advisory_xact_lock`). Only one process holds it at a time — the
    rest block until the holder's transaction commits and releases it. The
    lock auto-releases when the transaction ends (the `with _conn()` block
    commits), so there is no explicit unlock and a crashed process cannot
    leave it stuck.

    Genuine idempotency via version tracking (Finding 1 fix). The advisory
    lock only *serialises* migrators — it does not make a re-run safe. Two of
    the migration files (002, 003) do `DROP TABLE ... CASCADE` + unconditional
    `CREATE TABLE`. Re-running them drops the catalogs and recreates them
    empty, silently destroying any rows another process committed in between.
    So `_apply_migrations()` now keeps a `schema_migrations(version, ...)`
    ledger and applies a migration file ONLY IF its version is not already
    recorded. A re-run of `migrate()` therefore applies nothing and touches no
    data — the destructive 002/003 run exactly once, on a fresh DB.

    Bootstrap of a legacy DB. A dev database created by the old (no-ledger)
    code already has 001-005 applied but no `schema_migrations` table. When
    this function first creates the ledger on such a DB it must NOT then re-run
    (and drop) everything. Detection: if a core table (`tenants`) already
    exists when the ledger was just created, the schema is already current —
    every current version is backfilled into the ledger as applied before the
    apply loop runs, so the loop finds nothing to do. A genuinely fresh DB has
    no `tenants` table, so nothing is backfilled and all migrations apply
    normally.
    """
    migrations_dir = Path(__file__).parent / "migrations"
    # A dedicated, non-pooled connection. Migrations run once at startup, off
    # the hot path, and before the pool is otherwise needed — there is nothing
    # to gain from pooling them, and a dedicated connection keeps the pool
    # unbuilt until the schema actually exists. `contextlib.closing` closes the
    # connection on every exit path; the inner `with conn` manages only the
    # transaction (commit on success / rollback on error).
    with contextlib.closing(_state._conn()) as conn, conn, conn.cursor() as cur:
        # Serialise concurrent migrators in the database. Blocking (not
        # `try`): a process that loses the race must wait and then proceed,
        # not skip — it still needs the schema applied before it serves
        # traffic, and after the lock is released the version-tracked apply
        # loop is a clean no-op for it.
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATE_LOCK_KEY,))

        # The version ledger. Created if absent; `IF NOT EXISTS` so a process
        # that lost the advisory-lock race (ledger already created by the
        # winner) does not error.
        cur.execute("SELECT to_regclass('public.schema_migrations')")
        ledger_existed = cur.fetchone()[0] is not None
        cur.execute(
            """
CREATE TABLE IF NOT EXISTS schema_migrations (
  version    TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
            """
        )

        # Bootstrap: a legacy DB has the full schema applied by the old code
        # but no ledger. If we JUST created the ledger AND a core table
        # already exists, the schema is already current — backfill every
        # version as applied so the loop below re-runs (and re-drops) nothing.
        if not ledger_existed:
            cur.execute("SELECT to_regclass('public.tenants')")
            legacy_schema_present = cur.fetchone()[0] is not None
            if legacy_schema_present:
                for version in _MIGRATION_VERSIONS:
                    cur.execute(
                        "INSERT INTO schema_migrations(version) VALUES (%s) "
                        "ON CONFLICT (version) DO NOTHING",
                        (version,),
                    )

        # Which versions are already applied (after any bootstrap backfill).
        cur.execute("SELECT version FROM schema_migrations")
        applied = {row[0] for row in cur.fetchall()}

        # Apply each migration file exactly once, in order. Recording the
        # version in the SAME transaction means the apply + the ledger write
        # commit (or roll back) together — a crash mid-migration never leaves
        # a version marked applied that did not run, or vice versa.
        for version in _MIGRATION_VERSIONS:
            if version in applied:
                continue
            sql = (migrations_dir / f"{version}.sql").read_text(encoding="utf-8")
            cur.execute(sql)
            cur.execute(
                "INSERT INTO schema_migrations(version) VALUES (%s) "
                "ON CONFLICT (version) DO NOTHING",
                (version,),
            )

        # OSS bootstrap: idempotently insert the `default` tenant.
        #
        # When `RB_REQUIRE_AUTH` is unset/false (the self-host headline path)
        # `current_tenant_id` short-circuits to the literal string "default"
        # for every request — every dataset / api_keys / shard row created in
        # OSS mode therefore carries `tenant_id = 'default'`. Those tables FK
        # to `tenants(id)` (see 002_datasets_tenant_isolation.sql) so the row
        # must exist before any request lands; the simplest place to guarantee
        # that is at migration time. ON CONFLICT keeps the bootstrap idempotent
        # across restarts and across the legacy-DB backfill path above.
        #
        # `password_hash='!disabled!'` is intentionally a non-bcrypt-hash
        # sentinel: bcrypt.checkpw() will never match against it, so even if a
        # self-hoster later flips `RB_REQUIRE_AUTH=true` no one can `/auth/login`
        # as this tenant by guessing a password. The email is a documented dud
        # (`self-host@localhost`) so it cannot collide with any real signup.
        cur.execute(
            """
INSERT INTO tenants (id, email, password_hash, plan)
VALUES ('default', 'self-host@localhost', '!disabled!', 'oss')
ON CONFLICT (id) DO NOTHING
            """
        )


def migrate_recall(force: bool = False) -> bool:
    """Apply the recall-tier (pgvector) schema against `RB_RECALL_DSN`.

    Returns True if migrations were applied (or would have been — i.e. a recall
    DSN is configured), False if the recall tier is OFF (no `RB_RECALL_DSN`) and
    nothing was done. The boolean lets the migration entrypoint print an accurate
    "skipped (recall tier off)" vs "applied" line.

    DEFAULT-OFF: with no `RB_RECALL_DSN` this is a pure no-op — it never opens a
    connection, never imports a pgvector dependency, and leaves behaviour
    identical to today. This is the property that keeps a flag-off
    `docker compose up` byte-identical.

    Memory mode (`DATABASE_URL=memory://...`) does NOT suppress this: the recall
    tier is a wholly separate instance addressed by its own DSN, so a test or a
    deploy can point `RB_RECALL_DSN` at a real pgvector while the control plane
    runs in memory. The gate is `RB_RECALL_DSN`, not `DATABASE_URL`.

    The `_RECALL_MIGRATE_LOCK` / `_RECALL_MIGRATED` pair serialises within a
    process; `_apply_recall_migrations()` takes a Postgres `pg_advisory_xact_lock`
    so several processes booting at once serialise in the recall database instead
    of racing the DDL locks — identical discipline to `migrate()`.

    `force=True` mirrors `migrate(force=True)`: it bypasses the `RB_SKIP_MIGRATE`
    early-return so the dedicated migration entrypoint always applies the schema
    even when it inherits that flag from the service env.
    """
    dsn = _state._recall_dsn()
    if dsn is None:
        # Recall tier off — nothing to migrate, nothing connects to a recall store.
        return False
    if not force and os.getenv("RB_SKIP_MIGRATE", "").lower() in ("1", "true", "yes"):
        # Schema applied out-of-band (same contract as the control-plane
        # migrate()); a recall DSN IS configured, so report True.
        return True
    with _RECALL_MIGRATE_LOCK:
        if _state._RECALL_MIGRATED:
            return True
        _apply_recall_migrations(dsn)
        _state._RECALL_MIGRATED = True
    return True


def _apply_recall_migrations(dsn: str) -> None:
    """Apply the ordered recall migration files exactly once each (version-tracked).

    A faithful copy of `_apply_migrations()`'s cross-process-safe pattern, but
    against the recall DSN and the `recall_schema_migrations` ledger:

      - a dedicated (non-pooled) connection to `dsn` — the recall instance has no
        application connection pool;
      - `pg_advisory_xact_lock(_RECALL_MIGRATE_LOCK_KEY)` first, so concurrent
        migrators serialise in the recall database rather than deadlocking on the
        `CREATE EXTENSION` / `CREATE TABLE` locks;
      - a `recall_schema_migrations(version, applied_at)` ledger so a re-run
        applies only un-applied versions. The recall migrations are all
        `IF NOT EXISTS` (additive, non-destructive), so even a direct re-execute
        is safe; the ledger keeps the common path a clean skip.

    There is no legacy-bootstrap branch (unlike the control plane): the recall
    instance is brand new with the recall tier, so there is never a pre-existing
    recall schema without a ledger to reconcile.
    """
    migrations_dir = Path(__file__).parent / "migrations" / "recall"
    # Route the migrator's dedicated connection through the single recall
    # connection seam too, so EVERY recall connection — pooled, dedicated, and
    # the migration runner — carries the txn-mode `prepare_threshold=None`
    # marker. The migrator's `pg_advisory_xact_lock` is TRANSACTION-scoped and
    # its DDL runs in one transaction on this dedicated (non-pooled) connection,
    # so `_recall_connect()`'s contract is a perfect fit.
    conn = _state._recall_connect(dsn)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(%s)", (_RECALL_MIGRATE_LOCK_KEY,)
            )
            cur.execute(
                """
CREATE TABLE IF NOT EXISTS recall_schema_migrations (
  version    TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
                """
            )
            cur.execute("SELECT version FROM recall_schema_migrations")
            applied = {row[0] for row in cur.fetchall()}
            for version in _RECALL_MIGRATION_VERSIONS:
                if version in applied:
                    continue
                sql = (migrations_dir / f"{version}.sql").read_text(
                    encoding="utf-8"
                )
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO recall_schema_migrations(version) VALUES (%s) "
                    "ON CONFLICT (version) DO NOTHING",
                    (version,),
                )
    finally:
        conn.close()
