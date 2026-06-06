from __future__ import annotations

"""State adapter for dataset and shard catalogs.

Supports an in-memory mode (default when `DATABASE_URL` starts with `memory://`)
and a Postgres-backed mode otherwise. Provides functions to create schema,
manage datasets, and track index shards.

Every dataset/shard function takes `tenant_id` as its FIRST positional
argument. The in-memory mode keys `_MEM_DATASETS` by `(tenant_id, dataset_name)`
tuples and adds a `tenant_id` field to every `_MEM_SHARDS` row. Migration
002_datasets_tenant_isolation.sql rewrites the catalog tables to include
`tenant_id` in the primary key.

Module organisation (post-split)
--------------------------------
This module is now a THIN FACADE. Its public surface — every name a caller ever
reached via `import adapters.state.state as state_mod` or `from
adapters.state.state import X` — is unchanged, but the implementations live in
focused sibling modules and are re-exported here at end-of-module:

  * `adapters.state.pooling`    — control-plane pool + request-scoped conn
                                  lifecycle + the dedicated `_conn()`;
  * `adapters.state.migrations` — both schema-migration runners + ledgers + the
                                  OSS bootstrap seed;
  * `adapters.state.generations`— delta-tier generation logic (migration 009);
  * `adapters.state.quota`      — tenants + quotas + API keys + import jobs +
                                  DP-residency;
  * `adapters.state.catalog`    — dataset + shard catalog CRUD + memory-mode
                                  notify hooks + the per-dataset build lock;
  * `adapters.recall`           — the recall (pgvector) tier sub-adapter: connection
                                  factory + pool + vector CRUD + search +
                                  consolidation + the typed `RecallUnavailable`.

THIS module still OWNS the mutable process-wide state (the `_MEMORY_MODE` flag,
the in-memory `_MEM_*` stores + their locks, the run-once migration flags, the
control-plane + recall pool handles, the `_REQUEST_CONN` contextvar) and the
shared helpers (`_now_iso`, `_quota_defaults` + the quota-default constants).
Every sibling module reaches those through an `import adapters.state.state as
_state` reference, accessed at CALL time, so `monkeypatch.setattr(state_mod, …)`
and `importlib.reload(state_mod)` are both honoured exactly as before the split.
The `psycopg2.extras.execute_values` helper is also kept here as a module
attribute because the recall write path reaches it via `_state.execute_values`
(the test suite monkeypatches it on this module).
"""

import contextvars
import datetime as _dt
import os
import threading
from typing import Callable, Optional, Tuple

import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor, execute_values  # noqa: F401 (RealDictCursor/execute_values re-exported; execute_values reached by recall write path via _state.execute_values)


_MEMORY_MODE = os.getenv("DATABASE_URL", "memory://local").startswith("memory://")

# A single process-wide lock guards the in-memory quota counters so that the
# lazy daily reset + the consume check + the increment happen as one atomic
# step. Postgres mode does not use this — it relies on a single
# `UPDATE ... WHERE ... RETURNING` statement, which the DB serialises for us.
_MEM_QUOTA_LOCK = threading.Lock()

# `migrate()` is idempotent, but its idempotent DDL still takes brief
# `AccessExclusiveLock`s on Postgres. When several services run as threads in
# one process (a single-process dev/test harness) their `migrate()` calls would
# race and deadlock on those locks. The `_MIGRATE_LOCK` (in `migrations.py`) +
# this run-once flag serialise migration within a process: the first caller
# applies the schema, the rest return immediately. Harmless in production where
# each service process calls `migrate()` once. OWNED here so
# `importlib.reload(state)` resets it; `migrations.py` reads/writes it via
# `_state._MIGRATED`.
_MIGRATED = False

# Process-local run-once flag for recall migrations, mirroring `_MIGRATED`.
# OWNED here (reset on reload); `migrations.py` accesses it via
# `_state._RECALL_MIGRATED`.
_RECALL_MIGRATED = False

# Datasets are keyed by (tenant_id, dataset_name) so two tenants can reuse
# the same dataset name without collision. Shards carry tenant_id so the
# index builder and ephemeral runner can filter without joining back through
# the dataset table. OWNED here; the catalog CRUD in `catalog.py` reaches them
# via `_state._MEM_DATASETS` / `_MEM_SHARDS` / `_MEM_SHARD_ID`.
_MEM_DATASETS: dict[Tuple[str, str], dict] = {}
_MEM_SHARDS: list[dict] = []
_MEM_SHARD_ID = 0

# In-memory tenant + api_key stores. Indexed by primary identifier for
# O(1) lookup; we also expose `get_tenant_by_email` and `get_tenant_by_id`
# so callers do not depend on which key we used. OWNED here; the quota/tenant
# code in `quota.py` and the OSS bootstrap in `migrations.py` reach them via
# `_state._MEM_TENANTS` / `_MEM_TENANTS_BY_EMAIL`.
_MEM_TENANTS: dict[str, dict] = {}  # tenant_id -> row
_MEM_TENANTS_BY_EMAIL: dict[str, str] = {}  # email -> tenant_id

# In-memory api_keys store. A flat list preserving insertion order (used by
# `list_api_keys`), plus a `key_hash -> row` index so auth-time resolution
# is O(1) — see `get_api_key_by_hash`. Postgres mode reads from the
# `api_keys` table populated by migration 001_tenants_and_keys.sql; the
# `api_keys_hash_idx` index makes the equivalent lookup O(1) there too.
# Rows carry SHA-256(raw_key), not the raw key: the raw key only exists at
# creation time and is handed to the caller in the HTTP response. SHA-256
# (not bcrypt) is used so the hash is deterministic and directly indexable
# — correct for the high-entropy random `rb_live_` tokens (bcrypt stays for
# passwords).
_MEM_API_KEYS: list[dict] = []  # rows, insertion order
_MEM_API_KEYS_BY_HASH: dict[str, dict] = {}  # key_hash -> row

# In-process notify hooks for catalog invalidation (memory backend). Postgres
# mode emits `pg_notify('catalog_updates', payload)` from `add_shard`, which
# the DP's `services._common.catalog_listener` thread relays to its
# subscribers. The memory backend has no Postgres connection to NOTIFY through,
# so unit tests that need to observe an invalidation register a hook here
# instead — see `tests/unit/test_catalog_invalidation.py`. Hooks fire in
# registration order and a raising hook is logged-and-skipped (best-effort:
# the catalog INSERT must not depend on a subscriber's correctness). OWNED
# here; the notify hooks in `catalog.py` reach them via `_state.X`.
_CATALOG_NOTIFY_HOOKS: list[Callable[[dict], None]] = []
_CATALOG_NOTIFY_HOOKS_LOCK = threading.Lock()

# In-memory import_jobs store. Keyed by (tenant_id, import_id) so the
# cross-tenant 404 contract is enforced in the data layer, exactly like
# `_MEM_DATASETS`. Postgres mode reads from the `import_jobs` table created
# by migration 004_import_jobs.sql. OWNED here; `quota.py` reaches them via
# `_state._MEM_IMPORTS` / `_MEM_IMPORT_SEQ`.
_MEM_IMPORTS: dict[Tuple[str, str], dict] = {}  # (tenant_id, import_id) -> row
# Monotonic insertion counter for `_MEM_IMPORTS`. `created_at` is only
# second-resolution, so several jobs created within one second would tie on a
# newest-first sort; this sequence is the tiebreaker. Postgres mode does not
# need it — `now()` there is microsecond-resolution.
_MEM_IMPORT_SEQ = 0

# In-memory mirror of the `dp_shard_residency` table (SSD-cache feature).
# Keyed by `(dp_id, shard_uri)` so the PK semantics match the SQL branch —
# two DPs holding the same shard are independent rows. The value tuple is
# `(warm_since, last_query_at)` matching the column order. Postgres mode
# reads/writes the `dp_shard_residency` table populated by migration
# 007_dp_shard_residency.sql; the writer (services/_common/residency_writer.py)
# is the only producer in either backend. OWNED here; `quota.py` reaches them
# via `_state._MEM_DP_RESIDENCY` / `_MEM_DP_RESIDENCY_LOCK`.
_MEM_DP_RESIDENCY: dict[Tuple[str, str], Tuple[float, float]] = {}
_MEM_DP_RESIDENCY_LOCK = threading.Lock()


# --- Control-plane connection pool handles --------------------------------
#
# The process-wide pool. Built lazily on the first `pooled_conn()` checkout
# (NOT at import time — in tests `DATABASE_URL` is `memory://local`, which has
# no SQL server and no pool). `_POOL_LOCK` guards construction so two threads
# racing the first checkout do not build two pools. OWNED here; `pooling.py`
# mutates/reads them via `_state._POOL` / `_state._POOL_LOCK`.
_POOL: Optional[psycopg2.pool.ThreadedConnectionPool] = None
_POOL_LOCK = threading.Lock()


# --- request-scoped connection ------------------------------------------------
#
# Each `pooled_conn()` block historically did its own `getconn`/`commit`/
# `putconn`. The `request_scoped_connection()` context manager (driven by the
# ASGI middleware in `adapters/state/conn_middleware.py`) checks ONE connection
# out of the pool for the whole request, binds it to `_REQUEST_CONN`, and
# returns it to the pool exactly once at request end. While a request connection
# is bound, `pooled_conn()` yields THAT connection. OWNED here; `pooling.py`
# binds/reads it via `_state._REQUEST_CONN`.
#
# `contextvars.ContextVar` is the right primitive: it is task/thread-local and
# is *copied* into the thread `asyncio.to_thread` runs work on, so a connection
# bound by the middleware before an offloaded sync state call is still visible
# inside that call.
_REQUEST_CONN: contextvars.ContextVar[Optional["psycopg2.extensions.connection"]] = (
    contextvars.ContextVar("rb_request_conn", default=None)
)


# --- Recall-tier connection pool handles ----------------------------------
#
# The process-wide recall pool, plus the DSN it was built for. Built lazily on
# the first `recall_pooled_conn()` checkout (NOT at import — flag-off never gets
# here). Keyed on the DSN so a test (or a reconfigure) that rebinds
# `RB_RECALL_DSN` transparently rebuilds the pool against the new instance rather
# than handing out connections to the old one. `_RECALL_POOL_LOCK` guards
# construction so two threads racing the first checkout do not build two pools.
# OWNED here; the recall sub-adapter (`adapters.recall`) mutates/reads them via
# `_state._RECALL_POOL` / `_RECALL_POOL_DSN` / `_RECALL_POOL_LOCK`, so a
# `monkeypatch.setattr(state, "_RECALL_POOL", …)` and `importlib.reload(state)`
# are both observed (the recall pool unit tests rely on both).
_RECALL_POOL: Optional[psycopg2.pool.ThreadedConnectionPool] = None
_RECALL_POOL_DSN: Optional[str] = None
_RECALL_POOL_LOCK = threading.Lock()


# --- Shared helpers (used by more than one sibling module) -----------------


def _now_iso() -> str:
    """Return current UTC time in ISO 8601 with trailing Z (per v1 contract).

    Shared helper: used by `quota.py` (tenants / api-keys / import-jobs),
    `catalog.py` (dataset CRUD), and the OSS bootstrap in `migrations.py`. Lives
    HERE (the lowest-level owner) so every sibling reaches it via `_state._now_iso`
    without an import cycle.
    """
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_DEFAULT_VECTOR_QUOTA = 100000
_DEFAULT_DAILY_QUERY_QUOTA = 10000


def _quota_defaults() -> Tuple[int, int]:
    """Return `(vector_quota, daily_query_quota)` to stamp on a new tenant.

    Test hook: `RB_TEST_VECTOR_QUOTA` / `RB_TEST_QUERY_QUOTA` env vars, when
    set, override the default per-tenant limits at tenant-creation time. They are
    read fresh on every signup so a test (or the E2E harness) can flip them
    before calling `POST /auth/signup` to get a low-quota tenant and trigger a
    real 429 without issuing thousands of requests. NOT for production use —
    leaving them unset yields the contract defaults (100000 / 10000).

    Shared helper: used by `quota.py` (`create_tenant`) AND by the OSS bootstrap
    in `migrations.py`. Lives HERE so both reach it via `_state._quota_defaults`.
    """
    vq = os.getenv("RB_TEST_VECTOR_QUOTA")
    qq = os.getenv("RB_TEST_QUERY_QUOTA")
    vector_quota = int(vq) if vq and vq.isdigit() else _DEFAULT_VECTOR_QUOTA
    query_quota = int(qq) if qq and qq.isdigit() else _DEFAULT_DAILY_QUERY_QUOTA
    return vector_quota, query_quota


# Default DP pool — the value migration 006 stamps on every tenant. The CP
# query proxy falls back to this for an unknown tenant or a NULL column so a
# routing lookup can never fail open into an unroutable pool. OWNED here; `quota.py`
# reaches it via `_state._DEFAULT_DP_POOL`.
_DEFAULT_DP_POOL = "shared"


# ==========================================================================
# Back-compat re-exports (post-split public surface)
# ==========================================================================
#
# Every implementation now lives in a focused sibling module, but
# `adapters.state.state` MUST keep its full public surface so that every
# `import adapters.state.state as state_mod` (which reaches each name as an
# attribute) and every `from adapters.state.state import X` keeps resolving with
# ZERO caller changes. Each moved name is imported BACK here, so it resolves from
# `adapters.state.state` exactly as before — and as the SAME object, so identity
# / `isinstance` / `except` checks are unaffected (the `RecallUnavailable` /
# `PoolCheckoutTimeout` raised in the submodules are the SAME classes re-exported
# here).
#
# The extracted functions read this module's mutable globals (`_MEMORY_MODE`,
# `_POOL`, `_MEM_*`, `_MIGRATED`, `_RECALL_POOL`, …) and shared helpers
# (`_now_iso`, `_quota_defaults`, `_recall_dsn`, `_recall_connect`, `list_shards`,
# …) through an `import adapters.state.state as _state` reference, accessed at
# CALL time — so `monkeypatch.setattr(state_mod, …)` and
# `importlib.reload(state_mod)` are both honoured exactly as before the split
# (those globals + helpers are still OWNED and defined ABOVE in this module).
# Placed at the END of the module so all of them are defined before the
# submodules (which import this module) load.
#
# These imports also re-export each submodule's own module-level constants
# (e.g. `_DEFAULT_PG_POOL_MAX`, `_MIGRATION_VERSIONS`, `_DEFAULT_RECALL_POOL_MAX`)
# for `state_mod.X` completeness.
#
# IMPORT-ORDER CONTRACT: `adapters.state.state` is the canonical entry point and
# must be imported before any of its submodules — the submodules import THIS
# module at their top, and this module re-imports from them only here at the
# bottom, so importing a submodule first in a fresh interpreter would hit a
# partially-initialised cycle. In practice nothing does (every entrypoint, test,
# and tool reaches the tier via `adapters.state.state`), so the cycle never
# triggers; it is a deliberate facade trade-off, not an accident.

from adapters.state.pooling import (  # noqa: E402,F401  (intentional end-of-module re-export)
    _DEFAULT_PG_POOL_MAX,
    _DEFAULT_POOL_CHECKOUT_TIMEOUT_S,
    _PG_POOL_MIN,
    _POOL_CHECKOUT_POLL_S,
    _checkout_request_conn,
    _close_pool,
    _conn,
    _dsn,
    _get_pool,
    _getconn_with_timeout,
    _pool_checkout_timeout_s,
    _pool_max_size,
    bind_request_conn,
    finish_request_conn,
    pooled_conn,
    request_scoped_connection,
    unbind_request_conn,
)
from adapters.state.migrations import (  # noqa: E402,F401
    _MIGRATE_LOCK,
    _MIGRATE_LOCK_KEY,
    _MIGRATION_VERSIONS,
    _RECALL_MIGRATE_LOCK,
    _RECALL_MIGRATE_LOCK_KEY,
    _RECALL_MIGRATION_VERSIONS,
    _apply_migrations,
    _apply_recall_migrations,
    _bootstrap_default_tenant_memory,
    migrate,
    migrate_recall,
)
from adapters.recall import (  # noqa: E402,F401
    _DEFAULT_RECALL_POOL_MAX,
    _RECALL_CONNECTIVITY_ERRORS,
    _RECALL_POOL_MIN,
    _RECALL_PREPARE_THRESHOLD,
    _close_recall_pool,
    _get_recall_pool,
    _metadata_matches_filter,
    _pgvector_literal_to_list,
    _recall_connect,
    _recall_conn,
    _recall_dsn,
    _recall_pool_max_size,
    _to_pgvector_literal,
    recall_delete_vector,
    recall_enabled,
    recall_get_vector,
    recall_get_vector_with_embedding,
    recall_idle_partitions,
    recall_list_rows,
    recall_partition_count,
    recall_pooled_conn,
    recall_search,
    recall_snapshot_for_consolidation,
    recall_trim,
    recall_upsert_vectors,
)
from adapters.errors import (  # noqa: E402,F401  (shared class identity; re-exported here)
    PoolCheckoutTimeout,
    RecallUnavailable,
)
from adapters.state.catalog import (  # noqa: E402,F401
    _BUILD_LOCK_CLASS,
    _NON_TERMINAL_STATUSES,
    _dataset_lock_objid,
    _fire_catalog_notify_memory,
    _parse_iso,
    add_shard,
    create_dataset,
    dataset_build_lock,
    delete_dataset,
    delete_shards,
    fail_dataset_if_stale,
    find_stale_datasets,
    get_dataset,
    get_latest_shard,
    increment_row_count,
    list_datasets,
    list_shards,
    set_row_count,
    subscribe_catalog_notify_memory,
    unsubscribe_catalog_notify_memory,
    update_dataset_status,
    upsert_dataset,
)
from adapters.state.generations import (  # noqa: E402,F401
    _generations,
    _shard_frontier,
    _shard_level,
    dataset_watermark,
    grace_watermark,
    live_generation,
    superseded_shards,
)
from adapters.state.quota import (  # noqa: E402,F401
    _IMPORT_FIELDS,
    _mem_reset_locked,
    _usage_from_row,
    create_api_key,
    create_import_job,
    create_tenant,
    get_api_key,
    get_api_key_by_hash,
    get_import_job,
    get_import_job_by_id,
    get_tenant_by_email,
    get_tenant_by_id,
    get_tenant_dp_pool,
    get_usage,
    list_api_keys,
    list_dp_residency_for_dp,
    list_dp_residency_for_shard,
    list_import_jobs,
    register_dp_shard_warm,
    reset_daily_if_needed,
    revoke_api_key,
    touch_api_key_last_used,
    try_consume_query,
    try_consume_vectors,
    unregister_dp_shard_warm,
    update_import_job,
)
