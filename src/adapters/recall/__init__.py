from __future__ import annotations

"""Recall-tier (pgvector) sub-adapter for the state layer.

Extracted from `adapters.state.state` (behaviour-preserving). This package holds
the WHOLE recall tier — the separate data-plane pgvector instance addressed by
`RB_RECALL_DSN`, kept off the control-plane Postgres so a tenant write-storm
cannot starve the catalog reads on every query's critical path
(docs/architecture/recall-consolidate.md, "Blast radius"):

  * the master switch `recall_enabled()` + the DSN resolver `_recall_dsn()`;
  * the pgbouncer-TRANSACTION-mode-safe connection seam (`_recall_connect`,
    `_recall_conn`, `_RECALL_PREPARE_THRESHOLD`);
  * the recall application-side pool (`_get_recall_pool`, `recall_pooled_conn`,
    `_close_recall_pool`, `_recall_pool_max_size`);
  * the synchronous write path (`recall_upsert_vectors`);
  * the brute-force exact L2 read path (`recall_search`);
  * the get/list/delete CRUD union helpers (`recall_get_vector`,
    `recall_get_vector_with_embedding`, `recall_list_rows`,
    `recall_delete_vector`); and
  * the recall->consolidated flush helpers (`recall_snapshot_for_consolidation`,
    `recall_partition_count`, `recall_trim`, `recall_idle_partitions`).

The whole recall path is DEFAULT-OFF and gated on `recall_enabled()` (an
`RB_RECALL` + `RB_RECALL_DSN` switch); it has ZERO `_MEMORY_MODE` branches, so it
does not participate in the Memory-vs-Postgres backend split at all.

Mutable process-wide state — the recall pool + its DSN + lock (`_RECALL_POOL`,
`_RECALL_POOL_DSN`, `_RECALL_POOL_LOCK`), the `_RECALL_MIGRATED` run-once flag,
and `execute_values` (the psycopg2 helper the test suite monkeypatches on the
state module) — is OWNED by `adapters.state.state` and reached here through the
`_state.X` reference at CALL time (never at import time), so that:

  * `monkeypatch.setattr(state, "_RECALL_POOL", fake)` / `..., "execute_values",
    ...` / `..., "recall_pooled_conn", ...` / `..., "_get_recall_pool", ...` /
    `..., "_recall_conn", ...` is observed by these functions verbatim (the test
    suite relies on this); and
  * `importlib.reload(state)` (which recreates those globals fresh) is observed
    too — the `state` module object identity is stable across a reload, so a held
    `_state` reference always sees the current globals.

This is the same seam `pooling.py` / `migrations.py` use; see `pooling.py`'s
module docstring for the full rationale. `RecallUnavailable` and
`PoolCheckoutTimeout` are the SINGLE classes from `adapters.errors`, so every
`except RecallUnavailable` / `except PoolCheckoutTimeout` frame (here and in
callers) keeps matching by identity.
"""

import contextlib
import datetime as _dt
import json
import time
from typing import Iterator, List, Optional, Tuple

import psycopg2
import psycopg2.pool

from adapters import config
from adapters.errors import PoolCheckoutTimeout, RecallUnavailable
from adapters.observability import metrics as obs_metrics
from adapters.observability.tracing import recall_search_span, state_connect_span

# The state module owns the mutable recall globals (`_RECALL_POOL`,
# `_RECALL_POOL_DSN`, `_RECALL_POOL_LOCK`, `_RECALL_MIGRATED`) + `execute_values`,
# and re-exports the pooling helpers this module reuses (`_getconn_with_timeout`,
# `_pool_checkout_timeout_s`). Reference everything through `_state.X` at call
# time so monkeypatches and `importlib.reload(state)` are both honoured. Imported
# here, but every access is deferred to call time (no import-time use), so the
# partial-init of `state` during its own import of this module is safe.
from adapters.state._lazy_state import state as _state  # lazy proxy: resolves the facade at call time (breaks the import cycle)

# Embedded in-process numpy recall backend (the memtable). Selected by
# `_use_memory_backend()` when recall is on with no DSN (the all-in-one /
# no-docker mode). `memtable` is stdlib + numpy only (no psycopg2), and it
# imports `_metadata_matches_filter` from this package LAZILY (at call time
# inside `_filter_matches`), so this module-top import has no import cycle.
from adapters.recall import memtable  # noqa: E402


_DEFAULT_RECALL_POOL_MAX = 10  # per-process ceiling; override with RB_RECALL_POOL_MAX
_RECALL_POOL_MIN = 1  # connections kept warm even when idle


# psycopg2 connectivity error types that mean "the recall store is unreachable"
# (as opposed to a query/constraint error against a healthy backend). Both are
# subclasses of `psycopg2.Error`, NOT of `OSError`, so the hot-path classifier's
# `OSError` branch never caught them — that is exactly why an un-wrapped one used
# to fall through to the generic `ephemeral_error` 500 (benchmark finding C2).
_RECALL_CONNECTIVITY_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)


# txn-mode marker: never persist named prepared stmts (see `_recall_connect`).
_RECALL_PREPARE_THRESHOLD = None


def _recall_dsn() -> Optional[str]:
    """Return the recall-tier (pgvector) DSN from `RB_RECALL_DSN`, or None if unset.

    `None` means the recall tier is not configured — every recall path is a
    no-op and the deploy behaves exactly as it does today. A blank/whitespace
    value is treated as unset so an empty compose default cannot accidentally
    enable it.
    """
    return config.recall_dsn()


def _use_memory_backend() -> bool:
    """Whether the EMBEDDED in-process numpy memtable backs the recall tier.

    Read fresh on every call (the seam is `config.recall_backend()` —
    `RB_RECALL_BACKEND`, default `auto`). The single decision point routing every
    recall_* function to `memtable` instead of pgvector:

      - `memory` — always the embedded memtable (forces no-docker mode).
      - `auto`   — the embedded memtable when recall is ON and no `RB_RECALL_DSN`
        is configured (the all-in-one eval default); else the pgvector path.
      - `pgvector` (or any other value) — never the memtable.

    Keeps the pgvector path byte-identical whenever a DSN is set: with a DSN,
    `auto` resolves to pgvector, so a real deploy is unchanged.
    """
    backend = config.recall_backend()
    if backend == "memory":
        return True
    if backend == "auto":
        return config.recall() and _recall_dsn() is None
    return False


def recall_enabled() -> bool:
    """Whether the synchronous recall-tier write path is active.

    Mirrors `quotas_enabled()` (the OSS opt-in idiom): defaults OFF and reads
    the env fresh on every call so a test can flip it via monkeypatch without a
    module reload. It is the MASTER switch for recall-tier behaviour.

    `RB_RECALL` must be truthy (`1`/`true`/`yes`/`on`, case-insensitive). Then
    EITHER of two stores must be available:
      - the pgvector recall tier — `RB_RECALL_DSN` is configured (non-empty); or
      - the embedded in-process memtable — `_use_memory_backend()` selects it
        (the all-in-one / no-docker mode, which needs NO DSN).

    A deploy that flips `RB_RECALL=true` but neither points `RB_RECALL_DSN` at an
    instance NOR selects the embedded backend stays on the byte-identical
    flag-off path rather than erroring on every write.
    """
    if not config.recall():
        return False
    return _recall_dsn() is not None or _use_memory_backend()


def _recall_conn() -> "psycopg2.extensions.connection":
    """Return a fresh, DEDICATED psycopg2 connection to the recall store.

    A brand-new connection to `RB_RECALL_DSN` (a full TCP + TLS + auth
    handshake). This is NO LONGER the per-op recall path: every recall CRUD /
    consolidation function now checks a connection out of the recall pool via
    `recall_pooled_conn()` instead (see below).

    RETAINED FOR TESTS / LOW-LEVEL USE: this has no production callers today (the
    migration runner mints its dedicated connection via `_recall_connect()`
    directly, and the hot path is pooled). It is kept as the thin
    DSN-resolving + recall-off-guard wrapper around `_recall_connect()` for the
    unit suite (which monkeypatches / drives it) and as a low-level dedicated-
    connection factory — not dead code. It was never the hot per-query path it
    used to be (~48ms/op of pure handshake when one was opened per recall op).

    Lazy by construction: only reached when `recall_enabled()` is True. With the
    flag off no caller reaches here and no recall pool is built, so no connection
    to a recall instance is ever opened — the property that keeps a flag-off
    deploy byte-identical.

    Raises `RuntimeError` if `RB_RECALL_DSN` is unset — that is a programming
    error (a caller reached the recall path with the tier off), not an expected
    runtime state.
    """
    dsn = _recall_dsn()
    if dsn is None:
        raise RuntimeError(
            "recall tier is off (RB_RECALL_DSN unset); no recall connection"
        )
    return _recall_connect(dsn)


# --- Recall-tier connection factory: pgbouncer TRANSACTION-mode safety -----
#
# The recall tier is run behind pgbouncer in TRANSACTION pooling mode in prod
# (a server connection is held only for the duration of a transaction, then
# returned to pgbouncer's pool — so NO session-level state may span
# transactions). EVERY recall connection — pooled (`_get_recall_pool`), the
# dedicated factory (`_recall_conn`), AND the migration runner
# (`_apply_recall_migrations`) — is minted HERE so that txn-mode guarantee
# lives in exactly one place.
#
# TXN-MODE SAFETY AUDIT of the recall code path (recall_upsert_vectors,
# recall_delete_vector, recall_search, recall_list_rows, recall_get_vector*,
# recall_partition_count, recall_trim, recall_snapshot_for_consolidation,
# recall_idle_partitions, and the recall LSN-block allocation):
#
#   * Server-side PREPARED STATEMENTS — psycopg2 (this codebase; psycopg3 is NOT
#     installed) does NOT emit named server-side prepared statements: a
#     parameterised `cur.execute` uses the unnamed statement, bound+executed in
#     one round-trip, never persisted on the server across checkouts. So there is
#     no named prepared statement to outlive a pgbouncer-pooled server
#     connection. (psycopg3's `prepare_threshold` — the knob the task names — does
#     not exist on psycopg2.connect; the equivalent guarantee is structural here.)
#     `prepare_threshold` is set to None as an EXPLICIT, FORWARD-COMPATIBLE marker
#     of intent: documented, asserted in the unit suite, and the single seam a
#     future psycopg3 migration must keep at None for txn-mode safety. psycopg2
#     silently ignores the unknown kwarg, so behaviour is unchanged today.
#   * SET / session GUCs — none. No `SET`, no `set_session()`, no per-connection
#     options string; isolation stays the server default (READ COMMITTED) and
#     each recall op relies only on a STATEMENT-level snapshot (recall_search /
#     recall_list_rows / recall_snapshot_for_consolidation are single-statement
#     by design — see their docstrings), never on session-spanning state.
#   * LISTEN/NOTIFY — none on the recall tier (catalog NOTIFY is control-plane
#     only).
#   * ADVISORY LOCKS across transactions — none on the recall path. The recall
#     MIGRATION runner (`_apply_recall_migrations`) takes `pg_advisory_xact_lock`
#     (TRANSACTION-scoped, auto-released at commit/rollback) on a DEDICATED,
#     non-pooled connection — txn-mode-safe and not on the hot pooled path. The
#     only SESSION-level advisory lock (`dataset_build_lock`, autocommit) is on
#     the CONTROL-PLANE connection (`_conn()`), NOT the recall tier, and uses a
#     dedicated connection it owns and closes — it never touches a pgbouncer-
#     pooled recall checkout.
#   * Server-side NAMED CURSORS — none. Every recall cursor is a plain client
#     cursor; results are `fetchall()`/`fetchone()`'d within the same
#     transaction, so no portal spans a checkout.
#   * Two statements assuming a shared session beyond one txn — none. Every
#     recall op opens ONE transaction on ONE `recall_pooled_conn()` checkout and
#     commits/rolls back before the connection is returned (the multi-statement
#     ops — upsert's LSN-alloc+UPSERT, delete's LSN-alloc+tombstone — are the
#     SAME single transaction on the SAME checkout). Nothing is carried across.
#
# CONCLUSION: the recall path was ALREADY txn-mode-pgbouncer-safe by
# construction; this factory only HARDENS and DOCUMENTS that (the
# `prepare_threshold=None` marker + the single-seam guarantee), it does not fix
# an unsafe operation.


def _recall_connect(dsn: str) -> "psycopg2.extensions.connection":
    """Open ONE recall connection in a pgbouncer-TRANSACTION-mode-safe way.

    The single seam through which every recall connection (pooled, dedicated,
    and the migration runner) is created, so the transaction-pooling guarantee
    documented above lives in one place. Sets `prepare_threshold=None` so no NAMED server-side prepared
    statement is ever created on a recall connection — the property that lets a
    server connection be safely returned to pgbouncer's transaction pool after
    each transaction without leaking session state to the next borrower.

    `prepare_threshold` is a psycopg3 knob; psycopg2 (the installed driver) never
    auto-prepares. Passing it as `None` is a deliberate no-op on psycopg2:
    `psycopg2.extensions.make_dsn` DROPS kwargs whose value is None, so the
    connection params are byte-identical to a plain `psycopg2.connect(dsn)` today
    — the safety on psycopg2 is structural, this is only the documented, tested
    marker. (NOTE: a NON-None value here — e.g. an int — would raise psycopg2's
    "invalid connection option"; the txn-safe value is None, so keep it None.) A
    future psycopg3 migration MUST keep this at None to stay txn-mode-safe.
    """
    return psycopg2.connect(dsn, prepare_threshold=_RECALL_PREPARE_THRESHOLD)


# --- Recall-tier application-side connection pool --------------------------
#
# `_recall_conn()` opened a brand-new connection per recall op — a full TCP +
# TLS + auth handshake (~48ms against a managed pgvector instance). A single
# `POST /vectors` or `POST /v1/query` paid that tax on every recall read/write.
# This pool mirrors the control-plane pool (`_POOL` / `pooled_conn()`)
# one-for-one, but against the SEPARATE recall DSN (`RB_RECALL_DSN`): it opens a
# small set of recall connections once and hands them out / takes them back via
# `recall_pooled_conn()`.
#
# Independence from the control-plane pool is deliberate and load-bearing: the
# recall tier is a DIFFERENT database instance (its own DSN), so it MUST have its
# own pool — a control-plane connection cannot talk to the recall store and vice
# versa. One recall pool PER PROCESS, exactly like the control-plane pool.
#
# DEFAULT-OFF / lazy: the pool is built on the FIRST `recall_pooled_conn()`
# checkout, which is only ever reached when `recall_enabled()` is True. With
# `RB_RECALL_DSN` unset no recall function runs, so the pool is NEVER built and
# no connection to a recall instance is ever opened — the flag-off deploy stays
# byte-identical (the `psycopg2.connect`-raises tests still pass untouched).
#
# The mutable pool globals (`_RECALL_POOL`, `_RECALL_POOL_DSN`, `_RECALL_POOL_LOCK`)
# are OWNED by `adapters.state.state` and reached via `_state.X` so that a
# `monkeypatch.setattr(state, "_RECALL_POOL", ...)` and `importlib.reload(state)`
# are both observed here (the recall pool unit tests rely on both).


def _recall_pool_max_size() -> int:
    """Return the per-process recall-pool ceiling.

    Mirrors `_pool_max_size()`: defaults to `_DEFAULT_RECALL_POOL_MAX` (10).
    `RB_RECALL_POOL_MAX`, when set to a positive integer, overrides it — useful
    to fit the recall pool under the recall instance's connection cap when many
    worker processes run side by side. A missing or malformed value falls back
    to the default rather than crashing.
    """
    return config.recall_pool_max()


def _get_recall_pool(
    maxconn_override: Optional[int] = None,
) -> "psycopg2.pool.ThreadedConnectionPool":
    """Return the process-wide recall pool, building it lazily on first use.

    Mirrors `_get_pool()`: a `ThreadedConnectionPool` (the services run threaded
    under uvicorn, so checkout/return must be thread-safe) built against the
    recall DSN. Keyed on `RB_RECALL_DSN` — if the DSN changed since the pool was
    built (a test rebind, a reconfigure), the old pool is torn down and a fresh
    one is built against the new instance, so a checkout never returns a
    connection to a stale recall store.

    `maxconn_override` forces the pool's max size; it is a test hook (a tiny
    max-1 pool makes a connection leak immediately fatal). In production the size
    comes from `_recall_pool_max_size()`.

    Raises `RuntimeError` if `RB_RECALL_DSN` is unset — reaching here with the
    tier off is a programming error, identical to `_recall_conn()`.
    """
    dsn = _recall_dsn()
    if dsn is None:
        raise RuntimeError(
            "recall tier is off (RB_RECALL_DSN unset); no recall connection pool"
        )
    if _state._RECALL_POOL is None or _state._RECALL_POOL_DSN != dsn:
        with _state._RECALL_POOL_LOCK:
            # Re-check under the lock — another thread may have built it while
            # this one waited.
            if _state._RECALL_POOL is None or _state._RECALL_POOL_DSN != dsn:
                if _state._RECALL_POOL is not None:
                    # DSN changed: discard the pool bound to the old instance.
                    try:
                        _state._RECALL_POOL.closeall()
                    except Exception:
                        pass
                maxconn = maxconn_override or _recall_pool_max_size()
                # `ThreadedConnectionPool` forwards **kwargs to `psycopg2.connect`
                # for EVERY backend it opens, so threading the txn-mode marker
                # (`prepare_threshold=None`) through here makes every POOLED recall
                # connection match `_recall_connect`'s pgbouncer-transaction-mode
                # guarantee — no named server-side prepared statement is ever
                # created on a recall connection, so a server connection can be
                # safely recycled by pgbouncer's transaction pool after each txn.
                # (None is dropped by psycopg2's make_dsn → byte-identical params
                # today; it is the explicit, tested, psycopg3-forward marker.)
                _state._RECALL_POOL = psycopg2.pool.ThreadedConnectionPool(
                    minconn=_RECALL_POOL_MIN,
                    maxconn=max(_RECALL_POOL_MIN, maxconn),
                    dsn=dsn,
                    prepare_threshold=_RECALL_PREPARE_THRESHOLD,
                )
                _state._RECALL_POOL_DSN = dsn
    return _state._RECALL_POOL


def _close_recall_pool() -> None:
    """Close every connection in the recall pool and discard it.

    Mainly a test teardown hook so a test that rebinds `RB_RECALL_DSN` does not
    leave a pool pinned to a stopped recall container. Safe to call when no pool
    exists. Production processes are long-lived and let the pool live for the
    process lifetime.
    """
    with _state._RECALL_POOL_LOCK:
        if _state._RECALL_POOL is not None:
            _state._RECALL_POOL.closeall()
            _state._RECALL_POOL = None
            _state._RECALL_POOL_DSN = None


@contextlib.contextmanager
def recall_pooled_conn(
    maxconn_override: Optional[int] = None,
) -> Iterator["psycopg2.extensions.connection"]:
    """Check a recall connection out of the recall pool; commit/return on exit.

    The pool-aware replacement for `with contextlib.closing(_recall_conn()) as
    conn, conn:` at every recall call site. It mirrors `pooled_conn()`'s
    STANDALONE path exactly (the recall tier has no request-scoped connection
    middleware — it is a separate data-plane instance, so there is no
    `_REQUEST_CONN` analogue here; every recall op is its own short transaction):

      - on enter: check a connection out of the recall pool (block-with-timeout —
        on `PoolError` it poll-retries until the checkout deadline, so a
        transient burst blocks rather than 500s, and only a sustained exhaustion
        raises `PoolCheckoutTimeout`);
      - on normal exit: `commit()`, then return the connection to the pool;
      - on exception: `rollback()`, then return the connection, then re-raise —
        the connection is returned in EVERY path, never leaked, and a borrower
        never inherits a half-applied or aborted transaction.

    This yields ONE connection for the WHOLE `with` block, so a multi-statement
    recall transaction (e.g. `recall_upsert_vectors`: allocate an LSN block with
    `UPDATE ... RETURNING`, then UPSERT, then commit) runs entirely on a SINGLE
    checked-out connection — the LSN allocation and the UPSERT are the same
    transaction on the same backend, committed once on clean exit. The
    single-connection-per-transaction guarantee the old `closing(_recall_conn())`
    block gave is preserved verbatim; only the connection's lifecycle changed
    from open/close to checkout/return.

    `maxconn_override` is a test hook forwarded to `_get_recall_pool`.

    Only ever reached when `recall_enabled()` is True (every caller is gated), so
    with the flag off the recall pool is never built and no recall connection is
    ever opened.
    """
    # `_get_recall_pool` is reached via `_state` so a `monkeypatch.setattr(state,
    # "_get_recall_pool", ...)` is observed (the recall pool unit tests rely on
    # the same `state.X` patch seam the control-plane pool uses).
    pool = _state._get_recall_pool(maxconn_override=maxconn_override)
    # `state.connect` span — annotated so a trace distinguishes a ~0ms reuse from
    # a genuine new-backend open, exactly as the control-plane pool does. The
    # pool keeps `_RECALL_POOL_MIN` warm and only opens a new backend when every
    # kept connection is checked out, so a checkout that finds a free connection
    # in `_pool` is a reuse.
    reused = bool(getattr(pool, "_pool", None))
    with state_connect_span(reused=reused):
        conn = _state._getconn_with_timeout(pool, _state._pool_checkout_timeout_s())
    try:
        yield conn
    except BaseException:
        # Roll back so the connection is returned to the pool clean — never
        # carrying a half-applied or aborted transaction into the next borrower —
        # then re-raise.
        try:
            conn.rollback()
        finally:
            pool.putconn(conn)
        raise
    else:
        # Return the connection on EVERY exit — even if commit() itself raises
        # (broken backend / TLS reset / statement_timeout on COMMIT). Otherwise a
        # commit failure leaks the conn from the pool's accounting, permanently
        # shrinking the pool. psycopg2._putconn inspects transaction_status and
        # rolls back / closes a non-idle or server-lost conn before re-pooling,
        # so a post-commit-failure conn is returned clean, not poisoned.
        try:
            conn.commit()
        finally:
            pool.putconn(conn)


def _to_pgvector_literal(values: List[float]) -> str:
    """Format a float list as a pgvector text literal, e.g. `[1.0,2.0,3.0]`.

    pgvector accepts its `vector` input as a bracketed, comma-separated string;
    psycopg2 binds it as the parameter for the unparameterised `embedding`
    column. The dimension is NOT enforced here — the service validates each
    record's length against `dataset.dimension` before this is called, exactly
    as the cold path does.
    """
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def recall_upsert_vectors(
    tenant_id: str,
    dataset: str,
    records: List[dict],
) -> int:
    """Synchronously UPSERT validated records into the recall tier; return the count.

    For each record (`{"id", "values", "metadata"}`, already validated by the
    service): the row is UPSERTed into `recall_vectors` with last-write-wins on
    `(tenant_id, dataset, id)` and `deleted = false`, stamped with a strictly-
    monotonic per-`(tenant, dataset)` LSN from the recall store's `recall_lsn_seq`.
    A re-sent id overwrites the prior row (new embedding, metadata, and a fresh
    higher LSN) — never a duplicate.

    The LSN is generated in the RECALL store, so this path never touches the
    control-plane Postgres (docs/architecture/recall-consolidate.md, "The
    watermark"). All records for one call share a single recall connection +
    transaction: the whole batch commits atomically, so a mid-batch failure
    leaves the recall tier unchanged rather than half-applied.

    Set-based round-trips (was 2N, now ~2): the prior implementation allocated
    one LSN and ran one UPSERT *per record* (2N round-trips on the read-your-
    writes path). Instead this allocates ALL N LSNs in a SINGLE upsert-increment
    (`last_lsn = last_lsn + N RETURNING last_lsn`) and applies the batch in a
    SINGLE multi-row UPSERT. The seq-row upsert is still serialised by Postgres,
    so `last_lsn` stays strictly monotonic per (tenant, dataset) with no
    cross-dataset contention; the allocated block is `last_lsn-N+1 .. last_lsn`,
    assigned in input order.

    Intra-batch duplicate ids: if the same id appears more than once in one
    batch, only the LAST occurrence is written (last input wins), so the batch
    is collapsed to one row per id BEFORE allocation — N is the number of
    distinct ids, and a single multi-row UPSERT never lists the same conflict
    key twice (which Postgres would reject as "cannot affect row a second time").

    Only ever called when `recall_enabled()` is True (the service gates it),
    so it never runs — and never opens a connection — with the flag off.
    """
    if _use_memory_backend():
        return memtable.recall_upsert_vectors(tenant_id, dataset, records)
    if not records:
        return 0

    # Collapse intra-batch duplicate ids: later input wins (last-write-wins,
    # consistent with the cross-batch ON CONFLICT below). Preserve input order
    # of each surviving id — its position is that of its LAST occurrence, so the
    # LSN block is assigned in the order the winning records were sent. A dict
    # keeps insertion order (Python 3.7+); re-assigning an existing key updates
    # the value but NOT its position, so we del-then-set to move a repeated id
    # to the end (its latest occurrence).
    deduped: dict[str, dict] = {}
    for rec in records:
        rid = rec["id"]
        if rid in deduped:
            del deduped[rid]
        deduped[rid] = rec
    winners = list(deduped.values())
    n = len(winners)

    # ONE pooled recall connection for the WHOLE batch — the LSN-block allocation
    # and the multi-row UPSERT run in a SINGLE transaction on a SINGLE checked-out
    # connection. `recall_pooled_conn()` owns the lifecycle: it commits on clean
    # exit (the batch is all-or-nothing), rolls back on error, and returns the
    # connection to the recall pool on EVERY path. Holding one connection for the
    # duration of the txn — allocate, upsert, commit, then return — is what keeps
    # the LSN allocation and the UPSERT atomic (read-your-writes correctness).
    with _state.recall_pooled_conn() as conn, conn.cursor() as cur:
        # 1. Allocate the whole LSN block in ONE statement. The upsert-increment
        #    is serialised by Postgres on the single seq row, so `last_lsn` is
        #    strictly monotonic with no cross-dataset contention. Bumping by N at
        #    once reserves a contiguous block: the returned value is the LAST LSN
        #    in the block, and the block is `last_lsn-N+1 .. last_lsn`.
        cur.execute(
            """
INSERT INTO recall_lsn_seq (tenant_id, dataset, last_lsn)
VALUES (%s, %s, %s)
ON CONFLICT (tenant_id, dataset)
DO UPDATE SET last_lsn = recall_lsn_seq.last_lsn + %s
RETURNING last_lsn
            """,
            (tenant_id, dataset, n, n),
        )
        last_lsn = cur.fetchone()[0]
        first_lsn = last_lsn - n + 1
        # 2. SINGLE multi-row UPSERT. Each winner is stamped with its LSN from
        #    the block in input order. Last-write-wins on (tenant, dataset, id):
        #    a re-sent id overwrites embedding/metadata/lsn and clears any prior
        #    tombstone (deleted -> false), so an upserted id is live.
        rows = [
            (
                tenant_id,
                dataset,
                rec["id"],
                _to_pgvector_literal(rec["values"]),
                json.dumps(rec.get("metadata") or {}),
                first_lsn + offset,
            )
            for offset, rec in enumerate(winners)
        ]
        _state.execute_values(
            cur,
            """
INSERT INTO recall_vectors (tenant_id, dataset, id, embedding, metadata, lsn, deleted)
VALUES %s
ON CONFLICT (tenant_id, dataset, id)
DO UPDATE SET
  embedding = EXCLUDED.embedding,
  metadata  = EXCLUDED.metadata,
  lsn       = EXCLUDED.lsn,
  deleted   = FALSE
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s, FALSE)",
        )
    return n


# --- Recall-tier read path (brute-force exact search) ----------------------
#
# The query union (PR4): when the recall tier is on, `POST /v1/query` searches
# BOTH the consolidated (cold) shard AND the recall tier, then merges. This is
# the recall half — a brute-force exact L2 scan over `recall_vectors`, scoped to
# one `(tenant, dataset)` partition and to the rows ABOVE the resolved shard's
# watermark (`lsn > consolidated_lsn`). It returns BOTH live rows and tombstones
# (`deleted` rows above the watermark) so the merge in `v1_query` can suppress a
# consolidated id whose recall version is a tombstone. The metric is aligned to
# FAISS by SQUARING pgvector's plain-L2 `<->` (both tiers store the same
# un-normalised vectors). See docs/architecture/recall-consolidate.md, "Read
# path — the union" and invariants I1/I3.


def recall_search(
    tenant_id: str,
    dataset: str,
    vector: List[float],
    top_k: int,
    watermark: int,
    flt: Optional[dict] = None,
) -> Tuple[set, List[dict]]:
    """Brute-force exact L2 search of the recall tier above `watermark`.

    Returns `(suppress_ids, matches)`:

      - `suppress_ids` — the set of EVERY recall id above the watermark (every
        live row AND every tombstone), regardless of the filter and regardless of
        `top_k`. Recall is authoritative for any id with `lsn > watermark`, so the
        merge uses this set to drop the stale cold copy of that id unconditionally
        (live-but-filtered-out, ranked-past-top_k, and tombstoned ids all
        suppress their cold twin). Keying suppression on "any recall row for this
        id" — not "a recall row that became a match" — is the fix for the leak
        where a re-upsert that fails the filter (or ranks past `top_k`) let a
        stale cold copy survive.
      - `matches` — up to `top_k` filter-passing LIVE rows as dicts
        `{"id", "score", "metadata", "deleted"}`, ascending by `score`, the
        candidate recall MATCHES. Tombstones are NEVER matches; a live row that
        fails the filter is NOT a match (it only suppresses, via `suppress_ids`).

    `score` is the FAISS-aligned **L2-squared** distance — pgvector's `<->` (plain
    Euclidean L2) SQUARED, so it can be merged directly with the cold shard's
    FAISS L2² distances. Both tiers store the same UN-normalised vectors, so
    squaring is the only alignment needed (the most likely silent ranking bug —
    it has a dedicated test).

    Scope (I1 partition + I3 watermark pairing):
      WHERE tenant_id=? AND dataset=? AND lsn > :watermark
    The recall tier owns `lsn > consolidated_lsn`; the cold shard owns `<=` — so
    the union is complete with no double-count. `watermark` MUST be the
    `consolidated_lsn` of the shard the cold search ACTUALLY resolved (the caller
    pairs them), never a watermark read independently.

    The metadata `filter` (AND-of-equals, same semantics as the cold path's
    `metadata_matches_filter`) is applied to LIVE rows in Python after the scan,
    NOT pushed into SQL — the recall set is small by construction (bounded by
    consolidation cadence, not data size) so an exact scan + Python filter is
    sub-millisecond and keeps the equality semantics byte-identical to the cold
    path (exact type+value, no JSONB coercion surprises). The filter gates only
    whether a live row becomes a MATCH; it never removes an id from
    `suppress_ids`.

    SINGLE-SNAPSHOT SCAN — the suppress/match split comes from ONE SQL statement
    (task #17, b1 root cause). `suppress_ids` and `matches` are NOT two separate
    scans: they are derived in Python from the rows of a SINGLE SELECT over
    `lsn > watermark`. This is load-bearing for correctness, not a style choice:

      - The old shape ran TWO statements on a default READ COMMITTED recall
        connection — (a) a MATCH scan over live rows, then (b) a SUPPRESS scan
        over all ids. Under READ COMMITTED EACH `cur.execute` takes a FRESH MVCC
        snapshot, so a re-UPSERT that COMMITS BETWEEN scan (a) and scan (b) makes
        (b) observe ids that (a) never saw. The result: `suppress_ids ⊋
        match_ids`, the union over-suppresses, and a query that should return N
        live rows transiently returns 0 (benchmark case b1; root cause in
        bench-lab/analysis/b1-rootcause.md). The write/consolidate side
        (`recall_snapshot_for_consolidation`) was already hardened to a single
        statement for exactly this reason — this read path now matches it.
      - A SINGLE SELECT is evaluated against ONE MVCC snapshot EVEN UNDER READ
        COMMITTED (a statement-level snapshot is taken once, at statement start,
        and is stable for the whole statement). So `suppress_ids` (every returned
        id) and `matches` (the live subset) are split from the SAME consistent
        row set — `suppress_ids ⊇ match_ids` ALWAYS holds, no concurrent commit
        can wedge between them. This is precisely what eliminates the b1 race.

    The scan returns `(id, deleted, score, metadata)` so the tombstone/live split
    is done in Python BY THE EXPLICIT `deleted` FLAG, not by a `... AND NOT
    deleted` clause and not by SQL row ordering. That preserves the "no `LIMIT`
    pushdown can crowd out a real match with a tombstone" property: tombstones are
    excluded from `matches` by `if deleted` in Python (they only ever land in
    `suppress_ids`), so even a tombstone's zero-vector placeholder embedding
    (distance 0 to any query) can never displace a real live match. There is no
    SQL `LIMIT`; the live-row `top_k` truncation happens in Python over the
    live-only subset, and the recall set is small by construction (bounded by
    consolidation cadence, not data size — §Recall search).

    Only ever called when `recall_enabled()` is True (the query path gates it),
    so it never runs — and never opens a recall connection — with the flag off.
    """
    if _use_memory_backend():
        return memtable.recall_search(
            tenant_id, dataset, vector, top_k, watermark, flt
        )
    qlit = _to_pgvector_literal(vector)
    flt = flt or {}
    # `recall.search` span + `rosalinddb.recall_search.duration` metric (task #20):
    # the recall half of the union was previously invisible in traces/metrics even
    # though the query critical path waits on it. The span times the whole call
    # (the single-snapshot SQL scan + the Python suppress/match split) and is
    # stamped with `rows_scanned` (the single scan's row count) and `match_count`
    # once known. Only ever reached on the recall path (`recall_search` runs only
    # when `recall_enabled()` is True), so this never fires with the flag off.
    started = time.perf_counter()
    with recall_search_span(
        tenant=tenant_id, dataset=dataset, top_k=top_k, watermark=watermark
    ) as _span:
        # ONE statement, ONE MVCC snapshot. Select every row above the watermark
        # with its `deleted` flag and FAISS-aligned squared-L2 `score`, and split
        # it in Python: `suppress_ids` = every id, `matches` = the live,
        # filter-passing, top_k-closest subset. DO NOT split this back into two
        # scans — two `cur.execute`s are two READ COMMITTED snapshots and a
        # between-scan commit re-opens the b1 over-suppression race (see docstring).
        #
        # `power(embedding <-> %s, 2)` squares pgvector's plain-L2 `<->` so the
        # score matches FAISS L2². The query vector is bound TWICE — once for the
        # score expression, once for the ORDER BY — so the same literal drives
        # both. The `ORDER BY` ranks ALL rows (live and tombstoned) by raw
        # distance, but the Python split filters tombstones out by the `deleted`
        # flag, so ordering never lets a tombstone become a match. No SQL `LIMIT`:
        # the `top_k` truncation happens in Python over the live-only subset (the
        # recall set is small by construction — bounded by consolidation cadence,
        # not data size).
        #
        # Recall-connectivity boundary (benchmark finding C2). The recall store is
        # a SEPARATE data-plane instance; if it is down, the pool checkout (opening
        # a new backend) or the `cur.execute()` raises a `psycopg2.OperationalError`
        # / `InterfaceError`, and a sustained recall-pool exhaustion raises
        # `PoolCheckoutTimeout`. None of those are `OSError`s, so the hot-path
        # classifier used to bucket them as the generic `ephemeral_error` 500. Wrap
        # them HERE — at the recall boundary, where we KNOW the failure is the
        # recall tier — in a typed `RecallUnavailable`, so the query path maps it to
        # a retryable 503 `recall_unavailable` WITHOUT misclassifying an identical
        # exception raised by the control-plane/cold path. The original is preserved
        # as `__cause__`; a SQL/constraint error against a HEALTHY backend (anything
        # outside the connectivity set) is NOT wrapped and bubbles up unchanged —
        # only "the recall store is unreachable" becomes RecallUnavailable. The
        # `try` wraps ONLY the I/O; the Python suppress/match split below stays
        # outside it so a bug there is never masked as a recall outage.
        try:
            with _state.recall_pooled_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """
SELECT id,
       deleted,
       power(embedding <-> %s, 2) AS score,
       metadata
FROM recall_vectors
WHERE tenant_id = %s AND dataset = %s AND lsn > %s
ORDER BY embedding <-> %s ASC
                    """,
                    (qlit, tenant_id, dataset, watermark, qlit),
                )
                rows = cur.fetchall()
        except (PoolCheckoutTimeout, *_RECALL_CONNECTIVITY_ERRORS) as exc:
            raise RecallUnavailable(
                f"recall store unreachable: {type(exc).__name__}"
            ) from exc

        # `suppress_ids`: EVERY id the single scan returned (live AND tombstoned).
        # Recall is authoritative for `lsn > watermark`, so the stale cold copy of
        # any such id is always dropped — whether the recall row is a tombstone,
        # fails the filter, or ranks past `top_k`. The cold id's FAISS distance is
        # over the CONSOLIDATED embedding, which may differ from the (stale) recall
        # embedding, so we cannot assume the cold match ranks beyond `top_k` just
        # because the recall row does. Because suppress and matches come from the
        # SAME snapshot's rows, `suppress_ids ⊇ match_ids` always holds (b1 fix).
        suppress_ids: set = set()
        matches: List[dict] = []
        live_count = 0
        for rid, deleted, score, metadata in rows:
            # Every returned id suppresses its cold twin, ALWAYS — added before any
            # match/filter/top_k decision so a tombstone, a filter-failing row, or
            # a past-top_k row still drops its stale cold copy.
            suppress_ids.add(rid)
            # Tombstone: NEVER a match. It only suppresses (already added above).
            # Splitting on the explicit `deleted` flag — not a `NOT deleted` WHERE
            # clause — is what keeps the no-leak property safe even though the scan
            # has no SQL `LIMIT` and orders all rows together by raw distance.
            if deleted:
                continue
            # Live row past the top_k closest live survivors: it cannot be a MATCH
            # (the `top_k` closer live rows already beat it). Its id is already in
            # `suppress_ids`, so skip it as a candidate and keep scanning (so its
            # later id is still suppressed, even though it is not a match).
            if live_count >= top_k:
                continue
            # `metadata` is JSONB → psycopg2 decodes it to a Python dict (or None
            # for a SQL NULL); coalesce None to `{}`.
            meta = metadata or {}
            # Live row: apply the AND-of-equals filter to decide whether it is a
            # MATCH. A live row that fails the filter is NOT a match — but its id
            # is ALREADY in `suppress_ids`, so the stale cold copy is still dropped
            # (the P1 fix: an authoritative live re-upsert that fails the query
            # filter must not let a stale, filter-matching cold copy leak). The
            # predicate must match `metadata_matches_filter` exactly (type+value).
            if flt and not _metadata_matches_filter(meta, flt):
                continue
            matches.append(
                {"id": rid, "score": float(score), "metadata": meta, "deleted": False}
            )
            live_count += 1

        # Stamp the scan size + match count on the span (tenant/dataset/top_k/
        # watermark were stamped at span open). `rows` is the single scan's full
        # row count (recall rows above the watermark); `match_count` is the live
        # filter-passing top_k subset.
        try:
            _span.set_attribute("rosalinddb.rows_scanned", len(rows))
            _span.set_attribute("rosalinddb.match_count", len(matches))
        except Exception:  # noqa: BLE001 — never let observability break the query
            pass

    # Recall query-duration histogram (ms), mirroring the cold query-duration
    # metric. Recorded after the span closes so it captures the whole call.
    obs_metrics.record_recall_search_duration(
        (time.perf_counter() - started) * 1000.0
    )
    return suppress_ids, matches


def _metadata_matches_filter(metadata: dict, flt: dict) -> bool:
    """AND-of-equals predicate — the recall-side mirror of the cold path's
    `services.query_api.v1_query.metadata_matches_filter`.

    Duplicated (kept tiny and in sync) rather than imported so the state adapter
    does not take a dependency on the query service package — the same
    classifier-duplication rationale the query path already uses for its
    ephemeral-runner twin. Semantics are identical: every filter key must be
    present with an EXACTLY-equal value (same `type()` and `==`); a `null` filter
    value never matches; an empty filter matches everything.
    """
    for key, want in flt.items():
        if want is None:
            return False
        if key not in metadata:
            return False
        got = metadata[key]
        if type(got) is not type(want):
            return False
        if got != want:
            return False
    return True


# --- Recall-tier CRUD union helpers (get / list / delete) -----------------
#
# The get/list/delete-by-id union (PR6): when the recall tier is on, the
# consolidated-tier CRUD endpoints (`services.source_registry.main`) union the
# immutable shard sidecar with the recall tier, with RECALL AUTHORITATIVE for any
# id above the resolved shard's watermark — the same recall-wins / tombstone-
# suppress rule the QUERY union uses (`recall_search` + `_merge_recall_and_cold`).
# These point-lookup / partition-scan helpers back that union; they ONLY ever run
# when the endpoint reaches the `recall_enabled()`-gated path, so a flag-off
# deploy never opens a recall connection through here. See
# docs/architecture/recall-consolidate.md, "Read path — the union" / PR6.


def recall_get_vector(
    tenant_id: str, dataset: str, vector_id: str, watermark: int
) -> Tuple[Optional[str], Optional[dict]]:
    """Point-look up one id in the recall tier above `watermark`. Recall-authoritative.

    Returns a `(status, metadata)` pair the CRUD get-by-id union consumes:

      - `("live", metadata)` — a LIVE recall row with `lsn > watermark`: recall
        wins, the endpoint returns this `{id, metadata}` (its version is newer
        than any consolidated copy).
      - `("tombstone", None)` — a recall TOMBSTONE with `lsn > watermark`: the id
        was deleted in recall; the endpoint returns `404 not_found` and MUST NOT
        fall back to the (stale) cold sidecar.
      - `(None, None)` — NO recall row above the watermark for this id: the
        endpoint falls back to the cold sidecar lookup (the id, if it exists, is
        consolidated).

    Scope (I1 partition + I3 watermark pairing): `WHERE tenant_id=? AND dataset=?
    AND id=? AND lsn > :watermark`. The recall tier owns `lsn > consolidated_lsn`;
    a row at or below the watermark is already consolidated into the cold shard
    and is harmlessly ignored here (the cold sidecar is then authoritative for it).
    `watermark` MUST be the `consolidated_lsn` of the shard the caller resolved.

    Tenant-scoped via the partition key; a cross-tenant id is simply absent from
    this partition → `(None, None)` → the caller's cold lookup (also tenant-
    scoped) yields the 404. Only ever called when `recall_enabled()` is True.
    """
    if _use_memory_backend():
        return memtable.recall_get_vector(tenant_id, dataset, vector_id, watermark)
    with _state.recall_pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
SELECT metadata, deleted
FROM recall_vectors
WHERE tenant_id = %s AND dataset = %s AND id = %s AND lsn > %s
            """,
            (tenant_id, dataset, vector_id, watermark),
        )
        row = cur.fetchone()
    if row is None:
        return None, None
    metadata, deleted = row
    if deleted:
        return "tombstone", None
    return "live", (metadata or {})


def recall_get_vector_with_embedding(
    tenant_id: str, dataset: str, vector_id: str, watermark: int
) -> Tuple[Optional[str], Optional[dict], Optional[List[float]]]:
    """Like :func:`recall_get_vector`, but ALSO return the stored embedding.

    Backs the `?include_values=true` recall path for get-by-id — the cheap
    recall-resident case (no FAISS): the embedding is a plain `vector` COLUMN on
    `recall_vectors`, so this is one SELECT over the same partition/watermark
    scope as `recall_get_vector`, plus a text→`list[float]` parse of the column.

    Returns a `(status, metadata, embedding)` triple:

      - `("live", metadata, [float, ...])` — a LIVE recall row above the
        watermark: its real stored embedding is returned alongside the metadata.
      - `("tombstone", None, None)` — a recall TOMBSTONE above the watermark.
      - `(None, None, None)` — NO recall row above the watermark for this id; the
        caller falls back to the cold tier (which, for `include_values`, cannot
        currently reconstruct the embedding — a deferred FAISS `reconstruct`).

    Scope is byte-identical to `recall_get_vector` (I1 partition + I3 watermark
    pairing). Only ever called when `recall_enabled()` is True.
    """
    if _use_memory_backend():
        return memtable.recall_get_vector_with_embedding(
            tenant_id, dataset, vector_id, watermark
        )
    with _state.recall_pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
SELECT metadata, deleted, embedding
FROM recall_vectors
WHERE tenant_id = %s AND dataset = %s AND id = %s AND lsn > %s
            """,
            (tenant_id, dataset, vector_id, watermark),
        )
        row = cur.fetchone()
    if row is None:
        return None, None, None
    metadata, deleted, embedding = row
    if deleted:
        return "tombstone", None, None
    return "live", (metadata or {}), _pgvector_literal_to_list(embedding)


def recall_list_rows(
    tenant_id: str, dataset: str, watermark: int
) -> Tuple[List[dict], set]:
    """List the recall partition above `watermark` for the CRUD list union.

    Returns `(live_rows, suppress_ids)`, mirroring `recall_search`'s split so the
    list union dedups identically to the query union:

      - `live_rows` — `[{"id", "metadata"}, ...]` for every LIVE (not-deleted)
        recall row with `lsn > watermark`. These are unioned with the cold sidecar
        entries, recall-wins on the same id (recall metadata overrides cold).
      - `suppress_ids` — the FULL set of EVERY recall id above the watermark (live
        AND tombstoned). Recall is authoritative for `lsn > watermark`, so the
        stale cold copy of any such id is dropped: a tombstone HIDES the cold id
        from the list (the mirror of the query union's tombstone suppression), and
        a live recall row replaces the cold copy with the recall version.

    No metadata filter is applied here — the endpoint applies the AND-of-equals
    filter to the merged result (matching the cold list, which filters after the
    sidecar read), so the filter semantics stay byte-identical across tiers.

    Scope (I1 + I3): `WHERE tenant_id=? AND dataset=? AND lsn > :watermark`. The
    recall set is small by construction (consolidation-bounded), so materialising
    the whole partition is cheap. Only ever called when `recall_enabled()` is True.

    SINGLE-SNAPSHOT SCAN — `live_rows` and `suppress_ids` come from ONE SQL
    statement (task #30, the b1 twin on the LIST path). They are NOT two separate
    scans: they are derived in Python from the rows of a SINGLE SELECT over
    `lsn > watermark`. This is load-bearing for correctness, not a style choice,
    and mirrors what `recall_search` does on the query path (task #17):

      - The old shape ran TWO statements on a default READ COMMITTED recall
        connection — (a) a LIVE scan (`... AND NOT deleted ...`), then (b) a
        SUPPRESS scan over all ids. Under READ COMMITTED EACH `cur.execute` takes
        a FRESH MVCC snapshot, so a re-UPSERT that COMMITS BETWEEN scan (a) and
        scan (b) makes (b) observe ids that (a) never saw → `suppress_ids ⊋
        live_ids`. At the call site (`services/source_registry/main.py` list
        endpoint) the cold copy of that id is DROPPED (it is in `suppress_ids`)
        AND no recall live row is appended (the live scan missed it), so the
        record transiently VANISHES from the list — the identical b1 over-
        suppression race, on the list path.
      - A SINGLE SELECT is evaluated against ONE MVCC snapshot EVEN UNDER READ
        COMMITTED (a statement-level snapshot is taken once, at statement start,
        and is stable for the whole statement). So `suppress_ids` (every returned
        id) and `live_rows` (the not-deleted subset) are split from the SAME
        consistent row set — `suppress_ids ⊇ live_ids` ALWAYS holds, no concurrent
        commit can wedge between them. That is precisely what eliminates the race.

    The scan returns `(id, deleted, metadata)` so the tombstone/live split is done
    in Python BY THE EXPLICIT `deleted` FLAG, not by a `... AND NOT deleted`
    clause. Splitting on the flag (rather than re-deriving suppression from the
    live scan alone) is what keeps tombstone-suppress correct: deriving
    `suppress_ids` from only the live rows would re-open the tombstone leak (a
    tombstone would no longer hide its stale cold twin). Every returned id —
    live AND tombstoned — suppresses; only `deleted = false` rows become
    `live_rows`.
    """
    if _use_memory_backend():
        return memtable.recall_list_rows(tenant_id, dataset, watermark)
    # ONE statement, ONE MVCC snapshot. Select every row above the watermark with
    # its `deleted` flag and split it in Python: `suppress_ids` = every id (live
    # AND tombstoned), `live_rows` = the not-deleted subset. DO NOT split this back
    # into two scans — two `cur.execute`s are two READ COMMITTED snapshots and a
    # between-scan commit re-opens the b1-twin over-suppression race (see docstring
    # / `recall_search`, which is now single-snapshot too).
    with _state.recall_pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
SELECT id, deleted, metadata
FROM recall_vectors
WHERE tenant_id = %s AND dataset = %s AND lsn > %s
            """,
            (tenant_id, dataset, watermark),
        )
        rows = cur.fetchall()
    # Split the SINGLE snapshot's rows in Python:
    #   - `suppress_ids` — EVERY returned id (tombstone OR live). Recall is
    #     authoritative for `lsn > watermark`, so the stale cold twin of any such
    #     id is dropped, exactly as `recall_search` builds `suppress_ids`. Added
    #     for every row, before the live/tombstone decision, so a tombstone still
    #     suppresses its cold copy.
    #   - `live_rows` — `{"id", "metadata"}` for the `deleted = false` rows only;
    #     a tombstone's placeholder metadata never leaks into the list. NULL
    #     metadata (JSONB SQL NULL → None) coalesces to `{}`.
    suppress_ids: set = set()
    live_rows: List[dict] = []
    for rid, deleted, metadata in rows:
        suppress_ids.add(rid)
        if deleted:
            continue
        live_rows.append({"id": rid, "metadata": metadata or {}})
    return live_rows, suppress_ids


def recall_delete_vector(
    tenant_id: str, dataset: str, vector_id: str, dimension: int
) -> int:
    """Write an ABOVE-watermark tombstone for one id; return the allocated lsn.

    The recall-delete write path (PR6). DELETE is a last-write-wins UPSERT of a
    TOMBSTONE row (`deleted = true`) keyed on `(tenant, dataset, id)`, stamped
    with a FRESH lsn allocated from `recall_lsn_seq` — the SAME atomic upsert-
    increment the recall UPSERT uses. Returns that lsn.

    HARD CONTRACT — the lsn MUST be strictly ABOVE the current watermark (it is,
    because every allocated lsn is `> max(lsn) >= consolidated_lsn`). We
    deliberately ALLOCATE A NEW LSN rather than flip `deleted=true` in place /
    reuse the row's old lsn: a tombstone at or below the watermark would be
    EXCLUDED from every union (`lsn > consolidated_lsn` is false → the id never
    deletes) AND would be trim-eligible-but-unapplied (the consolidation could
    GC the row before folding the delete → the id resurrects). Allocating fresh,
    above the max, guarantees the tombstone is in the recall partition the union
    scans and the next consolidation applies. See
    docs/architecture/recall-consolidate.md, "Write path" (Delete) + invariants
    I1/I2 + the failure-mode "Delete then immediate query".

    `dimension` is the dataset's embedding dimension (the caller passes
    `dataset.dimension`). A brand-new tombstone — deleting an id present only in
    the COLD tier, with no prior recall row — still needs an `embedding` for the
    `NOT NULL` column, and we write a zero-vector of the dataset dimension so it
    matches the partition's dimension. NOTE: `recall_search`'s single-snapshot
    scan ranks ALL rows (live AND tombstoned) by `ORDER BY embedding <-> q`, so the
    placeholder zero-vector IS still fed to pgvector's `<->` — its dimension MUST
    match the column or the operator raises pgvector's "different vector
    dimensions" error (this is a real correctness dependency, not belt-and-
    suspenders). What a tombstone can NOT do is become a MATCH or crowd out a real
    live match: the Python split skips tombstones with `if deleted: continue`
    BEFORE the top_k / live_count logic, so the placeholder never lands in
    `matches` even though it sorts first by raw distance. It is never folded into a
    shard either (consolidation only `_remove_ids`'s a tombstoned id). On a
    conflict the existing embedding is left untouched.

    Synchronous + read-your-deletes: the row is committed before this returns, so
    an immediate GET sees the tombstone and a `POST /query` no longer returns the
    id. Only ever called when `recall_enabled()` is True (the endpoint gates it),
    so it never opens a recall connection with the flag off.
    """
    # Zero-vector placeholder of the dataset dimension (see docstring): same
    # dimension as every live row in the partition, so the search scan's `<->`
    # never hits a dimension mismatch on a cold-only delete's fresh tombstone.
    if _use_memory_backend():
        return memtable.recall_delete_vector(tenant_id, dataset, vector_id, dimension)
    placeholder = _to_pgvector_literal([0.0] * max(1, int(dimension)))
    with _state.recall_pooled_conn() as conn, conn.cursor() as cur:
        # 1. Allocate a SINGLE fresh lsn from the per-(tenant, dataset) sequence —
        #    the same atomic upsert-increment the write path uses, so the tombstone
        #    is stamped strictly above every prior lsn (and thus above any
        #    watermark, which can only be <= max(lsn)). This is the contract that
        #    keeps the tombstone inside the union's scan window.
        cur.execute(
            """
INSERT INTO recall_lsn_seq (tenant_id, dataset, last_lsn)
VALUES (%s, %s, 1)
ON CONFLICT (tenant_id, dataset)
DO UPDATE SET last_lsn = recall_lsn_seq.last_lsn + 1
RETURNING last_lsn
            """,
            (tenant_id, dataset),
        )
        lsn = cur.fetchone()[0]
        # 2. UPSERT the tombstone (last-write-wins on the partition key). An
        #    existing live/deleted row for this id is overwritten with deleted=true
        #    and the fresh higher lsn; a brand-new tombstone gets the dimension-
        #    matched zero placeholder. On a conflict the embedding is untouched.
        cur.execute(
            """
INSERT INTO recall_vectors (tenant_id, dataset, id, embedding, metadata, lsn, deleted)
VALUES (%s, %s, %s, %s, '{}'::jsonb, %s, TRUE)
ON CONFLICT (tenant_id, dataset, id)
DO UPDATE SET
  lsn     = EXCLUDED.lsn,
  deleted = TRUE
            """,
            (tenant_id, dataset, vector_id, placeholder, lsn),
        )
    return int(lsn)


# --- Recall-tier consolidation (flush) helpers ----------------------------
#
# The recall→consolidated flush (PR5): the index builder snapshots a
# (tenant, dataset) recall partition up to the current max LSN, folds the LIVE
# rows into a new Consolidated shard, applies tombstones, commits the catalog
# row with that LSN as the watermark, and THEN trims the consolidated recall
# rows — grace-bounded and idempotent (I2 + I4). The four read/finder helpers
# plus the trim below back that operation; they ONLY ever run when the builder
# reaches the consolidation path, which is gated on `recall_enabled()`, so a
# flag-off deploy never opens a recall connection through here. See
# docs/architecture/recall-consolidate.md, "Consolidation / flush".


def recall_snapshot_for_consolidation(
    tenant_id: str, dataset: str
) -> Tuple[int, List[dict]]:
    """Snapshot a recall partition for a consolidation: `(max_lsn, rows)`.

    Reads EVERY row for `(tenant, dataset)` — live rows AND tombstones — up to
    the partition's `max(lsn) = N`, in ONE statement so the bound `N` and the
    rows it selects come from a SINGLE MVCC snapshot. The bound is derived in a
    scalar sub-SELECT (`lsn <= (SELECT COALESCE(MAX(lsn),0) FROM recall_vectors
    WHERE ...)`), so the whole `SELECT` is one statement and the rows are exactly
    those at or below the max LSN as of that statement's snapshot — self-
    consistent independent of writer internals (no inter-statement gap, no
    reliance on the recall-write seq-lock / commit-order==LSN-order coupling).
    A concurrent write that commits during the snapshot has a higher LSN than the
    max captured by the sub-SELECT, so it is excluded by the `lsn <=` bound and
    left in recall — not partly-included and not lost.

    The watermark `N` is then derived from the max LSN of the RETURNED rows (`0`
    if empty), so it matches the row set exactly and the caller commits the
    catalog row with `consolidated_lsn = N`. The builder folds the live rows into
    the new shard, applies the tombstones (`deleted=true` ids are removed and
    never added), and stamps that watermark.

    Returns:
      - `max_lsn` — the highest LSN among the returned rows, the watermark the
        consolidation will stamp (`0` when the partition is empty — nothing to
        consolidate).
      - `rows` — `[{"id", "values", "metadata", "lsn", "deleted"}, ...]` for
        every row with `lsn <= max_lsn`, ascending by LSN. `values` is the
        parsed float list (pgvector's text `vector` literal → `list[float]`);
        `metadata` is the JSONB dict (coalesced to `{}`).

    A write that lands AFTER the snapshot (a higher LSN) is left in recall
    untouched: it is simply not in this consolidation's set, stays queryable via
    the union (`lsn > consolidated_lsn`), and the next consolidation folds it.
    This is what makes read-your-writes hold THROUGH a consolidation (I1 + I2).
    """
    if _use_memory_backend():
        return memtable.recall_snapshot_for_consolidation(tenant_id, dataset)
    with _state.recall_pooled_conn() as conn, conn.cursor() as cur:
        # ONE statement: derive the bound N inside the same query via a scalar
        # sub-SELECT, so N and the selected rows share a single MVCC snapshot. A
        # concurrent write (higher LSN) is excluded by the `lsn <=` bound, so the
        # snapshot is internally consistent without depending on transaction
        # isolation level or on the recall writer's seq-lock allocation order.
        cur.execute(
            """
SELECT id, embedding, metadata, lsn, deleted
FROM recall_vectors
WHERE tenant_id = %s AND dataset = %s
  AND lsn <= (
    SELECT COALESCE(MAX(lsn), 0) FROM recall_vectors
    WHERE tenant_id = %s AND dataset = %s
  )
ORDER BY lsn ASC
            """,
            (tenant_id, dataset, tenant_id, dataset),
        )
        rows: List[dict] = []
        for rid, embedding, metadata, lsn, deleted in cur.fetchall():
            rows.append(
                {
                    "id": rid,
                    "values": _pgvector_literal_to_list(embedding),
                    "metadata": metadata or {},
                    "lsn": int(lsn),
                    "deleted": bool(deleted),
                }
            )
    # Derive the watermark from the returned rows so it matches the set exactly
    # (the rows are ascending by LSN). Empty partition -> 0, nothing to do.
    max_lsn = rows[-1]["lsn"] if rows else 0
    return max_lsn, rows


def _pgvector_literal_to_list(literal) -> List[float]:
    """Parse pgvector's text `vector` literal (`[1,2,3]`) to a float list.

    psycopg2 reads an unparameterised pgvector `vector` column back as its text
    representation (a bracketed, comma-separated string) since no typecaster is
    registered. Invert `_to_pgvector_literal`. A `list`/`tuple` (already parsed
    by a future typecaster) is passed through; an empty/blank literal yields
    `[]`.
    """
    if isinstance(literal, (list, tuple)):
        return [float(v) for v in literal]
    s = str(literal).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    s = s.strip()
    if not s:
        return []
    return [float(part) for part in s.split(",")]


def recall_partition_count(tenant_id: str, dataset: str) -> int:
    """Return the number of recall rows for `(tenant, dataset)` (live + tombstones).

    Backs the per-tenant cap (`RB_RECALL_MAX_ROWS`): the write path calls this
    after a recall UPSERT and enqueues `CONSOLIDATE` when the count exceeds the
    cap. Counts every row in the partition (a tombstone still occupies a row and
    still costs the brute-force scan), so the cap genuinely bounds the recall
    set the union scans.
    """
    if _use_memory_backend():
        return memtable.recall_partition_count(tenant_id, dataset)
    with _state.recall_pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM recall_vectors "
            "WHERE tenant_id = %s AND dataset = %s",
            (tenant_id, dataset),
        )
        return int(cur.fetchone()[0] or 0)


def recall_trim(tenant_id: str, dataset: str, grace_watermark: int) -> int:
    """Hard-delete consolidated recall rows up to `grace_watermark`; return the count.

    The FINAL step of a consolidation, run STRICTLY AFTER the catalog commit
    (I2): `DELETE FROM recall_vectors WHERE lsn <= grace_watermark`. Idempotent
    and re-runnable — a crash between commit and trim leaves rows that the union
    harmlessly excludes (`lsn > consolidated_lsn`) and the next consolidation's
    trim GCs (cross-DB crash safety).

    `grace_watermark` is the GRACE-BOUNDED watermark (I4): the caller passes the
    `consolidated_lsn` of the shard that is now the **2nd-newest** (NOT the
    newest just committed), so an in-flight query that resolved an older shard
    still finds its recall rows. `grace_watermark <= 0` is a no-op (nothing has
    aged into the grace window yet — e.g. the very first consolidation, whose
    new shard is the only one).
    """
    if _use_memory_backend():
        return memtable.recall_trim(tenant_id, dataset, grace_watermark)
    if grace_watermark <= 0:
        return 0
    with _state.recall_pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM recall_vectors "
            "WHERE tenant_id = %s AND dataset = %s AND lsn <= %s",
            (tenant_id, dataset, grace_watermark),
        )
        return cur.rowcount


def recall_idle_partitions(idle_seconds: float) -> List[Tuple[str, str]]:
    """Return `(tenant_id, dataset)` partitions whose newest recall write is idle.

    Backs consolidate-on-idle (`RB_RECALL_IDLE_S`): the builder's idle-tick
    sweep calls this, then enqueues `CONSOLIDATE` for each returned partition so
    it drains to ZERO recall rows → idle queries skip pgvector entirely
    (scale-to-zero preserved). "Idle" = the partition's most recent write
    (`max(created_at)`) is older than `idle_seconds` ago AND the partition still
    has rows to drain. Returns an empty list when nothing is idle.

    Grouped per `(tenant, dataset)` so a single SQL round-trip finds every idle
    partition across the whole recall instance in one pass.
    """
    if _use_memory_backend():
        return memtable.recall_idle_partitions(idle_seconds)
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=idle_seconds)
    with _state.recall_pooled_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
SELECT tenant_id, dataset
FROM recall_vectors
GROUP BY tenant_id, dataset
HAVING MAX(created_at) <= %s
            """,
            (cutoff,),
        )
        return [(row[0], row[1]) for row in cur.fetchall()]
