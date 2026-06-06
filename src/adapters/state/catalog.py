from __future__ import annotations

"""Dataset + shard catalog CRUD for the state adapter.

Extracted from `adapters.state.state` (behaviour-preserving). This module holds:

  * the memory-mode catalog-invalidation notify hooks
    (`subscribe_catalog_notify_memory`, `unsubscribe_catalog_notify_memory`,
    `_fire_catalog_notify_memory`);
  * the per-dataset build advisory lock (`dataset_build_lock`,
    `_dataset_lock_objid`, `_BUILD_LOCK_CLASS`) — a SESSION-level
    `pg_advisory_lock` on the dedicated `_conn()`;
  * dataset CRUD (`create_dataset`, `get_dataset`, `list_datasets`,
    `delete_dataset`, `update_dataset_status`, `find_stale_datasets`,
    `fail_dataset_if_stale`, `upsert_dataset`, `increment_row_count`,
    `set_row_count`) + the reaper helpers (`_parse_iso`,
    `_NON_TERMINAL_STATUSES`); and
  * shard CRUD (`add_shard`, `list_shards`, `get_latest_shard`,
    `delete_shards`).

Mutable process-wide state — `_MEMORY_MODE`, the in-memory catalog stores
(`_MEM_DATASETS`, `_MEM_SHARDS`, `_MEM_SHARD_ID`) and the notify-hook registry
(`_CATALOG_NOTIFY_HOOKS` + its lock) — is OWNED by `adapters.state.state` and
reached here through the `_state.X` reference at CALL time (never at import time),
so that `importlib.reload(state)` (which recreates those stores fresh) and
`monkeypatch.setattr(state, …)` (the test suite reads/patches
`state_mod._MEM_DATASETS` / `_MEM_SHARDS` / `_MEM_SHARD_ID` /
`_CATALOG_NOTIFY_HOOKS` directly) are both honoured. The shared helper `_now_iso`
and the control-plane connection helpers (`pooled_conn`, `_conn`) likewise live
in `state` and are reached via `_state.X` so a monkeypatch of `pooled_conn` is
observed. See `pooling.py` for the full seam rationale.

`generations.py` reaches `list_shards` (which lives here) via `_state.list_shards`,
so the delta-tier generation logic keeps resolving it through the same seam.

Backend split (memory:// vs Postgres)
-------------------------------------
The dataset/shard CRUD functions used to be an `if _state._MEMORY_MODE:` memory
block followed by a Postgres block. Those forks are now collapsed: each such
public function has ONE body that selects the backend ONCE via `_backend()` and
delegates to it. `_backend()` reads `_state._MEMORY_MODE` at CALL time, so a
`monkeypatch.setattr(state, "_MEMORY_MODE", …)` / `importlib.reload(state)` flips
which backend a subsequent call uses, exactly as the inline branch did. The
memory and Postgres code moved verbatim into `_MemoryBackend` / `_PostgresBackend`,
so behaviour is identical for BOTH backends. `dataset_build_lock` is the one
catalog function NOT collapsed — its memory path is a yield-only no-op while its
Postgres path acquires/releases a SESSION-level advisory lock on a dedicated
`_conn()`, a connection-lifecycle divergence (not a store fork) that a backend
delegate would obscure; it keeps its explicit `_MEMORY_MODE` branch.
"""

import contextlib
import datetime as _dt
import hashlib
import json
import logging
import time
from typing import Callable, List, Optional, Tuple

from psycopg2.extras import RealDictCursor

# The state module owns the mutable process-wide globals (`_MEMORY_MODE`, the
# `_MEM_*` catalog stores, the notify-hook registry) + the shared helper
# `_now_iso` + the control-plane connection helpers (`pooled_conn`, `_conn`).
# Reference them through `_state.X` at call time. Imported here, but every access
# is deferred to call time (no import-time use), so the partial-init of `state`
# during its own import of this module is safe.
import adapters.state.state as _state


# --- Memory-mode catalog-invalidation notify hooks ------------------------


def subscribe_catalog_notify_memory(
    callback: Callable[[dict], None]
) -> Callable[[dict], None]:
    """Register a hook fired on `add_shard` in memory mode.

    Returns the callback so a test can pass it directly to `unsubscribe`
    without keeping a separate handle. Idempotent: registering the same
    callback twice fires it twice (mirrors Postgres LISTEN semantics —
    each subscriber gets one delivery per notify).
    """
    with _state._CATALOG_NOTIFY_HOOKS_LOCK:
        _state._CATALOG_NOTIFY_HOOKS.append(callback)
    return callback


def unsubscribe_catalog_notify_memory(callback: Callable[[dict], None]) -> bool:
    """Remove a previously registered hook. Returns True if removed."""
    with _state._CATALOG_NOTIFY_HOOKS_LOCK:
        try:
            _state._CATALOG_NOTIFY_HOOKS.remove(callback)
            return True
        except ValueError:
            return False


def _fire_catalog_notify_memory(payload: dict) -> None:
    """Fan a notify payload out to memory-mode hook subscribers.

    A subscriber that raises is logged-and-skipped — the catalog insert
    has already completed before this runs, and we MUST NOT let a buggy
    cache-invalidation subscriber make the catalog row look like it
    failed. Snapshot the list under the lock so a concurrent
    unsubscribe does not mutate it mid-iteration.
    """
    with _state._CATALOG_NOTIFY_HOOKS_LOCK:
        snapshot = list(_state._CATALOG_NOTIFY_HOOKS)
    for cb in snapshot:
        try:
            cb(payload)
        except Exception:  # noqa: BLE001 - best-effort dispatch
            logging.getLogger(__name__).warning(
                "catalog notify hook raised; continuing", exc_info=True,
            )


# --- Per-dataset build advisory lock (multi-worker safety) ----------------


# Namespace constant for the per-dataset builder advisory lock. The two-int
# form `pg_try_advisory_lock(classid, objid)` is used: `_BUILD_LOCK_CLASS`
# fixes the high 32 bits so a builder lock can never collide with any other
# advisory lock in the system, and the low 32 bits are a hash of
# `tenant + dataset` so distinct datasets get distinct locks.
_BUILD_LOCK_CLASS = 0x42554C44  # ASCII "BULD" — the builder-lock namespace


def _dataset_lock_objid(tenant: str, dataset: str) -> int:
    """Hash `(tenant, dataset)` to a signed 32-bit advisory-lock object id.

    A stable hash (not Python's salted `hash()`) so every process derives the
    same lock id for the same dataset. Masked to a signed 32-bit int — the
    range `pg_try_advisory_lock(int4, int4)` accepts.
    """
    digest = hashlib.sha1(f"{tenant}\x00{dataset}".encode("utf-8")).digest()
    val = int.from_bytes(digest[:4], "big")
    # Map an unsigned 32-bit value into the signed int4 range Postgres expects.
    if val >= 0x80000000:
        val -= 0x100000000
    return val


@contextlib.contextmanager
def dataset_build_lock(tenant: str, dataset: str):
    """Try to acquire the per-dataset builder advisory lock; yield True/False.

    Multi-worker safety (Change 3): when `index_builder` is replicated, two
    builder replicas can pick up two `DATASET_READY` messages for the SAME
    dataset concurrently (or a redelivered message races the original). Both
    would read the same landing parts and fold them in — double-indexing.

    This serialises builds *per dataset* with a Postgres SESSION-level advisory
    lock (`pg_try_advisory_lock`). It is NON-blocking: a caller that loses the
    race yields `False` and the builder skips the build. Skipping is only safe
    if the skipped `DATASET_READY` message is RE-DELIVERED, not discarded — the
    skipped message may represent a NEWER upload than the in-progress build, so
    those parts must still be indexed eventually. The consume loop must
    therefore `nack(msg, requeue=True)` on a skip (NOT `ack`); the retry then
    either re-indexes any still-unindexed landing parts or is a clean no-op via
    the newest shard's `indexed_landing_uris` manifest. See `run_once` /
    `index_builder.main_loop`.

    Connection-scope dependency (Finding 3). The session-level advisory lock
    MUST be acquired and released on the same connection, which is why this
    uses a dedicated `_conn()`. Correctness depends on `_conn()` NOT being
    pooled — a pooled connection returned to the pool while still holding the
    session lock would leak the lock to the next borrower. See `_conn`.

    Concurrent builds of *different* datasets get distinct lock ids and still
    run fully in parallel.

    `memory://` / single-process test mode has no Postgres and no concurrency,
    so there is nothing to serialise — it always yields `True` (the build
    proceeds) and is a pure no-op.

    NOT collapsed into the `_backend()` split: the two paths diverge in
    CONNECTION LIFECYCLE (a yield-only no-op contextmanager vs an
    acquire/yield/release around a dedicated session connection), not in which
    store a value comes from. Keeping the explicit `_MEMORY_MODE` branch makes
    that divergence legible instead of hiding it behind a backend delegate.

    Usage:
        with dataset_build_lock(tenant, dataset) as acquired:
            if not acquired:
                return  # another builder owns this dataset; skip
            ...  # do the build
    """
    if _state._MEMORY_MODE:
        yield True
        return
    objid = _dataset_lock_objid(tenant, dataset)
    conn = _state._conn()
    try:
        # autocommit so the lock is held at the SESSION level for the whole
        # `with` body, independent of any transaction the build itself runs.
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_try_advisory_lock(%s, %s)",
                (_BUILD_LOCK_CLASS, objid),
            )
            acquired = bool(cur.fetchone()[0])
        try:
            yield acquired
        finally:
            if acquired:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT pg_advisory_unlock(%s, %s)",
                        (_BUILD_LOCK_CLASS, objid),
                    )
    finally:
        # Closing the connection also releases any still-held session lock —
        # a safety net if the explicit unlock above could not run.
        conn.close()


# --- Reaper helpers (backend-independent) ---------------------------------


_NON_TERMINAL_STATUSES = ("validating", "indexing")


def _parse_iso(value) -> Optional[_dt.datetime]:
    """Parse an ISO-8601 string (or pass through a datetime) to aware UTC."""
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=_dt.timezone.utc)
    if isinstance(value, str):
        try:
            dt = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=_dt.timezone.utc)
    return None


# ==========================================================================
# Backend split: memory:// store vs Postgres
# ==========================================================================
#
# Every method below is the verbatim body of one half of a former
# `if _state._MEMORY_MODE:` fork in the dataset/shard CRUD. The two classes
# share identical signatures and contracts; only the storage differs.
# `_backend()` picks one per call. The shared notify payload, the `now`
# timestamp, the reaper `cutoff`, and the shard-column normalisation are all
# computed in the public wrapper (backend-independent), exactly as before, and
# handed to the backend so neither method recomputes them.


class _MemoryBackend:
    """In-memory (`memory://`) implementation of the dataset/shard CRUD. All
    mutable state lives on `_state`."""

    # --- Datasets ---------------------------------------------------------

    def create_dataset(self, tenant_id, dataset_name, dimension, now):
        key = (tenant_id, dataset_name)
        existing = _state._MEM_DATASETS.get(key)
        if existing is not None and not existing.get("deleted_at"):
            raise ValueError("dataset_exists")
        row = {
            "tenant_id": tenant_id,
            "dataset_name": dataset_name,
            "dimension": dimension,
            "row_count": 0,
            "status": "empty",
            "error_message": None,
            "source_uris": [],
            "landing_format": "jsonl",
            "object_store_uri": None,
            "last_indexed_at": None,
            "status_updated_at": now,
            "deleted_at": None,
            "created_at": now,
        }
        _state._MEM_DATASETS[key] = row
        return dict(row)

    def get_dataset(self, tenant_id, dataset_name):
        row = _state._MEM_DATASETS.get((tenant_id, dataset_name))
        if row is None or row.get("deleted_at"):
            return None
        return dict(row)

    def list_datasets(self, tenant_id):
        out = [
            dict(row)
            for (tid, _), row in _state._MEM_DATASETS.items()
            if tid == tenant_id and not row.get("deleted_at")
        ]
        out.sort(key=lambda r: r["dataset_name"])
        return out

    def delete_dataset(self, tenant_id, dataset_name):
        row = _state._MEM_DATASETS.get((tenant_id, dataset_name))
        if row is None or row.get("deleted_at"):
            return False
        row["deleted_at"] = _state._now_iso()
        # Hard-delete the matching shard rows so list_shards returns []
        # for this (tenant, dataset) immediately.
        _state._MEM_SHARDS[:] = [
            r for r in _state._MEM_SHARDS
            if not (
                r["tenant_id"] == tenant_id
                and r["dataset_name"] == dataset_name
            )
        ]
        # Fire the memory-backend NOTIFY so the DP's catalog cache (and
        # any other in-process subscriber) sees the invalidation.
        _fire_catalog_notify_memory({
            "tenant": tenant_id,
            "dataset": dataset_name,
            "shard_uri": "",
        })
        return True

    def update_dataset_status(self, tenant_id, dataset_name, status, error_message, last_indexed_at):
        row = _state._MEM_DATASETS.get((tenant_id, dataset_name))
        if row is None:
            return
        row["status"] = status
        row["status_updated_at"] = _state._now_iso()
        if error_message is not None:
            row["error_message"] = error_message
        elif status != "error":
            # Clear stale error on a non-error transition.
            row["error_message"] = None
        if last_indexed_at is not None:
            row["last_indexed_at"] = last_indexed_at

    def find_stale_datasets(self, statuses, cutoff):
        out: List[dict] = []
        for row in _state._MEM_DATASETS.values():
            if row.get("deleted_at") or row.get("status") not in statuses:
                continue
            updated = _parse_iso(row.get("status_updated_at"))
            if updated is None or updated <= cutoff:
                out.append(dict(row))
        return out

    def fail_dataset_if_stale(self, tenant_id, dataset_name, error_message, statuses, cutoff):
        row = _state._MEM_DATASETS.get((tenant_id, dataset_name))
        if row is None or row.get("deleted_at"):
            return False
        if row.get("status") not in statuses:
            # A worker already moved it to a terminal status — do not clobber.
            return False
        updated = _parse_iso(row.get("status_updated_at"))
        if updated is not None and updated > cutoff:
            return False
        row["status"] = "error"
        row["status_updated_at"] = _state._now_iso()
        row["error_message"] = error_message
        return True

    def upsert_dataset(self, tenant_id, dataset_name, dimension, source_uri, landing_format):
        key = (tenant_id, dataset_name)
        row = _state._MEM_DATASETS.get(key)
        if row is None:
            row = {
                "tenant_id": tenant_id,
                "dataset_name": dataset_name,
                "dimension": dimension,
                "row_count": 0,
                "status": "empty",
                "error_message": None,
                "source_uris": [],
                "landing_format": landing_format,
                "object_store_uri": None,
                "last_indexed_at": None,
                "status_updated_at": _state._now_iso(),
                "deleted_at": None,
                "created_at": _state._now_iso(),
            }
            _state._MEM_DATASETS[key] = row
        row["landing_format"] = landing_format
        row.setdefault("source_uris", []).append(source_uri)

    def increment_row_count(self, tenant_id, dataset_name, count):
        row = _state._MEM_DATASETS.get((tenant_id, dataset_name))
        if row:
            row["row_count"] = row.get("row_count", 0) + count

    def set_row_count(self, tenant_id, dataset_name, value):
        row = _state._MEM_DATASETS.get((tenant_id, dataset_name))
        if row:
            row["row_count"] = value

    # --- Shards -----------------------------------------------------------

    def add_shard(self, tenant_id, dataset_name, shard_uri, checksum, vector_count,
                  index_type, build_type, uris, consolidated_lsn, quantizer_version,
                  parent_shard_id, level, covered_lsn_lo, covered_lsn_hi, tombstones,
                  notify_payload):
        _state._MEM_SHARD_ID += 1
        record = {
            "id": _state._MEM_SHARD_ID,
            "tenant_id": tenant_id,
            "dataset_name": dataset_name,
            "shard_uri": shard_uri,
            "checksum": checksum,
            "vector_count": vector_count,
            "index_type": index_type,
            "build_type": build_type,
            "indexed_landing_uris": uris,
            "consolidated_lsn": consolidated_lsn,
            "sealed": True,
            "supersedes": [],
            "created_at": time.time(),
            "quantizer_version": quantizer_version,
            "parent_shard_id": parent_shard_id,
            "level": level,
            "covered_lsn_lo": covered_lsn_lo,
            "covered_lsn_hi": covered_lsn_hi,
            "tombstone_int_ids": tombstones,
        }
        _state._MEM_SHARDS.append(record)
        # Fire AFTER the row is appended so any subscriber that immediately
        # re-reads `list_shards` sees the new row (the contract the DP cache
        # listener relies on).
        _fire_catalog_notify_memory(notify_payload)
        return _state._MEM_SHARD_ID

    def list_shards(self, tenant_id, dataset_name):
        return [
            dict(r)
            for r in sorted(_state._MEM_SHARDS, key=lambda x: x["id"], reverse=True)
            if r["tenant_id"] == tenant_id and r["dataset_name"] == dataset_name
        ]

    def delete_shards(self, tenant_id, dataset_name, id_set):
        before = len(_state._MEM_SHARDS)
        _state._MEM_SHARDS[:] = [
            r for r in _state._MEM_SHARDS
            if not (
                r["tenant_id"] == tenant_id
                and r["dataset_name"] == dataset_name
                and r["id"] in id_set
            )
        ]
        return before - len(_state._MEM_SHARDS)


class _PostgresBackend:
    """Postgres implementation of the dataset/shard CRUD. Every method runs
    through `_state.pooled_conn()` so a monkeypatched pool is observed and the
    request-scoped connection seam is honoured."""

    # --- Datasets ---------------------------------------------------------

    def create_dataset(self, tenant_id, dataset_name, dimension, now):
        with _state.pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check existing non-deleted row first to give a clean error code.
            cur.execute(
                """
SELECT 1 FROM dataset_catalog
WHERE tenant_id=%s AND dataset_name=%s AND deleted_at IS NULL
                """,
                (tenant_id, dataset_name),
            )
            if cur.fetchone() is not None:
                raise ValueError("dataset_exists")
            # Upsert: a soft-deleted row with the same PK gets resurrected so we
            # do not violate the (tenant_id, dataset_name) primary key.
            cur.execute(
                """
INSERT INTO dataset_catalog(tenant_id, dataset_name, dimension)
VALUES (%s, %s, %s)
ON CONFLICT (tenant_id, dataset_name) DO UPDATE
SET dimension=EXCLUDED.dimension,
    row_count=0,
    status='empty',
    error_message=NULL,
    source_uris='{}',
    last_indexed_at=NULL,
    deleted_at=NULL,
    created_at=now()
RETURNING *
                """,
                (tenant_id, dataset_name, dimension),
            )
            return dict(cur.fetchone())

    def get_dataset(self, tenant_id, dataset_name):
        with _state.pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
SELECT * FROM dataset_catalog
WHERE tenant_id=%s AND dataset_name=%s AND deleted_at IS NULL
                """,
                (tenant_id, dataset_name),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def list_datasets(self, tenant_id):
        with _state.pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
SELECT * FROM dataset_catalog
WHERE tenant_id=%s AND deleted_at IS NULL
ORDER BY dataset_name
                """,
                (tenant_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def delete_dataset(self, tenant_id, dataset_name):
        with _state.pooled_conn() as conn, conn.cursor() as cur:
            # Purge first to keep the FK valid throughout the transaction.
            cur.execute(
                """
DELETE FROM shard_catalog
WHERE tenant_id=%s AND dataset_name=%s
                """,
                (tenant_id, dataset_name),
            )
            cur.execute(
                """
UPDATE dataset_catalog
SET deleted_at = now()
WHERE tenant_id=%s AND dataset_name=%s AND deleted_at IS NULL
                """,
                (tenant_id, dataset_name),
            )
            modified = cur.rowcount > 0
            if modified:
                # Best-effort NOTIFY — payload schema matches `add_shard` so
                # the DP's existing `_on_catalog_notify` invalidates the
                # `(tenant, dataset)` cache entry without any new wiring. The
                # shard purge above is the correctness mechanism; this is the
                # latency optimisation that closes the `RB_CATALOG_FRESHNESS_S`
                # stale-cache window.
                try:
                    cur.execute(
                        "SELECT pg_notify(%s, %s)",
                        (
                            "catalog_updates",
                            json.dumps({
                                "tenant": tenant_id,
                                "dataset": dataset_name,
                                "shard_uri": "",
                            }),
                        ),
                    )
                except Exception:  # noqa: BLE001 - best-effort emission
                    logging.getLogger(__name__).warning(
                        "pg_notify(catalog_updates) on delete_dataset failed; "
                        "TTL safety net will cover",
                        exc_info=True,
                    )
            return modified

    def update_dataset_status(self, tenant_id, dataset_name, status, error_message, last_indexed_at):
        with _state.pooled_conn() as conn, conn.cursor() as cur:
            if error_message is not None:
                cur.execute(
                    """
UPDATE dataset_catalog
SET status=%s, error_message=%s, status_updated_at=now()
    {set_indexed}
WHERE tenant_id=%s AND dataset_name=%s
                    """.format(set_indexed=", last_indexed_at=now()" if last_indexed_at else ""),
                    (status, error_message, tenant_id, dataset_name),
                )
            else:
                cur.execute(
                    """
UPDATE dataset_catalog
SET status=%s, status_updated_at=now(),
    error_message = CASE WHEN %s = 'error' THEN error_message ELSE NULL END
    {set_indexed}
WHERE tenant_id=%s AND dataset_name=%s
                    """.format(set_indexed=", last_indexed_at=now()" if last_indexed_at else ""),
                    (status, status, tenant_id, dataset_name),
                )

    def find_stale_datasets(self, statuses, cutoff):
        with _state.pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
SELECT * FROM dataset_catalog
WHERE deleted_at IS NULL
  AND status = ANY(%s)
  AND status_updated_at <= %s
                """,
                (list(statuses), cutoff),
            )
            return [dict(r) for r in cur.fetchall()]

    def fail_dataset_if_stale(self, tenant_id, dataset_name, error_message, statuses, cutoff):
        with _state.pooled_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
UPDATE dataset_catalog
SET status='error', error_message=%s, status_updated_at=now()
WHERE tenant_id=%s AND dataset_name=%s
  AND deleted_at IS NULL
  AND status = ANY(%s)
  AND status_updated_at <= %s
                """,
                (error_message, tenant_id, dataset_name, list(statuses), cutoff),
            )
            return cur.rowcount > 0

    def upsert_dataset(self, tenant_id, dataset_name, dimension, source_uri, landing_format):
        with _state.pooled_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
INSERT INTO dataset_catalog(tenant_id, dataset_name, dimension, source_uris, landing_format)
VALUES (%s, %s, %s, ARRAY[%s]::text[], %s)
ON CONFLICT (tenant_id, dataset_name) DO UPDATE
SET source_uris = array_append(dataset_catalog.source_uris, EXCLUDED.source_uris[1]),
    landing_format = EXCLUDED.landing_format
                """,
                (tenant_id, dataset_name, dimension, source_uri, landing_format),
            )

    def increment_row_count(self, tenant_id, dataset_name, count):
        with _state.pooled_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
UPDATE dataset_catalog
SET row_count = row_count + %s
WHERE tenant_id=%s AND dataset_name=%s
                """,
                (count, tenant_id, dataset_name),
            )

    def set_row_count(self, tenant_id, dataset_name, value):
        with _state.pooled_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
UPDATE dataset_catalog
SET row_count = %s
WHERE tenant_id=%s AND dataset_name=%s
                """,
                (value, tenant_id, dataset_name),
            )

    # --- Shards -----------------------------------------------------------

    def add_shard(self, tenant_id, dataset_name, shard_uri, checksum, vector_count,
                  index_type, build_type, uris, consolidated_lsn, quantizer_version,
                  parent_shard_id, level, covered_lsn_lo, covered_lsn_hi, tombstones,
                  notify_payload):
        with _state.pooled_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
INSERT INTO shard_catalog(
  tenant_id, dataset_name, shard_uri, checksum, vector_count, index_type,
  build_type, indexed_landing_uris, consolidated_lsn,
  quantizer_version, parent_shard_id, level, covered_lsn_lo, covered_lsn_hi,
  tombstone_int_ids)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                """,
                (
                    tenant_id, dataset_name, shard_uri, checksum, vector_count,
                    index_type, build_type, uris, consolidated_lsn,
                    quantizer_version, parent_shard_id, level, covered_lsn_lo,
                    covered_lsn_hi, tombstones,
                ),
            )
            shard_id = cur.fetchone()[0]
            # `pg_notify(channel, payload)` runs IN THE SAME TRANSACTION as the
            # INSERT. Postgres holds the notify until commit and delivers it to
            # every subscriber atomically with the row becoming visible — so no
            # listener can observe the notify and then re-query `list_shards`
            # without seeing the new row. Best-effort: if `pg_notify` itself
            # raises (it shouldn't — the call is in-process to Postgres), log and
            # continue. The catalog row is the source of truth; the notify is
            # the optimisation, not the correctness mechanism (the DP's TTL
            # safety net still discovers the row within `RB_CATALOG_FRESHNESS_S`).
            try:
                cur.execute(
                    "SELECT pg_notify(%s, %s)",
                    ("catalog_updates", json.dumps(notify_payload)),
                )
            except Exception:  # noqa: BLE001 - best-effort emission
                logging.getLogger(__name__).warning(
                    "pg_notify(catalog_updates) failed; TTL safety net will cover",
                    exc_info=True,
                )
            return shard_id

    def list_shards(self, tenant_id, dataset_name):
        with _state.pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
SELECT * FROM shard_catalog
WHERE tenant_id=%s AND dataset_name=%s
ORDER BY created_at DESC, id DESC
                """,
                (tenant_id, dataset_name),
            )
            return [dict(r) for r in cur.fetchall()]

    def delete_shards(self, tenant_id, dataset_name, id_set):
        with _state.pooled_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
DELETE FROM shard_catalog
WHERE tenant_id=%s AND dataset_name=%s AND id = ANY(%s)
                """,
                (tenant_id, dataset_name, list(id_set)),
            )
            return cur.rowcount


_MEMORY_BACKEND = _MemoryBackend()
_POSTGRES_BACKEND = _PostgresBackend()


def _backend():
    """Select the storage backend ONCE, reading `_state._MEMORY_MODE` at CALL
    time so a `monkeypatch.setattr(state, "_MEMORY_MODE", …)` /
    `importlib.reload(state)` flips it exactly as the inline branch did."""
    return _MEMORY_BACKEND if _state._MEMORY_MODE else _POSTGRES_BACKEND


# --- Datasets -------------------------------------------------------------


def create_dataset(tenant_id: str, dataset_name: str, dimension: int) -> dict:
    """Insert a brand-new dataset row owned by `tenant_id`.

    Returns the persisted row dict. Raises ValueError("dataset_exists") if
    a non-deleted row with the same `(tenant_id, dataset_name)` already
    exists. Soft-deleted rows (`deleted_at IS NOT NULL`) are treated as
    absent: re-creating a previously-deleted dataset is permitted and
    resurrects the slot with status=`empty`.
    """
    now = _state._now_iso()
    return _backend().create_dataset(tenant_id, dataset_name, dimension, now)


def get_dataset(tenant_id: str, dataset_name: str) -> Optional[dict]:
    """Return the dataset row for `(tenant_id, dataset_name)`, excluding
    soft-deleted rows. Returns None if missing OR not owned by caller —
    the v1 contract maps both to 404 `dataset_not_found`.
    """
    return _backend().get_dataset(tenant_id, dataset_name)


def list_datasets(tenant_id: str) -> List[dict]:
    """Return non-deleted datasets owned by `tenant_id`, ordered by name."""
    return _backend().list_datasets(tenant_id)


def delete_dataset(tenant_id: str, dataset_name: str) -> bool:
    """Soft-delete a dataset AND purge its `shard_catalog` rows.

    Returns True iff a non-deleted dataset row was found and flipped.

    Why the shard purge:

      `dataset_catalog` is soft-deleted (`deleted_at=now()`) so audit /
      tenancy lookups can still resolve the row, but `shard_catalog` is
      HARD-deleted for the `(tenant_id, dataset_name)` pair. Without the
      shard purge a same-name re-create resurrects the dataset row but
      `list_shards` still returns the old shards — the query path resolves
      `latest = shards[0]`, hits whatever is still in the FAISS shard
      cache (or refaults the bytes from object storage), and serves the
      deleted dataset's vectors under the new dataset's name. The stress
      driver's I-01 scenario was exactly this: create -> ingest 5 -> delete
      -> create -> query returns 5 ghost rows with `mode:"hot"`.

      Object-storage bytes are left in place (no readers can reach them
      without a catalog row); the documented background sweeper claim shifts
      from "shards too" to "S3 objects only". The in-process FAISS shard
      cache and SSD shard tier are keyed by `shard_id` / `shard_uri`
      respectively and become unreachable on the next `list_shards`
      returning [] — LRU eviction reclaims them naturally; no special
      teardown is required for correctness.

    Why notify on delete:

      The per-`(tenant, dataset)` `_CATALOG_CACHE` on the DP caches the
      `list_shards` result for `RB_CATALOG_FRESHNESS_S` (default 5 s). A
      stale cached entry could serve the OLD shard list for up to that TTL
      after delete, which means a fast delete -> create -> query sequence
      can still hit ghosts within the 5 s window. We fire the same
      `pg_notify('catalog_updates', ...)` payload `add_shard` uses so the
      DP's existing `_on_catalog_notify` evicts the cached entry
      synchronously — no new wiring on the DP side. The notify is
      best-effort (with `RB_CATALOG_LISTEN=false` the TTL pull is the
      only invalidation), but the shard-row purge above is the
      correctness mechanism: even with the cache stale, a stale list of
      shards points at rows that no longer exist, so the query path
      resolves `shards = []` after the cache expires.

    The shard purge runs BEFORE the dataset UPDATE so the FK
    `shard_catalog -> dataset_catalog` constraint stays valid throughout
    (a child without a parent is impossible for a row instant). Both
    statements ride the same transaction.
    """
    return _backend().delete_dataset(tenant_id, dataset_name)


def update_dataset_status(
    tenant_id: str,
    dataset_name: str,
    status: str,
    error_message: Optional[str] = None,
    last_indexed_at: Optional[str] = None,
) -> None:
    """Set the dataset's status and optional `error_message`/`last_indexed_at`.

    Used by the validator (`validating` → `indexing`/`error`) and the index
    builder (`indexed` on success, `error` on failure). Passing a non-None
    `error_message` overwrites any previous error. Passing None does NOT leave
    `error_message` untouched: on any non-`error` transition the column is
    actively cleared to NULL (the SQL `CASE` and the memory path both do
    this), so a stale failure message never lingers on a dataset that has
    since moved on to a healthy status. A None `error_message` *into* an
    `error` status leaves the existing message in place.

    Every call stamps `status_updated_at = now()` so the reconciliation reaper
    (`adapters/queue/reaper.py`) can age out a dataset stranded in a
    non-terminal status by a hung/dead worker.
    """
    return _backend().update_dataset_status(
        tenant_id, dataset_name, status, error_message, last_indexed_at,
    )


def find_stale_datasets(
    older_than_seconds: float,
    statuses: Tuple[str, ...] = _NON_TERMINAL_STATUSES,
) -> List[dict]:
    """Return non-deleted datasets stuck in a non-terminal `status` too long.

    "Too long" means `status_updated_at` is older than `older_than_seconds`
    ago. This is the reconciliation reaper's backstop for a worker that hangs
    (or dies mid-job in a way that escaped queue redelivery): the reaper flips
    each returned dataset to `error` so a customer's `GET /v1/datasets/{name}`
    can never report a silently-stuck `validating`/`indexing` forever.

    Each returned dict carries at least `tenant_id`, `dataset_name`, `status`.
    """
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(
        seconds=older_than_seconds
    )
    return _backend().find_stale_datasets(statuses, cutoff)


def fail_dataset_if_stale(
    tenant_id: str,
    dataset_name: str,
    older_than_seconds: float,
    error_message: str,
    statuses: Tuple[str, ...] = _NON_TERMINAL_STATUSES,
) -> bool:
    """Conditionally flip a stuck dataset to `error` — compare-and-set.

    The reconciliation reaper observes a dataset as stale, then flips it. In
    the gap between those two steps a worker may legitimately finish the job
    and write a terminal status (`indexed`). An unconditional
    `update_dataset_status(..., "error")` would then clobber that good result.

    This is the guarded flip: it sets `status='error'` ONLY IF the dataset is
    STILL in a non-terminal status AND still stale. In Postgres the `WHERE`
    clause is the compare-and-set — the DB serialises it against the worker's
    own `UPDATE`, so whichever commits last is the only writer that matters
    and a terminal status is never overwritten. The memory path re-checks the
    status and the timestamp under no lock but in a single function (the
    `_MEM_QUOTA_LOCK` does not cover datasets); it is best-effort guarded —
    good enough for the test-only in-process mode.

    Returns True iff the dataset was actually flipped to `error`.
    """
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(
        seconds=older_than_seconds
    )
    return _backend().fail_dataset_if_stale(
        tenant_id, dataset_name, error_message, statuses, cutoff,
    )


def upsert_dataset(
    tenant_id: str,
    dataset_name: str,
    dimension: int,
    source_uri: str,
    landing_format: str,
) -> None:
    """Insert or update a dataset catalog row (validator-side helper).

    Adds the source URI to `source_uris` and updates `landing_format`. Used
    by the validator worker which may receive records for a dataset that
    was registered out-of-band (legacy `/register-source`) and not yet in
    the catalog.
    """
    return _backend().upsert_dataset(
        tenant_id, dataset_name, dimension, source_uri, landing_format,
    )


def increment_row_count(tenant_id: str, dataset_name: str, count: int) -> None:
    """Increment accumulated ingested row count for a dataset.

    Called by the validator after a successful landing write so the dataset's
    `row_count` reflects newly-accepted rows even before the builder commits.
    This is a transient over-count when the batch upserts existing ids:
    `set_row_count` (called by the builder after `add_shard`) reconciles
    `row_count` to the true count of unique live ids — the newest shard's
    `vector_count` — at build commit. Without that reconcile, a re-ingest of
    the same id would double `row_count` on every retry.
    """
    return _backend().increment_row_count(tenant_id, dataset_name, count)


def set_row_count(tenant_id: str, dataset_name: str, count: int) -> None:
    """Set the dataset's `row_count` to an exact value (build-time reconcile).

    The index builder calls this AFTER `add_shard` with the just-built
    shard's `vector_count` — equal to `index.ntotal` after the upsert's
    `remove_ids` + `add_with_ids`, which is the authoritative count of
    unique live ids in the dataset (one shard per dataset is the steady-
    state invariant: the sweep retains the newest shard, the second-newest
    is the in-flight-query grace buffer, older are purged).

    This reconciles the validator's `increment_row_count` over-count when
    a batch upserts existing ids — the validator increments by `len(good)`
    regardless of whether those ids already exist, so a re-ingest of the
    same id would double `row_count`. Setting (not incrementing)
    makes the reconcile idempotent and self-healing for any pre-existing drift.

    Floored at 0 — a negative `count` is treated as 0 rather than letting a
    bug elsewhere store a nonsense value.
    """
    value = max(0, int(count))
    return _backend().set_row_count(tenant_id, dataset_name, value)


# --- Shards ---------------------------------------------------------------


def add_shard(
    tenant_id: str,
    dataset_name: str,
    shard_uri: str,
    checksum: str,
    vector_count: int,
    index_type: str,
    build_type: str = "full",
    indexed_landing_uris: Optional[List[str]] = None,
    consolidated_lsn: int = 0,
    quantizer_version: int = 0,
    parent_shard_id: Optional[int] = None,
    level: int = 0,
    covered_lsn_lo: int = 0,
    covered_lsn_hi: int = 0,
    tombstone_int_ids: Optional[List[int]] = None,
) -> int:
    """Insert a new shard record and return its ID.

    The six delta-tier columns (migration 009) describe a shard's place in the
    base+delta LSM generation (see docs/architecture/recall-consolidate.md and
    bench-lab compaction redesign). They default to a `level=0` base with no
    parent so every existing caller is unchanged:
      - `quantizer_version`: the frozen-coarse-quantizer generation this shard's
        IVF cells share. Only same-version shards may MERGE (search across
        versions is fine — each shard is searched independently).
      - `parent_shard_id`: for a `level=1` delta, the base shard it layers on
        (NULL for a base). Liveness/sweep is by generation membership via this
        pointer, never by list position.
      - `level`: `0` = base, `1` = delta.
      - `covered_lsn_lo`/`covered_lsn_hi`: the recall-LSN band this shard covers,
        used to build the query's contiguous-frontier watermark (I1). A base
        covers `[0, consolidated_lsn]`; a consolidate-delta covers
        `[prev_hi+1, its consolidated_lsn]`.
      - `tombstone_int_ids`: int64 ids (hash of the string id) deleted from the
        cold tier by this fold, suppressed at query time and physically purged
        at major compaction (no S3 tombstone object, no IVF `remove_ids`).

    Two columns support incremental indexing:
      - `build_type`: `'full'` (trained-from-scratch), `'incremental'`
        (existing index loaded, only new vectors `index.add()`-ed),
        `'delete'` (existing index loaded, one vector removed by id — a
        delete-driven rebuild, labelled distinctly so deletes are not
        miscounted as ingests in `build_type`-keyed metrics), or
        `'consolidate'` (recall→consolidated flush: the recall partition up to
        `consolidated_lsn` is folded into a new shard — see
        docs/architecture/recall-consolidate.md, "Consolidation / flush").
      - `indexed_landing_uris`: the manifest of landing parquet part URIs
        already folded into this shard. The index builder reads the *newest*
        shard's manifest to decide which landing parts are new, so a
        subsequent ingest never re-reads previously indexed uploads.

    `consolidated_lsn` (migration 008) is the recall-tier watermark: the highest
    recall LSN folded into any shard of this dataset so far (a per-dataset high-
    water mark). It partitions every vector into exactly one tier —
    `lsn <= consolidated_lsn` lives in the cold shard, `lsn >` lives in recall
    (I1). A consolidation advances it to the snapshot's `max(lsn)`; every other
    build (ingest/incremental/delete) carries the prior newest shard's value
    forward so the watermark stays monotonic — a non-consolidate fold only
    touches recall-owned rows (`lsn > watermark`) and must NOT regress it (a
    regression stalls the grace-trim and re-unions already-consolidated rows).
    The default `0` is correct only for a dataset's very first shard (no
    consolidated predecessor) and, with the flag off, for every shard. The value
    is set here at every build commit, never per recall write (the seam lives
    across two databases — I2's commit-then-trim keeps it safe).
    """
    uris = list(indexed_landing_uris or [])
    consolidated_lsn = int(consolidated_lsn or 0)
    tombstones = [int(x) for x in (tombstone_int_ids or [])]
    parent_shard_id = int(parent_shard_id) if parent_shard_id is not None else None
    quantizer_version = int(quantizer_version or 0)
    level = int(level or 0)
    covered_lsn_lo = int(covered_lsn_lo or 0)
    covered_lsn_hi = int(covered_lsn_hi or 0)
    # The payload format is shared across the memory hook and the `pg_notify`
    # channel so the DP's catalog-cache invalidator can use one parser. Keep
    # keys minimal — `pg_notify`'s payload is capped at 8000 bytes by
    # Postgres, and a DP only needs `(tenant, dataset)` to route the
    # eviction. `shard_uri` is included for diagnostics (operator can
    # `LISTEN catalog_updates` from psql and see which shard fired).
    notify_payload = {
        "tenant": tenant_id,
        "dataset": dataset_name,
        "shard_uri": shard_uri,
    }
    return _backend().add_shard(
        tenant_id, dataset_name, shard_uri, checksum, vector_count, index_type,
        build_type, uris, consolidated_lsn, quantizer_version, parent_shard_id,
        level, covered_lsn_lo, covered_lsn_hi, tombstones, notify_payload,
    )


def list_shards(tenant_id: str, dataset_name: str) -> List[dict]:
    """Return shards for a `(tenant_id, dataset_name)` sorted newest-first.

    The ordering must be a TOTAL order: `get_latest_shard` selects `shards[0]`
    and `_generations` walks the list to assign each base/delta to a generation,
    so a tie would make those selections nondeterministic and could mis-bound the
    sweep/grace-trim. Memory mode orders by the monotonic insertion `id` (already
    total). The
    Postgres path orders by `created_at DESC` but `created_at` is
    `TIMESTAMPTZ DEFAULT now()` (transaction-start time), so two shards built in
    the same transaction window could share it; the `id DESC` tiebreaker (the
    serial PK, strictly increasing) makes the order provably total and matches
    memory mode's `id`-desc semantics.
    """
    return _backend().list_shards(tenant_id, dataset_name)


def get_latest_shard(tenant_id: str, dataset_name: str) -> Optional[dict]:
    """Return the newest shard for `(tenant_id, dataset_name)`, or None.

    The newest shard is the current/authoritative one: the query path loads
    it, and the index builder reads its `indexed_landing_uris` manifest to
    decide which landing parts still need indexing. Newest-first ordering is
    `id` desc in memory mode and `created_at` desc in Postgres — identical to
    `list_shards`, just the head element.
    """
    shards = list_shards(tenant_id, dataset_name)
    return shards[0] if shards else None


def delete_shards(tenant_id: str, dataset_name: str, shard_ids: List[int]) -> int:
    """Delete `shard_catalog` rows by id for a `(tenant_id, dataset_name)`.

    Returns the number of rows removed. Object-storage cleanup of the shard
    `.bin`/`.meta.json` is the caller's responsibility (the catalog adapter
    does not reach into storage). Scoped by tenant/dataset so a stray id from
    another tenant can never be deleted.
    """
    if not shard_ids:
        return 0
    id_set = set(int(s) for s in shard_ids)
    return _backend().delete_shards(tenant_id, dataset_name, id_set)
