from __future__ import annotations

"""Tenants + quotas + API keys + import jobs + DP-residency for the state adapter.

Extracted from `adapters.state.state` (behaviour-preserving). This module holds:

  * tenants: `create_tenant`, `get_tenant_by_email`, `get_tenant_by_id`,
    `get_tenant_dp_pool`;
  * quotas: `get_usage`, `reset_daily_if_needed` (+ the lazy daily-reset helper
    `_mem_reset_locked` and the `usage`-projection `_usage_from_row`),
    `try_consume_query`, `try_consume_vectors`;
  * API keys: `create_api_key`, `list_api_keys`, `get_api_key`, `revoke_api_key`,
    `get_api_key_by_hash`, `touch_api_key_last_used`;
  * import jobs: `create_import_job`, `get_import_job`, `get_import_job_by_id`,
    `list_import_jobs`, `update_import_job`;
  * DP residency: `register_dp_shard_warm`, `unregister_dp_shard_warm`,
    `list_dp_residency_for_shard`, `list_dp_residency_for_dp`.

Mutable process-wide state — `_MEMORY_MODE`, the in-memory stores + their locks
(`_MEM_QUOTA_LOCK`, `_MEM_TENANTS`/`_MEM_TENANTS_BY_EMAIL`, `_MEM_API_KEYS`/
`_MEM_API_KEYS_BY_HASH`, `_MEM_IMPORTS`/`_MEM_IMPORT_SEQ`, `_MEM_DP_RESIDENCY`/
`_MEM_DP_RESIDENCY_LOCK`), the quota-default consts, and the shared helpers
`_now_iso` / `_quota_defaults` — is OWNED by `adapters.state.state` and reached
here through `_state.X` at CALL time (never at import time). This keeps
`importlib.reload(state)` (which recreates the stores fresh) and
`monkeypatch.setattr(state, …)` honoured, and lets the migration bootstrap and
the catalog code (still in `state`) share the SAME store objects. `pooled_conn`
is likewise reached via `_state.pooled_conn` so a monkeypatch of it is observed.
See `pooling.py` for the full seam rationale.

Backend split (memory:// vs Postgres)
-------------------------------------
The previous shape of every function below was an `if _state._MEMORY_MODE:`
memory block followed by a Postgres block. Those forks are now collapsed: each
public function has ONE body that selects the backend ONCE via `_backend()` and
delegates to it. `_backend()` reads `_state._MEMORY_MODE` at CALL time, so a
`monkeypatch.setattr(state, "_MEMORY_MODE", …)` / `importlib.reload(state)`
flips which backend a subsequent call uses, exactly as the inline branch did.
Both backends are stateless singletons (all mutable state lives on `_state`), so
selecting one per call is free. Behaviour is byte-for-byte identical to the
pre-collapse branches for BOTH backends — the memory and Postgres code moved
verbatim into `_MemoryBackend` / `_PostgresBackend`.
"""

import datetime as _dt
from typing import List, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor

# The state module owns the mutable process-wide globals + the shared helpers
# (`_now_iso`, `_quota_defaults`) + the quota consts + `pooled_conn`. Reference
# them through `_state.X` at call time. Imported here, but every access is
# deferred to call time (no import-time use), so the partial-init of `state`
# during its own import of this module is safe.
import adapters.state.state as _state


# --- Shared (backend-independent) projection helpers ----------------------


def _usage_from_row(row: dict) -> dict:
    """Project a tenant row down to the v1 `usage` shape.

    `queries_reset_at` is normalised to a `YYYY-MM-DD` string regardless of
    whether it came from Postgres (a `date`) or the memory adapter (a string).
    """
    reset = row.get("queries_reset_at")
    if isinstance(reset, (_dt.date, _dt.datetime)):
        reset = reset.isoformat()[:10]
    elif isinstance(reset, str):
        reset = reset[:10]
    return {
        "vectors_used": int(row.get("vectors_used", 0)),
        "vector_quota": int(row.get("vector_quota", _state._DEFAULT_VECTOR_QUOTA)),
        "queries_today": int(row.get("queries_today", 0)),
        "daily_query_quota": int(row.get("daily_query_quota", _state._DEFAULT_DAILY_QUERY_QUOTA)),
        "queries_reset_at": reset,
    }


def _mem_reset_locked(row: dict, today: _dt.date) -> None:
    """Apply the lazy daily reset to an in-memory row. Caller holds the lock."""
    reset = row.get("queries_reset_at")
    if isinstance(reset, str):
        try:
            reset_date = _dt.date.fromisoformat(reset[:10])
        except ValueError:
            reset_date = today
    elif isinstance(reset, (_dt.date, _dt.datetime)):
        reset_date = reset if isinstance(reset, _dt.date) else reset.date()
    else:
        reset_date = today
    if reset_date < today:
        row["queries_today"] = 0
        row["queries_reset_at"] = today.isoformat()


_IMPORT_FIELDS = (
    "import_id", "tenant_id", "dataset", "format", "status", "error_mode",
    "max_bad_records", "upload_uri", "records_processed", "records_accepted",
    "records_rejected", "rejected_uri", "error_message", "created_at",
    "completed_at",
)

# Columns `update_import_job` is allowed to patch (the immutable identity /
# creation columns are intentionally absent). Backend-independent.
_IMPORT_MUTABLE_FIELDS = frozenset((
    "status", "records_processed", "records_accepted", "records_rejected",
    "rejected_uri", "error_message", "completed_at",
))


# ==========================================================================
# Backend split: memory:// store vs Postgres
# ==========================================================================
#
# Every method below is the verbatim body of one half of a former
# `if _state._MEMORY_MODE:` fork. The two classes share identical signatures and
# contracts; only the storage differs. `_backend()` picks one per call.


class _MemoryBackend:
    """In-memory (`memory://`) implementation of the quota/tenant/key/import/
    residency primitives. All mutable state lives on `_state`."""

    # --- Tenants ----------------------------------------------------------

    def create_tenant(self, tenant_id, email, password_hash, vector_quota, query_quota):
        if email in _state._MEM_TENANTS_BY_EMAIL:
            raise ValueError("duplicate_email")
        row = {
            "id": tenant_id,
            "email": email,
            "password_hash": password_hash,
            "plan": "free",
            "vector_quota": vector_quota,
            "daily_query_quota": query_quota,
            "vectors_used": 0,
            "queries_today": 0,
            "queries_reset_at": _dt.date.today().isoformat(),
            "created_at": _state._now_iso(),
        }
        _state._MEM_TENANTS[tenant_id] = row
        _state._MEM_TENANTS_BY_EMAIL[email] = tenant_id
        return dict(row)

    def get_tenant_by_email(self, email):
        tenant_id = _state._MEM_TENANTS_BY_EMAIL.get(email)
        if tenant_id is None:
            return None
        return dict(_state._MEM_TENANTS[tenant_id])

    def get_tenant_by_id(self, tenant_id):
        row = _state._MEM_TENANTS.get(tenant_id)
        return dict(row) if row else None

    def get_tenant_dp_pool(self, tenant_id):
        row = _state._MEM_TENANTS.get(tenant_id)
        if row is None:
            return _state._DEFAULT_DP_POOL
        return row.get("dp_pool") or _state._DEFAULT_DP_POOL

    # --- Quotas -----------------------------------------------------------

    def reset_daily_if_needed(self, tenant_id, today):
        with _state._MEM_QUOTA_LOCK:
            row = _state._MEM_TENANTS.get(tenant_id)
            if row is None:
                return
            _mem_reset_locked(row, today)

    def get_usage(self, tenant_id, today):
        with _state._MEM_QUOTA_LOCK:
            row = _state._MEM_TENANTS.get(tenant_id)
            if row is None:
                raise ValueError("tenant_not_found")
            _mem_reset_locked(row, today)
            return _usage_from_row(row)

    def try_consume_query(self, tenant_id, today):
        with _state._MEM_QUOTA_LOCK:
            row = _state._MEM_TENANTS.get(tenant_id)
            if row is None:
                raise ValueError("tenant_not_found")
            _mem_reset_locked(row, today)
            quota = int(row.get("daily_query_quota", _state._DEFAULT_DAILY_QUERY_QUOTA))
            used = int(row.get("queries_today", 0))
            if used < quota:
                row["queries_today"] = used + 1
                return True, _usage_from_row(row)
            return False, _usage_from_row(row)

    def try_consume_vectors(self, tenant_id, count):
        with _state._MEM_QUOTA_LOCK:
            row = _state._MEM_TENANTS.get(tenant_id)
            if row is None:
                raise ValueError("tenant_not_found")
            quota = int(row.get("vector_quota", _state._DEFAULT_VECTOR_QUOTA))
            used = int(row.get("vectors_used", 0))
            if used + count <= quota:
                row["vectors_used"] = used + count
                return True, _usage_from_row(row)
            return False, _usage_from_row(row)

    # --- API keys ---------------------------------------------------------

    def create_api_key(self, key_id, tenant_id, key_hash, name):
        row = {
            "id": key_id,
            "tenant_id": tenant_id,
            "key_hash": key_hash,
            "name": name,
            "created_at": _state._now_iso(),
            "last_used_at": None,
            "revoked_at": None,
        }
        _state._MEM_API_KEYS.append(row)
        _state._MEM_API_KEYS_BY_HASH[key_hash] = row
        return dict(row)

    def list_api_keys(self, tenant_id):
        rows = [r for r in _state._MEM_API_KEYS if r["tenant_id"] == tenant_id]
        return [dict(r) for r in rows]

    def get_api_key(self, key_id, tenant_id):
        for r in _state._MEM_API_KEYS:
            if r["id"] == key_id and r["tenant_id"] == tenant_id:
                return dict(r)
        return None

    def revoke_api_key(self, key_id, tenant_id):
        for r in _state._MEM_API_KEYS:
            if r["id"] == key_id and r["tenant_id"] == tenant_id:
                if r["revoked_at"] is not None:
                    return False
                r["revoked_at"] = _state._now_iso()
                return True
        return False

    def get_api_key_by_hash(self, key_hash):
        row = _state._MEM_API_KEYS_BY_HASH.get(key_hash)
        return dict(row) if row else None

    def touch_api_key_last_used(self, key_id):
        for r in _state._MEM_API_KEYS:
            if r["id"] == key_id:
                r["last_used_at"] = _state._now_iso()
                return
        return

    # --- Import jobs ------------------------------------------------------

    def create_import_job(self, import_id, tenant_id, row):
        _state._MEM_IMPORT_SEQ += 1
        row["_seq"] = _state._MEM_IMPORT_SEQ
        _state._MEM_IMPORTS[(tenant_id, import_id)] = row
        return dict(row)

    def get_import_job(self, tenant_id, import_id):
        row = _state._MEM_IMPORTS.get((tenant_id, import_id))
        return dict(row) if row else None

    def get_import_job_by_id(self, import_id):
        for row in _state._MEM_IMPORTS.values():
            if row["import_id"] == import_id:
                return dict(row)
        return None

    def list_import_jobs(self, tenant_id, dataset):
        rows = [
            dict(r)
            for (tid, _), r in _state._MEM_IMPORTS.items()
            if tid == tenant_id and r["dataset"] == dataset
        ]
        rows.sort(key=lambda r: r.get("_seq", 0), reverse=True)
        return rows

    def update_import_job(self, import_id, patch):
        for row in _state._MEM_IMPORTS.values():
            if row["import_id"] == import_id:
                row.update(patch)
                return
        return

    # --- DP residency -----------------------------------------------------

    def register_dp_shard_warm(self, dp_id, shard_uri, warm_since, last_query_at):
        # The lock serialises concurrent writers in-process. The full
        # check-then-set must happen under one lock so a concurrent insert
        # cannot land between the existence check and the update.
        with _state._MEM_DP_RESIDENCY_LOCK:
            existing = _state._MEM_DP_RESIDENCY.get((dp_id, shard_uri))
            if existing is None:
                _state._MEM_DP_RESIDENCY[(dp_id, shard_uri)] = (warm_since, last_query_at)
            else:
                # Preserve the original warm_since; only refresh last_query_at.
                _state._MEM_DP_RESIDENCY[(dp_id, shard_uri)] = (existing[0], last_query_at)

    def unregister_dp_shard_warm(self, dp_id, shard_uri):
        with _state._MEM_DP_RESIDENCY_LOCK:
            _state._MEM_DP_RESIDENCY.pop((dp_id, shard_uri), None)

    def list_dp_residency_for_shard(self, shard_uri):
        with _state._MEM_DP_RESIDENCY_LOCK:
            return [
                (dp_id, warm_since, last_query_at)
                for (dp_id, uri), (warm_since, last_query_at) in _state._MEM_DP_RESIDENCY.items()
                if uri == shard_uri
            ]

    def list_dp_residency_for_dp(self, dp_id):
        with _state._MEM_DP_RESIDENCY_LOCK:
            return [
                (uri, warm_since, last_query_at)
                for (rid, uri), (warm_since, last_query_at) in _state._MEM_DP_RESIDENCY.items()
                if rid == dp_id
            ]


class _PostgresBackend:
    """Postgres implementation of the quota/tenant/key/import/residency
    primitives. Every method runs through `_state.pooled_conn()` so a
    monkeypatched pool is observed and the request-scoped connection seam is
    honoured (hot-row writes pass `standalone=True`, as before)."""

    # --- Tenants ----------------------------------------------------------

    def create_tenant(self, tenant_id, email, password_hash, vector_quota, query_quota):
        try:
            with _state.pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
INSERT INTO tenants(id, email, password_hash, vector_quota, daily_query_quota)
VALUES (%s, %s, %s, %s, %s)
RETURNING *
                    """,
                    (tenant_id, email, password_hash, vector_quota, query_quota),
                )
                row = cur.fetchone()
                return dict(row)
        except psycopg2.errors.UniqueViolation as exc:
            raise ValueError("duplicate_email") from exc

    def get_tenant_by_email(self, email):
        with _state.pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM tenants WHERE email=%s", (email,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_tenant_by_id(self, tenant_id):
        with _state.pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM tenants WHERE id=%s", (tenant_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_tenant_dp_pool(self, tenant_id):
        with _state.pooled_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT dp_pool FROM tenants WHERE id=%s", (tenant_id,))
            row = cur.fetchone()
            if row is None or row[0] is None:
                return _state._DEFAULT_DP_POOL
            return row[0]

    # --- Quotas -----------------------------------------------------------

    def reset_daily_if_needed(self, tenant_id, today):
        with _state.pooled_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
UPDATE tenants
SET queries_today = 0, queries_reset_at = CURRENT_DATE
WHERE id = %s AND queries_reset_at < CURRENT_DATE
                """,
                (tenant_id,),
            )

    def get_usage(self, tenant_id, today):
        with _state.pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
UPDATE tenants
SET queries_today = CASE WHEN queries_reset_at < CURRENT_DATE THEN 0 ELSE queries_today END,
    queries_reset_at = CASE WHEN queries_reset_at < CURRENT_DATE THEN CURRENT_DATE ELSE queries_reset_at END
WHERE id = %s
RETURNING vectors_used, vector_quota, queries_today, daily_query_quota, queries_reset_at
                """,
                (tenant_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError("tenant_not_found")
            return _usage_from_row(dict(row))

    def try_consume_query(self, tenant_id, today):
        with _state.pooled_conn(standalone=True) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            # First: apply the lazy reset so the conditional increment below sees
            # a fresh counter at the day boundary. Both run in one transaction.
            cur.execute(
                """
UPDATE tenants
SET queries_today = 0, queries_reset_at = CURRENT_DATE
WHERE id = %s AND queries_reset_at < CURRENT_DATE
                """,
                (tenant_id,),
            )
            cur.execute(
                """
UPDATE tenants
SET queries_today = queries_today + 1
WHERE id = %s AND queries_today < daily_query_quota
RETURNING vectors_used, vector_quota, queries_today, daily_query_quota, queries_reset_at
                """,
                (tenant_id,),
            )
            row = cur.fetchone()
            if row is not None:
                return True, _usage_from_row(dict(row))
            # Either the tenant is missing or the cap is hit — read back to tell.
            cur.execute(
                """
SELECT vectors_used, vector_quota, queries_today, daily_query_quota, queries_reset_at
FROM tenants WHERE id = %s
                """,
                (tenant_id,),
            )
            snap = cur.fetchone()
            if snap is None:
                raise ValueError("tenant_not_found")
            return False, _usage_from_row(dict(snap))

    def try_consume_vectors(self, tenant_id, count):
        with _state.pooled_conn(standalone=True) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
UPDATE tenants
SET vectors_used = vectors_used + %s
WHERE id = %s AND vectors_used + %s <= vector_quota
RETURNING vectors_used, vector_quota, queries_today, daily_query_quota, queries_reset_at
                """,
                (count, tenant_id, count),
            )
            row = cur.fetchone()
            if row is not None:
                return True, _usage_from_row(dict(row))
            cur.execute(
                """
SELECT vectors_used, vector_quota, queries_today, daily_query_quota, queries_reset_at
FROM tenants WHERE id = %s
                """,
                (tenant_id,),
            )
            snap = cur.fetchone()
            if snap is None:
                raise ValueError("tenant_not_found")
            return False, _usage_from_row(dict(snap))

    # --- API keys ---------------------------------------------------------

    def create_api_key(self, key_id, tenant_id, key_hash, name):
        with _state.pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
INSERT INTO api_keys(id, tenant_id, key_hash, name)
VALUES (%s, %s, %s, %s)
RETURNING *
                """,
                (key_id, tenant_id, key_hash, name),
            )
            return dict(cur.fetchone())

    def list_api_keys(self, tenant_id):
        with _state.pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM api_keys WHERE tenant_id=%s ORDER BY created_at ASC",
                (tenant_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_api_key(self, key_id, tenant_id):
        with _state.pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM api_keys WHERE id=%s AND tenant_id=%s",
                (key_id, tenant_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def revoke_api_key(self, key_id, tenant_id):
        with _state.pooled_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
UPDATE api_keys
SET revoked_at = now()
WHERE id=%s AND tenant_id=%s AND revoked_at IS NULL
                """,
                (key_id, tenant_id),
            )
            return cur.rowcount > 0

    def get_api_key_by_hash(self, key_hash):
        with _state.pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM api_keys WHERE key_hash=%s", (key_hash,))
            row = cur.fetchone()
            return dict(row) if row else None

    def touch_api_key_last_used(self, key_id):
        with _state.pooled_conn(standalone=True) as conn, conn.cursor() as cur:
            cur.execute("UPDATE api_keys SET last_used_at = now() WHERE id=%s", (key_id,))

    # --- Import jobs ------------------------------------------------------

    def create_import_job(self, import_id, tenant_id, row):
        with _state.pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
INSERT INTO import_jobs(import_id, tenant_id, dataset, format, error_mode,
                        max_bad_records, upload_uri)
VALUES (%s, %s, %s, %s, %s, %s, %s)
RETURNING *
                """,
                (
                    row["import_id"], row["tenant_id"], row["dataset"], row["format"],
                    row["error_mode"], row["max_bad_records"], row["upload_uri"],
                ),
            )
            return dict(cur.fetchone())

    def get_import_job(self, tenant_id, import_id):
        with _state.pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM import_jobs WHERE tenant_id=%s AND import_id=%s",
                (tenant_id, import_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_import_job_by_id(self, import_id):
        with _state.pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM import_jobs WHERE import_id=%s", (import_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def list_import_jobs(self, tenant_id, dataset):
        with _state.pooled_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
SELECT * FROM import_jobs
WHERE tenant_id=%s AND dataset=%s
ORDER BY created_at DESC
                """,
                (tenant_id, dataset),
            )
            return [dict(r) for r in cur.fetchall()]

    def update_import_job(self, import_id, patch):
        cols = ", ".join(f"{k}=%s" for k in patch)
        with _state.pooled_conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"UPDATE import_jobs SET {cols} WHERE import_id=%s",
                (*patch.values(), import_id),
            )

    # --- DP residency -----------------------------------------------------

    def register_dp_shard_warm(self, dp_id, shard_uri, warm_since, last_query_at):
        with _state.pooled_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
INSERT INTO dp_shard_residency(dp_id, shard_uri, warm_since, last_query_at)
VALUES (%s, %s, %s, %s)
ON CONFLICT (dp_id, shard_uri)
DO UPDATE SET last_query_at = EXCLUDED.last_query_at
                """,
                (dp_id, shard_uri, warm_since, last_query_at),
            )

    def unregister_dp_shard_warm(self, dp_id, shard_uri):
        with _state.pooled_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM dp_shard_residency WHERE dp_id=%s AND shard_uri=%s",
                (dp_id, shard_uri),
            )

    def list_dp_residency_for_shard(self, shard_uri):
        with _state.pooled_conn() as conn, conn.cursor() as cur:
            # The `dp_shard_residency_shard_uri_idx` (migration 007) makes this
            # an indexed scan even for a popular shard with many resident DPs.
            cur.execute(
                "SELECT dp_id, warm_since, last_query_at "
                "FROM dp_shard_residency WHERE shard_uri=%s",
                (shard_uri,),
            )
            return [(row[0], float(row[1]), float(row[2])) for row in cur.fetchall()]

    def list_dp_residency_for_dp(self, dp_id):
        with _state.pooled_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT shard_uri, warm_since, last_query_at "
                "FROM dp_shard_residency WHERE dp_id=%s",
                (dp_id,),
            )
            return [(row[0], float(row[1]), float(row[2])) for row in cur.fetchall()]


_MEMORY_BACKEND = _MemoryBackend()
_POSTGRES_BACKEND = _PostgresBackend()


def _backend():
    """Select the storage backend ONCE, reading `_state._MEMORY_MODE` at CALL
    time so a `monkeypatch.setattr(state, "_MEMORY_MODE", …)` /
    `importlib.reload(state)` flips it exactly as the inline branch did."""
    return _MEMORY_BACKEND if _state._MEMORY_MODE else _POSTGRES_BACKEND


# --- Tenants --------------------------------------------------------------


def create_tenant(tenant_id: str, email: str, password_hash: str) -> dict:
    """Insert a new tenant row.

    Returns the persisted row as a dict. Raises ValueError("duplicate_email")
    if the email already exists. Defaults (plan='free', quotas, etc.) are
    populated server-side to match the v1 API contract. Quota values honour
    the `RB_TEST_*` overrides (see `_quota_defaults`).
    """
    vector_quota, query_quota = _state._quota_defaults()
    return _backend().create_tenant(tenant_id, email, password_hash, vector_quota, query_quota)


# --- Quotas ---------------------------------------------------------------


def reset_daily_if_needed(tenant_id: str) -> None:
    """Lazy daily reset: if `queries_reset_at` is older than today, zero
    `queries_today` and bump the date to today.

    Standalone helper; `get_usage` / `try_consume_query` perform the same
    reset inline so the reset and the consume are atomic. Memory mode takes
    the quota lock; Postgres mode does it in one UPDATE.
    """
    return _backend().reset_daily_if_needed(tenant_id, _dt.date.today())


def get_usage(tenant_id: str) -> dict:
    """Return the current usage/quota snapshot for `tenant_id`.

    Performs a lazy daily reset first so a stale `queries_today` from a prior
    day is never reported. Returns the v1 `usage` shape; raises
    ValueError("tenant_not_found") if the tenant does not exist.
    """
    return _backend().get_usage(tenant_id, _dt.date.today())


def try_consume_query(tenant_id: str) -> Tuple[bool, dict]:
    """Atomically consume one unit of daily query quota.

    Lazy-resets first, then if `queries_today < daily_query_quota` increments
    `queries_today` and returns `(True, usage)`; otherwise `(False, usage)`
    without incrementing. The post-state `usage` is returned in both cases so
    the caller can surface `details.limit` / `details.reset_at` on a 429.

    Memory mode: the reset + check + increment run under one lock. Postgres
    mode: a single conditional UPDATE — the DB serialises concurrent callers,
    so two requests can never both slip past the cap.

    This is a **hot-row write** that MUST commit and release its row lock
    immediately. It uses `pooled_conn(standalone=True)` so it always runs in
    its own short transaction, never riding the request-scoped transaction —
    under the request scope the row lock would be held for the whole request
    (including the CP→DP proxy hop), serialising concurrent queries from one
    tenant.
    """
    return _backend().try_consume_query(tenant_id, _dt.date.today())


def try_consume_vectors(tenant_id: str, count: int) -> Tuple[bool, dict]:
    """Atomically consume `count` units of the lifetime vector quota.

    If `vectors_used + count <= vector_quota` adds `count` to `vectors_used`
    and returns `(True, usage)`; otherwise `(False, usage)` unchanged. A
    `count` of 0 always succeeds without touching the row.

    Same atomicity discipline as `try_consume_query`: memory mode under the
    quota lock, Postgres mode via one conditional UPDATE ... RETURNING.

    Like `try_consume_query` this is a hot-row write and uses
    `pooled_conn(standalone=True)` so the `UPDATE` commits and releases its
    row lock in its own short transaction instead of being held for the whole
    request-scoped transaction.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    return _backend().try_consume_vectors(tenant_id, count)


def get_tenant_by_email(email: str) -> Optional[dict]:
    """Return the tenant row for `email`, or None."""
    return _backend().get_tenant_by_email(email)


def get_tenant_by_id(tenant_id: str) -> Optional[dict]:
    """Return the tenant row for `tenant_id`, or None."""
    return _backend().get_tenant_by_id(tenant_id)


def get_tenant_dp_pool(tenant_id: str) -> str:
    """Return the Query-DP pool name a tenant's `/v1/query` traffic routes to.

    The CP reverse proxy calls this per request to resolve `tenant_id -> DP
    pool` (then `resolve_dp_base_url` maps the pool to a base URL). The value
    comes from the `tenants.dp_pool` column (migration 006).

    Defaults to `'shared'` for an unknown tenant or a NULL column — a missing
    routing target must never fail open, so an unrecognised tenant transparently
    uses the shared pool exactly as a freshly-created one does.

    Memory mode: the in-memory tenant rows created by `create_tenant` predate
    this column, so a row with no `dp_pool` key reads back as `'shared'`. A test
    (or a future provisioning path) can set `_MEM_TENANTS[tid]["dp_pool"]` to
    simulate a dedicated-pool tenant and this returns that value.
    """
    return _backend().get_tenant_dp_pool(tenant_id)


# --- API keys -------------------------------------------------------------


def create_api_key(key_id: str, tenant_id: str, key_hash: str, name: str) -> dict:
    """Insert a new api_keys row and return the persisted dict.

    `key_hash` is the SHA-256 hex digest of the raw `rb_live_...` token;
    the raw value is never stored. SHA-256 is deterministic so the hash
    can be looked up directly via the `api_keys_hash_idx` index (see
    `get_api_key_by_hash`) — bcrypt's per-row salt would have made that
    impossible. `last_used_at` and `revoked_at` start as NULL.
    """
    return _backend().create_api_key(key_id, tenant_id, key_hash, name)


def list_api_keys(tenant_id: str) -> List[dict]:
    """Return all api_keys rows for `tenant_id`, oldest-first.

    Revoked keys are included; the row carries `revoked_at`. Callers
    project to the v1 response shape themselves (no `key_hash` leaks).
    """
    return _backend().list_api_keys(tenant_id)


def get_api_key(key_id: str, tenant_id: str) -> Optional[dict]:
    """Return the api_keys row for `(key_id, tenant_id)`, or None.

    Filtering on `tenant_id` here keeps the cross-tenant 404 contract
    enforced in the data layer rather than relying on the caller.
    """
    return _backend().get_api_key(key_id, tenant_id)


def revoke_api_key(key_id: str, tenant_id: str) -> bool:
    """Mark the key revoked. Returns True iff a row was modified.

    Idempotent: revoking an already-revoked key leaves the original
    `revoked_at` in place and returns False, so callers can distinguish
    "no such key" from "already revoked" if they need to (the API maps
    both to 404 per the contract, but the primitive stays honest).
    """
    return _backend().revoke_api_key(key_id, tenant_id)


def get_api_key_by_hash(key_hash: str) -> Optional[dict]:
    """Return the api_keys row whose `key_hash` equals `key_hash`, or None.

    This is the auth-time resolution primitive: an inbound `rb_live_...`
    token is reduced to its SHA-256 hex digest by the caller, then looked
    up here in O(1) — a dict lookup in memory mode, an indexed
    `WHERE key_hash = %s` (backed by `api_keys_hash_idx`, also UNIQUE) in
    Postgres mode. The cost is independent of the total number of keys in
    the system. Revocation is NOT filtered here; the caller inspects
    `revoked_at` on the returned row so it can reject a revoked key.
    """
    return _backend().get_api_key_by_hash(key_hash)


def touch_api_key_last_used(key_id: str) -> None:
    """Set `last_used_at = now()` on a successful auth.

    Fire-and-forget: we do not return success/failure. If the row is
    gone (deleted tenant cascade) the update is a no-op.

    Uses `pooled_conn(standalone=True)` so this `UPDATE api_keys` commits and
    releases the api_keys row lock immediately, in its own short transaction.
    It runs during auth at the START of every authenticated request; left on
    the request-scoped connection its row lock would be held for the WHOLE
    request (including the CP->DP proxy hop), so concurrent requests sharing
    one API key would serialize on it — the same hot-row contention that
    `try_consume_query` addresses via `standalone=True`.
    """
    return _backend().touch_api_key_last_used(key_id)


# --- Import jobs (async bulk ingest) --------------------------------------


def create_import_job(
    import_id: str,
    tenant_id: str,
    dataset: str,
    fmt: str,
    error_mode: str,
    max_bad_records: Optional[int],
    upload_uri: str,
) -> dict:
    """Insert a new `import_jobs` row in status `awaiting_upload`.

    `upload_uri` is the deterministic object-storage key the client stages the
    file at via its presigned upload. Counters start at 0; the job advances
    through `validating` → `indexing` → `completed` (or `failed`) as the
    validator/builder process it.
    """
    now = _state._now_iso()
    row = {
        "import_id": import_id,
        "tenant_id": tenant_id,
        "dataset": dataset,
        "format": fmt,
        "status": "awaiting_upload",
        "error_mode": error_mode,
        "max_bad_records": max_bad_records,
        "upload_uri": upload_uri,
        "records_processed": 0,
        "records_accepted": 0,
        "records_rejected": 0,
        "rejected_uri": None,
        "error_message": None,
        "created_at": now,
        "completed_at": None,
    }
    return _backend().create_import_job(import_id, tenant_id, row)


def get_import_job(tenant_id: str, import_id: str) -> Optional[dict]:
    """Return the import job for `(tenant_id, import_id)`, or None.

    Tenant-scoped: a job belonging to another tenant returns None, which the
    HTTP layer maps to 404 (the v1 never-leak-existence rule).
    """
    return _backend().get_import_job(tenant_id, import_id)


def get_import_job_by_id(import_id: str) -> Optional[dict]:
    """Return an import job by `import_id` alone (worker-side helper).

    The validator worker only carries the `import_id` on the queue message;
    it has already been admission-checked at create time, so a tenant-scoped
    lookup is unnecessary here.
    """
    return _backend().get_import_job_by_id(import_id)


def list_import_jobs(tenant_id: str, dataset: str) -> List[dict]:
    """Return a dataset's import jobs, newest-first."""
    return _backend().list_import_jobs(tenant_id, dataset)


def update_import_job(import_id: str, **fields) -> None:
    """Patch an import job's mutable columns by `import_id`.

    Accepts any of: `status`, `records_processed`, `records_accepted`,
    `records_rejected`, `rejected_uri`, `error_message`, `completed_at`.
    Unknown keys are ignored. Used by the validator worker as it advances a
    job through its lifecycle.
    """
    patch = {k: v for k, v in fields.items() if k in _IMPORT_MUTABLE_FIELDS}
    if not patch:
        return
    return _backend().update_import_job(import_id, patch)


# --- DP residency registry (SSD-cache feature) ----------------------------
#
# The four functions below back the `dp_shard_residency` table (migration
# 007). The producer is the residency writer daemon
# (`services/_common/residency_writer.py`); `list_dp_residency_for_shard`
# is the intended read path for residency-aware CP routing (which prefers
# DPs that already have a shard cached), but is not yet wired into the CP.
# Each function selects the backend ONCE via `_backend()` so the unit suite
# (in-memory) and the integration suite (Postgres) share the same call sites.
#
# Why URI-keyed not catalog-id-keyed: the SSD tier itself is URI-keyed
# (`shard_tier.fetch(uri)` / `shard_tier.evict(uri)`) because a
# content-addressed URI is the stable identifier across builds. Keying
# the registry the same way avoids a join through `shard_catalog` every
# time the writer reconciles.


def register_dp_shard_warm(
    dp_id: str,
    shard_uri: str,
    warm_since: float,
    last_query_at: float,
) -> None:
    """UPSERT a residency row for `(dp_id, shard_uri)`.

    First write inserts; subsequent writes update only `last_query_at`.
    `warm_since` is set once on first admit and intentionally left alone on
    refresh — the operator reads it as "how long has this DP held the
    shard cached", which would be defeated by refreshing it on every cycle.
    The Postgres branch's `ON CONFLICT ... DO UPDATE SET last_query_at` is
    the canonical pattern; the memory branch mirrors it explicitly.

    Both `warm_since` and `last_query_at` are unix epoch seconds
    (`time.time()`), matching the `DOUBLE PRECISION` column type.
    """
    return _backend().register_dp_shard_warm(dp_id, shard_uri, warm_since, last_query_at)


def unregister_dp_shard_warm(dp_id: str, shard_uri: str) -> None:
    """DELETE the residency row for `(dp_id, shard_uri)`. Idempotent.

    Removing a row that does not exist is a no-op — the SQL `DELETE`'s
    natural semantics, mirrored in memory by `dict.pop(..., None)`. The
    writer relies on this: a diff cycle may compute the same delete twice
    if a residency entry was already missing on the previous cycle.
    """
    return _backend().unregister_dp_shard_warm(dp_id, shard_uri)


def list_dp_residency_for_shard(
    shard_uri: str,
) -> List[Tuple[str, float, float]]:
    """Return `[(dp_id, warm_since, last_query_at), ...]` for `shard_uri`.

    Intended read path for residency-aware routing: "given a shard the
    request needs, which DPs already hold it cached?". Not yet wired into
    the CP. The returned list is unordered — the caller picks a winner
    from the set (e.g. most recently queried, or any). Returns an empty
    list when no DP holds the shard, so callers can write
    `for dp_id, _, _ in list_dp_residency_for_shard(...)` without a None
    check.
    """
    return _backend().list_dp_residency_for_shard(shard_uri)


def list_dp_residency_for_dp(
    dp_id: str,
) -> List[Tuple[str, float, float]]:
    """Return `[(shard_uri, warm_since, last_query_at), ...]` for `dp_id`.

    Operator / observability primitive: "what is this DP currently
    holding?". Used by a future admin surface and operator dashboards.
    The PK `(dp_id, shard_uri)` is the index that makes this an indexed
    lookup.
    """
    return _backend().list_dp_residency_for_dp(dp_id)
