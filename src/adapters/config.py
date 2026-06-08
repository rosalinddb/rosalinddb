"""Canonical, READ-FRESH configuration for RosalindDB.

This is a LEAF module (stdlib only) that is the single source of truth for every
environment variable the codebase reads. Each variable is exposed as ONE
accessor function that reads `os.getenv` FRESH on every call, parsed with the
SAME coercion semantics the original ad-hoc read sites use, with the SAME
default. There is one authoritative place to see the full config surface.

Read-fresh contract
-------------------
The codebase reads config FRESH. Readers are either in-function `os.getenv(...)`
calls, OR module-level constants (`X = os.getenv(...)`) that tests re-read via
`importlib.reload(module)`. The unit suite flips ~50 vars at runtime with
`monkeypatch.setenv` / `monkeypatch.delenv`.

Therefore EVERY accessor here reads `os.getenv` on EVERY call and NEVER caches a
value at import. A module-level consumer keeps the shape `X = config.x()` so a
test reload re-invokes the fresh accessor; an in-function consumer calls
`config.x()` at use time. Importing this module performs NO env reads and has NO
side effects: `validate()` is NOT called here.

Important semantics preserved
-----------------------------
- `truthy(value)` is the canonical flag parser: `"1"/"true"/"yes"/"on"`
  (case-insensitive); empty / None -> False. This matches the four `_truthy`
  copies and the inline truthy checks across the tree.
- A handful of flags use NARROWER or DIFFERENT parsing than `truthy`, and that
  is preserved verbatim:
    * `RB_SKIP_MIGRATE` uses `("1","true","yes")` (NO "on").
    * `TENANT_PREFIX` uses bare `.lower() == "true"` (only the literal "true").
- `_env_float(name, default)` mirrors `shard_tier._env_float`: it treats BOTH a
  missing var AND an empty-string value (Compose `${X:-}` passthrough) as "use
  the default".
- `DATABASE_URL` legitimately has two readings: the memory-mode detector
  defaults to `memory://local` (`database_url()`), while the DSN resolver falls
  back to the postgres DSN (`database_url_dsn()`).
- `METRICS_PORT` is one env key backing three per-service accessors with
  different defaults (validator 9100 / index 9101 / ephemeral 9102). If the env
  key is set, all three collapse to that value — preserved verbatim.
- Several numeric vars are floored/capped at their read site (e.g.
  `max(1, ...)`); those clamps are applied inside the accessor so the value
  equals what the call site would compute.
"""

from __future__ import annotations

import os
from typing import List, Optional


# --- canonical parsing helpers -------------------------------------------

_TRUTHY_VALUES = ("1", "true", "yes", "on")


def truthy(value: Optional[str]) -> bool:
    """Canonical env-flag parser.

    Truthy values: `1`, `true`, `yes`, `on` (case-insensitive). Empty string
    and `None` are False. Identical semantics to the `_truthy` copies in
    `services.query_api.v1_query`, `services.index_builder.run`,
    `services.query_api.dp_app`, and `adapters.observability.otel`, and to the
    inline checks in `auth.jwt_utils` / `auth.quota` / `state` / `residency_writer`.
    """
    return (value or "").strip().lower() in _TRUTHY_VALUES


def _env_float(name: str, default: str) -> str:
    """Read `name`, treating empty string as unset (= use `default`).

    Mirror of `adapters.storage.shard_tier._env_float`: Compose forwards an
    unset shell var as an empty string, not a missing var, so both "missing"
    and "empty" must collapse to "use default". Returns a STRING (the caller
    coerces) exactly like the original.
    """
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _pool_max(name: str, default: int) -> int:
    """`int(env)` when a positive integer literal, else default. Mirror of
    `state._pool_max_size` / `_recall_pool_max` (`isdigit() and > 0`)."""
    raw = os.getenv(name, "")
    if raw.isdigit():
        v = int(raw)
        if v > 0:
            return v
    return default


def _int_or_default(name: str, default: int) -> int:
    """`int(env)` when the value is a non-negative integer literal, else default.

    Mirrors the `.isdigit()` guarded reads used for the test quotas.
    """
    raw = os.getenv(name, "")
    return int(raw) if raw.isdigit() else default


def _positive_float(name: str, default: float) -> float:
    """`float(env)` when > 0, else default (try/except). Mirror of
    `_pool_checkout_timeout_s` / `RB_RECALL_IDLE_S`."""
    raw = os.getenv(name)
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    return default


def _nonneg_float(name: str, default: float) -> float:
    """`max(0.0, float(env))` with try/except -> default. Mirror of
    `RB_CATALOG_FRESHNESS_S`."""
    raw = os.getenv(name, str(default))
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return default


def _positive_int(name: str, default: int) -> int:
    """`int(env)` when > 0, else default (try/except). Mirror of
    `_max_deltas` / `_max_deltas_hard`."""
    raw = os.getenv(name)
    if raw is not None:
        try:
            v = int(raw)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    return default


def _positive_int_lstrip(name: str, default: int) -> int:
    """`int(env)` when `lstrip("-").isdigit()` and > 0, else default. Mirror of
    `source_registry._recall_max_rows`."""
    raw = os.getenv(name, "")
    if raw.strip().lstrip("-").isdigit():
        v = int(raw)
        if v > 0:
            return v
    return default


def _max0_int(name: str, default: int) -> int:
    """`max(0, int(env))` with try/except -> default. Mirror of
    `RB_QUERY_DP_CONNECT_RETRIES`."""
    raw = os.getenv(name, str(default))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return default


def _max0_float(name: str, default: float) -> float:
    """`max(0.0, float(env))` with try/except -> default. Mirror of
    `RB_QUERY_DP_CONNECT_BACKOFF_S`."""
    raw = os.getenv(name, str(default))
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return default


def _otel_timeout(name: str, default: int) -> int:
    """`int(float(env))` with try/except -> default. Mirror of
    `OTEL_EXPORTER_OTLP_TIMEOUT`."""
    raw = os.getenv(name)
    if raw:
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            pass
    return default


# Defaults that several vars reference (kept as named constants so the accessors
# match the call-site literals exactly).
_DEFAULT_PG_POOL_MAX = 10
_DEFAULT_POOL_CHECKOUT_TIMEOUT_S = 2.5
_DEFAULT_RECALL_POOL_MAX = 10
_DEFAULT_VECTOR_QUOTA = 100000
_DEFAULT_DAILY_QUERY_QUOTA = 10000
_DEFAULT_RECALL_MAX_ROWS = 2000
_DEFAULT_NPROBE = 64
_MAX_NPROBE = 1024
_DEFAULT_CACHE_DIR = "/var/cache/shards"
_DEFAULT_LANDING_PREFIX = "s3://rosalinddb/landing"
_DEFAULT_POSTGRES_DSN = "postgresql://postgres:postgres@localhost:5432/vectors"


# === ACCESSORS (read-fresh, one per env var) =============================
#
# Each function reads os.getenv FRESH on every call with the exact default and
# parsing the original read site uses. Ordering mirrors the former Config
# dataclass for ease of cross-reference.


# --- required-at-boot / security-sensitive -------------------------------

def jwt_secret() -> Optional[str]:
    return os.getenv("JWT_SECRET")


def require_auth() -> bool:
    return truthy(os.getenv("RB_REQUIRE_AUTH"))


def proxy_secret() -> Optional[str]:
    return os.getenv("RB_PROXY_SECRET")


def enable_quotas() -> bool:
    return truthy(os.getenv("RB_ENABLE_QUOTAS"))


# --- auth / quota --------------------------------------------------------

def rate_limit_rps() -> float:
    return _float("RB_RATE_LIMIT_RPS", "50")


def rate_limit_burst() -> float:
    return _float("RB_RATE_LIMIT_BURST", "100")


def test_vector_quota() -> int:
    return _int_or_default("RB_TEST_VECTOR_QUOTA", _DEFAULT_VECTOR_QUOTA)


def test_query_quota() -> int:
    return _int_or_default("RB_TEST_QUERY_QUOTA", _DEFAULT_DAILY_QUERY_QUOTA)


# --- state / postgres / recall tier --------------------------------------

def database_url() -> str:
    """Memory-mode detector reading (state.py): default `memory://local`."""
    return os.getenv("DATABASE_URL", "memory://local")


def database_url_dsn() -> str:
    """DSN-resolver reading (pooling.py / catalog_listener.py): postgres DSN."""
    return os.getenv("DATABASE_URL", _DEFAULT_POSTGRES_DSN)


def pg_pool_max() -> int:
    return _pool_max("RB_PG_POOL_MAX", _DEFAULT_PG_POOL_MAX)


def pg_pool_checkout_timeout_s() -> float:
    return _positive_float(
        "RB_PG_POOL_CHECKOUT_TIMEOUT_S", _DEFAULT_POOL_CHECKOUT_TIMEOUT_S
    )


def skip_migrate() -> bool:
    """`RB_SKIP_MIGRATE` uses the NARROWER tuple (no "on") and NO `.strip()` —
    both preserved verbatim from the original migrations.py reads."""
    return os.getenv("RB_SKIP_MIGRATE", "").lower() in ("1", "true", "yes")


def recall_dsn() -> Optional[str]:
    """`RB_RECALL_DSN` with a blank/whitespace value treated as unset (-> None).

    Mirrors `adapters.recall._recall_dsn`: an empty Compose default must not
    accidentally enable the recall tier, so a whitespace-only value collapses to
    None exactly like a missing var.
    """
    raw = os.getenv("RB_RECALL_DSN")
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def recall() -> bool:
    return truthy(os.getenv("RB_RECALL"))


def recall_backend() -> str:
    """Which recall-tier backend to use (`RB_RECALL_BACKEND`, default `auto`).

    Read fresh on every call (no module reload needed for a test to flip it).
    The seam that selects the recall storage engine:

      - `auto` (default) — resolve at call time: the EMBEDDED in-process numpy
        memtable when recall is on AND no `RB_RECALL_DSN` is configured (the
        all-in-one / single-process eval default), else the pgvector path.
      - `memory` — force the embedded memtable regardless of DSN (the no-docker
        path; `RB_RECALL_DSN` is ignored / left unset).
      - `pgvector` — force the pgvector path; it still requires a DSN to enable.

    The resolution lives in `adapters.recall._use_memory_backend()`; this is
    only the single read-fresh config surface for it.
    """
    return os.getenv("RB_RECALL_BACKEND", "auto")


def recall_pool_max() -> int:
    return _pool_max("RB_RECALL_POOL_MAX", _DEFAULT_RECALL_POOL_MAX)


# --- query API hot path --------------------------------------------------

def cache_dir() -> str:
    return os.getenv("CACHE_DIR", _DEFAULT_CACHE_DIR)


def query_nprobe() -> int:
    return min(max(1, _int("RB_QUERY_NPROBE", str(_DEFAULT_NPROBE))), _MAX_NPROBE)


def delta_tier() -> bool:
    return truthy(os.getenv("RB_DELTA_TIER"))


def shard_cache_bytes() -> int:
    return max(1, _int("RB_SHARD_CACHE_BYTES", str(1 << 30)))


def shard_cache_size() -> int:
    return max(0, _int("RB_SHARD_CACHE_SIZE", "0"))


def faiss_mmap() -> bool:
    return truthy(os.getenv("RB_FAISS_MMAP"))


def catalog_freshness_s() -> float:
    return _nonneg_float("RB_CATALOG_FRESHNESS_S", 5.0)


def shard_tier_bytes_set() -> bool:
    """Presence/truthy test of RB_SHARD_TIER_BYTES (shard_fetch / ephemeral /
    v1_query). Distinct read semantics from the int budget `shard_tier_bytes()`.
    """
    return bool(os.getenv("RB_SHARD_TIER_BYTES"))


def catalog_listen() -> bool:
    return truthy(os.getenv("RB_CATALOG_LISTEN"))


def download_coalesce_wait_s() -> float:
    return _float("RB_DOWNLOAD_COALESCE_WAIT_S", "300")


def recall_overlap_workers() -> int:
    return max(1, _int("RB_RECALL_OVERLAP_WORKERS", "32"))


def result_topic() -> str:
    return os.getenv("RESULT_TOPIC", "RESULT_READY")


def top_k() -> int:
    return _int("TOP_K", "10")


def query_mode() -> str:
    return os.getenv("QUERY_MODE", "auto")


def query_result_ttl() -> int:
    return _int("RB_QUERY_RESULT_TTL", "3600")


def prewarm_consumer() -> bool:
    return truthy(os.getenv("RB_PREWARM_CONSUMER"))


def dp_residency_registry() -> bool:
    return truthy(os.getenv("RB_DP_RESIDENCY_REGISTRY"))


def admin_endpoints() -> bool:
    return truthy(os.getenv("RB_ADMIN_ENDPOINTS"))


# --- CP -> DP proxy ------------------------------------------------------

def query_dp_url() -> str:
    return os.getenv("QUERY_DP_URL", "http://localhost:8090")


def query_dp_url_for(pool: str) -> Optional[str]:
    """Per-tenant override `QUERY_DP_URL_<POOL>` (dynamic key, no default).

    The key space is unbounded (runtime-computed from the sanitized pool name),
    so this cannot be a static accessor — it reads the computed key fresh.
    """
    return os.getenv("QUERY_DP_URL_" + pool)


def routing_rendezvous() -> bool:
    return truthy(os.getenv("RB_ROUTING_RENDEZVOUS"))


def query_dp_read_timeout_s() -> Optional[str]:
    return os.getenv("RB_QUERY_DP_READ_TIMEOUT_S")


def query_dp_timeout_s() -> Optional[str]:
    return os.getenv("RB_QUERY_DP_TIMEOUT_S")


def query_dp_connect_timeout_s() -> Optional[str]:
    return os.getenv("RB_QUERY_DP_CONNECT_TIMEOUT_S")


def query_dp_connect_retries() -> int:
    return _max0_int("RB_QUERY_DP_CONNECT_RETRIES", 2)


def query_dp_connect_backoff_s() -> float:
    return _max0_float("RB_QUERY_DP_CONNECT_BACKOFF_S", 0.025)


def query_dp_max_connections() -> int:
    return _int("RB_QUERY_DP_MAX_CONNECTIONS", "100")


def query_dp_max_keepalive() -> int:
    return _int("RB_QUERY_DP_MAX_KEEPALIVE", "20")


# --- index builder -------------------------------------------------------

def vector_dim() -> int:
    """VECTOR_DIM then DIMENSION then 1536 (nested getenv)."""
    return int(os.getenv("VECTOR_DIM", os.getenv("DIMENSION", "1536")))


def vector_dim_set() -> bool:
    """True when EITHER VECTOR_DIM or DIMENSION is present in the env.

    Presence test distinct from the value accessor `vector_dim()`. The
    validator's `validate_record` uses this to decide between a fresh
    `vector_dim()` read and its import-time `DIMENSION` snapshot — when neither
    env var is set it keeps the snapshot, matching the original
    `os.getenv("VECTOR_DIM") is None and os.getenv("DIMENSION") is None` guard.
    """
    return os.getenv("VECTOR_DIM") is not None or os.getenv("DIMENSION") is not None


def index_type() -> str:
    return os.getenv("INDEX_TYPE", "ivfflat")


def indexes_prefix() -> str:
    return os.getenv("INDEXES_PREFIX", "s3://rosalinddb/indexes")


def landing_prefix() -> str:
    return os.getenv("LANDING_PREFIX", _DEFAULT_LANDING_PREFIX)


def tenant_prefix() -> bool:
    """`TENANT_PREFIX` uses bare `.lower() == "true"` (literal "true") — preserved."""
    return os.getenv("TENANT_PREFIX", "true").lower() == "true"


def index_metrics_port() -> int:
    """METRICS_PORT (index_builder default 9101). See module docstring re: the
    shared env key collision across the three services."""
    return _int("METRICS_PORT", "9101")


def recall_idle_s() -> float:
    return _positive_float("RB_RECALL_IDLE_S", 60.0)


def max_deltas() -> int:
    return _positive_int("RB_MAX_DELTAS", 8)


def max_deltas_hard() -> int:
    return _positive_int("RB_MAX_DELTAS_HARD", 16)


def shard_versioned_uris() -> bool:
    return truthy(os.getenv("RB_SHARD_VERSIONED_URIS"))


def ivf_training_floor() -> int:
    return max(4, _int("IVF_TRAINING_FLOOR", "64"))


def ivf_nlist() -> int:
    return _int("IVF_NLIST", "4096")


def shard_keep() -> int:
    return max(2, _int("SHARD_KEEP", "2"))


def prewarm_on_build() -> bool:
    return truthy(os.getenv("RB_PREWARM_ON_BUILD"))


# --- validator worker ----------------------------------------------------

def validator_metrics_port() -> int:
    """METRICS_PORT (validator default 9100)."""
    return _int("METRICS_PORT", "9100")


def import_max_bytes() -> int:
    return _int("IMPORT_MAX_BYTES", str(5 * 1024 * 1024 * 1024))


# --- ephemeral runner ----------------------------------------------------

def ephemeral_metrics_port() -> int:
    """METRICS_PORT (ephemeral default 9102)."""
    return _int("METRICS_PORT", "9102")


# --- source registry -----------------------------------------------------

def recall_max_rows() -> int:
    return _positive_int_lstrip("RB_RECALL_MAX_ROWS", _DEFAULT_RECALL_MAX_ROWS)


def cors_allow_origins() -> List[str]:
    """Comma-split, stripped, empties dropped. Mirror of `CORS_ALLOW_ORIGINS`."""
    raw = os.getenv("CORS_ALLOW_ORIGINS", "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def ingest_max_bytes() -> int:
    return _int("INGEST_MAX_BYTES", str(10 * 1024 * 1024))


def staging_prefix() -> Optional[str]:
    """Explicit STAGING_PREFIX override only (the call site derives a default
    from LANDING_PREFIX; None => derive downstream)."""
    return os.getenv("STAGING_PREFIX")


def import_upload_ttl_s() -> int:
    return _int("IMPORT_UPLOAD_TTL_S", "3600")


# --- common helpers ------------------------------------------------------

def dp_id() -> Optional[str]:
    return os.getenv("RB_DP_ID")


def hostname() -> Optional[str]:
    return os.getenv("HOSTNAME")


def dp_residency_sync_s() -> Optional[str]:
    return os.getenv("RB_DP_RESIDENCY_SYNC_S")


# --- storage / S3 --------------------------------------------------------

def s3_endpoint_url() -> Optional[str]:
    return os.getenv("S3_ENDPOINT_URL")


def s3_access_key() -> Optional[str]:
    return os.getenv("S3_ACCESS_KEY")


def s3_secret_key() -> Optional[str]:
    return os.getenv("S3_SECRET_KEY")


def s3_region() -> str:
    return os.getenv("S3_REGION", "us-east-1")


def shard_tier_bytes() -> int:
    """Int budget reading of RB_SHARD_TIER_BYTES (shard_tier.py), default 2 GiB.
    Distinct from the presence test `shard_tier_bytes_set()`."""
    return max(
        1, int(_env_float("RB_SHARD_TIER_BYTES", str(2 * 1024 * 1024 * 1024)))
    )


def shard_tier_dir() -> Optional[str]:
    """Explicit RB_SHARD_TIER_DIR override only (call site defaults to
    `${CACHE_DIR}/tier-managed`; None => derive downstream)."""
    return _env_float("RB_SHARD_TIER_DIR", "") or None


def shard_tier_coalesce_wait_s() -> float:
    return float(_env_float("RB_SHARD_TIER_COALESCE_WAIT_S", "300"))


def shard_tier_tmp_max_age_s() -> float:
    return float(_env_float("RB_SHARD_TIER_TMP_MAX_AGE_S", "3600"))


def shard_tier_min_resident_s() -> float:
    return max(0.0, float(_env_float("RB_SHARD_TIER_MIN_RESIDENT_S", "30")))


# --- queue / reaper ------------------------------------------------------

def queue_max_attempts() -> int:
    return _int("QUEUE_MAX_ATTEMPTS", "5")


def redis_url() -> Optional[str]:
    return os.getenv("REDIS_URL")


def queue_reclaim_timeout() -> float:
    return _float("QUEUE_RECLAIM_TIMEOUT", "300")


def dataset_stuck_timeout() -> float:
    return _float("DATASET_STUCK_TIMEOUT", "900")


def reaper_interval() -> float:
    return _float("REAPER_INTERVAL", "30")


def reaper_lock_ttl() -> float:
    """REAPER_LOCK_TTL defaults to `REAPER_INTERVAL + 30` (read fresh)."""
    return _float("REAPER_LOCK_TTL", str(reaper_interval() + 30))


# --- observability -------------------------------------------------------

def cloud_provider() -> str:
    return os.getenv("CLOUD_PROVIDER", "local")


def cloudwatch_namespace() -> str:
    return os.getenv("CLOUDWATCH_NAMESPACE", "RosalindDB")


def service_role() -> str:
    return os.getenv("SERVICE_ROLE", "unknown")


def otel_sdk_disabled() -> bool:
    return truthy(os.getenv("OTEL_SDK_DISABLED"))


def otel_exporter_otlp_endpoint() -> str:
    return os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318").rstrip("/")


def otel_exporter_otlp_timeout() -> int:
    return _otel_timeout("OTEL_EXPORTER_OTLP_TIMEOUT", 3)


def otel_service_name() -> Optional[str]:
    return os.getenv("OTEL_SERVICE_NAME")


def otel_metric_export_interval() -> int:
    return _int("OTEL_METRIC_EXPORT_INTERVAL", "10000")


# === validation ==========================================================

class ConfigError(RuntimeError):
    """Raised by `validate()` when a required-at-boot variable is missing."""


def validate() -> None:
    """Assert required-at-boot configuration is present (reading env FRESH).

    Intentionally NOT called at import (this module stays side-effect-free).
    Services call it explicitly at startup. Today it enforces the single hard
    rule the config layer owns:

      - When auth is required (`RB_REQUIRE_AUTH` truthy), `JWT_SECRET` MUST be
        set. Otherwise tokens fall back to an ephemeral per-process secret and
        do not survive a restart — silently broken auth in production.

    Raises `ConfigError` with a clear message on the first failure.
    """
    if require_auth() and not jwt_secret():
        raise ConfigError(
            "JWT_SECRET must be set when RB_REQUIRE_AUTH is enabled; "
            "without it tokens use an ephemeral per-process secret and are "
            "invalidated on every restart."
        )


__all__ = ["ConfigError", "truthy", "validate"]
