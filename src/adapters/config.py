"""Canonical, read-once configuration for RosalindDB.

This is a LEAF module (stdlib only) that snapshots every environment variable
the codebase reads into a single frozen, typed `CONFIG` singleton AT IMPORT.
Reading env once here gives one authoritative place to see the full config
surface, with each variable parsed using the SAME coercion semantics the
original ad-hoc read sites use.

Status: the canonical flag parser `truthy()` has replaced the four former
`_truthy` copies (`services.query_api.v1_query` / `.dp_app`,
`services.index_builder.run`, `adapters.observability.otel`). The frozen
`CONFIG` snapshot is the single authoritative place that reads the full env
surface; migrating the remaining ad-hoc readers onto `CONFIG` is intentionally
incremental (see the note on read-fresh helpers below). The module stays
side-effect-free at import — it only reads env, and `validate()` is NOT called
here — so it changes no behavior; it mirrors the existing defaults.

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
- `DATABASE_URL` legitimately has two readings: the module-level memory-mode
  detector defaults to `memory://local`, while the DSN resolver falls back to
  the postgres DSN. Both are captured (`database_url` / `database_url_dsn`).
- Several numeric vars are floored/capped at their read site (e.g.
  `max(1, ...)`); those clamps are applied here too so the snapshot equals what
  the call site would compute.

Because env is read ONCE at import, a process that mutates `os.environ` after
import will NOT see the change via `CONFIG`. The existing read-fresh helpers
(`recall_enabled`, `auth_required`, `quotas_enabled`, ...) are deliberately
NOT replaced yet for exactly that reason — tests monkeypatch env and expect a
fresh read. Migrating those is a later, careful step.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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


def _str(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)


def _int_or_default(name: str, default: int) -> int:
    """`int(env)` when the value is a non-negative integer literal, else default.

    Mirrors the `.isdigit()` guarded reads used for pool sizes / quotas.
    """
    raw = os.getenv(name, "")
    return int(raw) if raw.isdigit() else default


def _csv(name: str) -> List[str]:
    """Comma-split, stripped, empties dropped. Mirror of `CORS_ALLOW_ORIGINS`."""
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


# Defaults that several vars reference (kept as named constants so the snapshot
# matches the call-site literals exactly).
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


@dataclass(frozen=True)
class Config:
    """Immutable snapshot of the full RosalindDB env surface, read once.

    Field names mirror the env var names (lowercased). Where a single env var
    has two distinct readings in the codebase, two fields capture both.
    """

    # --- required-at-boot / security-sensitive ---------------------------
    jwt_secret: Optional[str] = None
    require_auth: bool = False
    proxy_secret: Optional[str] = None
    enable_quotas: bool = False

    # --- auth / quota ----------------------------------------------------
    rate_limit_rps: float = 50.0
    rate_limit_burst: float = 100.0
    test_vector_quota: int = _DEFAULT_VECTOR_QUOTA
    test_query_quota: int = _DEFAULT_DAILY_QUERY_QUOTA

    # --- state / postgres / recall tier ----------------------------------
    # Module-level memory-mode detector default (state.py:36).
    database_url: str = "memory://local"
    # DSN-resolver fallback (state.py / catalog_listener.py).
    database_url_dsn: str = _DEFAULT_POSTGRES_DSN
    pg_pool_max: int = _DEFAULT_PG_POOL_MAX
    pg_pool_checkout_timeout_s: float = _DEFAULT_POOL_CHECKOUT_TIMEOUT_S
    skip_migrate: bool = False
    recall_dsn: Optional[str] = None
    recall: bool = False
    recall_pool_max: int = _DEFAULT_RECALL_POOL_MAX

    # --- query API hot path ----------------------------------------------
    cache_dir: str = _DEFAULT_CACHE_DIR
    query_nprobe: int = _DEFAULT_NPROBE
    delta_tier: bool = False
    shard_cache_bytes: int = 1 << 30
    shard_cache_size: int = 0
    faiss_mmap: bool = False
    catalog_freshness_s: float = 5.0
    shard_tier_bytes_set: bool = False  # presence test (v1_query / ephemeral)
    catalog_listen: bool = False
    download_coalesce_wait_s: float = 300.0
    recall_overlap_workers: int = 32
    result_topic: str = "RESULT_READY"
    top_k: int = 10
    query_mode: str = "auto"
    query_result_ttl: int = 3600
    prewarm_consumer: bool = False
    dp_residency_registry: bool = False
    admin_endpoints: bool = False

    # --- CP -> DP proxy --------------------------------------------------
    query_dp_url: str = "http://localhost:8090"
    routing_rendezvous: bool = False
    query_dp_read_timeout_s: Optional[str] = None
    query_dp_timeout_s: Optional[str] = None
    query_dp_connect_timeout_s: Optional[str] = None
    query_dp_connect_retries: int = 2
    query_dp_connect_backoff_s: float = 0.025
    query_dp_max_connections: int = 100
    query_dp_max_keepalive: int = 20

    # --- index builder ---------------------------------------------------
    vector_dim: int = 1536
    index_type: str = "ivfflat"
    indexes_prefix: str = "s3://rosalinddb/indexes"
    landing_prefix: str = _DEFAULT_LANDING_PREFIX
    tenant_prefix: bool = True
    index_metrics_port: int = 9101
    recall_idle_s: float = 60.0
    max_deltas: int = 8
    max_deltas_hard: int = 16
    shard_versioned_uris: bool = False
    ivf_training_floor: int = 64
    ivf_nlist: int = 4096
    shard_keep: int = 2
    prewarm_on_build: bool = False

    # --- validator worker ------------------------------------------------
    validator_metrics_port: int = 9100
    import_max_bytes: int = 5 * 1024 * 1024 * 1024

    # --- ephemeral runner ------------------------------------------------
    ephemeral_metrics_port: int = 9102

    # --- source registry -------------------------------------------------
    recall_max_rows: int = _DEFAULT_RECALL_MAX_ROWS
    cors_allow_origins: List[str] = field(default_factory=list)
    ingest_max_bytes: int = 10 * 1024 * 1024
    staging_prefix: Optional[str] = None
    import_upload_ttl_s: int = 3600

    # --- common helpers --------------------------------------------------
    dp_id: Optional[str] = None
    hostname: Optional[str] = None
    dp_residency_sync_s: Optional[str] = None

    # --- storage / S3 ----------------------------------------------------
    s3_endpoint_url: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_region: str = "us-east-1"
    shard_tier_bytes: int = 2 * 1024 * 1024 * 1024
    shard_tier_dir: Optional[str] = None
    shard_tier_coalesce_wait_s: float = 300.0
    shard_tier_tmp_max_age_s: float = 3600.0
    shard_tier_min_resident_s: float = 30.0

    # --- queue / reaper --------------------------------------------------
    queue_max_attempts: int = 5
    redis_url: Optional[str] = None
    queue_reclaim_timeout: float = 300.0
    dataset_stuck_timeout: float = 900.0
    reaper_interval: float = 30.0
    reaper_lock_ttl: Optional[float] = None

    # --- observability ---------------------------------------------------
    cloud_provider: str = "local"
    cloudwatch_namespace: str = "RosalindDB"
    service_role: str = "unknown"
    otel_sdk_disabled: bool = False
    otel_exporter_otlp_endpoint: str = "http://localhost:4318"
    otel_exporter_otlp_timeout: int = 3
    otel_service_name: Optional[str] = None
    otel_metric_export_interval: int = 10000


def _load() -> Config:
    """Read every env var ONCE and build the frozen snapshot.

    Each field is parsed with the SAME coercion the original read site uses.
    Clamps applied at the read site (e.g. `max(1, ...)`) are applied here so
    the snapshot equals what the call site computes.
    """
    # vector_dim honours VECTOR_DIM then DIMENSION then 1536 (nested getenv).
    vector_dim = int(os.getenv("VECTOR_DIM", os.getenv("DIMENSION", "1536")))

    # staging_prefix has no static default here (the call site derives it from
    # LANDING_PREFIX); we capture only an explicit override.
    staging_prefix = os.getenv("STAGING_PREFIX")

    # shard_tier_dir defaults to ${CACHE_DIR}/tier-managed at the call site;
    # capture only an explicit override (None => derive downstream).
    shard_tier_dir = _env_float("RB_SHARD_TIER_DIR", "") or None

    reaper_interval = _float("REAPER_INTERVAL", "30")

    return Config(
        # required-at-boot / security-sensitive
        jwt_secret=os.getenv("JWT_SECRET"),
        require_auth=truthy(os.getenv("RB_REQUIRE_AUTH")),
        proxy_secret=os.getenv("RB_PROXY_SECRET"),
        enable_quotas=truthy(os.getenv("RB_ENABLE_QUOTAS")),
        # auth / quota
        rate_limit_rps=_float("RB_RATE_LIMIT_RPS", "50"),
        rate_limit_burst=_float("RB_RATE_LIMIT_BURST", "100"),
        test_vector_quota=_int_or_default("RB_TEST_VECTOR_QUOTA", _DEFAULT_VECTOR_QUOTA),
        test_query_quota=_int_or_default("RB_TEST_QUERY_QUOTA", _DEFAULT_DAILY_QUERY_QUOTA),
        # state / postgres / recall tier
        database_url=os.getenv("DATABASE_URL", "memory://local"),
        database_url_dsn=os.getenv("DATABASE_URL", _DEFAULT_POSTGRES_DSN),
        pg_pool_max=_pool_max("RB_PG_POOL_MAX", _DEFAULT_PG_POOL_MAX),
        pg_pool_checkout_timeout_s=_positive_float(
            "RB_PG_POOL_CHECKOUT_TIMEOUT_S", _DEFAULT_POOL_CHECKOUT_TIMEOUT_S
        ),
        # RB_SKIP_MIGRATE uses the NARROWER tuple (no "on") — preserved.
        skip_migrate=os.getenv("RB_SKIP_MIGRATE", "").strip().lower()
        in ("1", "true", "yes"),
        recall_dsn=(os.getenv("RB_RECALL_DSN") or None),
        recall=truthy(os.getenv("RB_RECALL")),
        recall_pool_max=_pool_max("RB_RECALL_POOL_MAX", _DEFAULT_RECALL_POOL_MAX),
        # query API hot path
        cache_dir=os.getenv("CACHE_DIR", _DEFAULT_CACHE_DIR),
        query_nprobe=min(
            max(1, _int("RB_QUERY_NPROBE", str(_DEFAULT_NPROBE))), _MAX_NPROBE
        ),
        delta_tier=truthy(os.getenv("RB_DELTA_TIER")),
        shard_cache_bytes=max(1, _int("RB_SHARD_CACHE_BYTES", str(1 << 30))),
        shard_cache_size=max(0, _int("RB_SHARD_CACHE_SIZE", "0")),
        faiss_mmap=truthy(os.getenv("RB_FAISS_MMAP")),
        catalog_freshness_s=_nonneg_float("RB_CATALOG_FRESHNESS_S", 5.0),
        shard_tier_bytes_set=bool(os.getenv("RB_SHARD_TIER_BYTES")),
        catalog_listen=truthy(os.getenv("RB_CATALOG_LISTEN")),
        download_coalesce_wait_s=_float("RB_DOWNLOAD_COALESCE_WAIT_S", "300"),
        recall_overlap_workers=max(1, _int("RB_RECALL_OVERLAP_WORKERS", "32")),
        result_topic=os.getenv("RESULT_TOPIC", "RESULT_READY"),
        top_k=_int("TOP_K", "10"),
        query_mode=os.getenv("QUERY_MODE", "auto"),
        query_result_ttl=_int("RB_QUERY_RESULT_TTL", "3600"),
        prewarm_consumer=truthy(os.getenv("RB_PREWARM_CONSUMER")),
        dp_residency_registry=truthy(os.getenv("RB_DP_RESIDENCY_REGISTRY")),
        admin_endpoints=truthy(os.getenv("RB_ADMIN_ENDPOINTS")),
        # CP -> DP proxy (timeout strings parsed lazily at the call site)
        query_dp_url=os.getenv("QUERY_DP_URL", "http://localhost:8090"),
        routing_rendezvous=truthy(os.getenv("RB_ROUTING_RENDEZVOUS")),
        query_dp_read_timeout_s=os.getenv("RB_QUERY_DP_READ_TIMEOUT_S"),
        query_dp_timeout_s=os.getenv("RB_QUERY_DP_TIMEOUT_S"),
        query_dp_connect_timeout_s=os.getenv("RB_QUERY_DP_CONNECT_TIMEOUT_S"),
        query_dp_connect_retries=_max0_int("RB_QUERY_DP_CONNECT_RETRIES", 2),
        query_dp_connect_backoff_s=_max0_float("RB_QUERY_DP_CONNECT_BACKOFF_S", 0.025),
        query_dp_max_connections=_int("RB_QUERY_DP_MAX_CONNECTIONS", "100"),
        query_dp_max_keepalive=_int("RB_QUERY_DP_MAX_KEEPALIVE", "20"),
        # index builder
        vector_dim=vector_dim,
        index_type=os.getenv("INDEX_TYPE", "ivfflat"),
        indexes_prefix=os.getenv("INDEXES_PREFIX", "s3://rosalinddb/indexes"),
        landing_prefix=os.getenv("LANDING_PREFIX", _DEFAULT_LANDING_PREFIX),
        # TENANT_PREFIX uses bare `.lower() == "true"` — preserved.
        tenant_prefix=os.getenv("TENANT_PREFIX", "true").lower() == "true",
        index_metrics_port=_int("METRICS_PORT", "9101"),
        recall_idle_s=_positive_float("RB_RECALL_IDLE_S", 60.0),
        max_deltas=_positive_int("RB_MAX_DELTAS", 8),
        max_deltas_hard=_positive_int("RB_MAX_DELTAS_HARD", 16),
        shard_versioned_uris=truthy(os.getenv("RB_SHARD_VERSIONED_URIS")),
        ivf_training_floor=max(4, _int("IVF_TRAINING_FLOOR", "64")),
        ivf_nlist=_int("IVF_NLIST", "4096"),
        shard_keep=max(2, _int("SHARD_KEEP", "2")),
        prewarm_on_build=truthy(os.getenv("RB_PREWARM_ON_BUILD")),
        # validator worker
        validator_metrics_port=_int("METRICS_PORT", "9100"),
        import_max_bytes=_int("IMPORT_MAX_BYTES", str(5 * 1024 * 1024 * 1024)),
        # ephemeral runner
        ephemeral_metrics_port=_int("METRICS_PORT", "9102"),
        # source registry
        recall_max_rows=_positive_int_lstrip("RB_RECALL_MAX_ROWS", _DEFAULT_RECALL_MAX_ROWS),
        cors_allow_origins=_csv("CORS_ALLOW_ORIGINS"),
        ingest_max_bytes=_int("INGEST_MAX_BYTES", str(10 * 1024 * 1024)),
        staging_prefix=staging_prefix,
        import_upload_ttl_s=_int("IMPORT_UPLOAD_TTL_S", "3600"),
        # common helpers
        dp_id=os.getenv("RB_DP_ID"),
        hostname=os.getenv("HOSTNAME"),
        dp_residency_sync_s=os.getenv("RB_DP_RESIDENCY_SYNC_S"),
        # storage / S3
        s3_endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        s3_access_key=os.getenv("S3_ACCESS_KEY"),
        s3_secret_key=os.getenv("S3_SECRET_KEY"),
        s3_region=os.getenv("S3_REGION", "us-east-1"),
        shard_tier_bytes=max(
            1, int(_env_float("RB_SHARD_TIER_BYTES", str(2 * 1024 * 1024 * 1024)))
        ),
        shard_tier_dir=shard_tier_dir,
        shard_tier_coalesce_wait_s=float(
            _env_float("RB_SHARD_TIER_COALESCE_WAIT_S", "300")
        ),
        shard_tier_tmp_max_age_s=float(
            _env_float("RB_SHARD_TIER_TMP_MAX_AGE_S", "3600")
        ),
        shard_tier_min_resident_s=max(
            0.0, float(_env_float("RB_SHARD_TIER_MIN_RESIDENT_S", "30"))
        ),
        # queue / reaper
        queue_max_attempts=_int("QUEUE_MAX_ATTEMPTS", "5"),
        redis_url=os.getenv("REDIS_URL"),
        queue_reclaim_timeout=_float("QUEUE_RECLAIM_TIMEOUT", "300"),
        dataset_stuck_timeout=_float("DATASET_STUCK_TIMEOUT", "900"),
        reaper_interval=reaper_interval,
        reaper_lock_ttl=_float("REAPER_LOCK_TTL", str(reaper_interval + 30)),
        # observability
        cloud_provider=os.getenv("CLOUD_PROVIDER", "local"),
        cloudwatch_namespace=os.getenv("CLOUDWATCH_NAMESPACE", "RosalindDB"),
        service_role=os.getenv("SERVICE_ROLE", "unknown"),
        otel_sdk_disabled=truthy(os.getenv("OTEL_SDK_DISABLED")),
        otel_exporter_otlp_endpoint=os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"
        ).rstrip("/"),
        otel_exporter_otlp_timeout=_otel_timeout("OTEL_EXPORTER_OTLP_TIMEOUT", 3),
        otel_service_name=os.getenv("OTEL_SERVICE_NAME"),
        otel_metric_export_interval=_int("OTEL_METRIC_EXPORT_INTERVAL", "10000"),
    )


# --- coercion helpers that need try/except or clamps ---------------------


def _pool_max(name: str, default: int) -> int:
    """`int(env)` when a positive integer literal, else default. Mirror of
    `state._pool_max_size` / `_recall_pool_max` (`isdigit() and > 0`)."""
    raw = os.getenv(name, "")
    if raw.isdigit():
        v = int(raw)
        if v > 0:
            return v
    return default


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
    if raw.lstrip("-").isdigit():
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


class ConfigError(RuntimeError):
    """Raised by `validate()` when a required-at-boot variable is missing."""


def validate(config: Optional["Config"] = None) -> None:
    """Assert required-at-boot configuration is present.

    Intentionally NOT called at import (this module stays side-effect-free).
    Services may call it explicitly at startup. Today it enforces the single
    hard rule the config layer owns:

      - When auth is required (`RB_REQUIRE_AUTH` truthy), `JWT_SECRET` MUST be
        set. Otherwise tokens fall back to an ephemeral per-process secret and
        do not survive a restart — silently broken auth in production.

    Raises `ConfigError` with a clear message on the first failure.
    """
    cfg = config if config is not None else CONFIG
    if cfg.require_auth and not cfg.jwt_secret:
        raise ConfigError(
            "JWT_SECRET must be set when RB_REQUIRE_AUTH is enabled; "
            "without it tokens use an ephemeral per-process secret and are "
            "invalidated on every restart."
        )


# Read-once singleton. Importing this module performs NO validation and has no
# side effects beyond reading os.environ.
CONFIG = _load()


__all__ = ["CONFIG", "Config", "ConfigError", "truthy", "validate"]
