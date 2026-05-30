-- 001_tenants_and_keys.sql
-- Additive only: introduces tenants and api_keys.
--
-- The existing dataset_catalog and shard_catalog are left alone;
-- 002_datasets_tenant_isolation.sql evolves them to include tenant_id.
-- Splitting migrations this way keeps each migration's blast radius to one
-- concern and prevents validator/index_builder/query_api from breaking on a
-- half-applied schema.

CREATE TABLE IF NOT EXISTS tenants (
  id                  TEXT PRIMARY KEY,
  email               TEXT NOT NULL UNIQUE,
  password_hash       TEXT NOT NULL,
  plan                TEXT NOT NULL DEFAULT 'free',
  vector_quota        BIGINT NOT NULL DEFAULT 100000,
  daily_query_quota   BIGINT NOT NULL DEFAULT 10000,
  vectors_used        BIGINT NOT NULL DEFAULT 0,
  queries_today       BIGINT NOT NULL DEFAULT 0,
  queries_reset_at    DATE   NOT NULL DEFAULT CURRENT_DATE,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS tenants_email_idx ON tenants(email);

CREATE TABLE IF NOT EXISTS api_keys (
  id            TEXT PRIMARY KEY,
  tenant_id     TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  key_hash      TEXT NOT NULL UNIQUE,
  name          TEXT NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_used_at  TIMESTAMPTZ,
  revoked_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS api_keys_tenant_idx ON api_keys(tenant_id);
CREATE INDEX IF NOT EXISTS api_keys_hash_idx   ON api_keys(key_hash);
