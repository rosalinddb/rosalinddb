-- 002_datasets_tenant_isolation.sql
-- Destructive: drops and recreates `dataset_catalog` and `shard_catalog` so
-- the primary key includes `tenant_id`. There is no production data yet (the
-- initial release runs against memory:// or fresh Postgres in dev), so this
-- is the accepted trade-off.
--
-- After this migration, every call site that previously looked up a dataset
-- or shard by `dataset_name` alone must pass `tenant_id` as well.
-- `adapters/state/state.py` and every service that writes/reads state
-- (validator_worker, index_builder, query_api, ephemeral_runner,
-- source_registry) must be updated atomically.

DROP TABLE IF EXISTS shard_catalog CASCADE;
DROP TABLE IF EXISTS dataset_catalog CASCADE;

CREATE TABLE dataset_catalog (
  tenant_id        TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  dataset_name     TEXT NOT NULL,
  dimension        INT  NOT NULL,
  row_count        BIGINT NOT NULL DEFAULT 0,
  status           TEXT NOT NULL DEFAULT 'empty',
  error_message    TEXT,
  source_uris      TEXT[] NOT NULL DEFAULT '{}',
  landing_format   TEXT NOT NULL DEFAULT 'parquet',
  object_store_uri TEXT,
  last_indexed_at  TIMESTAMPTZ,
  deleted_at       TIMESTAMPTZ,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, dataset_name)
);

CREATE INDEX dataset_catalog_tenant_idx ON dataset_catalog(tenant_id);
CREATE INDEX dataset_catalog_status_idx ON dataset_catalog(tenant_id, status);

CREATE TABLE shard_catalog (
  id            BIGSERIAL PRIMARY KEY,
  tenant_id     TEXT NOT NULL,
  dataset_name  TEXT NOT NULL,
  shard_uri     TEXT NOT NULL,
  checksum      TEXT NOT NULL,
  vector_count  BIGINT NOT NULL,
  index_type    TEXT NOT NULL,
  sealed        BOOLEAN NOT NULL DEFAULT TRUE,
  supersedes    BIGINT[] NOT NULL DEFAULT '{}',
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  FOREIGN KEY (tenant_id, dataset_name)
    REFERENCES dataset_catalog(tenant_id, dataset_name)
    ON DELETE CASCADE
);

CREATE INDEX shard_catalog_lookup_idx
  ON shard_catalog(tenant_id, dataset_name, created_at DESC);
