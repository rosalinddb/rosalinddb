-- 003_shard_incremental_indexing.sql
-- Incremental indexing support.
--
-- Adds `indexed_landing_uris` to `shard_catalog`: the manifest of landing
-- parquet part URIs that have already been folded into this shard. The index
-- builder uses the *newest* shard's manifest as the authoritative record of
-- "what has already been indexed" so a subsequent ingest only `index.add()`s
-- the new parts instead of rebuilding the whole shard from all landing data.
--
-- Destructive: drops and recreates `shard_catalog` to add the column. Applied
-- against memory:// or fresh dev Postgres, so there is no data to preserve.
-- `dataset_catalog` is left untouched; only `shard_catalog` is recreated.

DROP TABLE IF EXISTS shard_catalog CASCADE;

CREATE TABLE shard_catalog (
  id                   BIGSERIAL PRIMARY KEY,
  tenant_id            TEXT NOT NULL,
  dataset_name         TEXT NOT NULL,
  shard_uri            TEXT NOT NULL,
  checksum             TEXT NOT NULL,
  vector_count         BIGINT NOT NULL,
  index_type           TEXT NOT NULL,
  build_type           TEXT NOT NULL DEFAULT 'full',
  indexed_landing_uris TEXT[] NOT NULL DEFAULT '{}',
  sealed               BOOLEAN NOT NULL DEFAULT TRUE,
  supersedes           BIGINT[] NOT NULL DEFAULT '{}',
  created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
  FOREIGN KEY (tenant_id, dataset_name)
    REFERENCES dataset_catalog(tenant_id, dataset_name)
    ON DELETE CASCADE
);

CREATE INDEX shard_catalog_lookup_idx
  ON shard_catalog(tenant_id, dataset_name, created_at DESC);
