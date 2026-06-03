-- 008_shard_durable_lsn.sql
-- Delta tier (hot tier) watermark: per-shard durable LSN.
--
-- Adds `durable_lsn` to `shard_catalog`: the highest hot-tier LSN that has been
-- folded into this shard. It is the single number that partitions every vector
-- into exactly one tier (see docs/architecture/delta-tier.md, "The watermark"):
--
--     lsn <= durable_lsn   -> lives in COLD  (this shard)
--     lsn >  durable_lsn   -> lives in HOT   (the pgvector hot instance)
--
-- The LSN itself is generated in the *hot* store (a separate data-plane
-- pgvector instance, addressed via RB_HOT_DSN) so the per-write path never
-- touches the control-plane Postgres; `durable_lsn` is written here only at
-- flush time, never per write.
--
-- DEFAULT-OFF / backward compatible. The delta tier ships behind
-- `RB_DELTA_TIER` (default off). With the flag off nothing ever sets a non-zero
-- value: every existing and future shard simply carries `durable_lsn = 0`, the
-- query path takes the pure cold path, and existing shard logic is untouched.
--
-- Non-destructive and idempotent: a plain `ADD COLUMN ... IF NOT EXISTS` with a
-- `NOT NULL DEFAULT 0`, so every existing shard transparently gets `0` with no
-- backfill step and a re-run is a clean no-op (mirrors 006_tenants_dp_pool.sql).

ALTER TABLE shard_catalog
  ADD COLUMN IF NOT EXISTS durable_lsn BIGINT NOT NULL DEFAULT 0;
