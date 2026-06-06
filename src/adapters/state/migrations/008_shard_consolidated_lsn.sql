-- 008_shard_consolidated_lsn.sql
-- Recall-tier watermark: per-shard consolidated LSN.
--
-- Adds `consolidated_lsn` to `shard_catalog`: the highest recall-tier LSN that
-- has been folded into this shard. It is the single number that partitions every
-- vector into exactly one tier (see docs/architecture/recall-consolidate.md,
-- "The watermark"):
--
--     lsn <= consolidated_lsn   -> lives in CONSOLIDATED  (this shard)
--     lsn >  consolidated_lsn   -> lives in RECALL         (the pgvector recall instance)
--
-- The LSN itself is generated in the *recall* store (a separate data-plane
-- pgvector instance, addressed via RB_RECALL_DSN) so the per-write path never
-- touches the control-plane Postgres; `consolidated_lsn` is written here only at
-- consolidation time, never per write.
--
-- DEFAULT-OFF / backward compatible. The recall tier ships behind
-- `RB_RECALL` (default off). With the flag off nothing ever sets a non-zero
-- value: every existing and future shard simply carries `consolidated_lsn = 0`,
-- the query path takes the pure consolidated path, and existing shard logic is
-- untouched.
--
-- Non-destructive and idempotent: a plain `ADD COLUMN ... IF NOT EXISTS` with a
-- `NOT NULL DEFAULT 0`, so every existing shard transparently gets `0` with no
-- backfill step and a re-run is a clean no-op (mirrors 006_tenants_dp_pool.sql).

ALTER TABLE shard_catalog
  ADD COLUMN IF NOT EXISTS consolidated_lsn BIGINT NOT NULL DEFAULT 0;
