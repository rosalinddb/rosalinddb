-- 009_delta_tier.sql
-- Delta-shard LSM: base + small delta shards over one frozen coarse quantizer.
--
-- Adds the columns that describe a shard's place in a base+delta generation, so
-- a recall fold can write a CHEAP delta shard (O(rows folded)) instead of
-- rebuilding the whole base, and the read path can union base + live deltas with
-- a correct watermark, sweep, and grace. See:
--   bench-lab/research/compaction-redesign.md  (parent design)
--   bench-lab/research/phase1-spec.md          (this slice)
--   docs/architecture/recall-consolidate.md
--
--   quantizer_version : the frozen coarse-quantizer generation whose Voronoi
--                       cells this shard's IVF shares. Only same-version shards
--                       may MERGE (cross-version SEARCH is fine — each shard is
--                       searched independently and merged by exact L2).
--   parent_shard_id   : for a level=1 delta, the base shard it layers on (NULL
--                       for a base). Sweep/grace liveness is by generation
--                       membership via this pointer, never by list position.
--   level             : 0 = base, 1 = delta.
--   covered_lsn_lo /
--   covered_lsn_hi    : the recall-LSN band this shard covers. The query builds
--                       a CONTIGUOUS-FRONTIER watermark from these (I1) so a
--                       missing/unreadable delta clamps the watermark (recall
--                       re-serves the band) instead of dropping vectors.
--   tombstone_int_ids : int64 ids (SHA1->int64 hash of the string id) deleted
--                       from the cold tier by this fold. Suppressed at query
--                       time and physically purged at major compaction — no S3
--                       tombstone object, no IVF remove_ids (which aborts on
--                       IVFFlat in FAISS 1.8.0).
--
-- DEFAULT-OFF / backward compatible. Every column is additive with a safe
-- default, so every existing shard becomes a `level=0` base with no parent and
-- an empty tombstone set — the pre-delta sweep/grace/watermark behaviour is
-- preserved exactly. The delta-write path itself ships behind `RB_DELTA_TIER`
-- (default off); with the flag off nothing ever writes a level=1 row.
--
-- Non-destructive and idempotent: plain `ADD COLUMN ... IF NOT EXISTS` +
-- `CREATE INDEX ... IF NOT EXISTS`, so re-running is a clean no-op (mirrors
-- 006_tenants_dp_pool.sql and 008_shard_consolidated_lsn.sql).

ALTER TABLE shard_catalog
  ADD COLUMN IF NOT EXISTS quantizer_version INTEGER  NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS parent_shard_id   BIGINT,
  ADD COLUMN IF NOT EXISTS level             SMALLINT NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS covered_lsn_lo    BIGINT   NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS covered_lsn_hi    BIGINT   NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS tombstone_int_ids BIGINT[] NOT NULL DEFAULT '{}';

-- Speeds the generation-membership query (deltas by parent base).
CREATE INDEX IF NOT EXISTS shard_catalog_gen_idx
  ON shard_catalog (tenant_id, dataset_name, parent_shard_id);
