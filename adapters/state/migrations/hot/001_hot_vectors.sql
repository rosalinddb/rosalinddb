-- hot/001_hot_vectors.sql
-- Delta tier (hot tier) schema. Runs against RB_HOT_DSN — the SEPARATE
-- data-plane pgvector instance, NOT the control-plane Postgres (see
-- docs/architecture/delta-tier.md, "Blast radius & control/data-plane
-- isolation"). Applied by the hot migration runner only when RB_HOT_DSN is set;
-- a no-op when the delta tier is off.
--
-- This is the in-front-of-the-shards "memtable": synchronously written, exactly
-- queryable, and drained into immutable FAISS shards on flush.

-- pgvector. The hot instance uses the `pgvector/pgvector:pg15` image, which
-- ships the extension; `IF NOT EXISTS` keeps a re-run a clean no-op.
CREATE EXTENSION IF NOT EXISTS vector;

-- The hot vectors table — one row per live (tenant, dataset, id) plus tombstones.
--
-- Embedding dimension is PER-DATASET, so the `embedding` column is declared as
-- an UNPARAMETERISED `vector` (no `vector(N)` typmod). pgvector (>= 0.5.0, which
-- the pg15 image ships) permits this: each row may store a vector of any
-- dimension, and the dimension is enforced per (tenant, dataset) by the
-- application at write time (matching `dataset_catalog.dimension` in the
-- control plane) rather than by the column type. A single shared hot instance
-- therefore serves datasets of differing dimensions without a table-per-dataset
-- explosion. (Alternative considered: one `vector(N)`-typed partition per
-- dataset — rejected as far heavier for the small, flush-bounded hot set.)
--
--   tenant_id, dataset : the partition key. Every read is scoped to one
--                        (tenant, dataset) via the b-tree index below, then an
--                        exact brute-force L2 scan runs over just those rows
--                        (the hot set is kept small by flush, so this is
--                        sub-millisecond with zero recall loss).
--   id                 : the caller-supplied vector id (last-write-wins on
--                        re-upsert).
--   embedding          : the vector itself, stored UN-normalised. The query
--                        path squares pgvector's plain-L2 `<->` distance to
--                        align with FAISS L2-squared before merging the tiers.
--                        NOTE: this is an UNPARAMETERISED pgvector `vector` (no
--                        `vector(N)` typmod) so one column can serve mixed
--                        per-dataset dimensions under brute-force exact search.
--                        A pgvector HNSW/IVFFlat index CANNOT be built on a
--                        mixed-dimension column — ANN indexes require a fixed
--                        dimension. So enabling `RB_HOT_INDEX=hnsw` later is NOT
--                        a drop-in: it first requires migrating to a
--                        fixed-dimension layout (e.g. a table/partition per
--                        embedding dimension). The escape hatch is real but gated
--                        on that schema change.
--   metadata           : arbitrary JSON metadata (AND-of-equals filtering).
--   lsn                : the per-(tenant, dataset) monotonic log sequence
--                        number (see hot_lsn_seq below). Partitions hot vs cold.
--   deleted            : tombstone flag. Delete = UPDATE ... SET deleted=true so
--                        the hot tier can immediately suppress a matching cold id.
--   created_at         : insert/observe time (diagnostics; LSN is the ordering).
CREATE TABLE IF NOT EXISTS hot_vectors (
  tenant_id  TEXT        NOT NULL,
  dataset    TEXT        NOT NULL,
  id         TEXT        NOT NULL,
  embedding  vector      NOT NULL,
  metadata   JSONB       NOT NULL DEFAULT '{}'::jsonb,
  lsn        BIGINT      NOT NULL,
  deleted    BOOLEAN     NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, dataset, id)
);

-- Partition-scan index. The hot read path filters
--   WHERE tenant_id = ? AND dataset = ? AND NOT deleted AND lsn > durable_lsn
-- so a composite b-tree on (tenant_id, dataset, lsn) lets Postgres jump
-- straight to the rows for one (tenant, dataset) above the watermark; the exact
-- L2 distance is then computed over that bounded set (brute force by design —
-- no ANN index; HNSW is a flagged escape hatch, not created here).
CREATE INDEX IF NOT EXISTS hot_vectors_partition_idx
  ON hot_vectors (tenant_id, dataset, lsn);

-- The flush trim (`lsn <= N`) and the visibility filter both range over `lsn`
-- within a partition, already covered by the index above.

-- Per-(tenant, dataset) LSN sequence.
--
-- Why a table and not a Postgres SEQUENCE: the LSN is per (tenant, dataset),
-- and there is an unbounded, dynamic set of those pairs — one real SEQUENCE per
-- pair does not scale and cannot be created from a parameterised statement on a
-- hot write. Instead a single row per pair holds the current value, and the
-- next LSN is allocated with an atomic upsert-increment:
--
--   INSERT INTO hot_lsn_seq (tenant_id, dataset, last_lsn)
--   VALUES (?, ?, 1)
--   ON CONFLICT (tenant_id, dataset)
--   DO UPDATE SET last_lsn = hot_lsn_seq.last_lsn + 1
--   RETURNING last_lsn;
--
-- Postgres serialises concurrent upserts on the same row, so the returned
-- `last_lsn` is strictly monotonic per (tenant, dataset) with no gaps required
-- and no cross-dataset contention. The sequence lives HERE in the hot store so
-- the per-write path never touches the control-plane Postgres (the watermark
-- `durable_lsn` is written to the control plane only at flush).
CREATE TABLE IF NOT EXISTS hot_lsn_seq (
  tenant_id TEXT   NOT NULL,
  dataset   TEXT   NOT NULL,
  last_lsn  BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (tenant_id, dataset)
);
