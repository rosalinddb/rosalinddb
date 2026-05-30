-- 007_dp_shard_residency.sql
-- SSD-cache feature: DP residency registry.
--
-- A row per (dp_id, shard_uri) for every shard a DP currently holds warm in
-- its SSD tier. The DP's residency_writer daemon (services/_common/
-- residency_writer.py) periodically reconciles `shard_tier.residency()` to
-- this table — UPSERTing on every cycle so `last_query_at` stays recent, and
-- DELETEing rows for shards the tier no longer holds.
--
-- This table is ADVISORY: a stale row only causes a cache miss (and a
-- re-fetch) on the wrong DP, never a correctness break. It is the intended
-- data source for residency-aware CP routing (which would prefer DPs that
-- already have the shard warm), but is not yet wired into the CP.
--
-- Schema choices:
--   - `shard_uri TEXT` (not `shard_id BIGINT`): the tier itself is keyed on
--     URI (the cache's primary key is the immutable, content-addressed shard
--     URI). Keying the registry the same way means the writer does not need
--     a catalog join to translate.
--   - `warm_since` is set on first admit (the UPSERT's INSERT branch) and
--     left alone on refresh. Reads it as "how long has this DP held the
--     shard cached" — refreshing it on every cycle would defeat that.
--   - `last_query_at` is refreshed on every UPSERT (the ON CONFLICT branch
--     overwrites it) so a CP freshness filter sees recent activity.
--   - DOUBLE PRECISION for epoch seconds: matches Python `time.time()`'s
--     float, avoids the `to_timestamp` conversion the writer would need if
--     we used `TIMESTAMPTZ`.
--
-- The `shard_uri` index supports the "which DPs have shard X warm?" read
-- path. The PK index supports the writer's own UPSERT and the "what is
-- dp Y holding?" operator read path.

CREATE TABLE IF NOT EXISTS dp_shard_residency (
    dp_id TEXT NOT NULL,
    shard_uri TEXT NOT NULL,
    warm_since DOUBLE PRECISION NOT NULL,
    last_query_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (dp_id, shard_uri)
);

CREATE INDEX IF NOT EXISTS dp_shard_residency_shard_uri_idx
    ON dp_shard_residency (shard_uri);
