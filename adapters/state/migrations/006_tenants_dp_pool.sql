-- 006_tenants_dp_pool.sql
-- Control Plane / Data Plane routing: per-tenant DP pool assignment.
--
-- Adds `dp_pool` to `tenants`: the name of the Query Data Plane (DP) pool a
-- tenant's `/v1/query` traffic is routed to by the CP reverse proxy. The CP
-- router reads this column per request to resolve `tenant_id -> DP base URL`
-- (shared vs dedicated DP pool routing).
--
--   - 'shared'            -> the shared Query-DP pool (the default; every
--                            tenant uses it until provisioned onto a
--                            dedicated pool).
--   - 'dedicated-<tenant>' -> that tenant's dedicated Query-DP pool.
--
-- Non-destructive and idempotent: a plain `ADD COLUMN ... IF NOT EXISTS` with
-- a `NOT NULL DEFAULT 'shared'`, so every existing tenant transparently lands
-- on the shared pool with no backfill step and a re-run is a clean no-op.

ALTER TABLE tenants
  ADD COLUMN IF NOT EXISTS dp_pool TEXT NOT NULL DEFAULT 'shared';
