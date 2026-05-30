-- 005_dataset_status_updated_at.sql
-- feat/reliable-queue: deployment-resilience hardening of the queue.
--
-- Adds `status_updated_at` to `dataset_catalog`: the wall-clock time the
-- dataset's `status` last changed. The reconciliation reaper uses it as the
-- backstop for a worker that hangs (or dies mid-job and whose queue message
-- is somehow not redelivered) — any dataset stuck in a non-terminal status
-- (`validating`/`indexing`) for longer than the reaper's timeout is flipped to
-- `error` so a customer-facing `GET /v1/datasets/{name}` can never report a
-- silently-stuck dataset forever.
--
-- Non-destructive: a plain `ADD COLUMN ... IF NOT EXISTS`. Existing rows get
-- `now()` as a sane starting point (a freshly-deployed schema has no stuck
-- datasets to reconcile anyway).

ALTER TABLE dataset_catalog
  ADD COLUMN IF NOT EXISTS status_updated_at TIMESTAMPTZ NOT NULL DEFAULT now();

CREATE INDEX IF NOT EXISTS dataset_catalog_status_updated_idx
  ON dataset_catalog(status, status_updated_at);
