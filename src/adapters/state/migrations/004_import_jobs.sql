-- 004_import_jobs.sql
-- Async bulk ingest (the import-job pattern).
--
-- Adds the `import_jobs` table that tracks an async bulk-import: the client
-- stages a large NDJSON/Parquet file directly into object storage via a
-- presigned upload, then a job validates + indexes it asynchronously.
--
-- Tenant-scoped: `(tenant_id, import_id)` makes cross-tenant reads/completes
-- impossible at the data layer (the v1 404-not-403 isolation rule). A FK to
-- `dataset_catalog` ties each job to a live dataset and cascades on delete.
--
-- Non-destructive: this migration only CREATEs a new table — `tenants`,
-- `api_keys`, `dataset_catalog`, and `shard_catalog` are left untouched.

CREATE TABLE IF NOT EXISTS import_jobs (
  import_id          TEXT NOT NULL,
  tenant_id          TEXT NOT NULL,
  dataset            TEXT NOT NULL,
  format             TEXT NOT NULL,                       -- ndjson | parquet
  status             TEXT NOT NULL DEFAULT 'awaiting_upload',
  error_mode         TEXT NOT NULL DEFAULT 'continue',     -- continue | abort
  max_bad_records    BIGINT,                               -- NULL = unlimited
  upload_uri         TEXT NOT NULL,                        -- staged object key
  records_processed  BIGINT NOT NULL DEFAULT 0,
  records_accepted   BIGINT NOT NULL DEFAULT 0,
  records_rejected   BIGINT NOT NULL DEFAULT 0,
  rejected_uri       TEXT,                                 -- rejected.jsonl key
  error_message      TEXT,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at       TIMESTAMPTZ,
  PRIMARY KEY (tenant_id, import_id),
  FOREIGN KEY (tenant_id, dataset)
    REFERENCES dataset_catalog(tenant_id, dataset_name)
    ON DELETE CASCADE
);

-- Newest-first listing per dataset (`GET /v1/datasets/{name}/imports`).
CREATE INDEX IF NOT EXISTS import_jobs_lookup_idx
  ON import_jobs(tenant_id, dataset, created_at DESC);

-- Plain import_id lookup for the worker (it only carries the import_id).
CREATE INDEX IF NOT EXISTS import_jobs_id_idx
  ON import_jobs(import_id);
