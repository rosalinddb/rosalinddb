# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

Initial public release.

### Added
- Object-storage-first vector search: FAISS IVFFlat shards stored on any
  S3-compatible object store, served from a byte-budgeted in-process cache.
- Optional on-disk (SSD) shard cache tier for shards larger than the in-process
  budget. Off by default (`RB_SHARD_TIER_BYTES` unset).
- Optional FAISS mmap load mode for indexes larger than the cache budget. Off by
  default (`RB_FAISS_MMAP=false`).
- Control-plane / data-plane split: a public control plane handles ingest,
  dataset management, and query routing; private data planes run the search.
- NDJSON ingest and bulk import via presigned upload.
- Queue-driven validate, build, and query workers that scale to zero.
- Optional authentication (JWT + API keys) and optional per-tenant limits, both
  off by default for self-hosting (`RB_REQUIRE_AUTH`, `RB_ENABLE_QUOTAS`).

[Unreleased]: https://github.com/rosalinddb/rosalinddb/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rosalinddb/rosalinddb/releases/tag/v0.1.0
