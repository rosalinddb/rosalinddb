"""Shared utilities for cross-service modules.

Lives under `services._common` (not at the repo root or under `adapters/`)
to avoid the existing one-way import constraints: `services.query_api`
and `services.ephemeral_runner` deliberately do NOT import each other
(circular), so anything they both need lives here. Keep this package
narrow — code that genuinely belongs to `adapters` should land there
instead.
"""
