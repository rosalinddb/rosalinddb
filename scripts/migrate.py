"""Idempotent one-shot database migration entrypoint.

Run before starting services; safe to run from any init container or k8s job.
Applies the schema a single time per deploy rather than racing across the
long-running process groups on startup.

`force=True` is load-bearing. When the long-running services are started with
`RB_SKIP_MIGRATE=1` (so they do NOT migrate on boot), this entrypoint will
typically inherit the same env and see the same flag. A plain `migrate()`
would then short-circuit — applying nothing, while the workers also skip it:
a fresh deploy would come up with no database schema. `force=True` bypasses
the `RB_SKIP_MIGRATE` guard so this entrypoint always applies the schema; the
guard still works as intended for the service process groups.

`migrate()` is idempotent and serialises safely (a `pg_advisory_xact_lock`
plus a version ledger), but running it once up front keeps every long-running
process from taking Postgres DDL locks on boot.

Usage:

    python -m scripts.migrate
"""
from __future__ import annotations

import sys


def main() -> None:
    """Apply the database schema once, then exit non-zero on failure."""
    from adapters.state.state import migrate, migrate_hot

    print("migrate: applying database schema...")
    # force=True: this entrypoint must apply the schema even when it inherits
    # `RB_SKIP_MIGRATE=1` from the surrounding env (the same env the
    # long-running services see, where that flag suppresses on-boot migration).
    migrate(force=True)
    print("migrate: schema is up to date.")

    # Hot-tier (delta tier) schema. Runs against the SEPARATE pgvector instance
    # addressed by RB_HOT_DSN. DEFAULT-OFF: when RB_HOT_DSN is unset this is a
    # pure no-op — no connection is opened and the line below reports it skipped,
    # so a flag-off deploy behaves byte-identically to today.
    if migrate_hot(force=True):
        print("migrate: hot-tier (pgvector) schema is up to date.")
    else:
        print("migrate: hot tier off (RB_HOT_DSN unset); skipping hot schema.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"migrate: FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
