# Reporting Security Issues

If you believe you've found a security vulnerability in RosalindDB, please **do not** open a public GitHub issue.

Report it via GitHub Security Advisories: https://github.com/rosalinddb/rosalinddb/security/advisories/new

We'll acknowledge receipt within 72 hours and aim to address confirmed reports within 30 days. We'll coordinate disclosure with you once a fix is available.

## Scope

In scope:
- Authentication / authorization bypass (`/auth/*`, API key handling, tenant isolation)
- Vector / metadata data exposure across tenants
- Storage / queue privilege escalation
- Remote code execution via ingest paths (NDJSON / Parquet)

Out of scope:
- Findings that require physical access to a self-hosted deployment
- DoS via large legitimate workloads (quota / rate-limit configuration is the operator's responsibility)
- Issues in third-party dependencies (please report those upstream)

## Self-hosters

RosalindDB ships sensible defaults but production hardening is your responsibility. See `docs/deploy/self-host.md` for guidance on `JWT_SECRET`, network isolation between CP and DP, and S3 / Postgres / Redis credentials.
