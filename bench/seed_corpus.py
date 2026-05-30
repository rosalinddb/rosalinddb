"""Seed N tenants x M vectors at a given dimension for the k6 bench.

Signs up tenants, creates a dataset per tenant, ingests M random
vectors per dataset, polls until each is `status='indexed'`, and
writes the resulting `(api_key, dataset)` pairs as JSON for
load_test_queries.js to consume.

Usage:
    python bench/seed_corpus.py \
        --base-url http://localhost:8080 \
        --dim 128 --tenants 15 --vectors-per 309 \
        --out bench/cache/dim-128.json

    # single-tenant high-volume mode (e.g. the mmap-comparison bench):
    python bench/seed_corpus.py \
        --base-url http://localhost:8080 \
        --dim 1536 --vectors-per 1000000 \
        --out bench/cache/dim-1536-1M.json \
        --single-tenant --index-timeout 1800

Vector values are uniform[-1, 1] floats. Each record carries a
`category` metadata field cycling through {books, movies, music}
so filtered queries always hit a non-empty filter.

Bodies are POSTed in NDJSON chunks of at most ~8 MiB (safe margin below
the CP `INGEST_MAX_BYTES` default of 10 MiB) so a million-record seed
stays under the per-request cap.
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import requests

CATEGORIES = ["books", "movies", "music"]
_DEFAULT_TENANTS = 15

# Default chunk byte budget for a single POST /v1/datasets/.../vectors body.
# The CP `INGEST_MAX_BYTES` default is 10 MiB; 8 MiB leaves headroom for
# the trailing newline accounting + a little jitter in record sizes.
_DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024


def rand_email() -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
    return f"load-{suffix}@bench.rosalinddb.example.com"


def signup(base_url: str, session: requests.Session) -> str | None:
    email = rand_email()
    resp = session.post(
        f"{base_url}/auth/signup",
        json={"email": email, "password": "pw-bench-12345"},
        timeout=30,
    )
    if resp.status_code != 201:
        print(f"  signup failed: {resp.status_code} {resp.text[:200]}", file=sys.stderr)
        return None
    body = resp.json()
    key = body.get("first_api_key", {}).get("key")
    if not key:
        print(f"  signup OK but no first_api_key.key: {body}", file=sys.stderr)
        return None
    return key


def create_dataset(base_url: str, session: requests.Session, api_key: str, name: str, dim: int) -> bool:
    resp = session.post(
        f"{base_url}/v1/datasets",
        json={"name": name, "dimension": dim},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        print(f"  create_dataset {name}: {resp.status_code} {resp.text[:200]}", file=sys.stderr)
        return False
    return True


def _gen_records(name: str, n: int, dim: int) -> Iterator[dict]:
    """Yield n synthetic records, deterministic per dataset name."""
    rng = random.Random(hash(name) & 0xFFFFFFFF)
    for i in range(n):
        yield {
            "id": f"v{i}",
            "values": [rng.uniform(-1.0, 1.0) for _ in range(dim)],
            "metadata": {"category": CATEGORIES[i % len(CATEGORIES)]},
        }


def _chunk_ndjson(
    records: Iterable[dict],
    max_bytes: int = _DEFAULT_CHUNK_BYTES,
) -> Iterator[list[dict]]:
    """Yield lists of records whose serialised NDJSON body fits in `max_bytes`.

    Records are emitted in input order, never split across chunks. The body
    accounting matches what `_post_chunk` actually POSTs: each record's
    `json.dumps(rec)` plus one byte for the newline separator (the last
    record contributes no newline, which is accounted for by tracking
    running size without the trailing newline).

    If a single record's serialised size alone exceeds `max_bytes`, it is
    yielded as a one-element chunk. The seeder leaves the rejection to the
    CP (413 `payload_too_large`); this helper never silently drops data.
    """
    chunk: list[dict] = []
    chunk_bytes = 0  # bytes for current chunk's body, no trailing newline
    for rec in records:
        line = json.dumps(rec)
        line_bytes = len(line.encode("utf-8"))
        # Cost of appending: a newline separator + the new line, unless this is
        # the first record in the chunk (no separator).
        added = line_bytes if not chunk else line_bytes + 1
        if chunk and chunk_bytes + added > max_bytes:
            yield chunk
            chunk = [rec]
            chunk_bytes = line_bytes
            continue
        chunk.append(rec)
        chunk_bytes += added
    if chunk:
        yield chunk


def _post_chunk(
    base_url: str,
    session: requests.Session,
    api_key: str,
    name: str,
    chunk: Sequence[dict],
) -> bool:
    body = "\n".join(json.dumps(r) for r in chunk).encode("utf-8")
    resp = session.post(
        f"{base_url}/v1/datasets/{name}/vectors",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/x-ndjson",
        },
        timeout=300,
    )
    if resp.status_code not in (200, 201, 202):
        print(
            f"  ingest {name}: {resp.status_code} {resp.text[:200]}",
            file=sys.stderr,
        )
        return False
    return True


def ingest(
    base_url: str,
    session: requests.Session,
    api_key: str,
    name: str,
    n: int,
    dim: int,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
) -> bool:
    """POST n random vectors to /vectors, chunked under `chunk_bytes` per body.

    Small seeds (the existing 15 x 309 case) fit in a single chunk and
    behave exactly as before. Large seeds (1M x 1536) are streamed across
    many chunks with progress printed every chunk.
    """
    chunks = list(_chunk_ndjson(_gen_records(name, n, dim), chunk_bytes))
    total_chunks = len(chunks)
    sent = 0
    for idx, chunk in enumerate(chunks, start=1):
        if not _post_chunk(base_url, session, api_key, name, chunk):
            return False
        sent += len(chunk)
        if total_chunks > 1:
            print(
                f"  ingested {sent}/{n} records (chunk {idx}/{total_chunks})",
                flush=True,
            )
    return True


def wait_indexed(
    base_url: str,
    session: requests.Session,
    api_key: str,
    name: str,
    timeout_s: int = 180,
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = session.get(
            f"{base_url}/v1/datasets/{name}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if resp.status_code == 200:
            status = resp.json().get("status")
            if status == "indexed":
                return True
            if status in {"error", "failed"}:
                print(f"  {name} reached terminal failure status={status}", file=sys.stderr)
                return False
        time.sleep(1.5)
    print(f"  {name} did not index within {timeout_s}s", file=sys.stderr)
    return False


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--tenants", type=int, default=_DEFAULT_TENANTS)
    ap.add_argument("--vectors-per", type=int, default=309)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=None, help="random seed for reproducible emails")
    ap.add_argument(
        "--single-tenant",
        action="store_true",
        help="seed exactly one tenant (overrides --tenants); pair with a large --vectors-per",
    )
    ap.add_argument(
        "--chunk-bytes",
        type=int,
        default=_DEFAULT_CHUNK_BYTES,
        help=f"max NDJSON body size per POST (default {_DEFAULT_CHUNK_BYTES} bytes ~ 8 MiB)",
    )
    ap.add_argument(
        "--index-timeout",
        type=int,
        default=180,
        help="seconds to wait for each dataset to reach status=indexed (default 180)",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

    # --single-tenant is the cleaner override: ignore --tenants entirely.
    # If the operator passed BOTH a non-default --tenants and --single-tenant
    # (e.g. CI wrapper drift), warn so the silent override is visible.
    if args.single_tenant and args.tenants != _DEFAULT_TENANTS:
        print(
            f"WARNING: --single-tenant overrides --tenants={args.tenants}",
            file=sys.stderr,
        )
    tenants = 1 if args.single_tenant else args.tenants

    sess = requests.Session()
    corpus: list[dict[str, str]] = []
    start = time.time()
    print(
        f"Seeding corpus: dim={args.dim} tenants={tenants} "
        f"vectors/tenant={args.vectors_per} chunk_bytes={args.chunk_bytes}"
    )
    for i in range(tenants):
        t_start = time.time()
        api_key = signup(args.base_url, sess)
        if api_key is None:
            continue
        dataset = f"bench{i}"
        if not create_dataset(args.base_url, sess, api_key, dataset, args.dim):
            continue
        if not ingest(
            args.base_url, sess, api_key, dataset,
            args.vectors_per, args.dim,
            chunk_bytes=args.chunk_bytes,
        ):
            continue
        if not wait_indexed(args.base_url, sess, api_key, dataset, timeout_s=args.index_timeout):
            continue
        corpus.append({"api_key": api_key, "dataset": dataset})
        print(f"  [{i+1}/{tenants}] {dataset} indexed in {time.time()-t_start:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(corpus, indent=2))
    elapsed = time.time() - start
    print(f"\nDone. Wrote {len(corpus)} corpus entries to {args.out} in {elapsed:.1f}s")
    if len(corpus) < tenants:
        print(f"WARNING: {tenants - len(corpus)} tenants failed to seed", file=sys.stderr)
        return 1 if len(corpus) == 0 else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
