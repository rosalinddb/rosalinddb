"""Build a large FAISS shard directly, bypassing the CP ingest path.

The CP ingest path (auth + per-record validation + Parquet landing + Redis
queue + builder worker) is the right shape for measuring ingest throughput
but the wrong shape for landing a multi-GB shard for query benchmarking:
at single-tenant 1M x 1536-dim, it would take many hours regardless of
whether mmap is on. This tool produces the same shard the builder would
have produced, but in two minutes of local compute.

What it does:
  1. POST /auth/signup against the running CP -> api_key, tenant_id.
  2. POST /v1/datasets {name, dimension} -> dataset row in the catalog.
  3. Generate N random float32 vectors of the requested dimension.
  4. Build an IndexIDMap2(IndexIVFFlat) locally, train + add the vectors.
  5. Serialise via faiss.write_index, upload the bytes to MinIO at the
     standard `indexes/{tenant}/{dataset}/shard-<id>.bin` URI.
  6. Build the `{shard_uri}.meta.json` sidecar and upload it alongside.
  7. INSERT into `shard_catalog` so the query path resolves the shard.
  8. Write the bench cache file `[{api_key, dataset}]` for k6 to consume.

After this, `bench/run_mmap_comparison.sh` (or a manual cell-runner) can
drive queries against the shard with no further setup.

Usage:
    python bench/build_shard_directly.py \\
        --base-url http://localhost:8080 \\
        --dim 1536 --vectors 1000000 --nlist 4096 \\
        --out bench/cache/dim-1536-mmap.json

Defaults match what `run_mmap_comparison.sh` reads from `bench/cache/`.

Env reads (must match the running compose stack):
  S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY, S3_REGION  (MinIO creds)
  INDEXES_PREFIX                                            (s3://vectors/indexes)
  DATABASE_URL                                              (postgres dsn)
"""
from __future__ import annotations

import argparse
import json
import os
import string
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import boto3
import faiss  # type: ignore
import numpy as np
import psycopg2
import requests


_DEFAULT_INDEXES_PREFIX = "s3://vectors/indexes"
_DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/vectors"


def _s3_client():
    """Build a boto3 S3 client wired at the bench MinIO.

    `S3_ACCESS_KEY` and `S3_SECRET_KEY` are REQUIRED — the bench compose sets
    them explicitly via `bench/docker-compose.bench.yml`. We refuse to fall
    back to plaintext defaults here so a reader of this file does not mistake
    the test creds for production-relevant values; if you are running this
    script standalone, source `bench/.env` or export the two vars yourself.
    """
    try:
        access_key = os.environ["S3_ACCESS_KEY"]
        secret_key = os.environ["S3_SECRET_KEY"]
    except KeyError as missing:
        raise RuntimeError(
            f"{missing.args[0]} is required; export it or source bench/.env "
            "before running this script."
        ) from missing
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=os.environ.get("S3_REGION", "us-east-1"),
    )


def _split_s3_uri(uri: str) -> tuple[str, str]:
    """Parse `s3://bucket/key/path` into (bucket, key)."""
    assert uri.startswith("s3://"), f"not an s3 URI: {uri}"
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def signup(base_url: str) -> tuple[str, str]:
    """Sign up a fresh tenant, return (api_key, tenant_id)."""
    suffix = uuid.uuid4().hex[:12]
    email = f"directbuild-{suffix}@bench.example.com"
    resp = requests.post(
        f"{base_url}/auth/signup",
        json={"email": email, "password": "pw-directbuild-12345"},
        timeout=30,
    )
    if resp.status_code != 201:
        raise RuntimeError(f"signup failed: {resp.status_code} {resp.text[:200]}")
    body = resp.json()
    api_key = body["first_api_key"]["key"]
    tenant_id = body["tenant"]["id"]
    return api_key, tenant_id


def create_dataset(base_url: str, api_key: str, name: str, dim: int) -> None:
    """Create the dataset row so the CP knows it exists."""
    resp = requests.post(
        f"{base_url}/v1/datasets",
        json={"name": name, "dimension": dim},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"create_dataset {name}: {resp.status_code} {resp.text[:200]}"
        )


def build_ivfflat(vectors: np.ndarray, nlist: int) -> faiss.Index:
    """Build an `IndexIDMap2(IndexIVFFlat)` over the given vectors.

    IVFFlat is the production index type and is mmap-clean against the
    standard FAISS file format (per the FAISS wiki "Indexes that do not
    fit in RAM"); IndexFlat is not. The id table is wrapped in
    `IndexIDMap2` to match what `services/index_builder/run.py` produces.
    """
    n, dim = vectors.shape
    print(f"  training IVFFlat (nlist={nlist}) on {n:,} vectors...", flush=True)
    t0 = time.time()
    quantizer = faiss.IndexFlatL2(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    ivf.train(vectors)
    print(f"  trained in {time.time()-t0:.1f}s", flush=True)

    inner = faiss.IndexIDMap2(ivf)
    ids = np.arange(1, n + 1, dtype=np.int64)
    print(f"  add_with_ids ({n:,} vectors)...", flush=True)
    t0 = time.time()
    inner.add_with_ids(vectors, ids)
    print(f"  added in {time.time()-t0:.1f}s", flush=True)
    return inner


def upload_shard(
    index: faiss.Index, vectors_n: int, tenant: str, dataset: str
) -> str:
    """Serialise the index, upload to MinIO. Returns the s3:// shard URI."""
    indexes_prefix = os.environ.get("INDEXES_PREFIX", _DEFAULT_INDEXES_PREFIX)
    shard_uri = f"{indexes_prefix}/{tenant}/{dataset}/shard-{uuid.uuid4().hex[:8]}.bin"

    print(f"  serialising index...", flush=True)
    t0 = time.time()
    payload = faiss.serialize_index(index).tobytes()
    print(f"  serialised {len(payload)/1e9:.2f} GB in {time.time()-t0:.1f}s", flush=True)

    bucket, key = _split_s3_uri(shard_uri)
    s3 = _s3_client()
    print(f"  uploading to s3://{bucket}/{key}...", flush=True)
    t0 = time.time()
    s3.put_object(Bucket=bucket, Key=key, Body=payload)
    print(f"  uploaded in {time.time()-t0:.1f}s", flush=True)
    return shard_uri


def upload_sidecar(shard_uri: str, vectors_n: int) -> None:
    """Build + upload the `{shard_uri}.meta.json` sidecar.

    Sidecar maps each int64 id to `{id, metadata}`. The id table in this
    shard uses contiguous ids 1..N, so a generator-style dict comprehension
    is fine; for 1M entries it's ~120 MB of JSON.
    """
    sidecar = {
        str(i): {
            "id": f"r{i}",
            "metadata": {"category": ["books", "movies", "music"][i % 3]},
        }
        for i in range(1, vectors_n + 1)
    }
    bucket, key = _split_s3_uri(f"{shard_uri}.meta.json")
    s3 = _s3_client()
    print(f"  uploading sidecar to s3://{bucket}/{key}...", flush=True)
    t0 = time.time()
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(sidecar).encode("utf-8"))
    print(f"  uploaded in {time.time()-t0:.1f}s", flush=True)


def register_shard(
    tenant: str, dataset: str, shard_uri: str, vectors_n: int
) -> int:
    """INSERT a row into `shard_catalog` so the query path resolves it.

    Mirrors what `adapters.state.state.add_shard` would have written for an
    IVFFlat full-coverage shard. We talk to Postgres directly (no adapter
    import) so this script needs no PYTHONPATH gymnastics. The `id`
    column is a `bigint` serial — let the DB auto-assign it; we just
    capture the value for the operator log.
    """
    dsn = os.environ.get("DATABASE_URL", _DEFAULT_DATABASE_URL)
    print("  inserting shard_catalog row...", flush=True)
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO shard_catalog (
                    tenant_id, dataset_name, shard_uri, checksum,
                    vector_count, index_type, build_type,
                    indexed_landing_uris, sealed
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    tenant, dataset, shard_uri,
                    "direct-build-no-checksum",
                    vectors_n, "ivf", "full",
                    [],   # empty landing list — this shard didn't come from landing
                    True,
                ),
            )
            shard_id = cur.fetchone()[0]
        conn.commit()
    return shard_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--vectors", type=int, required=True)
    ap.add_argument(
        "--nlist", type=int, default=4096,
        help="IVF cell count; FAISS rule-of-thumb is ~sqrt(n) (defaults work for 1M)",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--seed", type=int, default=20240524,
        help="numpy seed; reproducible vectors and ids",
    )
    args = ap.parse_args()

    print(f"build_shard_directly: dim={args.dim} vectors={args.vectors:,} nlist={args.nlist}")
    print(f"  base_url={args.base_url}")
    print(f"  out={args.out}")
    start = time.time()

    print("[step 1/7] signing up tenant", flush=True)
    api_key, tenant_id = signup(args.base_url)
    print(f"  tenant={tenant_id}  api_key={api_key[:20]}...", flush=True)

    dataset = f"directbuild-{uuid.uuid4().hex[:8]}"
    print(f"[step 2/7] creating dataset {dataset}", flush=True)
    create_dataset(args.base_url, api_key, dataset, args.dim)

    print(f"[step 3/7] generating {args.vectors:,} x {args.dim} random vectors", flush=True)
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    vectors = rng.random((args.vectors, args.dim), dtype=np.float32)
    print(f"  generated {vectors.nbytes/1e9:.2f} GB in {time.time()-t0:.1f}s", flush=True)

    print(f"[step 4/7] building IVFFlat index", flush=True)
    index = build_ivfflat(vectors, args.nlist)

    print(f"[step 5/7] uploading shard to MinIO", flush=True)
    shard_uri = upload_shard(index, args.vectors, tenant_id, dataset)

    print(f"[step 6/7] uploading sidecar", flush=True)
    upload_sidecar(shard_uri, args.vectors)

    print(f"[step 7/7] registering in shard_catalog", flush=True)
    shard_id = register_shard(tenant_id, dataset, shard_uri, args.vectors)

    # Cache file matches the format `bench/run_mmap_comparison.sh` expects.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([{"api_key": api_key, "dataset": dataset}], indent=2))
    print(
        f"\nDone in {time.time()-start:.1f}s. "
        f"Wrote 1 corpus entry to {args.out}\n"
        f"  tenant={tenant_id}\n  dataset={dataset}\n"
        f"  shard_id={shard_id}\n  shard_uri={shard_uri}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
