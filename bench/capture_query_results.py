"""Capture per-query top-K IDs for a single bench cell.

The SSD-cache bench's correctness claim is "the tier does not change query
semantics" — every cell must return the same matches for the same query.
That claim does NOT require absolute recall against brute force; it
requires *agreement* between cells (tier-on-cold vs tier-off, tier-on-warm
vs tier-off, etc.).

This script captures (query_id -> returned_ids) for a fixed query set in
one cell. The analyzer then compares the per-cell captures to compute
agreement (fraction of queries where the cell's returned IDs match the
baseline cell's returned IDs).

Why not do this inside the k6 driver: k6 is the load driver — its job is
to push QPS and measure latency under load. Tagging per-query response
payloads into k6 metrics either explodes the metric cardinality (one
series per query) or requires a custom output sink. A separate Python
step runs deterministically at the END of each cell against the same
backend without contending with the load test, so the captured IDs
reflect a quiescent backend rather than a saturated one.

Usage:
    python bench/capture_query_results.py \\
        --base-url http://localhost:8080 \\
        --corpus bench/cache/dim-128-ssd-cache.json \\
        --query-set bench/cache/query-set-dim-128-n50000.json \\
        --num-queries 200 \\
        --out bench/results/<cell_dir>/query_results.json

The corpus file is what `seed_corpus.py --single-tenant` wrote (gives us
the dataset name + api_key). The query set is the test queries from
`build_ground_truth.py` (we ignore its `top_k_ids` field — we only need
the query vectors). On a fresh deployment a query set can also be
generated inline by `--queries-from-corpus`, which samples vectors from
the corpus with 1% Gaussian noise.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any


def _load_corpus(path: str) -> dict:
    with open(path) as f:
        records = json.load(f)
    if not records:
        raise RuntimeError(f"corpus {path} is empty")
    if isinstance(records, list):
        # Single-tenant cache shape from seed_corpus.py
        return records[0]
    return records


def _load_or_build_queries(
    query_set_path: str | None,
    dataset: str,
    dim: int,
    vectors_per: int,
    num_queries: int,
) -> list[dict]:
    """Return [{"qid": int, "vector": list[float]}].

    If `query_set_path` exists, load it (skip the `top_k_ids` field — the
    cell-agreement check does not use brute-force ground truth). Otherwise
    derive a deterministic query set from the corpus generator inline so
    the script works without a prior `build_ground_truth.py` run.
    """
    if query_set_path and os.path.exists(query_set_path):
        with open(query_set_path) as f:
            data = json.load(f)
        queries = data.get("queries") or []
        return [{"qid": q["qid"], "vector": q["vector"]} for q in queries[:num_queries]]

    # Inline derivation — mirrors build_ground_truth.py's query strategy
    # (sample a corpus vector + 1% Gaussian noise) without requiring numpy.
    rng = random.Random(hash(dataset) & 0xFFFFFFFF)
    corpus = [
        [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        for _ in range(vectors_per)
    ]
    qrng = random.Random(42)  # fixed seed: same queries across cells
    queries = []
    for qid in range(num_queries):
        idx = qrng.randrange(0, vectors_per)
        base = corpus[idx]
        # 1% Gaussian noise per dim (approximated; we avoid numpy here so
        # the script has no heavy deps at the bench-cell boundary).
        noise = [qrng.gauss(0.0, 0.01) for _ in range(dim)]
        vec = [b + n for b, n in zip(base, noise)]
        queries.append({"qid": qid, "vector": vec})
    return queries


def _post_query(
    base_url: str,
    api_key: str,
    dataset: str,
    vector: list[float],
    top_k: int,
    timeout: float,
) -> list[str]:
    """POST /v1/query, return the list of match IDs. Bubble HTTP errors."""
    import urllib.request

    body = json.dumps(
        {"dataset": dataset, "vector": vector, "top_k": top_k}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/query",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return [m["id"] for m in payload.get("matches", [])]


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8080")
    p.add_argument(
        "--corpus",
        required=True,
        help="seed_corpus.py cache file (single-tenant shape)",
    )
    p.add_argument(
        "--query-set",
        default=None,
        help=(
            "Optional: build_ground_truth.py query file (only the "
            "`queries[].vector` field is used; `top_k_ids` is ignored)."
        ),
    )
    p.add_argument("--num-queries", type=int, default=200)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument(
        "--dim", type=int, default=None,
        help=(
            "Vector dimension. Only needed when --query-set is absent "
            "(then we regenerate the corpus locally to derive queries). "
            "When --query-set is provided, this is ignored."
        ),
    )
    p.add_argument(
        "--vectors-per", type=int, default=None,
        help="Vectors per dataset (corpus cardinality). Same condition as --dim.",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output JSON path: {qid: [returned_id, ...]} per query.",
    )
    args = p.parse_args(argv)

    tenant = _load_corpus(args.corpus)
    api_key = tenant["api_key"]
    dataset = tenant["dataset"]
    # When --query-set is present we only need (api_key, dataset) from the
    # tenant record; the queries come from the JSON file. When it is
    # absent we need to regenerate the corpus locally, which requires
    # --dim and --vectors-per (the CLI knows them; seed_corpus.py does
    # not record them in its output).
    have_query_set = bool(args.query_set and os.path.exists(args.query_set))
    if not have_query_set:
        if args.dim is None or args.vectors_per is None:
            raise RuntimeError(
                f"--query-set was not provided (or {args.query_set!r} does "
                f"not exist), so capture_query_results would have to "
                f"regenerate the corpus to derive queries. That requires "
                f"--dim and --vectors-per. Either pass --query-set pointing "
                f"at a build_ground_truth.py output, or pass both --dim and "
                f"--vectors-per."
            )

    queries = _load_or_build_queries(
        args.query_set, dataset, args.dim or 0, args.vectors_per or 0,
        args.num_queries,
    )
    if not queries:
        print(f"no queries available (--num-queries={args.num_queries})", file=sys.stderr)
        return 2

    print(
        f"capturing {len(queries)} queries against {args.base_url} "
        f"dataset={dataset} top_k={args.top_k}",
        file=sys.stderr,
    )

    results: dict[str, list[str]] = {}
    started = time.time()
    failures = 0
    for q in queries:
        try:
            ids = _post_query(
                args.base_url, api_key, dataset, q["vector"],
                args.top_k, args.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  qid={q['qid']} FAILED: {exc}", file=sys.stderr)
            continue
        results[str(q["qid"])] = ids

    elapsed = time.time() - started
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "dataset": dataset,
                "num_queries": len(queries),
                "captured": len(results),
                "failures": failures,
                "top_k": args.top_k,
                "elapsed_s": elapsed,
                "results": results,
            },
            f,
            indent=2,
        )
    print(
        f"captured {len(results)}/{len(queries)} queries "
        f"({failures} failures) in {elapsed:.1f}s -> {args.out}",
        file=sys.stderr,
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
