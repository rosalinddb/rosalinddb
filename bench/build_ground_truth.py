"""Build brute-force ground-truth top-10 ids for the recall@10 bench.

The SSD-cache bench (`bench/run_ssd_cache.sh`) needs a ground-truth file so
the k6 script can compute recall against a known-correct top-10. This script
produces that file once per (dataset, dim, vectors_per) tuple and caches it
on disk for reuse across bench runs.

How "ground truth" is constructed here:

1. The bench's corpus generator (`bench/seed_corpus.py::_gen_records`) is
   deterministic per dataset name: `random.Random(hash(name) & 0xFFFFFFFF)`
   re-produces the exact same vectors a server already ingested, given the
   same `(name, n, dim)`. We re-build the corpus locally rather than fetch
   it back from the server.
2. We pick `--num-queries` test queries by sampling vectors from the same
   deterministic corpus (seeded separately under `--query-seed`). Each test
   query is therefore a "near-duplicate" of one corpus entry — by
   construction, a correct top-1 hit includes that entry.
3. For each test query we run `faiss.IndexFlatL2.search(query, k)` over the
   full corpus and record the top-K ids.

The output JSON shape (consumed by `bench/load_test_queries_ssd_cache.js`):

    {
      "dataset": "bench0",
      "dim": 128,
      "vectors_per": 309,
      "k": 10,
      "seed_corpus_name": "bench0",
      "query_seed": 1234,
      "queries": [
        {"qid": 0, "vector": [...], "top_k_ids": ["v17", "v203", ...]},
        ...
      ]
    }

Usage:
    python bench/build_ground_truth.py \\
        --dataset bench0 --dim 128 --vectors-per 309 \\
        --num-queries 200 --out bench/cache/ground-truth-dim-128.json

    # Reuse if the file already exists, with shape match:
    python bench/build_ground_truth.py ... --reuse-if-exists

The bench harness invokes this once per (dim, dataset_size) tuple before
the first cell; subsequent runs reuse the cached file.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as e:
    print(f"faiss import failed: {e}\nInstall via `pip install faiss-cpu`.", file=sys.stderr)
    sys.exit(2)


CATEGORIES = ["books", "movies", "music"]


def gen_corpus(name: str, n: int, dim: int) -> tuple[np.ndarray, list[str]]:
    """Re-build the deterministic corpus that seed_corpus.py POSTed.

    Mirrors `_gen_records` in `bench/seed_corpus.py` exactly. Any drift
    between these two would silently break recall.
    """
    rng = random.Random(hash(name) & 0xFFFFFFFF)
    vectors = np.empty((n, dim), dtype=np.float32)
    ids: list[str] = []
    for i in range(n):
        for j in range(dim):
            vectors[i, j] = rng.uniform(-1.0, 1.0)
        ids.append(f"v{i}")
    return vectors, ids


def gen_test_queries(
    name: str,
    n_corpus: int,
    dim: int,
    num_queries: int,
    query_seed: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (qid, vector) pairs for the test query set.

    Strategy: sample `num_queries` corpus indices under `query_seed`, then
    perturb each by a tiny gaussian. The perturbation keeps the queries
    distinct from any single corpus vector (so the engine can't trivially
    return the exact match) while keeping recall@10 high enough that a
    regression (returning the wrong shard's bytes or skipping a real
    neighbour) is visible against the floor.

    `_gen_records` and this helper share `(name, n, dim)`; together with
    `query_seed` they fully determine the query set.
    """
    base_vectors, _ = gen_corpus(name, n_corpus, dim)
    rng = np.random.default_rng(query_seed)
    indices = rng.integers(0, n_corpus, size=num_queries)
    noise_scale = 0.01  # ~1% of unit interval; small enough to keep top-1 near
    for qid, idx in enumerate(indices):
        q = base_vectors[idx].copy()
        q = q + rng.normal(0.0, noise_scale, size=dim).astype(np.float32)
        yield qid, q


def compute_ground_truth(
    dataset: str,
    dim: int,
    vectors_per: int,
    num_queries: int,
    k: int,
    query_seed: int,
) -> dict:
    """Drive an IndexFlatL2 over the corpus to get exact top-k for each query."""
    print(
        f"  building corpus dataset={dataset} dim={dim} n={vectors_per}",
        flush=True,
    )
    corpus_vectors, corpus_ids = gen_corpus(dataset, vectors_per, dim)

    print(f"  building IndexFlatL2 (n={vectors_per} x dim={dim})", flush=True)
    index = faiss.IndexFlatL2(dim)
    index.add(corpus_vectors)

    print(f"  computing ground-truth top-{k} for {num_queries} queries", flush=True)
    queries_out: list[dict] = []
    # Batch the search for speed — FAISS is internally parallel and the
    # bench's ground-truth shape is small enough to fit a single batch.
    query_matrix = np.empty((num_queries, dim), dtype=np.float32)
    qids: list[int] = []
    for qid, vec in gen_test_queries(
        dataset, vectors_per, dim, num_queries, query_seed
    ):
        query_matrix[qid] = vec
        qids.append(qid)

    distances, top_idx = index.search(query_matrix, k)

    for qid in qids:
        top_k_ids = [corpus_ids[int(i)] for i in top_idx[qid] if i >= 0]
        queries_out.append(
            {
                "qid": qid,
                "vector": [float(x) for x in query_matrix[qid]],
                "top_k_ids": top_k_ids,
            }
        )

    return {
        "dataset": dataset,
        "dim": dim,
        "vectors_per": vectors_per,
        "k": k,
        "seed_corpus_name": dataset,
        "query_seed": query_seed,
        "queries": queries_out,
    }


def _shape_matches(existing: dict, dataset: str, dim: int, vectors_per: int, k: int, num_queries: int) -> bool:
    """Cheap sanity check that an existing ground-truth file matches the requested shape."""
    return (
        existing.get("dataset") == dataset
        and existing.get("dim") == dim
        and existing.get("vectors_per") == vectors_per
        and existing.get("k") == k
        and len(existing.get("queries", [])) == num_queries
    )


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, help="dataset name (matches seed_corpus.py's dataset arg)")
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--vectors-per", type=int, required=True, help="corpus size for this dataset")
    ap.add_argument("--num-queries", type=int, default=200, help="how many test queries to generate (default 200)")
    ap.add_argument("--k", type=int, default=10, help="top-k for ground truth (default 10)")
    ap.add_argument("--query-seed", type=int, default=1234, help="rng seed for the test queries (default 1234)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--reuse-if-exists",
        action="store_true",
        help="skip rebuild when --out already exists with a matching shape",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.reuse_if_exists and args.out.exists():
        try:
            existing = json.loads(args.out.read_text())
        except (json.JSONDecodeError, OSError):
            existing = None
        if existing is not None and _shape_matches(
            existing,
            args.dataset,
            args.dim,
            args.vectors_per,
            args.k,
            args.num_queries,
        ):
            print(
                f"  reusing existing ground truth at {args.out} "
                f"({len(existing['queries'])} queries)"
            )
            return 0
        print(f"  existing file shape mismatch — rebuilding {args.out}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    gt = compute_ground_truth(
        dataset=args.dataset,
        dim=args.dim,
        vectors_per=args.vectors_per,
        num_queries=args.num_queries,
        k=args.k,
        query_seed=args.query_seed,
    )
    args.out.write_text(json.dumps(gt))
    print(f"  wrote {len(gt['queries'])} queries to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
