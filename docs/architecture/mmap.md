# FAISS mmap for indexes larger than cache budget

> Status: source of truth for the `RB_FAISS_MMAP` flag. Pairs with
> [`architecture.md`](architecture.md), which describes the byte-budgeted
> shard cache this flag reshapes.

## Why this exists

The per-tenant default `_DEFAULT_VECTOR_QUOTA = 100000`
([`adapters/state/state.py`](../../adapters/state/state.py)) is not a product
limit. It is the largest single shard that fits the legacy
`faiss.read_index(local_path)` path against a 1 GiB cache budget.

A 1M-vector, 1536-dim shard is ~6.3 GB on disk and ~6.1 GB resident once
deserialised — it cannot enter a 1 GiB LRU without evicting itself and
everything around it, so past that threshold the cache stops working as a
cache. Mmap lifts the limit at the kernel layer: the on-disk index is mapped
into the query process's address space and only the touched IVF cells are
paged in. That decouples a shard's *on-disk* size from its *resident* size and
lets one shard exceed the cache budget by an order of magnitude without
thrashing.

## The flag contract

`RB_FAISS_MMAP` — environment variable, read at `query_dp` module import
time. Default: `false`.

- **`false`** (default): legacy `faiss.read_index(local_path)` — fully
  deserialises the index into the Python process. This is the
  current-behaviour-preserving default.
- **`true`**: `faiss.read_index(local_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)`.
  The kernel page-caches the on-disk index; only touched IVF cells stay
  resident.

Invariants under both settings:

- Query API surface is identical (request/response shapes, status codes).
- Top-K results are bit-identical for the same shard + same query + same
  `nprobe`. Verified by `tests/integration/test_mmap_query_parity.py`.
- On-disk shard format is unchanged.
- Builder is unchanged.
- Catalog schema is unchanged.

What `RB_FAISS_MMAP=true` does NOT do:

- Does not change the per-tenant vector quota (`_DEFAULT_VECTOR_QUOTA`
  stays 100,000; raising it is a configuration decision).
- Does not enable mmap for `IndexFlat` or `HNSW` — only IVF-family
  indexes support `IO_FLAG_MMAP` in upstream FAISS today.
- Does not change the byte-budgeted shard cache logic — the cache still
  exists and still budgets sidecars; the FAISS object's contribution to
  the budget drops to a small fixed estimate.

## How it works

`faiss.read_index(path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)` opens
the on-disk index file via `mmap(2)` instead of slurping it into RSS. The flag
replaces the in-RAM `InvertedLists` object with an `OnDiskInvertedLists` that
keeps a small offset table in RAM (one entry per IVF cell) and lets the kernel
fault posting-list pages into the page cache on first access. The flag is
threaded through exactly one call site — the cold-load branch in `_hot_search`
in [`services/query_api/v1_query.py`](../../services/query_api/v1_query.py) and
its mirror in [`services/ephemeral_runner/run.py`](../../services/ephemeral_runner/run.py)
— and is gated on `RB_FAISS_MMAP` so a rollback is one env-var flip.

The cold-load branch that precedes `read_index` — `_ensure_cached` in the same
file — single-flights concurrent downloads of the same shard URI through a
per-URI `threading.Event`, so a burst of N queries against an uncached shard
issues exactly one object-store GET and N−1 waiters reuse the resulting local
path. The coalescing applies whether mmap is on or off; under mmap it is
load-bearing because the cold path now opens a multi-GB file that nothing
else in the process is prepared to redundantly fetch.

The byte-budgeted cache accounts for an mmap'd entry with the
`_MMAP_INDEX_ESTIMATE_BYTES = 32 MiB` constant in `v1_query.py`. That number
is deliberately conservative — large enough that an unbounded number of mmap'd
entries still pressures the LRU toward eviction, small enough that the
default 1 GiB cache holds many warm shards. Tune it in one place when
production traces give us a better number.

The shape of an mmap'd FAISS object inside the query process is small in
RAM but addressable as if it were fully loaded. `OnDiskInvertedLists` holds
the per-cell offset table; the IVF centroid quantizer (one `IndexFlat` over
the centroids) stays in normal RAM because it is scanned in full on every
query; the `IDMap2` id table is mapped alongside the inverted lists. On a
query, FAISS computes the `nprobe` nearest centroids in RAM, dereferences the
offset table for each chosen cell, and the kernel faults the 4 KiB pages
covering those cell payloads into the page cache. Hot cells stay resident;
cold cells get evicted under memory pressure exactly like any other
file-backed cache page.

Cgroup accounting is the subtle part. Under cgroup v2, file-backed page-cache
pages are charged to the cgroup that *first touched them*. Container runtimes
report those pages as part of the container's `memory.current`, and the OOM
killer respects the cgroup limit — but the kernel evicts clean file-backed
pages first under pressure, so a 6 GB shard mapped into a 2 GB container does
not OOM as long as the *actually-touched* working set fits.
The corollary: `memory.current` will hover near the limit because the kernel
happily uses every free byte for cache. That is expected and is not a leak.

## The working-set model

The numbers below are for a 1M × 1536-dim IVFFlat shard — the size that
motivates the change.

On-disk shape:

| Component | Size |
|---|---|
| Raw vectors (1,000,000 × 1536 × 4 B) | **6.144 GB** |
| Centroid quantizer (4096 × 1536 × 4 B) | **24 MB** |
| IDMap2 id table (1M × 8 B) | **8 MB** |
| Sidecar JSON (~120 B/vector) | **~120 MB** |
| Total on disk | **~6.3 GB** |

What gets paged in on a cold first query at `nprobe=64`, `top_k=10`:

- Centroid scan runs over the 24 MB quantizer already in RAM (one in-process
  pass, ~1 ms on a modern CPU).
- 64 of 4096 IVF cells are probed; at an even distribution that is
  ~244 vectors per cell, ~15,616 vectors total.
- Inverted-list payload paged in: 15,616 × 1536 × 4 B = **~96 MB**.
- Id-table pages paged in: 15,616 × 8 B = **~125 KB**.
- At local-SSD throughput (1–3 GB/s sequential) the cold page-in cost is
  ~30–100 ms on top of the FAISS search itself.

Warm steady-state, single dataset under realistic query diversity:

- Centroid quantizer: 24 MB (always resident).
- Resident inverted-list pages: roughly the cold-query footprint scaled by
  the Zipfian top of the cell-hit distribution — call it 2× the single-query
  payload.
- **~200 MB of resident page cache per dataset** comfortably handles the
  steady-state load. A 2 GB DP container holds several datasets co-resident
  with margin.

The same shard under the legacy `read_index` path requires ~6.1 GB of RSS
to even be admitted to the cache. The mmap'd version's *working set* is what
matters, not the total file size, and the kernel's page-cache LRU manages
the eviction policy automatically across queries.

## Operator notes

### `memory.current` will sit near the cgroup limit

Under load with `RB_FAISS_MMAP=true`, `docker stats` and any cgroup-aware
metrics tool will show the container's `memory.current` hovering close to
the cgroup limit.
This is the kernel using every otherwise-free byte for the file-backed page
cache, not a leak. The OOM killer respects the cgroup limit and evicts clean
file-backed pages first under pressure, so memory utilisation reported as
"99%" with mmap on is the steady state, not an incident. The signal to
watch for an actual problem is sustained anonymous-RSS growth (the Python
heap), not file-backed cache usage. Both are visible in
`memory.stat` — `file` vs `anon`.

### Never back `CACHE_DIR` with a FUSE object-store mount

`mmap(2)` against `s3fs`, `goofys`, `mountpoint-s3`, or any other FUSE-mounted
object store is unsafe. The page-fault path can return stale or partial data,
the offset table is read once and assumed stable, and a FUSE-side error during
a fault surfaces as `SIGBUS` inside the query process. `CACHE_DIR` must be a
real local filesystem (ext4, xfs, tmpfs, or the container overlayfs). A
startup guardrail logs a `WARNING` if the cache directory's filesystem type
matches a known FUSE pattern; treat the warning as a deployment bug.

### Page-fault counter for cold/warm visibility

A `rosalinddb.shard.page_faults` counter is sourced from the delta in the
process's major-fault count around each `faiss.search` call.
Chart the cold/warm ratio in Prometheus to monitor mmap behaviour under
load — a sudden rise in major faults per query is the signal that the
working set has outgrown available RAM and the kernel is re-faulting pages
on every query. The right response is more RAM on the DP container, not a
flag flip; turning mmap off would convert the same pressure into hard
admit-then-evict cache churn against the byte budget, which is strictly
worse.

## Index-type coverage caveat

Mmap support in upstream FAISS is not uniform across index types. The current
state of play:

- `IVFFlat`, `IVFPQ`, `IVFScalarQuantizer` are mmap-clean against the
  standard FAISS file format. `IndexIDMap2` wrapping any of these is fine —
  the wrapper's small id table is mapped alongside.
- `IndexFlat` and `HNSW` are **not** mmap-clean. See
  [FAISS issue #4101](https://github.com/facebookresearch/faiss/issues/4101)
  for the long-running RFC adding `IndexFlatCodes` and `HNSW` mmap support;
  it has been open and stale for months.

RosalindDB's small-dataset fallback (`build_flat` in
[`services/index_builder/run.py`](../../services/index_builder/run.py))
builds `IndexFlatL2` for datasets below the IVF training floor. Those
datasets are tiny by design (≤64 vectors) and fit in RAM trivially, so the
mmap flag has no observable effect on them — they take the same byte-budget
admit path under both flag settings. Do not claim "mmap everywhere"
externally: the honest framing is "mmap for IVF-family indexes, which is
every shard the builder produces above the training floor."

## References

- [FAISS wiki: Indexes that do not fit in RAM](https://github.com/facebookresearch/faiss/wiki/Indexes-that-do-not-fit-in-RAM)
- [FAISS issue #4101: memory-mapped faiss indices](https://github.com/facebookresearch/faiss/issues/4101)
- [FAISS issue #3165: mmap for IndexFlat in multi-process usage](https://github.com/facebookresearch/faiss/issues/3165)
- [cgroup v2 memory controller documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html)
- [`docs/architecture/architecture.md`](architecture.md) — the broader RosalindDB
  architecture this flag fits into; covers the byte-budgeted shard cache, the
  CP/DP split, and the object-storage-first design the mmap path preserves.
