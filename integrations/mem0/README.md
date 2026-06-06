# RosalindDB ↔ mem0 integration

A first-party [mem0](https://github.com/mem0ai/mem0) `VectorStoreBase` adapter
backed by RosalindDB, plus a thin REST client and a runnable demo. It lets you
use RosalindDB as the vector store behind mem0's agent-memory layer and get
**read-your-writes** (store a fact, retrieve it on the very next turn) when the
server runs with the recall tier on.

## What's here

| File | Purpose |
|---|---|
| `rosalinddb_client.py` | Dependency-light REST client for the [v1 API](../../docs/api/v1.md) (uses `requests`, falls back to stdlib `urllib`). |
| `rosalinddb.py` | `class RosalindDB(VectorStoreBase)` — the mem0 2.0.4 adapter. |
| `demo.py` | Runnable, no-API-key demo proving insert → search → get → list → delete (read-your-writes). |
| `requirements.txt` | `mem0ai==2.0.4` as an **optional** dependency (not in the repo's core requirements). |

## Install

```bash
pip install mem0ai==2.0.4          # or: pip install -r integrations/mem0/requirements.txt
# RosalindDB's `requests` core dep already covers the REST client.
```

Then either import the adapter directly, or copy `rosalinddb.py` +
`rosalinddb_client.py` into your project.

## Quickstart

```python
from rosalinddb import RosalindDB

store = RosalindDB(
    collection_name="mem0_app",      # = one RosalindDB dataset
    embedding_model_dims=1536,        # match your embedder
    base_url="http://localhost:8080", # the Control Plane origin
    token=None,                       # a JWT / rb_live_... key, or None for OSS no-auth
)

store.insert(
    vectors=[[...]],                  # your embeddings
    payloads=[{"user_id": "alice", "data": "allergic to peanuts"}],
    ids=["mem-1"],
)
hits = store.search(query="what to avoid?", vectors=[...], top_k=5,
                    filters={"user_id": "alice"})
for h in hits:
    print(h.id, h.score, h.payload)   # score is higher-is-better (see below)
```

### Using it through mem0's `Memory`

RosalindDB is not (yet) in mem0's built-in provider registry, so the simplest
wiring is to build a `Memory()` and swap in the adapter as its vector store:

```python
from mem0 import Memory
from rosalinddb import RosalindDB

m = Memory()                          # Memory.add needs an LLM + embedder (e.g. OPENAI_API_KEY)
m.vector_store = RosalindDB(
    collection_name="mem0_app",
    embedding_model_dims=1536,
    base_url="http://localhost:8080",
)
m.add("I'm allergic to peanuts", user_id="alice")     # LLM extracts the fact
print(m.search("what foods are unsafe?", user_id="alice"))
```

`Memory.add` runs an LLM fact-extraction step, so it needs an LLM + a real
embedder configured. The vector-store half (insert/search/get/list/delete) needs
neither — that's what `demo.py` exercises with a local hashing embedder and **no
API keys**:

```bash
export ROSALINDDB_URL=http://localhost:8080   # default
# export ROSALINDDB_TOKEN=...                  # omit for the OSS no-auth default
python integrations/mem0/demo.py
```

## The `VectorStoreBase` → RosalindDB mapping

All 13 methods mem0 2.0.4 requires:

| mem0 method | RosalindDB endpoint / behaviour |
|---|---|
| `create_col(name, vector_size, distance)` | `POST /v1/datasets` `{name, dimension}` — `distance` **ignored** (L2-only); `dataset_exists` is an idempotent no-op |
| `insert(vectors, payloads, ids)` | `POST /v1/datasets/{name}/vectors` (NDJSON upsert, last-write-wins) |
| `update(vector_id, vector, payload)` | re-upsert via `POST .../vectors` (last-write-wins); omitted `payload` is backfilled. **A metadata-only update (`vector=None`) preserves the stored embedding** by reading it back via `GET ...?include_values=true` — it never writes a placeholder. See the v1 limitation below. |
| `delete(vector_id)` | `DELETE /v1/datasets/{name}/vectors/{id}` |
| `get(vector_id)` | `GET /v1/datasets/{name}/vectors/{id}` → `OutputData` (or `None` on 404) |
| `search(query, vectors, top_k, filters)` | `POST /v1/query` → L2² distance **converted to similarity** → `list[OutputData]` |
| `list(filters, top_k)` | `GET /v1/datasets/{name}/vectors` (filter + `limit`) → `[[OutputData, ...]]` |
| `list_cols()` | `GET /v1/datasets` → list of names |
| `col_info()` | `GET /v1/datasets/{name}` → the dataset row |
| `delete_col()` | `DELETE /v1/datasets/{name}` |
| `reset()` | `delete_col()` then `create_col()` |
| `search_batch(queries, vectors_list, ...)` | inherited default — loops over `search()` |
| `keyword_search(query, ...)` | returns **`None`** — RosalindDB v1 has no BM25 / full-text index. (`Memory.search` calls this unconditionally and guards `if result is not None`, so it MUST return `None`, not raise — see the [base contract](https://github.com/mem0ai/mem0).) |

`OutputData` mirrors mem0's pgvector provider exactly: `id`, `score`, `payload`
(mem0 reads only those three). `search`/`get` return `OutputData`; `list`
returns it **double-wrapped** (`[[...]]`) because mem0 unwraps one level
(`list(...)[0]`).

## L2² distance → similarity

RosalindDB's `POST /v1/query` returns the raw **FAISS L2-squared distance** as
`score`, where **lower is closer** (`0.0` is an exact match). mem0 expects a
**similarity** where *higher is better*. The adapter converts each distance `d`:

```
similarity = 1 / (1 + d)
```

This is strictly decreasing in `d`, so the ranking is preserved (nearer →
higher), the exact match `d == 0` maps to the maximum `1.0`, and the value is
bounded in `(0, 1]`. It is **not** a cosine similarity — just a monotonic
distance-to-similarity transform.

> [!IMPORTANT]
> **The similarity scale is NOT cosine — mem0's hard-coded thresholds behave
> differently.** mem0 compares this `score` against thresholds that were tuned
> for a **cosine** store (where similarity ∈ `[-1, 1]`), e.g. the memory **dedup**
> gate (`>= 0.95`) and the search-result **gate** (default `0.1`). Here `score =
> 1/(1+d)` with `d = L2² (squared Euclidean) distance`, so the same numeric
> threshold means something different — two embeddings `0.32` apart in L2² already
> fall to `score ≈ 0.95`, so cosine-tuned cut-offs will over- or under-trigger.
>
> Mitigations:
> - **Normalize your embeddings to unit length.** For unit vectors,
>   `L2² = 2·(1 − cos)`, so `d` is *monotonic* in cosine distance and the ranking
>   (and `1/(1+d)`) tracks cosine ranking — the closest practical fix.
> - **Tune the mem0 thresholds** (the dedup `0.95` and the search-gate `0.1`) to
>   this `1/(1+d)` scale for your embedder / dimension if you can't normalize.

## Caveats (read these)

### 1. Filtered queries are exhaustive server-side — prefer dataset-per-tenant

mem0's flat `user_id` / `agent_id` / `run_id` filters pass straight through to
RosalindDB's `filter` (exact **AND-of-equals**). But a *filtered* `POST /v1/query`
is run **exhaustively** server-side — every IVF cell is scanned and the predicate
applied to every candidate. It's exact, but **O(n)** in the dataset size.

For multi-tenant agent memory, the recommended layout is therefore
**one dataset (collection) per tenant** — e.g. a `RosalindDB(collection_name=user_id, ...)`
per `user_id` — rather than one shared collection filtered by `user_id`. That
keeps each query's scan bounded to the tenant's own vectors and gives hard
isolation, at the cost of more datasets. The single-collection + filter layout
is fine for small/low-cardinality cases.

### 2. Read-your-writes requires `RB_RECALL` on

Immediate read-your-writes (insert, then `search` returns it *now*) requires the
server to run with the **recall tier enabled** (`RB_RECALL=true` +
`RB_RECALL_DSN` set — see
[`docs/architecture/recall-consolidate.md`](../../docs/architecture/recall-consolidate.md)).
With the flag on, `POST .../vectors` is synchronous (HTTP 200) and the vector is
immediately queryable; `DELETE` is a synchronous tombstone (HTTP 204,
read-your-deletes).

With `RB_RECALL` **off** (the default), writes are **eventually consistent**:
`POST .../vectors` returns 202 and an async build folds the vector into a shard
later, so a just-inserted vector is not queryable until the build lands. The
client's `poll_until_indexed(name)` helper can wait for that transition, but you
do **not** get turn-to-turn read-your-writes in that mode.

### 3. Metadata-only `update` only works for recall-resident vectors (v1)

A metadata-only `update(vector_id, vector=None, payload=...)` preserves the
stored embedding by reading it back via `GET ...?include_values=true` — so it
**never** clobbers the vector with a placeholder. But `include_values` can only
return the embedding for a **recall-resident** vector (a plain column read; no
FAISS). For a **consolidated / cold-only** vector the embedding is not yet
readable (cold FAISS `reconstruct` is a deferred follow-up), so rather than
corrupt it the adapter raises:

```
ValueError: metadata-only update for a consolidated vector requires passing
vector=...; cold reconstruct is a future include_values follow-up
```

Pass `vector=...` explicitly to update a consolidated vector. In practice mem0's
`Memory` always passes `vector`, so this only affects direct/vendored callers
that edit metadata in place. (Until a vector has been consolidated — i.e. while
it still lives in the recall tier — metadata-only updates work transparently.)
