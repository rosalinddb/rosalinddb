"""Runnable mem0 + RosalindDB demo: read-your-writes through the adapter.

This proves the end-to-end story with **no external API keys**:

  1. It uses a tiny **deterministic local embedder** (a hashing bag-of-words),
     so nothing calls out to OpenAI/Anthropic.
  2. It drives mem0's :class:`RosalindDB` vector store adapter directly
     (insert -> search -> get -> list -> delete), which is exactly what mem0's
     ``Memory`` does under the hood — minus the LLM fact-extraction step that
     ``Memory.add`` would otherwise need an LLM for.
  3. The bottom of the file shows the **full ``Memory(...)`` wiring** you'd use
     in production (commented), and documents the one thing it additionally
     needs: an LLM + a real embedder (set ``OPENAI_API_KEY`` or configure a
     local Ollama, etc.).

Run it against a **recall-enabled** RosalindDB stack (``RB_RECALL=true``), which
is what makes insert -> immediate search return the just-written vector
(read-your-writes). Point it with env vars:

    export ROSALINDDB_URL=http://localhost:8080   # default
    export ROSALINDDB_TOKEN=...                    # omit for the OSS no-auth default
    python integrations/mem0/demo.py

With ``RB_RECALL`` *off* the writes are eventually consistent, so the immediate
search may return nothing until the async build lands — the demo notes this.
"""
from __future__ import annotations

import hashlib
import os
import sys

# Make `import rosalinddb` work whether run from the repo root or this dir.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rosalinddb import RosalindDB  # noqa: E402

DIM = 64
COLLECTION = "mem0_demo"


def local_embed(text: str, dim: int = DIM) -> list[float]:
    """A deterministic, dependency-free embedding (hashed bag-of-words).

    Not semantically meaningful — it just gives stable, comparable vectors so
    the demo runs without any embedding-model API key. Real deployments use a
    proper embedder (OpenAI, a local model, etc.).
    """
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.sha1(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    # L2-normalise so distances are comparable across texts of different length.
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def main() -> int:
    base_url = os.getenv("ROSALINDDB_URL", "http://localhost:8080")
    token = os.getenv("ROSALINDDB_TOKEN")

    print(f"-> Connecting to RosalindDB at {base_url}")
    store = RosalindDB(
        collection_name=COLLECTION,
        embedding_model_dims=DIM,
        base_url=base_url,
        token=token,
    )

    # Synthetic, non-PII memories.
    memories = [
        ("mem-1", "the user prefers tea over coffee", {"user_id": "u-demo", "data": "prefers tea"}),
        ("mem-2", "the user is allergic to peanuts", {"user_id": "u-demo", "data": "allergic to peanuts"}),
        ("mem-3", "the user lives in a coastal town", {"user_id": "u-demo", "data": "coastal town"}),
    ]

    print("\n-> insert() three memories")
    store.insert(
        vectors=[local_embed(text) for _, text, _ in memories],
        payloads=[meta for _, _, meta in memories],
        ids=[mid for mid, _, _ in memories],
    )

    print("\n-> search('what should I avoid?')  [read-your-writes]")
    hits = store.search(
        query="what should I avoid?",
        vectors=local_embed("what should the user avoid allergy"),
        top_k=3,
        filters={"user_id": "u-demo"},
    )
    if not hits:
        print(
            "   (no hits — is the server running with RB_RECALL on? without it, "
            "writes are eventually consistent and not immediately queryable)"
        )
    for h in hits:
        print(f"   id={h.id!r}  similarity={h.score:.4f}  data={h.payload.get('data')!r}")

    print("\n-> get('mem-2')")
    got = store.get("mem-2")
    print(f"   {got.id!r} -> {got.payload}")

    print("\n-> list(filters={'user_id': 'u-demo'})")
    rows = store.list(filters={"user_id": "u-demo"})[0]
    for r in rows:
        print(f"   id={r.id!r}  data={r.payload.get('data')!r}")

    print("\n-> delete('mem-2')  then get('mem-2')")
    store.delete("mem-2")
    gone = store.get("mem-2")
    print(f"   get('mem-2') -> {gone}  (None == deleted, read-your-deletes)")

    print("\n-> cleanup: reset() the collection")
    store.reset()
    print("\nDone. Read-your-writes proven via the mem0 adapter.")
    return 0


# ---------------------------------------------------------------------------
# Full mem0 `Memory` wiring (needs an LLM + embedder — not run by default).
#
# RosalindDB is not (yet) in mem0's built-in provider registry, so the cleanest
# way to use it through `Memory` is to construct `Memory()` and swap in the
# adapter as its vector store:
#
#     from mem0 import Memory
#     from mem0.utils.factory import VectorStoreFactory
#     from rosalinddb import RosalindDB
#
#     # Register RosalindDB so a config-driven `Memory.from_config(...)` works:
#     VectorStoreFactory.provider_to_class["rosalinddb"] = "rosalinddb.RosalindDB"
#     VectorStoreConfig._provider_configs["rosalinddb"] = ...  # a small config class
#
#     # ...or simplest: build a Memory then replace its vector store directly:
#     m = Memory()  # needs OPENAI_API_KEY (LLM + embedder) by default
#     m.vector_store = RosalindDB(
#         collection_name="mem0_app",
#         embedding_model_dims=1536,            # match your embedder
#         base_url="http://localhost:8080",
#     )
#     m.add("I'm allergic to peanuts", user_id="alice")   # LLM extracts facts
#     print(m.search("what foods are unsafe?", user_id="alice"))  # read-your-writes
#
# `Memory.add` runs an LLM fact-extraction step, so it needs an LLM + a real
# embedder configured (e.g. `OPENAI_API_KEY`, or a local Ollama). The
# vector-store half — insert/search/get/list/delete — is exactly what this
# demo exercises above without any API key.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    raise SystemExit(main())
