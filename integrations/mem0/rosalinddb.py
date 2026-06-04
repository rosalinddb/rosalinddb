"""A first-party mem0 ``VectorStoreBase`` adapter backed by RosalindDB.

This maps mem0 2.0.4's vector-store interface onto the RosalindDB v1 REST API
(via :class:`rosalinddb_client.RosalindDBClient`). One mem0 *collection* maps to
exactly one RosalindDB *dataset*.

Two caveats are load-bearing and called out in the module-level docstring,
:meth:`search`, and ``integrations/mem0/README.md``:

  1. **L2-squared distance -> similarity.** RosalindDB's ``/v1/query`` returns a
     raw FAISS L2-squared distance (``score``, *lower is closer*). mem0 expects a
     *similarity* where *higher is better*. This adapter converts each distance
     ``d`` to ``1 / (1 + d)``: a strictly decreasing function of ``d``, so the
     ordering is preserved (smaller distance -> higher similarity), the exact
     match ``d == 0`` maps to the maximum ``1.0``, and the value is bounded in
     ``(0, 1]``. We do *not* claim it is a cosine similarity.

  2. **Filters are exhaustive server-side, and read-your-writes needs RB_RECALL.**
     mem0's flat ``user_id`` / ``agent_id`` / ``run_id`` filters pass straight
     through to RosalindDB's ``filter`` (exact AND-of-equals). A *filtered*
     ``/v1/query`` is run **exhaustively** server-side (every IVF cell scanned),
     which is exact but O(n) on the dataset. For strict per-tenant isolation and
     to keep filtered queries cheap, prefer a **dataset-per-tenant** layout
     (one ``RosalindDB`` instance / collection per ``user_id``) over a single
     shared collection with a ``user_id`` filter. Separately, **immediate
     read-your-writes** (insert then search returns it now) requires the server
     to run with ``RB_RECALL`` on; with the flag off, writes are eventually
     consistent and a just-inserted vector is not queryable until the async
     build lands.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from pydantic import BaseModel

try:  # Allow ``import rosalinddb`` both as a package and as a flat module.
    from .rosalinddb_client import (
        RosalindDBClient,
        VectorNotFoundError,
    )
except ImportError:  # pragma: no cover - flat-module / sys.path import
    from rosalinddb_client import (  # type: ignore
        RosalindDBClient,
        VectorNotFoundError,
    )

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    """The result shape mem0 expects from ``get`` / ``search`` / ``list``.

    Mirrors ``mem0.vector_stores.pgvector.OutputData`` exactly: mem0's
    ``Memory`` only ever reads ``.id``, ``.score`` and ``.payload`` off these.
    """

    id: Optional[str] = None
    score: Optional[float] = None
    payload: Optional[dict] = None


def l2_squared_to_similarity(distance: float) -> float:
    """Convert an L2-squared distance to a higher-is-better similarity.

    Uses ``1 / (1 + d)`` — a strictly decreasing function of ``d`` (so ranking
    by descending similarity equals ranking by ascending distance), mapping the
    exact match ``d == 0`` to ``1.0`` and large distances toward ``0``.
    """
    return 1.0 / (1.0 + float(distance))


class RosalindDB(VectorStoreBase):
    """mem0 vector store backed by a RosalindDB dataset (= mem0 collection).

    Args:
        collection_name: The RosalindDB dataset name (= the mem0 collection).
            Must match ``[a-z0-9_-]+``, 1-64 chars.
        embedding_model_dims: The embedding dimension; bound to the dataset on
            create.
        base_url: The RosalindDB Control Plane origin.
        token: Optional bearer token (JWT or ``rb_live_...`` API key). Omit when
            the server runs with the OSS no-auth default.
        client: An existing :class:`RosalindDBClient` (overrides ``base_url`` /
            ``token``); handy for tests with a mocked client.
        timeout: Per-request timeout in seconds (when constructing a client).
    """

    def __init__(
        self,
        collection_name: str,
        embedding_model_dims: int,
        base_url: str = "http://localhost:8080",
        token: Optional[str] = None,
        client: Optional[RosalindDBClient] = None,
        timeout: float = 30.0,
    ):
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.client = client or RosalindDBClient(
            base_url=base_url, token=token, timeout=timeout
        )
        self.create_col(collection_name, embedding_model_dims)

    # -- collections --------------------------------------------------------

    def create_col(self, name, vector_size, distance="l2"):
        """Create the backing dataset (idempotent on ``dataset_exists``).

        ``distance`` is accepted for interface compatibility but **ignored** —
        RosalindDB v1 is L2-squared only. A pre-existing dataset is treated as a
        success (mem0 calls ``create_col`` on every init).
        """
        if distance not in (None, "l2", "L2", "euclidean"):
            logger.warning(
                "RosalindDB only supports L2 distance; ignoring requested "
                "distance=%r for collection %r",
                distance,
                name,
            )
        try:
            self.client.create_dataset(name, vector_size)
        except Exception as exc:  # dataset_exists -> idempotent no-op
            code = getattr(exc, "code", None)
            if code == "dataset_exists":
                logger.debug("Dataset %r already exists; reusing it.", name)
                return
            raise

    def list_cols(self):
        """List collection (dataset) names."""
        return [d["name"] for d in self.client.list_datasets()]

    def delete_col(self):
        """Delete the backing dataset."""
        self.client.delete_dataset(self.collection_name)

    def col_info(self):
        """Return the dataset's metadata (name, dimension, status, row_count)."""
        return self.client.get_dataset(self.collection_name)

    def reset(self):
        """Delete and recreate the collection (dataset)."""
        logger.warning("Resetting collection %s ...", self.collection_name)
        try:
            self.delete_col()
        except Exception as exc:
            if getattr(exc, "code", None) != "dataset_not_found":
                raise
        self.create_col(self.collection_name, self.embedding_model_dims)

    # -- writes -------------------------------------------------------------

    def insert(self, vectors, payloads=None, ids=None):
        """Upsert vectors via the NDJSON endpoint (last-write-wins on id)."""
        records = []
        for idx, vector in enumerate(vectors):
            payload = payloads[idx] if payloads else {}
            vector_id = str(ids[idx]) if ids else str(idx)
            records.append(
                {
                    "id": vector_id,
                    "values": list(vector),
                    "metadata": payload or {},
                }
            )
        if records:
            self.client.upsert(self.collection_name, records)

    def update(self, vector_id, vector=None, payload=None):
        """Update = re-upsert the id (last-write-wins).

        RosalindDB ``POST .../vectors`` is an upsert, so an update is a single
        re-upsert of the full record. Because the endpoint replaces the whole
        record, a partial update needs the current value/metadata: any field
        left ``None`` is backfilled from the existing row's metadata. ``vector``
        cannot be read back (values are not returned in v1), so a metadata-only
        update reuses a zero placeholder vector ONLY if the row is absent;
        otherwise callers should pass ``vector`` to avoid clobbering it.
        """
        record: dict = {"id": str(vector_id)}
        if payload is None:
            # Preserve existing metadata when only the vector changes.
            try:
                existing = self.client.get(self.collection_name, str(vector_id))
                payload = existing.get("metadata", {})
            except VectorNotFoundError:
                payload = {}
        record["metadata"] = payload or {}
        record["values"] = (
            list(vector) if vector is not None else [0.0] * self.embedding_model_dims
        )
        self.client.upsert(self.collection_name, [record])

    def delete(self, vector_id):
        """Delete one vector by id (no-op if absent)."""
        self.client.delete(self.collection_name, str(vector_id))

    # -- reads --------------------------------------------------------------

    def get(self, vector_id):
        """Retrieve one vector's id + metadata, or ``None`` if absent."""
        try:
            row = self.client.get(self.collection_name, str(vector_id))
        except VectorNotFoundError:
            return None
        return OutputData(id=row["id"], score=None, payload=row.get("metadata") or {})

    def search(self, query, vectors, top_k=5, filters=None):
        """Search for similar vectors; returns a list of :class:`OutputData`.

        The server returns L2-squared distances (lower = closer); each is
        converted to a higher-is-better similarity via
        :func:`l2_squared_to_similarity`. ``filters`` (mem0's flat
        ``user_id`` / ``agent_id`` / ``run_id``) pass straight through as the
        v1 ``filter`` (exact AND-of-equals). See the module docstring for the
        exhaustive-filter and read-your-writes caveats.
        """
        result = self.client.query(
            self.collection_name,
            list(vectors),
            top_k=top_k,
            filter=self._normalize_filters(filters),
        )
        out = []
        for match in result.get("matches", []):
            out.append(
                OutputData(
                    id=match["id"],
                    score=l2_squared_to_similarity(match.get("score", 0.0)),
                    payload=match.get("metadata") or {},
                )
            )
        return out

    def list(self, filters=None, top_k=100):
        """List vectors (filtered); returns ``[[OutputData, ...]]``.

        mem0 expects the list wrapped in an outer list (it unwraps one level via
        ``list(...)[0]``), matching the pgvector provider's return shape.
        """
        response = self.client.list(
            self.collection_name,
            filter=self._normalize_filters(filters),
            limit=top_k,
        )
        rows = [
            OutputData(id=v["id"], score=None, payload=v.get("metadata") or {})
            for v in response.get("vectors", [])
        ]
        return [rows]

    def keyword_search(self, query, top_k=5, filters=None):
        """Not supported — RosalindDB v1 has no BM25 / full-text index."""
        raise NotImplementedError(
            "RosalindDB has no keyword/BM25 search; use search() (vector search)."
        )

    # search_batch is inherited from VectorStoreBase (loops over search()).

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _normalize_filters(filters):
        """Pass mem0's flat filters straight through, dropping ``None`` values.

        mem0 hands us a flat dict of ``field -> value`` (``user_id`` etc.).
        RosalindDB's ``filter`` is exact AND-of-equals over the same flat shape,
        so this is a near-identity map; we only strip ``None`` values (a ``null``
        filter value never matches anything in v1) and return ``None`` for an
        empty filter so no filtering is applied.
        """
        if not filters:
            return None
        clean = {k: v for k, v in filters.items() if v is not None}
        return clean or None


# Optional convenience: a self-hoster can point RB_RECALL etc. at a stack and
# build the adapter straight from env. Not required by mem0.
def from_env(collection_name: str, embedding_model_dims: int) -> RosalindDB:
    """Build a :class:`RosalindDB` from ``ROSALINDDB_URL`` / ``ROSALINDDB_TOKEN``."""
    return RosalindDB(
        collection_name=collection_name,
        embedding_model_dims=embedding_model_dims,
        base_url=os.getenv("ROSALINDDB_URL", "http://localhost:8080"),
        token=os.getenv("ROSALINDDB_TOKEN"),
    )
