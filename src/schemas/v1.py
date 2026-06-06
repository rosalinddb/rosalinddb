"""Pydantic models for the ``/v1`` wire contract.

These models describe — and ONLY describe — the JSON shapes the ``/v1/query``
endpoint already speaks. They are intentionally *lenient*: the model is the
typed front door for parsing, but the authoritative, order-sensitive semantic
validation still lives in
:func:`services.query_api.v1_query.validate_query_body`.

Why so lenient
--------------
The legacy handler validates the request body field-by-field in a
**load-bearing order** (dataset existence FIRST, then vector shape, then
dimension, then ``top_k``, then ``nprobe``, then ``filter``), and the FIRST
failure wins and maps to a specific v1 error code / HTTP status:

* missing/empty ``dataset`` and cross-tenant/missing dataset both collapse to
  ``404 dataset_not_found`` (existence is never leaked);
* a malformed ``vector`` → ``400 invalid_request``;
* a wrong-length ``vector`` → ``400 dimension_mismatch`` with
  ``details={"expected", "got"}``;
* an out-of-range ``top_k`` → ``400 top_k_out_of_range``;
* a non-positive ``nprobe`` → ``400 invalid_request``; an over-max ``nprobe``
  → ``400 nprobe_out_of_range``;
* a non-flat ``filter`` → ``400 invalid_request``.

If this model performed those checks itself (typed lengths, bounded ints,
forbidden extras, strict coercion) it would either reject inputs the current
code accepts, change the error code/status, or reorder which failure is
reported first. To stay strictly behavior-preserving, the model therefore:

* declares every field with a permissive type and round-trips the original
  Python value UNCHANGED (no coercion that ``validate_query_body`` would not
  itself perform);
* IGNORES unknown/extra fields (``extra="ignore"``) — exactly mirroring the
  legacy ``body.get(...)`` access pattern, which silently ignores unknown keys;
* leaves all range/shape/existence checks to ``validate_query_body``.

The response models (:class:`QuerySuccessResponse`,
:class:`QueryErrorResponse`) document the success and error envelopes the
handler emits. They are descriptive: the handler builds plain ``dict`` /
``JSONResponse`` bodies today, and that is unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    """Parsed ``POST /v1/query`` request body.

    Lenient by design (see module docstring): this captures the known request
    keys without enforcing the value-level rules that
    :func:`services.query_api.v1_query.validate_query_body` owns. Unknown keys
    are ignored, matching the legacy ``body.get(...)`` behaviour.

    Field types are deliberately permissive — ``vector``/``top_k``/``nprobe``/
    ``filter`` are typed as ``Any`` so Pydantic neither coerces nor rejects the
    values that ``validate_query_body`` is responsible for accepting or
    rejecting (e.g. a wrong-length vector, an out-of-range ``top_k``, a
    non-positive ``nprobe``, or a nested ``filter`` value must all reach the
    downstream validator unchanged so the correct v1 error code is produced).
    """

    # ``extra="ignore"`` mirrors the legacy ``body.get(...)`` access pattern,
    # which silently ignores unknown keys. Do NOT switch this to "forbid":
    # the current code accepts (and ignores) extra fields.
    model_config = ConfigDict(extra="ignore")

    # Required on the wire, but NOT enforced here: an absent/empty/non-string
    # ``dataset`` must reach ``validate_query_body`` so it maps to the
    # contract's ``404 dataset_not_found`` (with the exact "dataset is required"
    # message) rather than a 422 from Pydantic.
    dataset: Optional[Any] = None

    # The query vector. Typed ``Any`` so length/element validation (and the
    # ``dimension_mismatch`` / ``invalid_request`` codes) stays in
    # ``validate_query_body``.
    vector: Optional[Any] = None

    # Optional; server default (10) and the 1..1000 range check are applied by
    # ``validate_query_body``. ``Any`` so an out-of-range / wrong-type value is
    # passed through to produce ``top_k_out_of_range`` rather than a 422.
    top_k: Optional[Any] = None

    # Optional per-query IVF override; ``None``/absent → server default. Range
    # and positivity checks live downstream.
    nprobe: Optional[Any] = None

    # Optional flat AND-of-equals filter; ``None``/absent → ``{}`` downstream.
    # Scalar-only enforcement lives in ``validate_query_body``.
    filter: Optional[Any] = None


class QueryMatch(BaseModel):
    """A single similarity-search hit in a success response.

    Mirrors the per-match dict the handler builds: ``{"id", "score",
    "metadata"}`` where ``score`` is the FAISS L2² distance (ascending sort)
    and ``metadata`` defaults to ``{}`` when the sidecar carries none.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuerySuccessResponse(BaseModel):
    """The ``200`` success envelope for ``/v1/query`` (and its status poll).

    Descriptive only — the handler emits a plain dict today and that is
    unchanged. Shapes covered:

    * hot/cold/recall consolidated path: ``{matches, latency_ms, mode}`` where
      ``mode`` ∈ ``"hot" | "cold" | "recall"``;
    * ephemeral enqueue path: ``{matches, latency_ms, mode: "ephemeral",
      job_id}``.

    ``job_id`` is therefore optional (present only on the ephemeral path).
    """

    model_config = ConfigDict(extra="ignore")

    matches: List[QueryMatch] = Field(default_factory=list)
    latency_ms: Optional[int] = None
    mode: str
    job_id: Optional[str] = None


class QueryStatusResponse(BaseModel):
    """The ``GET /v1/query/status/{job_id}`` polling envelope.

    Descriptive only. Not-ready/unknown → ``{"ready": false}``; ready success
    → ``{"ready": true, "matches", "latency_ms", "mode": "ephemeral"}``. A
    runner error envelope is surfaced as a v1 error response (see
    :class:`QueryErrorResponse`), not via this model.
    """

    model_config = ConfigDict(extra="ignore")

    ready: bool
    matches: Optional[List[QueryMatch]] = None
    latency_ms: Optional[int] = None
    mode: Optional[str] = None


class ErrorDetail(BaseModel):
    """The inner ``error`` object of the v1 error envelope.

    ``details`` is present ONLY when non-null — matching the handler, which adds
    the key to the envelope only when ``details is not None``.
    """

    model_config = ConfigDict(extra="ignore")

    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class QueryErrorResponse(BaseModel):
    """The v1 error envelope: ``{"error": {"code", "message"[, "details"]}}``.

    Descriptive only — the handler builds this via
    :func:`adapters.errors.error_envelope`, which omits ``details`` when it is
    ``None``. This model documents that contract.
    """

    model_config = ConfigDict(extra="ignore")

    error: ErrorDetail
