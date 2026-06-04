"""A thin, dependency-light REST client for the RosalindDB v1 HTTP API.

This client speaks the exact v1 contract documented in ``docs/api/v1.md``,
``docs/api/vectors.md`` and ``docs/api/query.md``. It is deliberately small and
has no hard third-party dependency: it uses ``requests`` when it is importable
(it is a RosalindDB core dependency) and otherwise falls back to the standard
library ``urllib``.

Endpoints covered:

  - ``POST   /v1/datasets``                         -> :meth:`create_dataset`
  - ``GET    /v1/datasets``                         -> :meth:`list_datasets`
  - ``GET    /v1/datasets/{name}``                  -> :meth:`get_dataset`
  - ``DELETE /v1/datasets/{name}``                  -> :meth:`delete_dataset`
  - ``POST   /v1/datasets/{name}/vectors``          -> :meth:`upsert` (NDJSON)
  - ``GET    /v1/datasets/{name}/vectors/{id}``     -> :meth:`get`
    (``?include_values=true`` returns a recall-resident vector's ``embedding``)
  - ``GET    /v1/datasets/{name}/vectors``          -> :meth:`list`
  - ``DELETE /v1/datasets/{name}/vectors/{id}``     -> :meth:`delete`
  - ``POST   /v1/query`` (+ ephemeral status poll)  -> :meth:`query`
  - ``GET    /v1/query/status/{job_id}``            -> internal poll loop

The v1 error envelope ``{"error": {"code", "message", "details"}}`` is mapped
onto :class:`RosalindDBError` subclasses so callers can branch on a stable
``.code`` rather than parsing HTTP status text.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable, Optional

try:  # `requests` is a RosalindDB core dependency; use it when available.
    import requests  # type: ignore

    _HAVE_REQUESTS = True
except ImportError:  # pragma: no cover - stdlib fallback path
    requests = None  # type: ignore
    _HAVE_REQUESTS = False


__all__ = [
    "RosalindDBClient",
    "RosalindDBError",
    "DatasetNotFoundError",
    "VectorNotFoundError",
    "QueryTimeoutError",
    "TransportError",
]


class RosalindDBError(Exception):
    """A v1 API error.

    Carries the parsed ``{code, message, details}`` envelope plus the HTTP
    status code so callers can branch on a stable, transport-independent
    ``.code``.
    """

    def __init__(
        self,
        code: str,
        message: str,
        status: Optional[int] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.status = status
        self.details = details or {}


class DatasetNotFoundError(RosalindDBError):
    """Raised on ``404 dataset_not_found`` (missing or cross-tenant dataset)."""


class VectorNotFoundError(RosalindDBError):
    """Raised on ``404 not_found`` for a get/delete of an absent vector id."""


class QueryTimeoutError(RosalindDBError):
    """Raised when an ephemeral query never becomes ready within the timeout."""

    def __init__(self, job_id: str, timeout: float):
        super().__init__(
            "query_timeout",
            f"ephemeral query {job_id} did not complete within {timeout}s",
        )
        self.job_id = job_id


class TransportError(RosalindDBError):
    """A transport-layer failure (connection refused, DNS, socket timeout, ...).

    The request never reached a v1 error envelope — the server was unreachable or
    the socket timed out — so there is no ``status``. Carried under the
    :class:`RosalindDBError` hierarchy (``code == "transport_error"``) so callers
    can catch every client failure uniformly. The original exception is chained
    via ``__cause__``.
    """

    def __init__(self, message: str):
        super().__init__("transport_error", message)


def _raise_for_envelope(status: int, body: bytes) -> None:
    """Map a 4xx/5xx response body's v1 error envelope onto an exception."""
    code = "internal_error"
    message = ""
    details: dict = {}
    try:
        parsed = json.loads(body.decode("utf-8")) if body else {}
        err = parsed.get("error") or {}
        code = err.get("code") or code
        message = err.get("message") or ""
        details = err.get("details") or {}
    except (ValueError, AttributeError):
        message = body.decode("utf-8", "replace") if body else ""

    if code == "dataset_not_found":
        raise DatasetNotFoundError(code, message, status, details)
    if code == "not_found":
        raise VectorNotFoundError(code, message, status, details)
    raise RosalindDBError(code, message, status, details)


class _Response:
    """A minimal, transport-agnostic response wrapper (status + raw bytes)."""

    __slots__ = ("status", "content")

    def __init__(self, status: int, content: bytes):
        self.status = status
        self.content = content

    def json(self) -> Any:
        if not self.content:
            return None
        return json.loads(self.content.decode("utf-8"))


class RosalindDBClient:
    """A thin REST client for one RosalindDB deployment.

    Args:
        base_url: The Control Plane origin, e.g. ``http://localhost:8080``.
        token: Optional bearer token (a JWT or an ``rb_live_...`` API key). When
            the deployment runs with the OSS default (no auth) this can be left
            unset — the ``Authorization`` header is then ignored anyway.
        timeout: Per-request socket timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        token: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._session = requests.Session() if _HAVE_REQUESTS else None

    # -- low-level transport ------------------------------------------------

    def _headers(self, content_type: Optional[str] = None) -> dict:
        headers = {"Accept": "application/json"}
        if content_type:
            headers["Content-Type"] = content_type
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        data: Optional[bytes] = None,
        content_type: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> _Response:
        url = self.base_url + path
        if params:
            # Drop None values so an unset cursor/filter is simply omitted.
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                url = url + "?" + urllib.parse.urlencode(clean)

        body: Optional[bytes] = data
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            content_type = content_type or "application/json"

        if _HAVE_REQUESTS:
            try:
                resp = self._session.request(  # type: ignore[union-attr]
                    method,
                    url,
                    data=body,
                    headers=self._headers(content_type),
                    timeout=self.timeout,
                )
            except requests.exceptions.RequestException as exc:  # type: ignore[union-attr]
                # Connection refused / DNS / socket timeout / etc. — the request
                # never reached a v1 envelope. Wrap into the typed hierarchy.
                raise TransportError(f"{method} {url} failed: {exc}") from exc
            response = _Response(resp.status_code, resp.content)
        else:  # pragma: no cover - exercised only without `requests`
            req = urllib.request.Request(
                url, data=body, headers=self._headers(content_type), method=method
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as raw:
                    response = _Response(raw.status, raw.read())
            except urllib.error.HTTPError as exc:
                response = _Response(exc.code, exc.read())
            except (urllib.error.URLError, OSError) as exc:
                # URLError wraps connection refused/DNS; OSError covers socket
                # timeouts. Same typed wrapping as the `requests` path.
                raise TransportError(f"{method} {url} failed: {exc}") from exc

        if response.status >= 400:
            _raise_for_envelope(response.status, response.content)
        return response

    # -- datasets -----------------------------------------------------------

    def create_dataset(self, name: str, dimension: int) -> dict:
        """``POST /v1/datasets`` — create an empty dataset. Returns the row."""
        return self._request(
            "POST", "/v1/datasets", json_body={"name": name, "dimension": dimension}
        ).json()

    def list_datasets(self) -> list[dict]:
        """``GET /v1/datasets`` — list the caller's datasets."""
        return self._request("GET", "/v1/datasets").json().get("datasets", [])

    def get_dataset(self, name: str) -> dict:
        """``GET /v1/datasets/{name}`` — one dataset's metadata + status."""
        return self._request("GET", f"/v1/datasets/{name}").json()

    def delete_dataset(self, name: str) -> None:
        """``DELETE /v1/datasets/{name}`` — soft-delete a dataset (204)."""
        self._request("DELETE", f"/v1/datasets/{name}")

    # -- vectors ------------------------------------------------------------

    def upsert(self, name: str, records: Iterable[dict]) -> dict:
        """``POST /v1/datasets/{name}/vectors`` — NDJSON upsert (last-write-wins).

        ``records`` is an iterable of ``{"id", "values", "metadata"}`` dicts.
        Returns ``{accepted, rejected, errors[, job_id]}``. With ``RB_RECALL``
        on the write is synchronous (HTTP 200, no ``job_id``); off it is queued
        (HTTP 202, with ``job_id``). The body shape is identical either way.
        """
        ndjson = "\n".join(json.dumps(rec) for rec in records).encode("utf-8")
        return self._request(
            "POST",
            f"/v1/datasets/{name}/vectors",
            data=ndjson,
            content_type="application/x-ndjson",
        ).json()

    def get(self, name: str, vector_id: str, include_values: bool = False) -> dict:
        """``GET /v1/datasets/{name}/vectors/{id}`` — id + metadata.

        Returns ``{"id", "metadata"}``. With ``include_values=True``
        (``?include_values=true``) a **recall-resident** vector additionally
        carries its stored ``"embedding"`` (a ``list[float]``); a consolidated
        (cold-only) vector OMITS ``"embedding"`` (the cold FAISS ``reconstruct``
        is a deferred follow-up), so callers must treat its absence as "not
        recall-resident". This backs the adapter's metadata-only ``update``,
        which must re-upsert without clobbering the real embedding.

        Raises :class:`VectorNotFoundError` on a ``404 not_found`` (absent id or
        a recall tombstone).
        """
        quoted = urllib.parse.quote(vector_id, safe="")
        params = {"include_values": "true"} if include_values else None
        return self._request(
            "GET", f"/v1/datasets/{name}/vectors/{quoted}", params=params
        ).json()

    def list(
        self,
        name: str,
        filter: Optional[dict] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> dict:
        """``GET /v1/datasets/{name}/vectors`` — list + filter + paginate.

        ``filter`` is a flat AND-of-equals object, sent URL-encoded as JSON.
        Returns ``{"vectors": [...], "next_cursor": ...}``.
        """
        params: dict = {}
        if filter:
            params["filter"] = json.dumps(filter)
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._request(
            "GET", f"/v1/datasets/{name}/vectors", params=params
        ).json()

    def delete(self, name: str, vector_id: str) -> None:
        """``DELETE /v1/datasets/{name}/vectors/{id}`` — delete one vector.

        With ``RB_RECALL`` on this is a synchronous tombstone (204,
        read-your-deletes); off it queues a rebuild (202). Either way the call
        returns ``None``; deleting an absent id is a clean no-op.
        """
        quoted = urllib.parse.quote(vector_id, safe="")
        self._request("DELETE", f"/v1/datasets/{name}/vectors/{quoted}")

    # -- query --------------------------------------------------------------

    def query(
        self,
        name: str,
        vector: list[float],
        top_k: int = 10,
        filter: Optional[dict] = None,
        poll_timeout: float = 30.0,
        poll_interval: float = 0.1,
    ) -> dict:
        """``POST /v1/query`` — vector similarity search (lower score = closer).

        Returns ``{"matches": [{id, score, metadata}], "mode", ...}``. ``score``
        is the raw L2-squared distance — **lower is closer**.

        When the server answers ``mode == "ephemeral"`` (the dataset has no
        shard yet and recall had nothing), the immediate body carries an empty
        ``matches`` plus a ``job_id``; this method then polls
        ``GET /v1/query/status/{job_id}`` until ``ready`` is true, raising
        :class:`QueryTimeoutError` after ``poll_timeout`` seconds.
        """
        body: dict = {"dataset": name, "vector": list(vector), "top_k": top_k}
        if filter:
            body["filter"] = filter
        result = self._request("POST", "/v1/query", json_body=body).json()

        if result.get("mode") == "ephemeral" and result.get("job_id"):
            return self._poll_query(result["job_id"], poll_timeout, poll_interval)
        return result

    def _poll_query(
        self, job_id: str, timeout: float, interval: float
    ) -> dict:
        """Poll ``GET /v1/query/status/{job_id}`` until ready (or time out)."""
        deadline = time.monotonic() + timeout
        while True:
            status = self._request(
                "GET", f"/v1/query/status/{job_id}"
            ).json()
            if status.get("ready"):
                return status
            if time.monotonic() >= deadline:
                raise QueryTimeoutError(job_id, timeout)
            time.sleep(interval)

    # -- helpers ------------------------------------------------------------

    def poll_until_indexed(
        self, name: str, timeout: float = 30.0, interval: float = 0.25
    ) -> dict:
        """Poll ``GET /v1/datasets/{name}`` until ``status == "indexed"``.

        Useful in the eventually-consistent (``RB_RECALL`` off) mode after an
        upsert/delete, where the build is asynchronous. With the recall tier on,
        writes are immediately queryable, so this is usually unnecessary.

        Raises :class:`QueryTimeoutError` on timeout, or surfaces the dataset's
        ``error`` status as a :class:`RosalindDBError`.
        """
        deadline = time.monotonic() + timeout
        while True:
            ds = self.get_dataset(name)
            status = ds.get("status")
            if status == "indexed":
                return ds
            if status == "error":
                raise RosalindDBError(
                    "dataset_error", ds.get("error_message") or "dataset build failed"
                )
            if time.monotonic() >= deadline:
                raise QueryTimeoutError(f"dataset:{name}", timeout)
            time.sleep(interval)
