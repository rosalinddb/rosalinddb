"""Happy-path smoke check for a running RosalindDB instance.

This is the post-deploy "did the deploy actually work" gate: it exercises
RosalindDB's core path — health, signup, dataset create, ingest, the async
validate->index pipeline, and a query — against ONE base URL.

It works against any RosalindDB instance:

    python scripts/smoke.py                              # local dev backend
    python scripts/smoke.py --base-url https://api.example.com
    BASE_URL=https://api.example.com python scripts/smoke.py

`make smoke` wraps this; `make smoke BASE_URL=...` overrides the target.

The script makes HTTP calls ONLY — it does NOT bring up or tear down any
stack; the target instance must already be running. It is dependency-light
(stdlib `urllib` only, no venv required) and idempotent across runs: every
run uses a fresh unique email and dataset name so re-running never collides
with leftovers.

Each step prints a PASS/FAIL line; the process exits non-zero on the first
failure so it slots straight into CI / a deploy gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

# All vectors in this smoke run are 4-dimensional — small and quick. The
# dataset is created with this dimension and the query vector matches it.
_DIM = 4
_VECTOR_COUNT = 8
# The validate->index pipeline is async; poll the dataset until it reports
# `indexed`. Generous but bounded so a wedged pipeline fails the gate.
_INDEX_TIMEOUT_S = 60.0
# `POST /v1/query` can return an async `ephemeral` job when the query service
# has no hot shard cached; poll its status endpoint with the same bounded,
# generous-but-finite budget so a wedged ephemeral runner fails the gate.
_QUERY_TIMEOUT_S = 60.0
_POLL_INTERVAL_S = 2.0
_HTTP_TIMEOUT_S = 30.0


class SmokeFailure(Exception):
    """Raised when a smoke step fails — carries a human-readable reason."""


def _request(method: str, url: str, *, headers=None, body=None):
    """Make one HTTP request, returning `(status_code, parsed_json_or_text)`.

    Uses stdlib `urllib` so the script has zero third-party dependencies and
    can run against a deployed URL without the repo's virtualenv. A non-2xx
    response is NOT raised here — callers inspect the status themselves so
    they can print a precise per-step diagnostic.
    """
    data = None
    req_headers = dict(headers or {})
    if body is not None:
        if isinstance(body, (bytes, bytearray)):
            data = bytes(body)
        else:
            data = json.dumps(body).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, method=method, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    except urllib.error.URLError as exc:
        raise SmokeFailure(f"could not reach {url}: {exc.reason}") from exc
    text = raw.decode("utf-8", errors="replace")
    try:
        return status, json.loads(text) if text else None
    except json.JSONDecodeError:
        return status, text


def _step(num: int, name: str):
    """Print the start-of-step banner; the result line is printed by `_ok`/raise."""
    print(f"[{num}] {name} ... ", end="", flush=True)


def _ok(detail: str = "") -> None:
    """Print a PASS marker (with optional detail) for the current step."""
    print(f"PASS{(' — ' + detail) if detail else ''}")


def run_smoke(base_url: str) -> None:
    """Run the full happy-path smoke check against `base_url`.

    Raises `SmokeFailure` on the first failing step; the caller maps that to a
    non-zero exit code.
    """
    base_url = base_url.rstrip("/")
    # Unique per run -> idempotent: re-running never collides with leftovers.
    # The domain uses a real public TLD because signup runs the address
    # through `email-validator`, which rejects non-deliverable TLDs like
    # `.local`. `example.com` is the reserved-for-docs domain (RFC 2606).
    run_id = uuid.uuid4().hex[:10]
    email = f"smoke-{run_id}@smoke.rosalinddb.example.com"
    password = "smoke-password-123"
    dataset = f"smoke_{run_id}"

    print(f"RosalindDB smoke check against {base_url}")
    print(f"  run id: {run_id}  email: {email}  dataset: {dataset}")
    print("-" * 60)

    # 1. Health probe — unauthenticated, must be 200.
    _step(1, "GET /healthz")
    status, body = _request("GET", f"{base_url}/healthz")
    if status != 200:
        raise SmokeFailure(f"/healthz returned {status}, expected 200 (body: {body!r})")
    _ok(f"{body}")

    # 2. Signup — get a JWT and the auto-issued first API key.
    #
    # The same smoke runs against TWO deployment shapes:
    #   - OSS default (RB_REQUIRE_AUTH unset/false): `/auth/signup` is
    #     intentionally hidden and returns 404 `auth_disabled`. There is no
    #     per-tenant principal in this mode; every request resolves to the
    #     built-in `default` tenant without any Authorization header.
    #   - Auth-on (RB_REQUIRE_AUTH=true): signup returns 201 with a JWT and a
    #     first API key, and every subsequent request must carry the key as
    #     `Authorization: Bearer rb_live_...`.
    #
    # We probe the response and pick the right branch. Both modes must pass
    # the full happy-path check end-to-end.
    _step(2, "POST /auth/signup")
    status, body = _request(
        "POST", f"{base_url}/auth/signup", body={"email": email, "password": password}
    )
    if status == 404 and isinstance(body, dict) and (
        (body.get("error") or {}).get("code") == "auth_disabled"
    ):
        # OSS mode: auth is disabled by design. Run the rest of the smoke
        # unauthenticated; the CP short-circuits every caller to the built-in
        # `default` tenant.
        _ok("auth disabled (OSS mode) — running unauthenticated")
        key_headers: dict = {}
    elif status == 201 and isinstance(body, dict):
        token = body.get("token")
        api_key = (body.get("first_api_key") or {}).get("key")
        if not token or not api_key:
            raise SmokeFailure(
                f"signup response missing token/first_api_key (body: {body!r})"
            )
        _ok("got JWT + first API key")
        key_headers = {"Authorization": f"Bearer {api_key}"}
    else:
        raise SmokeFailure(f"signup returned {status} (body: {body!r})")

    # 3. Create a small dataset using the API key.
    _step(3, f"POST /v1/datasets ({dataset}, dim={_DIM})")
    status, body = _request(
        "POST",
        f"{base_url}/v1/datasets",
        headers=key_headers,
        body={"name": dataset, "dimension": _DIM},
    )
    if status != 201:
        raise SmokeFailure(f"dataset create returned {status} (body: {body!r})")
    _ok("dataset created")

    # 4. Ingest a handful of vectors via the small NDJSON endpoint.
    _step(4, f"POST /v1/datasets/{dataset}/vectors ({_VECTOR_COUNT} vectors)")
    ndjson_lines = []
    for i in range(_VECTOR_COUNT):
        # Simple deterministic values — content is irrelevant, we just need
        # real records to flow through validate -> index.
        values = [float((i + j) % 7) for j in range(_DIM)]
        ndjson_lines.append(
            json.dumps({"id": f"vec-{i}", "values": values, "metadata": {"n": i}})
        )
    ndjson_body = ("\n".join(ndjson_lines) + "\n").encode("utf-8")
    status, body = _request(
        "POST",
        f"{base_url}/v1/datasets/{dataset}/vectors",
        headers={**key_headers, "Content-Type": "application/x-ndjson"},
        body=ndjson_body,
    )
    if status != 202 or not isinstance(body, dict):
        raise SmokeFailure(f"ingest returned {status} (body: {body!r})")
    accepted = body.get("accepted")
    if accepted != _VECTOR_COUNT:
        raise SmokeFailure(
            f"ingest accepted {accepted}, expected {_VECTOR_COUNT} (body: {body!r})"
        )
    _ok(f"{accepted} vectors accepted")

    # 5. Poll the dataset until the async pipeline reports `indexed`.
    _step(5, f"GET /v1/datasets/{dataset} until status=indexed")
    deadline = time.time() + _INDEX_TIMEOUT_S
    last_status = None
    while time.time() < deadline:
        status, body = _request(
            "GET", f"{base_url}/v1/datasets/{dataset}", headers=key_headers
        )
        if status != 200 or not isinstance(body, dict):
            raise SmokeFailure(f"dataset get returned {status} (body: {body!r})")
        last_status = body.get("status")
        if last_status == "indexed":
            break
        if last_status == "error":
            raise SmokeFailure(
                f"dataset entered error state: {body.get('error_message')!r}"
            )
        time.sleep(_POLL_INTERVAL_S)
    if last_status != "indexed":
        raise SmokeFailure(
            f"dataset not indexed within {_INDEX_TIMEOUT_S:.0f}s "
            f"(last status: {last_status!r})"
        )
    _ok(f"status=indexed within {_INDEX_TIMEOUT_S:.0f}s budget")

    # 6. Query — assert results come back, via EITHER query path.
    #
    # `POST /v1/query` has two legitimate shapes, both HTTP 200:
    #   - hot/cold path: the query service has the shard and returns
    #     `{matches: [...], mode: "hot"|"cold"}` synchronously.
    #   - ephemeral path: the query service has no hot shard cached and
    #     enqueues the search async, returning `{matches: [], mode:
    #     "ephemeral", job_id: "job_..."}`. The result must then be polled
    #     from `GET /v1/query/status/{job_id}`, which returns `{ready: false}`
    #     while computing and `{ready: true, matches: [...], mode:
    #     "ephemeral"}` once done.
    #
    # Step 5 polls the *dataset* to `status=indexed` (set by the index
    # builder), but that does NOT guarantee the *query service* has a hot
    # shard at query time — against a real multi-service deployment the
    # query can legitimately land on the ephemeral fallback. Asserting on
    # the synchronous `matches` alone would false-red a healthy system, so
    # we handle both paths and only FAIL when results are genuinely absent.
    _step(6, "POST /v1/query")
    query_vector = [float(j % 7) for j in range(_DIM)]
    status, body = _request(
        "POST",
        f"{base_url}/v1/query",
        headers=key_headers,
        body={"dataset": dataset, "vector": query_vector, "top_k": 5},
    )
    if status != 200 or not isinstance(body, dict):
        raise SmokeFailure(f"query returned {status} (body: {body!r})")

    mode = body.get("mode")
    matches = body.get("matches")

    if mode == "ephemeral":
        # Async fallback: poll the job until the runner publishes a result.
        job_id = body.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise SmokeFailure(
                f"query returned mode=ephemeral without a job_id (body: {body!r})"
            )
        deadline = time.time() + _QUERY_TIMEOUT_S
        result = None
        while time.time() < deadline:
            status, poll = _request(
                "GET",
                f"{base_url}/v1/query/status/{job_id}",
                headers=key_headers,
            )
            if status != 200 or not isinstance(poll, dict):
                raise SmokeFailure(
                    f"query status returned {status} (body: {poll!r})"
                )
            if poll.get("ready") is True:
                result = poll
                break
            time.sleep(_POLL_INTERVAL_S)
        if result is None:
            raise SmokeFailure(
                f"ephemeral query {job_id} not ready within "
                f"{_QUERY_TIMEOUT_S:.0f}s"
            )
        matches = result.get("matches")
        if not isinstance(matches, list) or len(matches) == 0:
            raise SmokeFailure(
                f"ephemeral query returned no matches (job_id={job_id}, "
                f"body: {result!r})"
            )
        _ok(f"{len(matches)} matches (mode=ephemeral, polled {job_id})")
        return

    # Hot/cold path: matches are present synchronously.
    if not isinstance(matches, list) or len(matches) == 0:
        raise SmokeFailure(
            f"query returned no matches (mode={mode!r}, body: {body!r})"
        )
    _ok(f"{len(matches)} matches (mode={mode})")


def main() -> int:
    """CLI entry point. Returns the process exit code (0 = all steps passed)."""
    parser = argparse.ArgumentParser(
        description="Happy-path smoke check for a running RosalindDB instance."
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("BASE_URL", "http://localhost:8081"),
        help="Base URL of the RosalindDB instance "
        "(default: $BASE_URL or http://localhost:8081).",
    )
    args = parser.parse_args()

    start = time.time()
    try:
        run_smoke(args.base_url)
    except SmokeFailure as exc:
        print("FAIL")
        print("-" * 60)
        print(f"SMOKE FAILED: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 — surface anything unexpected as a failure.
        print("FAIL")
        print("-" * 60)
        print(f"SMOKE FAILED (unexpected error): {type(exc).__name__}: {exc}")
        return 1

    print("-" * 60)
    print(f"SMOKE PASSED — all 6 steps green in {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
