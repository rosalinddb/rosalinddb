"""Unit tests for `adapters/storage/shard_uri.py`.

Contract under test:

  - `build(bucket, tenant, dataset, shard_id, content_bytes) -> str`
    returns `s3://{bucket}/{tenant}/{dataset}/{shard_id}-{content_hash}.bin`
    where `content_hash` is the first 16 hex chars of
    `hashlib.sha256(content_bytes).hexdigest()`. Two identical builds collide
    (cheap dedup); two different builds cannot.
  - `parse(uri) -> ShardURI` (a NamedTuple with `bucket`, `tenant`, `dataset`,
    `shard_id`, `content_hash`) round-trips `build`.
  - `is_legacy(uri) -> bool` returns True for the pre-versioning shape
    `s3://bucket/tenant/dataset/shard-{id}.bin` so the SSD tier can identify
    files written before this PR and migrate them safely.
"""
from __future__ import annotations

import hashlib

import pytest


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------


def test_build_includes_content_hash_in_uri():
    """`build` produces a `{shard_id}-{content_hash}.bin` key with the
    expected sha256 prefix.
    """
    from adapters.storage import shard_uri

    content = b"hello-shard"
    expected_hash = hashlib.sha256(content).hexdigest()[:16]
    uri = shard_uri.build("rosalinddb", "tenantA", "datasetX", "42", content)
    assert uri == f"s3://rosalinddb/tenantA/datasetX/42-{expected_hash}.bin"


def test_build_is_deterministic_in_content():
    """Same content + same identifiers => same URI (cheap dedup); different
    content => different URI (no collisions possible)."""
    from adapters.storage import shard_uri

    args = ("b", "t", "d", "s")
    a = shard_uri.build(*args, b"payload-one")
    b = shard_uri.build(*args, b"payload-one")
    c = shard_uri.build(*args, b"payload-two")
    assert a == b
    assert a != c


def test_build_hash_is_exactly_16_hex_chars():
    """The hash slice length is part of the contract: it bounds key length
    and the SSD tier parses it positionally."""
    from adapters.storage import shard_uri

    uri = shard_uri.build("b", "t", "d", "s", b"x")
    suffix = uri.rsplit("-", 1)[-1].removesuffix(".bin")
    assert len(suffix) == 16
    assert all(ch in "0123456789abcdef" for ch in suffix)


def test_build_accepts_empty_content_bytes():
    """Empty content is allowed — the sha256 of the empty string is a stable,
    well-known value. The builder should never call build() with no bytes in
    practice, but the function must not blow up on the boundary."""
    from adapters.storage import shard_uri

    empty_hash = hashlib.sha256(b"").hexdigest()[:16]
    uri = shard_uri.build("b", "t", "d", "s", b"")
    assert uri.endswith(f"s-{empty_hash}.bin")


# ---------------------------------------------------------------------------
# parse()
# ---------------------------------------------------------------------------


def test_parse_round_trips_build():
    """`parse(build(...))` recovers every field."""
    from adapters.storage import shard_uri

    content = b"round-trip-me"
    uri = shard_uri.build("bk", "tnt", "dst", "shrd", content)
    parsed = shard_uri.parse(uri)
    assert parsed.bucket == "bk"
    assert parsed.tenant == "tnt"
    assert parsed.dataset == "dst"
    assert parsed.shard_id == "shrd"
    assert parsed.content_hash == hashlib.sha256(content).hexdigest()[:16]


def test_parse_round_trips_with_hyphen_in_shard_id():
    """`shard_id` is allowed to contain `-` (the existing builder produces
    names like `shard-1700000000000-abcd1234`). Parse must split only on the
    *last* hyphen of the filename so the content-hash is recovered correctly."""
    from adapters.storage import shard_uri

    content = b"hyphens-everywhere"
    parsed = shard_uri.parse(
        shard_uri.build("b", "tnt", "dst", "shard-1700000000000-abcd1234", content)
    )
    assert parsed.shard_id == "shard-1700000000000-abcd1234"
    assert parsed.content_hash == hashlib.sha256(content).hexdigest()[:16]


def test_parse_raises_on_legacy_uri():
    """`parse` is for the versioned shape only — handing it a legacy URI is a
    bug the caller should hear about loudly."""
    from adapters.storage import shard_uri

    with pytest.raises(ValueError):
        shard_uri.parse("s3://bucket/tenant/dataset/shard-42.bin")


def test_parse_raises_on_garbage_uri():
    """Non-`s3://`, missing `.bin`, missing components — all malformed."""
    from adapters.storage import shard_uri

    for garbage in (
        "",
        "not-a-uri",
        "http://bucket/tenant/dataset/shard-abc.bin",
        "s3://bucket/tenant/dataset/file.txt",  # wrong suffix
        "s3://bucket/tenant/dataset/shard.bin",  # no hash component
        "s3://bucket/onlytwo.bin",  # missing tenant/dataset
        "s3://bucket/t/d/shard-NOTHEX1234567890.bin",  # 16 chars but not hex
    ):
        with pytest.raises(ValueError):
            shard_uri.parse(garbage)


# ---------------------------------------------------------------------------
# is_legacy()
# ---------------------------------------------------------------------------


def test_is_legacy_identifies_pre_versioning_shape():
    """`s3://b/t/d/shard-42.bin` is legacy."""
    from adapters.storage import shard_uri

    assert shard_uri.is_legacy("s3://bucket/tenant/dataset/shard-42.bin") is True
    # The existing index_builder also produced this longer shape; it is
    # equally legacy and the SSD tier needs to recognise it for migration.
    assert (
        shard_uri.is_legacy(
            "s3://rosalinddb/indexes/tnt/dst/indexes/2026-01-01/"
            "shard-1700000000000-abcd1234.bin"
        )
        is True
    )


def test_is_legacy_rejects_versioned_uri():
    """A freshly-built versioned URI is NOT legacy."""
    from adapters.storage import shard_uri

    uri = shard_uri.build("bucket", "tenant", "dataset", "42", b"some-bytes")
    assert shard_uri.is_legacy(uri) is False


def test_is_legacy_rejects_non_s3_or_malformed_uris():
    """A URI we cannot classify (garbage, wrong scheme) is not "legacy" —
    legacy means specifically a pre-versioning *shard* file."""
    from adapters.storage import shard_uri

    for not_legacy in (
        "",
        "http://bucket/tenant/dataset/shard-1.bin",
        "s3://bucket/tenant/dataset/notashard.bin",
        "s3://bucket/tenant/dataset/file.txt",
    ):
        assert shard_uri.is_legacy(not_legacy) is False


# ---------------------------------------------------------------------------
# Builder integration: env-flag-gated URI selection.
# ---------------------------------------------------------------------------
#
# `RB_SHARD_VERSIONED_URIS` flips the builder between the legacy mutable URI
# shape (default — preserves current behaviour bit-identically) and the new
# content-addressed shape. The flag-off path is the rollback contract.


def test_compute_shard_uri_off_preserves_legacy_shape(monkeypatch):
    """Default / flag-off: returns the existing
    `{INDEXES_PREFIX}/{tenant}/{dataset}/indexes/{date}/shard-{ts}-{uuid8}.bin`
    shape. Bit-identical to today is the rollback contract."""
    monkeypatch.delenv("RB_SHARD_VERSIONED_URIS", raising=False)
    monkeypatch.setenv("INDEXES_PREFIX", "s3://rosalinddb/indexes")
    import importlib

    import services.index_builder.run as run

    importlib.reload(run)
    uri = run._compute_shard_uri(
        tenant="acme",
        dataset="docs",
        shard_name="shard-1700000000000-abcd1234.bin",
        blob=b"any-bytes",
    )
    assert uri == (
        "s3://rosalinddb/indexes/acme/docs/indexes/"
        f"{run.time.strftime('%Y-%m-%d')}/shard-1700000000000-abcd1234.bin"
    )


def test_compute_shard_uri_on_returns_versioned_shape(monkeypatch):
    """Flag-on: returns the new `s3://{bucket}/{tenant}/{dataset}/{shard_id}-{hash}.bin`
    content-addressed shape via `shard_uri.build`."""
    monkeypatch.setenv("RB_SHARD_VERSIONED_URIS", "true")
    monkeypatch.setenv("INDEXES_PREFIX", "s3://rosalinddb/indexes")
    import importlib

    from adapters.storage import shard_uri

    import services.index_builder.run as run

    importlib.reload(run)
    blob = b"deterministic-bytes"
    uri = run._compute_shard_uri(
        tenant="acme",
        dataset="docs",
        shard_name="shard-1700000000000-abcd1234.bin",
        blob=blob,
    )
    # The versioned shape comes straight from `shard_uri.build`. The shard_id
    # passed in is the shard_name stem (without `.bin`); the URI itself is
    # round-trippable.
    parsed = shard_uri.parse(uri)
    assert parsed.tenant == "acme"
    assert parsed.dataset == "docs"
    assert parsed.shard_id == "shard-1700000000000-abcd1234"
    import hashlib as _hashlib

    assert parsed.content_hash == _hashlib.sha256(blob).hexdigest()[:16]
    assert shard_uri.is_legacy(uri) is False


def test_compute_shard_uri_off_for_falsy_values(monkeypatch):
    """Only truthy values (`1`, `true`, `yes`, `on`) flip the flag; anything
    else (including `0`, empty string, garbage) leaves the legacy path active."""
    monkeypatch.setenv("INDEXES_PREFIX", "s3://b/indexes")
    import importlib

    import services.index_builder.run as run

    # "FALSE" uppercase is explicitly included to pin the contract that
    # `_truthy` lowercases before matching the truthy set — without that,
    # an operator who exports `RB_SHARD_VERSIONED_URIS=FALSE` (yelling)
    # would silently miss the flag-off rollback path.
    for falsy in ("", "0", "false", "FALSE", "no", "off", "anything-not-truthy"):
        monkeypatch.setenv("RB_SHARD_VERSIONED_URIS", falsy)
        importlib.reload(run)
        uri = run._compute_shard_uri(
            tenant="t",
            dataset="d",
            shard_name="shard-1-aa.bin",
            blob=b"x",
        )
        # Legacy path is keyed by an `indexes/{date}/` segment between the
        # dataset and the shard filename. (The leading `/` is consumed by the
        # split delimiter; we assert the substring as it appears in the tail.)
        tail = uri.split("/d/", 1)[1]
        assert tail.startswith("indexes/")
        assert uri.endswith("shard-1-aa.bin")
