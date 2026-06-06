"""Versioned, content-addressed shard URI helpers.

Today's catalog row points at a `shard_uri` string that *looks* immutable but
isn't enforced — the builder could in principle overwrite a key, and a DP that
re-resolves the catalog can read newer bytes through an older URI in transit.
Content-addressed URIs (Git's blob-by-SHA model) make the version visible in
the URI itself: two builds producing the same bytes converge on the same URI
(cheap dedup), and two builds producing different bytes can never collide.

The new shape: `s3://{bucket}/{tenant}/{dataset}/{shard_id}-{content_hash}.bin`
where `content_hash` is the first 16 hex chars of `sha256(serialised_index)`.
16 hex chars = 64 bits of entropy, which is comfortably above birthday-collision
risk for the shard-counts we will ever see (10^9 shards = ~1 in 10^10 collision
odds) while keeping the key short.

This module is intentionally tiny and pure (no I/O, no env reads): the SSD
tier, the index builder, and the orphan GC all import it. Keeping it small
also keeps it cheap to test exhaustively.
"""

from __future__ import annotations

import hashlib
from typing import NamedTuple


_SCHEME = "s3://"
_SUFFIX = ".bin"
_HASH_LEN = 16  # first 16 hex chars of sha256(content)
_HEX_DIGITS = frozenset("0123456789abcdef")


class ShardURI(NamedTuple):
    """Decomposed versioned shard URI. `parse(build(...))` returns this."""

    bucket: str
    tenant: str
    dataset: str
    shard_id: str
    content_hash: str


def build(
    bucket: str,
    tenant: str,
    dataset: str,
    shard_id: str,
    content_bytes: bytes,
) -> str:
    """Build a versioned, content-addressed shard URI.

    Hashing the bytes (not the inputs) is what makes the URI a verifiable
    receipt for what was written: two builds with identical content collapse
    onto the same key, and a corrupted byte stream produces a different URI
    that the catalog never references.
    """
    content_hash = hashlib.sha256(content_bytes).hexdigest()[:_HASH_LEN]
    return (
        f"{_SCHEME}{bucket}/{tenant}/{dataset}/"
        f"{shard_id}-{content_hash}{_SUFFIX}"
    )


def parse(uri: str) -> ShardURI:
    """Round-trip with `build()`. Raise `ValueError` on a legacy or malformed URI.

    Split policy: the path has exactly `bucket/tenant/dataset/<file>` (4
    segments). The filename is `{shard_id}-{content_hash}.bin` and is split on
    the LAST hyphen, since `shard_id` is allowed to contain hyphens (e.g. the
    existing builder's `shard-{ts}-{uuid8}` naming).
    """
    if not isinstance(uri, str) or not uri.startswith(_SCHEME):
        raise ValueError(f"not an s3:// URI: {uri!r}")
    path = uri[len(_SCHEME):]
    parts = path.split("/")
    if len(parts) != 4:
        raise ValueError(
            f"versioned shard URI must have shape "
            f"s3://bucket/tenant/dataset/<file>, got: {uri!r}"
        )
    bucket, tenant, dataset, filename = parts
    if not all((bucket, tenant, dataset, filename)):
        raise ValueError(f"empty path component in URI: {uri!r}")
    if not filename.endswith(_SUFFIX):
        raise ValueError(f"shard URI must end with {_SUFFIX!r}: {uri!r}")
    stem = filename[: -len(_SUFFIX)]
    # Refuse the legacy `shard-{id}.bin` shape outright: its trailing chunk
    # is not a content hash, and silently treating it as one would let stale
    # bytes ride a versioned key.
    if "-" not in stem:
        raise ValueError(f"shard URI is missing the content-hash suffix: {uri!r}")
    shard_id, _, content_hash = stem.rpartition("-")
    if not shard_id:
        raise ValueError(f"shard URI has empty shard_id: {uri!r}")
    if len(content_hash) != _HASH_LEN or not set(content_hash).issubset(_HEX_DIGITS):
        raise ValueError(
            f"content-hash suffix must be {_HASH_LEN} lowercase hex chars: {uri!r}"
        )
    return ShardURI(
        bucket=bucket,
        tenant=tenant,
        dataset=dataset,
        shard_id=shard_id,
        content_hash=content_hash,
    )


def is_legacy(uri: str) -> bool:
    """Return True for any pre-versioning shard file.

    A shard file is "legacy" iff its filename starts with `shard-` and ends
    with `.bin` AND the URI is not a valid versioned URI. The SSD tier uses
    this to decide whether to migrate a file in place vs. accept it as-is.
    The function never raises — non-shard URIs, garbage, and wrong-scheme
    strings all return False (they are not "legacy shards" per se).
    """
    if not isinstance(uri, str) or not uri.startswith(_SCHEME):
        return False
    if not uri.endswith(_SUFFIX):
        return False
    filename = uri.rsplit("/", 1)[-1]
    # A versioned filename is `{shard_id}-{16-hex}.bin`. If it parses as one,
    # it is NOT legacy regardless of how the prefix looks.
    try:
        parse(uri)
        return False
    except ValueError:
        pass
    return filename.startswith("shard-")
