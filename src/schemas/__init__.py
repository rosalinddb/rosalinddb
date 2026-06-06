"""Wire-contract schemas for RosalindDB's public HTTP surface.

This package holds the Pydantic models that describe the JSON request and
response shapes RosalindDB speaks on the wire, versioned by API major
(``v1`` today). The models are deliberately a thin, *descriptive* layer: they
mirror the shapes the hand-rolled handlers already produce, so introducing
them does not change a single byte on the wire and does not move where
semantic validation (dataset existence, dimension match, range checks)
happens.

See :mod:`schemas.v1` for the ``/v1/query`` request and the success / error
response envelopes.
"""

from __future__ import annotations
