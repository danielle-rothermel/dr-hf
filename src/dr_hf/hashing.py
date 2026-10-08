"""Content hashing for dr-hf identity values.

``content_hash`` is the single hashing seam in dr-hf. It is a stand-in for
canonical hashing from dr-serialize: when dr-hf adopts dr-serialize, this
function is replaced and every hash computed through it changes. Hashes
produced here (pin hashes, subset content hashes) are therefore stable for a
given dr-hf release but are not durable identities across that swap.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["content_hash"]


def content_hash(payload: Mapping[str, object]) -> str:
    """Return the sha256 hex digest of a JSON-compatible mapping.

    The payload is encoded as UTF-8 JSON with sorted keys, compact
    separators (``","`` and ``":"``), ``ensure_ascii=False``, and NaN or
    infinite floats rejected.
    """
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
