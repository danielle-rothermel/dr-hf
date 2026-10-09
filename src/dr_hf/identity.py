"""Identity hashes for dr-hf values, computed by dr-serialize.

Every hash dr-hf produces (pin hashes, named subset content hashes) is the
dr-serialize identity hash of an identity document: a fixed schema name, a
schema version, and the value's JSON payload. dr-hf owns the schema names and
payload shapes; dr-serialize owns canonical bytes and hashing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dr_serialize import build_identity_document, identity_document_hash

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "DATASET_PIN_SCHEMA",
    "IDENTITY_SCHEMA_VERSION",
    "NAMED_SUBSET_SCHEMA",
    "identity_hash",
]

DATASET_PIN_SCHEMA = "dr_hf.dataset_pin"
NAMED_SUBSET_SCHEMA = "dr_hf.named_subset"
IDENTITY_SCHEMA_VERSION = 1


def identity_hash(schema: str, payload: Mapping[str, object]) -> str:
    """Return the dr-serialize identity hash of ``payload`` under ``schema``.

    The payload must be strict JSON (the output of ``model_dump(mode="json")``
    on a dr-hf model). Non-finite floats and non-string keys are rejected by
    dr-serialize before hashing.
    """
    document = build_identity_document(
        schema=schema,
        schema_version=IDENTITY_SCHEMA_VERSION,
        payload=dict(payload),
    )
    return str(identity_document_hash(document))
