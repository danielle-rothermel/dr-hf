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
    document = build_identity_document(
        schema=schema,
        schema_version=IDENTITY_SCHEMA_VERSION,
        payload=dict(payload),
    )
    return str(identity_document_hash(document))
