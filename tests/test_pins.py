from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_hf import (
    DATASET_PIN_SCHEMA,
    ROW_INDEX,
    DatasetPin,
    ModelPin,
    identity_hash,
    resolve_dataset_pin,
    resolve_model_pin,
)
from tests.hf_hub_http_mock import stub_revision_info

SHA = "91c54ad1727ee830252e457677f467be0bfd8a57"


def _pin(**overrides: object) -> DatasetPin:
    fields: dict[str, object] = {
        "repo_id": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "revision": SHA,
        "native_id_field": "id",
    }
    fields.update(overrides)
    return DatasetPin.model_validate(fields)


@pytest.mark.parametrize(
    "revision",
    ["main", SHA[:7], SHA.upper(), SHA + "0", "g" * 40, ""],
)
def test_dataset_pin_rejects_non_sha_revision(revision: str) -> None:
    with pytest.raises(ValidationError, match="40-character"):
        _pin(revision=revision)


def test_model_pin_rejects_branch_name() -> None:
    with pytest.raises(ValidationError, match="40-character"):
        ModelPin(repo_id="org/model", revision="step1000-seed0")


def test_dataset_pin_rejects_blank_repo_and_field() -> None:
    with pytest.raises(ValidationError):
        _pin(repo_id="")
    with pytest.raises(ValidationError):
        _pin(native_id_field="")


def test_dataset_pin_is_frozen_and_hash_covers_every_field() -> None:
    pin = _pin()
    with pytest.raises(ValidationError):
        pin.revision = "0" * 40  # ty: ignore[invalid-assignment]
    assert pin.pin_hash == identity_hash(
        DATASET_PIN_SCHEMA, pin.model_dump(mode="json")
    )
    variants = [
        _pin(config=None),
        _pin(revision="0" * 40),
        _pin(native_id_field=ROW_INDEX),
        _pin(repo_id="allenai/other"),
    ]
    assert len({pin.pin_hash, *(v.pin_hash for v in variants)}) == 5


def test_resolve_dataset_pin_records_resolved_sha() -> None:
    with stub_revision_info(repo_id="allenai/ai2_arc", sha=SHA) as calls:
        pin = resolve_dataset_pin(
            "allenai/ai2_arc",
            config="ARC-Challenge",
            ref="refs/convert/parquet",
            native_id_field="id",
        )
    assert pin == _pin()
    assert calls == [
        (
            "GET",
            (
                "https://huggingface.co/api/datasets/allenai/ai2_arc/"
                "revision/refs%2Fconvert%2Fparquet"
            ),
        )
    ]


def test_resolve_model_pin_uses_model_endpoint() -> None:
    with stub_revision_info(repo_id="org/model", sha=SHA) as calls:
        pin = resolve_model_pin("org/model", ref="step1000-seed0")
    assert pin == ModelPin(repo_id="org/model", revision=SHA)
    assert calls == [
        (
            "GET",
            (
                "https://huggingface.co/api/models/org/model/revision/"
                "step1000-seed0"
            ),
        )
    ]


def test_resolve_rejects_missing_sha() -> None:
    with (
        stub_revision_info(repo_id="org/model", sha=None),
        pytest.raises(ValueError, match="no commit SHA"),
    ):
        resolve_model_pin("org/model")
