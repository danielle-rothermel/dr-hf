from __future__ import annotations

import pytest

from dr_hf import (
    DatasetPin,
    origin_subsets,
    read_rows,
    resolve_dataset_pin,
    split_sizes,
)

pytestmark = pytest.mark.hub

ARC_SIZES = {"train": 1119, "validation": 299, "test": 1172}


@pytest.fixture(scope="module")
def arc_pin() -> DatasetPin:
    return resolve_dataset_pin(
        "allenai/ai2_arc", config="ARC-Challenge", native_id_field="id"
    )


def test_arc_challenge_resolves_and_reads(arc_pin: DatasetPin) -> None:
    assert len(arc_pin.revision) == 40
    assert dict(split_sizes(arc_pin)) == ARC_SIZES

    rows = list(read_rows(arc_pin))
    assert len(rows) == sum(ARC_SIZES.values())
    assert set(rows[0].fields) == {"id", "question", "choices", "answerKey"}

    subsets = {s.key.name: s for s in origin_subsets(arc_pin, rows)}
    assert {name: len(s.native_ids) for name, s in subsets.items()} == (
        ARC_SIZES
    )
    version = f"origin-{arc_pin.revision[:7]}"
    assert all(s.key.version == version for s in subsets.values())


def test_datadecide_sample_eval_reads_arc_test_ids(
    arc_pin: DatasetPin,
) -> None:
    pin = resolve_dataset_pin(
        "allenai/DataDecide-eval-instances",
        config="sample-eval",
        native_id_field="native_id",
    )
    assert dict(split_sizes(pin)) == {"test": 1172}
    instance_ids = {row.native_id for row in read_rows(pin)}
    arc_test_ids = {
        row.native_id for row in read_rows(arc_pin, splits=["test"])
    }
    assert instance_ids == arc_test_ids
