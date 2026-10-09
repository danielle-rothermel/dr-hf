from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from datasets import Dataset, DatasetDict

import dr_hf.datasets as datasets_module
from dr_hf import (
    ROW_INDEX,
    DatasetPin,
    NativeIdError,
    read_rows,
    split_sizes,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

SHA = "91c54ad1727ee830252e457677f467be0bfd8a57"


def _pin(native_id_field: str = "id") -> DatasetPin:
    return DatasetPin(
        repo_id="org/data",
        config="cfg",
        revision=SHA,
        native_id_field=native_id_field,
    )


def _install(
    monkeypatch: pytest.MonkeyPatch,
    splits: Mapping[str, Mapping[str, list[object]]],
) -> list[tuple[tuple[object, ...], dict[str, object]]]:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    dataset_dict = DatasetDict(
        {name: Dataset.from_dict(dict(cols)) for name, cols in splits.items()}
    )

    def fake_load_dataset(
        *args: object, **kwargs: object
    ) -> DatasetDict | Dataset:
        calls.append((args, kwargs))
        split = kwargs.get("split")
        if split is None:
            return dataset_dict
        return dataset_dict[str(split)]

    monkeypatch.setattr(datasets_module, "load_dataset", fake_load_dataset)
    return calls


def test_read_rows_tags_split_and_coerces_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install(
        monkeypatch,
        {
            "train": {"id": [3, 1], "q": ["a", "b"]},
            "test": {"id": [4, 2], "q": ["c", "d"]},
        },
    )
    rows = list(read_rows(_pin()))
    assert calls == [(("org/data", "cfg"), {"revision": SHA})]
    assert [(r.origin_split, r.native_id) for r in rows] == [
        ("train", "3"),
        ("train", "1"),
        ("test", "4"),
        ("test", "2"),
    ]
    assert rows[0].fields == {"id": 3, "q": "a"}
    assert all(row.pin == _pin() for row in rows)


def test_read_rows_row_index_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"validation": {"q": ["a", "b", "c"]}})
    rows = list(read_rows(_pin(ROW_INDEX)))
    assert [r.native_id for r in rows] == [
        "validation:0",
        "validation:1",
        "validation:2",
    ]


def test_read_rows_selected_splits_load_each_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install(
        monkeypatch,
        {"train": {"id": ["a"]}, "test": {"id": ["b"]}},
    )
    rows = list(read_rows(_pin(), splits=["test"]))
    assert [(r.origin_split, r.native_id) for r in rows] == [("test", "b")]
    assert calls == [(("org/data", "cfg"), {"revision": SHA})]


def test_read_rows_rejects_unknown_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, {"train": {"id": ["a"]}})
    with pytest.raises(ValueError, match="unknown splits"):
        list(read_rows(_pin(), splits=["test"]))


def test_duplicate_id_across_splits_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        {"train": {"id": ["a", "3"]}, "test": {"id": ["3", "b"]}},
    )
    rows = read_rows(_pin(), splits=["test"])
    with pytest.raises(NativeIdError, match=r"'3'.*splits 'train' and 'test'"):
        next(rows)


def test_row_index_ids_are_unique_across_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, {"train": {"q": ["a"]}, "test": {"q": ["b"]}})
    rows = list(read_rows(_pin(ROW_INDEX)))
    assert [r.native_id for r in rows] == ["train:0", "test:0"]


def test_read_rows_rejects_repeated_split_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, {"test": {"id": ["b"]}})
    with pytest.raises(ValueError, match="unique"):
        list(read_rows(_pin(), splits=["test", "test"]))


def test_duplicate_id_raises_before_any_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        {"train": {"id": ["a", "b"]}, "test": {"id": ["1", "1"]}},
    )
    rows = read_rows(_pin())
    with pytest.raises(NativeIdError, match="duplicate native id '1'"):
        next(rows)


def test_missing_field_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"train": {"q": ["a"]}})
    with pytest.raises(NativeIdError, match="not a column"):
        list(read_rows(_pin()))


def test_null_id_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"train": {"id": ["a", None]}})
    with pytest.raises(NativeIdError, match="null native id at row 1"):
        list(read_rows(_pin()))


def test_split_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        {"train": {"id": [1, 2, 3]}, "test": {"id": [4]}},
    )
    assert split_sizes(_pin()) == {"train": 3, "test": 1}
