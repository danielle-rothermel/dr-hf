from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from datasets import Dataset, load_dataset
from pydantic import BaseModel, ConfigDict

from .pins import ROW_INDEX, DatasetPin

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = ["NativeIdError", "SourceRow", "read_rows", "split_sizes"]


class NativeIdError(ValueError): ...


class SourceRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pin: DatasetPin
    origin_split: str
    native_id: str
    fields: Mapping[str, object]


def read_rows(
    pin: DatasetPin, *, splits: Sequence[str] | None = None
) -> Iterator[SourceRow]:
    loaded = _load_splits(pin, None)
    ids_by_split = {
        split: _native_ids(pin, split, dataset)
        for split, dataset in loaded.items()
    }
    _require_unique_across_splits(pin, ids_by_split)
    for split in _selected_splits(loaded, splits):
        dataset = loaded[split]
        for native_id, fields in zip(
            ids_by_split[split], dataset, strict=True
        ):
            yield SourceRow(
                pin=pin,
                origin_split=split,
                native_id=native_id,
                fields=fields,
            )


def split_sizes(pin: DatasetPin) -> Mapping[str, int]:
    return {
        split: dataset.num_rows
        for split, dataset in _load_splits(pin, None).items()
    }


def _load_splits(
    pin: DatasetPin, splits: Sequence[str] | None
) -> dict[str, Dataset]:
    dataset_dict = load_dataset(pin.repo_id, pin.config, revision=pin.revision)
    loaded = {str(name): ds for name, ds in dataset_dict.items()}
    if splits is None:
        return loaded
    return {split: loaded[split] for split in _selected_splits(loaded, splits)}


def _selected_splits(
    loaded: Mapping[str, Dataset], splits: Sequence[str] | None
) -> list[str]:
    if splits is None:
        return list(loaded)
    if len(set(splits)) != len(splits):
        msg = f"splits must be unique, got {list(splits)!r}"
        raise ValueError(msg)
    unknown = [split for split in splits if split not in loaded]
    if unknown:
        msg = f"unknown splits {unknown!r}; available: {list(loaded)!r}"
        raise ValueError(msg)
    return list(splits)


def _require_unique_across_splits(
    pin: DatasetPin, ids_by_split: Mapping[str, Sequence[str]]
) -> None:
    owner: dict[str, str] = {}
    for split, native_ids in ids_by_split.items():
        for native_id in native_ids:
            first = owner.setdefault(native_id, split)
            if first != split:
                msg = (
                    f"native id {native_id!r} of {pin.repo_id!r} appears in "
                    f"splits {first!r} and {split!r}; native ids must be "
                    f"unique across the pinned dataset (use ROW_INDEX for "
                    "datasets whose id field repeats across splits)"
                )
                raise NativeIdError(msg)


def _native_ids(pin: DatasetPin, split: str, dataset: Dataset) -> list[str]:
    if pin.native_id_field == ROW_INDEX:
        return [f"{split}:{position}" for position in range(dataset.num_rows)]
    if pin.native_id_field not in dataset.column_names:
        msg = (
            f"native id field {pin.native_id_field!r} is not a column of "
            f"{pin.repo_id!r} split {split!r}; columns: "
            f"{dataset.column_names!r}"
        )
        raise NativeIdError(msg)
    native_ids: list[str] = []
    seen: set[str] = set()
    for position, value in enumerate(dataset[pin.native_id_field]):
        if value is None:
            msg = (
                f"null native id at row {position} of {pin.repo_id!r} "
                f"split {split!r}"
            )
            raise NativeIdError(msg)
        native_id = str(value)
        if native_id in seen:
            msg = (
                f"duplicate native id {native_id!r} in {pin.repo_id!r} "
                f"split {split!r}"
            )
            raise NativeIdError(msg)
        seen.add(native_id)
        native_ids.append(native_id)
    return native_ids
