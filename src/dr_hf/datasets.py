"""Reading pinned Hub datasets as rows tagged with origin split and id."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from datasets import Dataset, load_dataset
from pydantic import BaseModel, ConfigDict

from .pins import ROW_INDEX, DatasetPin

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = ["NativeIdError", "SourceRow", "read_rows", "split_sizes"]


class NativeIdError(ValueError):
    """The native id field is missing, null, or not unique within a split."""


class SourceRow(BaseModel):
    """One dataset row with its pin, origin split, and native id."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pin: DatasetPin
    origin_split: str
    native_id: str
    fields: Mapping[str, object]


def read_rows(
    pin: DatasetPin, *, splits: Sequence[str] | None = None
) -> Iterator[SourceRow]:
    """Yield every row of the requested splits at ``pin.revision``.

    With ``splits=None`` every split the dataset provides is read, in the
    order ``datasets`` reports them. Native ids are validated for every
    requested split before the first row is yielded: the field must exist,
    values must be non-null, and their ``str`` forms must be unique within a
    split. With ``native_id_field == ROW_INDEX`` the id is
    ``f"{origin_split}:{row_position}"``.
    """
    loaded = _load_splits(pin, splits)
    ids_by_split = {
        split: _native_ids(pin, split, dataset)
        for split, dataset in loaded.items()
    }
    for split, dataset in loaded.items():
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
    """Return the row count of every split the dataset provides."""
    return {
        split: dataset.num_rows
        for split, dataset in _load_splits(pin, None).items()
    }


def _load_splits(
    pin: DatasetPin, splits: Sequence[str] | None
) -> dict[str, Dataset]:
    if splits is None:
        dataset_dict = load_dataset(
            pin.repo_id, pin.config, revision=pin.revision
        )
        return {str(name): ds for name, ds in dataset_dict.items()}
    if len(set(splits)) != len(splits):
        msg = f"splits must be unique, got {list(splits)!r}"
        raise ValueError(msg)
    return {
        split: load_dataset(
            pin.repo_id, pin.config, split=split, revision=pin.revision
        )
        for split in splits
    }


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
