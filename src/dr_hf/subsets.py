from __future__ import annotations

import json
import math
import random
import re
import tempfile
from enum import StrEnum, unique
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from .identity import NAMED_SUBSET_SCHEMA, identity_hash

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable, Mapping

    from .datasets import SourceRow
    from .pins import DatasetPin

__all__ = [
    "Derivation",
    "DerivationKind",
    "NamedSubset",
    "PartitionShare",
    "SplitKey",
    "SubsetConflictError",
    "SubsetNotFoundError",
    "SubsetRegistry",
    "exclude",
    "explicit",
    "origin_key",
    "origin_subsets",
    "seeded_partition",
    "seeded_sample",
    "union",
]

_KEY_SEPARATOR = "@"
_PIN_HASH_RE = re.compile(r"[0-9a-f]{64}")


class SubsetConflictError(ValueError): ...


class SubsetNotFoundError(LookupError): ...


def _validate_pin_hash(value: str) -> str:
    if not _PIN_HASH_RE.fullmatch(value):
        msg = f"pin_hash must be a 64-character sha256 hex digest: {value!r}"
        raise ValueError(msg)
    return value


PinHash = Annotated[str, AfterValidator(_validate_pin_hash)]
KeyPart = Annotated[str, Field(min_length=1, pattern=r"^[^@]+$")]


class SplitKey(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: KeyPart
    version: KeyPart

    def __str__(self) -> str:
        return f"{self.name}{_KEY_SEPARATOR}{self.version}"

    @classmethod
    def parse(cls, text: str) -> SplitKey:
        name, separator, version = text.partition(_KEY_SEPARATOR)
        if not separator:
            msg = f"split key must have the form 'name@version': {text!r}"
            raise ValueError(msg)
        return cls(name=name, version=version)


@unique
class DerivationKind(StrEnum):
    ORIGIN_SPLIT = "origin_split"
    SEEDED_SAMPLE = "seeded_sample"
    SEEDED_PARTITION = "seeded_partition"
    EXCLUSION = "exclusion"
    UNION = "union"
    EXPLICIT_IDS = "explicit_ids"


class PartitionShare(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: KeyPart
    fraction: float = Field(gt=0, le=1)


_OPTIONAL_FIELDS = (
    "parents",
    "origin_split",
    "seed",
    "size",
    "fraction",
    "partition",
    "excluded_native_ids",
)
_FIELDS_BY_KIND: dict[DerivationKind, frozenset[str]] = {
    DerivationKind.ORIGIN_SPLIT: frozenset({"origin_split"}),
    DerivationKind.SEEDED_SAMPLE: frozenset({"parents", "seed", "size"}),
    DerivationKind.SEEDED_PARTITION: frozenset(
        {"parents", "seed", "fraction", "partition"}
    ),
    DerivationKind.EXCLUSION: frozenset({"parents", "excluded_native_ids"}),
    DerivationKind.UNION: frozenset({"parents"}),
    DerivationKind.EXPLICIT_IDS: frozenset(),
}


class Derivation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: DerivationKind
    parents: tuple[SplitKey, ...] = ()
    origin_split: str | None = None
    seed: int | None = Field(default=None, ge=0)
    size: int | None = Field(default=None, ge=1)
    fraction: float | None = Field(default=None, gt=0, le=1)
    partition: tuple[PartitionShare, ...] = ()
    excluded_native_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _fields_match_kind(self) -> Self:
        allowed = _FIELDS_BY_KIND[self.kind]
        for field in _OPTIONAL_FIELDS:
            value = getattr(self, field)
            is_set = value is not None and value != ()
            if field in allowed and not is_set:
                msg = f"{self.kind} derivation requires {field}"
                raise ValueError(msg)
            if field not in allowed and is_set:
                msg = f"{self.kind} derivation does not take {field}"
                raise ValueError(msg)
        if self.kind is DerivationKind.UNION:
            if len(self.parents) < 2:  # noqa: PLR2004
                raise ValueError("union derivation requires two parents")
        elif self.parents and len(self.parents) != 1:
            msg = f"{self.kind} derivation takes exactly one parent"
            raise ValueError(msg)
        if self.partition:
            names = [share.name for share in self.partition]
            if len(set(names)) != len(names):
                raise ValueError("partition share names must be unique")
            if _exact_total(self.partition) > 1:
                raise ValueError("partition fractions sum to more than 1")
        return self


class NamedSubset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pin_hash: PinHash
    key: SplitKey
    native_ids: tuple[str, ...] = Field(min_length=1)
    derivation: Derivation

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if len(set(self.native_ids)) != len(self.native_ids):
            msg = f"native_ids of {self.key} must be unique"
            raise ValueError(msg)
        derivation = self.derivation
        if (
            derivation.kind is DerivationKind.SEEDED_SAMPLE
            and len(self.native_ids) != derivation.size
        ):
            msg = f"{self.key} has {len(self.native_ids)} ids, not size"
            raise ValueError(msg)
        if derivation.kind is DerivationKind.SEEDED_PARTITION:
            share = PartitionShare(
                name=self.key.name,
                fraction=derivation.fraction or 0.0,
            )
            if share not in derivation.partition:
                msg = f"{self.key} is not a share of its partition"
                raise ValueError(msg)
        return self

    @property
    def content_hash(self) -> str:
        return identity_hash(NAMED_SUBSET_SCHEMA, self.model_dump(mode="json"))


def origin_key(split: str, pin: DatasetPin) -> SplitKey:
    return SplitKey(name=split, version=f"origin-{pin.revision[:7]}")


def origin_subsets(
    pin: DatasetPin, rows: Iterable[SourceRow]
) -> tuple[NamedSubset, ...]:
    ids_by_split: dict[str, list[str]] = {}
    for row in rows:
        if row.pin != pin:
            msg = f"row {row.native_id!r} belongs to a different pin"
            raise ValueError(msg)
        ids_by_split.setdefault(row.origin_split, []).append(row.native_id)
    return tuple(
        NamedSubset(
            pin_hash=pin.pin_hash,
            key=origin_key(split, pin),
            native_ids=tuple(native_ids),
            derivation=Derivation(
                kind=DerivationKind.ORIGIN_SPLIT, origin_split=split
            ),
        )
        for split, native_ids in ids_by_split.items()
    )


def seeded_sample(
    parent: NamedSubset, *, name: str, version: str, size: int, seed: int
) -> NamedSubset:
    if not 1 <= size <= len(parent.native_ids):
        msg = (
            f"size must be between 1 and {len(parent.native_ids)} "
            f"(size of {parent.key}), got {size}"
        )
        raise ValueError(msg)
    chosen = set(_seeded_order(parent.native_ids, seed)[:size])
    return NamedSubset(
        pin_hash=parent.pin_hash,
        key=SplitKey(name=name, version=version),
        native_ids=_in_parent_order(parent, chosen),
        derivation=Derivation(
            kind=DerivationKind.SEEDED_SAMPLE,
            parents=(parent.key,),
            seed=seed,
            size=size,
        ),
    )


def seeded_partition(
    parent: NamedSubset,
    *,
    names: Mapping[str, float],
    version: str,
    seed: int,
) -> tuple[NamedSubset, ...]:
    if not names:
        raise ValueError("names must contain at least one part")
    shares = tuple(
        PartitionShare(name=name, fraction=fraction)
        for name, fraction in names.items()
    )
    total = _exact_total(shares)
    if total > 1:
        msg = f"partition fractions sum to {float(total)}, more than 1"
        raise ValueError(msg)
    order = _seeded_order(parent.native_ids, seed)
    size = len(order)
    cumulative = Fraction()
    start = 0
    parts: list[NamedSubset] = []
    for share in shares:
        cumulative += Fraction(repr(share.fraction))
        end = math.floor(size * cumulative)
        if end == start:
            msg = f"part {share.name!r} is empty for parent size {size}"
            raise ValueError(msg)
        parts.append(
            NamedSubset(
                pin_hash=parent.pin_hash,
                key=SplitKey(name=share.name, version=version),
                native_ids=_in_parent_order(parent, set(order[start:end])),
                derivation=Derivation(
                    kind=DerivationKind.SEEDED_PARTITION,
                    parents=(parent.key,),
                    seed=seed,
                    fraction=share.fraction,
                    partition=shares,
                ),
            )
        )
        start = end
    return tuple(parts)


def exclude(
    parent: NamedSubset,
    *,
    name: str,
    version: str,
    native_ids: Iterable[str],
) -> NamedSubset:
    excluded = set(native_ids)
    if not excluded:
        raise ValueError("native_ids to exclude must be non-empty")
    unknown = excluded - set(parent.native_ids)
    if unknown:
        msg = f"ids not in {parent.key}: {sorted(unknown)!r}"
        raise ValueError(msg)
    return NamedSubset(
        pin_hash=parent.pin_hash,
        key=SplitKey(name=name, version=version),
        native_ids=tuple(i for i in parent.native_ids if i not in excluded),
        derivation=Derivation(
            kind=DerivationKind.EXCLUSION,
            parents=(parent.key,),
            excluded_native_ids=tuple(sorted(excluded)),
        ),
    )


def union(
    parents: Iterable[NamedSubset], *, name: str, version: str
) -> NamedSubset:
    parent_list = list(parents)
    if len(parent_list) < 2:  # noqa: PLR2004
        raise ValueError("union requires at least two parents")
    if len({parent.pin_hash for parent in parent_list}) > 1:
        raise ValueError("union parents must share one pin")
    members = dict.fromkeys(
        native_id for parent in parent_list for native_id in parent.native_ids
    )
    return NamedSubset(
        pin_hash=parent_list[0].pin_hash,
        key=SplitKey(name=name, version=version),
        native_ids=tuple(members),
        derivation=Derivation(
            kind=DerivationKind.UNION,
            parents=tuple(parent.key for parent in parent_list),
        ),
    )


def explicit(
    pin: DatasetPin,
    *,
    name: str,
    version: str,
    native_ids: Iterable[str],
) -> NamedSubset:
    return NamedSubset(
        pin_hash=pin.pin_hash,
        key=SplitKey(name=name, version=version),
        native_ids=tuple(native_ids),
        derivation=Derivation(kind=DerivationKind.EXPLICIT_IDS),
    )


def _exact_total(shares: Iterable[PartitionShare]) -> Fraction:
    return sum(
        (Fraction(repr(share.fraction)) for share in shares), Fraction()
    )


def _seeded_order(native_ids: Iterable[str], seed: int) -> list[str]:
    if seed < 0:
        msg = f"seed must be a non-negative int, got {seed}"
        raise ValueError(msg)
    rng = random.Random(seed)  # noqa: S311
    order = list(native_ids)
    for i in range(len(order) - 1, 0, -1):
        j = int(rng.random() * (i + 1))
        order[i], order[j] = order[j], order[i]
    return order


def _in_parent_order(parent: NamedSubset, chosen: set[str]) -> tuple[str, ...]:
    return tuple(i for i in parent.native_ids if i in chosen)


class _RegistryFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pin_hash: PinHash
    subsets: tuple[NamedSubset, ...]


class SubsetRegistry:
    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)

    def register(self, subset: NamedSubset) -> None:
        subsets = self._load(subset.pin_hash)
        existing = subsets.get(subset.key)
        if existing is not None:
            if existing.content_hash == subset.content_hash:
                return
            msg = (
                f"{subset.key} is registered for pin {subset.pin_hash} with "
                f"content hash {existing.content_hash}, not "
                f"{subset.content_hash}"
            )
            raise SubsetConflictError(msg)
        subsets[subset.key] = subset
        self._write(subset.pin_hash, subsets)

    def get(self, pin_hash: str, key: SplitKey) -> NamedSubset:
        subset = self._load(pin_hash).get(key)
        if subset is None:
            msg = f"no subset {key} registered for pin {pin_hash}"
            raise SubsetNotFoundError(msg)
        return subset

    def keys(
        self, pin_hash: str, name: str | None = None
    ) -> tuple[SplitKey, ...]:
        return tuple(
            key
            for key in sorted(
                self._load(pin_hash), key=lambda k: (k.name, k.version)
            )
            if name is None or key.name == name
        )

    def _path(self, pin_hash: str) -> Path:
        return self.directory / f"{_validate_pin_hash(pin_hash)}.json"

    def _load(self, pin_hash: str) -> dict[SplitKey, NamedSubset]:
        path = self._path(pin_hash)
        if not path.exists():
            return {}
        stored = _RegistryFile.model_validate_json(path.read_bytes())
        if stored.pin_hash != pin_hash or any(
            subset.pin_hash != pin_hash for subset in stored.subsets
        ):
            msg = f"registry file {path} holds subsets of another pin"
            raise ValueError(msg)
        subsets = {subset.key: subset for subset in stored.subsets}
        if len(subsets) != len(stored.subsets):
            msg = f"registry file {path} repeats a split key"
            raise ValueError(msg)
        return subsets

    def _write(
        self, pin_hash: str, subsets: Mapping[SplitKey, NamedSubset]
    ) -> None:
        ordered = tuple(
            subsets[key]
            for key in sorted(subsets, key=lambda k: (k.name, k.version))
        )
        payload = _RegistryFile(pin_hash=pin_hash, subsets=ordered)
        text = json.dumps(
            payload.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(pin_hash)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.directory,
            prefix=f".{pin_hash}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(text + "\n")
            except BaseException:
                handle.close()
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(path)
