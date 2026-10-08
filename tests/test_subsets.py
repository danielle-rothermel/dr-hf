from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from dr_hf import (
    DatasetPin,
    Derivation,
    DerivationKind,
    NamedSubset,
    PartitionShare,
    SourceRow,
    SplitKey,
    SubsetConflictError,
    SubsetNotFoundError,
    SubsetRegistry,
    content_hash,
    exclude,
    explicit,
    origin_key,
    origin_subsets,
    seeded_partition,
    seeded_sample,
    union,
)

if TYPE_CHECKING:
    from pathlib import Path

SHA = "91c54ad1727ee830252e457677f467be0bfd8a57"
IDS = tuple(str(i) for i in range(10))


@pytest.fixture
def pin() -> DatasetPin:
    return DatasetPin(
        repo_id="org/data", config=None, revision=SHA, native_id_field="id"
    )


@pytest.fixture
def parent(pin: DatasetPin) -> NamedSubset:
    return explicit(pin, name="all", version="v1", native_ids=IDS)


# ------------------------------------------------------- persisted literals


def test_content_hash_encoding_is_pinned() -> None:
    payload = {"b": [1, "é"], "a": None}
    expected = hashlib.sha256('{"a":null,"b":[1,"é"]}'.encode()).hexdigest()
    assert content_hash(payload) == expected
    assert expected == (
        "f8f17faab95c024891d173fa43442b0e52007736a1f36715ac721ab22deeefc5"
    )


def test_content_hash_rejects_nan() -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        content_hash({"x": float("nan")})


def test_derivation_kind_literals_are_pinned() -> None:
    assert [kind.value for kind in DerivationKind] == [
        "origin_split",
        "seeded_sample",
        "seeded_partition",
        "exclusion",
        "union",
        "explicit_ids",
    ]


def test_pin_and_subset_hashes_are_pinned(
    pin: DatasetPin, parent: NamedSubset
) -> None:
    assert pin.pin_hash == (
        "3d64868b913b8c67f0b1457373ae422436d9a6cfff586b591999dd506317caec"
    )
    assert parent.content_hash == (
        "c911511162ce90e2c32fbb9cae31b97f1cdd33f3ed56bb690abc6728f562b120"
    )


# ---------------------------------------------------------------- split key


def test_split_key_string_round_trip() -> None:
    key = SplitKey(name="test", version="origin-91c54ad")
    assert str(key) == "test@origin-91c54ad"
    assert SplitKey.parse(str(key)) == key


@pytest.mark.parametrize("text", ["test", "@v1", "test@", "a@b@c"])
def test_split_key_parse_rejects_malformed(text: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        SplitKey.parse(text)


# ------------------------------------------------------------ derivations


def test_origin_subsets_group_rows_by_split(pin: DatasetPin) -> None:
    rows = [
        SourceRow(pin=pin, origin_split=split, native_id=nid, fields={})
        for split, nid in [("train", "a"), ("test", "c"), ("train", "b")]
    ]
    train, test = origin_subsets(pin, rows)
    assert train.key == SplitKey(name="train", version="origin-91c54ad")
    assert train.key == origin_key("train", pin)
    assert train.native_ids == ("a", "b")
    assert test.native_ids == ("c",)
    assert train.derivation == Derivation(
        kind=DerivationKind.ORIGIN_SPLIT, origin_split="train"
    )
    assert train.pin_hash == pin.pin_hash


def test_origin_subsets_reject_foreign_rows(pin: DatasetPin) -> None:
    other = pin.model_copy(update={"config": "other"})
    row = SourceRow(pin=other, origin_split="train", native_id="a", fields={})
    with pytest.raises(ValueError, match="different pin"):
        origin_subsets(pin, [row])


def test_seeded_sample_rule_is_pinned(parent: NamedSubset) -> None:
    sample = seeded_sample(parent, name="s", version="v1", size=4, seed=0)
    # Seeded order for seed 0 is 9 4 0 5 2 7 1 3 6 8; the first four ids
    # are listed in parent order.
    assert sample.native_ids == ("0", "4", "5", "9")
    assert sample.derivation == Derivation(
        kind=DerivationKind.SEEDED_SAMPLE,
        parents=(parent.key,),
        seed=0,
        size=4,
    )


def test_seeded_sample_is_deterministic_and_seed_sensitive(
    parent: NamedSubset,
) -> None:
    def ids(seed: int) -> tuple[str, ...]:
        return seeded_sample(
            parent, name="s", version="v1", size=5, seed=seed
        ).native_ids

    assert ids(7) == ids(7)
    assert len({ids(seed) for seed in range(10)}) > 1


@pytest.mark.parametrize("size", [0, 11])
def test_seeded_sample_rejects_bad_size(
    parent: NamedSubset, size: int
) -> None:
    with pytest.raises(ValueError, match="size must be between"):
        seeded_sample(parent, name="s", version="v1", size=size, seed=0)


def test_seeded_sample_rejects_negative_seed(parent: NamedSubset) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seeded_sample(parent, name="s", version="v1", size=2, seed=-1)


def test_seeded_partition_rule_is_pinned(parent: NamedSubset) -> None:
    parts = seeded_partition(
        parent, names={"a": 0.5, "b": 0.3, "c": 0.2}, version="v1", seed=0
    )
    # Cuts of the seed-0 order at floor(10 * 0.5) and floor(10 * 0.8).
    assert [p.native_ids for p in parts] == [
        ("0", "2", "4", "5", "9"),
        ("1", "3", "7"),
        ("6", "8"),
    ]
    assert [str(p.key) for p in parts] == ["a@v1", "b@v1", "c@v1"]
    assert parts[1].derivation.fraction == 0.3
    assert [s.name for s in parts[1].derivation.partition] == ["a", "b", "c"]


def test_seeded_partition_parts_are_disjoint_and_cover(
    parent: NamedSubset,
) -> None:
    for seed in range(20):
        parts = seeded_partition(
            parent, names={"x": 0.7, "y": 0.3}, version="v1", seed=seed
        )
        members = [set(p.native_ids) for p in parts]
        assert not members[0] & members[1]
        assert members[0] | members[1] == set(IDS)
        assert [len(m) for m in members] == [7, 3]


def test_seeded_partition_may_leave_a_remainder(parent: NamedSubset) -> None:
    parts = seeded_partition(
        parent, names={"x": 0.25, "y": 0.25}, version="v1", seed=3
    )
    assert [len(p.native_ids) for p in parts] == [2, 3]


def test_seeded_partition_rejects_bad_fractions(parent: NamedSubset) -> None:
    with pytest.raises(ValueError, match="more than 1"):
        seeded_partition(
            parent, names={"x": 0.7, "y": 0.4}, version="v1", seed=0
        )
    with pytest.raises(ValueError, match="empty"):
        seeded_partition(
            parent, names={"x": 0.96, "y": 0.01}, version="v1", seed=0
        )
    with pytest.raises(ValidationError):
        seeded_partition(parent, names={"x": 0.0}, version="v1", seed=0)


def test_exclude_keeps_parent_order(parent: NamedSubset) -> None:
    subset = exclude(parent, name="clean", version="v1", native_ids=["3", "0"])
    assert subset.native_ids == ("1", "2", "4", "5", "6", "7", "8", "9")
    assert subset.derivation.excluded_native_ids == ("0", "3")


def test_exclude_rejects_unknown_ids(parent: NamedSubset) -> None:
    with pytest.raises(ValueError, match="ids not in all@v1"):
        exclude(parent, name="clean", version="v1", native_ids=["nope"])


def test_union_keeps_first_occurrence(pin: DatasetPin) -> None:
    left = explicit(pin, name="l", version="v1", native_ids=["b", "a"])
    right = explicit(pin, name="r", version="v1", native_ids=["a", "c"])
    merged = union([left, right], name="u", version="v1")
    assert merged.native_ids == ("b", "a", "c")
    assert merged.derivation.parents == (left.key, right.key)


def test_union_rejects_mixed_pins_and_single_parent(pin: DatasetPin) -> None:
    other = pin.model_copy(update={"config": "other"})
    left = explicit(pin, name="l", version="v1", native_ids=["a"])
    right = explicit(other, name="r", version="v1", native_ids=["b"])
    with pytest.raises(ValueError, match="share one pin"):
        union([left, right], name="u", version="v1")
    with pytest.raises(ValueError, match="two parents"):
        union([left], name="u", version="v1")


def test_explicit_rejects_duplicates_and_empty(pin: DatasetPin) -> None:
    with pytest.raises(ValidationError, match="unique"):
        explicit(pin, name="e", version="v1", native_ids=["a", "a"])
    with pytest.raises(ValidationError):
        explicit(pin, name="e", version="v1", native_ids=[])


def test_derivation_fields_must_match_kind() -> None:
    parent_key = SplitKey(name="p", version="v1")
    with pytest.raises(ValidationError, match="requires seed"):
        Derivation(kind=DerivationKind.SEEDED_SAMPLE, parents=(parent_key,))
    with pytest.raises(ValidationError, match="does not take seed"):
        Derivation(kind=DerivationKind.EXPLICIT_IDS, seed=1)
    with pytest.raises(ValidationError, match="exactly one parent"):
        Derivation(
            kind=DerivationKind.EXCLUSION,
            parents=(parent_key, parent_key),
            excluded_native_ids=("a",),
        )


def test_sample_size_must_match_member_count(parent: NamedSubset) -> None:
    sample = seeded_sample(parent, name="s", version="v1", size=3, seed=1)
    tampered = {**sample.model_dump(), "native_ids": ("0",)}
    with pytest.raises(ValidationError, match="not size"):
        NamedSubset.model_validate(tampered)


# ----------------------------------------------------------------- registry


def test_registry_register_get_keys(
    tmp_path: Path, pin: DatasetPin, parent: NamedSubset
) -> None:
    registry = SubsetRegistry(tmp_path / "subsets")
    sample = seeded_sample(parent, name="s", version="v1", size=3, seed=1)
    registry.register(sample)
    registry.register(parent)
    assert registry.get(pin.pin_hash, sample.key) == sample
    assert registry.keys(pin.pin_hash) == (parent.key, sample.key)
    assert registry.keys(pin.pin_hash, name="s") == (sample.key,)
    reopened = SubsetRegistry(tmp_path / "subsets")
    assert reopened.get(pin.pin_hash, parent.key) == parent
    assert [p.name for p in (tmp_path / "subsets").iterdir()] == [
        f"{pin.pin_hash}.json"
    ]


def test_registry_same_content_is_noop_and_conflict_raises(
    tmp_path: Path, pin: DatasetPin, parent: NamedSubset
) -> None:
    registry = SubsetRegistry(tmp_path)
    registry.register(parent)
    before = (tmp_path / f"{pin.pin_hash}.json").read_bytes()
    registry.register(parent.model_copy())
    assert (tmp_path / f"{pin.pin_hash}.json").read_bytes() == before
    changed = explicit(pin, name="all", version="v1", native_ids=IDS[:-1])
    with pytest.raises(SubsetConflictError, match="all@v1"):
        registry.register(changed)
    assert registry.get(pin.pin_hash, parent.key) == parent


def test_registry_missing_key_raises(tmp_path: Path, pin: DatasetPin) -> None:
    registry = SubsetRegistry(tmp_path)
    with pytest.raises(SubsetNotFoundError):
        registry.get(pin.pin_hash, SplitKey(name="x", version="v1"))
    assert registry.keys(pin.pin_hash) == ()


def test_registry_rejects_non_hash_pin(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sha256"):
        SubsetRegistry(tmp_path).keys("../escape")


def test_registry_file_shape_is_pinned(
    tmp_path: Path, pin: DatasetPin
) -> None:
    subset = explicit(pin, name="e", version="v1", native_ids=["b", "a"])
    SubsetRegistry(tmp_path).register(subset)
    stored = json.loads((tmp_path / f"{pin.pin_hash}.json").read_text())
    assert stored == {
        "pin_hash": pin.pin_hash,
        "subsets": [
            {
                "pin_hash": pin.pin_hash,
                "key": {"name": "e", "version": "v1"},
                "native_ids": ["b", "a"],
                "derivation": {
                    "kind": "explicit_ids",
                    "parents": [],
                    "origin_split": None,
                    "seed": None,
                    "size": None,
                    "fraction": None,
                    "partition": [],
                    "excluded_native_ids": [],
                },
            }
        ],
    }


def test_partition_part_must_be_a_share(parent: NamedSubset) -> None:
    part = seeded_partition(parent, names={"x": 0.5}, version="v1", seed=0)[0]
    renamed = {**part.model_dump(), "key": {"name": "z", "version": "v1"}}
    with pytest.raises(ValidationError, match="not a share"):
        NamedSubset.model_validate(renamed)


def test_partition_derivation_rejects_incoherent_shares() -> None:
    parent_key = SplitKey(name="p", version="v1")
    with pytest.raises(ValidationError, match="unique"):
        Derivation(
            kind=DerivationKind.SEEDED_PARTITION,
            parents=(parent_key,),
            seed=0,
            fraction=0.5,
            partition=(
                PartitionShare(name="a", fraction=0.5),
                PartitionShare(name="a", fraction=0.5),
            ),
        )
    with pytest.raises(ValidationError, match="more than 1"):
        Derivation(
            kind=DerivationKind.SEEDED_PARTITION,
            parents=(parent_key,),
            seed=0,
            fraction=0.6,
            partition=(
                PartitionShare(name="a", fraction=0.6),
                PartitionShare(name="b", fraction=0.6),
            ),
        )
