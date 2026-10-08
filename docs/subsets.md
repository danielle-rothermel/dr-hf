# subsets

Named subsets of one dataset pin, their derivations, and a file registry.

## Types

```python
class SplitKey(BaseModel, frozen=True):    # str form "name@version"
    name: str
    version: str
    @classmethod
    def parse(cls, text: str) -> SplitKey: ...

class DerivationKind(StrEnum):
    ORIGIN_SPLIT = "origin_split"
    SEEDED_SAMPLE = "seeded_sample"
    SEEDED_PARTITION = "seeded_partition"
    EXCLUSION = "exclusion"
    UNION = "union"
    EXPLICIT_IDS = "explicit_ids"

class PartitionShare(BaseModel, frozen=True):
    name: str
    fraction: float                       # 0 < fraction <= 1

class Derivation(BaseModel, frozen=True):
    kind: DerivationKind
    parents: tuple[SplitKey, ...] = ()
    origin_split: str | None = None
    seed: int | None = None
    size: int | None = None
    fraction: float | None = None
    partition: tuple[PartitionShare, ...] = ()
    excluded_native_ids: tuple[str, ...] = ()

class NamedSubset(BaseModel, frozen=True):
    pin_hash: str
    key: SplitKey
    native_ids: tuple[str, ...]           # unique, non-empty, ordered
    derivation: Derivation
    content_hash: str                     # property
```

Each derivation kind sets exactly its own fields (`origin_split`; one parent
with `seed` and `size`; one parent with `seed`, `fraction`, and the full
`partition`; one parent with `excluded_native_ids`; two or more parents;
nothing for explicit ids). The content hash covers pin hash, key, ordered
ids, and derivation.

## Derivations

```python
def origin_key(split: str, pin: DatasetPin) -> SplitKey
def origin_subsets(pin, rows) -> tuple[NamedSubset, ...]
def seeded_sample(parent, *, name, version, size, seed) -> NamedSubset
def seeded_partition(parent, *, names: Mapping[str, float], version, seed)
    -> tuple[NamedSubset, ...]
def exclude(parent, *, name, version, native_ids) -> NamedSubset
def union(parents, *, name, version) -> NamedSubset
def explicit(pin, *, name, version, native_ids) -> NamedSubset
```

Origin keys are `(split, "origin-<first 7 SHA characters>")`.

Seeded rules, reproducible from a derivation and its parent alone:

1. Seeded order: copy the parent's ids, create `random.Random(seed)` with a
   non-negative int seed, and for `i` from `n - 1` down to `1` swap
   positions `i` and `int(rng.random() * (i + 1))`. Only `Random.random`,
   whose output Python guarantees across versions, is used.
2. Sample: the first `size` ids of the seeded order.
3. Partition: read each fraction as the exact decimal
   `Fraction(repr(fraction))`; cumulative sums must not exceed 1; part `k`
   is the slice `[floor(n * c_(k-1)), floor(n * c_k))` of the seeded order.
   Parts are disjoint and cover the parent when the fractions sum to 1. An
   empty part raises.
4. Members of every derived subset are listed in parent order (unions keep
   the first occurrence across parents in the given order). Exclusion ids
   must all be in the parent. Explicit ids are not checked against the
   dataset.

## Registry

```python
class SubsetRegistry:
    def __init__(self, directory: str | PathLike[str]) -> None
    def register(self, subset: NamedSubset) -> None
    def get(self, pin_hash: str, key: SplitKey) -> NamedSubset
    def keys(self, pin_hash: str, name: str | None = None) -> tuple[SplitKey, ...]
```

One JSON file per pin hash (`<directory>/<pin_hash>.json`). One
`(pin_hash, key)` maps to one content hash: registering identical content is
a no-op and different content raises `SubsetConflictError`. Missing keys
raise `SubsetNotFoundError`. Files are replaced atomically; concurrent
writers to the same pin hash are not coordinated.

## Usage

```python
from dr_hf import (
    SubsetRegistry, origin_subsets, read_rows, resolve_dataset_pin,
    seeded_partition,
)

pin = resolve_dataset_pin(
    "allenai/ai2_arc", config="ARC-Challenge", native_id_field="id"
)
origins = {s.key.name: s for s in origin_subsets(pin, read_rows(pin))}
optimize, select = seeded_partition(
    origins["train"], names={"optimize": 0.8, "select": 0.2},
    version="v1", seed=0,
)
registry = SubsetRegistry("subsets/")
for subset in (*origins.values(), optimize, select):
    registry.register(subset)
```
