# datasets

Reading every split of a pinned Hub dataset as rows tagged with origin split
and native id.

## Types

```python
class SourceRow(BaseModel, frozen=True):
    pin: DatasetPin
    origin_split: str
    native_id: str
    fields: Mapping[str, object]

class NativeIdError(ValueError): ...
```

## Functions

```python
def read_rows(
    pin: DatasetPin, *, splits: Sequence[str] | None = None
) -> Iterator[SourceRow]

def split_sizes(pin: DatasetPin) -> Mapping[str, int]
```

`read_rows` calls `datasets.load_dataset(repo_id, config,
revision=pin.revision)` (one call per split when `splits` is given). Before
yielding any row it checks every requested split: the native id field must be
a column, values must be non-null, and their `str` forms must be unique
within the split; otherwise it raises `NativeIdError`. `split_sizes` returns
the row count of every provided split.

## Usage

```python
from dr_hf import read_rows, resolve_dataset_pin, split_sizes

pin = resolve_dataset_pin(
    "allenai/ai2_arc", config="ARC-Challenge", native_id_field="id"
)
split_sizes(pin)  # {"train": 1119, "test": 1172, "validation": 299}
for row in read_rows(pin, splits=["test"]):
    print(row.native_id, row.fields["question"])
```
