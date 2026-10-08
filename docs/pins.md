# pins

Dataset and model pins: Hub repositories fixed at a full commit SHA.

## Types

```python
CommitSha = Annotated[str, ...]   # 40-character lowercase hex
ROW_INDEX = "__row__"

class DatasetPin(BaseModel, frozen=True):
    repo_id: str
    config: str | None
    revision: CommitSha
    native_id_field: str          # a column name, or ROW_INDEX
    pin_hash: str                 # property: content_hash of the fields

class ModelPin(BaseModel, frozen=True):
    repo_id: str
    revision: CommitSha
```

A branch or tag name is rejected as a `revision`; resolve it first.
`native_id_field=ROW_INDEX` gives each row the id
`f"{origin_split}:{row_position}"`.

## Functions

```python
def resolve_dataset_pin(
    repo_id: str, *, config: str | None = None, ref: str = "main",
    native_id_field: str,
) -> DatasetPin

def resolve_model_pin(repo_id: str, *, ref: str = "main") -> ModelPin
```

Both resolve `ref` (branch, tag, or SHA) through `HfApi.dataset_info` or
`HfApi.model_info` and record the returned commit SHA. The config and native
id field are checked when rows are read, not at resolution.

## Usage

```python
from dr_hf import resolve_dataset_pin, resolve_model_pin

arc = resolve_dataset_pin(
    "allenai/ai2_arc", config="ARC-Challenge", native_id_field="id"
)
model = resolve_model_pin("allenai/DataDecide-dolma1_7-150M", ref="main")
```

`content_hash` (in `dr_hf.hashing`) is the only hash function used for pin
and subset identities. It will be replaced by dr-serialize canonical hashing,
which changes every hash.
