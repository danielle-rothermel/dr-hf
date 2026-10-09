# dr-hf

[Definitions](https://danielle-rothermel.github.io/dr-hf/) ·
[Terms](https://github.com/danielle-rothermel/dr-hf/blob/main/.defs/terms.toml) ·
[Contracts](https://github.com/danielle-rothermel/dr-hf/blob/main/.defs/contracts.toml)

Hugging Face Hub identity layer for dr-* research infrastructure: dataset
and model pins fixed at commit SHAs, an all-split dataset reader, named
subsets with recorded derivations, checkpoint branch parsing, and publishing
of converted models with provenance. Loading models and running inference
live in dr-providers.

## Installation

```bash
uv add dr-hf
```

Runtime dependencies: `datasets`, `huggingface_hub`, `pydantic`.

## Quick start

```python
from dr_hf import (
    SubsetRegistry,
    origin_subsets,
    read_rows,
    resolve_dataset_pin,
    seeded_sample,
)

# Pin a dataset config at the commit "main" points to now.
pin = resolve_dataset_pin(
    "allenai/ai2_arc", config="ARC-Challenge", native_id_field="id"
)

# Read every split; each row carries its origin split and native id.
rows = list(read_rows(pin))

# Origin subsets are keyed (split, "origin-<sha7>"); derive more from them.
origins = {s.key.name: s for s in origin_subsets(pin, rows)}
dev = seeded_sample(origins["test"], name="dev", version="v1", size=100, seed=0)

registry = SubsetRegistry("subsets/")
registry.register(dev)  # same key + different content -> SubsetConflictError
```

## Surface

| Module | Purpose | Key exports |
|--------|---------|-------------|
| **pins** | Hub repos fixed at full commit SHAs | `DatasetPin`, `ModelPin`, `ROW_INDEX`, `CommitSha`, `resolve_dataset_pin`, `resolve_model_pin` |
| **datasets** | All-split reader | `SourceRow`, `read_rows`, `split_sizes`, `NativeIdError` |
| **subsets** | Named subsets and registry | `SplitKey`, `Derivation`, `DerivationKind`, `PartitionShare`, `NamedSubset`, `origin_key`, `origin_subsets`, `seeded_sample`, `seeded_partition`, `exclude`, `union`, `explicit`, `SubsetRegistry`, `SubsetConflictError`, `SubsetNotFoundError` |
| **identity** | Identity hashes through dr-serialize | `identity_hash`, `DATASET_PIN_SCHEMA`, `NAMED_SUBSET_SCHEMA`, `IDENTITY_SCHEMA_VERSION` |
| **publishing** | Model upload with card and provenance | `ModelProvenance`, `publish_model`, `TagConflictError`, `MODEL_CARD_FILENAME`, `PROVENANCE_FILENAME` |
| **branches** | Checkpoint branch discovery and `stepN-seed-*` parsing | `get_checkpoint_branches`, `parse_branch_name`, `create_branch_metadata`, ... |
| **io** | Atomic multi-file dataset commits | `commit_dataset_files_to_hf` |
| **location** | Dataset repository references | `HFLocation`, `HFRepoID`, `HFResource` |
| **models** | Branch metadata and commit models | `BranchInfo`, `BranchMetadata`, `DatasetFileCommitEntry`, `DatasetCommitResult`, ... |

Key rules (see [contracts](.defs/contracts.toml)):

- Pins and provenance accept only full 40-character commit SHAs.
- `read_rows` validates that native ids exist, are non-null, and are unique
  across every split of the pinned dataset before yielding any row; datasets
  whose id field repeats across splits use `ROW_INDEX`.
- Seeded samples and partitions follow a documented rule driven only by
  `random.Random(seed).random()`, so membership is reproducible from the
  derivation and parent alone.
- The subset registry maps one `(pin_hash, split key)` to one content hash.
- `publish_model` uploads weights, `README.md`, and `provenance.json` in one
  commit and never moves an existing tag.
- Every hash is a dr-serialize identity hash of an identity document with a
  dr-hf-owned schema name (`dr_hf.dataset_pin`, `dr_hf.named_subset`) and
  schema version 1; dr-hf computes no digests of its own.

## Publishing a converted model

```python
from dr_hf import ModelProvenance, publish_model

pin = publish_model(
    "converted/step1000-seed0",
    repo_id="drotherm/DataDecide-dolma1_7-150M",
    branch="step1000-seed0",
    tags=(),
    card_markdown=card_text,
    provenance=ModelProvenance(
        source_repo_id="allenai/DataDecide-dolma1_7-150M",
        source_revision=source_sha,
        conversion_tool="datadec.models.convert 0.1.0",
        verification={"max_abs_logit_diff": 2.1e-5, "batch": "8 x 256"},
    ),
    private=True,
    commit_message="Add step1000 seed 0",
)
```

## Documentation

- [Definitions site](https://danielle-rothermel.github.io/dr-hf/) — shared vocabulary and binding contracts ([terms](.defs/terms.toml), [contracts](.defs/contracts.toml); agents read the TOML directly)
- [Changelog](CHANGELOG.md)

## Development

Install dependencies and the commit hook once per clone:

```bash
uv sync --locked
uv run pre-commit install
```

The hook runs `scripts/pre-check.sh` (TOML lint of `.defs`, Ruff format,
Ruff lint, ty) followed by the test suite. Hub-backed conformance tests are
opt-in:

```bash
uv run pytest -m hub
```

## License

MIT
