# io

Atomic multi-file commits to Hugging Face dataset repositories.

## Functions

### commit_dataset_files_to_hf
```python
def commit_dataset_files_to_hf(
    files: list[DatasetFileCommitEntry],
    hf_loc: HFLocation,
    *,
    revision: str,
    expected_parent: str,
    commit_message: str,
    commit_description: str | None = None,
    create_pr: bool = False,
    hf_token: str | None = None,
) -> DatasetCommitResult
```
Publish multiple local files to a Hugging Face dataset repository in one atomic commit. Each entry pairs a local path with a unique relative POSIX repository path. All inputs are validated before any remote mutation. Requires `expected_parent`: for direct commits it enforces optimistic concurrency against the target `revision`; for `create_pr=True` it names the PR base only. Returns `created=False` when the Hub performs a matching-parent no-op. Set `create_pr=True` to open a Hub PR instead of committing directly.

## Usage

```python
from pathlib import Path
from dr_hf import (
    commit_dataset_files_to_hf,
    DatasetFileCommitEntry,
    HFLocation,
)

loc = HFLocation(org="username", repo_name="my-dataset")
result = commit_dataset_files_to_hf(
    [
        DatasetFileCommitEntry(
            local_path=Path("results/a.parquet"),
            repo_path="data/a.parquet",
        ),
        DatasetFileCommitEntry(
            local_path=Path("results/b.parquet"),
            repo_path="data/b.parquet",
        ),
    ],
    loc,
    revision="main",
    expected_parent="abc1234567890",
    commit_message="Add evaluation results",
)
print(f"Committed: {result.commit_oid}")
print(f"Files: {result.file_urls}")
```
