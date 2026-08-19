# io

HfApi upload/download operations for files and parquet datasets.

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

### cached_download_tables_from_hf
```python
def cached_download_tables_from_hf(
    hf_loc: HFLocation,
    *,
    cache_dir: str | Path,
    hf_token: str | None = None,
    force_download: bool = False,
    verbose: bool = True
) -> dict[str, str | Path]
```
Download parquet files from HuggingFace with local caching. Returns a dictionary mapping remote filepaths to local file paths. Files are only downloaded if they don't exist locally (unless `force_download=True`).

### get_tables_from_cache
```python
def get_tables_from_cache(
    hf_loc: HFLocation,
    cache_dir: str | Path
) -> dict[str, pd.DataFrame]
```
Read parquet files from local cache directory. Returns a dictionary mapping file stems to DataFrames.

### read_local_parquet_paths
```python
def read_local_parquet_paths(
    local_paths: list[str] | list[Path]
) -> dict[str, pd.DataFrame]
```
Read parquet files from a list of local paths. Returns a dictionary mapping file stems to DataFrames.

### query_hf_with_duckdb (requires `[duckdb]`)
```python
def query_hf_with_duckdb(
    hf_loc: HFLocation,
    connection: duckdb.DuckDBPyConnection
) -> dict[str, pd.DataFrame]
```
Query a HuggingFace dataset directly using DuckDB. Requires a DuckDB connection object. Returns a dictionary mapping file stems to DataFrames.

## Usage

```python
from pathlib import Path
from dr_hf import (
    commit_dataset_files_to_hf,
    DatasetFileCommitEntry,
    cached_download_tables_from_hf,
    query_hf_with_duckdb,
    HFLocation,
)

# Atomically commit multiple files to a dataset repo
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
    hf_token="hf_...",
)
print(f"Committed: {result.commit_oid}")
print(f"Files: {result.file_urls}")

# Download parquet tables with caching
loc = HFLocation(
    org="allenai",
    repo_name="c4",
    filepaths=["en/train-00000-of-01024.parquet"]
)
tables = cached_download_tables_from_hf(
    loc,
    cache_dir=Path("./cache"),
    hf_token="hf_..."
)
# tables is a dict: {"en/train-00000-of-01024": Path("./cache/...")}

# Read cached tables as DataFrames
from dr_hf import get_tables_from_cache
dfs = get_tables_from_cache(loc, cache_dir=Path("./cache"))
# dfs is a dict: {"en/train-00000-of-01024": pd.DataFrame}

# Query with DuckDB (requires [duckdb] optional dependency)
import duckdb  # Optional: install with pip install dr-hf[duckdb]
conn = duckdb.connect()
loc = HFLocation.from_uri("hf://datasets/squad/squad")
results = query_hf_with_duckdb(loc, conn)
```
