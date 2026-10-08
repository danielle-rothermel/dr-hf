# models

Pydantic models for checkpoint branch metadata and dataset commits. Pin, subset, and provenance models are documented with their modules: [pins](pins.md), [subsets](subsets.md), [publishing](publishing.md).

## Branch Models

### BranchInfo
```python
class BranchInfo(BaseModel):
    branch: str
    valid: bool = False
    step: int | None = None
    seed: str | None = None
```
Parsed branch information from `parse_branch_name()`.

### SeedBranchInfo
```python
class SeedBranchInfo(BaseModel):
    step: int
    branch: str
```
Step and branch name pair for seed configurations.

### SeedConfiguration
```python
class SeedConfiguration(BaseModel):
    count: int
    step_range: tuple[int, int]
    branches: list[SeedBranchInfo]
```
Metadata for a seed group.

### BranchMetadata
```python
class BranchMetadata(BaseModel):
    repo_id: str
    last_updated: datetime
    total_branches: int
    checkpoint_branches: int
    seed_configurations: dict[str, SeedConfiguration]
    other_branches: list[str]
    all_checkpoint_branches: list[str]
```
Complete branch metadata for a repository.

## I/O Models

### DatasetFileCommitEntry
```python
class DatasetFileCommitEntry(BaseModel):
    local_path: Path
    repo_path: str
```
Local file paired with its target relative POSIX path in a dataset repository.

### DatasetCommitResult
```python
class DatasetCommitResult(BaseModel):
    created: bool
    commit_oid: str
    commit_url: HttpUrl
    commit_message: str
    commit_description: str
    pr_url: HttpUrl | None = None
    pr_num: int | None = None
    pr_revision: str | None = None
    file_urls: dict[str, HttpUrl]
```
Structured result from `commit_dataset_files_to_hf()`, including whether a new commit was created, commit metadata, optional PR details, and per-file resolve URLs at the applicable revision.
