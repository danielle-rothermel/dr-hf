from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, HttpUrl


class BranchInfo(BaseModel):
    branch: str
    valid: bool = False
    step: int | None = None
    seed: str | None = None


class SeedBranchInfo(BaseModel):
    step: int
    branch: str


class SeedConfiguration(BaseModel):
    count: int
    step_range: tuple[int, int]
    branches: list[SeedBranchInfo]


class BranchMetadata(BaseModel):
    repo_id: str
    last_updated: datetime
    total_branches: int
    checkpoint_branches: int
    seed_configurations: dict[str, SeedConfiguration]
    other_branches: list[str]
    all_checkpoint_branches: list[str]


class DatasetFileCommitEntry(BaseModel):
    local_path: Path
    repo_path: str


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
