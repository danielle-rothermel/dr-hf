"""Hugging Face Hub identity layer for dr-* research infrastructure."""

from __future__ import annotations

__version__ = "0.1.2"

from .branches import (
    create_branch_metadata,
    extract_seed_from_branch,
    extract_step_from_branch,
    get_all_repo_branches,
    get_checkpoint_branches,
    get_step_range_for_seed,
    group_branches_by_seed,
    is_checkpoint_branch,
    parse_branch_name,
    sort_branches_by_step,
)
from .datasets import NativeIdError, SourceRow, read_rows, split_sizes
from .hashing import content_hash
from .io import commit_dataset_files_to_hf
from .location import HFLocation, HFRepoID, HFResource
from .models import (
    BranchInfo,
    BranchMetadata,
    DatasetCommitResult,
    DatasetFileCommitEntry,
    SeedBranchInfo,
    SeedConfiguration,
)
from .pins import (
    ROW_INDEX,
    CommitSha,
    DatasetPin,
    ModelPin,
    resolve_dataset_pin,
    resolve_model_pin,
)
from .publishing import (
    MODEL_CARD_FILENAME,
    PROVENANCE_FILENAME,
    ModelProvenance,
    TagConflictError,
    publish_model,
)
from .subsets import (
    Derivation,
    DerivationKind,
    NamedSubset,
    PartitionShare,
    SplitKey,
    SubsetConflictError,
    SubsetNotFoundError,
    SubsetRegistry,
    exclude,
    explicit,
    origin_key,
    origin_subsets,
    seeded_partition,
    seeded_sample,
    union,
)

__all__ = [
    "MODEL_CARD_FILENAME",
    "PROVENANCE_FILENAME",
    "ROW_INDEX",
    "BranchInfo",
    "BranchMetadata",
    "CommitSha",
    "DatasetCommitResult",
    "DatasetFileCommitEntry",
    "DatasetPin",
    "Derivation",
    "DerivationKind",
    "HFLocation",
    "HFRepoID",
    "HFResource",
    "ModelPin",
    "ModelProvenance",
    "NamedSubset",
    "NativeIdError",
    "PartitionShare",
    "SeedBranchInfo",
    "SeedConfiguration",
    "SourceRow",
    "SplitKey",
    "SubsetConflictError",
    "SubsetNotFoundError",
    "SubsetRegistry",
    "TagConflictError",
    "__version__",
    "commit_dataset_files_to_hf",
    "content_hash",
    "create_branch_metadata",
    "exclude",
    "explicit",
    "extract_seed_from_branch",
    "extract_step_from_branch",
    "get_all_repo_branches",
    "get_checkpoint_branches",
    "get_step_range_for_seed",
    "group_branches_by_seed",
    "is_checkpoint_branch",
    "origin_key",
    "origin_subsets",
    "parse_branch_name",
    "publish_model",
    "read_rows",
    "resolve_dataset_pin",
    "resolve_model_pin",
    "seeded_partition",
    "seeded_sample",
    "sort_branches_by_step",
    "split_sizes",
    "union",
]
