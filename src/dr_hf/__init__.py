from __future__ import annotations

__version__ = "0.1.1"

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
from .configs import (
    analyze_model_config,
    download_config_file,
    estimate_parameter_count,
    extract_model_architecture_info,
)
from .datasets import (
    download_dataset,
    load_or_download_dataset,
    sanitize_repo_name,
)
from .io import (
    cached_download_tables_from_hf,
    commit_dataset_files_to_hf,
    get_tables_from_cache,
    read_local_parquet_paths,
)
from .location import (
    HFLocation,
    HFRepoID,
    HFResource,
)
from .models import DatasetCommitResult, DatasetFileCommitEntry
from .paths import (
    get_data_dir,
    get_repo_dir,
)

__all__ = [
    "DatasetCommitResult",
    "DatasetFileCommitEntry",
    "HFLocation",
    "HFRepoID",
    "HFResource",
    "__version__",
    "analyze_model_config",
    "cached_download_tables_from_hf",
    "commit_dataset_files_to_hf",
    "create_branch_metadata",
    "download_config_file",
    "download_dataset",
    "estimate_parameter_count",
    "extract_model_architecture_info",
    "extract_seed_from_branch",
    "extract_step_from_branch",
    "get_all_repo_branches",
    "get_checkpoint_branches",
    "get_data_dir",
    "get_repo_dir",
    "get_step_range_for_seed",
    "get_tables_from_cache",
    "group_branches_by_seed",
    "is_checkpoint_branch",
    "load_or_download_dataset",
    "parse_branch_name",
    "read_local_parquet_paths",
    "sanitize_repo_name",
    "sort_branches_by_step",
]


def __getattr__(name: str):
    if name == "query_hf_with_duckdb":
        from .io import query_hf_with_duckdb

        return query_hf_with_duckdb

    weights_exports = {
        "analyze_layer_structure",
        "analyze_model_weights",
        "calculate_global_weight_stats",
        "calculate_tensor_stats",
        "calculate_weight_statistics",
        "discover_model_weight_files",
        "download_model_weights",
    }
    if name in weights_exports:
        from . import weights

        return getattr(weights, name)

    checkpoint_exports = {
        "analyze_complete_checkpoint",
        "analyze_optimizer_checkpoint",
        "create_comprehensive_summary",
        "create_learning_rate_summary",
        "download_optimizer_checkpoint",
        "process_all_checkpoints",
        "process_single_checkpoint",
        "save_all_analyses_outputs",
        "save_checkpoint_analysis",
    }
    if name in checkpoint_exports:
        from . import checkpoints

        return getattr(checkpoints, name)

    model_exports = {
        "ArchitectureInfo",
        "BranchInfo",
        "BranchMetadata",
        "CheckpointAnalysis",
        "CheckpointComponents",
        "CheckpointSummaryRow",
        "ConfigAnalysis",
        "GlobalWeightStats",
        "LayerAnalysis",
        "LayerCategorization",
        "LayerCounts",
        "LearningRateInfo",
        "OptimizerAnalysis",
        "OptimizerComponentInfo",
        "ParamGroupInfo",
        "ParameterEstimate",
        "ParameterStats",
        "SeedBranchInfo",
        "SeedConfiguration",
        "TensorInfo",
        "TensorStats",
        "WeightFileStatistics",
        "WeightsAnalysis",
        "WeightsSummary",
    }
    if name in model_exports:
        from . import models

        return getattr(models, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
