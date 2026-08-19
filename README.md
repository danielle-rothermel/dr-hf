# dr-hf

[Definitions](https://danielle-rothermel.github.io/dr-hf/) ·
[Terms](https://github.com/danielle-rothermel/dr-hf/blob/main/.defs/terms.toml) ·
[Contracts](https://github.com/danielle-rothermel/dr-hf/blob/main/.defs/contracts.toml)

HuggingFace utilities for repository management, dataset operations, and model analysis.

## Installation

```bash
uv add dr-hf
```

For model weight analysis (requires PyTorch):
```bash
uv add dr-hf[weights]
```

For DuckDB query support:
```bash
uv add dr-hf[duckdb]
```

## Quick Start

```python
from dr_hf import (
    get_checkpoint_branches,
    parse_branch_name,
    HFLocation,
    download_dataset,
)

# Parse checkpoint branches from a repo
branches = get_checkpoint_branches("org/model-checkpoints")
for branch in branches:
    info = parse_branch_name(branch)
    print(f"Step {info.step}, Seed: {info.seed}")

# Create a location reference for HF datasets
loc = HFLocation(org="allenai", repo_name="my-dataset")
print(loc.repo_uri)  # hf://datasets/allenai/my-dataset

# Download a dataset to local parquet
from pathlib import Path
download_dataset(Path("./data/squad_train.parquet"), repo_id="squad", split="train")
```

## Module Overview

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| **branches** | Branch discovery & parsing | `get_checkpoint_branches`, `parse_branch_name`, `create_branch_metadata` |
| **configs** | Model config analysis | `download_config_file`, `analyze_model_config`, `estimate_parameter_count` |
| **weights** | Model weight analysis | `analyze_model_weights`, `calculate_weight_statistics` ⚡ |
| **checkpoints** | Checkpoint orchestration | `analyze_complete_checkpoint`, `process_all_checkpoints` ⚡ |
| **datasets** | Dataset loading & caching | `load_or_download_dataset`, `download_dataset` |
| **io** | HfApi upload/download | `commit_dataset_files_to_hf`, `cached_download_tables_from_hf` |
| **location** | HF resource URIs | `HFLocation`, `HFRepoID`, `HFResource` |
| **paths** | Environment paths | `get_data_dir`, `get_repo_dir` |
| **models** | Pydantic data models | `BranchInfo`, `ConfigAnalysis`, `WeightsAnalysis`, ... |

⚡ = Requires `[weights]` optional dependency

## Documentation

- [Definitions site](https://danielle-rothermel.github.io/dr-hf/) — shared vocabulary and binding contracts ([terms](.defs/terms.toml), [contracts](.defs/contracts.toml); agents read the TOML directly)
- [Full API Reference](docs/api.md)
- Module guides: [branches](docs/branches.md) | [configs](docs/configs.md) | [weights](docs/weights.md) | [checkpoints](docs/checkpoints.md) | [datasets](docs/datasets.md) | [io](docs/io.md) | [location](docs/location.md) | [paths](docs/paths.md)
- [Pydantic Models](docs/models.md)
- [Recipes & Patterns](docs/recipes.md)

### Auto-generated API Docs

```bash
# Serve interactive docs locally
uv run pdoc dr_hf

# Generate static HTML
uv run pdoc dr_hf -o docs/api_html
```

## Quick Reference

### Branch Operations
```python
from dr_hf import (
    get_all_repo_branches,    # list all branches in repo
    get_checkpoint_branches,  # filter to stepN-seed-* branches
    is_checkpoint_branch,     # check if branch matches pattern
    parse_branch_name,        # extract step/seed -> BranchInfo
    extract_step_from_branch, # get step number
    extract_seed_from_branch, # get seed string
    sort_branches_by_step,    # sort branches by step
    group_branches_by_seed,   # group branches by seed
    create_branch_metadata,   # full repo metadata -> BranchMetadata
)
```

### Config Analysis
```python
from dr_hf import (
    download_config_file,           # download config.json
    analyze_model_config,           # parse config -> ConfigAnalysis
    extract_model_architecture_info,# extract architecture -> ArchitectureInfo
    estimate_parameter_count,       # estimate params -> ParameterEstimate
)
```

### Weight Analysis (requires `[weights]`)
```python
from dr_hf import (
    discover_model_weight_files,  # find weight files in repo
    download_model_weights,       # download specific weights
    calculate_weight_statistics,  # analyze weights -> WeightFileStatistics
    calculate_tensor_stats,       # per-tensor stats -> TensorStats
    analyze_layer_structure,      # categorize layers -> LayerAnalysis
    calculate_global_weight_stats,# global stats -> GlobalWeightStats
    analyze_model_weights,        # full workflow -> WeightsAnalysis
)
```

### Checkpoint Analysis (requires `[weights]`)
```python
from dr_hf import (
    download_optimizer_checkpoint, # download optim.pt
    analyze_optimizer_checkpoint,  # parse optimizer -> OptimizerAnalysis
    analyze_complete_checkpoint,   # full analysis -> CheckpointAnalysis
    process_single_checkpoint,     # single branch analysis
    process_all_checkpoints,       # parallel multi-branch
    create_comprehensive_summary,  # DataFrame summary
    create_learning_rate_summary,  # LR-focused summary
    save_checkpoint_analysis,      # save to JSON
    save_all_analyses_outputs,     # save CSVs + JSON
)
```

### Dataset Operations
```python
from dr_hf import (
    load_or_download_dataset, # load from cache or download
    download_dataset,         # download HF dataset to parquet
    sanitize_repo_name,       # convert repo ID to safe filename
)
```

### HfApi I/O
```python
from dr_hf import (
    commit_dataset_files_to_hf,   # atomic multi-file dataset commit
    DatasetFileCommitEntry,       # local path + repo path pair
    cached_download_tables_from_hf,# download parquet with caching
    get_tables_from_cache,        # read cached parquet files
    read_local_parquet_paths,     # list local parquet files
    query_hf_with_duckdb,         # query HF with DuckDB (requires [duckdb])
)
```

### Location Management
```python
from dr_hf import (
    HFLocation,   # Pydantic model for HF dataset locations
    HFRepoID,     # Type alias: "org/repo-name"
    HFResource,   # Type alias: "hf://datasets/org/repo"
)

loc = HFLocation(org="allenai", repo_name="c4")
loc.repo_id        # "allenai/c4"
loc.repo_uri       # "hf://datasets/allenai/c4"
loc.repo_link      # HttpUrl to HF page

# Parse from URI
loc = HFLocation.from_uri("hf://datasets/squad/squad")
```

### Environment Paths
```python
from dr_hf import (
    get_data_dir,  # get DATA_DIR from env
    get_repo_dir,  # get REPO_DIR from env
)
```

### Pydantic Models
```python
from dr_hf import (
    # Branch models
    BranchInfo,           # parsed branch (step, seed, valid)
    SeedBranchInfo,       # step + branch name
    SeedConfiguration,    # seed group metadata
    BranchMetadata,       # full repo branch info

    # Config models
    ConfigAnalysis,       # config.json analysis result
    ArchitectureInfo,     # model architecture details
    ParameterEstimate,    # estimated parameter counts

    # Weight models
    WeightsAnalysis,      # full weight analysis result
    WeightsSummary,       # aggregated weight stats
    WeightFileStatistics, # per-file statistics
    TensorInfo,           # per-tensor metadata
    TensorStats,          # tensor statistics
    LayerAnalysis,        # layer categorization
    LayerCategorization,  # layers by type
    LayerCounts,          # layer count summary
    GlobalWeightStats,    # global weight statistics
    ParameterStats,       # parameter counts

    # Checkpoint models
    CheckpointAnalysis,   # full checkpoint analysis
    CheckpointComponents, # optimizer + config + weights
    CheckpointSummaryRow, # DataFrame row model
    OptimizerAnalysis,    # optimizer state analysis
    OptimizerComponentInfo,# optimizer component details
    LearningRateInfo,     # learning rate extraction
    ParamGroupInfo,       # param group details
)
```

## Development

Install dependencies and the commit hook once per clone:

```bash
uv sync --locked
uv run pre-commit install
```

The hook runs `scripts/pre-check.sh` for Ruff formatting, Ruff lint, and type
checking, followed by the test suite.

## License

MIT
