# models

Pydantic data models for type-safe data handling across all modules.

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

## Config Models

### ConfigAnalysis
```python
class ConfigAnalysis(BaseModel):
    available: bool = False
    error: str | None = None
    raw_config: dict | None = None
    architecture_info: ArchitectureInfo | None = None
    config_keys: list[str] = []
    config_type: str | None = None
```
Result of config.json analysis.

### ArchitectureInfo
```python
class ArchitectureInfo(BaseModel):
    hidden_size: int | None = None
    num_layers: int | None = None
    num_attention_heads: int | None = None
    intermediate_size: int | None = None
    vocab_size: int | None = None
    max_position_embeddings: int | None = None
    model_type: str | None = None
    # ... additional fields
    estimated_parameters: ParameterEstimate | None = None
```
Standardized architecture information extracted from config.

### ParameterEstimate
```python
class ParameterEstimate(BaseModel):
    embedding_parameters: int
    per_layer_parameters: int
    total_layer_parameters: int
    output_head_parameters: int
    estimated_total_parameters: int

    # Computed fields
    estimated_total_millions: float  # computed from estimated_total_parameters
```
Estimated parameter counts with computed scaling fields.

## Weight Models

### WeightsAnalysis
```python
class WeightsAnalysis(BaseModel):
    available: bool = False
    error: str | None = None
    discovered_files: list[str] = []
    file_analyses: dict[str, WeightFileStatistics] = {}
    summary: WeightsSummary | None = None

    # Computed field
    weights_available: bool  # alias for available
```
Complete weight analysis result.

### WeightsSummary
```python
class WeightsSummary(BaseModel):
    total_files_analyzed: int = 0
    total_parameters: int = 0
    total_size_mb: float = 0.0

    # Computed fields
    total_parameters_millions: float
    total_parameters_billions: float
    total_size_gb: float
```
Aggregated weight statistics with computed scaling fields.

### WeightFileStatistics
```python
class WeightFileStatistics(BaseModel):
    file_path: str
    file_size_mb: float
    num_tensors: int
    tensor_info: list[TensorInfo] = []
    parameter_stats: ParameterStats | None = None
    layer_analysis: LayerAnalysis | None = None
    global_statistics: GlobalWeightStats | None = None
    error: str | None = None
```
Per-file weight statistics.

### TensorInfo
```python
class TensorInfo(BaseModel):
    name: str
    shape: list[int]
    dtype: str
    parameters: int
    size_mb: float
    statistics: TensorStats | None = None
```
Per-tensor metadata.

### TensorStats
```python
class TensorStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    median: float
    abs_mean: float
    zero_fraction: float
    percentile_25: float | None = None
    percentile_75: float | None = None
    percentile_90: float | None = None
    percentile_95: float | None = None
    percentile_99: float | None = None
```
Tensor statistics.

### ParameterStats
```python
class ParameterStats(BaseModel):
    total_parameters: int

    # Computed fields
    total_parameters_millions: float
    total_parameters_billions: float
```
Parameter counts with computed scaling fields.

### LayerAnalysis
```python
class LayerAnalysis(BaseModel):
    layer_categorization: LayerCategorization
    layer_counts: LayerCounts
    transformer_layer_indices: list[int] = []
```
Layer structure analysis.

### LayerCategorization
```python
class LayerCategorization(BaseModel):
    embedding_layers: list[str] = []
    transformer_layers: list[str] = []
    output_layers: list[str] = []
    layer_norm_layers: list[str] = []
    attention_layers: list[str] = []
    feedforward_layers: list[str] = []
    other_layers: list[str] = []
```
Layers categorized by type.

### LayerCounts
```python
class LayerCounts(BaseModel):
    total_layers: int = 0
    attention_heads: int = 0
    feedforward_layers: int = 0
    layer_norms: int = 0
    estimated_transformer_layers: int = 0
```
Layer count summary.

### GlobalWeightStats
```python
class GlobalWeightStats(BaseModel):
    global_mean: float
    global_std: float
    global_min: float
    global_max: float
    global_abs_mean: float
    global_zero_fraction: float
    global_percentile_1: float | None = None
    # ... additional percentiles
```
Global statistics across all weights.

## Checkpoint Models

### CheckpointAnalysis
```python
class CheckpointAnalysis(BaseModel):
    branch: str
    step: int
    error: str | None = None
    components: CheckpointComponents = CheckpointComponents()
```
Complete checkpoint analysis.

### CheckpointComponents
```python
class CheckpointComponents(BaseModel):
    optimizer: OptimizerAnalysis = OptimizerAnalysis()
    config: ConfigAnalysis = ConfigAnalysis()
    weights: WeightsAnalysis = WeightsAnalysis()
```
Container for all checkpoint component analyses.

### CheckpointSummaryRow
```python
class CheckpointSummaryRow(BaseModel):
    branch: str
    step: int
    optimizer_available: bool = False
    config_available: bool = False
    weights_available: bool = False
    current_lr: float | None = None
    # ... additional fields

    @classmethod
    def from_analysis(cls, analysis: CheckpointAnalysis) -> CheckpointSummaryRow
```
DataFrame row model with factory method.

### OptimizerAnalysis
```python
class OptimizerAnalysis(BaseModel):
    available: bool = False
    error: str | None = None
    checkpoint_keys: list[str] = []
    checkpoint_type: str | None = None
    optimizer_info: dict[str, OptimizerComponentInfo] = {}
    learning_rate_info: LearningRateInfo | None = None
```
Optimizer checkpoint analysis.

### OptimizerComponentInfo
```python
class OptimizerComponentInfo(BaseModel):
    type: str
    size: int | str = "N/A"
    num_param_groups: int | None = None
    param_group_keys: list[str] = []
    learning_rate: float | None = None
    initial_learning_rate: float | None = None
    num_parameters: int | None = None
    state_keys: list[str] = []
```
Optimizer component details.

### LearningRateInfo
```python
class LearningRateInfo(BaseModel):
    param_group_lrs: list[ParamGroupInfo] = []
    lr_scheduler_present: bool = False
    lr_scheduler_keys: list[str] = []
    scheduler_last_epoch: int | None = None
```
Learning rate information extraction.

### ParamGroupInfo
```python
class ParamGroupInfo(BaseModel):
    group_index: int
    current_lr: float | None = None
    initial_lr: float | None = None
    weight_decay: float | None = None
    momentum: float | None = None
```
Per-parameter-group optimizer settings.

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
    commit_oid: str
    commit_url: HttpUrl
    commit_message: str
    commit_description: str
    pr_url: HttpUrl | None = None
    pr_num: int | None = None
    pr_revision: str | None = None
    file_urls: dict[str, HttpUrl]
```
Structured result from `commit_dataset_files_to_hf()`, including commit metadata, optional PR details, and per-file resolve URLs at the target revision.

## Computed Fields

Several models use Pydantic `@computed_field` to derive values from base fields:

- `ParameterStats.total_parameters_millions` - computed from `total_parameters`
- `ParameterStats.total_parameters_billions` - computed from `total_parameters`
- `ParameterEstimate.estimated_total_millions` - computed from `estimated_total_parameters`
- `WeightsSummary.total_parameters_millions` - computed from `total_parameters`
- `WeightsSummary.total_parameters_billions` - computed from `total_parameters`
- `WeightsSummary.total_size_gb` - computed from `total_size_mb`
- `WeightsAnalysis.weights_available` - alias for `available`

Computed fields are included in `model_dump()` output automatically.
