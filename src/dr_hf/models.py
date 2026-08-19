from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, HttpUrl, computed_field


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


class TensorInfo(BaseModel):
    name: str
    shape: list[int]
    dtype: str
    parameters: int
    size_mb: float
    statistics: TensorStats | None = None


class LayerCategorization(BaseModel):
    embedding_layers: list[str] = []
    transformer_layers: list[str] = []
    output_layers: list[str] = []
    layer_norm_layers: list[str] = []
    attention_layers: list[str] = []
    feedforward_layers: list[str] = []
    other_layers: list[str] = []


class LayerCounts(BaseModel):
    total_layers: int = 0
    attention_heads: int = 0
    feedforward_layers: int = 0
    layer_norms: int = 0
    estimated_transformer_layers: int = 0


class LayerAnalysis(BaseModel):
    layer_categorization: LayerCategorization
    layer_counts: LayerCounts
    transformer_layer_indices: list[int] = []


class GlobalWeightStats(BaseModel):
    global_mean: float
    global_std: float
    global_min: float
    global_max: float
    global_abs_mean: float
    global_zero_fraction: float
    global_percentile_1: float | None = None
    global_percentile_5: float | None = None
    global_percentile_25: float | None = None
    global_percentile_50: float | None = None
    global_percentile_75: float | None = None
    global_percentile_95: float | None = None
    global_percentile_99: float | None = None


class ParameterStats(BaseModel):
    total_parameters: int

    @computed_field
    @property
    def total_parameters_millions(self) -> float:
        return round(self.total_parameters / 1_000_000, 2)

    @computed_field
    @property
    def total_parameters_billions(self) -> float:
        return round(self.total_parameters / 1_000_000_000, 3)


class WeightFileStatistics(BaseModel):
    file_path: str
    file_size_mb: float
    num_tensors: int
    tensor_info: list[TensorInfo] = []
    parameter_stats: ParameterStats | None = None
    layer_analysis: LayerAnalysis | None = None
    global_statistics: GlobalWeightStats | None = None
    error: str | None = None


class ParamGroupInfo(BaseModel):
    group_index: int
    current_lr: float | None = None
    initial_lr: float | None = None
    weight_decay: float | None = None
    momentum: float | None = None


class LearningRateInfo(BaseModel):
    param_group_lrs: list[ParamGroupInfo] = []
    lr_scheduler_present: bool = False
    lr_scheduler_keys: list[str] = []
    scheduler_last_epoch: int | None = None


class OptimizerComponentInfo(BaseModel):
    type: str
    size: int | str = "N/A"
    num_param_groups: int | None = None
    param_group_keys: list[str] = []
    learning_rate: float | None = None
    initial_learning_rate: float | None = None
    num_parameters: int | None = None
    state_keys: list[str] = []


class OptimizerAnalysis(BaseModel):
    available: bool = False
    error: str | None = None
    checkpoint_keys: list[str] = []
    checkpoint_type: str | None = None
    optimizer_info: dict[str, OptimizerComponentInfo] = {}
    learning_rate_info: LearningRateInfo | None = None


class ParameterEstimate(BaseModel):
    embedding_parameters: int
    per_layer_parameters: int
    total_layer_parameters: int
    output_head_parameters: int
    estimated_total_parameters: int

    @computed_field
    @property
    def estimated_total_millions(self) -> float:
        return round(self.estimated_total_parameters / 1_000_000, 1)


class ArchitectureInfo(BaseModel):
    hidden_size: int | None = None
    num_layers: int | None = None
    num_attention_heads: int | None = None
    intermediate_size: int | None = None
    vocab_size: int | None = None
    max_position_embeddings: int | None = None
    sequence_length: int | None = None
    model_type: str | None = None
    activation_function: str | None = None
    layer_norm_eps: float | None = None
    dropout: float | None = None
    pad_token_id: int | None = None
    eos_token_id: int | None = None
    bos_token_id: int | None = None
    torch_dtype: str | None = None
    use_cache: bool | None = None
    tie_word_embeddings: bool | None = None
    rope_scaling: dict | None = None
    estimated_parameters: ParameterEstimate | None = None


class ConfigAnalysis(BaseModel):
    available: bool = False
    error: str | None = None
    raw_config: dict | None = None
    architecture_info: ArchitectureInfo | None = None
    config_keys: list[str] = []
    config_type: str | None = None


class WeightsSummary(BaseModel):
    total_files_analyzed: int = 0
    total_parameters: int = 0
    total_size_mb: float = 0.0

    @computed_field
    @property
    def total_parameters_millions(self) -> float:
        return round(self.total_parameters / 1_000_000, 2)

    @computed_field
    @property
    def total_parameters_billions(self) -> float:
        return round(self.total_parameters / 1_000_000_000, 3)

    @computed_field
    @property
    def total_size_gb(self) -> float:
        return round(self.total_size_mb / 1024, 3)


class WeightsAnalysis(BaseModel):
    available: bool = False
    error: str | None = None
    discovered_files: list[str] = []
    file_analyses: dict[str, WeightFileStatistics] = {}
    summary: WeightsSummary | None = None

    @computed_field
    @property
    def weights_available(self) -> bool:
        """Alias for `available`, kept for API/serialization stability.

        This property mirrors `available` to provide a consistent naming
        convention in serialized outputs (e.g., JSON) where
        `weights_available` may be expected by external consumers or for
        consistency with other component fields.
        """
        return self.available


class CheckpointComponents(BaseModel):
    optimizer: OptimizerAnalysis = OptimizerAnalysis()
    config: ConfigAnalysis = ConfigAnalysis()
    weights: WeightsAnalysis = WeightsAnalysis()


class CheckpointAnalysis(BaseModel):
    branch: str
    step: int
    error: str | None = None
    components: CheckpointComponents = CheckpointComponents()


class CheckpointSummaryRow(BaseModel):
    branch: str
    step: int
    optimizer_available: bool = False
    config_available: bool = False
    weights_available: bool = False
    current_lr: float | None = None
    weight_decay: float | None = None
    momentum: float | None = None
    num_param_groups: int | None = None
    model_type: str | None = None
    hidden_size: int | None = None
    num_layers: int | None = None
    vocab_size: int | None = None
    estimated_params_millions: float | None = None
    total_weight_params_millions: float | None = None
    optimizer_error: str = ""
    config_error: str = ""
    weights_error: str = ""

    @classmethod
    def from_analysis(
        cls, analysis: CheckpointAnalysis
    ) -> CheckpointSummaryRow:
        row = cls(branch=analysis.branch, step=analysis.step)

        opt = analysis.components.optimizer
        row.optimizer_available = opt.available
        if opt.available and opt.learning_rate_info:
            lr_info = opt.learning_rate_info
            if lr_info.param_group_lrs:
                first = lr_info.param_group_lrs[0]
                row.current_lr = first.current_lr
                row.weight_decay = first.weight_decay
                row.momentum = first.momentum
                row.num_param_groups = len(lr_info.param_group_lrs)
        elif opt.error:
            row.optimizer_error = opt.error

        cfg = analysis.components.config
        row.config_available = cfg.available
        if cfg.available and cfg.architecture_info:
            arch = cfg.architecture_info
            row.model_type = arch.model_type
            row.hidden_size = arch.hidden_size
            row.num_layers = arch.num_layers
            row.vocab_size = arch.vocab_size
            if arch.estimated_parameters:
                row.estimated_params_millions = (
                    arch.estimated_parameters.estimated_total_millions
                )
        elif cfg.error:
            row.config_error = cfg.error

        wgt = analysis.components.weights
        row.weights_available = wgt.available
        if wgt.available and wgt.summary:
            row.total_weight_params_millions = (
                wgt.summary.total_parameters_millions
            )
        elif wgt.error:
            row.weights_error = wgt.error

        return row


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
