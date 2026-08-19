from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

from ._torch import get_torch
from .branches import extract_step_from_branch, get_checkpoint_branches
from .configs import analyze_model_config, download_config_file
from .models import (
    CheckpointAnalysis,
    CheckpointComponents,
    CheckpointSummaryRow,
    ConfigAnalysis,
    LearningRateInfo,
    OptimizerAnalysis,
    OptimizerComponentInfo,
    ParamGroupInfo,
    WeightsAnalysis,
)
from .weights import analyze_model_weights

logger = logging.getLogger(__name__)


def download_optimizer_checkpoint(
    repo_id: str, branch: str = "main", local_dir: str | None = None
) -> tuple[str | None, bool, str]:
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename="training/optim.pt",
            revision=branch,
            local_dir=local_dir,
        )
    except (EntryNotFoundError, HfHubHTTPError, OSError) as e:
        error_msg = str(e)
        if "404" in error_msg or "Entry Not Found" in error_msg:
            return None, False, "Optimizer checkpoint not available"
        return None, False, error_msg
    else:
        return file_path, True, ""


def _parse_optimizer_component(
    component: Any, component_name: str
) -> OptimizerComponentInfo:
    info = OptimizerComponentInfo(
        type=type(component).__name__,
        size=len(component) if hasattr(component, "__len__") else "N/A",
    )

    if component_name == "param_groups" and isinstance(component, list):
        info.num_param_groups = len(component)
        if component:
            first_group = component[0]
            if isinstance(first_group, dict):
                info.param_group_keys = list(first_group.keys())
                if "lr" in first_group:
                    info.learning_rate = first_group["lr"]
                if "initial_lr" in first_group:
                    info.initial_learning_rate = first_group["initial_lr"]

    elif component_name == "state" and isinstance(component, dict):
        info.num_parameters = len(component)
        if component:
            first_param_state = next(iter(component.values()))
            if isinstance(first_param_state, dict):
                info.state_keys = list(first_param_state.keys())

    return info


def _parse_learning_rate_info(checkpoint: dict[str, Any]) -> LearningRateInfo:
    lr_info = LearningRateInfo()

    if "param_groups" in checkpoint and isinstance(
        checkpoint["param_groups"], list
    ):
        for i, group in enumerate(checkpoint["param_groups"]):
            if isinstance(group, dict):
                group_info = ParamGroupInfo(
                    group_index=i,
                    current_lr=group.get("lr"),
                    initial_lr=group.get("initial_lr"),
                    weight_decay=group.get("weight_decay"),
                    momentum=group.get("momentum"),
                )
                lr_info.param_group_lrs.append(group_info)

    if "lr_scheduler" in checkpoint:
        lr_info.lr_scheduler_present = True
        scheduler = checkpoint["lr_scheduler"]
        if isinstance(scheduler, dict):
            lr_info.lr_scheduler_keys = list(scheduler.keys())
            if "last_epoch" in scheduler:
                lr_info.scheduler_last_epoch = scheduler["last_epoch"]

    return lr_info


def _check_pytorch_version_for_weights_only() -> bool:
    """Check PyTorch >= 2.6.0 for weights_only=True support."""
    torch = get_torch()
    version_str = torch.__version__
    # Handle version strings like "2.6.0" or "2.6.0+cu118"
    version_parts = version_str.split("+")[0].split(".")
    try:
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
    except (ValueError, IndexError):
        # If version parsing fails, assume it's too old to be safe
        return False
    else:
        return (major, minor) >= (2, 6)


def analyze_optimizer_checkpoint(checkpoint_path: str) -> OptimizerAnalysis:
    """
    Analyze an optimizer checkpoint file.

    This function loads and analyzes optimizer checkpoint files. Note that
    optimizer checkpoint files must be trusted as torch.load() can execute
    arbitrary code. For PyTorch >= 2.6, this function uses weights_only=True
    to mitigate code execution risks. For older PyTorch versions, the
    checkpoint is loaded without this protection.

    It is recommended that optimizer checkpoints be saved as state_dicts to
    avoid code execution risks.

    Args:
        checkpoint_path: Path to the optimizer checkpoint file.

    Returns:
        OptimizerAnalysis object containing the analysis results.
    """
    torch = get_torch()

    try:
        # Use weights_only=True for PyTorch >= 2.6 to prevent code execution
        if _check_pytorch_version_for_weights_only():
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        else:
            logger.warning(
                "PyTorch version %s < 2.6.0 does not support "
                "weights_only=True. Loading optimizer checkpoint without this "
                "protection. Optimizer checkpoint files must be trusted as "
                "torch.load() can execute arbitrary code. Consider upgrading "
                "PyTorch or ensuring checkpoints are saved as state_dicts.",
                torch.__version__,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        analysis = OptimizerAnalysis(
            available=True,
            checkpoint_keys=list(checkpoint.keys())
            if isinstance(checkpoint, dict)
            else ["non_dict_checkpoint"],
            checkpoint_type=type(checkpoint).__name__,
        )

        if isinstance(checkpoint, dict):
            for key in ["state", "param_groups", "optimizer", "lr_scheduler"]:
                if key in checkpoint:
                    analysis.optimizer_info[key] = _parse_optimizer_component(
                        checkpoint[key], key
                    )
            analysis.learning_rate_info = _parse_learning_rate_info(checkpoint)

    except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as e:
        return OptimizerAnalysis(available=False, error=str(e))
    else:
        return analysis


def analyze_complete_checkpoint(
    repo_id: str,
    branch: str,
    *,
    include_weights: bool = False,
    weight_files: list[str] | None = None,
    delete_weights_after: bool = False,
) -> CheckpointAnalysis:
    components = CheckpointComponents()

    optimizer_path, optimizer_success, optimizer_error = (
        download_optimizer_checkpoint(repo_id, branch)
    )

    if optimizer_success and optimizer_path:
        components.optimizer = analyze_optimizer_checkpoint(optimizer_path)
    else:
        components.optimizer = OptimizerAnalysis(
            available=False, error=optimizer_error
        )

    config_path, config_success, config_error = download_config_file(
        repo_id, branch
    )

    if config_success and config_path:
        components.config = analyze_model_config(config_path)
    else:
        components.config = ConfigAnalysis(available=False, error=config_error)

    if include_weights:
        components.weights = analyze_model_weights(
            repo_id,
            branch,
            weight_files,
            delete_after_analysis=delete_weights_after,
        )

    return CheckpointAnalysis(
        branch=branch,
        step=extract_step_from_branch(branch),
        components=components,
    )


def process_single_checkpoint(
    repo_id: str,
    branch: str,
    *,
    include_weights: bool = False,
    delete_weights_after: bool = False,
) -> tuple[str, CheckpointAnalysis]:
    try:
        analysis = analyze_complete_checkpoint(
            repo_id,
            branch,
            include_weights=include_weights,
            delete_weights_after=delete_weights_after,
        )
    except (OSError, RuntimeError, ValueError, TypeError) as e:
        error_analysis = CheckpointAnalysis(
            branch=branch,
            step=extract_step_from_branch(branch),
            error=str(e),
            components=CheckpointComponents(
                optimizer=OptimizerAnalysis(available=False, error=str(e)),
                config=ConfigAnalysis(available=False, error=str(e)),
                weights=WeightsAnalysis(available=False, error=str(e)),
            ),
        )
        return branch, error_analysis
    else:
        return branch, analysis


def process_all_checkpoints(
    repo_id: str,
    max_workers: int = 4,
    *,
    include_weights: bool = False,
    delete_weights_after: bool = False,
) -> dict[str, CheckpointAnalysis]:
    branches = get_checkpoint_branches(repo_id)

    if not branches:
        return {}

    results: dict[str, CheckpointAnalysis] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_branch = {
            executor.submit(
                process_single_checkpoint,
                repo_id,
                branch,
                include_weights=include_weights,
                delete_weights_after=delete_weights_after,
            ): branch
            for branch in branches
        }

        for future in as_completed(future_to_branch):
            branch, analysis = future.result()
            results[branch] = analysis

    return results


def create_comprehensive_summary(
    all_analyses: dict[str, CheckpointAnalysis],
) -> pd.DataFrame:
    summary_data = []

    for analysis in all_analyses.values():
        if analysis.error:
            continue
        row = CheckpointSummaryRow.from_analysis(analysis)
        summary_data.append(row.model_dump())

    df = pd.DataFrame(summary_data)
    return df.sort_values("step") if not df.empty else df


def create_learning_rate_summary(
    all_analyses: dict[str, CheckpointAnalysis],
) -> pd.DataFrame:
    comprehensive_df = create_comprehensive_summary(all_analyses)

    lr_columns = [
        "branch",
        "step",
        "optimizer_available",
        "current_lr",
        "weight_decay",
        "momentum",
        "num_param_groups",
        "optimizer_error",
    ]

    return (
        comprehensive_df[lr_columns]
        if not comprehensive_df.empty
        else comprehensive_df
    )


def save_checkpoint_analysis(
    analysis: CheckpointAnalysis, branch: str, output_dir: str | None = None
) -> str:
    sanitized = branch.replace("/", "_").replace("-", "_")
    filename = f"optimizer_analysis_{sanitized}.json"

    if output_dir:
        filepath = Path(output_dir) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
    else:
        filepath = Path(filename)

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(analysis.model_dump(), f, indent=2, default=str)

    return str(filepath)


def save_all_analyses_outputs(
    all_analyses: dict[str, CheckpointAnalysis], output_dir: str | None = None
) -> tuple[str | None, str | None, str]:
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        comprehensive_csv_path = (
            output_path / "comprehensive_checkpoint_summary.csv"
        )
        lr_csv_path = output_path / "learning_rate_summary.csv"
        json_path = output_path / "all_checkpoint_analyses.json"
    else:
        comprehensive_csv_path = Path("comprehensive_checkpoint_summary.csv")
        lr_csv_path = Path("learning_rate_summary.csv")
        json_path = Path("all_checkpoint_analyses.json")

    comprehensive_summary = create_comprehensive_summary(all_analyses)
    if not comprehensive_summary.empty:
        comprehensive_summary.to_csv(comprehensive_csv_path, index=False)
        comprehensive_csv_path_str: str | None = str(comprehensive_csv_path)
    else:
        comprehensive_csv_path_str = None

    lr_summary = create_learning_rate_summary(all_analyses)
    if not lr_summary.empty:
        lr_summary.to_csv(lr_csv_path, index=False)
        lr_csv_path_str: str | None = str(lr_csv_path)
    else:
        lr_csv_path_str = None

    serializable = {k: v.model_dump() for k, v in all_analyses.items()}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str)

    return comprehensive_csv_path_str, lr_csv_path_str, str(json_path)
