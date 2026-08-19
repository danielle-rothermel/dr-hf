from __future__ import annotations

import importlib
import logging
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, list_repo_files

from ._torch import get_torch
from .models import (
    GlobalWeightStats,
    LayerAnalysis,
    LayerCategorization,
    LayerCounts,
    ParameterStats,
    TensorInfo,
    TensorStats,
    WeightFileStatistics,
    WeightsAnalysis,
    WeightsSummary,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _check_safetensors() -> bool:
    try:
        importlib.import_module("safetensors")
    except ImportError:
        return False
    else:
        return True


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


def _require_pytorch_for_bin_weights() -> None:
    """Raise if PyTorch is too old for safe .bin weight loading."""
    if _check_pytorch_version_for_weights_only():
        return
    torch = get_torch()
    raise RuntimeError(
        "PyTorch >= 2.6.0 is required for safe loading of .bin weight "
        f"files. Current version: {torch.__version__}. "
        "Please upgrade PyTorch or use .safetensors files instead."
    )


def discover_model_weight_files(
    repo_id: str, branch: str = "main"
) -> list[str]:
    try:
        all_files = list_repo_files(repo_id=repo_id, revision=branch)
    except Exception:
        return []
    else:
        weight_patterns = [
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model-00001-of-00001.bin",
        ]
        weight_files = [
            file
            for file in all_files
            if any(pattern in file for pattern in weight_patterns)
            or ("pytorch_model-" in file and file.endswith(".bin"))
            or file.endswith(".safetensors")
        ]
        return sorted(weight_files)


def download_model_weights(
    repo_id: str, branch: str, filename: str, local_dir: str | None = None
) -> tuple[str | None, bool, str]:
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=branch,
            local_dir=local_dir,
        )
        return file_path, True, ""
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "Entry Not Found" in error_msg:
            return None, False, f"Weight file {filename} not available"
        return None, False, error_msg


def calculate_weight_statistics(weight_path: str) -> WeightFileStatistics:
    torch = get_torch()

    try:
        if weight_path.endswith(".safetensors"):
            if not _check_safetensors():
                return WeightFileStatistics(
                    file_path=weight_path,
                    file_size_mb=0.0,
                    num_tensors=0,
                    error="safetensors library not available",
                )
            safe_open = importlib.import_module("safetensors").safe_open

            weights: dict[str, Any] = {}
            with safe_open(weight_path, framework="pt") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
        else:
            # Use weights_only=True for security (requires PyTorch >= 2.6.0)
            # Safetensors remains the preferred path (checked first above)
            _require_pytorch_for_bin_weights()
            weights = torch.load(
                weight_path, map_location="cpu", weights_only=True
            )

        if not isinstance(weights, dict):
            return WeightFileStatistics(
                file_path=weight_path,
                file_size_mb=0.0,
                num_tensors=0,
                error="Weights file is not a dictionary",
            )

        total_params = 0
        tensor_infos: list[TensorInfo] = []

        for name, tensor in weights.items():
            if torch.is_tensor(tensor):
                tensor_params = tensor.numel()
                total_params += tensor_params

                tensor_infos.append(
                    TensorInfo(
                        name=name,
                        shape=list(tensor.shape),
                        dtype=str(tensor.dtype),
                        parameters=tensor_params,
                        size_mb=round(
                            tensor.numel()
                            * tensor.element_size()
                            / (1024 * 1024),
                            3,
                        ),
                        statistics=calculate_tensor_stats(tensor),
                    )
                )

        return WeightFileStatistics(
            file_path=weight_path,
            file_size_mb=round(
                Path(weight_path).stat().st_size / (1024 * 1024), 2
            ),
            num_tensors=len(weights),
            tensor_info=tensor_infos,
            parameter_stats=ParameterStats(total_parameters=total_params),
            layer_analysis=analyze_layer_structure(weights),
            global_statistics=calculate_global_weight_stats(weights),
        )

    except Exception as e:
        return WeightFileStatistics(
            file_path=weight_path,
            file_size_mb=0.0,
            num_tensors=0,
            error=f"Failed to analyze weights: {e!s}",
        )


def calculate_tensor_stats(tensor: Any) -> TensorStats | None:
    torch = get_torch()

    try:
        flat_tensor = tensor.flatten().float()

        return TensorStats(
            mean=float(torch.mean(flat_tensor)),
            std=float(torch.std(flat_tensor)),
            min=float(torch.min(flat_tensor)),
            max=float(torch.max(flat_tensor)),
            median=float(torch.median(flat_tensor)),
            abs_mean=float(torch.mean(torch.abs(flat_tensor))),
            zero_fraction=float(
                torch.sum(flat_tensor == 0.0) / flat_tensor.numel()
            ),
            percentile_25=float(torch.quantile(flat_tensor, 0.25)),
            percentile_75=float(torch.quantile(flat_tensor, 0.75)),
            percentile_90=float(torch.quantile(flat_tensor, 0.90)),
            percentile_95=float(torch.quantile(flat_tensor, 0.95)),
            percentile_99=float(torch.quantile(flat_tensor, 0.99)),
        )

    except (RuntimeError, ValueError, TypeError):
        return None


def analyze_layer_structure(weights: dict[str, Any]) -> LayerAnalysis:
    torch = get_torch()

    categorization = LayerCategorization()
    counts = LayerCounts()

    for name, tensor in weights.items():
        if not torch.is_tensor(tensor):
            continue

        name_lower = name.lower()

        if "embed" in name_lower:
            categorization.embedding_layers.append(name)
        elif (
            "ln" in name_lower
            or "layer_norm" in name_lower
            or "layernorm" in name_lower
        ):
            categorization.layer_norm_layers.append(name)
            counts.layer_norms += 1
        elif any(
            attn_key in name_lower
            for attn_key in ["attn", "attention", "self_attention"]
        ):
            categorization.attention_layers.append(name)
        elif any(
            ff_key in name_lower
            for ff_key in ["mlp", "ffn", "feed_forward", "intermediate"]
        ):
            categorization.feedforward_layers.append(name)
            counts.feedforward_layers += 1
        elif "lm_head" in name_lower or "output" in name_lower:
            categorization.output_layers.append(name)
        elif "transformer" in name_lower or "layer" in name_lower:
            categorization.transformer_layers.append(name)
        else:
            categorization.other_layers.append(name)

    transformer_layers: set[int] = set()
    for name in weights:
        layer_matches = re.findall(r"(?:layer|h)\.(\d+)\.", name.lower())
        if layer_matches:
            transformer_layers.update(int(match) for match in layer_matches)

    counts.estimated_transformer_layers = len(transformer_layers)
    counts.total_layers = len(weights)

    return LayerAnalysis(
        layer_categorization=categorization,
        layer_counts=counts,
        transformer_layer_indices=sorted(transformer_layers)
        if transformer_layers
        else [],
    )


def calculate_global_weight_stats(  # noqa: PLR0912, PLR0915
    weights: dict[str, Any],
) -> GlobalWeightStats | None:
    torch = get_torch()

    try:
        # Welford's algorithm state for mean/variance
        count = 0
        mean = 0.0
        m2 = 0.0  # Sum of squared differences from mean

        # Running min/max
        global_min = float("inf")
        global_max = float("-inf")

        # For abs_mean: sum of absolute values and total count
        abs_sum = 0.0

        # For zero_fraction: count of zeros and total count
        zero_count = 0

        # Reservoir sampling for percentiles (fixed size: 10,000 samples)
        reservoir_size = 10000
        reservoir_samples = []
        reservoir_count = 0

        # Iterate over all tensors without concatenating
        for tensor in weights.values():
            if not torch.is_tensor(tensor):
                continue

            flat_tensor = tensor.flatten().float()
            tensor_size = flat_tensor.numel()

            if tensor_size == 0:
                continue

            # Update min/max
            tensor_min = float(torch.min(flat_tensor))
            tensor_max = float(torch.max(flat_tensor))
            global_min = min(global_min, tensor_min)
            global_max = max(global_max, tensor_max)

            # Update abs_sum
            abs_sum += float(torch.sum(torch.abs(flat_tensor)))

            # Update zero count
            zero_count += int(torch.sum(flat_tensor == 0.0))

            # Welford's algorithm for incremental mean/variance
            # Batch update: new_mean = (old_count * old_mean + n * m) / n_total
            chunk_mean = float(torch.mean(flat_tensor))
            chunk_count = tensor_size

            # Update mean using batch formula
            new_count = count + chunk_count
            if new_count > 0:
                # Batch update for mean
                new_mean = (
                    count * mean + chunk_count * chunk_mean
                ) / new_count

                # Combined variance formula for two groups
                chunk_variance = float(torch.var(flat_tensor, unbiased=False))
                if chunk_count > 0:
                    # m2_total combines old m2, chunk variance, and mean delta
                    delta_mean = chunk_mean - mean
                    m2 += chunk_count * (
                        chunk_variance
                        + delta_mean * delta_mean * count / new_count
                    )
                    mean = new_mean
                    count = new_count

            # Reservoir sampling for percentiles (chunked for large tensors)
            chunk_size = 100000

            for i in range(0, tensor_size, chunk_size):
                end_idx = min(i + chunk_size, tensor_size)
                chunk = flat_tensor[i:end_idx].cpu().numpy().flatten()

                for value in chunk:
                    reservoir_count += 1
                    if len(reservoir_samples) < reservoir_size:
                        # Fill reservoir initially
                        reservoir_samples.append(float(value))
                    else:
                        # Replace with prob. reservoir_size / reservoir_count
                        replace_idx = random.randint(  # noqa: S311
                            0, reservoir_count - 1
                        )
                        if replace_idx < reservoir_size:
                            reservoir_samples[replace_idx] = float(value)

        if count == 0:
            return None

        # Compute final statistics
        global_mean = mean
        # Population std: sqrt(m2 / count)
        global_std = (
            float(torch.sqrt(torch.tensor(m2 / count))) if m2 > 0 else 0.0
        )

        global_abs_mean = abs_sum / count
        global_zero_fraction = zero_count / count

        # Compute percentiles from reservoir sample
        percentile_values = {}
        if reservoir_samples:
            reservoir_tensor = torch.tensor(
                reservoir_samples, dtype=torch.float32
            )
            for p in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
                percentile_key = f"percentile_{int(p * 100)}"
                percentile_values[f"global_{percentile_key}"] = float(
                    torch.quantile(reservoir_tensor, p)
                )
        else:
            # Fallback if no samples collected
            for p in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
                percentile_key = f"percentile_{int(p * 100)}"
                percentile_values[f"global_{percentile_key}"] = None

        return GlobalWeightStats(
            global_mean=global_mean,
            global_std=global_std,
            global_min=global_min if global_min != float("inf") else 0.0,
            global_max=global_max if global_max != float("-inf") else 0.0,
            global_abs_mean=global_abs_mean,
            global_zero_fraction=global_zero_fraction,
            global_percentile_1=percentile_values.get("global_percentile_1"),
            global_percentile_5=percentile_values.get("global_percentile_5"),
            global_percentile_25=percentile_values.get("global_percentile_25"),
            global_percentile_50=percentile_values.get("global_percentile_50"),
            global_percentile_75=percentile_values.get("global_percentile_75"),
            global_percentile_95=percentile_values.get("global_percentile_95"),
            global_percentile_99=percentile_values.get("global_percentile_99"),
        )
    except (RuntimeError, ValueError, TypeError):
        return None


def analyze_model_weights(
    repo_id: str,
    branch: str,
    weight_files: list[str] | None = None,
    delete_after_analysis: bool = False,
) -> WeightsAnalysis:
    if weight_files is None:
        weight_files = discover_model_weight_files(repo_id, branch)

    if not weight_files:
        return WeightsAnalysis(
            available=False,
            error="No weight files found",
            discovered_files=[],
        )

    file_analyses: dict[str, WeightFileStatistics] = {}
    total_params = 0
    total_size_mb = 0.0
    downloaded_files: list[str] = []
    successful_analyses = 0

    try:
        for weight_file in weight_files:
            file_path, success, error_msg = download_model_weights(
                repo_id, branch, weight_file
            )

            if success and file_path:
                downloaded_files.append(file_path)

                file_stats = calculate_weight_statistics(file_path)
                file_analyses[weight_file] = file_stats

                if file_stats.error is None:
                    successful_analyses += 1
                    if file_stats.parameter_stats:
                        total_params += (
                            file_stats.parameter_stats.total_parameters
                        )
                    total_size_mb += file_stats.file_size_mb
            else:
                file_analyses[weight_file] = WeightFileStatistics(
                    file_path="",
                    file_size_mb=0.0,
                    num_tensors=0,
                    error=error_msg,
                )

        return WeightsAnalysis(
            available=True,
            discovered_files=weight_files,
            file_analyses=file_analyses,
            summary=WeightsSummary(
                total_files_analyzed=successful_analyses,
                total_parameters=total_params,
                total_size_mb=round(total_size_mb, 2),
            ),
        )

    finally:
        if delete_after_analysis and downloaded_files:
            for file_path in downloaded_files:
                try:
                    Path(file_path).unlink()
                except OSError as e:
                    logger.warning(
                        "Failed to remove downloaded file %s: %s",
                        file_path,
                        e,
                    )
