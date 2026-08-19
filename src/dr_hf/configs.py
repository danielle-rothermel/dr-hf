from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

from .models import ArchitectureInfo, ConfigAnalysis, ParameterEstimate


def download_config_file(
    repo_id: str, branch: str = "main", local_dir: str | None = None
) -> tuple[str | None, bool, str]:
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=branch,
            local_dir=local_dir,
        )
    except (EntryNotFoundError, HfHubHTTPError, OSError) as e:
        error_msg = str(e)
        if "404" in error_msg or "Entry Not Found" in error_msg:
            return None, False, "Config file not available"
        return None, False, error_msg
    else:
        return file_path, True, ""


def analyze_model_config(config_path: str) -> ConfigAnalysis:
    try:
        with Path(config_path).open(encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        return ConfigAnalysis(available=False, error=str(e))
    else:
        return ConfigAnalysis(
            available=True,
            raw_config=config,
            architecture_info=extract_model_architecture_info(config),
            config_keys=list(config.keys())
            if isinstance(config, dict)
            else [],
            config_type=type(config).__name__,
        )


def extract_model_architecture_info(
    config: dict[str, Any],
) -> ArchitectureInfo:
    arch_fields = {
        "hidden_size": ["hidden_size", "d_model", "n_embd"],
        "num_layers": ["num_hidden_layers", "n_layer", "num_layers"],
        "num_attention_heads": ["num_attention_heads", "n_head", "num_heads"],
        "intermediate_size": ["intermediate_size", "ffn_dim", "n_inner"],
        "vocab_size": ["vocab_size", "vocabulary_size"],
        "max_position_embeddings": [
            "max_position_embeddings",
            "n_positions",
            "max_seq_len",
        ],
        "sequence_length": ["max_sequence_length", "seq_length"],
        "model_type": ["model_type", "architectures"],
        "activation_function": ["hidden_act", "activation_function"],
        "layer_norm_eps": ["layer_norm_eps", "layer_norm_epsilon"],
        "dropout": [
            "hidden_dropout_prob",
            "dropout",
            "attention_probs_dropout_prob",
        ],
        "pad_token_id": ["pad_token_id"],
        "eos_token_id": ["eos_token_id"],
        "bos_token_id": ["bos_token_id"],
    }

    extracted: dict[str, Any] = {}
    for standard_key, possible_keys in arch_fields.items():
        for key in possible_keys:
            if key in config:
                extracted[standard_key] = config[key]
                break

    estimated_parameters = None
    if all(
        k in extracted for k in ["hidden_size", "num_layers", "vocab_size"]
    ):
        estimated_parameters = estimate_parameter_count(
            hidden_size=extracted["hidden_size"],
            num_layers=extracted["num_layers"],
            vocab_size=extracted["vocab_size"],
            intermediate_size=extracted.get("intermediate_size"),
        )

    interesting_keys = [
        "torch_dtype",
        "use_cache",
        "tie_word_embeddings",
        "rope_scaling",
    ]
    for key in interesting_keys:
        if key in config:
            extracted[key] = config[key]

    return ArchitectureInfo(
        hidden_size=extracted.get("hidden_size"),
        num_layers=extracted.get("num_layers"),
        num_attention_heads=extracted.get("num_attention_heads"),
        intermediate_size=extracted.get("intermediate_size"),
        vocab_size=extracted.get("vocab_size"),
        max_position_embeddings=extracted.get("max_position_embeddings"),
        sequence_length=extracted.get("sequence_length"),
        model_type=extracted.get("model_type"),
        activation_function=extracted.get("activation_function"),
        layer_norm_eps=extracted.get("layer_norm_eps"),
        dropout=extracted.get("dropout"),
        pad_token_id=extracted.get("pad_token_id"),
        eos_token_id=extracted.get("eos_token_id"),
        bos_token_id=extracted.get("bos_token_id"),
        torch_dtype=extracted.get("torch_dtype"),
        use_cache=extracted.get("use_cache"),
        tie_word_embeddings=extracted.get("tie_word_embeddings"),
        rope_scaling=extracted.get("rope_scaling"),
        estimated_parameters=estimated_parameters,
    )


def estimate_parameter_count(
    hidden_size: int,
    num_layers: int,
    vocab_size: int,
    intermediate_size: int | None = None,
) -> ParameterEstimate | None:
    if not all([hidden_size, num_layers, vocab_size]):
        return None

    if intermediate_size is None:
        intermediate_size = hidden_size * 4

    embedding_params = vocab_size * hidden_size
    attention_params = 4 * hidden_size * hidden_size
    ff_params = 2 * hidden_size * intermediate_size
    layer_norm_params = 2 * hidden_size

    per_layer_params = attention_params + ff_params + layer_norm_params
    total_layer_params = per_layer_params * num_layers
    output_head_params = vocab_size * hidden_size

    total_params = embedding_params + total_layer_params + output_head_params

    return ParameterEstimate(
        embedding_parameters=embedding_params,
        per_layer_parameters=per_layer_params,
        total_layer_parameters=total_layer_params,
        output_head_parameters=output_head_params,
        estimated_total_parameters=total_params,
    )
