from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from dr_hf.weights import calculate_weight_statistics

if TYPE_CHECKING:
    from pathlib import Path


def test_calculate_weight_statistics_reads_safetensors_keys(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "model.safetensors"
    weight_path.write_bytes(b"fake")

    mock_file = MagicMock()
    mock_file.keys.return_value = ["layer.weight"]
    mock_file.get_tensor.return_value = object()

    mock_safe_open = MagicMock()
    mock_safe_open.return_value.__enter__.return_value = mock_file
    mock_module = MagicMock()
    mock_module.safe_open = mock_safe_open

    with (
        patch("dr_hf.weights._check_safetensors", return_value=True),
        patch(
            "dr_hf.weights.importlib.import_module",
            return_value=mock_module,
        ),
        patch("dr_hf.weights.get_torch") as mock_get_torch,
    ):
        mock_torch = MagicMock()
        mock_torch.is_tensor.return_value = False
        mock_get_torch.return_value = mock_torch

        stats = calculate_weight_statistics(str(weight_path))

    mock_file.keys.assert_called_once_with()
    assert stats.error is None
