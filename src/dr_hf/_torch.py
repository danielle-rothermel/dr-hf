from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_torch: ModuleType | None = None


def get_torch() -> ModuleType:
    global _torch
    if _torch is None:
        try:
            import torch

            _torch = torch
        except ImportError as e:
            raise ImportError(
                "torch is required for this operation. "
                "Install with: uv add dr-hf[weights]"
            ) from e
    return _torch
