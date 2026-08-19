from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from datasets import Dataset, load_dataset

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def load_or_download_dataset(
    path: Path, repo_id: str, split: str = "train"
) -> pd.DataFrame:
    download_dataset(path, repo_id, split, force_reload=False)
    return pd.read_parquet(path)


def download_dataset(
    path: Path,
    repo_id: str,
    split: str = "train",
    *,
    force_reload: bool = False,
) -> None:
    if force_reload or not path.exists():
        try:
            raw_ds: Dataset = load_dataset(repo_id, split=split)
            path.parent.mkdir(parents=True, exist_ok=True)
            raw_ds.to_parquet(path)
        except Exception as e:
            logger.exception(
                "Failed to download dataset: repo_id=%s, split=%s, path=%s",
                repo_id,
                split,
                path,
            )
            raise RuntimeError(
                f"Failed to download dataset '{repo_id}' "
                f"(split='{split}') to {path}"
            ) from e


def sanitize_repo_name(repo_id: str) -> str:
    return repo_id.replace("/", "--").replace(" ", "-")
