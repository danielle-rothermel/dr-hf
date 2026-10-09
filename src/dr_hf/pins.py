"""Dataset and model pins: Hub repositories resolved to commit SHAs."""

from __future__ import annotations

import re
from typing import Annotated, Final

from huggingface_hub import HfApi
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from .identity import DATASET_PIN_SCHEMA, identity_hash

__all__ = [
    "ROW_INDEX",
    "CommitSha",
    "DatasetPin",
    "ModelPin",
    "resolve_dataset_pin",
    "resolve_model_pin",
]

ROW_INDEX: Final = "__row__"
"""Native id sentinel: use ``f"{origin_split}:{row_position}"`` as the id."""

_COMMIT_SHA_RE = re.compile(r"[0-9a-f]{40}")


def _validate_commit_sha(value: str) -> str:
    if not _COMMIT_SHA_RE.fullmatch(value):
        msg = (
            "revision must be a full 40-character lowercase hexadecimal "
            f"commit SHA, got {value!r}"
        )
        raise ValueError(msg)
    return value


CommitSha = Annotated[str, AfterValidator(_validate_commit_sha)]
"""A full 40-character lowercase hexadecimal Hub commit SHA."""

RepoId = Annotated[str, Field(min_length=1, pattern=r"^\S+$")]


class DatasetPin(BaseModel):
    """One Hub dataset configuration fixed at one commit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo_id: RepoId
    config: str | None
    revision: CommitSha
    native_id_field: str = Field(min_length=1)

    @property
    def pin_hash(self) -> str:
        """dr-serialize identity hash of the pin; subsets are scoped by it."""
        return identity_hash(DATASET_PIN_SCHEMA, self.model_dump(mode="json"))


class ModelPin(BaseModel):
    """One Hub model repository fixed at one commit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo_id: RepoId
    revision: CommitSha


def resolve_dataset_pin(
    repo_id: str,
    *,
    config: str | None = None,
    ref: str = "main",
    native_id_field: str,
) -> DatasetPin:
    """Resolve a dataset branch, tag, or SHA to a ``DatasetPin``.

    The commit SHA comes from ``HfApi.dataset_info(repo_id, revision=ref)``.
    The config and native id field are recorded as given; they are checked
    when rows are read.
    """
    info = HfApi().dataset_info(repo_id, revision=ref)
    return DatasetPin(
        repo_id=repo_id,
        config=config,
        revision=_require_sha(info.sha, repo_id=repo_id, ref=ref),
        native_id_field=native_id_field,
    )


def resolve_model_pin(repo_id: str, *, ref: str = "main") -> ModelPin:
    """Resolve a model branch, tag, or SHA to a ``ModelPin``.

    The commit SHA comes from ``HfApi.model_info(repo_id, revision=ref)``.
    """
    info = HfApi().model_info(repo_id, revision=ref)
    return ModelPin(
        repo_id=repo_id,
        revision=_require_sha(info.sha, repo_id=repo_id, ref=ref),
    )


def _require_sha(sha: str | None, *, repo_id: str, ref: str) -> str:
    if not sha:
        msg = f"Hub returned no commit SHA for {repo_id!r} at {ref!r}"
        raise ValueError(msg)
    return sha
