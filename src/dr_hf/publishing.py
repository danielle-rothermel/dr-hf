"""Publishing a local model directory to the Hub with card and provenance."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final

from huggingface_hub import CommitOperationAdd, HfApi
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .pins import CommitSha, ModelPin

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence

__all__ = [
    "MODEL_CARD_FILENAME",
    "PROVENANCE_FILENAME",
    "ModelProvenance",
    "TagConflictError",
    "publish_model",
]

MODEL_CARD_FILENAME: Final = "README.md"
PROVENANCE_FILENAME: Final = "provenance.json"
_MODEL_REPO_TYPE: Final = "model"
_MAIN_BRANCH: Final = "main"
_IGNORED_TOP_LEVEL_DIRS: Final = frozenset({".git", ".cache"})


class TagConflictError(RuntimeError):
    """A requested tag already points at a different commit."""


class ModelProvenance(BaseModel):
    """Where a published model came from and how it was verified.

    ``verification`` must be strict JSON (no NaN or infinity, only JSON
    types); it is written verbatim into the ``provenance.json`` sidecar.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_repo_id: str = Field(min_length=1)
    source_revision: CommitSha
    conversion_tool: str = Field(min_length=1)
    verification: Mapping[str, object]
    notes: str | None = None

    @field_validator("verification")
    @classmethod
    def _strict_json(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as error:
            msg = f"verification must be strict JSON: {error}"
            raise ValueError(msg) from error
        return value


def publish_model(  # noqa: PLR0913
    local_dir: str | os.PathLike[str],
    *,
    repo_id: str,
    branch: str,
    tags: Sequence[str] = (),
    card_markdown: str,
    provenance: ModelProvenance,
    private: bool,
    commit_message: str,
) -> ModelPin:
    """Upload ``local_dir`` to ``branch`` of a Hub model repo in one commit.

    Steps: ``create_repo(exist_ok=True)`` (``private`` applies only when the
    repo is created), ``create_branch(exist_ok=True)`` unless ``branch`` is
    ``main``, then one ``create_commit`` holding every file under
    ``local_dir`` (skipping top-level ``.git`` and ``.cache``), the card as
    ``README.md``, and ``provenance.json``. ``local_dir`` must not contain
    either of those two files itself. Each tag is then created at the
    resulting commit; an existing tag at that commit is accepted and an
    existing tag elsewhere raises ``TagConflictError`` (the commit has
    already landed by then). Unchanged uploads create no commit and return
    the branch head. Authentication uses the standard ``huggingface_hub``
    token resolution.
    """
    root = Path(local_dir)
    files = _local_files(root)
    if not commit_message.strip():
        raise ValueError("commit_message must be non-empty")
    if not branch.strip():
        raise ValueError("branch must be non-empty")
    provenance_json = json.dumps(
        provenance.model_dump(mode="json"),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    operations = [
        *(
            CommitOperationAdd(
                path_in_repo=path.relative_to(root).as_posix(),
                path_or_fileobj=str(path),
            )
            for path in files
        ),
        CommitOperationAdd(
            path_in_repo=MODEL_CARD_FILENAME,
            path_or_fileobj=card_markdown.encode("utf-8"),
        ),
        CommitOperationAdd(
            path_in_repo=PROVENANCE_FILENAME,
            path_or_fileobj=(provenance_json + "\n").encode("utf-8"),
        ),
    ]

    api = HfApi()
    api.create_repo(
        repo_id, repo_type=_MODEL_REPO_TYPE, private=private, exist_ok=True
    )
    if branch != _MAIN_BRANCH:
        api.create_branch(
            repo_id, branch=branch, repo_type=_MODEL_REPO_TYPE, exist_ok=True
        )
    commit = api.create_commit(
        repo_id,
        operations,
        commit_message=commit_message,
        repo_type=_MODEL_REPO_TYPE,
        revision=branch,
    )
    pin = ModelPin(repo_id=repo_id, revision=commit.oid)
    if tags:
        _create_tags(api, pin, tags)
    return pin


def _local_files(root: Path) -> list[Path]:
    if not root.is_dir():
        msg = f"local_dir is not a directory: {root}"
        raise NotADirectoryError(msg)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).parts[0] not in _IGNORED_TOP_LEVEL_DIRS
    )
    reserved = {MODEL_CARD_FILENAME, PROVENANCE_FILENAME}
    clashes = [
        path.name
        for path in files
        if path.relative_to(root).as_posix() in reserved
    ]
    if clashes:
        msg = (
            f"local_dir must not contain {sorted(clashes)!r}; pass the card "
            "as card_markdown and the provenance as provenance"
        )
        raise ValueError(msg)
    if not files:
        msg = f"local_dir has no files to publish: {root}"
        raise ValueError(msg)
    return files


def _create_tags(api: HfApi, pin: ModelPin, tags: Sequence[str]) -> None:
    refs = api.list_repo_refs(pin.repo_id, repo_type=_MODEL_REPO_TYPE)
    existing = {ref.name: ref.target_commit for ref in refs.tags}
    for tag in tags:
        target = existing.get(tag)
        if target == pin.revision:
            continue
        if target is not None:
            msg = (
                f"tag {tag!r} of {pin.repo_id!r} points at {target}, not "
                f"{pin.revision}"
            )
            raise TagConflictError(msg)
        api.create_tag(
            pin.repo_id,
            tag=tag,
            revision=pin.revision,
            repo_type=_MODEL_REPO_TYPE,
        )
