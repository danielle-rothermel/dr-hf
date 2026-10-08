from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from huggingface_hub import (
    CommitInfo,
    CommitOperationAdd,
    HfApi,
)
from pydantic import HttpUrl

from .location import HFLocation
from .models import DatasetCommitResult, DatasetFileCommitEntry

__all__ = ["commit_dataset_files_to_hf"]

_COMMIT_OID_RE = re.compile(r"^[0-9a-fA-F]{5,40}$")


def _normalize_commit_oid(
    oid: str,
    *,
    field_name: str = "expected_parent",
) -> str:
    stripped = oid.strip()
    if not stripped:
        msg = f"{field_name} must be non-empty"
        raise ValueError(msg)
    if not _COMMIT_OID_RE.fullmatch(stripped):
        msg = (
            f"{field_name} must be a valid commit OID "
            f"(5-40 hexadecimal characters), got {oid!r}"
        )
        raise ValueError(msg)
    return stripped.lower()


def _commit_oids_equal(left: str, right: str) -> bool:
    normalized_left = _normalize_commit_oid(left)
    normalized_right = _normalize_commit_oid(right)
    if len(normalized_left) <= len(normalized_right):
        shorter, longer = normalized_left, normalized_right
    else:
        shorter, longer = normalized_right, normalized_left
    return longer.startswith(shorter)


def _validate_repo_path(repo_path: str) -> str:
    if PurePosixPath(repo_path).is_absolute():
        raise ValueError(f"repo_path must be relative, got: {repo_path!r}")
    normalized = HFLocation.norm_posix(repo_path)
    if not normalized or normalized == ".":
        raise ValueError(f"Invalid repo_path: {repo_path!r}")
    if ".." in PurePosixPath(normalized).parts:
        raise ValueError(f"repo_path must not contain '..': {repo_path!r}")
    return normalized


def _validate_dataset_commit_inputs(
    files: list[DatasetFileCommitEntry],
    *,
    revision: str,
    expected_parent: str,
    commit_message: str,
) -> tuple[list[DatasetFileCommitEntry], str]:
    if not files:
        raise ValueError("files must be non-empty")
    if not revision.strip():
        raise ValueError("revision must be non-empty")
    if not commit_message.strip():
        raise ValueError("commit_message must be non-empty")
    canonical_parent = _normalize_commit_oid(expected_parent)

    normalized: list[DatasetFileCommitEntry] = []
    seen_repo_paths: set[str] = set()
    for entry in files:
        local_path = Path(entry.local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        if not local_path.is_file():
            raise ValueError(f"Local path is not a file: {local_path}")
        repo_path = _validate_repo_path(entry.repo_path)
        if repo_path in seen_repo_paths:
            raise ValueError(f"Duplicate repo_path: {repo_path!r}")
        seen_repo_paths.add(repo_path)
        normalized.append(
            DatasetFileCommitEntry(local_path=local_path, repo_path=repo_path)
        )
    return normalized, canonical_parent


def _revision_head_sha(
    api: HfApi,
    hf_loc: HFLocation,
    *,
    revision: str,
) -> str:
    repo_head = api.repo_info(
        repo_id=hf_loc.repo_id,
        repo_type=hf_loc.hf_hub_repo_type,
        revision=revision,
    ).sha
    if not repo_head:
        msg = f"Could not resolve revision head for {revision!r}"
        raise ValueError(msg)
    return _normalize_commit_oid(repo_head, field_name="revision head")


def _commit_was_created(
    commit_info: CommitInfo,
    *,
    expected_parent: str,
    head_before: str,
    create_pr: bool,
    committed: bool,
) -> bool:
    if create_pr:
        return bool(commit_info.pr_url or commit_info.pr_revision)

    if committed:
        return True

    oid = _normalize_commit_oid(commit_info.oid)
    if _commit_oids_equal(oid, expected_parent):
        return False

    if _commit_oids_equal(oid, head_before):
        if not _commit_oids_equal(expected_parent, head_before):
            msg = (
                "expected_parent is stale: Hub returned revision head "
                f"{commit_info.oid!r} without creating a commit"
            )
            raise ValueError(msg)
        return False

    return False


def commit_dataset_files_to_hf(  # noqa: PLR0913
    files: list[DatasetFileCommitEntry],
    hf_loc: HFLocation,
    *,
    revision: str,
    expected_parent: str,
    commit_message: str,
    commit_description: str | None = None,
    create_pr: bool = False,
    hf_token: str | None = None,
) -> DatasetCommitResult:
    normalized_files, canonical_parent = _validate_dataset_commit_inputs(
        files,
        revision=revision,
        expected_parent=expected_parent,
        commit_message=commit_message,
    )
    operations = [
        CommitOperationAdd(
            path_in_repo=entry.repo_path,
            path_or_fileobj=str(entry.local_path),
        )
        for entry in normalized_files
    ]
    api = HfApi(token=hf_token)
    head_before = _revision_head_sha(api, hf_loc, revision=revision)
    commit_info = api.create_commit(
        repo_id=hf_loc.repo_id,
        operations=operations,
        commit_message=commit_message,
        commit_description=commit_description or "",
        repo_type=hf_loc.hf_hub_repo_type,
        revision=revision,
        parent_commit=canonical_parent,
        create_pr=create_pr,
    )
    created = _commit_was_created(
        commit_info,
        expected_parent=canonical_parent,
        head_before=head_before,
        create_pr=create_pr,
        committed=any(op._is_committed for op in operations),
    )
    url_revision = commit_info.pr_revision or revision
    file_urls = {
        entry.repo_path: hf_loc.get_file_download_link_for_revision(
            entry.repo_path, url_revision
        )
        for entry in normalized_files
    }
    return DatasetCommitResult(
        created=created,
        commit_oid=commit_info.oid,
        commit_url=HttpUrl(commit_info.commit_url),
        commit_message=commit_info.commit_message,
        commit_description=commit_info.commit_description,
        pr_url=HttpUrl(commit_info.pr_url) if commit_info.pr_url else None,
        pr_num=commit_info.pr_num,
        pr_revision=commit_info.pr_revision,
        file_urls=file_urls,
    )
