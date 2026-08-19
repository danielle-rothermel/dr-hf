from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import CommitInfo

from dr_hf import commit_dataset_files_to_hf
from dr_hf.location import HFLocation
from dr_hf.models import DatasetFileCommitEntry

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def hf_loc() -> HFLocation:
    return HFLocation(org="test-org", repo_name="test-dataset")


@pytest.fixture
def local_file(tmp_path: Path) -> Path:
    path = tmp_path / "data.parquet"
    path.write_bytes(b"parquet")
    return path


def _make_entry(local_path: Path, repo_path: str) -> DatasetFileCommitEntry:
    return DatasetFileCommitEntry(local_path=local_path, repo_path=repo_path)


class _CommitKwargs(TypedDict):
    revision: str
    expected_parent: str
    commit_message: str


def _commit_kwargs(
    *,
    revision: str = "main",
    expected_parent: str = "abc1234",
    commit_message: str = "Add dataset files",
) -> _CommitKwargs:
    return {
        "revision": revision,
        "expected_parent": expected_parent,
        "commit_message": commit_message,
    }


def test_commit_rejects_empty_files(hf_loc: HFLocation) -> None:
    with pytest.raises(ValueError, match="files must be non-empty"):
        commit_dataset_files_to_hf([], hf_loc, **_commit_kwargs())


@patch("dr_hf.io.HfApi.create_commit")
def test_commit_rejects_missing_local_file(
    mock_create_commit: MagicMock,
    hf_loc: HFLocation,
    tmp_path: Path,
) -> None:
    entry = DatasetFileCommitEntry(
        local_path=tmp_path / "missing.parquet",
        repo_path="data/missing.parquet",
    )
    with pytest.raises(FileNotFoundError, match="Local file not found"):
        commit_dataset_files_to_hf([entry], hf_loc, **_commit_kwargs())
    mock_create_commit.assert_not_called()


@patch("dr_hf.io.HfApi.create_commit")
def test_commit_rejects_duplicate_repo_paths(
    mock_create_commit: MagicMock,
    hf_loc: HFLocation,
    tmp_path: Path,
) -> None:
    file_a = tmp_path / "a.parquet"
    file_b = tmp_path / "b.parquet"
    file_a.write_bytes(b"a")
    file_b.write_bytes(b"b")
    entries = [
        DatasetFileCommitEntry(
            local_path=file_a, repo_path="data/same.parquet"
        ),
        DatasetFileCommitEntry(
            local_path=file_b, repo_path="data/same.parquet"
        ),
    ]
    with pytest.raises(ValueError, match="Duplicate repo_path"):
        commit_dataset_files_to_hf(entries, hf_loc, **_commit_kwargs())
    mock_create_commit.assert_not_called()


@patch("dr_hf.io.HfApi.create_commit")
def test_commit_rejects_empty_expected_parent(
    mock_create_commit: MagicMock,
    hf_loc: HFLocation,
    local_file: Path,
) -> None:
    entry = _make_entry(local_file, "data/file.parquet")
    with pytest.raises(ValueError, match="expected_parent must be non-empty"):
        commit_dataset_files_to_hf(
            [entry],
            hf_loc,
            revision="main",
            expected_parent="",
            commit_message="Add file",
        )
    mock_create_commit.assert_not_called()


@patch("dr_hf.io.HfApi.create_commit")
def test_commit_rejects_invalid_repo_path_with_parent_dir(
    mock_create_commit: MagicMock,
    hf_loc: HFLocation,
    local_file: Path,
) -> None:
    entry = _make_entry(local_file, "../escape.parquet")
    with pytest.raises(ValueError, match="repo_path must not contain"):
        commit_dataset_files_to_hf([entry], hf_loc, **_commit_kwargs())
    mock_create_commit.assert_not_called()


@patch("dr_hf.io.HfApi.create_commit")
def test_commit_rejects_absolute_repo_path(
    mock_create_commit: MagicMock,
    hf_loc: HFLocation,
    local_file: Path,
) -> None:
    entry = _make_entry(local_file, "/absolute/path.parquet")
    with pytest.raises(ValueError, match="repo_path must be relative"):
        commit_dataset_files_to_hf([entry], hf_loc, **_commit_kwargs())
    mock_create_commit.assert_not_called()


@patch("dr_hf.io.HfApi")
def test_commit_calls_create_commit_with_expected_args(
    mock_hf_api_cls: MagicMock,
    hf_loc: HFLocation,
    tmp_path: Path,
) -> None:
    file_a = tmp_path / "a.parquet"
    file_b = tmp_path / "b.parquet"
    file_a.write_bytes(b"a")
    file_b.write_bytes(b"b")
    entries = [
        DatasetFileCommitEntry(local_path=file_a, repo_path="data/a.parquet"),
        DatasetFileCommitEntry(local_path=file_b, repo_path="data/b.parquet"),
    ]
    mock_hf_api_cls.return_value.create_commit.return_value = CommitInfo(
        commit_url="https://huggingface.co/datasets/test-org/test-dataset/commit/oid123",
        commit_message="Add dataset files",
        commit_description="",
        oid="oid123",
    )

    result = commit_dataset_files_to_hf(
        entries,
        hf_loc,
        revision="dev",
        expected_parent="parent123",
        commit_message="Add dataset files",
        create_pr=False,
        hf_token="hf_test",
    )

    mock_hf_api_cls.assert_called_once_with(token="hf_test")
    mock_create_commit = mock_hf_api_cls.return_value.create_commit
    mock_create_commit.assert_called_once()
    _, kwargs = mock_create_commit.call_args
    assert kwargs["repo_id"] == "test-org/test-dataset"
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["revision"] == "dev"
    assert kwargs["parent_commit"] == "parent123"
    assert kwargs["create_pr"] is False
    assert kwargs["commit_message"] == "Add dataset files"
    assert len(kwargs["operations"]) == 2
    assert kwargs["operations"][0].path_in_repo == "data/a.parquet"
    assert kwargs["operations"][1].path_in_repo == "data/b.parquet"

    assert result.commit_oid == "oid123"
    assert result.created is True
    assert str(result.commit_url).endswith("/commit/oid123")
    assert result.pr_url is None
    assert str(result.file_urls["data/a.parquet"]).endswith(
        "/resolve/dev/data/a.parquet"
    )
    assert str(result.file_urls["data/b.parquet"]).endswith(
        "/resolve/dev/data/b.parquet"
    )


@patch("dr_hf.io.HfApi.create_commit")
def test_commit_maps_pr_fields(
    mock_create_commit: MagicMock,
    hf_loc: HFLocation,
    local_file: Path,
) -> None:
    entry = _make_entry(local_file, "data/file.parquet")
    mock_create_commit.return_value = CommitInfo(
        commit_url="https://huggingface.co/datasets/test-org/test-dataset/commit/oid123",
        commit_message="Add dataset files",
        commit_description="details",
        oid="oid123",
        pr_url="https://huggingface.co/datasets/test-org/test-dataset/discussions/7",
    )

    result = commit_dataset_files_to_hf(
        [entry],
        hf_loc,
        **_commit_kwargs(),
        create_pr=True,
    )

    assert mock_create_commit.call_args.kwargs["create_pr"] is True
    assert result.created is True
    assert result.pr_num == 7
    assert result.pr_revision == "refs/pr/7"
    assert str(result.pr_url).endswith("/discussions/7")
    assert str(result.file_urls["data/file.parquet"]).endswith(
        "/resolve/refs/pr/7/data/file.parquet"
    )


@patch("dr_hf.io.HfApi.create_commit")
def test_commit_noop_when_oid_matches_expected_parent(
    mock_create_commit: MagicMock,
    hf_loc: HFLocation,
    local_file: Path,
) -> None:
    entry = _make_entry(local_file, "data/file.parquet")
    mock_create_commit.return_value = CommitInfo(
        commit_url="https://huggingface.co/datasets/test-org/test-dataset/commit/abc1234",
        commit_message="Add dataset files",
        commit_description="",
        oid="abc1234",
    )

    result = commit_dataset_files_to_hf(
        [entry],
        hf_loc,
        revision="main",
        expected_parent="abc1234",
        commit_message="Add dataset files",
    )

    assert result.created is False
    assert result.commit_oid == "abc1234"


@patch("dr_hf.io.HfApi.create_commit")
def test_commit_rejects_stale_parent_noop(
    mock_create_commit: MagicMock,
    hf_loc: HFLocation,
    local_file: Path,
) -> None:
    entry = _make_entry(local_file, "data/file.parquet")
    mock_create_commit.return_value = CommitInfo(
        commit_url="https://huggingface.co/datasets/test-org/test-dataset/commit/current_head",
        commit_message="",
        commit_description="",
        oid="current_head",
    )

    with pytest.raises(ValueError, match="expected_parent is stale"):
        commit_dataset_files_to_hf(
            [entry],
            hf_loc,
            revision="main",
            expected_parent="stale_parent",
            commit_message="Add dataset files",
        )
