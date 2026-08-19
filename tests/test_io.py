from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import CommitInfo, CommitOperationAdd

from dr_hf import commit_dataset_files_to_hf
from dr_hf.location import HFLocation
from dr_hf.models import DatasetFileCommitEntry

if TYPE_CHECKING:
    from pathlib import Path

PARENT_OID = "91c54ad1727ee830252e457677f467be0bfd8a57"
PARENT_OID_ABBREV = "91c54ad1"
HEAD_OID_B = "b2c3d4e5f6789012345678901234567890abcd01"
NEW_COMMIT_OID = "c3d4e5f6789012345678901234567890ab12cd"
STALE_PARENT_OID = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


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
    expected_parent: str = PARENT_OID,
    commit_message: str = "Add dataset files",
) -> _CommitKwargs:
    return {
        "revision": revision,
        "expected_parent": expected_parent,
        "commit_message": commit_message,
    }


def _mark_operations_committed(
    operations: list[CommitOperationAdd] | None,
    *,
    committed: bool,
) -> None:
    if operations is None:
        return
    for operation in operations:
        operation._is_committed = committed


def _mock_hf_api(
    mock_hf_api_cls: MagicMock,
    *,
    head_before: str,
    commit_info: CommitInfo,
    committed: bool = False,
) -> MagicMock:
    mock_api = mock_hf_api_cls.return_value
    mock_api.repo_info.return_value = MagicMock(sha=head_before)

    def _create_commit(
        *args: object,
        operations: list[CommitOperationAdd] | None = None,
        **kwargs: object,
    ) -> CommitInfo:
        _mark_operations_committed(operations, committed=committed)
        return commit_info

    mock_api.create_commit.side_effect = _create_commit
    return mock_api


def test_commit_rejects_empty_files(hf_loc: HFLocation) -> None:
    with pytest.raises(ValueError, match="files must be non-empty"):
        commit_dataset_files_to_hf([], hf_loc, **_commit_kwargs())


@patch("dr_hf.io.HfApi")
def test_commit_rejects_missing_local_file(
    mock_hf_api_cls: MagicMock,
    hf_loc: HFLocation,
    tmp_path: Path,
) -> None:
    entry = DatasetFileCommitEntry(
        local_path=tmp_path / "missing.parquet",
        repo_path="data/missing.parquet",
    )
    with pytest.raises(FileNotFoundError, match="Local file not found"):
        commit_dataset_files_to_hf([entry], hf_loc, **_commit_kwargs())
    mock_hf_api_cls.assert_not_called()


@patch("dr_hf.io.HfApi")
def test_commit_rejects_duplicate_repo_paths(
    mock_hf_api_cls: MagicMock,
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
    mock_hf_api_cls.assert_not_called()


@patch("dr_hf.io.HfApi")
def test_commit_rejects_empty_expected_parent(
    mock_hf_api_cls: MagicMock,
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
    mock_hf_api_cls.assert_not_called()


@patch("dr_hf.io.HfApi")
def test_commit_rejects_invalid_expected_parent_oid(
    mock_hf_api_cls: MagicMock,
    hf_loc: HFLocation,
    local_file: Path,
) -> None:
    entry = _make_entry(local_file, "data/file.parquet")
    with pytest.raises(
        ValueError, match="expected_parent must be a valid commit OID"
    ):
        commit_dataset_files_to_hf(
            [entry],
            hf_loc,
            revision="main",
            expected_parent="not-a-sha",
            commit_message="Add file",
        )
    mock_hf_api_cls.assert_not_called()


@patch("dr_hf.io.HfApi")
def test_commit_rejects_four_char_expected_parent_oid(
    mock_hf_api_cls: MagicMock,
    hf_loc: HFLocation,
    local_file: Path,
) -> None:
    entry = _make_entry(local_file, "data/file.parquet")
    with pytest.raises(
        ValueError, match="expected_parent must be a valid commit OID"
    ):
        commit_dataset_files_to_hf(
            [entry],
            hf_loc,
            revision="main",
            expected_parent="abcd",
            commit_message="Add file",
        )
    mock_hf_api_cls.assert_not_called()


@patch("dr_hf.io.HfApi")
def test_commit_rejects_invalid_repo_path_with_parent_dir(
    mock_hf_api_cls: MagicMock,
    hf_loc: HFLocation,
    local_file: Path,
) -> None:
    entry = _make_entry(local_file, "../escape.parquet")
    with pytest.raises(ValueError, match="repo_path must not contain"):
        commit_dataset_files_to_hf([entry], hf_loc, **_commit_kwargs())
    mock_hf_api_cls.assert_not_called()


@patch("dr_hf.io.HfApi")
def test_commit_rejects_absolute_repo_path(
    mock_hf_api_cls: MagicMock,
    hf_loc: HFLocation,
    local_file: Path,
) -> None:
    entry = _make_entry(local_file, "/absolute/path.parquet")
    with pytest.raises(ValueError, match="repo_path must be relative"):
        commit_dataset_files_to_hf([entry], hf_loc, **_commit_kwargs())
    mock_hf_api_cls.assert_not_called()


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
    _mock_hf_api(
        mock_hf_api_cls,
        head_before=PARENT_OID,
        commit_info=CommitInfo(
            commit_url=(
                "https://huggingface.co/datasets/test-org/test-dataset/"
                f"commit/{NEW_COMMIT_OID}"
            ),
            commit_message="Add dataset files",
            commit_description="",
            oid=NEW_COMMIT_OID,
        ),
        committed=True,
    )

    result = commit_dataset_files_to_hf(
        entries,
        hf_loc,
        revision="dev",
        expected_parent=PARENT_OID,
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
    assert kwargs["parent_commit"] == PARENT_OID
    assert kwargs["create_pr"] is False
    assert kwargs["commit_message"] == "Add dataset files"
    assert len(kwargs["operations"]) == 2
    assert kwargs["operations"][0].path_in_repo == "data/a.parquet"
    assert kwargs["operations"][1].path_in_repo == "data/b.parquet"
    assert all(op._is_committed for op in kwargs["operations"])

    assert result.commit_oid == NEW_COMMIT_OID
    assert result.created is True
    assert str(result.commit_url).endswith(f"/commit/{NEW_COMMIT_OID}")
    assert result.pr_url is None
    assert str(result.file_urls["data/a.parquet"]).endswith(
        "/resolve/dev/data/a.parquet"
    )
    assert str(result.file_urls["data/b.parquet"]).endswith(
        "/resolve/dev/data/b.parquet"
    )


@patch("dr_hf.io.HfApi")
def test_commit_maps_pr_fields(
    mock_hf_api_cls: MagicMock,
    hf_loc: HFLocation,
    local_file: Path,
) -> None:
    entry = _make_entry(local_file, "data/file.parquet")
    _mock_hf_api(
        mock_hf_api_cls,
        head_before=PARENT_OID,
        commit_info=CommitInfo(
            commit_url=(
                "https://huggingface.co/datasets/test-org/test-dataset/"
                f"commit/{NEW_COMMIT_OID}"
            ),
            commit_message="Add dataset files",
            commit_description="details",
            oid=NEW_COMMIT_OID,
            pr_url="https://huggingface.co/datasets/test-org/test-dataset/discussions/7",
        ),
        committed=True,
    )

    result = commit_dataset_files_to_hf(
        [entry],
        hf_loc,
        **_commit_kwargs(),
        create_pr=True,
    )

    assert (
        mock_hf_api_cls.return_value.create_commit.call_args.kwargs[
            "create_pr"
        ]
        is True
    )
    assert result.created is True
    assert result.pr_num == 7
    assert result.pr_revision == "refs/pr/7"
    assert str(result.pr_url).endswith("/discussions/7")
    assert str(result.file_urls["data/file.parquet"]).endswith(
        "/resolve/refs%2Fpr%2F7/data/file.parquet"
    )


@patch("dr_hf.io.HfApi")
def test_commit_noop_when_oid_matches_expected_parent(
    mock_hf_api_cls: MagicMock,
    hf_loc: HFLocation,
    local_file: Path,
) -> None:
    entry = _make_entry(local_file, "data/file.parquet")
    _mock_hf_api(
        mock_hf_api_cls,
        head_before=PARENT_OID,
        commit_info=CommitInfo(
            commit_url=(
                "https://huggingface.co/datasets/test-org/test-dataset/"
                f"commit/{PARENT_OID}"
            ),
            commit_message="Add dataset files",
            commit_description="",
            oid=PARENT_OID,
        ),
        committed=False,
    )

    result = commit_dataset_files_to_hf(
        [entry],
        hf_loc,
        revision="main",
        expected_parent=PARENT_OID,
        commit_message="Add dataset files",
    )

    assert result.created is False
    assert result.commit_oid == PARENT_OID


@patch("dr_hf.io.HfApi")
def test_commit_noop_abbreviated_parent_matches_full_oid(
    mock_hf_api_cls: MagicMock,
    hf_loc: HFLocation,
    local_file: Path,
) -> None:
    entry = _make_entry(local_file, "data/file.parquet")
    _mock_hf_api(
        mock_hf_api_cls,
        head_before=PARENT_OID,
        commit_info=CommitInfo(
            commit_url=(
                "https://huggingface.co/datasets/test-org/test-dataset/"
                f"commit/{PARENT_OID}"
            ),
            commit_message="Add dataset files",
            commit_description="",
            oid=PARENT_OID,
        ),
        committed=False,
    )

    result = commit_dataset_files_to_hf(
        [entry],
        hf_loc,
        revision="main",
        expected_parent=PARENT_OID_ABBREV,
        commit_message="Add dataset files",
    )

    assert result.created is False
    assert result.commit_oid == PARENT_OID


@patch("dr_hf.io.HfApi")
def test_commit_rejects_stale_parent_noop(
    mock_hf_api_cls: MagicMock,
    hf_loc: HFLocation,
    local_file: Path,
) -> None:
    entry = _make_entry(local_file, "data/file.parquet")
    _mock_hf_api(
        mock_hf_api_cls,
        head_before=HEAD_OID_B,
        commit_info=CommitInfo(
            commit_url=(
                "https://huggingface.co/datasets/test-org/test-dataset/"
                f"commit/{HEAD_OID_B}"
            ),
            commit_message="Add dataset files",
            commit_description="",
            oid=HEAD_OID_B,
        ),
        committed=False,
    )

    with pytest.raises(ValueError, match="expected_parent is stale"):
        commit_dataset_files_to_hf(
            [entry],
            hf_loc,
            revision="main",
            expected_parent=STALE_PARENT_OID,
            commit_message="Add dataset files",
        )


@patch("dr_hf.io.HfApi")
def test_commit_race_noop_after_external_advance(
    mock_hf_api_cls: MagicMock,
    hf_loc: HFLocation,
    local_file: Path,
) -> None:
    entry = _make_entry(local_file, "data/file.parquet")
    _mock_hf_api(
        mock_hf_api_cls,
        head_before=PARENT_OID,
        commit_info=CommitInfo(
            commit_url=(
                "https://huggingface.co/datasets/test-org/test-dataset/"
                f"commit/{HEAD_OID_B}"
            ),
            commit_message="Add dataset files",
            commit_description="",
            oid=HEAD_OID_B,
        ),
        committed=False,
    )

    result = commit_dataset_files_to_hf(
        [entry],
        hf_loc,
        revision="main",
        expected_parent=PARENT_OID,
        commit_message="Add dataset files",
    )

    assert result.created is False
    assert result.commit_oid == HEAD_OID_B


@patch("dr_hf.io.HfApi")
def test_commit_pr_noop_when_oid_matches_expected_parent(
    mock_hf_api_cls: MagicMock,
    hf_loc: HFLocation,
    local_file: Path,
) -> None:
    entry = _make_entry(local_file, "data/file.parquet")
    _mock_hf_api(
        mock_hf_api_cls,
        head_before=PARENT_OID,
        commit_info=CommitInfo(
            commit_url=(
                "https://huggingface.co/datasets/test-org/test-dataset/"
                f"commit/{PARENT_OID}"
            ),
            commit_message="Add dataset files",
            commit_description="",
            oid=PARENT_OID,
        ),
        committed=False,
    )

    result = commit_dataset_files_to_hf(
        [entry],
        hf_loc,
        **_commit_kwargs(),
        create_pr=True,
    )

    assert result.created is False
    assert result.commit_oid == PARENT_OID
    assert result.pr_url is None


@patch("dr_hf.io.HfApi")
def test_commit_pr_noop_when_parent_older_than_head(
    mock_hf_api_cls: MagicMock,
    hf_loc: HFLocation,
    local_file: Path,
) -> None:
    entry = _make_entry(local_file, "data/file.parquet")
    _mock_hf_api(
        mock_hf_api_cls,
        head_before=HEAD_OID_B,
        commit_info=CommitInfo(
            commit_url=(
                "https://huggingface.co/datasets/test-org/test-dataset/"
                f"commit/{HEAD_OID_B}"
            ),
            commit_message="Add dataset files",
            commit_description="",
            oid=HEAD_OID_B,
        ),
        committed=False,
    )

    result = commit_dataset_files_to_hf(
        [entry],
        hf_loc,
        revision="main",
        expected_parent=PARENT_OID,
        commit_message="Add dataset files",
        create_pr=True,
    )

    assert result.created is False
    assert result.commit_oid == HEAD_OID_B
    assert result.pr_url is None


@patch("dr_hf.io.HfApi")
def test_commit_real_commit_uses_new_oid(
    mock_hf_api_cls: MagicMock,
    hf_loc: HFLocation,
    local_file: Path,
) -> None:
    entry = _make_entry(local_file, "data/file.parquet")
    _mock_hf_api(
        mock_hf_api_cls,
        head_before=PARENT_OID,
        commit_info=CommitInfo(
            commit_url=(
                "https://huggingface.co/datasets/test-org/test-dataset/"
                f"commit/{NEW_COMMIT_OID}"
            ),
            commit_message="Add dataset files",
            commit_description="",
            oid=NEW_COMMIT_OID,
        ),
        committed=True,
    )

    result = commit_dataset_files_to_hf(
        [entry],
        hf_loc,
        **_commit_kwargs(),
    )

    assert result.created is True
    assert result.commit_oid == NEW_COMMIT_OID
