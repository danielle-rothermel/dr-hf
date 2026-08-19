from __future__ import annotations

from typing import TYPE_CHECKING

import huggingface_hub
import pytest
from huggingface_hub import CommitOperationAdd, HfApi

from dr_hf.io import commit_dataset_files_to_hf
from dr_hf.location import HFLocation
from dr_hf.models import DatasetFileCommitEntry
from tests.hf_hub_http_mock import (
    NEW_COMMIT_OID,
    PARENT_OID,
    REPO_ID,
    stub_hf_hub_http,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.usefixtures("hf_hub_version")


@pytest.fixture
def hf_hub_version() -> str:
    version = huggingface_hub.__version__
    if version.startswith("0.24."):
        return version
    if version.startswith("1.3."):
        return version
    pytest.skip(
        "SDK commit contract tests run under locked Hub 1.3.x or floor 0.24.x"
    )


@pytest.fixture
def hf_loc() -> HFLocation:
    return HFLocation(org="test-org", repo_name="test-dataset")


def test_sdk_create_commit_marks_is_committed_on_changed_file(
    tmp_path: Path,
) -> None:
    repo_path = "data/file.parquet"
    local_path = tmp_path / "file.parquet"
    content = b"changed-file-content"
    local_path.write_bytes(content)

    operation = CommitOperationAdd(
        path_in_repo=repo_path,
        path_or_fileobj=str(local_path),
    )
    with stub_hf_hub_http(
        repo_path=repo_path,
        file_content=content,
        upload_case="changed",
    ) as calls:
        commit_info = HfApi(token="hf_test").create_commit(
            REPO_ID,
            [operation],
            commit_message="Add dataset files",
            repo_type="dataset",
            revision="main",
            parent_commit=PARENT_OID,
        )

    assert operation._is_committed is True
    assert commit_info.oid == NEW_COMMIT_OID
    assert any(method == "POST" and "/commit/" in url for method, url in calls)


def test_sdk_create_commit_leaves_is_committed_false_on_unchanged_early_return(
    tmp_path: Path,
) -> None:
    repo_path = "data/file.parquet"
    local_path = tmp_path / "file.parquet"
    content = b"unchanged-file-content"
    local_path.write_bytes(content)

    operation = CommitOperationAdd(
        path_in_repo=repo_path,
        path_or_fileobj=str(local_path),
    )
    with stub_hf_hub_http(
        repo_path=repo_path,
        file_content=content,
        upload_case="unchanged",
    ) as calls:
        commit_info = HfApi(token="hf_test").create_commit(
            REPO_ID,
            [operation],
            commit_message="Add dataset files",
            repo_type="dataset",
            revision="main",
            parent_commit=PARENT_OID,
        )

    assert operation._is_committed is False
    assert commit_info.oid == PARENT_OID
    assert not any("/commit/" in url for _, url in calls)


def test_commit_dataset_files_to_hf_created_from_sdk_is_committed_changed(
    hf_loc: HFLocation,
    tmp_path: Path,
) -> None:
    repo_path = "data/file.parquet"
    local_path = tmp_path / "file.parquet"
    content = b"changed-file-content"
    local_path.write_bytes(content)
    entry = DatasetFileCommitEntry(
        local_path=local_path,
        repo_path=repo_path,
    )

    with stub_hf_hub_http(
        repo_path=repo_path,
        file_content=content,
        upload_case="changed",
    ):
        result = commit_dataset_files_to_hf(
            [entry],
            hf_loc,
            revision="main",
            expected_parent=PARENT_OID,
            commit_message="Add dataset files",
            hf_token="hf_test",
        )

    assert result.created is True
    assert result.commit_oid == NEW_COMMIT_OID


def test_commit_dataset_files_to_hf_created_from_sdk_is_committed_unchanged(
    hf_loc: HFLocation,
    tmp_path: Path,
) -> None:
    repo_path = "data/file.parquet"
    local_path = tmp_path / "file.parquet"
    content = b"unchanged-file-content"
    local_path.write_bytes(content)
    entry = DatasetFileCommitEntry(
        local_path=local_path,
        repo_path=repo_path,
    )

    with stub_hf_hub_http(
        repo_path=repo_path,
        file_content=content,
        upload_case="unchanged",
    ):
        result = commit_dataset_files_to_hf(
            [entry],
            hf_loc,
            revision="main",
            expected_parent=PARENT_OID,
            commit_message="Add dataset files",
            hf_token="hf_test",
        )

    assert result.created is False
    assert result.commit_oid == PARENT_OID
