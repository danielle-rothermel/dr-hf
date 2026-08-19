from __future__ import annotations

import pytest

from dr_hf.location import HFLocation


def test_hflocation_basic() -> None:
    loc = HFLocation(org="allenai", repo_name="test-dataset")
    assert loc.repo_id == "allenai/test-dataset"
    assert loc.repo_uri == "hf://datasets/allenai/test-dataset"


def test_hflocation_with_filepaths() -> None:
    loc = HFLocation(
        org="allenai",
        repo_name="test-dataset",
        filepaths=["data/train.parquet", "data/test.parquet"],
    )
    assert loc.filepaths == ["data/train.parquet", "data/test.parquet"]
    paths = loc.resolve_filepaths()
    assert len(paths) == 2


def test_hflocation_from_uri_simple() -> None:
    loc = HFLocation.from_uri("hf://datasets/allenai/test-dataset")
    assert loc.org == "allenai"
    assert loc.repo_name == "test-dataset"


def test_hflocation_from_uri_with_path() -> None:
    loc = HFLocation.from_uri(
        "hf://datasets/allenai/test-dataset/data/train.parquet"
    )
    assert loc.org == "allenai"
    assert loc.repo_name == "test-dataset"
    assert loc.filepaths == ["data/train.parquet"]


def test_hflocation_from_uri_filepaths_keyword_only() -> None:
    with pytest.raises(TypeError):
        HFLocation.from_uri(
            "hf://datasets/allenai/test-dataset",
            ["data/train.parquet"],  # ty: ignore[too-many-positional-arguments]
        )


def test_hflocation_from_uri_invalid() -> None:
    with pytest.raises(AssertionError):
        HFLocation.from_uri("invalid://not-hf")

    with pytest.raises(AssertionError):
        HFLocation.from_uri("")


def test_hflocation_norm_posix() -> None:
    assert HFLocation.norm_posix("/data/file.parquet") == "data/file.parquet"
    assert HFLocation.norm_posix("data/file.parquet") == "data/file.parquet"


def test_hflocation_build_local_dir() -> None:
    loc = HFLocation(org="allenai", repo_name="test-dataset")
    local_dir = loc.build_local_dir("/cache")
    assert local_dir == "/cache/allenai/test-dataset"


def test_hflocation_get_path_uri() -> None:
    loc = HFLocation(org="allenai", repo_name="test-dataset")
    uri = loc.get_path_uri("data/train.parquet")
    assert uri == "hf://datasets/allenai/test-dataset/data/train.parquet"
