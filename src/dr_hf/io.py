from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import pandas as pd
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from pydantic import HttpUrl

from .location import HFLocation
from .models import DatasetCommitResult, DatasetFileCommitEntry

if TYPE_CHECKING:
    import duckdb as duckdb_module

__all__ = [
    "cached_download_tables_from_hf",
    "commit_dataset_files_to_hf",
    "get_tables_from_cache",
    "query_hf_with_duckdb",
    "read_local_parquet_paths",
]


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
) -> list[DatasetFileCommitEntry]:
    if not files:
        raise ValueError("files must be non-empty")
    if not revision.strip():
        raise ValueError("revision must be non-empty")
    if not expected_parent.strip():
        raise ValueError("expected_parent must be non-empty")
    if not commit_message.strip():
        raise ValueError("commit_message must be non-empty")

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
    return normalized


def commit_dataset_files_to_hf(
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
    normalized_files = _validate_dataset_commit_inputs(
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
    commit_info = HfApi(token=hf_token).create_commit(
        repo_id=hf_loc.repo_id,
        operations=operations,
        commit_message=commit_message,
        commit_description=commit_description or "",
        repo_type=hf_loc.hf_hub_repo_type,
        revision=revision,
        parent_commit=expected_parent,
        create_pr=create_pr,
    )
    file_urls = {
        entry.repo_path: hf_loc.get_file_download_link_for_revision(
            entry.repo_path, revision
        )
        for entry in normalized_files
    }
    return DatasetCommitResult(
        commit_oid=commit_info.oid,
        commit_url=HttpUrl(commit_info.commit_url),
        commit_message=commit_info.commit_message,
        commit_description=commit_info.commit_description,
        pr_url=HttpUrl(commit_info.pr_url) if commit_info.pr_url else None,
        pr_num=commit_info.pr_num,
        pr_revision=commit_info.pr_revision,
        file_urls=file_urls,
    )


def get_tables_from_cache(
    hf_loc: HFLocation, cache_dir: str | Path
) -> dict[str, pd.DataFrame]:
    local_paths = hf_loc.resolve_filepaths(local_dir=cache_dir)
    for fp in local_paths:
        assert Path(fp).exists(), f"Local file not found: {fp}"
    return read_local_parquet_paths(local_paths)


def query_hf_with_duckdb(
    hf_loc: HFLocation,
    connection: duckdb_module.DuckDBPyConnection,
) -> dict[str, pd.DataFrame]:
    resolved_paths = hf_loc.resolve_filepaths()
    hf_uris = hf_loc.get_uris_for_files(resolved_paths, ignore_cfg_files=True)
    results: dict[str, pd.DataFrame] = {}
    try:
        for filepath, uri in zip(resolved_paths, hf_uris, strict=True):
            results[Path(filepath).stem] = connection.execute(
                "SELECT * FROM read_parquet(?)", [uri]
            ).df()
    except ValueError as e:
        raise ValueError(
            f"Mismatch between resolved_paths ({len(resolved_paths)} items) and "
            f"hf_uris from hf_loc.get_uris_for_files ({len(hf_uris)} items). "
            f"resolved_paths: {resolved_paths}"
        ) from e
    return results


def cached_download_tables_from_hf(
    hf_loc: HFLocation,
    *,
    cache_dir: str | Path,
    hf_token: str | None = None,
    force_download: bool = False,
    verbose: bool = True,
) -> dict[str, str | Path]:
    cache_path = Path(cache_dir)
    local_paths = hf_loc.resolve_filepaths(local_dir=cache_path)

    if not force_download and all(Path(fp).exists() for fp in local_paths):
        if verbose:
            print(f">> All tables already cached:\n - {'\n - '.join(local_paths)}")
        return {Path(fp).stem: fp for fp in local_paths}

    cache_path.mkdir(parents=True, exist_ok=True)
    remote_paths = hf_loc.resolve_filepaths()
    tables: dict[str, str | Path] = {}

    for remote_path in remote_paths:
        local_path = hf_hub_download(
            repo_id=hf_loc.repo_id,
            filename=remote_path,
            repo_type=hf_loc.hf_hub_repo_type,
            token=hf_token,
            local_dir=hf_loc.build_local_dir(cache_path),
            force_download=force_download,
        )
        tables[Path(remote_path).stem] = local_path

    if verbose:
        print(f">> Downloaded {hf_loc.org}/{hf_loc.repo_name} tables:")
        print("\n".join([f" - {rem} -> {loc}" for rem, loc in tables.items()]))
    return tables


def read_local_parquet_paths(
    local_paths: list[str] | list[Path],
) -> dict[str, pd.DataFrame]:
    return {Path(fp).stem: pd.read_parquet(fp) for fp in local_paths}
