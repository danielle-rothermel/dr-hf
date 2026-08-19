from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import BaseModel, Field, HttpUrl, computed_field, field_validator

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["HFLocation", "HFRepoID", "HFResource"]

HFRepoID = Annotated[
    str,
    Field(
        description=(
            "Hugging Face repo ID (e.g. allenai/DataDecide-eval-results)"
        )
    ),
]

HFResource = Annotated[
    str,
    Field(description="Hugging Face resource URI (e.g. hf://datasets/...)"),
]


class HFLocation(BaseModel):
    uri_prefix: ClassVar[str] = "hf://"
    repo_type: ClassVar[str] = "datasets"
    repo_type_aliases: ClassVar[tuple[str, ...]] = ("dataset", "datasets")

    org: str
    repo_name: str
    filepaths: list[str] | None = None

    local_path_include_org: bool = True
    local_path_include_repo: bool = True

    @computed_field
    @property
    def repo_id(self) -> HFRepoID:
        return f"{self.org}/{self.repo_name}"

    @computed_field
    @property
    def repo_uri(self) -> HFResource:
        return f"hf://{self.repo_type}/{self.repo_id}"

    @computed_field
    @property
    def repo_link(self) -> HttpUrl:
        return HttpUrl(
            f"https://huggingface.co/{self.repo_type}/{self.repo_id}"
        )

    @computed_field
    @property
    def rest_api_repo_url(self) -> HttpUrl:
        api_base_url = "https://huggingface.co/api"
        return HttpUrl(f"{api_base_url}/{self.repo_type}/{self.repo_id}")

    @computed_field
    @property
    def hf_hub_repo_type(self) -> str:
        if self.repo_type != "datasets":
            raise ValueError(f"Invalid repo type: {self.repo_type}")
        return "dataset"

    @classmethod
    def from_uri(
        cls,
        uri: str,
        *,
        filepaths: Iterable[str] | None = None,
    ) -> HFLocation:
        if not isinstance(uri, str):
            raise TypeError("HF URI must be a non-empty string.")
        if not uri:
            raise AssertionError("HF URI must be a non-empty string.")
        prefix = cls.uri_prefix
        if not uri.startswith(prefix):
            raise AssertionError(
                f"HF URI must start with '{prefix}'. Got: {uri!r}"
            )
        stripped = uri[len(prefix) :]

        parts = [part for part in stripped.split("/") if part]
        if len(parts) < 2:  # noqa: PLR2004
            raise ValueError(
                "HF URI must include at least org and repo, e.g. 'hf://datasets/org/repo'."
            )

        expected_repo_type = cls.repo_type
        alias_map = {expected_repo_type: expected_repo_type}
        alias_map.update(
            dict.fromkeys(cls.repo_type_aliases, expected_repo_type)
        )

        potential_repo_type = parts[0]
        normalized = alias_map.get(potential_repo_type)
        if normalized:
            if normalized != expected_repo_type:
                raise ValueError(
                    f"HF URI repo type '{potential_repo_type}' does not "
                    f"match expected '{expected_repo_type}'."
                )
            org_repo_parts = parts[1:]
        elif len(parts) == 2:  # noqa: PLR2004
            org_repo_parts = parts
        else:
            raise ValueError(
                "HF URI with nested paths must start with "
                "'hf://datasets/' or 'hf://dataset/'."
            )

        if len(org_repo_parts) < 2:  # noqa: PLR2004
            raise ValueError(
                "HF URI must include both org and repo names, e.g. 'hf://datasets/org/repo'."
            )
        org, repo_name, *sub_path = org_repo_parts
        resolved_paths: list[str] = list(filepaths or [])
        if sub_path:
            resolved_paths.append("/".join(sub_path))
        return cls(
            org=org,
            repo_name=repo_name,
            filepaths=resolved_paths or None,
        )

    @field_validator("filepaths")
    @classmethod
    def posix_norm_filepaths(
        cls, v: list[str | Path] | None
    ) -> list[str] | None:
        if v is None:
            return None
        return [cls.norm_posix(p) for p in v]

    @staticmethod
    def norm_posix(path: str | Path) -> str:
        p = PurePosixPath(str(path))
        return str(p).lstrip("/")

    @staticmethod
    def _is_dir(path: str | Path) -> bool:
        if Path(path).exists():
            return Path(path).is_dir()
        return str(path).endswith(os.path.sep)

    def get_the_single_filepath(
        self, local_dir: str | Path | None = None
    ) -> str:
        return self.resolve_filepaths(local_dir=local_dir, expect_one=True)[0]

    def build_local_dir(
        self, local_dir: str | Path | None = None
    ) -> str | None:
        dir_parts = [str(local_dir or "")]
        if self.local_path_include_org:
            dir_parts.append(self.org)
        if self.local_path_include_repo:
            dir_parts.append(self.repo_name)
        return "/".join(dir_parts)

    def resolve_filepaths(
        self,
        extra_paths: list[str | Path] | None = None,
        local_dir: str | Path | None = None,
        *,
        required: bool = True,
        expect_one: bool = False,
    ) -> list[str]:
        seen: dict[str, None] = {}
        for path in [
            *(self.filepaths or []),
            *[self.norm_posix(p) for p in extra_paths or []],
        ]:
            if path in seen:
                continue
            seen[path] = None
        paths = list(seen.keys())
        if local_dir:
            paths = [
                f"{self.build_local_dir(local_dir)}/{path}" for path in paths
            ]
        assert not (required and not paths), "No filepaths found"
        assert not (expect_one and len(paths) != 1), (
            "Expected exactly one filepath"
        )
        return paths

    def get_path_uri(self, path: str | Path) -> HFResource:
        return f"{self.repo_uri}/{self.norm_posix(path)}"

    def get_path_link(self, path: str | Path) -> HttpUrl:
        path = self.norm_posix(path)
        path_type = "tree" if self._is_dir(path) else "blob"
        return HttpUrl(f"{self.repo_link}/{path_type}/main/{path}")

    def get_rest_path_url(self, path: str | Path) -> HttpUrl:
        path = self.norm_posix(path)
        assert self._is_dir(path), (
            "REST endpoint only supports directories, not files."
        )
        return HttpUrl(f"{self.rest_api_repo_url}/tree/main/{path}")

    def get_file_download_link(self, filepath: str | Path) -> HttpUrl:
        return self.get_file_download_link_for_revision(filepath, "main")

    def get_file_download_link_for_revision(
        self, filepath: str | Path, revision: str
    ) -> HttpUrl:
        path = self.norm_posix(filepath)
        assert not self._is_dir(path), (
            "Download link only supports files, not directories."
        )
        return HttpUrl(f"{self.repo_link}/resolve/{revision}/{path}")

    def get_uris_for_files(
        self,
        filepaths: Iterable[str | Path] | None = None,
        *,
        ignore_cfg_files: bool = False,
    ) -> list[HFResource]:
        items: list[str | Path] = list(filepaths or [])
        if not ignore_cfg_files:
            items.extend(self.filepaths or [])

        seen: dict[str, None] = {}
        for item in items:
            uri = self.get_path_uri(self.norm_posix(item))
            seen.setdefault(uri, None)
        return list(seen.keys())
