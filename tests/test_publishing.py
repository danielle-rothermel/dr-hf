from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import CommitInfo, CommitOperationAdd, GitRefInfo
from pydantic import ValidationError

from dr_hf import ModelPin, ModelProvenance, TagConflictError, publish_model

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

SOURCE_SHA = "91c54ad1727ee830252e457677f467be0bfd8a57"
NEW_SHA = "c3d4e5f6789012345678901234567890ab12cd34"
OTHER_SHA = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
REPO_ID = "drotherm/DataDecide-dolma1_7-150M"


@pytest.fixture
def provenance() -> ModelProvenance:
    return ModelProvenance(
        source_repo_id="allenai/DataDecide-dolma1_7-150M",
        source_revision=SOURCE_SHA,
        conversion_tool="datadec.models.convert 0.1.0",
        verification={"max_abs_logit_diff": 1.5e-5, "batch": "4 x 128"},
        notes="seed ordinal 0 is original seed 'default'",
    )


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    root = tmp_path / "converted"
    (root / "sub").mkdir(parents=True)
    (root / "config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "sub" / "extra.txt").write_text("x")
    (root / ".cache").mkdir()
    (root / ".cache" / "lock").write_text("ignored")
    return root


@pytest.fixture
def api() -> Generator[MagicMock]:
    with patch("dr_hf.publishing.HfApi") as api_cls:
        mock_api = api_cls.return_value
        mock_api.create_commit.return_value = CommitInfo(
            commit_url=f"https://huggingface.co/{REPO_ID}/commit/{NEW_SHA}",
            commit_message="msg",
            commit_description="",
            oid=NEW_SHA,
        )
        mock_api.list_repo_refs.return_value = MagicMock(tags=[])
        yield mock_api


def _publish(
    model_dir: Path,
    provenance: ModelProvenance,
    **overrides: object,
) -> ModelPin:
    kwargs: dict[str, object] = {
        "repo_id": REPO_ID,
        "branch": "step1000-seed0",
        "tags": (),
        "card_markdown": "# card\n",
        "provenance": provenance,
        "private": True,
        "commit_message": "Add step1000 seed 0",
    }
    kwargs.update(overrides)
    return publish_model(model_dir, **kwargs)  # type: ignore[arg-type]


def test_publish_one_commit_with_card_and_provenance(
    api: MagicMock, model_dir: Path, provenance: ModelProvenance
) -> None:
    pin = _publish(model_dir, provenance)

    assert pin == ModelPin(repo_id=REPO_ID, revision=NEW_SHA)
    api.create_repo.assert_called_once_with(
        REPO_ID, repo_type="model", private=True, exist_ok=True
    )
    api.create_branch.assert_called_once_with(
        REPO_ID, branch="step1000-seed0", repo_type="model", exist_ok=True
    )
    api.create_commit.assert_called_once()
    args, kwargs = api.create_commit.call_args
    assert args[0] == REPO_ID
    assert kwargs["revision"] == "step1000-seed0"
    assert kwargs["repo_type"] == "model"
    assert kwargs["commit_message"] == "Add step1000 seed 0"
    operations: list[CommitOperationAdd] = args[1]
    by_path = {op.path_in_repo: op for op in operations}
    assert sorted(by_path) == [
        "README.md",
        "config.json",
        "model.safetensors",
        "provenance.json",
        "sub/extra.txt",
    ]
    assert by_path["README.md"].path_or_fileobj == b"# card\n"
    sidecar = by_path["provenance.json"].path_or_fileobj
    assert isinstance(sidecar, bytes)
    assert json.loads(sidecar) == provenance.model_dump(mode="json")
    api.create_tag.assert_not_called()


def test_publish_to_main_skips_branch_creation(
    api: MagicMock, model_dir: Path, provenance: ModelProvenance
) -> None:
    _publish(model_dir, provenance, branch="main")
    api.create_branch.assert_not_called()


def test_publish_creates_tags_at_commit(
    api: MagicMock, model_dir: Path, provenance: ModelProvenance
) -> None:
    api.list_repo_refs.return_value = MagicMock(
        tags=[
            GitRefInfo(
                name="final-seed1",
                ref="refs/tags/final-seed1",
                target_commit=NEW_SHA,
            )
        ]
    )
    _publish(model_dir, provenance, tags=("final-seed0", "final-seed1"))
    api.create_tag.assert_called_once_with(
        REPO_ID, tag="final-seed0", revision=NEW_SHA, repo_type="model"
    )


def test_publish_tag_elsewhere_raises(
    api: MagicMock, model_dir: Path, provenance: ModelProvenance
) -> None:
    api.list_repo_refs.return_value = MagicMock(
        tags=[
            GitRefInfo(
                name="final-seed0",
                ref="refs/tags/final-seed0",
                target_commit=OTHER_SHA,
            )
        ]
    )
    with pytest.raises(TagConflictError, match="final-seed0"):
        _publish(model_dir, provenance, tags=("final-seed0",))
    api.create_tag.assert_not_called()


@pytest.mark.parametrize("reserved", ["README.md", "provenance.json"])
def test_publish_rejects_reserved_files_before_network(
    api: MagicMock,
    model_dir: Path,
    provenance: ModelProvenance,
    reserved: str,
) -> None:
    (model_dir / reserved).write_text("local")
    with pytest.raises(ValueError, match="must not contain"):
        _publish(model_dir, provenance)
    api.create_repo.assert_not_called()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_provenance_rejects_non_json_verification(value: object) -> None:
    with pytest.raises(ValidationError, match="strict JSON"):
        ModelProvenance(
            source_repo_id="org/src",
            source_revision=SOURCE_SHA,
            conversion_tool="tool 0.1",
            verification={"diff": value},
        )


def test_publish_rejects_missing_dir(
    api: MagicMock, tmp_path: Path, provenance: ModelProvenance
) -> None:
    with pytest.raises(NotADirectoryError):
        _publish(tmp_path / "missing", provenance)
    api.create_repo.assert_not_called()


def test_provenance_requires_source_sha() -> None:
    with pytest.raises(ValidationError, match="40-character"):
        ModelProvenance(
            source_repo_id="org/src",
            source_revision="main",
            conversion_tool="tool",
            verification={},
        )


def test_publish_creates_each_requested_tag_once(
    api: MagicMock, model_dir: Path, provenance: ModelProvenance
) -> None:
    _publish(model_dir, provenance, tags=("final-seed0", "final-seed0"))
    api.create_tag.assert_called_once_with(
        REPO_ID, tag="final-seed0", revision=NEW_SHA, repo_type="model"
    )
