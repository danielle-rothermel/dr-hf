from __future__ import annotations

from dr_hf.branches import (
    extract_seed_from_branch,
    extract_step_from_branch,
    group_branches_by_seed,
    is_checkpoint_branch,
    parse_branch_name,
    sort_branches_by_step,
)


def test_is_checkpoint_branch_valid() -> None:
    assert is_checkpoint_branch("step0-seed-default")
    assert is_checkpoint_branch("step1000-seed-custom-config")
    assert is_checkpoint_branch("step123456-seed-a")


def test_is_checkpoint_branch_invalid() -> None:
    assert not is_checkpoint_branch("main")
    assert not is_checkpoint_branch("step-seed-default")
    assert not is_checkpoint_branch("stepX-seed-default")
    assert not is_checkpoint_branch("")


def test_parse_branch_name_valid() -> None:
    result = parse_branch_name("step1000-seed-default")
    assert result.valid
    assert result.step == 1000
    assert result.seed == "default"
    assert result.branch == "step1000-seed-default"


def test_parse_branch_name_invalid() -> None:
    result = parse_branch_name("main")
    assert not result.valid
    assert result.step is None
    assert result.seed is None


def test_extract_step_from_branch() -> None:
    assert extract_step_from_branch("step1000-seed-default") == 1000
    assert extract_step_from_branch("step0-seed-test") == 0
    assert extract_step_from_branch("main") == 0


def test_extract_seed_from_branch() -> None:
    assert extract_seed_from_branch("step1000-seed-default") == "default"
    assert (
        extract_seed_from_branch("step0-seed-custom-config") == "custom-config"
    )
    assert extract_seed_from_branch("main") == "unknown"


def test_sort_branches_by_step() -> None:
    branches = ["step1000-seed-a", "step0-seed-a", "step500-seed-a"]
    sorted_branches = sort_branches_by_step(branches)
    assert sorted_branches == [
        "step0-seed-a",
        "step500-seed-a",
        "step1000-seed-a",
    ]


def test_group_branches_by_seed() -> None:
    branches = [
        "step0-seed-default",
        "step1000-seed-default",
        "step0-seed-custom",
        "step500-seed-custom",
    ]
    groups = group_branches_by_seed(branches)

    assert "default" in groups
    assert "custom" in groups
    assert len(groups["default"]) == 2
    assert len(groups["custom"]) == 2
    assert groups["default"] == ["step0-seed-default", "step1000-seed-default"]
    assert groups["custom"] == ["step0-seed-custom", "step500-seed-custom"]
