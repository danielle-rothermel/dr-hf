from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, datetime

from huggingface_hub import list_repo_refs

from .models import (
    BranchInfo,
    BranchMetadata,
    SeedBranchInfo,
    SeedConfiguration,
)

CHECKPOINT_BRANCH_RE = re.compile(r"^step\d+-seed-.+$")


def get_all_repo_branches(repo_id: str) -> list[str]:
    refs = list_repo_refs(repo_id)
    return [ref.name for ref in refs.branches]


def is_checkpoint_branch(branch: str) -> bool:
    return bool(CHECKPOINT_BRANCH_RE.match(branch))


def get_checkpoint_branches(repo_id: str) -> list[str]:
    all_branches = get_all_repo_branches(repo_id)
    checkpoint_branches = [b for b in all_branches if is_checkpoint_branch(b)]
    return sort_branches_by_step(checkpoint_branches)


def parse_branch_name(branch: str) -> BranchInfo:
    if not is_checkpoint_branch(branch):
        return BranchInfo(branch=branch)

    step: int | None = None
    seed: str | None = None

    step_match = re.search(r"step(\d+)-", branch)
    if step_match:
        step = int(step_match.group(1))

    seed_match = re.search(r"seed-(.+)$", branch)
    if seed_match:
        seed = seed_match.group(1)

    valid = step is not None and seed is not None
    return BranchInfo(branch=branch, valid=valid, step=step, seed=seed)


def extract_step_from_branch(branch: str) -> int:
    parsed = parse_branch_name(branch)
    return parsed.step or 0


def extract_seed_from_branch(branch: str) -> str:
    parsed = parse_branch_name(branch)
    return parsed.seed or "unknown"


def sort_branches_by_step(branches: list[str]) -> list[str]:
    return sorted(branches, key=extract_step_from_branch)


def group_branches_by_seed(branches: list[str]) -> dict[str, list[str]]:
    groups = defaultdict(list)
    for branch in branches:
        seed = extract_seed_from_branch(branch)
        groups[seed].append(branch)

    for seed in groups:
        groups[seed] = sort_branches_by_step(groups[seed])

    return dict(groups)


def get_step_range_for_seed(branches: list[str]) -> tuple[int, int]:
    if not branches:
        return (0, 0)

    steps = [extract_step_from_branch(b) for b in branches]
    return (min(steps), max(steps))


def create_branch_metadata(repo_id: str) -> BranchMetadata:
    all_branches = get_all_repo_branches(repo_id)
    checkpoint_branches = []
    other_branches = []
    for branch in all_branches:
        if is_checkpoint_branch(branch):
            checkpoint_branches.append(branch)
        else:
            other_branches.append(branch)

    seed_groups = group_branches_by_seed(checkpoint_branches)

    seed_configurations: dict[str, SeedConfiguration] = {}
    for seed, branches in seed_groups.items():
        step_range = get_step_range_for_seed(branches)
        seed_configurations[seed] = SeedConfiguration(
            count=len(branches),
            step_range=step_range,
            branches=[
                SeedBranchInfo(step=extract_step_from_branch(b), branch=b)
                for b in branches
            ],
        )

    return BranchMetadata(
        repo_id=repo_id,
        last_updated=datetime.now(UTC),
        total_branches=len(all_branches),
        checkpoint_branches=len(checkpoint_branches),
        seed_configurations=seed_configurations,
        other_branches=sorted(other_branches),
        all_checkpoint_branches=sorted(
            checkpoint_branches, key=extract_step_from_branch
        ),
    )
