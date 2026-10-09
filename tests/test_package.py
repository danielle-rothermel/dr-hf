from __future__ import annotations

import tomllib
from pathlib import Path


def test_import_dr_hf_version_matches_pyproject() -> None:
    """Locked-env import smoke test.

    See CI minimum-deps job for floor coverage.
    """
    import dr_hf

    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    with pyproject.open("rb") as file:
        version = tomllib.load(file)["project"]["version"]
    assert dr_hf.__version__ == version
