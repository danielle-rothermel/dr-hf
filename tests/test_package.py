from __future__ import annotations


def test_import_dr_hf() -> None:
    """Locked-env import smoke test.

    See CI minimum-deps job for floor coverage.
    """
    import dr_hf

    assert dr_hf.__version__
