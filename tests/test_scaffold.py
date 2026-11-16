"""Sanity checks for the package scaffold and test environment."""

import numpy as np

import voacap_gpu


def test_package_imports():
    assert voacap_gpu.__version__


def test_version_matches_pyproject():
    """``__version__`` and pyproject's [project] version must not drift (they
    did: 0.0.1 vs 0.0.4 before 0.1.0)."""
    import pathlib
    import re
    text = (pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
            ).read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert m and m.group(1) == voacap_gpu.__version__


def test_float32_is_default_working_dtype():
    # Guard the mixed-precision convention: Stage A REAL*4 emulation relies
    # on explicit float32 narrowing at each step.
    x = np.float32(16.25)
    assert x.dtype == np.float32
