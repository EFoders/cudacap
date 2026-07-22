"""Sanity checks for the package scaffold and test environment."""

import numpy as np

import voacap_gpu


def test_package_imports():
    assert voacap_gpu.__version__


def test_float32_is_default_working_dtype():
    # Guard the mixed-precision convention: Stage A REAL*4 emulation relies
    # on explicit float32 narrowing at each step.
    x = np.float32(16.25)
    assert x.dtype == np.float32
