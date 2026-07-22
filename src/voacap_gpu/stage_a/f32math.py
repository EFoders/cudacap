"""REAL*4 arithmetic helpers for Stage A ports of float32 Fortran.

gfortran maps REAL*4 intrinsics onto the C float functions (sinf, cosf,
acosf, atanf), which modern glibc implements correctly rounded.  The
closest portable equivalent is to evaluate in binary64 and round once to
binary32: that equals the correctly-rounded binary32 result except for
double-rounding coincidences (odds ~2^-29 per call), which the dump tests'
ULP bars absorb.  +, -, *, / and sqrt on float32 are correctly rounded in
numpy and gfortran alike, so those are used directly.
"""

import numpy as np

F32 = np.float32
F64 = np.float64


def sin32(x):
    return F32(np.sin(F64(x)))


def cos32(x):
    return F32(np.cos(F64(x)))


def acos32(x):
    return F32(np.arccos(F64(x)))


def atan32(x):
    return F32(np.arctan(F64(x)))
