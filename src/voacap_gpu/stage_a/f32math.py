"""REAL*4 arithmetic helpers for Stage A ports of float32 Fortran.

gfortran maps REAL*4 intrinsics onto the C float functions (sinf, cosf,
acosf, atanf).  These helpers evaluate in binary64 and round once to
binary32, which yields the CORRECTLY ROUNDED binary32 result (modulo
~2^-29 double-rounding coincidences) — the same value GCC's compile-time
MPFR constant folding produces.  Runtime glibc, however, is only
1-ULP-accurate, not correctly rounded: it disagrees with these helpers on
~1% of arguments by 1 ULP (see docs/f32_libm.md for the diagnosed VERSY
case and the plan to emulate glibc's own flt-32 implementations exactly).
The dump tests' ULP bars absorb that noise floor — including where a
routine's conditioning amplifies it.  +, -, *, / and sqrt on float32 are
correctly rounded in numpy and gfortran alike, so those are used
directly.
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


def asin32(x):
    return F32(np.arcsin(F64(x)))


def log32(x):
    """REAL*4 ALOG (natural log)."""
    return F32(np.log(F64(x)))


def log10_32(x):
    """REAL*4 ALOG10 (base-10 log)."""
    return F32(np.log10(F64(x)))


def sqrt32(x):
    """REAL*4 SQRT — correctly rounded in numpy and gfortran alike."""
    return F32(np.sqrt(F64(x)))


def exp32(x):
    return F32(np.exp(F64(x)))


def pow32(x, y):
    """REAL*4 ``x**y`` with a REAL exponent — gfortran emits powf.

    The exponent must already be rounded to binary32 before the binary64
    evaluation: REGMOD's ``**1.98`` is a default-REAL literal, i.e. the
    float32 nearest to 1.98 (1.98000001907...), not 1.98 itself.  Raising to
    the binary64 1.98 would shift the result by far more than the libm floor.
    """
    return F32(np.power(F64(F32(x)), F64(F32(y))))
