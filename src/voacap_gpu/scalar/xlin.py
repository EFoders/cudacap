"""Stage A transliteration of the XLIN function
(reference/voacapl/src/voacapw/xlin.for).

General piecewise-linear table interpolation of YN vs XN over column K
(NTAB entries), tolerant of non-monotonic tables: it walks the table and
interpolates in the first bracketing segment, handling increasing, flat, and
decreasing runs separately, and clamps to the endpoints outside the table.
The Fortran flattens XN(NTAB,*)/YN(NTAB,*) and offsets by ISMIN=NTAB*(K-1);
column-major, that is exactly column K of a 2-D array, so this keeps XN/YN
as 1-based 2-D arrays and indexes [i, k] directly.  REAL*4, no transcendentals.
The arithmetic-IF web (labels 2/3/5/7/8/9/10/11/12/19) is preserved
branch-for-branch.
"""

from voacap_gpu.scalar.f32math import F32


def _interp(x, xn, yn, i, k):
    """Label 10: XLIN = A + B*C/D (left-to-right, REAL*4)."""
    a = yn[i, k]
    b = x - xn[i, k]
    c = yn[i + 1, k] - yn[i, k]
    d = xn[i + 1, k] - xn[i, k]
    return a + b * c / d


def xlin(x, xn, yn, ntab, k):
    """XN/YN: 1-based 2-D numpy arrays; interpolate YN vs XN in column K."""
    x = F32(x)
    if xn[1, k] - x > 0.0:  # IF(XN(1)-X) 3,3,2 : >0 -> return YN(1)
        return yn[1, k]
    for j in range(1, ntab):  # DO 19 J=1,NTAB-1
        i = j
        d = xn[i, k] - xn[i + 1, k]
        if d < 0.0:  # 8: increasing segment
            if xn[i, k] - x <= 0.0:  # -> 9
                if x - xn[i + 1, k] < 0.0:  # -> 10
                    return _interp(x, xn, yn, i, k)
                # else -> 19 (continue)
            else:  # -> 10
                return _interp(x, xn, yn, i, k)
        elif d == 0.0:  # 5: flat segment
            if x - xn[i, k] == 0.0:  # -> 7
                return yn[i, k]
            # else -> 19 (continue)
        else:  # 11: decreasing segment
            if xn[i, k] - x < 0.0:  # -> 19 (continue)
                pass
            else:  # -> 12
                if x - xn[i + 1, k] > 0.0:  # -> 10
                    return _interp(x, xn, yn, i, k)
                # else -> 19 (continue)
    return yn[ntab, k]  # XLIN = YN(ISMAX), ISMAX = NTAB*K -> row NTAB, col K
