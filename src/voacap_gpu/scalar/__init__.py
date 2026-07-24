"""Stage A: scalar NumPy transliterations of the reference Fortran.

Each module here is a line-faithful translation of its ``.for`` counterpart
(control flow, loop structure, and arithmetic order preserved — deliberately
not idiomatic Python). Validated against per-routine oracle dumps generated
by Fortran drivers in ``tools/oracle/dumps/``.

These implementations are kept permanently as a cross-check against the
Stage B (PyTorch) layer; they are not deleted once Stage B passes.

Precision convention: variables that are REAL*4 in the Fortran are computed
with ``np.float32``; the DOUBLE PRECISION exceptions (DAZEL internals, etc.)
use ``np.float64`` — see docs/precision_map.md.
"""
