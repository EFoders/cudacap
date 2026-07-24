"""COMMON /CON/ constants, verbatim from blkdat.for:129-131 (REAL*4).

Note these are deliberately NOT high-precision: PI is 3.1415926, D2R/R2D
are truncated.  DAZEL has its own (different, double-precision) constants.

(Named constants.py rather than con.py: CON is a reserved DOS device name
and the path would be unusable by most Windows tooling.)
"""

import numpy as np

F32 = np.float32

D2R = F32(0.01745329251)
DCL = F32(1.28)
GAMA = F32(0.57721566)
PI = F32(3.1415926)
PI2 = F32(6.283185307)
PIO2 = F32(1.570796326)
R2D = F32(57.295779513)
RZ = F32(6370.0)
VOFL = F32(299.79246)
