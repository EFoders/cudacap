"""COMMON /DATR/ — the Gaussian-integration setup GETHP uses.

WT/XT are the 20-point half Gauss-Legendre weights/abscissae from
blkdat.for (GETHP evaluates the folded 40-point rule via the 1-XT and
1+XT symmetric pair).  NT/NPL/XNPL/TWDIV are set once in voacapw.for's
startup: a 20-point rule with the linear (NPL=1) interval transformation,
so TWDIV = 1/2**NPL = 0.5 and XNPL = 1.0.
"""

import numpy as np

F32 = np.float32

NT = 20
NPL = 1
XNPL = F32(1.0)
TWDIV = F32(1.0) / F32(2.0 ** NPL)  # 0.5

# 1-based (21,): XT[1..20], WT[1..20]
XT = np.zeros(21, dtype=np.float32)
WT = np.zeros(21, dtype=np.float32)
XT[1:] = [0.0387724175, 0.1160840707, 0.1926975807, 0.2681521850,
          0.3419940908, 0.4137792043, 0.4830758017, 0.5494671251,
          0.6125538897, 0.6719566846, 0.7273182552, 0.7783056514,
          0.8246122308, 0.8659595032, 0.9020988070, 0.9328128083,
          0.9579168192, 0.9772599500, 0.9907262387, 0.9982377097]
WT[1:] = [0.0775059480, 0.0770398182, 0.0761103619, 0.0747231691,
          0.0728865824, 0.0706116474, 0.0679120458, 0.0648040135,
          0.0613062425, 0.0574397691, 0.0532278470, 0.0486958076,
          0.0438709082, 0.0387821680, 0.0334601953, 0.0279370070,
          0.0222458492, 0.0164210584, 0.0104982845, 0.0045212771]
