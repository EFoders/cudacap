"""SETLNG — replicate the computed sample area across the array slots.

For a short P2P path only one sample area is actually computed, but the
downstream frequency x hop loop indexes several sample-area slots.  SETLNG
fills them in by copying, choosing the pattern from KFX (the sample-area
count code in /RON/):

* ``KFX >= 3`` — all slots already populated; do nothing.
* ``KFX <= 1`` — one computed area: copy area 1's per-area scalars + layer
  parameters (FS/FI/YI/HI) into areas 2..5, and its ionogram
  (HPRIM/HTRUE/FVERT/AFAC/IFOB) into areas 2..3.
* ``KFX == 2`` — three computed areas: copy area 3 into areas 4..5, and the
  ionogram of area 2 into area 3.

Pure array replication — no arithmetic, hence bit-exact by construction; the
tier-B dump pins the control flow and the (deliberately asymmetric) index
ranges.  The routine touches /RON/, /ES/, /GEOG/ (LayerState) and the /RAYS/
reflectrix IFOB (RaysState), so it takes both.  Mirrors setlng.for.
"""

from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.raysstate import RaysState


def setlng(state: LayerState, rays: RaysState) -> None:
    s = state
    # IF(KFX-3) 105,100,100  — KFX>=3: everything already filled.
    if s.kfx >= 3:
        return
    # 105  IF(KFX-1) 110,110,135
    if s.kfx <= 1:
        # 110  ONE SAMPLE AREA: area 1 -> areas 2..5 (scalars + FS/FI/YI/HI)
        for a in range(2, 6):
            s.clat[a] = s.clat[1]
            s.clong[a] = s.clong[1]
            s.glat[a] = s.glat[1]
            s.rd[a] = s.rd[1]
            s.gyz[a] = s.gyz[1]
            s.rat[a] = s.rat[1]
            s.gmdip[a] = s.gmdip[1]
            s.clck[a] = s.clck[1]
            s.abiy[a] = s.abiy[1]
            s.artic[a] = s.artic[1]
            s.sigpat[a] = s.sigpat[1]
            s.epspat[a] = s.epspat[1]
            s.hs[a] = s.hs[1]
            s.fs[1:4, a] = s.fs[1:4, 1]
            s.fi[1:4, a] = s.fi[1:4, 1]
            s.yi[1:4, a] = s.yi[1:4, 1]
            s.hi[1:4, a] = s.hi[1:4, 1]
        # ionogram: area 1 -> areas 2..3 only
        for a in range(2, 4):
            s.hprim[1:31, a] = s.hprim[1:31, 1]
            s.htrue[1:31, a] = s.htrue[1:31, 1]
            s.fvert[1:31, a] = s.fvert[1:31, 1]
            s.afac[1:31, a] = s.afac[1:31, 1]
            rays.ifob[1:41, 1:31, a] = rays.ifob[1:41, 1:31, 1]
        return
    # 135  THREE SAMPLE AREAS: area 3 -> areas 4..5 (scalars + FS/FI/YI/HI)
    for a in range(4, 6):
        s.clat[a] = s.clat[3]
        s.clong[a] = s.clong[3]
        s.glat[a] = s.glat[3]
        s.rd[a] = s.rd[3]
        s.gyz[a] = s.gyz[3]
        s.rat[a] = s.rat[3]
        s.gmdip[a] = s.gmdip[3]
        s.clck[a] = s.clck[3]
        s.abiy[a] = s.abiy[3]
        s.artic[a] = s.artic[3]
        s.sigpat[a] = s.sigpat[3]
        s.epspat[a] = s.epspat[3]
        s.hs[a] = s.hs[3]
        s.fs[1:4, a] = s.fs[1:4, 3]
        s.fi[1:4, a] = s.fi[1:4, 3]
        s.yi[1:4, a] = s.yi[1:4, 3]
        s.hi[1:4, a] = s.hi[1:4, 3]
    # ionogram: area 2 -> area 3
    s.hprim[1:31, 3] = s.hprim[1:31, 2]
    s.htrue[1:31, 3] = s.htrue[1:31, 2]
    s.fvert[1:31, 3] = s.fvert[1:31, 2]
    s.afac[1:31, 3] = s.afac[1:31, 2]
    rays.ifob[1:41, 1:31, 3] = rays.ifob[1:41, 1:31, 2]
