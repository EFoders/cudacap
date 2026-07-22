"""INMUF -- insert over-the-MUF and zero-distance modes, then run REGMOD.

FDIST only finds modes that actually appear in the reflectrix.  Above the MUF
there are none, and at very short distances the ray table does not reach, yet
the prediction still needs a mode to attach a loss to.  INMUF supplies them so
the frequency sweep stays continuous instead of falling off a cliff at the MUF.

It runs in four stages:

1. **Compact** the /MODES/ list, dropping anything with a virtual height below
   70 km and packing what survives into slots 1..INUM.
2. **Insert the circuit MUF mode** for any layer that has none, when the
   frequency is at or above that layer's MUF and the hop count matches.
3. **Recompute a per-hop MUF** for layers whose MUF was established at a
   different hop count, via the ionogram (XLIN) plus the same Martyn spherical
   correction CURMUF uses, and insert a mode where the frequency clears it.
   This stage *overwrites* the circuit MUF distributions YMUF/SIGL/YFOT/SIGU/
   YHPF, which are saved and restored around the REGMOD call.
4. Or, for a very short path, **synthesise a zero-distance mode** straight off
   the ionogram at 90 degrees take-off.

Two behaviours preserved deliberately because they look like bugs:

* The compaction in stage 1 does **not** clear the slots above INUM, so a
  vacated slot keeps its previous occupant.  With contiguous input -- which is
  what FDIST produces -- nothing is left behind, but the trailing NUMMOD count
  scans all six slots and would see a stale mode if it were.
* Stage 3 assigns ``GHOP = 2*PSI``, mutating the hop distance in /MODES/ that
  REGMOD then reads.  The insert changes the geometry for every mode, not just
  the inserted one.

The INFO-gated writes to unit 99 are diagnostics that compute nothing reaching
/MODES/ or /ZON/ (HIMUFS is only ever read inside one), so they are not ported.

float32; COS/SIN/SQRT/ATAN carry the [[f32-libm-noise]] floor.
"""

from voacap_gpu.stage_a.constants import R2D, RZ
from voacap_gpu.stage_a.f32math import F32, atan32, cos32, sin32, sqrt32
from voacap_gpu.stage_a.regmod import regmod
from voacap_gpu.stage_a.xlin import xlin

_EPS = F32(0.4001)
_HMIN = F32(70.0)


def inmuf(state, modes, muf, sigd, anois, rays, cant, zon,
          freq, jmode, gcd, ihop) -> int:
    """Insert modes, call REGMOD, return NUMMOD.  Mutates `modes` and `zon`.

    Arguments match :func:`~voacap_gpu.stage_a.regmod.regmod` plus `ihop`, the
    hop count under test.  `muf` is restored on the way out if stage 3 changed
    it.
    """
    freq = F32(freq)
    gcd = F32(gcd)
    k = jmode
    ireset = False
    inm = [0, -1, -1, -1]                  # INM(1..3): is layer ILY present?

    # --- 1. compact the mode list ------------------------------------------
    inum = 0
    for im in range(1, 7):
        if modes.hpmod[im, k] - _HMIN < 0.0:        # IF(...) 125,100,100
            continue
        inum += 1
        inm[int(modes.itmod[im, k])] = 1
        modes.delmod[inum, k] = modes.delmod[im, k]
        modes.hpmod[inum, k] = modes.hpmod[im, k]
        modes.htmod[inum, k] = modes.htmod[im, k]
        modes.fvmod[inum, k] = modes.fvmod[im, k]
        modes.afmod[inum, k] = modes.afmod[im, k]
        modes.itmod[inum, k] = modes.itmod[im, k]
        # NB: slots above INUM are deliberately left as they were
    isav = inum

    osigl = [F32(0.0)] * 4
    osigu = [F32(0.0)] * 4
    oyfot = [F32(0.0)] * 4
    oyhpf = [F32(0.0)] * 4
    oymuf = [F32(0.0)] * 4

    # very short distance (take-off angle >= 89.9) -> zero-distance mode
    zero_distance = muf.delmuf[muf.modmuf] >= F32(89.9)

    if not zero_distance:
        # --- 2. insert the circuit MUF mode for any missing layer ----------
        for ily in range(1, 4):
            if ily == 2 and state.fi[2, k] <= 0.0:      # no F1 layer
                continue
            if inum >= 5:                               # all layers are in
                break
            if inm[ily] < 0:
                if (freq + _EPS) >= muf.ymuf[ily] and ihop == muf.nhopmf[ily]:
                    inum += 1
                    modes.delmod[inum, k] = muf.delmuf[ily]
                    modes.hpmod[inum, k] = muf.hpmuf[ily]
                    modes.htmod[inum, k] = muf.htmuf[ily]
                    modes.fvmod[inum, k] = muf.fvmuf[ily]
                    modes.afmod[inum, k] = muf.afmuf[ily]
                    modes.itmod[inum, k] = ily
                    inm[ily] = 1
        if inum > isav:
            isav = inum

        if ihop == muf.nhopmf[muf.modmuf]:
            # a mode already covers this hop order -> straight to the budget;
            # none at all -> fall back to the zero-distance construction
            zero_distance = inum <= 0
        else:
            # --- 3. per-hop MUF, above the circuit MUF for this hop order --
            ireset = True
            for il in range(1, 4):
                osigl[il] = muf.sigl[il]
                osigu[il] = muf.sigu[il]
                oyfot[il] = muf.yfot[il]
                oyhpf[il] = muf.yhpf[il]
                oymuf[il] = muf.ymuf[il]
            for jh in range(1, 4):
                if inum >= 5:                           # all layers are in
                    break
                if inm[jh] > 0:                         # layer already has one
                    continue
                if jh == 2 and state.fi[2, k] <= 0.0:   # no F1 layer
                    continue
                if ihop <= muf.nhopmf[jh]:
                    continue
                fv = muf.fvmuf[jh]
                ht = muf.htmuf[jh]
                hpver = xlin(fv, state.fvert, state.hprim, 30, k)
                xhp = (hpver - ht) / RZ
                hpnow = hpver
                hop = F32(ihop)
                psi = F32(0.5) * gcd / hop
                tdel = (cos32(psi) - RZ / (RZ + hpnow)) / sin32(psi)
                cdel = F32(1.0) / sqrt32(F32(1.0) + tdel * tdel)
                sphe = RZ * cdel / (RZ + ht)
                secp = F32(1.0) / sqrt32(F32(1.0) - sphe * sphe)
                esd = fv * secp
                # correction to Martyn's theorem, as in CURMUF
                fob1 = esd
                xmut = sphe * sphe
                xfsq = fob1 * fob1 / (state.fi[jh, k] * state.fi[jh, k])
                sph = xfsq * xmut * xhp * (ht + F32(2.0) * (RZ + ht) * xhp)
                hpnow = hpver + sph
                tdel = (cos32(psi) - RZ / (RZ + hpnow)) / sin32(psi)
                cdel = F32(1.0) / sqrt32(F32(1.0) + tdel * tdel)
                sphe = RZ * cdel / (RZ + ht)
                secp = F32(1.0) / sqrt32(F32(1.0) - sphe * sphe)
                esd = fv * secp
                if freq - esd + _EPS < 0.0:     # IF(...) 240,160,160
                    continue
                # a new MUF has been found and can be used
                muf.ymuf[jh] = esd
                muf.sigl[jh] = esd * osigl[jh] / oymuf[jh]
                muf.yfot[jh] = muf.ymuf[jh] - F32(1.28) * muf.sigl[jh]
                muf.sigu[jh] = esd * osigu[jh] / oymuf[jh]
                muf.yhpf[jh] = muf.ymuf[jh] + F32(1.28) * muf.sigu[jh]
                inum += 1
                modes.delmod[inum, k] = atan32(tdel) * R2D
                modes.hpmod[inum, k] = hpnow
                modes.htmod[inum, k] = muf.htmuf[jh]
                modes.fvmod[inum, k] = muf.fvmuf[jh]
                modes.afmod[inum, k] = muf.afmuf[jh]
                modes.itmod[inum, k] = jh
                modes.ghop = F32(2.0) * psi     # mutates the hop for REGMOD

    # --- 4. zero-distance mode straight off the ionogram --------------------
    if zero_distance:
        inum += 1
        fv = freq - F32(0.001)
        # find FV in the ionogram
        ihx = 30
        hit = 0                             # 0 = above the table, 260, or 265
        for ih in range(2, 31):
            ihx = ih
            if fv - state.fvert[ih, k] < 0.0:       # IF(...) 260,265,255
                hit = 260
                break
            if fv - state.fvert[ih, k] == 0.0:
                hit = 265
                break
        if hit == 260:
            # interpolate between table values
            slope = state.fvert[ihx, k] - state.fvert[ihx - 1, k]
            if slope > F32(0.01):
                slp = (fv - state.fvert[ihx - 1, k]) / slope
                modes.delmod[inum, k] = F32(90.0)
                modes.hpmod[inum, k] = (
                    state.hprim[ihx - 1, k]
                    + (state.hprim[ihx, k] - state.hprim[ihx - 1, k]) * slp)
                modes.htmod[inum, k] = (
                    state.htrue[ihx - 1, k]
                    + (state.htrue[ihx, k] - state.htrue[ihx - 1, k]) * slp)
                modes.fvmod[inum, k] = fv
                modes.afmod[inum, k] = (
                    state.afac[ihx - 1, k]
                    + (state.afac[ihx, k] - state.afac[ihx - 1, k]) * slp)
        elif hit == 265:
            # exact hit in the FVERT table, so use its entries
            modes.delmod[inum, k] = F32(90.0)
            modes.hpmod[inum, k] = state.hprim[ihx, k]
            modes.htmod[inum, k] = state.htrue[ihx, k]
            modes.fvmod[inum, k] = state.fvert[ihx, k]
            modes.afmod[inum, k] = state.afac[ihx, k]
        if hit:                             # 270: assign the mode a layer
            for ily in range(1, 4):
                if ily == 2 and state.fi[2, k] <= 0.0:
                    continue
                if modes.htmod[inum, k] <= state.hi[ily, k]:
                    modes.itmod[inum, k] = ily
                    inm[ily] = 1
                    break

    # --- 275: count the modes, run the loss budget, put /MUFS/ back ---------
    nummod = 0
    for ilm in range(1, 7):
        if modes.hpmod[ilm, k] < 0.0:
            continue
        nummod += 1
    regmod(state, modes, muf, sigd, anois, rays, cant, zon, freq, jmode, gcd)
    if ireset:
        for il in range(1, 4):
            muf.sigl[il] = osigl[il]
            muf.sigu[il] = osigu[il]
            muf.yfot[il] = oyfot[il]
            muf.yhpf[il] = oyhpf[il]
            muf.ymuf[il] = oymuf[il]
    return nummod
