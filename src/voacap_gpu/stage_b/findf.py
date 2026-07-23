"""Stage B (PyTorch, batched) ray-set search — FINDF.

FINDF is the routine FDIST exists to consume: it walks the reflectrix table
IFOB layer by layer and angle by angle, and for every ray that reflects at the
operating frequency it records the take-off angle, the true and virtual
heights, the vertical frequency, the deviative loss and the layer.  Martyn's
theorem, corrected for a spherical ionosphere, turns each of those into a
ground distance, and the smallest and largest are the skip distance and the
maximum single-hop distance.

It is the SECOND lockstep port and by some distance the largest -- twenty-two
Fortran labels against FDIST's thirteen, seven cursors against three, and four
separate write sites instead of two.  ``lockstep.run_lockstep`` is reused
unchanged, which is the point of having factored it out: the pattern is what
generalises, not the routine.

THE STEP COUNT IS WHAT THIS COSTS.  Lockstep runs until the SLOWEST lane
finishes, and the bound here is a scan over up to forty angles nested inside
three layers with the height cursor threaded through -- roughly 1,200
statements against FDIST's ~250.  The bound is structural rather than hopeful
(IA and IH only ever increase within a layer, IL only ever advances, IAF caps
at 45) and ``_MAX_STEPS`` raises rather than truncating if it is reached.

Measured against the 222 us/call scalar Stage A, on the 88-lane oracle batch
tiled up:

           lanes    eager    graph   gain   graph vs scalar
              88   2117 ms   239 ms   8.9x   12x SLOWER
             880   2149 ms   283 ms   7.6x   1.4x slower
           3,520   1879 ms   294 ms   6.4x   2.7x faster
          35,200   1945 ms   296 ms   6.6x    26x faster
         176,000        --   483 ms     --    81x faster

The shape is FDIST's: eager is flat in the lane count because it is bound by
kernel LAUNCHES, and the graph path stays flat until ~35,000 lanes and only
then starts to grow, which is the compute-bound regime it should be in.  The
crossover against scalar sits around 1,100 lanes -- higher than FDIST's ~2,000
in absolute time but the same story, because the scalar routine it is racing
is also five times heavier.  Below that, use Stage A.

CURSOR UPDATES ARE ALL COMPUTED FROM THE STEP'S ENTRY STATE and applied at the
end.  That is not tidiness: label 255 sets ILOW from IHIGH and then sets IHIGH,
and label 375 stores IAFTXR from IAF and then increments it, so applying
updates as they are written would feed one label's output into another's
input.  Every ``n_*`` below is a pure expression of the buffers as they were
when the step began.

WRITES GO THROUGH ROW 0 AS A SINK, exactly as in FDIST: labels 325, 340, 345
and 360 each write a ray at the lane's own IAF cursor, and lanes doing
anything else scatter to row 0, which nothing reads (rays are 1..45).  That
keeps the store unconditional instead of masked.

Precision follows Stage A: float32, with COS/ASIN (Martyn) and PENANG's ACOS
carrying the [[f32-libm-noise]] floor.  IFOB is the input that matters most --
it is quantised to integer kHz, so a 1-ULP COS difference upstream in FOBBY
either vanishes or moves an entry a whole kHz, and near a cusp a small SLOPD
turns that into a visible ground-distance shift.  That sensitivity belongs to
the reflectrix design, not to this port; Stage A's tier-B isolates it the same
way, by feeding in the oracle's own IFOB.
"""

import numpy as np
import torch

from voacap_gpu.stage_a.constants import D2R, PIO2, RZ
from voacap_gpu.stage_b.lockstep import gather_rows as _g
from voacap_gpu.stage_b.lockstep import run_lockstep
from voacap_gpu.stage_b.penang import penang_batch
from voacap_gpu.stage_b.profile import _k

_F32 = torch.float32

NRAY = 45                       # /REFLX/ ray slots, 1-based
NFV = 30                        # ionogram sounding frequencies
_DONE = 400                     # the terminal label
_MAX_STEPS = 4096               # see the module docstring
_CHECK_EVERY = 32

_LABELS = (225, 235, 245, 255, 265, 275, 285, 295, 300, 305, 306, 315, 325,
           330, 335, 340, 345, 350, 355, 360, 375, 396)


def findf_batch(htrue, hprim, fvert, afac, fi2, fi3, ang, nang, ifob, freq,
                amind, use_cuda_graph=True):
    """Batched FINDF, PENANG included (the Fortran calls it on entry).

    htrue/hprim/fvert/afac: (30, N) ionogram + deviative loss, Fortran index
        IH = 1..30 as rows 0..29.
    fi2/fi3: (N,) F1 and F2 critical frequencies; fi2 <= 0 means no F1 ledge.
    ang: (A, N) the take-off angle scan SANG produced, IA = 1..A as rows 0..A-1.
    nang: (N,) int, how much of that scan is live per lane.
    ifob: (A, 30, N) int, the reflectrix from ``fobby_batch``.
    freq/amind: (N,) operating frequency (MHz) and minimum take-off angle.

    Returns ``(rays, scalars)``.  ``rays`` maps each /REFLX/ name to a (46, N)
    array 1-based in the ray index -- delfx, gdflx, hpflx, htflx, fvflx, afflx
    and the integer imode.  ``scalars`` maps iaftxr, dskpkm, delskp, hpskp,
    htskp, fvskp, iskp and dmaxkm to (N,).

    Slots above IAFTXR are left at their initial value, matching the Fortran,
    which leaves them stale rather than clearing them.
    """
    n = freq.shape[0]
    dev = freq.device
    n_ang = ang.shape[0]
    if ifob.shape[:2] != (n_ang, NFV):
        raise ValueError(
            f"ifob is {tuple(ifob.shape)}; expected ({n_ang}, {NFV}, {n}) to "
            "match the angle scan and the thirty sounding frequencies")

    lanes = torch.arange(n, device=dev)
    zero_i = torch.zeros(n, dtype=torch.long, device=dev)

    # ---- constants, hoisted: `_k` builds on the host and a host-to-device
    # copy is not capturable into a CUDA graph.  All loop-invariant anyway.
    k_rz, k_d2r, k_pio2 = _k(RZ, freq), _k(D2R, freq), _k(PIO2, freq)
    k_one, k_two = _k(1.0, freq), _k(2.0, freq)
    k_8999, k_899 = _k(89.99, freq), _k(89.9, freq)
    k_eps, k_khz = _k(0.001, freq), _k(1000.0, freq)

    # 1-based padding so a cursor of 1 means Fortran's index 1.
    pad = torch.zeros((1, n), dtype=_F32, device=dev)
    htrue_p = torch.cat([pad, htrue])                   # (31, N)
    hprim_p = torch.cat([pad, hprim])
    fvert_p = torch.cat([pad, fvert])
    afac_p = torch.cat([pad, afac])
    ang_p = torch.cat([pad, ang])                       # (A+1, N)
    # IFOB stays unpadded -- at (A, 30, N) it is the biggest thing here -- and
    # is addressed by a flattened (IA-1)*30 + (IH-1) instead.
    ifob_flat = ifob.reshape(n_ang * NFV, n).to(torch.int32)

    # JFHZ = 1000.*FREQ truncated to integer, the same quantisation FOBBY
    # applied to build IFOB; they have to agree or the equality test at 315
    # never fires.
    jfhz = torch.trunc(k_khz * freq).to(torch.int32)
    fc2 = fi3 * fi3

    delfx = torch.zeros((NRAY + 1, n), dtype=_F32, device=dev)
    gdflx = torch.zeros((NRAY + 1, n), dtype=_F32, device=dev)
    hpflx = torch.zeros((NRAY + 1, n), dtype=_F32, device=dev)
    htflx = torch.zeros((NRAY + 1, n), dtype=_F32, device=dev)
    fvflx = torch.zeros((NRAY + 1, n), dtype=_F32, device=dev)
    afflx = torch.zeros((NRAY + 1, n), dtype=_F32, device=dev)
    imode = torch.zeros((NRAY + 1, n), dtype=torch.int32, device=dev)

    dmaxkm = torch.zeros(n, dtype=_F32, device=dev)
    dskpkm = torch.full((n,), 10000.0, dtype=_F32, device=dev)
    delskp = torch.zeros(n, dtype=_F32, device=dev)
    hpskp = torch.zeros(n, dtype=_F32, device=dev)
    htskp = torch.zeros(n, dtype=_F32, device=dev)
    fvskp = torch.zeros(n, dtype=_F32, device=dev)
    iskp = torch.zeros(n, dtype=torch.int32, device=dev)
    iaftxr = torch.zeros(n, dtype=torch.int32, device=dev)

    # PENANG runs before the search and can veto the whole area by flagging
    # DSKPKM/DMAXKM negative when even the F2 nose is penetrated.
    delpen, nose_penetrates = penang_batch(htrue, fvert, fi2, freq)
    dskpkm = torch.where(nose_penetrates, -k_one, dskpkm)
    dmaxkm = torch.where(nose_penetrates, -k_one, dmaxkm)

    nang_l = nang.to(torch.long)
    has_f1 = fi2 > 0.0

    # ---- cursors ---------------------------------------------------------
    pc = torch.full((n,), 275, dtype=torch.long, device=dev)
    ia = zero_i.clone()
    iaf = torch.ones(n, dtype=torch.long, device=dev)
    icusp = torch.full((n,), -1, dtype=torch.long, device=dev)
    il = torch.ones(n, dtype=torch.long, device=dev)
    ih = torch.ones(n, dtype=torch.long, device=dev)
    ilow = torch.ones(n, dtype=torch.long, device=dev)
    ihigh = torch.full((n,), 10, dtype=torch.long, device=dev)
    fv = torch.zeros(n, dtype=_F32, device=dev)
    hp = torch.zeros(n, dtype=_F32, device=dev)

    def _step():
        """Advance every lane by one Fortran statement, IN PLACE."""
        at = {p: pc == p for p in _LABELS}

        def L(v):
            return torch.full_like(pc, v)

        # ---- gathers, all from the entry-state cursors -------------------
        iac = ia.clamp(0, n_ang)
        ihc = ih.clamp(1, NFV)
        ih1 = (ih + 1).clamp(1, NFV)
        ilowc = ilow.clamp(1, NFV)
        ihic = ihigh.clamp(1, NFV)
        ihi1 = (ihigh + 1).clamp(1, NFV)
        iafc = iaf.clamp(0, NRAY)
        iafm = (iaf - 1).clamp(0, NRAY)
        ilc = il.clamp(1, 3)

        ang_ia = _g(ang_p, iac)
        dpen = _g(delpen, ilc)

        iaz = (ia - 1).clamp(0, n_ang - 1) * NFV
        fob_lo = _g(ifob_flat, iaz + (ilowc - 1))
        fob_ih = _g(ifob_flat, iaz + (ihc - 1))
        fob_ih1 = _g(ifob_flat, iaz + (ih1 - 1))

        # ---- label 340: interpolate between IH and IH+1 ------------------
        slopd = torch.clamp((fob_ih1 - fob_ih).to(_F32), min=1.0)
        slope = (jfhz - fob_ih).to(_F32) / slopd

        def _interp(tab):
            a, b = _g(tab, ihc), _g(tab, ih1)
            return a + slope * (b - a)

        # ---- the four write sites ----------------------------------------
        w325, w340, w345, w360 = at[325], at[340], at[345], at[360]
        write = w325 | w340 | w345 | w360

        def _pick(tab, interp):
            """325 takes IH as it stands, 340 interpolates, 345 takes the
            layer's top index and 360 the first index of the next layer."""
            return torch.where(
                w345, _g(tab, ihic),
                torch.where(w360, _g(tab, ihi1),
                            torch.where(w340, interp, _g(tab, ihc))))

        ht_w = _pick(htrue_p, _interp(htrue_p))
        af_w = _pick(afac_p, _interp(afac_p))
        fv_w = _pick(fvert_p, _interp(fvert_p))
        hp_w = _pick(hprim_p, _interp(hprim_p))
        # 325 and 340 both take the scan angle; 345 the cusp angle; 360 nudges
        # past the previous ray so the two sides of the cusp stay ordered.
        del_w = torch.where(w345, dpen,
                            torch.where(w360, _g(delfx, iafm) + k_eps, ang_ia))
        it_w = torch.where(w360, torch.where(has_f1, il + 1, L(3)), il)

        row = torch.where(write, iafc, zero_i)          # row 0 is the sink
        delfx[row, lanes] = del_w
        htflx[row, lanes] = ht_w
        afflx[row, lanes] = af_w
        imode[row, lanes] = it_w.to(torch.int32)

        # ---- label 375: Martyn, corrected for a spherical ionosphere -----
        m = at[375]
        del_iaf = _g(delfx, iafc)
        ht = _g(htflx, iafc)
        del_ = del_iaf * k_d2r
        rcosd = k_rz * torch.cos(del_)
        xfsq = freq * freq / fc2
        xmut = k_one - fv * fv / (freq * freq)
        xhp = (hp - ht) / k_rz
        sph = xfsq * xmut * xhp * (ht + k_two * (k_rz + ht) * xhp)
        hp_new = hp + sph
        phe = torch.asin(rcosd / (k_rz + hp_new))
        gdr = k_two * k_rz * (k_pio2 - del_ - phe)

        mrow = torch.where(m, iafc, zero_i)
        gdflx[mrow, lanes] = gdr
        hpflx[mrow, lanes] = hp_new
        fvflx[mrow, lanes] = fv

        closer = m & (dskpkm - gdr > 0.0)
        dskpkm.copy_(torch.where(closer, gdr, dskpkm))
        delskp.copy_(torch.where(closer, del_iaf, delskp))
        htskp.copy_(torch.where(closer, ht, htskp))
        hpskp.copy_(torch.where(closer, hp_new, hpskp))
        fvskp.copy_(torch.where(closer, fv, fvskp))
        iskp.copy_(torch.where(closer, il.to(torch.int32), iskp))
        # the maximum only counts rays at or above the minimum take-off angle
        farther = m & (dmaxkm - gdr <= 0.0) & (del_iaf - amind >= 0.0)
        dmaxkm.copy_(torch.where(farther, gdr, dmaxkm))
        iaftxr.copy_(torch.where(m, iaf.to(torch.int32), iaftxr))

        # ---- cursor updates, all from the entry state --------------------
        n_ih = torch.where(at[225], torch.full_like(ih, 11), ih)
        n_ih = torch.where(at[255], torch.full_like(ih, 21), n_ih)
        n_ih = torch.where(at[330], ih + 1, n_ih)
        n_ilow = torch.where(at[225], torch.full_like(ilow, 11), ilow)
        n_ilow = torch.where(at[255], ihigh + 1, n_ilow)
        n_ihigh = torch.where(at[235] | at[255], torch.full_like(ihigh, 30),
                              ihigh)
        n_ihigh = torch.where(at[245], torch.full_like(ihigh, 20), n_ihigh)
        n_il = torch.where(at[235] | at[255], torch.full_like(il, 3), il)
        n_il = torch.where(at[245], torch.full_like(il, 2), n_il)
        n_icusp = torch.where(at[235] | at[245] | at[255],
                              torch.full_like(icusp, -1), icusp)
        n_icusp = torch.where(w345, torch.zeros_like(icusp), n_icusp)
        n_icusp = torch.where(w360, torch.ones_like(icusp), n_icusp)
        n_ia = torch.where(at[295], ia + 1, ia)
        n_ia = torch.where(w345, ia - 1, n_ia)
        n_iaf = torch.where(m, iaf + 1, iaf)
        n_fv = torch.where(write, fv_w, fv)
        n_hp = torch.where(write, hp_w, torch.where(m, hp_new, hp))

        # ---- transitions --------------------------------------------------
        nxt = pc.clone()
        nxt = torch.where(at[225], torch.where(has_f1, L(245), L(235)), nxt)
        nxt = torch.where(at[235] | at[245] | at[255], L(275), nxt)
        nxt = torch.where(at[265], torch.where(
            il == 1, L(225), torch.where(il == 2, L(255), L(400))), nxt)
        nxt = torch.where(at[275], torch.where(dpen <= 0.0, L(265), L(285)),
                          nxt)
        nxt = torch.where(at[285], torch.where(dpen - k_8999 <= 0.0, L(295),
                                               L(400)), nxt)
        nxt = torch.where(at[295], torch.where(n_ia - nang_l <= 0, L(300),
                                               L(400)), nxt)
        nxt = torch.where(at[300], torch.where(dpen - ang_ia <= 0.0, L(345),
                                               L(305)), nxt)
        nxt = torch.where(at[305], torch.where(fob_lo - jfhz < 0, L(306),
                                               L(325)), nxt)
        nxt = torch.where(at[306], torch.where(ih - ihigh < 0, L(315), L(275)),
                          nxt)
        d315 = fob_ih - jfhz
        nxt = torch.where(at[315], torch.where(
            d315 < 0, L(335), torch.where(d315 == 0, L(325), L(330))), nxt)
        nxt = torch.where(at[330], L(305), nxt)
        nxt = torch.where(at[335], torch.where(fob_ih1 - jfhz < 0, L(330),
                                               L(340)), nxt)
        nxt = torch.where(write, L(375), nxt)
        nxt = torch.where(at[350], torch.where(il - 3 < 0, L(355), L(400)),
                          nxt)
        nxt = torch.where(at[355], torch.where(dpen - k_899 < 0.0, L(360),
                                               L(400)), nxt)
        nxt = torch.where(m, torch.where(n_iaf - NRAY <= 0, L(396), L(400)),
                          nxt)
        nxt = torch.where(at[396], torch.where(
            icusp < 0, L(275), torch.where(icusp == 0, L(350), L(265))), nxt)

        for buf, val in ((ih, n_ih), (ilow, n_ilow), (ihigh, n_ihigh),
                         (il, n_il), (icusp, n_icusp), (ia, n_ia),
                         (iaf, n_iaf), (fv, n_fv), (hp, n_hp), (pc, nxt)):
            buf.copy_(val)

    run_lockstep(_step, pc,
                 (pc, ia, iaf, icusp, il, ih, ilow, ihigh, fv, hp,
                  delfx, gdflx, hpflx, htflx, fvflx, afflx, imode,
                  dmaxkm, dskpkm, delskp, hpskp, htskp, fvskp, iskp, iaftxr),
                 _DONE, _MAX_STEPS, "FINDF", use_cuda_graph,
                 check_every=_CHECK_EVERY)

    # Clear the sink: how many times row 0 was scattered into depends on how
    # long the SLOWEST lane in the batch ran, so leaving it would make the
    # result depend on batch composition.  Nothing reads it -- rays are 1..45.
    for arr in (delfx, gdflx, hpflx, htflx, fvflx, afflx, imode):
        arr[0] = 0

    rays = {"delfx": delfx, "gdflx": gdflx, "hpflx": hpflx, "htflx": htflx,
            "fvflx": fvflx, "afflx": afflx, "imode": imode}
    scalars = {"iaftxr": iaftxr, "dskpkm": dskpkm, "delskp": delskp,
               "hpskp": hpskp, "htskp": htskp, "fvskp": fvskp, "iskp": iskp,
               "dmaxkm": dmaxkm}
    return rays, scalars
