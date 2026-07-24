"""Stage B (PyTorch, batched) ray-set search — FINDF.

FINDF is the routine FDIST exists to consume: it walks the reflectrix table
IFOB layer by layer and angle by angle, and for every ray that reflects at the
operating frequency it records the take-off angle, the true and virtual
heights, the vertical frequency, the deviative loss and the layer.  Martyn's
theorem, corrected for a spherical ionosphere, turns each of those into a
ground distance, and the smallest and largest are the skip distance and the
maximum single-hop distance.

It is the largest lockstep port in the project and, before this rewrite, 78%
of the whole GPU pipeline's wall clock.  Two measurements drove the rewrite,
both taken on a 20,000-lane batch:

  * 44.8% of all live lane-steps sat at labels 305/306/315/335 -- the inner
    height search -- which cost FIVE steps per single increment of IH.
  * 73.7% of all live lane-steps sat at labels that change NO STATE AT ALL,
    only the program counter.

So the machine here is not the Fortran's twenty-two labels.  It is seven, and
the reduction is exact rather than approximate.  Both transformations below
preserve the answer bit for bit; the oracle test is unchanged and still
passes on the same cells.

THE SEARCH COLLAPSES TO ONE STEP, WITHOUT ASSUMING MONOTONICITY.  The inner
loop 305 -> 306 -> 315 -> 335 -> 330 -> 305 advances IH by one and re-tests,
and it stops at the first IH satisfying any of three conditions:

    IH >= IHIGH                                   -> leave the layer
    IFOB(IA,IH) == JFHZ                           -> take the ray as it stands
    IFOB(IA,IH) < JFHZ <= IFOB(IA,IH+1)           -> interpolate IH..IH+1

That is a FIRST-INDEX-SATISFYING-A-PREDICATE query over a thirty-entry
column, and such a query is computed exactly by masking the predicate, adding
the "at or after the cursor" constraint, and taking the minimum surviving
index.  No sortedness is required, which matters: IFOB is NOT monotone -- 46%
of its columns contain a decrease.  (Measured; the decreases sit at the layer
cusps, 43.9% of them at the E->F boundary IH 10->11, which is exactly where
FINDF resets ILOW/IHIGH and never scans across.)  A ``searchsorted`` would
have been wrong here, and would have been wrong only sometimes, which is
worse.

The one test that looks like it belongs in the loop and does not is label
305's own: ``IFOB(IA,ILOW) - JFHZ``.  ILOW and IA are both fixed for the
duration of the scan, so it is loop-INVARIANT -- it decides once, before any
scanning, whether the operating frequency is below the layer's floor.

DISPATCH-ONLY LABELS ARE RESOLVED TO CLOSURE, not executed.  Labels 265, 275,
285, 300, 350, 355 and 396 compute nothing; each is a pure function of state
the step has already loaded.  So instead of spending a lockstep step landing
on one, the transition is composed through them analytically -- ``_r396``
below is literally 396 -> 275 -> 285 -> 295 written as nested ``torch.where``.
The terminals are exactly the labels that DO something.

Together these take the machine from 586 lockstep steps to 88 on the same
20,000-lane batch, and the output is IDENTICAL -- 15262/15467 cells exact
against the oracle on CPU and 14658/15467 on CUDA, the same counts as before
the rewrite, not merely within the same bars.

Measured on an RTX 5080, against 2.46 us/lane before:

        lanes    ms    us/lane   peak GiB
       20,000    81     4.050       0.18
      100,000    92     0.922       0.89
      200,000   124     0.622       1.79
      400,000   203     0.508       3.57

CHUNK AT A FEW HUNDRED THOUSAND LANES.  Past that the working set stops
fitting: IFOB alone is 4,800 bytes per lane, so a million lanes is 4.8 GB of
input before any intermediate, and the measured cost per lane goes back UP --
6.8 us at a million, worse than the version this replaced.  That is not a
tuning constant to be raised later, it is the shape of the problem, and a
driver should chunk rather than batch harder.

WRITES AND THE MARTYN CORRECTION NOW HAPPEN IN ONE STEP.  Label 375 used to be
a separate visit that re-read DELFX/HTFLX from the row the write had just
filled; folding it in lets it use the values directly, and removes the FV/HP
carry buffers entirely -- they existed only to ferry two scalars across a step
boundary that no longer exists.

WRITES GO THROUGH ROW 0 AS A SINK, as in FDIST: a lane not writing this step
scatters to row 0, which nothing reads (rays are 1..45), so the store stays
unconditional instead of masked.  The sink is cleared before returning, or the
result would depend on which lanes shared the batch.

CURSOR UPDATES ARE ALL COMPUTED FROM THE STEP'S ENTRY STATE and applied at the
end.  Label 255 sets ILOW from IHIGH and then sets IHIGH; the write path
stores IAFTXR from IAF and then increments it.  Applying updates where they
are written would feed one label's output into another's input.

Precision follows Stage A: float32, with COS/ASIN (Martyn) and PENANG's ACOS
carrying the [[f32-libm-noise]] floor.  IFOB is the input that matters most --
it is quantised to integer kHz, so a 1-ULP COS difference upstream in FOBBY
either vanishes or moves an entry a whole kHz, and near a cusp a small SLOPD
turns that into a visible ground-distance shift.  That sensitivity belongs to
the reflectrix design, not to this port.
"""

import torch

from voacap_gpu.scalar.constants import D2R, PIO2, RZ
from voacap_gpu.batched.lockstep import gather_rows as _g
from voacap_gpu.batched.lockstep import run_lockstep
from voacap_gpu.batched.penang import penang_batch
from voacap_gpu.batched.profile import _k

_F32 = torch.float32

NRAY = 45                       # /REFLX/ ray slots, 1-based
NFV = 30                        # ionogram sounding frequencies
_DONE = 400                     # the terminal label

# Measured on the 20,000-lane benchmark batch: 586 steps before the rewrite,
# 88 after, with the mean lane needing 52.  512 leaves room for a batch whose
# slowest lane is far worse than anything seen, and raises rather than
# truncating if it is ever reached -- the cursors only ever advance, so
# hitting it means a wrong transition, not a batch that needed more room.
_MAX_STEPS = 512
_CHECK_EVERY = 8

# The labels that actually do something.  Everything else in the Fortran is
# dispatch and is resolved analytically by the ``_r*`` helpers below.
_LABELS = (225, 255, 295, 305, 345, 360)
_WRITE = (305, 345, 360)


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

    # ---- constants, hoisted: `_k` builds its scalar on the host, and a
    # host-to-device copy is not capturable into a CUDA graph.
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
    # applied to build IFOB; they have to agree or the equality test never
    # fires.
    jfhz = torch.trunc(k_khz * freq).to(torch.int32)
    fc2 = fi3 * fi3

    # The thirty Fortran height indices, as a ROW.  The search below works in
    # (lane, height) orientation rather than the (height, lane) the rest of
    # the module uses, because the whole column has to be read at once and
    # this way the per-lane block gather needs no materialised index tensor:
    # ``ifob[ia, :, lane]`` with two (N,) index vectors is (N, 30) directly.
    # Building a (30, N) int64 index instead costs 240 MB at a million lanes,
    # which measured as a 4x SLOWDOWN there -- the arithmetic was never the
    # problem, the index was.
    jrow = torch.arange(1, NFV + 1, device=dev, dtype=torch.int32)
    big_j = torch.tensor(NFV + 1, dtype=torch.int32, device=dev)

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
    pc = torch.full((n,), 225, dtype=torch.long, device=dev)
    ia = zero_i.clone()
    iaf = torch.ones(n, dtype=torch.long, device=dev)
    icusp = torch.full((n,), -1, dtype=torch.long, device=dev)
    il = torch.ones(n, dtype=torch.long, device=dev)
    ih = torch.ones(n, dtype=torch.long, device=dev)
    ilow = torch.ones(n, dtype=torch.long, device=dev)
    ihigh = torch.full((n,), 10, dtype=torch.long, device=dev)

    # The Fortran enters at 275 with IL=1; that is a dispatch label, so the
    # entry point is resolved here once instead of costing a step.
    def _L(v):
        return torch.full((n,), v, dtype=torch.long, device=dev)

    def _r265(il_):
        """GO TO (225,255,400), IL."""
        return torch.where(il_ == 1, _L(225),
                           torch.where(il_ == 2, _L(255), _L(_DONE)))

    def _r285(dp):
        return torch.where(dp - k_8999 <= 0.0, _L(295), _L(_DONE))

    def _r275(dp, il_):
        return torch.where(dp <= 0.0, _r265(il_), _r285(dp))

    def _r355(dp):
        return torch.where(dp - k_899 < 0.0, _L(360), _L(_DONE))

    def _r350(il_, dp):
        return torch.where(il_ - 3 < 0, _r355(dp), _L(_DONE))

    def _r396(ic, dp, il_):
        """IF(ICUSP) 275,350,265 -- resolved through everything past it."""
        return torch.where(ic < 0, _r275(dp, il_),
                           torch.where(ic == 0, _r350(il_, dp), _r265(il_)))

    def _r300(dp, ang_ia):
        return torch.where(dp - ang_ia <= 0.0, _L(345), _L(305))

    pc = _r275(_g(delpen, il.clamp(1, 3)), il)

    def _step():
        """Advance every lane by one FUSED statement group, IN PLACE."""
        at = {p: pc == p for p in _LABELS}

        # ---- gathers from the entry-state cursors -------------------------
        iac = ia.clamp(0, n_ang)
        ilc = il.clamp(1, 3)
        iafc = iaf.clamp(0, NRAY)
        iafm = (iaf - 1).clamp(0, NRAY)
        ihic = ihigh.clamp(1, NFV)
        ihi1 = (ihigh + 1).clamp(1, NFV)

        dpen = _g(delpen, ilc)
        # label 295 advances IA and label 300 reads ANG at the NEW index, so
        # the gather is at IA+1; the un-advanced value is never needed.
        ia_nxt = (ia + 1).clamp(0, n_ang)
        ang_nxt = _g(ang_p, ia_nxt)

        # ================= the height search, in one step ==================
        # IFOB(IA, 1..30) for every lane at once, as (N, 30).  Reading the
        # whole column is what lets the first-stop index be computed rather
        # than walked.
        iaz = (ia - 1).clamp(0, n_ang - 1)
        base = iaz * NFV
        blk = ifob[iaz, :, lanes]                            # (N, 30)

        jz = jfhz.unsqueeze(1)
        jj = jrow.unsqueeze(0)                               # (1, 30)
        s_exit = jj >= ihigh.unsqueeze(1)                    # label 306
        s_exact = ~s_exit & (blk == jz)                      # label 315, d==0
        # IFOB(IA, IH+1) >= JFHZ, as a SHIFTED BOOLEAN rather than a shifted
        # copy of the table: a bool block is a quarter the bytes of the int32
        # one, and the duplicated last column is only ever read at IH = 30,
        # where S_EXIT has already stopped the scan.
        ge = blk >= jz
        ge1 = torch.cat([ge[:, 1:], ge[:, -1:]], dim=1)
        s_interp = ~s_exit & (blk < jz) & ge1                # label 335
        # A lane whose IFOB(IH) EXCEEDS JFHZ keeps scanning -- that is the
        # Fortran's d>0 arm at 315, and it is why this is not a bisection.
        cand = (s_exit | s_exact | s_interp) & (jj >= ih.unsqueeze(1))
        # First true column = the smallest surviving index.  JROW is INT32 on
        # purpose: an int64 grid here is 240 MB at a million lanes and buys
        # nothing, since the values are at most 31.  A cumprod-of-negations
        # says the same thing in uint8 and measured SLOWER -- a scan against a
        # single reduction -- so the cheap dtype wins, not the cheap shape.
        jstar = torch.where(cand, jj.expand_as(cand), big_j).min(dim=1).values
        jstar = jstar.to(torch.long)

        # 305's own test is loop-INVARIANT (ILOW and IA are fixed for the
        # scan), so it decides once, before any scanning.
        fob_lo = _g(ifob_flat, base + (ilow.clamp(1, NFV) - 1))
        inv = fob_lo - jfhz >= 0                             # -> take IH as is

        js = jstar.clamp(1, NFV)
        fob_star = _g(ifob_flat, base + (js - 1))
        c_exit = ~inv & (jstar >= ihigh)
        c_exact = inv | (~inv & ~c_exit & (fob_star == jfhz))
        c_interp = ~inv & ~c_exit & ~c_exact
        # the index the write reads: the untouched cursor if 305 short-
        # circuited, otherwise wherever the scan stopped
        kx = torch.where(inv, ih.clamp(1, NFV), js)
        kx1 = (kx + 1).clamp(1, NFV)

        # ---- the write values --------------------------------------------
        w305, w345, w360 = at[305], at[345], at[360]
        w305_hit = w305 & ~c_exit
        write = w305_hit | w345 | w360

        slopd = torch.clamp(
            (_g(ifob_flat, base + (kx1 - 1)) - fob_star).to(_F32), min=1.0)
        slope = (jfhz - _g(ifob_flat, base + (kx - 1))).to(_F32) / slopd

        def _at305(tab):
            """Label 325 takes the entry as it stands, 340 interpolates."""
            a, b = _g(tab, kx), _g(tab, kx1)
            return torch.where(c_exact, a, a + slope * (b - a))

        def _pick(tab):
            return torch.where(w345, _g(tab, ihic),
                               torch.where(w360, _g(tab, ihi1), _at305(tab)))

        ht_w = _pick(htrue_p)
        af_w = _pick(afac_p)
        fv_w = _pick(fvert_p)
        hp_w = _pick(hprim_p)
        # 325 and 340 both take the scan angle; 345 the cusp angle; 360 nudges
        # past the previous ray so the two sides of the cusp stay ordered.
        ang_ia = _g(ang_p, iac)
        del_w = torch.where(w345, dpen,
                            torch.where(w360, _g(delfx, iafm) + k_eps,
                                        ang_ia))
        it_w = torch.where(w360, torch.where(has_f1, il + 1, _L(3)), il)

        # ---- Martyn, corrected for a spherical ionosphere (label 375) -----
        # Folded in: it used to re-read DELFX/HTFLX from the row just written,
        # which is these values.
        del_ = del_w * k_d2r
        rcosd = k_rz * torch.cos(del_)
        xfsq = freq * freq / fc2
        xmut = k_one - fv_w * fv_w / (freq * freq)
        xhp = (hp_w - ht_w) / k_rz
        sph = xfsq * xmut * xhp * (ht_w + k_two * (k_rz + ht_w) * xhp)
        hp_new = hp_w + sph
        phe = torch.asin(rcosd / (k_rz + hp_new))
        gdr = k_two * k_rz * (k_pio2 - del_ - phe)

        row = torch.where(write, iafc, zero_i)          # row 0 is the sink
        delfx[row, lanes] = del_w
        htflx[row, lanes] = ht_w
        afflx[row, lanes] = af_w
        imode[row, lanes] = it_w.to(torch.int32)
        gdflx[row, lanes] = gdr
        hpflx[row, lanes] = hp_new
        fvflx[row, lanes] = fv_w

        closer = write & (dskpkm - gdr > 0.0)
        dskpkm.copy_(torch.where(closer, gdr, dskpkm))
        delskp.copy_(torch.where(closer, del_w, delskp))
        htskp.copy_(torch.where(closer, ht_w, htskp))
        hpskp.copy_(torch.where(closer, hp_new, hpskp))
        fvskp.copy_(torch.where(closer, fv_w, fvskp))
        iskp.copy_(torch.where(closer, il.to(torch.int32), iskp))
        # the maximum only counts rays at or above the minimum take-off angle
        farther = write & (dmaxkm - gdr <= 0.0) & (del_w - amind >= 0.0)
        dmaxkm.copy_(torch.where(farther, gdr, dmaxkm))
        iaftxr.copy_(torch.where(write, iaf.to(torch.int32), iaftxr))

        # ---- cursor updates, all from the entry state --------------------
        # 225 is fused with 235/245: they set disjoint cursors and 225's only
        # exit is into one of them.
        n_ih = torch.where(at[225], torch.full_like(ih, 11), ih)
        n_ih = torch.where(at[255], torch.full_like(ih, 21), n_ih)
        n_ih = torch.where(w305, kx, n_ih)
        n_ilow = torch.where(at[225], torch.full_like(ilow, 11), ilow)
        n_ilow = torch.where(at[255], ihigh + 1, n_ilow)
        n_il = torch.where(at[225], torch.where(has_f1, torch.full_like(il, 2),
                                                torch.full_like(il, 3)), il)
        n_il = torch.where(at[255], torch.full_like(il, 3), n_il)
        n_ihigh = torch.where(
            at[225], torch.where(has_f1, torch.full_like(ihigh, 20),
                                 torch.full_like(ihigh, 30)), ihigh)
        n_ihigh = torch.where(at[255], torch.full_like(ihigh, 30), n_ihigh)
        n_icusp = torch.where(at[225] | at[255],
                              torch.full_like(icusp, -1), icusp)
        n_icusp = torch.where(w345, torch.zeros_like(icusp), n_icusp)
        n_icusp = torch.where(w360, torch.ones_like(icusp), n_icusp)
        n_ia = torch.where(at[295], ia + 1, ia)
        n_ia = torch.where(w345, ia - 1, n_ia)
        n_iaf = torch.where(write, iaf + 1, iaf)

        # ---- transitions, with every dispatch label resolved away ---------
        nxt = pc.clone()
        # a layer setup lands on 275, which resolves against the NEW IL/DELPEN
        dpen_new = _g(delpen, n_il.clamp(1, 3))
        setup = at[225] | at[255]
        nxt = torch.where(setup, _r275(dpen_new, n_il), nxt)
        # 295 -> 300 reads ANG at the advanced IA
        nxt = torch.where(at[295],
                          torch.where(n_ia - nang_l <= 0,
                                      _r300(dpen, ang_nxt), _L(_DONE)), nxt)
        # a completed write runs out through 375 -> 396 and everything past it
        nxt = torch.where(write,
                          torch.where(n_iaf - NRAY <= 0,
                                      _r396(n_icusp, dpen, il), _L(_DONE)),
                          nxt)
        # the search leaving its layer lands on 275 with IL unchanged
        nxt = torch.where(w305 & c_exit, _r275(dpen, il), nxt)

        for buf, val in ((ih, n_ih), (ilow, n_ilow), (ihigh, n_ihigh),
                         (il, n_il), (icusp, n_icusp), (ia, n_ia),
                         (iaf, n_iaf), (pc, nxt)):
            buf.copy_(val)

    run_lockstep(_step, pc,
                 (pc, ia, iaf, icusp, il, ih, ilow, ihigh,
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
