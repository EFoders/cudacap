"""Stage B (PyTorch, batched) sporadic-E modes — ESMOD.

REGMOD does the F- and E-layer modes; ESMOD does the ones that reflect off the
sporadic-E layer itself, and it writes them into /ZON/ slots 4 and 5, on top
of whatever REGMOD left there.  The two routines share a shape -- free-space
loss, absorption, ground loss, antenna gains, an over-the-MUF term, then field
strength and SNR -- and almost none of the detail, because an Es reflection is
a thin mirror at a fixed height rather than a refraction through a profile.
There is no deviative loss (ADV is 0), the virtual height IS the true height,
and the "over-the-MUF" term becomes a reflection loss.

THE BATCHING IS EASY AND THE CURSOR IS TINY.  Everything is elementwise across
grid points once the sample area is picked, and the only irregular part is the
hop loop, which runs over at most two values.  So the two hops are UNROLLED as
a Python loop and the per-lane slot is a two-row scatter.

WHICH ROW EACH HOP WRITES IS THE DETAIL MOST LIKELY TO BE MIS-PORTED.  IH is
incremented once per DO-150 trip, not once per mode written, and the loop runs
``IHOP = IHSRT, 2``.  A path that needs two hops therefore runs ``IHOP = 2,2``
-- ONE iteration, writing slot **4** and leaving slot 5 at its presets.  Slot
4 is "the first mode found", not "the one-hop mode".  Batched, the row is
``ihop - ihsrt``, which is exactly that statement.  A lane cut by the minimum
take-off angle still CONSUMES its slot (the ``go to 150`` skips the writes,
not the increment), so a one-hop mode below AMIND leaves slot 4 preset and
puts the two-hop mode in slot 5.

WRITES GO THROUGH A SINK ROW, the FDIST pattern: the scatter buffer is (3, N)
and lanes that are not writing this trip scatter into row 2.  That row is
DROPPED rather than cleared on the way out, so batch-composition independence
is structural here rather than something the caller has to trust.

Four things below look like slips and are not:

* the sample area K is the one with the **lowest** FS(2,.), not the highest --
  "all modes will be at least this good": the weakest link bounds the mode.
* ``AC`` indexes GYZ by **K**, the Es sample area, where REGMOD indexes it by
  LX(K); this routine has no LX at all.
* ``ADX``, the correction for an Es layer below the E layer, is added into
  SABPS *and then again* inside XTLOS, so it counts twice in the loss but
  only once in the reported ABPS.
* slots 4 and 5 are preset to the no-mode markers first and unconditionally,
  so the four early returns still leave them readable.

ESREG IS NOT PORTED.  Its 363 lines contain an unconditional RETURN after
presetting slots 6-7 and the source itself says "ERRORS IN THE CODE BELOW /
Nothing DONe here!!!"; VOACAP has never computed a mixed Es-F mode.  Stage A
keeps ``esreg.py`` so anyone hunting that physics lands on the explanation.
Nothing here depends on it: the slots are disjoint.

float32.  ALOG10/ALOG/SIN/COS/ATAN/ASIN/SQRT and the ``**1.98`` / ``**-0.7``
powers all carry the [[f32-libm-noise]] floor, and torch's libm is not
gfortran's, so unlike the pure-bookkeeping routines this one cannot be
bit-exact on either device.  MEASURED against the 466-case oracle dump, 21,436
cells: 99.22% exact on CPU and 98.22% on CUDA, worst 2,648 floored ULP (CPU,
an RGAIN of 0.334 dB moved 4e-4 by a 9e-5-degree take-off angle) and 478
(CUDA, a TLOSS of 52,935 dB moved 1.87 by the ``FREQ - ESD`` cancellation
feeding the far tail of PRBMUF and then ``PROS**(-0.7)``).  ADV, CREL, EFF,
HN, HP, NMODE, PROB, RELY and SPRO are bit-exact on both -- none of them
touches the trig chain.  The test module carries the full breakdown.
"""

import numpy as np
import torch

from voacap_gpu.stage_a.constants import PIO2, R2D, RZ, VOFL
from voacap_gpu.stage_b.gain import (antenna_gain_batch, ground_loss_batch,
                                     pwrdb_batch)
from voacap_gpu.stage_b.lockstep import gather_rows as _g
from voacap_gpu.stage_b.normal import prbmuf_batch
from voacap_gpu.stage_b.profile import _k

_F32 = torch.float32

NAREA = 5                       # sample areas along the path
NES = 2                         # /ZON/ 4 and 5 -- only two Es hops exist
_SINK = NES                     # the scatter row nothing reads

# DO 97, the no-mode presets.  Note CREL is -1000 here where ESREG's slots
# get +1000; the two disagree and both are transliterated.
_PRESET = {"obf": 1000.0, "adv": 1000.0, "fslos": 1000.0, "tloss": 1000.0,
           "abps": 1000.0, "eff": 0.0, "grlos": 1000.0, "rgain": 0.0,
           "tgain": 0.0, "hn": -1.0, "prob": 0.001, "crel": -1000.0,
           "rely": 0.001, "spro": 0.001, "fldst": -1000.0, "sigpow": -1000.0,
           "sn": -1000.0, "hp": -1.0, "b": -1.0, "tllow": 10.0,
           "timed": -1.0, "tlhgh": 10.0}
_PRESET_NMODE = 5


def esmod_batch(km, fs, hs, fi1, gyz, sigpat, epspat,
                acav, afe, bfe, asm, dsl, dsu, rcnse,
                sigl4, sigu4, ymuf4, yfot4, yhpf4,
                ifob13, cant, freq, gcd, gcdkm, amind):
    """Batched ESMOD.  Returns the /ZON/ Es pair.

    km:      (N,) int, how many sample areas are live (/RON/ KM).
    fs:      (3, 5, N) /ES/ FS -- rows 0..2 are FOT / MUF / HPF of the Es
             layer, columns 0..4 the five sample areas.
    hs:      (5, N) /ES/ HS, the Es height (virtual = true).
    fi1:     (5, N) /RON/ FI(1,.), the E-layer critical frequency.
    gyz:     (5, N) /GEOG/ gyrofrequency.
    sigpat, epspat: (5, N) /GEOG/ ground constants, for GAIN's ITR<0 arm.
    acav, afe, bfe, asm, dsl, dsu: (N,) /SIGD/.
    rcnse:   (N,) /ANOIS/ receiver noise.
    sigl4, sigu4, ymuf4, yfot4, yhpf4: (N,) the /MUFS/ Es row (index 4).
    ifob13:  (N,) IFOB(1,3,JMODE), the reflectrix entry the low-frequency
             cutoff is read off.  JMODE has already selected the area, which
             is why the whole table is not needed here.
    cant:    ``gain.BatchedCantenna``.
    freq:    (N,) MHz.  gcd/gcdkm: (N,) great-circle distance, radians and km.
    amind:   (N,) minimum usable take-off angle, degrees.

    Returns a dict of (2, N) tensors keyed by the /ZON/ field name, ROWS 0 AND
    1 BEING FORTRAN SLOTS 4 AND 5 -- splice them in with ``zon[name][4:6] =
    out[name]``.  ``nmode`` is int32.
    """
    n = freq.shape[0]
    dev = freq.device
    lanes = torch.arange(n, device=dev)
    sink = torch.full((n,), _SINK, dtype=torch.long, device=dev)

    k_one, k_two, k_half = _k(1.0, freq), _k(2.0, freq), _k(0.5, freq)
    k_rz, k_pio2, k_r2d = _k(RZ, freq), _k(PIO2, freq), _k(R2D, freq)
    k_vofl = _k(VOFL, freq)
    k_zero = _k(0.0, freq)
    k_kilo, k_three = _k(1000.0, freq), _k(3.0, freq)
    k_p198, k_pm07 = _k(1.98, freq), _k(-0.7, freq)
    k_refc = _k(8.9136, freq)               # the reflection-loss coefficient
    k_p90, k_p9 = _k(0.90, freq), _k(0.9, freq)
    k_25 = _k(25.0, freq)                   # "maximum observed is 25."
    k_10p2, k_677 = _k(10.2, freq), _k(677.2, freq)
    k_32p45, k_20 = _k(32.45, freq), _k(20.0, freq)
    k_100, k_107p2 = _k(100.0, freq), _k(107.2, freq)

    km_l = km.to(torch.long)

    # ---- DO 95: the sample area with the LOWEST Es critical frequency.
    # "All modes will be at least this good" -- the weakest link along the
    # path bounds the whole mode, so this is a minimum and not a maximum.
    # The running FK is FS(2,K) by construction, so no re-gather is needed.
    kidx = torch.zeros(n, dtype=torch.long, device=dev)      # 0-based area
    fk = fs[1, 0]
    for i in range(1, NAREA):                # Fortran IS = 2..KM
        take = (i < km_l) & (fk - fs[1, i] > 0.0)
        kidx = torch.where(take, torch.full_like(kidx, i), kidx)
        fk = torch.where(take, fs[1, i], fk)

    fs1k, fs2k, fs3k = _g(fs[0], kidx), fk, _g(fs[2], kidx)
    hsk, fi1k, gyzk = _g(hs, kidx), _g(fi1, kidx), _g(gyz, kidx)

    # The maps are Es or E, so they are not usable at the low end.
    fsdead = torch.minimum(ifob13.to(_F32) / k_kilo, k_three)

    out = {name: torch.full((NES + 1, n), float(np.float32(v)),
                            dtype=_F32, device=dev)
           for name, v in _PRESET.items()}
    out["nmode"] = torch.full((NES + 1, n), _PRESET_NMODE,
                              dtype=torch.int32, device=dev)

    # The first two early returns.  Everything below still evaluates on the
    # dead lanes; nothing they compute is ever stored, because their scatter
    # row is the sink.
    alive = (freq > fsdead) & (fs2k > 0.0)

    # Virtual heights are the true heights: a thin mirror at HS.  ASIN's
    # argument is <= 1 for any HS > 0, and the clamp only catches the
    # degenerate lanes whose answer is discarded anyway.
    asin_arg = torch.clamp(k_one / (k_one + hsk / k_rz), min=-1.0, max=1.0)
    sdmax = k_two * k_rz * (k_pio2 - torch.asin(asin_arg))
    safe_sdmax = torch.where(sdmax > 0.0, sdmax, torch.ones_like(sdmax))
    ihsrt = torch.trunc(gcdkm / safe_sdmax + k_one).to(torch.long)
    if bool((alive & ((ihsrt < 1) | (sdmax <= 0.0))).any()):
        raise ValueError(
            "ESMOD: a live lane has SDMAX <= 0 (HS <= 0), so IHSRT would "
            "start below 1 and the Fortran's DO 150 would run more than the "
            "two hops /ZON/ has room for.  Fix the Es height rather than "
            "letting the loop bound come out of a degenerate geometry")
    live = alive & (ihsrt <= 2)              # DO ONLY 2 Es HOPS

    # Absorption loss factor, as for an F2 mode.  GYZ is indexed by K, the Es
    # sample area, where REGMOD would index it by LX(K).
    ac = k_677 * acav / (torch.pow(freq + gyzk, k_p198) + k_10p2)

    for ihop in range(1, 3):                 # DO 150 IHOP = IHSRT, 2
        run = live & (ihop >= ihsrt)
        k_hop = _k(float(ihop), freq)        # GP = IHOP, then HOP = IHOP

        ghop = gcd / k_hop
        thet = k_half * ghop
        tans = torch.sin(thet) / (k_one - torch.cos(thet) + hsk / k_rz)
        psi = torch.atan(tans)
        secs = k_one / torch.cos(psi)
        sfvmod = freq / secs
        esd = fs2k * secs                    # MUF at this hop distance
        dele = k_pio2 - thet - psi
        cdel = torch.cos(dele)
        adel = dele * k_r2d
        # IF(ADEL.lt.AMIND) go to 150 -- the trip is skipped but IH was
        # already incremented, so the slot is consumed either way.
        ok = run & (adel >= amind)

        # group path, free-space loss
        path = k_two * torch.sin(k_half * ghop) * (k_rz + hsk) / cdel
        path = k_hop * path
        # a dead lane's PATH can be anything, including negative; the ALOG10
        # argument is sanitised so the discarded arm cannot make a NaN that
        # then has to be reasoned about downstream
        sflos = k_32p45 + k_20 * torch.log10(
            torch.where(ok, path * freq, torch.ones_like(path)))

        sinp = k_rz * cdel / (k_rz + k_100)
        cosp = torch.sqrt(k_one - sinp * sinp)
        adx_on = fi1k > sfvmod               # Es below E: remove E bending
        ratio = torch.where(adx_on, sfvmod / fi1k, torch.ones_like(sfvmod))
        adx = torch.where(adx_on, afe + bfe * torch.log(ratio),
                          torch.zeros_like(sfvmod))
        sabps = ac / cosp + adx

        # probability of reflection at the MUF and at both deciles
        pros = torch.minimum(
            prbmuf_batch(freq, esd, ymuf4, sigl4, sigu4), k_p90)
        ref = k_refc * torch.pow(pros, k_pm07)
        esd = fs1k * secs                    # lower decile: use the FOT
        ps = torch.minimum(
            prbmuf_batch(freq, esd, yfot4, sigl4, sigu4), k_p9)
        refl = k_refc * torch.pow(ps, k_pm07)
        esd = fs3k * secs                    # upper decile: use the HPF
        ps = torch.minimum(
            prbmuf_batch(freq, esd, yhpf4, sigl4, sigu4), k_p9)
        refu = k_refc * torch.pow(ps, k_pm07)

        tllow = torch.minimum(dsl + k_hop * (refl - ref), k_25)
        tlhgh = torch.minimum(dsu + k_hop * (ref - refu), k_25)

        # GAIN's elevation must not go negative -- it indexes ARRAY with no
        # bottom clamp and the batched GAIN refuses rather than wrapping the
        # index round the axis.  Only the discarded lanes can get there.
        dele_g = torch.where(ok, dele, torch.zeros_like(dele))

        # DO 130: ground loss averaged over the live sample areas.  Adding a
        # hard zero for the dead ones is bit-identical to not adding them.
        y = torch.zeros_like(freq)
        for ig in range(NAREA):
            gl = ground_loss_batch(sigpat[ig], epspat[ig], dele_g, freq)
            y = y + torch.where(ig < km_l, gl, torch.zeros_like(gl))
        sgrlos = y / km_l.to(_F32)

        stgain, _steff = antenna_gain_batch(cant, 1, dele_g, freq)
        srgain, reff = antenna_gain_batch(cant, 2, dele_g, freq)

        # ADX is inside SABPS already and is added a SECOND time here
        xtlos = (sflos + k_hop * (sabps + ref + adx)
                 + (k_hop - k_one) * sgrlos
                 - srgain - stgain + asm)

        pdb = pwrdb_batch(cant, freq)
        fldst = (k_107p2 + pdb + k_20 * torch.log10(freq) - xtlos - srgain)
        sigpow = pdb - xtlos

        # MUFday: the probability the frequency is below the Es MUF.  YMUF is
        # passed as BOTH the median and the reference frequency here.
        pros = prbmuf_batch(freq, ymuf4, ymuf4, sigl4, sigu4)
        # PROS**IHOP with an INTEGER exponent -- gfortran calls powi, so this
        # is repeated multiplication, not powf.  IHOP is only ever 1 or 2.
        prob = pros if ihop == 1 else pros * pros

        wrote = {
            "obf": k_refc * torch.pow(pros, k_pm07),   # F obscuration n/a
            "adv": k_zero.expand(n),                   # no deviative loss
            "fslos": sflos,
            "tloss": xtlos,
            "abps": ac / cosp + adx,                   # ADX counted once here
            "grlos": sgrlos,
            "rgain": srgain,
            "tgain": stgain,
            "hn": k_hop.expand(n),
            "eff": reff,
            "sigpow": sigpow,
            "sn": sigpow - rcnse - reff,
            "hp": hsk,
            "b": adel,
            "prob": prob,
            "timed": path / k_vofl,
            "fldst": fldst,
            "tllow": tllow,
            "tlhgh": tlhgh,
        }

        # IH = IH + 1 happens once per TRIP, so the row this trip writes is
        # IHOP - IHSRT: a two-hop-only path writes slot 4, not slot 5.
        row = torch.where(ok, (ihop - ihsrt).clamp(0, NES - 1), sink)
        for name, value in wrote.items():
            out[name][row, lanes] = value
        out["nmode"][row, lanes] = torch.full_like(out["nmode"][0], 4)

    # Drop the sink rather than clear it: nothing outside this function can
    # then read a row whose contents depend on who shared the batch.
    result = {name: arr[:NES] for name, arr in out.items()}
    return result
