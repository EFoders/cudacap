"""Stage B (PyTorch, batched) per-mode loss budget — REGMOD.

REGMOD turns the up-to-six regular modes FDIST left in /MODES/ into the numbers
a prediction actually reports: free-space loss, non-deviative absorption,
deviative loss, ground-reflection loss, sporadic-E obscuration, the two antenna
gains, the over-the-MUF loss and its deciles, and from those the field
strength, the signal power and the signal-to-noise.  Sporadic-E *modes* belong
to ESMOD; what appears here is only Es getting in the way of an F mode.  Mode
slot 7 gets nothing but its HN/HP = -1 preset.

THE BATCH AXIS IS THE LANE AXIS AND NOTHING ELSE.  The mode index runs 1..6 as
a PYTHON loop: six slots is a fixed table bound, not a data-dependent one, and
each slot writes its own row of every output.  Inside a slot everything is
elementwise over lanes, so this is the plain compute-both-and-select shape --
no lockstep, no scatter, no carry between iterations.  The two structural
branches (F-layer against D-E mode, and the D-E collision-frequency
sub-branch), the Es obscuration and the low-MUFday penalty are all evaluated in
full on every lane and merged with ``torch.where``.

THREE ORDERING FACTS ARE LOAD-BEARING and are preserved verbatim:

* ``DEL`` and ``CDEL`` are RECOMPUTED partway through, at the 1995 hop-MUF
  change, from the PATH's geometry rather than the mode's.  Everything before
  that point -- crucially the three GAIN calls -- uses the MODE's take-off
  angle; everything after uses the recomputed one.  Reordering the statements
  silently changes which angle the antenna is asked about.
* ``CPHET`` is computed from the OLD ``CDEL`` and then kept across the
  recomputation, so the over-the-MUF loss divides by a secant belonging to the
  mode while its MUF comes from the path.  That mismatch is the Fortran's.
* ``ITRY`` is vestigial -- set to 1 at the top of every iteration, so the
  ``ITRY.lt.1`` test can never fire and the trailing ``ITRY = -1`` is dead.
  Kept as this comment rather than as code, exactly as Stage A keeps it.

WHERE THE RESIDUAL COMES FROM -- TWO CANCELLATIONS, and it is worth keeping
them apart because they land on different fields:

* ``1 - COS(PSI)`` in PATH, with PSI the half-hop angle.  At a 700 km hop that
  leaves 0.0015 from operands near 1, a ~660x magnification of COS's last bit,
  and ``RZ*(1 - COS PSI)`` is a few percent of PATH's numerator.  Measured
  PATH relative error 8e-7..3e-6.  TIMED is PATH/VOFL so it inherits that
  directly; ``20*ALOG10`` damps it to ~3e-5 dB in FSLOS.
* ``1 - SPHET**2`` in CPHET, which is the LARGER effect despite looking
  tamer.  One ULP of CDEL moves CPHET by ~2e-7, XMUF by ~5e-7 relative -- and
  PRBMUF's normal tail turns that into a ~2e-4 RELATIVE move in P, which
  ``-10*ALOG10(P)/CPHET`` amplifies once more into ~3e-3 dB of XLS per hop.
  That, not PATH, is the worst cell in TLOSS/SIGPOW/FLDST/SN.

Every other term is well conditioned.  Those two are not, and they are why the
bars are a ULP budget rather than bit-exactness -- on top of them sit
ALOG10/ALOG/EXP/COS/SIN/ATAN/SQRT and two ``**1.98`` powers, all DIFFERENT
FUNCTIONS in torch than in this gfortran build's libm.

``**1.98`` is a default-REAL literal, so the exponent is the float32 nearest
1.98 (1.98000001907...), not the binary64 1.98; ``_k`` rounds it before
``torch.pow`` sees it.  Stage A goes further and evaluates the power in
binary64 to get the correctly rounded float32; torch's ``powf`` is only
~1 ULP accurate, which is the same class of noise as every other libm call
here and is not worth a float64 kernel on a consumer GPU.

NOT PORTED: nothing.  The Fortran's STOPs on a non-positive MUFday probability
or a non-positive frequency become :class:`RegmodError`, raised if ANY lane
with a live mode would have hit them -- silently returning NaN for one lane of
a million is the failure mode this project cannot afford.

THE ROUTINE IS SPLIT IN THREE SO THAT THE MIDDLE PART CAN BE A CUDA GRAPH.
Measured on an RTX 5080, REGMOD is FLAT at ~68-88 ms from one lane to a
million: it is bound by kernel LAUNCHES, not by arithmetic.  ``graph.py``
records the launches once and replays them, which is bit-for-bit identical
because it is the same kernels in the same order.  What that costs is a
reordering -- not of arithmetic, of WHICH BLOCK RUNS WHEN:

* ``_gain_prologue`` -- the mode take-off angles and the four GAIN answers
  (ground loss, both gains, the receive efficiency) plus PWRDB.  This block
  CANNOT be captured, and not by accident: ``gain.antenna_gain_batch`` pays
  two device syncs per call to refuse the Fortran's out-of-bounds table read,
  and its docstring says so.  So it stays eager and is hoisted to the front.
* ``_regmod_core`` -- everything else, one straight line, capturable.
* the MUFday STOP -- the Fortran raises inside the mode loop, which is a host
  sync and therefore illegal inside a capture.  The core counts the offending
  lanes per mode into a tensor instead, and the count is read back once, after
  the replay, and turned into the same :class:`RegmodError`.  The only visible
  difference is that all six modes are now computed before the first bad one
  raises.

MOVING THE GAIN BLOCK DOES NOT MOVE ANY ARITHMETIC.  GAIN is asked about the
MODE's take-off angle, which is computed from ``DELMOD`` alone, and its inputs
(SIGPAT/EPSPAT/KM/FREQ) are all loop-invariant, so hoisting the calls is a
pure reordering of independent work.  The "three ordering facts" above still
hold: the prologue uses the MODE's angle and the recomputed 1995 hop angle
lives entirely inside the core, after the point where GAIN was called.

MEASURED on an RTX 5080, best of five, synthetic batch, all six mode slots
live (so this is the worst case, not the typical one):

     lanes    eager   graph   GAIN prologue   captured core   whole call
     1,000    64 ms   36 ms       32 ms       33 -> 4.3 ms       1.8x
    10,000    63 ms   37 ms       32 ms       31 -> 5.0 ms       1.7x
   100,000    65 ms   39 ms       33 ms       31 -> 5.2 ms       1.7x
 1,000,000    77 ms   51 ms       34 ms       43 -> 16.8 ms      1.5x

READ THAT AS TWO NUMBERS, NOT ONE.  The core is 6-8x faster and stays flat to
100,000 lanes, which is what capture is supposed to buy; the whole call is
only 1.5-1.8x faster because the GAIN prologue, which is HALF of REGMOD, is
still 43 eager calls (12 antenna gains, 30 ground losses, one PWRDB) at
~0.8 ms each.  That half is not slow arithmetic, it is the same launch
overhead -- so the remaining lever is GAIN's, not REGMOD's, and it is one of
two things: batch the 43 calls into 3 by letting GAIN take a (modes, lanes)
delta (its docstring currently promises (N,), so that is its owner's call), or
make its two bounds guards capturable so the block can join the graph.
Neither is done here.

float32 throughout, mirroring Stage A.
"""

import numpy as np
import torch

from voacap_gpu.stage_a.constants import D2R, PIO2, RZ, VOFL
from voacap_gpu.stage_b.gain import (antenna_gain_batch, ground_loss_batch,
                                     pwrdb_batch)
from voacap_gpu.stage_b.gain import check_gain_indices
from voacap_gpu.stage_b.graph import StraightLineGraph
from voacap_gpu.stage_b.lockstep import gather_rows
from voacap_gpu.stage_b.normal import prbmuf_batch
from voacap_gpu.stage_b.profile import _k

_F32 = torch.float32

NMODES = 7                      # /ZON/ slots; REGMOD fills 1..6
NAREA = 5                       # sample areas the ground loss averages over
NLAYER = 4                      # /MUFS/ rows: E, F1, F2, Es

_P198 = 1.98                    # the default-REAL exponent, see the docstring
_DELMAX = float(np.float32(89.99) * D2R)
_PROB_FLOOR = 0.000001          # AMAX1 floor under every over-the-MUF P
_CPHET_FLOOR = 0.000001
_ES_LAYER = 4                   # /MUFS/ row the Es obscuration reads

# the /ZON/ fields REGMOD writes.  CREL/RELY/SPRO are RELBIL's and are not
# returned, because REGMOD does not touch them and a zero here would look
# like an answer.
FIELDS = ("abps", "adv", "b", "eff", "fldst", "fslos", "grlos", "hn", "hp",
          "nmode", "obf", "prob", "rgain", "sigpow", "sn", "tgain", "timed",
          "tllow", "tlhgh", "tloss")


class RegmodError(ValueError):
    """A state REGMOD refuses to run on, where the Fortran STOPs."""


def regmod_batch(modes, muf, sigd, cant, sigpat, epspat, km,
                 fi1, gyz, fs1, fs2, hs, ifob13, rcnse, freq, gcd,
                 use_cuda_graph=True):
    """Batched REGMOD.  Returns /ZON/ as a dict of (7, N) tensors.

    THE SAMPLE AREA IS ALREADY CHOSEN.  Stage A carries JMODE and indexes
    ``fi(1,K)``, ``gyz(LX(K))``, ``fs(*,K)``, ``hs(K)``, ``ifob(1,3,K)`` with
    it; in a batch JMODE is per-lane, so the caller does that gather once (with
    ``lockstep.gather_rows``) and passes the (N,) rows.  That is also what the
    oracle dump injects, so the interface and the fixture agree.

    modes: dict from ``fdist_batch`` -- ``delmod``/``hpmod``/``htmod``/
        ``fvmod``/``afmod``/``itmod`` as (7, N) with the FORTRAN 1-based mode
        index, i.e. rows 1..6 live and row 0 unused -- plus ``ghop``, the hop
        angle in radians, as (N,).
    muf: dict of (4, N) /MUFS/ tables -- ``sigl``, ``sigu``, ``ymuf``,
        ``yfot``, ``yhpf``, ``hpmuf``, ``htmuf``, ``fvmuf`` -- with rows 0..3
        for the Fortran layers 1..4 (E, F1, F2, Es), the ``areas`` convention.
    sigd: dict of (N,) /SIGD/ scalars -- ``acav``, ``htloss``, ``xnuz``,
        ``hnu``, ``xve``, ``afe``, ``bfe``, ``asm``, ``dsl``, ``dsu``.
    cant: a ``gain.BatchedCantenna``.
    sigpat/epspat: (5, N) ground constants per sample area, rows 0..4.
    km: (N,) integer count of live sample areas -- the ground loss averages
        over the first KM of them, and KM differs per lane, which is why the
        area loop is masked rather than sliced.
    fi1/gyz/fs1/fs2/hs/ifob13: (N,) the JMODE-selected E critical frequency,
        gyrofrequency, Es FOT and median, Es height, and the integer Es
        sounding floor from /RAYS/ IFOB(1,3).
    rcnse: (N,) receiver-site noise.  freq: (N,) MHz.  gcd: (N,) great-circle
        distance in RADIANS.
    use_cuda_graph: capture the straight-line core once per batch shape and
        replay it, the convention ``findf_batch``/``fdist_batch`` already use.
        Ignored on CPU.  A replay is the same kernels in the same order, so it
        is bit-for-bit the eager answer; what it costs is one captured graph
        (and its private memory pool) per distinct batch shape, which
        ``reset_graph_cache`` releases.

    The RETURNED arrays are DENSE (7, N): row 0 is mode slot 1 and row 6 is
    slot 7.  That differs from the 1-based `modes` input on purpose -- /ZON/
    has exactly seven slots and none of them is a padding row -- so read the
    output with ``out["tloss"][im - 1]``.
    """
    n = freq.shape[0]
    dev = freq.device

    live = torch.zeros(n, dtype=torch.bool, device=dev)
    for im in range(1, NMODES):
        live = live | (modes["hpmod"][im] > 0.0)
    # the Fortran STOPs on FREQ <= 0 inside the mode loop, so a lane with no
    # live mode never reaches it; the gate keeps that true here.  This is a
    # host sync and so it has to happen HERE, before any capture: measured,
    # a sync attempted inside a capture does not merely raise, it invalidates
    # the stream for the rest of the process.
    if bool((live & (freq <= 0.0)).any()):
        raise RegmodError(
            "frequency must be positive where a mode exists; the "
            "Fortran STOPs")

    # GAIN's refusal of the Fortran's out-of-bounds table read is the OTHER
    # host sync, and it used to force the whole GAIN block to stay eager.  It
    # is a policy check rather than a memory guard -- the gathers clamp
    # regardless -- so it runs once here, over every take-off angle GAIN will
    # be asked about, and the calls themselves then run sync-free.
    # ALL SIX MODES AT ONCE, in two calls rather than twelve.  Validating per
    # mode simply relocates the syncs it was meant to remove -- measured, the
    # check is 0.63 ms a call and REGMOD makes twelve of them, so the per-mode
    # loop was a wash.  Dead slots get a known-good angle rather than being
    # skipped, because skipping them is itself a sync.
    # THE MINIMUM LIVE TAKE-OFF ANGLE IS EXACTLY EQUIVALENT to checking all
    # six.  GAIN's frequency index does not depend on the angle at all, and
    # its elevation index fails only from BELOW -- J = INT(DELTD)+1 < 1 iff
    # DELTD < 0 -- so if the smallest live angle clears the bound, every angle
    # does.  Dead slots take a known-good 0.5 rad rather than being masked
    # out, because masking them is itself a sync.
    _safe = _k(0.5, freq).expand_as(freq)
    _dmin = _safe
    for _im in range(1, NMODES):
        _dmin = torch.minimum(
            _dmin, torch.where(modes["hpmod"][_im] > 0.0,
                               modes["delmod"][_im] * _k(D2R, freq), _safe))
    check_gain_indices(cant, 1, _dmin, freq)
    check_gain_indices(cant, 2, _dmin, freq)

    core = _gain_prologue(modes, cant, sigpat, epspat, km, freq)
    core.update({name: modes[name] for name in
                 ("delmod", "hpmod", "htmod", "fvmod", "afmod", "itmod",
                  "ghop")})
    core.update(muf)
    core.update(sigd)
    core.update(fi1=fi1, gyz=gyz, fs1=fs1, fs2=fs2, hs=hs, ifob13=ifob13,
                rcnse=rcnse, freq=freq, gcd=gcd)
    out = _CORE.run(core, use_cuda_graph=use_cuda_graph)

    # the SECOND Fortran STOP, deferred.  It lives inside the mode loop, where
    # a sync is not available under capture, so the core counts the offending
    # lanes per mode and the count is read back once, here.
    bad = out.pop("badmuf")
    if bool(bad.any()):
        im = int(torch.nonzero(bad)[0]) + 1
        raise RegmodError(
            f"MUFday probability is non-positive for mode {im} on "
            f"{int(bad[im - 1])} lane(s); the Fortran STOPs here")
    return out


def _gain_prologue(modes, cant, sigpat, epspat, km, freq):
    """The mode take-off angles and every GAIN answer.  EAGER, always.

    This is the block that cannot go inside a CUDA graph.
    ``gain.antenna_gain_batch`` spends two device syncs per call refusing the
    out-of-bounds table read the Fortran performs -- deliberately, and its
    docstring says so -- and a sync inside a capture invalidates the stream.
    So it is hoisted out of the mode loop and run first.  Nothing moves
    arithmetically: GAIN is asked about the MODE's take-off angle, which comes
    from DELMOD alone, and SIGPAT/EPSPAT/KM/FREQ are loop-invariant.

    Returns the (6, N) per-mode stacks the core consumes, row 0 = mode slot 1.
    """
    n = freq.shape[0]
    dev = freq.device
    k = lambda v: _k(v, freq)                            # noqa: E731
    k0 = k(0.0)
    kd2r = k(D2R)
    zero = torch.zeros(n, dtype=_F32, device=dev)

    dels, actives, grloss, tgains, rgains, effs = [], [], [], [], [], []
    for im in range(1, NMODES):
        delm = modes["delmod"][im]
        active = modes["hpmod"][im] > 0.0
        del_ = torch.clamp(kd2r * delm, max=_DELMAX)
        # ---- 160: ground reflection, averaged over the live sample areas ---
        # A dead slot's take-off angle can be anything, and GAIN indexes its
        # table with it; feeding zero there keeps the index in range without
        # touching a live lane's answer.
        del_g = torch.where(active, del_, zero)
        y = zero
        for ig in range(1, NAREA + 1):
            gl = ground_loss_batch(sigpat[ig - 1], epspat[ig - 1], del_g, freq)
            y = y + torch.where(ig <= km, gl, k0)      # + 0.0 is exact
        tgain, _teff = antenna_gain_batch(cant, 1, del_g, freq,
                                          validate=False)
        rgain, eff = antenna_gain_batch(cant, 2, del_g, freq,
                                        validate=False)
        dels.append(del_)
        actives.append(active)
        grloss.append(y / km.to(_F32))
        tgains.append(tgain)
        rgains.append(rgain)
        effs.append(eff)

    return {"delx": torch.stack(dels), "activex": torch.stack(actives),
            "grlosx": torch.stack(grloss), "tgainx": torch.stack(tgains),
            "rgainx": torch.stack(rgains), "effx": torch.stack(effs),
            # no mode dependence; hoisted, as it always was
            "pdb": pwrdb_batch(cant, freq)}


def _regmod_core(t):
    """Everything REGMOD does once GAIN is out of the way.  One straight line.

    Dict in, dict out, no host syncs and no data-dependent control flow, which
    is what makes it capturable.  `t` is ``_gain_prologue``'s output merged
    with every plain tensor input; the 1-based /MODES/ tables keep their
    (7, N) shape while the prologue's stacks are (6, N) with row 0 = slot 1.

    Returns /ZON/ plus ``badmuf``, a (6,) count of the lanes that would have
    hit the Fortran's non-positive-MUFday STOP, one entry per mode slot.  The
    caller reads it back and raises; a raise here would be a host sync.
    """
    freq, gyz, gcd = t["freq"], t["gyz"], t["gcd"]
    fi1, fs1, fs2, hs = t["fi1"], t["fs1"], t["fs2"], t["hs"]
    n = freq.shape[0]
    dev = freq.device
    k = lambda v: _k(v, freq)                            # noqa: E731
    zero = torch.zeros(n, dtype=_F32, device=dev)
    k0, k1, k2 = k(0.0), k(1.0), k(2.0)
    krz, kpio2 = k(RZ), k(PIO2)
    k10_2 = k(10.2)                 # XNSQ, the nominal collision term
    kp198 = k(_P198)

    ghop = t["ghop"]
    # average absorption index from SIGDIS, and the (FREQ + fH)**1.98
    # denominator both modes share -- neither depends on the mode
    ac = k(677.2) * t["acav"]
    bc = torch.pow(freq + gyz, kp198)
    ihop = torch.trunc(gcd / ghop + k(0.01))             # INT() truncates
    hop = ihop
    hops = torch.clamp(hop, max=2.0)      # only 2 hops carry the obscuration
    pdb = t["pdb"]

    # Es sounding floor: at low frequencies folding regular E into sporadic E
    # misbehaves, so FSDEAD applies (SIGDIS does the same).  INTEGER -> REAL.
    fsdead = torch.clamp(t["ifob13"].to(_F32) / k(1000.0), max=3.0)
    fmhz_es = torch.maximum(freq, fsdead)
    es_row = _ES_LAYER - 1
    sigl_es, sigu_es = t["sigl"][es_row], t["sigu"][es_row]

    out = {name: torch.zeros((NMODES, n), dtype=_F32, device=dev)
           for name in FIELDS}
    # NMODE is a layer tag, INTEGER in /ZON/, and stays one here
    out["nmode"] = torch.zeros((NMODES, n), dtype=torch.int32, device=dev)
    # every slot is preset unused; only a slot with a mode overwrites it, and
    # slot 7 never does -- ESMOD fills that one
    minus1 = torch.full((n,), -1.0, dtype=_F32, device=dev)
    out["hn"] += minus1
    out["hp"] += minus1
    badmuf = []

    for im in range(1, NMODES):
        # ITRY = 1 here in the Fortran; the ITRY test below it is unreachable
        row = im - 1
        delm = t["delmod"][im]
        hpm = t["hpmod"][im]
        htm = t["htmod"][im]
        fvm = t["fvmod"][im]
        afm = t["afmod"][im]
        itm = t["itmod"][im]
        active = t["activex"][row]
        # ISMOD indexes /MUFS/; an unused slot can hold anything (0 included),
        # and a 0 would gather row -1 and wrap to the far end of the axis, so
        # the gather index is clamped.  Only rows discarded below can differ.
        ismod = itm.to(torch.long).clamp(1, NLAYER) - 1
        ymuf_s = gather_rows(t["ymuf"], ismod)
        yfot_s = gather_rows(t["yfot"], ismod)
        yhpf_s = gather_rows(t["yhpf"], ismod)
        hpmuf_s = gather_rows(t["hpmuf"], ismod)
        htmuf_s = gather_rows(t["htmuf"], ismod)
        fvmuf_s = gather_rows(t["fvmuf"], ismod)
        sigl_s = gather_rows(t["sigl"], ismod)
        sigu_s = gather_rows(t["sigu"], ismod)

        # ---- the MODE's own geometry --------------------------------------
        # DEL is the prologue's, because GAIN was asked about that same angle
        del_ = t["delx"][row]
        cdel = torch.cos(del_)
        psi = ghop * k(0.5)
        phe = kpio2 - psi - del_
        path = k2 * (hpm + krz * (k1 - torch.cos(psi))) / torch.cos(phe)
        path = torch.abs(path * hop)
        timed = path / k(VOFL)
        fslos = k(32.45) + k(20.0) * torch.log10(path * freq)

        # SECP at the reflection point, shared by both arms' deviative term
        secp_dev = k1 / torch.sin(del_ + psi)
        # (FV + fH)**1.98 -- negative bases exist only on dead slots, where
        # torch.pow yields NaN and the select below discards it
        pow_fv = torch.pow(fvm + gyz, kp198)

        # ---- arm A (label 125): an F-layer mode ---------------------------
        # 100 km is the assumed absorbing height.  SINP <= COS(DEL) <= 1 by
        # construction, so the clamp under SQRT is a no-op on real geometry
        # and only stops a dead slot manufacturing a NaN.
        sinp = krz * cdel / (krz + k(100.0))
        secp = k1 / torch.sqrt(torch.clamp(k1 - sinp * sinp, min=0.0))
        abps_f = secp * ac / (bc + k10_2)
        adv_f = secp_dev * afm * (pow_fv + k10_2) / (bc + k10_2)

        # Es obscuration of that F mode.  The median MUF is for THIS hop, not
        # for the path; and note the deciles read FS(1) where the median reads
        # FS(2), which is the Fortran's own asymmetry.
        sins = krz * cdel / (krz + hs)
        secs = k1 / torch.sqrt(torch.clamp(k1 - sins * sins, min=0.0))

        def _obsc(fs, fset):
            pros = torch.clamp(prbmuf_batch(fmhz_es, fs * secs, fset,
                                            sigl_es, sigu_es), max=0.90)
            return k(-10.0) * torch.log10(k1 - pros)

        has_es = fs2 > 0.0
        obf_f = torch.where(has_es, _obsc(fs2, t["ymuf"][es_row]), zero)
        obfu_f = torch.where(has_es, _obsc(fs1, t["yfot"][es_row]), zero)
        obfl_f = torch.where(has_es, _obsc(fs1, t["yhpf"][es_row]), zero)

        # ---- arm B (label 106): a D-E mode --------------------------------
        hnux = k(61.0) + k(3.0) * (htm - k(70.0)) / k(18.0)
        xnsq = torch.where(t["htloss"] - htm <= 0.0, k10_2,
                           t["xnuz"] * torch.exp(k(-2.0) * (hnux - k(60.0))
                                                 / t["hnu"]))
        heff = torch.clamp(htm, max=100.0)
        sinp = krz * cdel / (krz + heff)
        secp = k1 / torch.sqrt(torch.clamp(k1 - sinp * sinp, min=0.0))
        abps_de = secp * ac / (bc + xnsq)
        # absorption loss with the E-layer bending effect removed
        xv = torch.maximum(fvm / fi1, t["xve"])
        adx = t["afe"] + t["bfe"] * torch.log(xv)
        adv_de = secp_dev * afm * (pow_fv + xnsq) / (bc + xnsq) + adx

        is_f = fi1 - fvm < 0.0
        abps = torch.where(is_f, abps_f, abps_de)
        adv = torch.where(is_f, adv_f, adv_de)
        obf = torch.where(is_f, obf_f, zero)        # no Es obscuration for D-E
        obfu = torch.where(is_f, obfu_f, zero)
        obfl = torch.where(is_f, obfl_f, zero)

        # ---- 160: the ground and antenna answers, from the prologue -------
        grlos = t["grlosx"][row]
        tgain = t["tgainx"][row]
        rgain = t["rgainx"][row]
        eff = t["effx"][row]

        xtlos = (fslos + hop * (abps + adv) + grlos * (hop - k1) + hops * obf
                 + t["asm"] - rgain - tgain)

        # CPHET belongs to the MODE, and survives the recomputation below
        sphet = krz * cdel / (krz + htm)
        cphet = torch.clamp(k1 - sphet * sphet, min=_CPHET_FLOOR)
        cphet = torch.sqrt(cphet)

        # ---- 1995 change: base MUFday on the specific HOP muf -------------
        # DEL/CDEL are rebuilt from the PATH here.  Everything above used the
        # mode's angle; everything below uses this one.
        psi = gcd / k2
        psi = psi / hop
        cpsi = torch.cos(psi)
        spsi = torch.sin(psi)
        tanp = spsi / (k1 - cpsi + hpmuf_s / krz)
        phe = torch.atan(tanp)
        del_ = kpio2 - phe - psi
        cdel = torch.cos(del_)
        sphe = krz * cdel / (krz + htmuf_s)
        xmuf = fvmuf_s / torch.sqrt(torch.clamp(k1 - sphe * sphe, min=0.0))
        prob = prbmuf_batch(freq, xmuf, ymuf_s, sigl_s, sigu_s)
        # the Fortran STOPs right here; under capture that is a host sync, so
        # the count is carried out and raised on by the caller
        badmuf.append((active & (prob <= 0.0)).sum().to(torch.int32))
        # extra loss as MUFday gets very low, 0 to 24 dB over 1e-4..1e-7
        xfac = torch.clamp(torch.log10(prob), min=-7.0, max=-4.0)
        ghlos = (xfac + k(4.0)) * k(8.0)
        xtlos = torch.where(prob < 0.0001, xtlos - ghlos, xtlos)

        def _xls(fgo, fset):
            p = torch.clamp(prbmuf_batch(freq, fgo / cphet, fset,
                                         sigl_s, sigu_s), min=_PROB_FLOOR)
            return k(-10.0) * torch.log10(p) / cphet

        xls = _xls(fvmuf_s, ymuf_s)
        xtlos = xtlos + xls * hop                          # median

        cpr = fvmuf_s / ymuf_s
        xlsl = _xls(yfot_s * cpr, yfot_s)
        xlsu = _xls(yhpf_s * cpr, yhpf_s)

        # despite the names these are the signal level's lower/upper deciles
        tllow = torch.clamp(t["dsl"] + hops * (obfl - obf)
                            + hop * (xlsl - xls), max=25.0)
        tlhgh = torch.clamp(t["dsu"] + hops * (obf - obfu)
                            + hop * (xls - xlsu), max=25.0)

        fldst = k(107.2) + pdb + k(20.0) * torch.log10(freq) - xtlos - rgain
        sigpow = pdb - xtlos
        # long-path receiver efficiency correction (1991)
        sn = sigpow - t["rcnse"] - eff

        for name, val in (("abps", abps), ("adv", adv), ("eff", eff),
                          ("fldst", fldst), ("fslos", fslos), ("grlos", grlos),
                          ("obf", obf), ("prob", prob), ("rgain", rgain),
                          ("sigpow", sigpow), ("sn", sn), ("tgain", tgain),
                          ("timed", timed), ("tllow", tllow),
                          ("tlhgh", tlhgh), ("tloss", xtlos),
                          ("b", delm)):
            out[name][row] = torch.where(active, val, k0)
        out["nmode"][row] = torch.where(active, itm.to(torch.int32),
                                        torch.zeros_like(out["nmode"][row]))
        out["hp"][row] = torch.where(active, hpm, minus1)
        out["hn"][row] = torch.where(active, hop, minus1)

    out["badmuf"] = torch.stack(badmuf)
    return out


# One instance, module level, so the captured graphs survive between calls --
# that is the entire point.  ``regmod`` is in the list because its own ``_k``
# calls have to be memoised too, not just PRBMUF's; names rather than modules
# because this line runs while ``regmod`` is still being imported.
_CORE = StraightLineGraph(_regmod_core, "REGMOD",
                          const_modules=("voacap_gpu.stage_b.regmod",
                                         "voacap_gpu.stage_b.normal"))


def reset_graph_cache():
    """Drop every captured REGMOD graph and the memory pools they pin.

    One graph per distinct batch shape, and a graph's private pool holds the
    routine's whole working set -- at a million lanes that is gigabytes, so a
    caller that sweeps shapes should call this between sweeps.
    """
    _CORE.reset()
