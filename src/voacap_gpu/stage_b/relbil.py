"""Stage B (PyTorch, batched) reliability — RELBIL.

The last real computation of the frequency loop.  Everything upstream produced
modes; RELBIL turns them into the numbers a user reads -- how often the circuit
works, on which mode, and how much more power it would take to hit the required
grade of service.  It is the first routine reading /allMODE/ rather than /ZON/,
so its input is a list of up to twenty candidate modes per lane and its output
is one column of the per-frequency /SON/ table.

THE MODE AXIS IS A PYTHON LOOP, THE LANE AXIS IS THE BATCH.  Twenty is a fixed
Fortran bound, and every one of the three phases below is a sequential fold
over it whose carried state differs per lane -- so twenty unrolled iterations
of a handful of elementwise kernels is both the cheap and the faithful port.
Slots above a lane's own NMMOD are masked out with ``im <= nmmod``.

Three separate things happen and conflating them is the easy mistake:

1. **Per-mode reliability.**  The required S/N is placed on a two-sided normal
   whose lower spread crosses the noise UPPER decile with the signal LOWER
   decile, and vice versa.  That crossing is deliberate -- a bad hour is high
   noise *and* low signal -- and the 1994 Lucas/Gibson correction that "fixed"
   it is still in the Fortran as commented-out lines next to the live ones.
   A mode at or below 70 km virtual height is not a mode; it gets 0.001.

   D10R/D50R/D90R ARE WRITTEN INSIDE THIS LOOP, so they end up holding the LAST
   LIVE MODE's values -- a last-active-wins carry, the same shape as ALOSFV's
   CF.  They live in /DON/ and PERSIST ACROSS CALLS, so a frequency whose modes
   are all dead leaves SNRLW/SNRUP/SNPR reading the PREVIOUS frequency's
   numbers.  That staleness is observable and is preserved here by taking the
   incoming triple as an argument and returning the updated one.

2. **Most-reliable-mode selection, in two passes.**  The first is a plain
   argmax with FIRST-WINS ties (the Fortran's test is a strict ``.GT.``).  The
   second walks the list again and will *replace* that choice with any mode
   whose reliability is within 0.05, preferring fewer hops and, among equal hop
   counts, the better median S/N.  The comment on the first pass says why it
   exists: the second pass alone "could produce an improper mode", because it
   compares each candidate against a running best it has already moved.
   XREL, XHN, XSN and IR are therefore true carried state and the loop cannot
   be turned into a reduction -- including the ``XREL = 1e-8`` floor, which is
   applied at the TOP of each iteration and STICKS.  (That floor is in fact
   UNREACHABLE: FNORML clamps |Z| at 5, so the smallest reliability it can
   return is about 5.8e-6, and a dead mode gets 0.001.  Ported anyway, and the
   test asserts the property that makes it dead rather than claiming to cover
   it.)

3. **The combined distribution.**  With more than one mode the signals add as
   powers with random phase: FOUR separate sums, since the median, the two
   deciles and the field strength each need their own reference level (field
   strength separately, because the receive antenna gain is already in it).
   Each sum is referred to its own maximum term so the ``10**x`` stays in
   range, and terms more than 100 dB down are dropped.  THE REDUCTION ORDER IS
   PRESERVED -- a sequential mode-order accumulation, never ``torch.sum``,
   because float32 addition is not associative and a parallel reduction
   reassociates differently on CPU and CUDA.

Deliberate transliterations, all of which look like bugs and are not:

* ``DSLF``/``DSUF`` are /SIGD/ variables used here as scratch; RELBIL clobbers
  them, so they are threaded in and out like any other state.
* the two decile clamps in phase 3 are ASYMMETRIC -- D10R gets only a floor and
  D90R only a ceiling.  Copied as written.
* the sums in phase 3 run over ALL modes, dead ones included.  There is no
  HP test in the DO 369 / DO 370 loops.
* DBLOS is written twice: first as the chosen mode's TLOSS, then immediately
  overwritten by the 2014 recalculation from the summed power.  Only the second
  survives; the order is kept because that is where DBLOS changed meaning.
* ``IRmethod`` is fixed at 1 in the source; the percentage test (2) and the
  "fall back to max reliability below 1%" rule (3) are commented out there and
  are comments here too.  ``CREL`` is set to 1000 and the source says it "is
  not used now".
* ``IS = NMODE(IM)`` inside the per-mode loop and ``MR = HN(IR)`` after it are
  assigned and never read.  Not ported.

float32.  LOG10 and the ``10.**`` powers carry the [[f32-libm-noise]] floor;
FNORML is pure polynomial and does not.  ``1. - FNORML(Z)`` is spelled out
rather than folded into the tail, because for Z >= 0 FNORML already computed
``1 - QX`` and the round trip through two float32 subtractions does not come
back to QX.
"""

import numpy as np
import torch

from voacap_gpu.stage_a.sonstate import LAYTYP
from voacap_gpu.stage_b.lockstep import gather_rows as _g
from voacap_gpu.stage_b.normal import fnorml_batch
from voacap_gpu.stage_b.profile import _k

_F32 = torch.float32
NM = 20                                     # /allMODE/ slots

# DATA TME -- the standard normal at 5% steps from the median.  TME(9) is the
# 90th percentile, the level the required-power ratio is taken against.
_TME = np.array([0.0, 0.1257, 0.2533, 0.3853, 0.5244, 0.6745, 0.8416,
                 1.0364, 1.2815, 1.6449], dtype=np.float32)
# TME(ITM)/TME(9) folded on the HOST in float32.  That is bit-for-bit the
# division Stage A does, and it keeps a per-lane divide by a table constant off
# the device entirely -- see profile._k on why dividing a float32 tensor by a
# Python float is not a float32 divide on CUDA.
_TMX = (_TME / _TME[8]).astype(np.float32)

_XEPS = float(np.float32(0.05))
_DCL = float(np.float32(1.28))              # deciles per sigma
_MIN_REL = float(np.float32(0.00000001))
_MIN_SPREAD = float(np.float32(0.2))
_MAX_SPREAD = float(np.float32(30.0))
_NO_MODE_HEIGHT = float(np.float32(70.0))
_FLOOR_DB = float(np.float32(-10.0))        # ZEXP floor: 100 dB down is gone
_DEAD_REL = float(np.float32(0.001))
_CREL = float(np.float32(1000.0))

MODE_NAMES = ("hp", "sn", "tllow", "tlhgh", "hn", "sigpow", "fldst", "tgain",
              "rgain", "b", "timed", "tloss", "prob", "eff", "nmode")

# the /SON/ + /DUDL_NOIS/ + /sncom/ + /cgains/ column fields RELBIL writes
SON_NAMES = ("angle", "cprob", "dblos", "dblosl", "dblosu", "dbu", "delay",
             "dbw", "nhp", "xnynois", "reliab", "sndb", "snpr", "snrlw",
             "snrup", "vhigh", "rneff", "du_nois", "dl_nois", "snxx",
             "gaint", "gainr", "modecode")


def _reliability(rsn, d50, d10, d90, k):
    """``1. - FNORML(Z)`` with Z placed on whichever tail RSN falls in.

    The Fortran divides by ``D??R/1.28`` unguarded.  A zero spread is reachable
    -- both noise deciles and the signal decile can be zero -- and gives IEEE
    infinity, which is harmless because FNORML clamps |Z| at 5 before doing
    anything with it.  Both arms are evaluated here and the untaken one can be
    +-inf without consequence; the single NaN case (Z exactly 0 over a zero
    spread) is 0/0 in the Fortran too.
    """
    z = rsn - d50
    lo = z / (d10 / k["dcl"])
    hi = z / (d90 / k["dcl"])
    return k["one"] - fnorml_batch(torch.where(z <= 0.0, lo, hi))


def relbil_batch(modes, nmmod, du, dl, rcnse, pwrdb, freq, rsn, lufp,
                 d10r, d50r, d90r, dslf, dsuf, nrel_prev=None, son_prev=None):
    """Batched RELBIL.  Returns ``(allm, son)``, both dicts of tensors.

    modes: dict of (NM+1, N) tensors 1-based in the mode axis (row 0 is
        padding, as in ``findf_batch``'s ray tables).  Required keys are
        ``MODE_NAMES``; ``nmode`` is integer, the rest float32.  ``crel`` and
        ``rely`` may also be supplied -- they are /allMODE/ outputs but RELBIL
        leaves the slots above NMMOD, and every slot of a lane with no modes
        at all, exactly as it found them.  A /allMODE/ state dict as
        ``stage_b.allmodes`` builds one can be handed straight in; extra keys
        are ignored, and the outputs come BACK rather than being written into
        it, so a driver assigns ``allm.update(relbil_out)``.
    nmmod: (N,) int, the live mode count.  <= 0 means the Fortran RETURNs
        before touching anything.
    du/dl/rcnse: (N,) /ANOIS/ noise deciles and median.
    pwrdb: (N,) transmitter power in dBW at FREQ, i.e.
        ``gain.pwrdb_batch(cant, freq)``.  It is passed IN rather than looked
        up here because /cantenna/ is shared across lanes in Stage B -- one
        transmitter, many grid points -- so a table lookup taking a shared
        table and a per-lane frequency belongs outside a per-lane routine.
        Stage A's tier-B harness fabricates a fresh one-entry table per case
        for the same reason.
    freq: (N,) MHz.  rsn: (N,) required S/N.  lufp: (N,) int, required
        reliability in percent.
    d10r/d50r/d90r: (N,) the /DON/ decile triple ON ENTRY.  These PERSIST
        across calls and RELBIL does not always assign them -- pass the
        previous frequency's values or the staleness is lost.
    dslf/dsuf: (N,) the /SIGD/ scratch on entry, same reason.
    nrel_prev: (N,) int NREL on entry; defaults to 1, which is BLOCK DATA's
        value and the slot MPATH/SERPRB would index before the first call.
    son_prev: optional dict of the /SON/ column's previous contents.  A lane
        with NMMOD <= 0 has none of these written; without son_prev the values
        returned for such a lane are the arithmetic's leftovers and mean
        nothing.

    allm: ``crel``/``rely`` as (NM+1, N) and ``nrel`` as (N,) int64.
    son: ``SON_NAMES`` as (N,) -- ``nhp`` and ``modecode`` int64, the rest
        float32 -- plus ``d10r``/``d50r``/``d90r`` (/DON/) and ``dslf``/
        ``dsuf`` (/SIGD/), which are state rather than column entries.
        ``modecode`` is the 1..5 LAYTYP index; the label is
        ``LAYTYP[modecode - 1]``.
    """
    hp, sn = modes["hp"], modes["sn"]
    tllow, tlhgh, hn = modes["tllow"], modes["tlhgh"], modes["hn"]
    sigpow, fldst = modes["sigpow"], modes["fldst"]
    n = freq.shape[0]
    dev = freq.device
    nmmod = nmmod.to(torch.long)
    lufp = lufp.to(torch.long)

    k = {name: _k(v, freq) for name, v in (
        ("one", 1.0), ("zero", 0.0), ("dcl", _DCL), ("xeps", _XEPS),
        ("minrel", _MIN_REL), ("dead", _DEAD_REL), ("crel", _CREL),
        ("no_mode_h", _NO_MODE_HEIGHT), ("tenth", 0.1), ("ten", 10.0),
        ("m500", -500.0), ("floor_db", _FLOOR_DB))}
    lo_spread, hi_spread = _k(_MIN_SPREAD, freq), _k(_MAX_SPREAD, freq)

    wrote = nmmod > 0                       # IF(INUM.LE.0) RETURN
    du2, dl2 = du * du, dl * dl

    crel = modes.get("crel")
    rely = modes.get("rely")
    crel = torch.zeros_like(hp) if crel is None else crel.clone()
    rely = torch.zeros_like(hp) if rely is None else rely.clone()

    # ---- 1. per-mode reliability (DO 310) --------------------------------
    for im in range(1, NM + 1):
        inloop = wrote & (im <= nmmod)
        dead = hp[im] <= k["no_mode_h"]
        live = inloop & ~dead
        # the crossing is deliberate: a bad hour is high noise AND low signal
        d10 = torch.sqrt(du2 + tllow[im] * tllow[im])
        d90 = torch.sqrt(dl2 + tlhgh[im] * tlhgh[im])
        rely_live = _reliability(rsn, sn[im], d10, d90, k)
        rely[im] = torch.where(inloop,
                               torch.where(dead, k["dead"], rely_live),
                               rely[im])
        crel[im] = torch.where(inloop,
                               torch.where(dead, k["dead"], k["crel"]),
                               crel[im])
        # last-live-wins: these are /SIGD/ and /DON/ scratch, written per mode
        dslf = torch.where(live, tllow[im], dslf)
        dsuf = torch.where(live, tlhgh[im], dsuf)
        d10r = torch.where(live, d10, d10r)
        d50r = torch.where(live, sn[im], d50r)
        d90r = torch.where(live, d90, d90r)

    # ---- 2a. the outright maximum (DO 10) --------------------------------
    # strict .GT., so the FIRST slot holding the maximum wins.
    irmax = torch.ones(n, dtype=torch.long, device=dev)
    best = rely[1].clone()
    for im in range(2, NM + 1):
        take = (im <= nmmod) & (rely[im] > best)
        irmax = torch.where(take, torch.full_like(irmax, im), irmax)
        best = torch.where(take, rely[im], best)

    # ---- 2b. the walk that can move away from it (DO 140) ----------------
    ir = irmax.clone()
    xrel, xhn, xsn = _g(rely, irmax), _g(hn, irmax), _g(sn, irmax)
    for im in range(1, NM + 1):
        act = (im <= nmmod) & (irmax != im)      # IF(IM.EQ.IRmax) GO TO 140
        # the floor is applied before the comparison and STICKS in XREL
        xrel = torch.where(act & (xrel < k["minrel"]), k["minrel"], xrel)
        reltest = torch.abs(rely[im] - xrel)
        # IRmethod 2 would divide RELtest by XREL for a percentage test
        equal_hops = torch.abs(xhn - hn[im]) <= k["xeps"]
        better = torch.where(equal_hops, xsn < sn[im], xhn > hn[im])
        upd = act & (reltest < k["xeps"]) & better
        ir = torch.where(upd, torch.full_like(ir, im), ir)
        xhn = torch.where(upd, hn[im], xhn)
        xsn = torch.where(upd, sn[im], xsn)
        xrel = torch.where(upd, rely[im], xrel)
    # IRmethod 3 would fall back to IRmax when RELY(IR) < .01

    rely_ir, sigpow_ir = _g(rely, ir), _g(sigpow, ir)

    # ---- 3. the frequency's summary --------------------------------------
    # arm A: a single mode is used as it stands.
    a_reliab, a_dblosl, a_dblosu = rely_ir, _g(tllow, ir), _g(tlhgh, ir)
    a_dbu, a_sndb, a_dbw = _g(fldst, ir), _g(sn, ir), sigpow_ir

    # arm B: add the signals with random phase, i.e. add the powers in watts.
    # Each sum is referred to its own maximum term to keep 10**x in range.
    inum_gt1 = wrote & (nmmod > 1)
    full = lambda v: torch.full((n,), float(np.float32(v)),  # noqa: E731
                               dtype=_F32, device=dev)
    dxsigs, dxfld = full(-1000.0), full(-1000.0)
    dxdslw, dxdsup = full(-1000.0), full(-1000.0)
    for iv in range(1, NM + 1):                 # DO 369
        act = wrote & (iv <= nmmod)
        dxsigs = torch.where(act, torch.maximum(dxsigs, sigpow[iv]), dxsigs)
        dxfld = torch.where(act, torch.maximum(dxfld, fldst[iv]), dxfld)
        dxdslw = torch.where(act,
                             torch.maximum(dxdslw, sigpow[iv] - tllow[iv]),
                             dxdslw)
        dxdsup = torch.where(act,
                             torch.maximum(dxdsup, sigpow[iv] + tlhgh[iv]),
                             dxdsup)

    def _add(acc, zexp, act):
        """``if(ZEXP.gt.-10.) ACC = ACC + 10.**ZEXP`` for one mode.

        Sequential, one mode at a time -- NOT ``torch.sum`` over the axis.  A
        parallel reduction reassociates, and float32 addition is not
        associative, so the two would disagree between CPU and CUDA.
        """
        return torch.where(act & (zexp > k["floor_db"]),
                           acc + torch.pow(k["ten"], zexp), acc)

    xdslw, xsigs, xdsup, xfld = full(0.0), full(0.0), full(0.0), full(0.0)
    for im in range(1, NM + 1):                 # DO 370
        act = wrote & (im <= nmmod)
        # NO HP TEST HERE: dead modes contribute to the sums.  Deliberate.
        xdslw = _add(xdslw, k["tenth"] * (sigpow[im] - tllow[im] - dxdslw),
                     act)
        xsigs = _add(xsigs, k["tenth"] * (sigpow[im] - dxsigs), act)
        xdsup = _add(xdsup, k["tenth"] * (sigpow[im] + tlhgh[im] - dxdsup),
                     act)
        # field strength is summed separately: the receive antenna gain is
        # already in it and must not be double-counted
        xfld = _add(xfld, k["tenth"] * (fldst[im] - dxfld), act)

    # every sum contains its own maximum term at ZEXP == 0, so all four are
    # >= 1 in practice; the .GT.0 guards below are the Fortran's and are dead.
    sigmed = torch.where(xsigs > 0.0, dxsigs + k["ten"] * torch.log10(xsigs),
                         k["m500"])
    b_dblosl = torch.where(
        xdslw > 0.0,
        torch.abs(sigmed - k["ten"] * torch.log10(xdslw) - dxdslw), k["zero"])
    b_dblosl = torch.minimum(torch.maximum(b_dblosl, lo_spread), hi_spread)
    b_dblosu = torch.where(
        xdsup > 0.0,
        torch.abs(dxdsup + k["ten"] * torch.log10(xdsup) - sigmed), k["zero"])
    b_dblosu = torch.minimum(torch.maximum(b_dblosu, lo_spread), hi_spread)
    b_dbw = sigmed
    delsig = sigmed - sigpow_ir
    b_dbu = torch.where(xfld > 0.0, dxfld + k["ten"] * torch.log10(xfld),
                        k["m500"])
    b_sndb = _g(sn, ir) + delsig

    # redo the reliability against the combined distribution.  The clamps are
    # ASYMMETRIC in the source -- D10R floored only, D90R capped only.
    b_d10r = torch.maximum(torch.sqrt(du2 + b_dblosl * b_dblosl), lo_spread)
    b_d50r = b_sndb
    b_d90r = torch.minimum(torch.sqrt(dl2 + b_dblosu * b_dblosu), hi_spread)
    b_reliab = _reliability(rsn, b_d50r, b_d10r, b_d90r, k)

    def pick(a, b):
        return torch.where(inum_gt1, b, a)

    reliab = pick(a_reliab, b_reliab)
    dblosl, dblosu = pick(a_dblosl, b_dblosl), pick(a_dblosu, b_dblosu)
    dbu, sndb, dbw = pick(a_dbu, b_dbu), pick(a_sndb, b_sndb), pick(a_dbw,
                                                                    b_dbw)
    d10r = pick(d10r, b_d10r)
    d50r = pick(d50r, b_d50r)
    d90r = pick(d90r, b_d90r)

    # ---- the tail, common to both arms -----------------------------------
    ilayer = _g(modes["nmode"].to(torch.long), ir)
    bad = wrote & ((ilayer < 1) | (ilayer > len(LAYTYP)))
    if bool(bad.any()):
        i = int(torch.argwhere(bad)[0][0])
        raise IndexError(
            f"NMODE({int(ir[i])}) = {int(ilayer[i])} on lane {i} is outside "
            f"LAYTYP 1..{len(LAYTYP)}; the Fortran reads out of bounds here")

    # DBLOS is assigned TLOSS(IR) and then immediately overwritten by the 2014
    # recalculation from the summed power.  Only the second survives; the order
    # is kept because that is where DBLOS's meaning changed.
    dblos = _g(modes["tloss"], ir)
    dblos = pwrdb - dbw

    hn_ir = _g(hn, ir)
    son = {
        "gaint": _g(modes["tgain"], ir), "gainr": _g(modes["rgain"], ir),
        "snrlw": d10r, "snrup": d90r, "angle": _g(modes["b"], ir),
        "vhigh": _g(hp, ir), "delay": _g(modes["timed"], ir), "dblos": dblos,
        "cprob": _g(modes["prob"], ir), "rneff": _g(modes["eff"], ir),
        "modecode": ilayer,
        # REAL -> INTEGER assignment truncates toward zero
        "nhp": torch.trunc(hn_ir).to(torch.long),
        "xnynois": rcnse, "du_nois": du, "dl_nois": dl,
        "reliab": reliab, "dblosl": dblosl, "dblosu": dblosu, "dbu": dbu,
        "sndb": sndb, "dbw": dbw,
    }

    # required power gain for the specified reliability
    itm = torch.clamp(
        torch.div(torch.abs(lufp - 50), 5, rounding_mode="floor") + 1, max=10)
    tmx = torch.as_tensor(_TMX, dtype=_F32, device=dev)[itm - 1]
    son["snpr"] = torch.where(lufp < 50,
                              -(d50r + tmx * d90r) + rsn,
                              -(d50r - tmx * d10r) + rsn)
    son["snxx"] = rsn - son["snpr"]

    if son_prev is not None:
        for name in SON_NAMES:
            old = son_prev.get(name)
            if old is not None:
                son[name] = torch.where(wrote, son[name], old.to(
                    son[name].dtype))
    son["d10r"], son["d50r"], son["d90r"] = d10r, d50r, d90r
    son["dslf"], son["dsuf"] = dslf, dsuf

    if nrel_prev is None:
        nrel_prev = torch.ones(n, dtype=torch.long, device=dev)
    allm = {"crel": crel, "rely": rely,
            "nrel": torch.where(wrote, ir, nrel_prev.to(torch.long))}
    return allm, son
