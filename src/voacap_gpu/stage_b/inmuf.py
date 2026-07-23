"""Stage B (PyTorch, batched) over-the-MUF mode insertion — INMUF.

FDIST only finds modes that actually appear in the reflectrix.  Above the MUF
there are none, and at very short distances the ray table does not reach --
yet the prediction still needs a mode to hang a loss on, or the frequency
sweep falls off a cliff at the MUF instead of degrading through it.  INMUF
supplies them, then hands the filled list to REGMOD.

Four stages, in order: compact the list; insert the circuit MUF mode for any
layer that has none; recompute a PER-HOP MUF for layers whose MUF was
established at a different hop count and insert there instead; or, for a very
short path, synthesise a zero-distance mode straight off the ionogram at 90
degrees take-off.

EVERY LOOP HERE HAS A COMPILE-TIME TRIP COUNT -- six mode slots, three layers,
thirty ionogram points -- so none of them needs the lockstep driver.  They
unroll in Python with a per-lane cursor and a per-lane done mask, which is the
cheaper shape when the bound is a constant: no program counter, no convergence
check, no graph capture.  Lockstep earns its cost when the trip count is
DATA-dependent, and here it never is.

THE CURSOR MAKES THIS A SCATTER, and the scatter goes through row 0 as a sink
exactly as in FDIST and FINDF: a lane not inserting this step writes to row 0,
which nothing reads, so the store stays unconditional instead of masked.  The
sink is cleared before returning, or the answer would depend on which other
lanes shared the batch.

STAGE 1 REWRITES THE ARRAY IT IS READING, and that is safe for the same reason
it is safe in the Fortran: the destination cursor never runs ahead of the
source index, so a slot is only ever overwritten after it has been read.  The
port keeps the in-place form rather than building a copy, because the two are
not equivalent -- see the deliberate non-clearing below.

Two behaviours preserved because they look like bugs and are not:

* The compaction does **not** clear the slots above INUM, so a vacated slot
  keeps its previous occupant.  With FDIST's contiguous input nothing is ever
  left behind, but the trailing NUMMOD count scans all six slots and would see
  a stale mode if it were.
* Stage 3 assigns ``GHOP = 2*PSI``, mutating the hop distance in /MODES/ that
  REGMOD then reads.  The insert changes the geometry for EVERY mode, not just
  the inserted one.

The INFO-gated writes to unit 99 are diagnostics computing nothing that
reaches /MODES/ or /ZON/, so they are not ported.

float32; COS/SIN/SQRT/ATAN carry the [[f32-libm-noise]] floor.
"""

import torch

from voacap_gpu.stage_a.constants import R2D, RZ
from voacap_gpu.stage_b.lockstep import gather_rows as _g
from voacap_gpu.stage_b.profile import _k
from voacap_gpu.stage_b.regmod import regmod_batch
from voacap_gpu.stage_b.xlin import xlin_batch

_F32 = torch.float32

NMODES = 6                      # /MODES/ slots
NLAYER = 3                      # E, F1, F2
NFV = 30                        # ionogram points
_EPS = 0.4001                   # the frequency slack on every MUF comparison
_HMIN = 70.0                    # a virtual height below this is not a mode
_FULL = 5                       # INUM >= 5 means every layer is represented

_MODE_FIELDS = ("delmod", "hpmod", "htmod", "fvmod", "afmod")


def _scatter(modes, row, lanes, values, itmod_value=None):
    """Write one mode at each lane's own cursor, sinking to row 0 elsewhere.

    ``itmod_value`` is optional because stage 4 does NOT set the layer when it
    writes the mode -- label 270 does that afterwards, and only if some layer
    actually contains the height.  Writing a placeholder here would overwrite
    what the Fortran leaves in place on the lanes where 270 finds nothing.
    """
    for name, v in zip(_MODE_FIELDS, values):
        modes[name][row, lanes] = v
    if itmod_value is not None:
        modes["itmod"][row, lanes] = itmod_value


def inmuf_batch(state, modes, muf, sigd, cant, sigpat, epspat, km,
                fi1, gyz, fs1, fs2, hs, ifob13, rcnse, freq, gcd, ihop,
                jmode):
    """Batched INMUF: insert modes, run REGMOD, return ``(nummod, zon)``.

    `modes` is MUTATED in place, as the Fortran mutates /MODES/ -- the caller
    needs the modified list, and stage 3's GHOP change in particular is
    observable downstream.

    state: dict of the JMODE-selected ionosphere -- ``fvert``/``hprim``/
        ``htrue``/``afac`` as (30, N), ``fi`` as (4, N) rows 0..3 for the
        Fortran layers 1..4, and ``hi`` likewise.  The area gather is the
        caller's job, matching ``regmod_batch``.
    modes/muf/sigd/cant/sigpat/epspat/km/fi1/gyz/fs1/fs2/hs/ifob13/rcnse/freq:
        exactly as ``regmod_batch`` takes them; `muf` additionally needs
        ``delmuf``, ``afmuf``, ``nhopmf`` and ``modmuf``.
    gcd: (N,) great-circle distance in RADIANS, as REGMOD takes it.
    ihop: (N,) integer hop count under test.
    jmode: (N,) the circuit mode, used only for the F1-presence test.

    /MUFS/ is restored on the way out wherever stage 3 changed it.
    """
    n = freq.shape[0]
    dev = freq.device
    lanes = torch.arange(n, device=dev)
    zero_i = torch.zeros(n, dtype=torch.long, device=dev)
    k_eps, k_hmin = _k(_EPS, freq), _k(_HMIN, freq)
    k_rz, k_one, k_two, k_half = (_k(RZ, freq), _k(1.0, freq), _k(2.0, freq),
                                  _k(0.5, freq))
    k_r2d = _k(R2D, freq)

    fvert, hprim, htrue = state["fvert"], state["hprim"], state["htrue"]
    afac, fi, hi = state["afac"], state["fi"], state["hi"]
    has_f1 = fi[1] > 0.0                        # FI(2,K) > 0
    ihop_l = ihop.to(torch.long)

    # ---- 1. compact the mode list -----------------------------------------
    # Reading and writing one array: safe because INUM <= IM always, so a slot
    # is overwritten only after being read.  Slots above INUM are LEFT ALONE.
    inum = zero_i.clone()
    inm = torch.zeros((NLAYER + 1, n), dtype=torch.bool, device=dev)
    for im in range(1, NMODES + 1):
        keep = modes["hpmod"][im] - k_hmin >= 0.0
        vals = [modes[name][im].clone() for name in _MODE_FIELDS]
        it = modes["itmod"][im].clone().to(torch.long)
        inum = torch.where(keep, inum + 1, inum)
        row = torch.where(keep, inum, zero_i)   # row 0 is the sink
        _scatter(modes, row, lanes, vals, it.to(modes["itmod"].dtype))
        lay = torch.where(keep, it.clamp(0, NLAYER), zero_i)
        inm[lay, lanes] |= keep
    inm[0] = False                          # the sink row of the presence table
    # ISAV is genuinely dead in the Fortran -- assigned twice, never read --
    # so it is not carried here.

    # ---- the very-short-path test -----------------------------------------
    # MODMUF is a Fortran LAYER NUMBER (1..4), so it indexes row MODMUF-1 of
    # the /MUFS/ tables.  Reading it as a row directly puts the whole lane on
    # the wrong layer, and on this dump that silently sent 22 lanes down the
    # zero-distance path because DELMUF(3) happens to be 90.98.
    modmuf = muf["modmuf"].to(torch.long) - 1
    zero_distance = _g(muf["delmuf"], modmuf) >= _k(89.9, freq)
    nhop_mm = _g(muf["nhopmf"].to(_F32), modmuf).to(torch.long)

    saved = {name: muf[name].clone()
             for name in ("sigl", "sigu", "yfot", "yhpf", "ymuf")}
    ireset = torch.zeros(n, dtype=torch.bool, device=dev)

    long_path = ~zero_distance

    # ---- 2. insert the circuit MUF mode for any missing layer -------------
    for ily in range(1, NLAYER + 1):
        skip = ~has_f1 if ily == 2 else torch.zeros_like(has_f1)
        # `INUM >= 5` is a BREAK in the Fortran, not a CONTINUE -- but the
        # only thing after it in the loop body is the insert, so on this loop
        # the two are the same and the mask form is faithful.
        room = inum < _FULL
        want = (long_path & room & ~skip & ~inm[ily]
                & ((freq + k_eps) >= muf["ymuf"][ily - 1])
                & (ihop_l == muf["nhopmf"][ily - 1].to(torch.long)))
        inum = torch.where(want, inum + 1, inum)
        row = torch.where(want, inum, zero_i)
        _scatter(modes, row, lanes,
                 [muf[nm][ily - 1] for nm in ("delmuf", "hpmuf", "htmuf",
                                              "fvmuf", "afmuf")],
                 torch.full_like(modes["itmod"][0], ily))
        inm[ily] |= want

    # A mode already covers this hop order -> straight to the loss budget;
    # none at all -> fall back to the zero-distance construction.
    same_hop = long_path & (ihop_l == nhop_mm)
    zero_distance = zero_distance | (same_hop & (inum <= 0))
    stage3 = long_path & ~same_hop
    ireset = stage3.clone()

    # ---- 3. a per-hop MUF, above the circuit MUF for this hop order -------
    hop = ihop.to(_F32)
    psi = k_half * gcd / hop
    for jh in range(1, NLAYER + 1):
        skip = ~has_f1 if jh == 2 else torch.zeros_like(has_f1)
        eligible = (stage3 & (inum < _FULL) & ~inm[jh] & ~skip
                    & (ihop_l > muf["nhopmf"][jh - 1].to(torch.long)))
        fv = muf["fvmuf"][jh - 1]
        ht = muf["htmuf"][jh - 1]
        hpver = xlin_batch(fv, fvert, hprim, NFV)
        xhp = (hpver - ht) / k_rz
        # first pass: Martyn's theorem uncorrected
        tdel = (torch.cos(psi) - k_rz / (k_rz + hpver)) / torch.sin(psi)
        cdel = k_one / torch.sqrt(k_one + tdel * tdel)
        sphe = k_rz * cdel / (k_rz + ht)
        secp = k_one / torch.sqrt(k_one - sphe * sphe)
        esd = fv * secp
        # the spherical correction CURMUF applies, then the same again
        xfsq = esd * esd / (fi[jh - 1] * fi[jh - 1])
        sph = xfsq * (sphe * sphe) * xhp * (ht + k_two * (k_rz + ht) * xhp)
        hpnow = hpver + sph
        tdel = (torch.cos(psi) - k_rz / (k_rz + hpnow)) / torch.sin(psi)
        cdel = k_one / torch.sqrt(k_one + tdel * tdel)
        sphe = k_rz * cdel / (k_rz + ht)
        secp = k_one / torch.sqrt(k_one - sphe * sphe)
        esd = fv * secp

        # NEGATE THE FORTRAN'S TEST, do not flip the operator.  The Fortran is
        # `IF(FREQ-ESD+EPS) 240,160,160`, i.e. skip when the value is
        # NEGATIVE -- and a ray that cannot reflect gives ESD = NaN, where
        # `< 0` is false and the Fortran therefore INSERTS a NaN mode.
        # Writing `>= 0` instead would silently drop it, which is a different
        # answer on exactly the pathological lanes worth reproducing.
        want = eligible & ~(freq - esd + k_eps < 0.0)
        # the new MUF replaces the circuit one, deciles scaled to match
        oym = saved["ymuf"][jh - 1]
        new_sigl = esd * saved["sigl"][jh - 1] / oym
        new_sigu = esd * saved["sigu"][jh - 1] / oym
        k128 = _k(1.28, freq)
        for name, v in (("ymuf", esd), ("sigl", new_sigl),
                        ("yfot", esd - k128 * new_sigl),
                        ("sigu", new_sigu),
                        ("yhpf", esd + k128 * new_sigu)):
            muf[name][jh - 1] = torch.where(want, v, muf[name][jh - 1])
        inum = torch.where(want, inum + 1, inum)
        row = torch.where(want, inum, zero_i)
        _scatter(modes, row, lanes,
                 [torch.atan(tdel) * k_r2d, hpnow, ht, fv,
                  muf["afmuf"][jh - 1]],
                 torch.full_like(modes["itmod"][0], jh))
        inm[jh] |= want
        # GHOP is mutated for EVERY mode, not just the inserted one
        modes["ghop"] = torch.where(want, k_two * psi, modes["ghop"])

    # ---- 4. a zero-distance mode straight off the ionogram ----------------
    fvz = freq - _k(0.001, freq)
    ihx = torch.full((n,), NFV, dtype=torch.long, device=dev)
    hit_lt = torch.zeros(n, dtype=torch.bool, device=dev)    # label 260
    hit_eq = torch.zeros(n, dtype=torch.bool, device=dev)    # label 265
    found = torch.zeros(n, dtype=torch.bool, device=dev)
    for ih in range(2, NFV + 1):
        d = fvz - fvert[ih - 1]
        lt, eq = ~found & (d < 0.0), ~found & (d == 0.0)
        ihx = torch.where(lt | eq, torch.full_like(ihx, ih), ihx)
        hit_lt |= lt
        hit_eq |= eq
        found = found | lt | eq

    # Fortran index IHX maps to row IHX-1, so IHX-1 maps to row IHX-2.  The
    # search starts at IH=2, so IHX >= 2 and IHX-2 is always in range.
    hi_row = (ihx - 1).clamp(0, NFV - 1)
    lo_row = (ihx - 2).clamp(0, NFV - 1)
    slope = _g(fvert, hi_row) - _g(fvert, lo_row)
    safe = torch.where(slope == 0.0, torch.ones_like(slope), slope)
    slp = (fvz - _g(fvert, lo_row)) / safe

    def _lerp(tab):
        a, b = _g(tab, lo_row), _g(tab, hi_row)
        return a + (b - a) * slp

    interp = hit_lt & (slope > _k(0.01, freq))
    exact = hit_eq
    wrote = interp | exact
    zd = zero_distance
    # INUM is bumped for EVERY zero-distance lane, even one whose search fell
    # off the end of the table or whose slope was too flat to interpolate --
    # the Fortran increments before the search and only the WRITE is guarded.
    inum = torch.where(zd, inum + 1, inum)
    row = torch.where(zd & wrote, inum, zero_i)
    _scatter(modes, row, lanes,
             [torch.full_like(freq, 90.0),
              torch.where(exact, _g(hprim, hi_row), _lerp(hprim)),
              torch.where(exact, _g(htrue, hi_row), _lerp(htrue)),
              torch.where(exact, _g(fvert, hi_row), fvz),
              torch.where(exact, _g(afac, hi_row), _lerp(afac))])
    # 270: give the new mode the first layer whose height contains it
    ht_new = _g(modes["htmod"], inum.clamp(0, NMODES))
    assigned = torch.zeros(n, dtype=torch.bool, device=dev)
    for ily in range(1, NLAYER + 1):
        skip = ~has_f1 if ily == 2 else torch.zeros_like(has_f1)
        take = (zd & wrote & ~assigned & ~skip & (ht_new <= hi[ily - 1]))
        arow = torch.where(take, inum.clamp(0, NMODES), zero_i)
        modes["itmod"][arow, lanes] = torch.full_like(modes["itmod"][0], ily)
        inm[ily] |= take
        assigned = assigned | take

    # ---- clear the sink ----------------------------------------------------
    for name in _MODE_FIELDS:
        modes[name][0] = 0.0
    modes["itmod"][0] = 0

    # ---- 275: count, run the loss budget, put /MUFS/ back ------------------
    nummod = torch.zeros(n, dtype=torch.int32, device=dev)
    for ilm in range(1, NMODES + 1):
        nummod = nummod + (modes["hpmod"][ilm] >= 0.0).to(torch.int32)

    zon = regmod_batch(modes, muf, sigd, cant, sigpat, epspat, km, fi1, gyz,
                       fs1, fs2, hs, ifob13, rcnse, freq, gcd)

    for name, old in saved.items():
        muf[name] = torch.where(ireset.unsqueeze(0), old, muf[name])
    return nummod, zon
