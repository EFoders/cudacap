"""Stage B (PyTorch, batched) mode accumulator — ALLMODES.

/ZON/ holds at most seven modes for ONE hop distance; the hop sweep visits
several hop counts and then the sporadic-E pair, and ALLMODES is what makes
the whole frequency's worth of modes readable afterwards.  It appends every
live /ZON/ slot into the twenty-slot /allMODE/ arrays behind a running cursor.

THERE IS NO ARITHMETIC HERE AT ALL -- only assignment and index selection --
so the port is bit-exact by construction on both devices, and the tests assert
exactly that.  MEASURED: 0 mismatches over the whole oracle dump replayed
eight ways at once, 249 calls x 8 lanes x 468 columns plus NMMOD and FREQ, on
CPU and on CUDA.  What there IS, and the reason this is the awkward one, is a
PER-LANE APPEND CURSOR: NMMOD is both the running count and the write index,
it accumulates across calls, and in a batch every lane is at a different slot.

SO THE WRITE IS A SCATTER WITH ROW 0 AS A SINK, the FDIST pattern.  A lane
whose /ZON/ slot is empty (HP <= 0) must not append; rather than mask the
store it scatters into row 0, which nothing reads -- the /allMODE/ slots are
1..20 and row 0 is the 1-based padding.  The sink is CLEARED before returning,
because a lane that never appends still writes there and leaving it would make
the returned arrays depend on who else was in the batch.

IFLG IS PER LANE HERE, unlike Stage A where it is a call-site constant.  It
costs almost nothing -- IST/LST become two ``torch.where`` results -- and it
buys the thing the batched driver actually needs, because the hop loop's
bounds are per-lane: in LUFFY the number of ``allmodes(zon, allm, ipfg, hop)``
calls is ``ihstp - ihsrt + 1`` and no two grid points agree on that.  The
``active`` mask is the other half of the same need: a lane past its own last
hop sits out the call entirely and its state must not move.

FOUR IFLG ARMS, and three things in the IFLG=1 arm look like bugs and are not:

======  ==============================================================
IFLG    behaviour
======  ==============================================================
0       reset -- the 20 slots back to their "no mode" markers, and
        FREQ = FVAL
1       slot 1 only.  It ALSO writes ``HP(2) = HP(1)`` back into /ZON/,
        parks a 32000 sentinel in ``zTLOSS(2)`` -- slot 2 absolutely,
        not relative to the cursor -- and writes NMODE/B into slot
        ``NMMOD+2`` rather than the cursor, reserving the next-but-one
        slot for a partner mode a later call fills in
2       slots 4-5, the sporadic-E pair ESMOD writes
other   slots 1-6, the regular modes
======  ==============================================================

FREQ LIVES IN THE STATE DICT rather than being returned.  Stage A returns it
because /FRQ/ is a different common and the scalar caller adopts it; with a
per-lane IFLG "the return value is only meaningful on a reset call" stops
being expressible, so ``allm["freq"]`` is carried and updated as
``where(reset, fval, freq)`` -- which is exactly what ``FREQ = FVAL`` inside
the IFLG=0 arm means.

OVERFLOW IS REFUSED, NOT WRAPPED.  NMMOD is never bound-checked in the Fortran
and accumulates across every call between resets, so a long enough sweep walks
off the end of the arrays and into whatever follows them in the common block.
Stage A raises; so does this, on ``.any()``, which costs one device sync per
call -- affordable, since ALLMODES is called once per hop and does no work.
The check is planned BEFORE any write, so a batch that would overflow leaves
the state untouched rather than half-updated.

The MSPEC=125 blocks throughout allmodes.for are commented out in the source
(an area-coverage variant), so nothing from them is ported.
"""

import torch

from voacap_gpu.stage_a.allmodestate import FIELDS, NSLOT
from voacap_gpu.stage_b.profile import _k

_F32 = torch.float32

NZON = 7                        # /ZON/ mode slots, 1-based

_NO_TLOSS = 99999.0
_NO_DECILE = 999.0
_NO_HP = -1.0
_MARKER_TLOSS = 32000.0


class AllModesOverflow(IndexError):
    """More than 20 modes accumulated -- the Fortran would write past the end.

    Raised if ANY lane would overflow.  The batch is planned before it is
    written, so the state is unchanged when this fires.
    """


def new_allmode_state(n, device="cpu"):
    """A fresh /allMODE/ for `n` lanes, at its BLOCK DATA values.

    The 22 float arrays and NMODE are (21, N), 1-based in the slot index with
    row 0 as the scatter sink.  ``nmmod`` and ``nrel`` start at **1, not 0**
    (``blkdat.for:181  DATA NMMOD/1/, NREL/1/``) -- see allmodestate.py for
    why that is observable rather than cosmetic.  ``freq`` is /FRQ/ FREQ,
    carried here because the IFLG=0 arm assigns it.
    """
    z = torch.zeros((NSLOT + 1, n), dtype=_F32, device=device)
    st = {name: z.clone() for name in FIELDS}
    st["nmode"] = torch.zeros((NSLOT + 1, n), dtype=torch.int32,
                              device=device)
    st["nmmod"] = torch.ones(n, dtype=torch.long, device=device)
    st["nrel"] = torch.ones(n, dtype=torch.long, device=device)
    st["freq"] = torch.zeros(n, dtype=_F32, device=device)
    return st


def new_zon_state(n, device="cpu"):
    """A zeroed /ZON/ for `n` lanes: (8, N) per field, 1-based in the slot."""
    z = torch.zeros((NZON + 1, n), dtype=_F32, device=device)
    zon = {name: z.clone() for name in FIELDS}
    zon["nmode"] = torch.zeros((NZON + 1, n), dtype=torch.int32,
                               device=device)
    return zon


def _lanes_long(v, n, dev):
    if torch.is_tensor(v):
        return v.to(device=dev, dtype=torch.long)
    return torch.full((n,), int(v), dtype=torch.long, device=dev)


def _lanes_f32(v, n, dev, like):
    if torch.is_tensor(v):
        return v.to(device=dev, dtype=_F32)
    return _k(v, like).expand(n)


def allmodes_batch(zon, allm, iflg, fval=0.0, active=None):
    """Batched ALLMODES.  MUTATES `allm` (and `zon["hp"]`), and returns `allm`.

    zon:  dict of (8, N) /ZON/ arrays, 1-based in the slot index; ``nmode`` is
          int32.  ``new_zon_state`` builds one.
    allm: dict from ``new_allmode_state``, the accumulator.  It is updated IN
          PLACE rather than rebuilt -- it is a COMMON block, the hop sweep
          calls this several times per frequency, and copying 23 x 21 x N
          tensors per call would be the entire cost of a routine that does no
          arithmetic.
    iflg: int, or an (N,) integer tensor for a per-lane arm.
    fval: float or (N,) tensor.  The reset arm assigns it to ``allm["freq"]``;
          the other arms ignore it, exactly as the Fortran does (LUFFY passes
          HOP there, which nothing reads).
    active: (N,) bool, or None for "every lane makes this call".  A False lane
          is untouched -- that is how a lane past its own last hop sits out.

    ``zon["hp"][2]`` is written back on the IFLG=1 lanes, matching the
    Fortran's ``hp(2)=hp(1)``; pass a /ZON/ you are willing to have mutated.
    """
    hp = zon["hp"]
    n = hp.shape[1]
    dev = hp.device
    lanes = torch.arange(n, device=dev)
    sink = torch.zeros(n, dtype=torch.long, device=dev)

    iflg_t = _lanes_long(iflg, n, dev)
    act = (torch.ones(n, dtype=torch.bool, device=dev) if active is None
           else active.to(device=dev, dtype=torch.bool))
    reset = act & (iflg_t == 0)
    app = act & (iflg_t != 0)
    arm1 = app & (iflg_t == 1)

    nmmod = allm["nmmod"]
    zeros_l = torch.zeros_like(nmmod)

    # ---- plan first: the overflow check has to fire before anything moves --
    # IST/LST per lane.  IFLG=1 takes slot 1 alone, IFLG=2 the Es pair 4-5,
    # anything else the six regular slots.
    one = torch.ones_like(iflg_t)
    ist = torch.where(iflg_t == 2, one * 4, one)
    lst = torch.where(iflg_t == 1, one,
                      torch.where(iflg_t == 2, one * 5, one * 6))

    cur = torch.where(reset, zeros_l, nmmod)     # the reset zeroes the cursor
    plan = []
    csr = cur
    for i in range(1, NZON):                     # DO 10 i = ist, lst, IST/LST
        # in 1..6 -- slot 7 is never appended by any arm
        copy = app & (i >= ist) & (i <= lst) & (hp[i] > 0.0)
        csr = torch.where(copy, csr + 1, csr)
        plan.append((i, copy, csr))
    reserved = cur + 2                           # the IFLG=1 next-but-one slot
    over = (csr > NSLOT) | (arm1 & (reserved > NSLOT))
    if bool(over.any()):
        worst = int(torch.maximum(csr, torch.where(arm1, reserved, csr)).max())
        raise AllModesOverflow(
            f"/allMODE/ slot {worst} is past the end of {NSLOT} on "
            f"{int(over.sum())} lane(s); the Fortran writes into whatever "
            f"follows the common block")

    # ---- IFLG = 0: back to the no-mode markers, and FREQ = FVAL ------------
    r = reset.unsqueeze(0)
    for name, value in (("tloss", _NO_TLOSS), ("tllow", _NO_DECILE),
                        ("tlhgh", _NO_DECILE), ("hp", _NO_HP)):
        arr = allm[name]
        arr[1:].copy_(torch.where(r, _k(value, arr), arr[1:]))
    allm["freq"].copy_(torch.where(reset, _lanes_f32(fval, n, dev, hp),
                                   allm["freq"]))

    # ---- IFLG = 1's three oddities, in the Fortran's order -----------------
    # HP(2)=HP(1) writes back into /ZON/; the 32000 goes into slot 2
    # ABSOLUTELY, not at the cursor; and NMODE/B go into NMMOD+2, reserving
    # the next-but-one slot for the partner mode a later call appends.  The
    # append loop below can and does overwrite slot 2 afterwards -- that is
    # the Fortran's order too.
    hp[2].copy_(torch.where(arm1, hp[1], hp[2]))
    tl = allm["tloss"]
    tl[2].copy_(torch.where(arm1, _k(_MARKER_TLOSS, tl), tl[2]))
    row = torch.where(arm1, reserved, sink)
    allm["nmode"][row, lanes] = zon["nmode"][2]
    allm["b"][row, lanes] = zon["b"][2]

    # ---- DO 10: append every live slot at the cursor ------------------------
    for i, copy, csr_i in plan:
        row = torch.where(copy, csr_i, sink)     # row 0 is the sink
        for name in FIELDS:
            allm[name][row, lanes] = zon[name][i]
        allm["nmode"][row, lanes] = zon["nmode"][i]

    nmmod.copy_(csr)

    # Clear the sink.  Lanes with nothing to append scatter into row 0 on
    # every call, so leaving it would make the arrays depend on batch
    # composition; nothing reads row 0 (the slots are 1..20), but "nothing
    # reads it" is not a property a test can hold on to.
    for name in FIELDS:
        allm[name][0] = 0.0
    allm["nmode"][0] = 0
    return allm
