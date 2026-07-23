"""Stage B (PyTorch, batched) service probability — SERPRB.

RELBIL asks "how often does this circuit work?" given the model.  SERPRB asks
the harder question: "how confident are we in that answer?"  It rebuilds the
same normal-distribution test but widens the spread by everything uncertain
about the *prediction* rather than about the hour -- the system-loss prediction
error ADS/SUS/SLS, the noise prediction error SIGM/SYGU/SYGL, and a flat 2 dB
for the required S/N itself.

Which tail binds depends on the required reliability.  Above 50% the bad case
is a bad hour (noise upper decile crossed with the signal lower decile) and XLH
is -1, so the reference level moves DOWN from the median; below 50% it is the
other way round and XLH is +1.  LUFP IS PER-LANE HERE, so both arms are
evaluated and selected -- they differ in four of the five quantities, and only
the sign XLH is shared structure.

Batching is the mode axis as a fixed twenty-iteration Python loop and the lane
axis as the tensor axis, the same shape as RELBIL.  Nothing carries between
modes except the final argmax, whose tie-break is FIRST-WINS (the Fortran's
``.GT.``).

Preserved as found, all three of which look wrong:

* the source's own header says it "has not been revised to use summed signal",
  i.e. it still works per mode off /allMODE/ where RELBIL now works off the
  combined distribution, then takes the maximum.  That inconsistency is real.
* ``SPR`` in /TON/ is zeroed on entry and never touched again.
* ``D90S`` is set equal to ``D10S`` rather than to an upper decile of its own.
* a mode is dead at HP <= 70 km here but at HP <= 0 in MPATH, which runs
  immediately afterwards on the same table.  Both thresholds are correct.

Dead modes get SPRO 0.001 and leave D10SA/D50SA at zero, so a lane whose best
SPRO belongs to a dead mode reports a zero decile pair.  Here the full
arithmetic is evaluated for every slot and then discarded by the HP test; the
denominator ``D10SA`` in the live arm is ``D50SA + sqrt(... + DR*DR)`` with
DR = 2, so it is never smaller than 2 and the divide needs no guard.

float32; SQRT is the one libm call and carries the [[f32-libm-noise]] floor.
FNORML is pure polynomial.  ``1. - FNORML(Z)`` is spelled out for the reason
in ``stage_b.normal``'s docstring.
"""

import numpy as np
import torch

from voacap_gpu.stage_a.constants import DCL
from voacap_gpu.stage_b.normal import fnorml_batch
from voacap_gpu.stage_b.profile import _k

_F32 = torch.float32
NM = 20

# the same standard-normal table RELBIL has, as a separate DATA statement in
# the Fortran; SERPRB uses TME(ITM) directly where RELBIL takes a ratio.
_TME = np.array([0.0, 0.1257, 0.2533, 0.3853, 0.5244, 0.6745, 0.8416,
                 1.0364, 1.2815, 1.6449], dtype=np.float32)
_DR = float(np.float32(2.0))            # prediction error in the required S/N
_NO_MODE_HEIGHT = float(np.float32(70.0))
_DEAD = float(np.float32(0.001))


class NoModesError(ValueError):
    """SERPRB with no modes reads uninitialised locals in the Fortran."""


def serprb_batch(modes, nmmod, du, dl, sigm, sygu, sygl, ads, sus, sls,
                 rsn, lufp):
    """Batched SERPRB.  Returns a dict of tensors.

    modes: dict of (NM+1, N) tensors 1-based in the mode axis -- ``hp``,
        ``sn``, ``tllow``, ``tlhgh`` are read, and ``spro`` may be supplied as
        the incoming /allMODE/ column so the slots above NMMOD keep whatever
        they held (SERPRB writes only 1..NMMOD).  A /allMODE/ state dict as
        ``stage_b.allmodes`` builds one can be handed straight in; extra keys
        are ignored.
    nmmod: (N,) int mode count.  Every lane must have at least one -- see
        below.
    du/dl/sigm/sygu/sygl: (N,) /ANOIS/ noise deciles and prediction errors.
    ads/sus/sls: (N,) /TON/ system-loss prediction errors.
    rsn: (N,) required S/N.  lufp: (N,) int, required reliability in percent.

    Returns ``spro`` (NM+1, N), ``sprob`` (N,), ``d10s``/``d50s``/``d90s`` (N,)
    and ``spr`` (N,), which is the /TON/ scalar SERPRB zeroes.

    Raises ``NoModesError`` if any lane has NMMOD <= 0: both Fortran loops
    would skip, leaving IMAX at 1 and D10S/D50S read from uninitialised local
    arrays.  That is undefined behaviour, not a branch, so it is refused rather
    than reproduced -- and refusing needs a device sync, which is why the check
    is a single ``.any()`` over the batch.
    """
    hp, sn = modes["hp"], modes["sn"]
    tllow, tlhgh = modes["tllow"], modes["tlhgh"]
    n = rsn.shape[0]
    dev = rsn.device
    nmmod = nmmod.to(torch.long)
    lufp = lufp.to(torch.long)

    if bool((nmmod <= 0).any()):
        bad = int((nmmod <= 0).sum())
        raise NoModesError(
            f"SERPRB needs at least one mode; {bad} lane(s) have NMMOD <= 0, "
            "where the Fortran reads uninitialised D10SA/D50SA locals")

    k = {name: _k(v, rsn) for name, v in (
        ("one", 1.0), ("dcl", DCL), ("dr", _DR), ("dead", _DEAD),
        ("no_mode_h", _NO_MODE_HEIGHT), ("plus", 1.0), ("minus", -1.0))}
    zero = torch.zeros(n, dtype=_F32, device=dev)

    itm = torch.clamp(
        torch.div(torch.abs(lufp - 50), 5, rounding_mode="floor") + 1, max=10)
    tmx = torch.as_tensor(_TME, dtype=_F32, device=dev)[itm - 1]

    upper = lufp >= 50
    # both arms in full: LUFP is a per-lane input, so which tail binds is a
    # data-dependent branch like any other.
    xlh = torch.where(upper, k["minus"], k["plus"])
    dn = tmx * torch.where(upper, du, dl) / k["dcl"]        # the noise
    dso = tmx * torch.where(upper, sus, sls)                # prediction errors
    dno = tmx * torch.where(upper, sygu, sygl) / k["dcl"]
    # the prediction-error term does not depend on the mode
    dpred = torch.sqrt(sigm * sigm + ads * ads + dno * dno + dso * dso
                       + k["dr"] * k["dr"])

    spro = modes.get("spro")
    spro = torch.zeros_like(hp) if spro is None else spro.clone()
    d10sa = torch.zeros_like(hp)
    d50sa = torch.zeros_like(hp)

    for im in range(1, NM + 1):                             # DO 145
        inloop = im <= nmmod
        dead = hp[im] <= k["no_mode_h"]
        live = inloop & ~dead
        ds = tmx * torch.where(upper, tllow[im], tlhgh[im]) / k["dcl"]
        d50 = torch.sqrt(dn * dn + ds * ds)
        d10 = d50 + dpred
        d50 = sn[im] + xlh * d50
        # D10 >= DR = 2 always, so this divide needs no guard
        spro_live = k["one"] - fnorml_batch((rsn - d50) / d10)
        spro[im] = torch.where(inloop,
                               torch.where(dead, k["dead"], spro_live),
                               spro[im])
        # a dead mode leaves both deciles at zero, as the Fortran's explicit
        # d10sa(im)=0. / d50sa(im)=0. does
        d10sa[im] = torch.where(live, d10, zero)
        d50sa[im] = torch.where(live, d50, zero)

    # DO 150 -- AMXX is dead code in the source; IMAX starts at 1 and the test
    # is a strict .GT., so the FIRST slot holding the maximum wins.
    imax = torch.ones(n, dtype=torch.long, device=dev)
    best = spro[1].clone()
    for i in range(2, NM + 1):
        take = (i <= nmmod) & (spro[i] > best)
        imax = torch.where(take, torch.full_like(imax, i), imax)
        best = torch.where(take, spro[i], best)

    d10s = torch.gather(d10sa, 0, imax.unsqueeze(0)).squeeze(0)
    return {"spro": spro, "sprob": best,
            "d10s": d10s,
            "d50s": torch.gather(d50sa, 0, imax.unsqueeze(0)).squeeze(0),
            "d90s": d10s,                   # not a separate decile
            "spr": zero.clone(),            # /TON/ SPR, zeroed and never read
            "imax": imax}
