"""The Stage-A setup chain: raw circuit inputs -> a built ``Circuit``.

Split out of :mod:`voacap_gpu.engine` (which still re-exports every name
here).  :func:`build_circuit` composes REDMAP, GEOM, SANG and the per-hour
GEOTIM / VIRTIM / TIMVAR / F2VAR / ESIND / IONSET / CURMUF chain on the host;
:func:`setup_config` is the per-SYSTEM-card row the device setup kernel takes
instead.  The two must agree field for field.
"""

import copy

import numpy as np

from voacap_gpu import system as _system
from voacap_gpu import validate as _validate
from voacap_gpu.antennas import isotropic_cant, offazim
from voacap_gpu.scalar import constants as _K
from voacap_gpu.scalar.circuit import Circuit
from voacap_gpu.scalar.curmuf import curmuf
from voacap_gpu.scalar.esind import esind
from voacap_gpu.scalar.f2var import f2var
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.geom import geom
from voacap_gpu.scalar.geotim import geotim
from voacap_gpu.scalar.ionset import ionset
from voacap_gpu.scalar.noisy import noisy
from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.scalar.sang import sang
from voacap_gpu.scalar.timvar import timvar
from voacap_gpu.scalar.virtim import virtim
from voacap_gpu.system import POLE_DEFAULT, System


_GCDLNG = F32(10000.0)          # blkdat long-path threshold, km


# ---------------------------------------------------------------------------
# METHOD / IPFLAG
# ---------------------------------------------------------------------------

def _resolve_method(method):
    """User METHOD -> (internal METHOD, MSPEC), as DECRED's METHOD card does."""
    if method == 30:            # short/long-path smoothing
        return 20, 121
    if method in (20, 25):
        return method, 0
    raise NotImplementedError(
        f"METHOD {method} is not supported; use 30 (smoothing), 20 (system "
        "performance) or 25 (all modes)")


def _resolve_model(model, coeff, system):
    """A Prop-SL ``model`` toggle -> (coeff, System).

    From Prop-SL / PropServer ``voacapx.dat`` vs ``icepacx.dat``:

    * ``"voacap"`` -> CCIR coefficients + FPROB ``(1, 1, 1, 0.0)`` -- exactly
      the validated golden decks.  Supported.
    * ``"icepac"`` -> URSI coefficients + FPROB ``(1, 1, 1, 0.70)``.

    FPROB is the PSC critical-frequency multiplier, consumed by the ported
    EF1VAR/ESIND/F2VAR chain; PSC(4)=0.70 turns sporadic E ON, which genuinely
    changes the answer.  BOTH arms are validated against voacapl -- the same
    binary Prop-SL runs for ICEPAC.  The ICEPAC oracle deck
    (``golden/dumps/luffy_icepac/tanbel_m07_s160_icepac``, the TANGIER-BELGRADE
    circuit under URSI + FPROB 0.70) reproduces at 99.31% bit-exact on the
    setup chain with Es present in all 24 hours, where its CCIR twin has none;
    ``tests/test_engine.py`` pins it.

    A ``model`` never silently overrides an explicit choice: ``coeff`` other
    than the model's (and other than the default 'CCIR', which is treated as
    "not given" because ``predict`` cannot tell it from an explicit 'CCIR'),
    or a ``system.fprob`` other than the default ``(1, 1, 1, 0)`` and the
    model's own, raises ``ValueError``.  The returned System is a shallow copy
    with ``fprob`` set (a copy, not ``dataclasses.replace``, so a deprecated
    multipath alias is not re-warned).
    """
    if model is None:
        return coeff, system
    m = model.lower()
    if m == "voacap":
        mcoeff, mfprob = "CCIR", (1.0, 1.0, 1.0, 0.0)
    elif m == "icepac":
        mcoeff, mfprob = "URSI", (1.0, 1.0, 1.0, 0.70)
    else:
        raise NotImplementedError(
            f"model={model!r}: only 'voacap' and 'icepac' are supported")
    _validate.check_model_conflict(m, mcoeff, mfprob, coeff, system)
    s = copy.copy(system) if system is not None else System()
    s.fprob = mfprob
    return mcoeff, s


def _ipflag(method, mspec, gcdkm):
    """The IPFLAG hfmufs.for hands LUFFY on the system-performance arm.

    ``method``/``mspec`` are the INTERNAL pair from :func:`_resolve_method`.
    """
    meth = 30 if mspec == 121 else method       # hfmufs' `meth`
    if method in (22, 25) or mspec == 125:
        return 100
    ipfg = 100
    if mspec != 121 and F32(gcdkm) >= _GCDLNG:
        ipfg = 200
    if meth == 30 and F32(gcdkm) >= _GCDLNG:
        ipfg = 200
    return ipfg


# ---------------------------------------------------------------------------
# The setup chain
# ---------------------------------------------------------------------------

def _install_geometry(circ, g, tlatd, tlongd, rlongd, amind):
    """GEOM result + the loose /DON/ scalars into a fresh Circuit's state."""
    lay = circ.layers
    lay.km = g.km
    lay.iedp = -1
    for i in range(1, g.km + 1):
        lay.clat[i] = g.clat[i - 1]
        lay.clong[i] = g.clong[i - 1]
        lay.glat[i] = g.glat[i - 1]
        lay.rd[i] = g.rd[i - 1]
        lay.gyz[i] = g.gyz[i - 1]
        lay.gmdip[i] = g.gmdip[i - 1]
        lay.sigpat[i] = g.sigpat[i - 1]
        lay.epspat[i] = g.epspat[i - 1]
    circ.gcd = g.gcd
    circ.gcdkm = g.gcdkm
    circ.amind = amind
    # /DON/ endpoint coordinates (radians), with GEOM's near-pole longitude
    # zeroing (geom.for:52-53).  GEOM may have nudged the receiver latitude.
    tl = F32(0.0) if abs(F32(tlatd)) > F32(89.9) else F32(tlongd) * _K.D2R
    rl = F32(0.0) if abs(F32(g.rlatd)) > F32(89.9) else F32(rlongd) * _K.D2R
    circ.tlong = tl
    circ.rlong = rl
    circ.rlongd = F32(rlongd)
    circ.rlat = F32(g.rlatd) * _K.D2R
    circ.rays.nang = sang(g.gcdkm, amind)


def _setup_hour(circ, jt):
    """Run one prediction hour's ionosphere/MUF build into ``circ`` in place.

    Fills /TIME/, the per-area layer parameters, IONSET's packing and CURMUF's
    /MUFS/, and the FREL(12)=MUF column -- everything the LUFFY kernel reads as
    input.  Leaves /SON/ untouched, so a Circuit reused across hours reproduces
    the Fortran COMMON carry-over the replay depends on.
    """
    st = circ.iono
    lay = circ.layers
    geotim(lay, circ.time, jt, circ.tlong, circ.rlong)
    ab = virtim(st, circ.time.gmt)
    timvar(lay, st, ab, circ.time.gmt)
    f2var(lay, st, ab)
    esind(lay, st, ab)
    ionset(lay)
    curmuf(lay, circ.muf, st, circ.gcdkm, circ.gcd, circ.amind,
           F32(lay.ssn), it=circ.time.it, iedp=lay.iedp, info=circ.info)
    circ.frq.frel[12] = circ.muf.allmuf[circ.time.it]


def build_circuit(tx, rx, month, ssn, hour, freqs, *, method=30,
                  system=None, cant=None, coeff="CCIR", pole=POLE_DEFAULT,
                  npsl=0, tx_bearing=None, rx_bearing=None, iono=None,
                  circ=None):
    """One fully-built Stage-A ``Circuit`` for ``hour`` (1-based UTC).

    ``tx``/``rx`` are (lat_deg, lon_deg), south/west negative.  Returns
    ``(circ, ipflag)``.  ``tx_bearing``/``rx_bearing`` (true azimuth, degrees,
    of each antenna main beam) set OFFAZIM against GEOM's path bearing ONLY on
    the isotropic /cantenna/ built here when ``cant`` is None; a ``cant``
    passed in keeps the OFFAZIM :func:`antcalc_cant` gave it.  ``None`` keeps
    the beam along the path (OFFAZIM 0).  ``iono`` (a REDMAP
    ``IonoState``) and ``circ`` (a Circuit to reuse for COMMON carry-over) may
    be supplied to avoid rebuilding them across hours.

    Inputs are checked as in :func:`predict` (``ValueError`` for an
    out-of-range frequency, month, hour, SSN, coordinate, coeff or System
    value; see :mod:`voacap_gpu.validate`).
    """
    system = system or System()
    _validate.check_system(system)
    psc, amind = system._psc()
    imeth, mspec = _resolve_method(method)
    freqs = [F32(f) for f in _validate.check_freqs(freqs)]
    month = _validate.check_month(month)
    hour = _validate.check_hour(hour)
    _validate.check_ssn(ssn)
    tx = _validate.check_point("tx", tx)         # longitude wrapped to ±180
    rx = _validate.check_point("rx", rx)
    _validate.check_coeff(coeff)

    st = iono if iono is not None else redmap(_system._COEFFS_DIR, coeff,
                                              int(month), F32(ssn))

    if circ is None:
        # geometry is fixed for the path; build it once
        tlatd, tlongd = tx
        rlatd, rlongd = rx
        wld = lambda a, b: noisy(st, 7, a, b)    # NOISY(7) land/sea map
        g = geom(tlatd, tlongd, rlatd, rlongd, int(npsl), pole[0], pole[1],
                 wld)
        if cant is None:
            # OFFAZIM from the beam headings against the path bearing at each
            # end -- only for the isotropic cant built here, where it is inert.
            off_tx = (offazim(g.btrd, tx_bearing) if tx_bearing is not None
                      else 0.0)
            off_rx = (offazim(g.brtd, rx_bearing) if rx_bearing is not None
                      else 0.0)
            cant = isotropic_cant(system.power_kw, system.tx_gos,
                                  system.rx_gos, off_tx, off_rx)
        # A caller's cant keeps its own OFFAZIM.  antcalc_cant already
        # resolved it (DAZEL0 + ANTCALC's single wrap) and installed the
        # type-13 azimuth slice AT that value; recomputing it here from
        # GEOM's bearing (and a % wrap) would leave cant.offazim describing a
        # different direction from the table actually installed.
        # tx_bearing / rx_bearing are then unused for that cant.

        circ = Circuit(iono=st)
        circ.cant = cant
        circ.muf.xluf[:] = F32(-1.0)           # SETOUT's preset
        circ.layers.month = int(month)
        circ.layers.ssn = F32(ssn)
        circ.layers.psc[:] = psc
        circ.method = imeth
        circ.mspec = mspec
        circ.noise = int(system.man_made_noise)
        circ.lufp = int(system.required_reliability)
        circ.rsn = F32(system.required_snr)
        circ.dmp = F32(system.dmp)             # DMP, max multipath delay, ms
        circ.pmp = F32(system.pmp)             # PMP, multipath power tol., dB
        circ.want_service_prob = imeth in (20, 25, 30)
        circ.time.itim = 1                     # hours are UTC (GMT)
        _install_geometry(circ, g, tlatd, tlongd, rlongd, amind)
        # user frequency list (FREL 1..nf), the DECRED flag, MUF column later
        for i, f in enumerate(freqs, start=1):
            circ.frq.frel[i] = f
        circ.frq.frel[14] = F32(1.0)

    _setup_hour(circ, int(hour))
    ipflag = _ipflag(imeth, mspec, circ.gcdkm)
    return circ, ipflag


# ---------------------------------------------------------------------------
# The device-setup path: the same inputs, none of the host work
# ---------------------------------------------------------------------------

def setup_config(system=None, *, method=30, freqs=(), pole=POLE_DEFAULT,
                 info=0):
    """A ``SETUP_CONFIG_DTYPE`` row: everything :func:`build_circuit` copies
    into a Circuit that does NOT vary per circuit-hour.

    The device setup kernel (``cuda/kernels/setup.cuh``) takes one of these per
    distinct SYSTEM card and 32 bytes per circuit-hour; between them they are
    exactly ``build_circuit``'s inputs.  This function and ``build_circuit``
    must agree field for field -- ``tests/cuda/test_setup_batch.py`` is what
    holds them to it -- so anything added to one belongs in the other.
    The same System / frequency checks as ``build_circuit`` apply.
    """
    from voacap_gpu.cuda.driver import SETUP_CONFIG_DTYPE

    system = system or System()
    _validate.check_system(system)
    psc, amind = system._psc()
    imeth, mspec = _resolve_method(method)
    freqs = [F32(f) for f in _validate.check_freqs(freqs)]

    cfg = np.zeros((), SETUP_CONFIG_DTYPE)
    cfg["pole_lat"] = F32(pole[0])
    cfg["pole_lon"] = F32(pole[1])
    cfg["amind"] = amind
    cfg["rsn"] = F32(system.required_snr)
    # DMP is the max multipath time delay (ms), PMP the multipath power
    # tolerance (dB) -- decred.for:245-259, scalar/circuit.py.  Both come from
    # System's resolved dmp/pmp, exactly as build_circuit takes them.
    cfg["dmp"] = F32(system.dmp)
    cfg["pmp"] = F32(system.pmp)
    cfg["psc"] = psc
    cfg["lufp"] = int(system.required_reliability)
    cfg["method"] = imeth
    cfg["noise"] = int(system.man_made_noise)
    cfg["mspec"] = mspec
    cfg["nyear_is_2089"] = 0            # Circuit's default NYEAR is ""
    cfg["want_service_prob"] = int(imeth in (20, 25, 30))
    cfg["info"] = int(info)
    cfg["itim"] = 1                     # hours are UTC; see setup.cuh on ITIM
    frel = np.zeros(30, np.float32)
    for i, f in enumerate(freqs, start=1):
        frel[i] = f
    frel[14] = F32(1.0)                 # DECRED's IFREQ flag
    cfg["frel"] = frel
    return cfg
