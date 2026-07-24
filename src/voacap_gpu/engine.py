"""The public VOACAP-GPU endpoint: raw circuit parameters in, results out.

Everything below this module is validated line-against-line with the Fortran;
nothing above it existed.  ``predict`` composes the Stage-A setup routines --
REDMAP, GEOM, SANG, and the per-hour GEOTIM / VIRTIM / TIMVAR / F2VAR / ESIND /
IONSET / CURMUF chain, in exactly the order ``voacapw/hfmufs.for`` drives them
-- into a fully-built Stage-A ``Circuit`` for each requested hour, then hands
those circuits to the production CUDA LUFFY kernel (``cuda/driver.py``) and
distils the /SON/ table into a structured :class:`Results`.

WHAT IS SUPPORTED
-----------------
* METHOD 30 (short/long-path smoothing -- the default), 20 (system
  performance, short path forced past the long-path threshold) and 25
  (all-modes, short path forced).  These are the three regimes the LUFFY
  replay validates.  Any other METHOD raises ``NotImplementedError``.
* Up to 11 frequencies per call (the FREL card's capacity).
* Antennas: an isotropic 0 dBi pattern (the default), or a vendored *type 14*
  ``.VOA`` sample file (30 frequencies x 91 elevations) read through
  ``scalar/readant.py`` and installed exactly as READANT + ANTCALC would leave
  it.  A type-14 pattern is azimuth-independent, so the beam is taken as
  pointing along the great-circle path (OFFAZIM 0).  Antenna types 1-10, 12
  (method-of-moments synthesis) and 13 (azimuth-dependent, needs a real beam
  heading) raise ``NotImplementedError`` rather than returning wrong gains.
* UTC hours 1..24.  ``hours`` given in local mean time at the transmitter is
  not exposed here (the Fortran's ITIM<0 path); pass UTC.

THE GEOMAGNETIC POLE
--------------------
GEOM's geomagnetic latitude needs the north geomagnetic pole.  The vendored
``database/north_pole.txt`` carries 79.5N, but ``set_magnetic_pole.for``'s
*runtime fallback* -- used when that file is not on the run path, which is how
the oracle LUFFY decks were captured -- is 78.5N, -69.0.  The default here is
that fallback, because it is what the validation oracle used and what
reproduces the captured entry state bit-for-bit; pass ``pole=(79.5, -69.0)`` to
match a modern voacapl install that finds the file.

VALIDATION
----------
``tests/test_engine.py`` builds circuits from three decks' raw inputs
(short-path lonpar, smoothing latok, long-path lonakl) and shows the built
geometry + per-area ionosphere is bit-exact to the captured LUFFY entry state
at the documented libm floor, then runs them through the kernel and shows the
/SON/ table matches the deck's captured output.
"""

import pathlib
from dataclasses import dataclass, field

import numpy as np

from voacap_gpu.scalar import constants as _K
from voacap_gpu.scalar.antennastate import CantennaState
from voacap_gpu.scalar.circuit import Circuit
from voacap_gpu.scalar.curmuf import curmuf
from voacap_gpu.scalar.esind import esind
from voacap_gpu.scalar.f2var import f2var
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.geom import geom
from voacap_gpu.scalar.geotim import geotim
from voacap_gpu.scalar.ionset import ionset
from voacap_gpu.scalar.noisy import noisy
from voacap_gpu.scalar.readant import UnsupportedAntennaType, read_antenna
from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.scalar.sang import sang
from voacap_gpu.scalar.timvar import timvar
from voacap_gpu.scalar.virtim import virtim

# The reference program's runtime-fallback geomagnetic north pole -- see the
# module docstring on why this, and not the file's 79.5, is the default.
POLE_DEFAULT = (78.5, -69.0)

# Vendored CCIR/URSI coefficient directory (the one the tier-B dumps used).
_COEFFS_DIR = (pathlib.Path(__file__).resolve().parents[2] / "reference"
               / "voacapl" / "voacapl" / "itshfbc" / "coeffs")

_GCDLNG = F32(10000.0)          # blkdat long-path threshold, km


# ---------------------------------------------------------------------------
# System / SYSTEM-card parameters
# ---------------------------------------------------------------------------

@dataclass
class System:
    """The SYSTEM and FPROB card values, defaulted to the golden decks'.

    ``fprob`` are the FPROB multipliers PSC applied to (E, F1, F2, Es) critical
    frequencies; the deck value ``(1, 1, 1, 0)`` disables sporadic E.  The
    stock VOACAP default (no FPROB card) is ``(1, 1, 1, 0.7)``.
    """

    required_snr: float = 73.0          # RSN, dB (S/N at the reliability pctile)
    min_takeoff_deg: float = 0.1        # AMIND (<=0 -> 3.0, as DECRED does)
    required_reliability: int = 90      # LUFP, percent
    max_delay_ms: float = 3.0           # PMP (multipath power-spread limit, dB)
    max_spread_db: float = 0.1          # DMP (multipath delay limit, ms)
    man_made_noise: float = 145.0       # NOISE, -dBW at 3 MHz (residential 145)
    power_kw: float = 1.0               # transmitter power (also on ANTENNA)
    fprob: tuple = (1.0, 1.0, 1.0, 0.0)

    def _psc(self):
        amind = self.min_takeoff_deg if self.min_takeoff_deg > 0.0 else 3.0
        psc = np.array([0.0, *self.fprob], dtype=np.float32)
        return psc, F32(amind)


# ---------------------------------------------------------------------------
# Antennas
# ---------------------------------------------------------------------------

def isotropic_cant(power_kw=1.0) -> CantennaState:
    """A two-entry /cantenna/: 0 dBi isotropic TX (1) and RX (2).

    ARRAY is zero (0 dBi at every frequency and elevation) and AEFF zero;
    PWRDBA carries the transmitter power.  Enough for a sane, gain-neutral
    prediction and for the pure-Python smoke path.
    """
    cant = CantennaState()
    for iat, itr in ((1, 1), (2, 2)):
        cant.iats[iat] = itr
        cant.xfqs[iat] = F32(0.0)
        cant.xfqe[iat] = F32(30.0)
        cant.offazim[iat] = F32(0.0)
        cant.numants = max(cant.numants, iat)
    pkw = max(float(power_kw), 1e-6)
    cant.pwrkw[1] = F32(power_kw)
    cant.pwrdba[1] = F32(30.0 + 10.0 * np.log10(pkw))
    return cant


def cant_from_files(tx_path, rx_path, power_kw=1.0) -> CantennaState:
    """Build a /cantenna/ from two type-14 ``.VOA`` sample files (TX, RX).

    Installed exactly as READANT + ANTCALC would (``CantennaState.load_p2p``);
    the beam is taken as pointing along the path (OFFAZIM 0), which is exact for
    a type-14 azimuth-independent pattern.  Raises for non-tabulated types.
    """
    cant = CantennaState()
    for iat, itr, path in ((1, 1, tx_path), (2, 2, rx_path)):
        try:
            ant = read_antenna(path)
        except UnsupportedAntennaType as e:
            raise NotImplementedError(str(e)) from e
        if ant.jant != 14:
            raise NotImplementedError(
                f"antenna file {path!r} is type {ant.jant}; only type 14 "
                "(30 freq x 91 elevation) point-to-point tables are supported")
        cant.load_p2p(iat, ant, itr)
    pkw = max(float(power_kw), 1e-6)
    cant.pwrkw[1] = F32(power_kw)
    cant.pwrdba[1] = F32(30.0 + 10.0 * np.log10(pkw))
    return cant


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
                  npsl=0, iono=None, circ=None):
    """One fully-built Stage-A ``Circuit`` for ``hour`` (1-based UTC).

    ``tx``/``rx`` are (lat_deg, lon_deg), south/west negative.  Returns
    ``(circ, ipflag)``.  ``iono`` (a REDMAP ``IonoState``) and ``circ`` (a
    Circuit to reuse for COMMON carry-over) may be supplied to avoid rebuilding
    them across hours.
    """
    system = system or System()
    psc, amind = system._psc()
    imeth, mspec = _resolve_method(method)
    freqs = [F32(f) for f in freqs]
    if not 1 <= len(freqs) <= 11:
        raise NotImplementedError(
            f"{len(freqs)} frequencies: 1..11 are supported per call")

    st = iono if iono is not None else redmap(_COEFFS_DIR, coeff, int(month),
                                              F32(ssn))
    if cant is None:
        cant = isotropic_cant(system.power_kw)

    if circ is None:
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
        circ.dmp = F32(system.max_spread_db)
        circ.pmp = F32(system.max_delay_ms)
        circ.want_service_prob = imeth in (20, 25, 30)
        circ.time.itim = 1                     # hours are UTC (GMT)
        # geometry is fixed for the path; build it once
        tlatd, tlongd = tx
        rlatd, rlongd = rx
        wld = lambda a, b: noisy(st, 7, a, b)    # NOISY(7) land/sea map
        g = geom(tlatd, tlongd, rlatd, rlongd, int(npsl), pole[0], pole[1],
                 wld)
        _install_geometry(circ, g, tlatd, tlongd, rlongd, amind)
        # user frequency list (FREL 1..nf), the DECRED flag, MUF column later
        for i, f in enumerate(freqs, start=1):
            circ.frq.frel[i] = f
        circ.frq.frel[14] = F32(1.0)

    _setup_hour(circ, int(hour))
    ipflag = _ipflag(imeth, mspec, circ.gcdkm)
    return circ, ipflag


# ---------------------------------------------------------------------------
# Running circuits on the GPU
# ---------------------------------------------------------------------------

def predict_circuits(circuits, ipflags, fast_math=False):
    """Run pre-built Stage-A circuits through the CUDA LUFFY kernel.

    Thin wrapper over ``cuda/driver.py``: shares each distinct ``iono`` / ``cant``
    on the device once, launches one thread per circuit, and writes the mutated
    /SON/ (and the rest) back into the given Circuit objects in place.  Returns
    ``(circuits, status)`` -- ``status`` is per-circuit (0 = success).

    ``fast_math`` opts into the native-f32 device libm, which is NOT bit-exact
    against Stage A (see ``cuda/f32math.cuh``); off by default.
    """
    from voacap_gpu.cuda.driver import (launch_luffy, pack_circuit,
                                         shared_antenna_tables,
                                         shared_iono_tables, unpack_circuit)
    I32 = np.int32
    ipflags = np.asarray(ipflags, I32)
    assert len(ipflags) == len(circuits)

    ionos, iono_idx, seen_i = [], [], {}
    cants, ant_idx, seen_a = [], [], {}
    for c in circuits:
        ki = id(c.iono)
        if ki not in seen_i:
            seen_i[ki] = len(ionos)
            ionos.append(c.iono)
        iono_idx.append(seen_i[ki])
        ka = id(c.cant)
        if ka not in seen_a:
            seen_a[ka] = len(cants)
            cants.append(c.cant)
        ant_idx.append(seen_a[ka])

    packed = [pack_circuit(c) for c in circuits]
    bufs = np.stack([p[0] for p in packed])
    rays_bufs = np.stack([p[1] for p in packed])
    out, out_rays, status = launch_luffy(
        bufs, rays_bufs, ipflags, shared_iono_tables(ionos),
        np.array(iono_idx, I32), shared_antenna_tables(cants),
        np.array(ant_idx, I32), fast_math=fast_math)
    for c, b, r in zip(circuits, out, out_rays):
        unpack_circuit(b, r, c)
    return circuits, status


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

# LUFFY's /SON/ MODE layer labels (blkdat LAYTYP), for reference.
_MODE_LABELS = (" E", "F1", "F2", "ES", " N")


@dataclass
class Results:
    """Per-(hour, frequency) HF-circuit predictions.

    Axes: ``hours`` (1-based UTC, shape ``(H,)``) and ``freqs`` (MHz, shape
    ``(F,)``).  The 2-D tables are ``(H, F)``:

        snr       median signal-to-noise ratio, dB
        dbw       median signal power, dBW
        reliab    circuit reliability, 0..1
        field_dbu median field strength, dB(uV/m)
        nhp       number of hops of the dominant mode
        mode      2-char reflecting-layer label ('F2', ' E', ...) or ''

    The per-hour 1-D vectors are ``muf`` (circuit MUF, MHz), ``fot`` and
    ``hpf`` (the 90 % / 10 % decile MUFs) and ``luf`` (lowest frequency in the
    request meeting the required reliability, NaN if none does).
    """

    hours: np.ndarray
    freqs: np.ndarray
    muf: np.ndarray
    fot: np.ndarray
    hpf: np.ndarray
    luf: np.ndarray
    snr: np.ndarray
    dbw: np.ndarray
    reliab: np.ndarray
    field_dbu: np.ndarray
    nhp: np.ndarray
    mode: np.ndarray
    status: np.ndarray = field(default=None)

    def array(self, name):
        """Named table as a plain numpy array (see the class docstring)."""
        return getattr(self, name)

    def at(self, hour, freq):
        """A dict of every quantity at one (hour, frequency) pair."""
        i = int(np.where(self.hours == hour)[0][0])
        j = int(np.argmin(np.abs(self.freqs - freq)))
        return {
            "hour": int(self.hours[i]), "freq": float(self.freqs[j]),
            "muf": float(self.muf[i]), "fot": float(self.fot[i]),
            "hpf": float(self.hpf[i]), "luf": float(self.luf[i]),
            "snr": float(self.snr[i, j]), "dbw": float(self.dbw[i, j]),
            "reliab": float(self.reliab[i, j]),
            "field_dbu": float(self.field_dbu[i, j]),
            "nhp": int(self.nhp[i, j]), "mode": str(self.mode[i, j]),
        }

    def to_records(self):
        """Flat list of :meth:`at` dicts over every (hour, frequency)."""
        return [self.at(int(h), float(f))
                for h in self.hours for f in self.freqs]


# ---------------------------------------------------------------------------
# The public endpoint
# ---------------------------------------------------------------------------

def predict(tx, rx, *, freqs, month, ssn, hours=range(1, 25),
            tx_antenna=None, rx_antenna=None, system=None, method=30,
            coeff="CCIR", pole=POLE_DEFAULT, long_path=False,
            fast_math=False) -> Results:
    """Predict an HF point-to-point circuit from raw inputs.

    Parameters
    ----------
    tx, rx : (lat_deg, lon_deg)
        Endpoint coordinates, south and west negative.
    freqs : sequence of float
        Frequencies in MHz (1..11 of them).
    month : int
        1..12.
    ssn : float
        Smoothed sunspot number.
    hours : iterable of int, default 1..24
        1-based UTC hours.
    tx_antenna, rx_antenna : path or None
        A type-14 ``.VOA`` sample file, or None for a 0 dBi isotropic pattern.
        Both must be given together or both omitted.
    system : System or None
        SYSTEM/FPROB parameters; defaults to the golden decks' values.
    method : int, default 30
        30 smoothing, 20 system performance, 25 all modes.
    coeff : {'CCIR', 'URSI'}
    pole : (lat, lon)
        Geomagnetic north pole; see the module docstring.
    fast_math : bool
        Opt into the non-bit-exact native-f32 device libm.

    Returns
    -------
    Results
    """
    system = system or System()
    hours = [int(h) for h in hours]
    freqs_f = np.array([float(f) for f in freqs], dtype=np.float32)
    if not 1 <= len(freqs_f) <= 11:
        raise NotImplementedError(
            f"{len(freqs_f)} frequencies: 1..11 are supported per call")

    if (tx_antenna is None) != (rx_antenna is None):
        raise NotImplementedError(
            "give both tx_antenna and rx_antenna, or neither (isotropic)")
    if tx_antenna is None:
        cant = isotropic_cant(system.power_kw)
    else:
        cant = cant_from_files(tx_antenna, rx_antenna, system.power_kw)

    # One coefficient map for the whole request; one INDEPENDENT circuit per
    # hour (fresh /SON/), all sharing the iono and antenna tables.
    st = redmap(_COEFFS_DIR, coeff, int(month), np.float32(ssn))
    circuits, ipflags = [], []
    muf = np.empty(len(hours), np.float32)
    fot = np.empty(len(hours), np.float32)
    hpf = np.empty(len(hours), np.float32)
    for k, h in enumerate(hours):
        c, ip = build_circuit(tx, rx, month, ssn, h, freqs_f, method=method,
                              system=system, cant=cant, coeff=coeff,
                              pole=pole, npsl=int(long_path), iono=st)
        it = c.time.it
        muf[k] = c.muf.allmuf[it]
        fot[k] = c.muf.fot[it]
        hpf[k] = c.muf.hpf[it]
        circuits.append(c)
        ipflags.append(ip)

    predict_circuits(circuits, ipflags, fast_math=fast_math)
    return _assemble(circuits, hours, freqs_f, muf, fot, hpf,
                     system.required_reliability)


def _assemble(circuits, hours, freqs_f, muf, fot, hpf, lufp):
    """Distil the mutated circuits' /SON/ into a :class:`Results`."""
    H, Fn = len(hours), len(freqs_f)
    snr = np.full((H, Fn), np.nan, np.float32)
    dbw = np.full((H, Fn), np.nan, np.float32)
    rel = np.full((H, Fn), np.nan, np.float32)
    dbu = np.full((H, Fn), np.nan, np.float32)
    nhp = np.zeros((H, Fn), np.int32)
    mode = np.empty((H, Fn), object)
    luf = np.full(H, np.nan, np.float32)
    status = np.zeros(H, np.int32)
    for k, c in enumerate(circuits):
        son = c.son
        for j in range(Fn):
            s = j + 1                          # /SON/ slot for freq j
            snr[k, j] = son.sndb[s]
            dbw[k, j] = son.dbw[s]
            rel[k, j] = son.reliab[s]
            dbu[k, j] = son.dbu[s]
            nhp[k, j] = int(son.nhp[s])
            mode[k, j] = son.mode[s]
        # practical LUF: lowest requested frequency meeting the reliability
        thresh = lufp / 100.0
        ok = [freqs_f[j] for j in range(Fn) if rel[k, j] >= thresh]
        if ok:
            luf[k] = min(ok)
    return Results(hours=np.array(hours, np.int32), freqs=freqs_f, muf=muf,
                   fot=fot, hpf=hpf, luf=luf, snr=snr, dbw=dbw, reliab=rel,
                   field_dbu=dbu, nhp=nhp, mode=mode, status=status)
