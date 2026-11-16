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
* Antennas, independently per end: an isotropic 0 dBi pattern (the default),
  a *type 14* ``.VOA`` table (30 frequencies x 91 elevations), or a *type 13*
  ``.VOA`` table (360 azimuths x 91 elevations at one design frequency) --
  each read through ``scalar/readant.py`` and installed exactly as READANT +
  ANTCALC would leave it.  Type 13 is azimuth-dependent: ANTCALC resolves
  OFFAZIM (DAZEL0 path azimuth minus the beam heading) at setup and installs
  the interpolated azimuth slice, so the /cantenna/ is per-circuit (see
  :func:`antcalc_cant`).  Types 1-12 (method-of-moments synthesis) raise
  ``NotImplementedError`` rather than returning wrong gains.
* Antenna bearings: ``tx_bearing`` / ``rx_bearing`` (degrees true) set
  OFFAZIM = DAZEL0 path azimuth at that end minus the main-beam heading,
  ANTCALC's rule, computed once in :func:`antcalc_cant`; default
  beam-along-path (a type-13 antenna with no bearing warns).
* Prop-SL ``model`` toggle, BOTH validated against voacapl (the same binary
  Prop-SL runs): ``'voacap'`` (CCIR + FPROB (1,1,1,0)) and ``'icepac'`` (URSI +
  FPROB (1,1,1,0.70), sporadic-E on).  'icepac' is Prop-SL's ICEPAC mode on
  the VOACAP engine, NOT the ITS ICEPAC program (no ICED model; Q/Kp is
  ignored by voacapl too).  See ``_resolve_model``.
  A ``model`` that contradicts an explicit ``coeff`` / ``fprob`` raises.
* The Prop-SL parameter surface (:class:`System`): required S/N (dB-Hz),
  reliability percentile, man-made noise (a -dBW/Hz level at 3 MHz or a CCIR
  category code), transmit power (kW), the multipath delay / power
  tolerances (DMP / PMP; MPATH only), and the ``tx_gos`` / ``rx_gos`` gain
  offsets (consumed); ``bandwidth`` / ``modulation`` are advisory (the
  S/N-derivation front-end is not ported).
* UTC hours 1..24.  ``hours`` given in local mean time at the transmitter is
  not exposed here (the Fortran's ITIM<0 path); pass UTC.

INPUT VALIDATION
----------------
``predict``, ``build_circuit`` and ``setup_config`` check their inputs with
:mod:`voacap_gpu.validate` before any of the chain runs: frequencies 1..30
MHz, month 1..12, integral hours 1..24, SSN 0..300, lat/lon in range, and the
:class:`System` values (see its docstring).  A bad value raises
``ValueError``; the Fortran-faithful ``int()`` truncation of a fractional
noise level or reliability is kept but warned about.  The checks never alter
a valid input, so the validated numbers below are untouched.

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
``tests/test_engine.py`` builds circuits from the raw inputs (the CIRCUIT
card's coordinates and S/L flag) of every captured LUFFY deck -- the 21
replay decks plus the ICEPAC one -- and shows the built geometry + per-area
ionosphere matches the captured LUFFY entry state at the documented libm
floor, then runs them through LUFFY on both backends in ``predict``'s
configuration and shows the /SON/ table matches the captured output: hop
counts exact, every numeric field within an absolute bound (0.01 dB on the
dB fields; the table is at the top of that file).

BACKENDS
--------
``predict(..., backend=)`` picks where LUFFY runs: ``'cuda'`` (default, the
production kernel), ``'cpu'`` (the scalar Stage-A reference, no cupy import,
bit-identical to the kernel on every oracle deck and ~1000x slower), or
``'auto'`` (CUDA if a device is usable, else CPU with a ``UserWarning``).
``tests/test_cpu_backend.py`` holds the two to bit-identity.
"""

import copy
import pathlib
import warnings
from dataclasses import dataclass, field, replace

import numpy as np

from voacap_gpu import validate as _validate
from voacap_gpu.scalar import constants as _K
from voacap_gpu.scalar.antennastate import CantennaState
from voacap_gpu.scalar.circuit import Circuit
from voacap_gpu.scalar.curmuf import curmuf
from voacap_gpu.scalar.dazel import dazel0
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

# CCIR/URSI coefficient directory.  Ships INSIDE the package (data/coeffs),
# so voacap_gpu is self-contained: predict() reads coeffNN.asc + fof2*.asc
# from here at runtime.  Falls back to the vendored oracle tree when running
# from a source checkout without the bundled copy (e.g. mid-rebuild), and can
# be overridden per call via ``build_circuit(..., iono=redmap(dir, ...))``.
_PKG_COEFFS = pathlib.Path(__file__).resolve().parent / "data" / "coeffs"
_VENDORED_COEFFS = (pathlib.Path(__file__).resolve().parents[2] / "reference"
                    / "voacapl" / "voacapl" / "itshfbc" / "coeffs")
_COEFFS_DIR = _PKG_COEFFS if _PKG_COEFFS.is_dir() else _VENDORED_COEFFS

_GCDLNG = F32(10000.0)          # blkdat long-path threshold, km


# ---------------------------------------------------------------------------
# System / SYSTEM-card parameters
# ---------------------------------------------------------------------------

@dataclass
class System:
    """The SYSTEM/FPROB card values plus the Prop-SL parameter surface.

    Field names are the ones a Prop-SL / PropServer caller uses.  Each defaults
    to the golden-deck SYSTEM/FPROB value, so a bare ``System()`` reproduces the
    validated decks.  Those defaults are TEST values (RSN 73 dB-Hz, AMIND 0.1
    degrees, 1 kW, ...), not recommended operating defaults: set every field
    that matters to your circuit.  Values are checked by
    :func:`voacap_gpu.validate.check_system` when the System is used
    (``predict`` / ``build_circuit`` / ``setup_config``), not at construction.

    * ``required_snr``            RSN -- required S/N at the reliability
      percentile, dB in a 1 Hz bandwidth (dB-Hz).  Prop-SL normally derives
      this from ``modulation`` and ``bandwidth``; that derivation is NOT
      ported, so pass the number.  Must be finite.
    * ``min_takeoff_deg``         AMIND -- minimum take-off angle, degrees.
      A value ``<= 0`` means 3.0 degrees, exactly as DECRED reads the card.
    * ``required_reliability``    LUFP -- the reliability percentile (the
      ``snrXXp`` percentile), percent, 1..99.  Read with ``int()`` as the
      Fortran does: a fraction (90.5) is truncated, with a ``UserWarning``.
    * ``man_made_noise``          NOISE -- a POSITIVE value is the man-made
      noise level at 3 MHz in -dBW/Hz (145 residential, 150 rural, 164
      quiet-rural, ...); ``0`` / ``-1`` / ``-2`` / ``-3`` / ``-4`` are the CCIR
      category codes (0 and -4 remote, -1 industrial, -2 residential, -3
      rural).  Any other value ``<= 0`` raises -- ``-145`` is a dBW level
      passed with a sign, not a category.  Truncated by ``int()`` as the
      Fortran does (144.7 -> 144, with a ``UserWarning``).
    * ``power_kw``                transmitter power, kW (ANTENNA-card PWRKW);
      must be > 0.
    * ``tx_gos`` / ``rx_gos``     Prop-SL gain-offset, dB, added to every
      value of the transmit / receive antenna gain table (a real, consumed
      offset).
    * ``fprob``                   the four FPROB multipliers applied to the
      (E, F1, F2, Es) critical frequencies (PSC).  ``(1, 1, 1, 0)`` disables
      sporadic E (the VOACAP deck value); ``(1, 1, 1, 0.7)`` is the ICEPAC
      value.  CONSUMED by the ported EF1VAR/ESIND/F2VAR chain, and both arms
      are validated against voacapl (the 0.7 arm against the ICEPAC oracle
      deck ``golden/dumps/luffy_icepac/tanbel_m07_s160_icepac``).
    * ``multipath_delay_ms``      DMP -- the maximum multipath time delay, ms
      (default 0.1).
    * ``multipath_power_db``      PMP -- the multipath power tolerance, dB
      (default 3.0).  Neither DMP nor PMP affects ``snr`` / ``reliab``: they
      feed only MPATH, the multipath probability.
    * ``max_delay_ms`` / ``max_spread_db``  DEPRECATED aliases, kept for one
      release with their original CROSSED mapping: ``max_delay_ms`` feeds PMP
      (dB, despite its name) and ``max_spread_db`` feeds DMP (ms).  Using
      either emits a ``FutureWarning`` (shown by default, unlike DeprecationWarning); mixing them with the new names
      raises.  Migrate ``max_delay_ms=X`` to ``multipath_power_db=X`` and
      ``max_spread_db=Y`` to ``multipath_delay_ms=Y`` to keep your numbers.
    * ``bandwidth`` / ``modulation``  ADVISORY only.  VOACAP folds bandwidth
      into the required S/N (a 1 Hz reference), and the modulation->S/N table
      is a Prop-SL front-end concern, neither of which is ported.  Carried for
      provenance; they do not change the computation.  Setting a non-default
      ``bandwidth`` without adjusting ``required_snr`` is a caller error.

    The effective DMP / PMP are the :attr:`dmp` / :attr:`pmp` properties.
    """

    required_snr: float = 73.0          # RSN, dB-Hz (S/N at the reliability pctile)
    min_takeoff_deg: float = 0.1        # AMIND, deg (<=0 -> 3.0, as DECRED does)
    required_reliability: int = 90      # LUFP, percent 1..99
    max_delay_ms: float = None          # DEPRECATED alias -> PMP (dB!), crossed
    max_spread_db: float = None         # DEPRECATED alias -> DMP (ms!), crossed
    man_made_noise: float = 145.0       # NOISE, -dBW/Hz at 3 MHz, or category
    power_kw: float = 1.0               # transmitter power, kW (also on ANTENNA)
    tx_gos: float = 0.0                 # Prop-SL transmit gain offset, dB
    rx_gos: float = 0.0                 # Prop-SL receive gain offset, dB
    fprob: tuple = (1.0, 1.0, 1.0, 0.0)  # (E, F1, F2, Es) multipliers
    bandwidth: float = None             # advisory (Hz); fold into required_snr
    modulation: str = None              # advisory; Prop-SL derives S/N from it
    multipath_delay_ms: float = None    # DMP, ms (None -> 0.1)
    multipath_power_db: float = None    # PMP, dB (None -> 3.0)

    def __post_init__(self):
        old = _validate.check_multipath_aliases(self)
        if old:
            warnings.warn(
                f"System({', '.join(old)}=...) is deprecated and its names are "
                "CROSSED: max_delay_ms feeds PMP (the multipath power "
                "tolerance, dB) and max_spread_db feeds DMP (the multipath "
                "delay, ms).  The old mapping is kept for now; use "
                "multipath_power_db= / multipath_delay_ms= instead (same "
                "numbers: max_delay_ms -> multipath_power_db, max_spread_db "
                "-> multipath_delay_ms)", FutureWarning, stacklevel=3)

    @property
    def dmp(self):
        """Effective DMP, the max multipath time delay, ms."""
        if self.multipath_delay_ms is not None:
            return self.multipath_delay_ms
        if self.max_spread_db is not None:        # deprecated, crossed name
            return self.max_spread_db
        return DMP_DEFAULT_MS

    @property
    def pmp(self):
        """Effective PMP, the multipath power tolerance, dB."""
        if self.multipath_power_db is not None:
            return self.multipath_power_db
        if self.max_delay_ms is not None:         # deprecated, crossed name
            return self.max_delay_ms
        return PMP_DEFAULT_DB

    def _psc(self):
        amind = self.min_takeoff_deg if self.min_takeoff_deg > 0.0 else 3.0
        psc = np.array([0.0, *self.fprob], dtype=np.float32)
        return psc, F32(amind)


# The golden decks' DMP / PMP (decred.for:245-259 reads DMP, then PMP).
DMP_DEFAULT_MS = 0.1
PMP_DEFAULT_DB = 3.0


# ---------------------------------------------------------------------------
# Antennas
# ---------------------------------------------------------------------------

def offazim(path_bearing_deg, beam_bearing_deg) -> float:
    """OFFAZIM: the angle a main beam sits off the great-circle path.

    Exactly ANTCALC's rule (antcalc.for:271-272): ``OFFAZIM = path_azimuth -
    beam_heading`` wrapped into ``[0, 360)``.  ``path_bearing_deg`` is the
    azimuth of the path AT that end (GEOM's BTRD at the transmitter, BRTD at
    the receiver); ``beam_bearing_deg`` is the antenna main-beam true azimuth.
    """
    return float(F32((F32(path_bearing_deg) - F32(beam_bearing_deg)) % F32(360.0)))


def _finish_cant(cant, power_kw, tx_gos, rx_gos, off_tx, off_rx):
    """Common tail: power, gain offsets, and OFFAZIM on a built /cantenna/."""
    pkw = max(float(power_kw), 1e-6)
    cant.pwrkw[1] = F32(power_kw)
    # REAL*4 arithmetic, as the Fortran does it: the float64 form was 1 ULP
    # off the oracle's PWRDBA on 28 of 200 sweep decks (docs/accuracy_sweep.md)
    # and this form reproduces it on all 200.
    from voacap_gpu.scalar.f32math import log10_32
    cant.pwrdba[1] = F32(F32(30.0) + F32(10.0) * log10_32(F32(pkw)))
    if tx_gos:
        cant.array[:, :, 1] = cant.array[:, :, 1] + F32(tx_gos)
    if rx_gos:
        cant.array[:, :, 2] = cant.array[:, :, 2] + F32(rx_gos)
    cant.offazim[1] = F32(off_tx)
    cant.offazim[2] = F32(off_rx)
    return cant


def isotropic_cant(power_kw=1.0, tx_gos=0.0, rx_gos=0.0,
                   off_tx=0.0, off_rx=0.0) -> CantennaState:
    """A two-entry /cantenna/: 0 dBi isotropic TX (1) and RX (2).

    ARRAY is zero (0 dBi at every frequency and elevation, plus any gain
    offset) and AEFF zero; PWRDBA carries the transmitter power.  ``off_tx`` /
    ``off_rx`` are the OFFAZIM values (see :func:`offazim`) -- inert to an
    isotropic gain but carried faithfully.
    """
    cant = CantennaState()
    for iat, itr in ((1, 1), (2, 2)):
        cant.iats[iat] = itr
        cant.xfqs[iat] = F32(0.0)
        cant.xfqe[iat] = F32(30.0)
        cant.numants = max(cant.numants, iat)
    return _finish_cant(cant, power_kw, tx_gos, rx_gos, off_tx, off_rx)


def cant_from_files(tx_path, rx_path, power_kw=1.0, tx_gos=0.0, rx_gos=0.0,
                    off_tx=0.0, off_rx=0.0, *, tx_freq_range=(2, 30),
                    rx_freq_range=(2, 30), tx_table_absolute=False,
                    rx_table_absolute=False) -> CantennaState:
    """Build a /cantenna/ from two type-14 ``.VOA`` sample files (TX, RX).

    Installed exactly as READANT + ANTCALC would (``CantennaState.load_p2p``):
    only rows MINFREQ..MAXFREQ (``*_freq_range``, the ANTENNA card's usual
    2..30 MHz, whole MHz) are installed, with XFQS/XFQE bracketing them;
    GAIN substitutes 0 dBi outside that range.  The tables are RELATIVE to
    header parameter 1 ("Max Gain dBi"), which is added to every gain as
    ANTCALC does; ``*_table_absolute=True`` skips that add for a table
    already in absolute dBi.

    A type-14 pattern is azimuth-independent, so OFFAZIM does not change the
    gain -- ``off_tx`` / ``off_rx`` are recorded for provenance.  Raises for
    non-tabulated types.  For azimuth-dependent (type 13) patterns use
    :func:`antcalc_cant`, which needs the circuit geometry.
    """
    cant = CantennaState()
    for iat, itr, path, rng, absolute in (
            (1, 1, tx_path, tx_freq_range, tx_table_absolute),
            (2, 2, rx_path, rx_freq_range, rx_table_absolute)):
        try:
            ant = read_antenna(path)
        except UnsupportedAntennaType as e:
            raise NotImplementedError(str(e)) from e
        if ant.jant != 14:
            raise NotImplementedError(
                f"antenna file {path!r} is type {ant.jant}; only type 14 "
                "(30 freq x 91 elevation) point-to-point tables are supported")
        cant.load_p2p(iat, ant, itr, fqs=rng[0], fqe=rng[1],
                      absolute=absolute)
    return _finish_cant(cant, power_kw, tx_gos, rx_gos, off_tx, off_rx)


def _antcalc_offazim(ztaz, beam_bearing):
    """ANTCALC's off-azimuth (antcalc.for:271-272, 336-337), REAL*4.

    The single +360 wrap matches the Fortran; the >=360 wrap does not exist
    there, but DAZEL0 can return exactly 360.0 (360 - atan2 of 0) and a
    tiny-negative difference rounds up to 360.0 in REAL*4 -- both circularly
    identical to 0, where the Fortran would read past its azimuth table.
    """
    if beam_bearing is None:
        return F32(0.0)                     # beam along the path
    off = F32(F32(ztaz) - F32(beam_bearing))
    if off < 0.0:
        off = F32(off + F32(360.0))
    if off >= F32(360.0):
        off = F32(off - F32(360.0))
    return off


def _check_bearing(end, ant, bearing):
    """Guard rails on a type-13 antenna's main-beam bearing (UserWarnings).

    Only type 13 consumes the bearing, so only type 13 is checked.  ``None``
    silently means "beam along the path", the BEST-CASE gain on every path;
    a small non-zero value is most likely radians passed for degrees.
    """
    import warnings
    if ant is None or ant.jant != 13:
        return
    if bearing is None:
        warnings.warn(
            f"{end} antenna is type 13 (azimuth-dependent) but {end}_bearing "
            "is None: the main beam is assumed to point along the path, which "
            "gives the best-case gain on every path.  Pass the antenna's real "
            "main-beam heading in degrees true azimuth.",
            UserWarning, stacklevel=3)
    elif bearing != 0.0 and -2.0 * np.pi <= float(bearing) <= 2.0 * np.pi:
        warnings.warn(
            f"{end}_bearing={float(bearing)!r} looks like it may be radians; "
            "bearings are degrees true azimuth (clockwise from true north).  "
            "Ignore this if a beam heading of "
            f"{float(bearing):g} degrees is intended.",
            UserWarning, stacklevel=3)


def antcalc_cant(tx_ant, rx_ant, tx, rx, *, power_kw=1.0, tx_gos=0.0,
                 rx_gos=0.0, tx_bearing=None, rx_bearing=None,
                 long_path=False, tx_freq_range=(2, 30),
                 rx_freq_range=(2, 30), tx_table_absolute=False,
                 rx_table_absolute=False, freqs=None) -> CantennaState:
    """ANTCALC for one circuit: a /cantenna/ from parsed antenna files.

    The per-CIRCUIT half of antcalc.for's point-to-point path (lines
    106-118 for the azimuths, 246-337 for the tables): compute the path
    azimuth at each end with DAZEL0 (long path flips it 180 degrees,
    antcalc.for:110-115), derive OFFAZIM from the main-beam heading, and
    install each end's pattern:

    * ``None``          -- 0 dBi isotropic entry (as :func:`isotropic_cant`),
      valid over 0..30 MHz,
    * type 14 ``.VOA``  -- the azimuth-independent table (load_p2p),
    * type 13 ``.VOA``  -- the azimuth slice at OFFAZIM (load_p2p_t13); this
      is the arm that makes the result GEOMETRY-DEPENDENT: a new tx/rx pair
      needs a new cant, exactly as voacapl reruns ANTCALC per CIRCUIT card.

    ``tx_ant`` / ``rx_ant`` are parsed :class:`AntennaFile` objects (see
    ``scalar.readant``) or None; ``tx`` / ``rx`` are (lat, lon) degrees.
    Other tabulated/synthesised types raise ``NotImplementedError``.

    ``tx_bearing`` / ``rx_bearing`` are the main-beam headings in DEGREES
    TRUE AZIMUTH (clockwise from true north).  Only type 13 consumes them;
    ``None`` points the beam along the path (best-case gain), and a type-13
    file given with ``None`` -- or with a small non-zero bearing that looks
    like radians -- emits a ``UserWarning``.

    ``tx_freq_range`` / ``rx_freq_range`` are the ANTENNA card's MINFREQ /
    MAXFREQ (whole MHz, default 2..30) for a FILE antenna: only those table
    rows are installed and XFQS/XFQE bracket them, as ANTCALC does.  GAIN
    returns 0 dBi / efficiency 0 for a frequency outside the range, silently;
    pass ``freqs`` (MHz) to have that raise ``ValueError`` here instead.

    Tables are RELATIVE to header parameter 1 ("Max Gain dBi"): ANTCALC adds
    PARM(1) to every tabulated gain, and so does this.  A table already in
    absolute dBi would be double-counted; ``*_table_absolute=True`` skips the
    add, and an installed peak above 30 dBi emits a ``UserWarning``.
    """
    ends = (("tx", tx_ant, tx_freq_range), ("rx", rx_ant, rx_freq_range))
    if freqs is not None:
        for end, ant, rng in ends:
            if ant is None:
                continue
            lo, hi = F32(rng[0]), F32(rng[1])
            for f in freqs:
                if not lo <= F32(f) <= hi:
                    raise ValueError(
                        f"{end} antenna ({ant.description.strip()!r}, type "
                        f"{ant.jant}) covers {float(lo):g}..{float(hi):g} MHz "
                        f"(MINFREQ..MAXFREQ) but {float(f):g} MHz was "
                        "requested; VOACAP's GAIN would silently substitute "
                        "0 dBi there")
    _check_bearing("tx", tx_ant, tx_bearing)
    _check_bearing("rx", rx_ant, rx_bearing)

    ztaz_t, _ = dazel0(tx[0], tx[1], rx[0], rx[1])
    ztaz_r, _ = dazel0(rx[0], rx[1], tx[0], tx[1])
    if long_path:                           # antcalc.for:110-115
        ztaz_t = F32(ztaz_t + F32(180.0))
        if ztaz_t >= 360.0:
            ztaz_t = F32(ztaz_t - F32(360.0))
        ztaz_r = F32(ztaz_r + F32(180.0))
        if ztaz_r >= 360.0:
            ztaz_r = F32(ztaz_r - F32(360.0))
    off_tx = _antcalc_offazim(ztaz_t, tx_bearing)
    off_rx = _antcalc_offazim(ztaz_r, rx_bearing)

    cant = CantennaState()
    for iat, itr, ant, off, rng, absolute in (
            (1, 1, tx_ant, off_tx, tx_freq_range, tx_table_absolute),
            (2, 2, rx_ant, off_rx, rx_freq_range, rx_table_absolute)):
        if ant is None:
            cant.iats[iat] = itr
            cant.xfqs[iat] = F32(0.0)
            cant.xfqe[iat] = F32(30.0)
            cant.numants = max(cant.numants, iat)
        elif ant.jant == 14:
            cant.load_p2p(iat, ant, itr, fqs=rng[0], fqe=rng[1],
                          absolute=absolute)
        elif ant.jant == 13:
            cant.load_p2p_t13(iat, ant, itr, off, fqs=rng[0], fqe=rng[1],
                              absolute=absolute)
        else:
            raise NotImplementedError(
                f"antenna type {ant.jant} is not ported; isotropic, type 13 "
                "and type 14 are supported")
    return _finish_cant(cant, power_kw, tx_gos, rx_gos, off_tx, off_rx)


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

    st = iono if iono is not None else redmap(_COEFFS_DIR, coeff, int(month),
                                              F32(ssn))

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


# ---------------------------------------------------------------------------
# Backends: the CUDA kernel, or the scalar Stage-A reference on the CPU
# ---------------------------------------------------------------------------

BACKENDS = ("cuda", "cpu", "auto")

# The CPU backend's own status code: the scalar reference raised somewhere the
# CUDA kernel has no identifiable counterpart status for (see Results, STATUS
# CODES).  Outside every callee-local code the kernel returns (1..6, 21, 22)
# and the driver's -1 preset.
STATUS_CPU_UNMAPPED = 99

# Exceptions the scalar LUFFY raises where the kernel returns a non-zero
# status: the explicit refusals (AllModesOverflow, ModmufError, RegmodError,
# OldNoiseModel, ... are IndexError / ValueError / NotImplementedError
# subclasses), numpy / list index errors at the Fortran's out-of-bounds reads,
# and FDIST/FINDF's RuntimeError.  Anything else -- TypeError, AttributeError,
# KeyError -- is a bug in the caller or the port and propagates.
_SCALAR_REFUSALS = (IndexError, ValueError, NotImplementedError, RuntimeError,
                    ArithmeticError)


def _cuda_usable() -> bool:
    """True if cupy imports and sees at least one CUDA device.

    Does not compile anything: a device that is present but cannot build the
    kernels (no NVRTC, an unsupported architecture) still fails at launch.
    """
    try:
        import cupy
        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def _resolve_backend(backend, fast_math=False) -> str:
    """``backend`` -> ``'cuda'`` or ``'cpu'`` (see :func:`predict`)."""
    if backend not in BACKENDS:
        raise ValueError(
            f"backend={backend!r}: expected one of {', '.join(BACKENDS)}")
    if backend == "auto":
        if _cuda_usable():
            backend = "cuda"
        else:
            warnings.warn(
                "backend='auto': no usable CUDA device (cupy missing or no "
                "GPU visible), falling back to the CPU scalar reference, "
                "which is roughly 1000x slower than the GPU kernel (~70 ms "
                "per circuit-hour).  Pass backend='cpu' to silence this.",
                UserWarning, stacklevel=3)
            backend = "cpu"
    if backend == "cpu" and fast_math:
        raise ValueError(
            "fast_math=True selects the CUDA native-f32 libm and has no CPU "
            "equivalent; use backend='cuda' or fast_math=False")
    return backend


def _scalar_status(exc) -> int:
    """The CUDA status code for a refusal the scalar LUFFY raised.

    The kernel's codes are callee-LOCAL (see :class:`Results`), so the code
    is identified from WHERE the scalar reference raised -- the innermost
    ``voacap_gpu.scalar`` frame of the traceback, plus GAIN's caller and,
    where one callee has two codes, the message or source line that tells
    them apart.  Each arm mirrors a ``return N`` in ``cuda/kernels``:

        allmodes / frqcom / lngpat / relbil / seltxr   1
        genois  OldNoiseModel (GENOIS_OLD)              1
        gain    via genois: IndexError 2, NotImplementedError (area cov.) 3
        gain    via regmod/esmod/settxr: IndexError 3, NotImplementedError 4
        regmod  RegmodError: MUFday prob <= 0 -> 1, frequency <= 0 -> 2
        settxr  ModmufError: DELMUF -> 1, YMUF -> 2
        inmuf   IndexError on INM(ITMOD) -> 5, on a /MODES/ row -> 6
        lsmooth IndexError on the smoothing store -> 21
        luffy   IndexError, /allMODE/ NMODE(2) outside LAYTYP -> 22

    Anything else (e.g. SERPRB's NoModesError or FDIST/FINDF's convergence
    RuntimeError, which the kernel has no status for) is
    :data:`STATUS_CPU_UNMAPPED`.
    """
    import linecache

    frames = []                                  # (module, line text)
    tb = exc.__traceback__
    while tb is not None:
        name = tb.tb_frame.f_globals.get("__name__", "")
        if name.startswith("voacap_gpu.scalar."):
            line = linecache.getline(tb.tb_frame.f_code.co_filename,
                                     tb.tb_lineno)
            frames.append((name.rsplit(".", 1)[-1], line))
        tb = tb.tb_next
    if not frames:
        return STATUS_CPU_UNMAPPED
    where, line = frames[-1]
    msg = str(exc)
    is_index = isinstance(exc, IndexError)
    if where == "gain":
        caller = frames[-2][0] if len(frames) > 1 else ""
        area = isinstance(exc, NotImplementedError)      # oob_flag 2
        if not (area or is_index):
            return STATUS_CPU_UNMAPPED
        if caller == "genois":
            return 3 if area else 2
        if caller in ("regmod", "esmod", "settxr"):
            return 4 if area else 3
        return STATUS_CPU_UNMAPPED
    if where in ("allmodes", "frqcom", "lngpat", "relbil", "seltxr"):
        return 1 if is_index else STATUS_CPU_UNMAPPED
    if where == "genois":
        return 1 if isinstance(exc, NotImplementedError) else \
            STATUS_CPU_UNMAPPED
    if where == "regmod" and isinstance(exc, ValueError):
        return 2 if "frequency must be positive" in msg else 1
    if where == "settxr" and is_index:
        if "YMUF" in msg:
            return 2
        if "DELMUF" in msg:
            return 1
        return STATUS_CPU_UNMAPPED
    if where == "inmuf" and is_index:
        return 5 if "inm[" in line else 6
    if where == "lsmooth" and is_index:
        return 21
    if where == "luffy" and is_index and "NMODE(2)" in msg:
        return 22
    return STATUS_CPU_UNMAPPED


def _predict_circuits_cpu(circuits, ipflags):
    """The CPU backend: the scalar reference LUFFY, one circuit at a time.

    Same contract as the CUDA path -- mutates each Circuit in place and
    returns ``(circuits, status)`` -- with a refusal caught per circuit and
    turned into the kernel's status code by :func:`_scalar_status`.  Like the
    kernel, a refused circuit keeps whatever partial /SON/ it had reached;
    :class:`Results` masks the whole hour.
    """
    from voacap_gpu.scalar.luffy import luffy

    ipflags = np.asarray(ipflags, np.int32)
    assert len(ipflags) == len(circuits)
    status = np.zeros(len(circuits), np.int32)
    for k, (c, ip) in enumerate(zip(circuits, ipflags)):
        try:
            luffy(c, int(ip))
        except _SCALAR_REFUSALS as e:
            status[k] = _scalar_status(e)
    return circuits, status


# ---------------------------------------------------------------------------
# Running circuits on the GPU
# ---------------------------------------------------------------------------

def predict_circuits(circuits, ipflags, fast_math=False, *, backend="cuda"):
    """Run pre-built Stage-A circuits through LUFFY.

    With ``backend='cuda'`` (the default) this is a thin wrapper over
    ``cuda/driver.py``: shares each distinct ``iono`` / ``cant`` on the device
    once, launches one thread per circuit, and writes the mutated /SON/ (and
    the rest) back into the given Circuit objects in place.  With
    ``backend='cpu'`` the scalar reference ``scalar.luffy.luffy`` runs each
    circuit in turn instead (no cupy import); ``'auto'`` picks as
    :func:`predict` does.  Returns ``(circuits, status)`` -- ``status`` is
    per-circuit (0 = success; the codes are the same on both backends, see
    :class:`Results`).

    ``fast_math`` opts into the native-f32 device libm, which is NOT bit-exact
    against Stage A (see ``cuda/f32math.cuh``); off by default, and refused
    (``ValueError``) on the CPU backend.
    """
    if _resolve_backend(backend, fast_math) == "cpu":
        return _predict_circuits_cpu(circuits, ipflags)
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


# Frequency match tolerance for Results.at, MHz.  ``freqs`` is float32, so a
# requested 7.1 comes back as 7.0999999; 1e-4 MHz (100 Hz) absorbs that and
# nothing a caller would mean as a different frequency.
_FREQ_ATOL = 1e-4


def _valid_mask(nhp, status):
    """Which (hour, freq) cells LUFFY actually computed; see :class:`Results`."""
    nhp = np.asarray(nhp)
    ok_hour = np.asarray(status).reshape(-1, 1) == 0
    return (nhp >= 1) & ok_hour


@dataclass
class Results:
    """Per-(hour, frequency) HF-circuit predictions.

    Axes: ``hours`` (1-based UTC, shape ``(H,)``) and ``freqs`` (MHz, float32,
    shape ``(F,)``).  The 2-D tables are ``(H, F)``:

        snr       median signal-to-noise ratio in a 1 Hz bandwidth, dB-Hz
                  (dB re 1 Hz).  The noise is computed per hertz (GENOIS
                  references kTb at 290 K in 1 Hz, ``scalar/genois.py``
                  ``_KTB``), so this is NOT the S/N in the receiver's
                  bandwidth: subtract ``10*log10(bandwidth_hz)`` for that.
        dbw       median signal power at the receiver, dBW
        reliab    circuit reliability, a fraction 0..1 (not percent)
        field_dbu median field strength at the receiver, dB(uV/m)
        nhp       number of hops of the dominant mode (>= 1 where valid)
        mode      2-char reflecting-layer label ('F2', ' E', ...) or ''
        valid     bool: True where LUFFY computed this cell (see below)

    The per-hour 1-D vectors are ``muf`` (circuit MUF, MHz), ``fot`` and
    ``hpf`` (the 90 % / 10 % decile MUFs, MHz), ``luf`` (see
    :attr:`lowest_reliable_freq`) and ``status`` (int32, see below).

    VALID AND NOT-COMPUTED CELLS
    ----------------------------
    ``valid[h, f]`` is False where LUFFY did not produce an answer for that
    cell, and there ``snr``, ``dbw``, ``reliab`` and ``field_dbu`` are NaN,
    ``mode`` is ``''`` and ``nhp`` keeps its raw value (``-1`` or ``0``).  A
    cell is valid iff ``status[h] == 0`` AND ``nhp[h, f] >= 1``:

    * On the system-performance arms ``predict`` uses (IPFLAG 100 / 200),
      LUFFY presets ``NHP(IFX) = -1`` at the top of every frequency slot
      (``luffy.cuh`` "PRESET IN COMMON/SON/").  The ONLY writer of NHP after
      that is RELBIL's tail (or the smoothing blend copying a saved RELBIL
      column), which writes ``INT(HN)`` of the chosen mode, and every mode
      that reaches /allMODE/ has HN >= 1 (a hop count; LNGPAT clamps it with
      ``MAX(1., ...)``).  So ``nhp >= 1`` <=> RELBIL ran for that slot.  The
      skip paths -- "no frequency in this slot" (FREL <= 0) and RELBIL's own
      NMMOD <= 0 no-op -- leave -1 and the /SON/ presets (0 dB, 0 dBW, ...),
      which look like data but are not.
    * A non-zero ``status`` means the kernel stopped part-way through the
      hour.  Slots after the failing one are never reached (their NHP is the
      fresh circuit's 0), and the slots before it are NOT the final answer
      either: on a 7000-10000 km path the short/long blend runs only after
      the whole loop, so they hold one model's unblended pass.  The whole
      hour is therefore invalid.

    VERY NEGATIVE VALUES ARE REAL, NOT MISSING
    ------------------------------------------
    Above the MUF VOACAP still computes an over-the-MUF mode, whose loss
    grows steeply with the frequency ratio, so valid cells can hold e.g.
    ``dbw`` of -500 to -650 dBW, ``snr`` of -380 dB-Hz and ``reliab`` around
    5.8e-6.  These ARE what VOACAP computes and match the Fortran oracle; they
    mean "no usable signal".  Two of them are floors rather than physics:
    ``dbw``/``field_dbu`` of exactly -500 is RELBIL's sentinel when the
    multi-mode power sum underflows, and ``reliab`` ~5.84e-6 is FNORML's
    5-sigma clamp (``scalar/fnorml.py``).  They are deliberately left as is;
    use ``valid`` for "not computed", and a threshold of your own for "no
    signal".

    STATUS CODES (per hour)
    -----------------------
    ``status[h]`` is the LUFFY kernel's return for that hour's circuit
    (``cuda/driver.py``).  0 = success.  -1 = the kernel never wrote it (the
    driver's preset).  Anything else is the FIRST refusal a callee reported,
    propagated unchanged -- each one marks a place where the Stage-A Python
    raises because the Fortran would misbehave (read out of bounds, STOP, or
    hit an unported feature).  The numbers are callee-LOCAL, so the same
    value can mean different things; the ones seen in practice:

        1   ALLMODES overflow (> 20 modes), FRQCOM index, RELBIL / LNGPAT /
            SELTXR / SETTXR mode or ray index out of range, REGMOD STOP,
            or GENOIS_OLD (unported)
        2   GENOIS: receive-antenna GAIN lookup left the table
            (``gain.cuh`` oob_flag 1 -- e.g. a frequency below 1 MHz);
            also REGMOD STOP / SETTXR YMUF index
        3   GENOIS: receive antenna is AREA-COVERAGE (oob_flag 2,
            OFFAZIM == -999, not ported); REGMOD / ESMOD / SETTXR:
            antenna GAIN lookup left the table (oob_flag 1)
        4   REGMOD / ESMOD / SETTXR: area-coverage antenna (oob_flag 2)
        5, 6  INMUF index out of range
        21  smoothing store index (LUF-arm IPFLAG in the blend)
        22  LUFFY: /allMODE/ NMODE(2) outside LAYTYP on the long path

        99  CPU backend only (``STATUS_CPU_UNMAPPED``): the scalar reference
            raised where the kernel has no identifiable status (e.g.
            SERPRB's NoModesError)

    The CPU backend (``predict(..., backend='cpu')``) runs the scalar
    reference, which RAISES at each of these places; the exception is caught
    per circuit-hour and mapped back to the kernel's code from where it was
    raised (``engine._scalar_status``), so ``status`` and ``valid`` read the
    same on either backend.

    Treat any non-zero status as "this hour has no answer"; ``valid`` already
    does.
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
    valid: np.ndarray = field(default=None)

    def __post_init__(self):
        # Directly-constructed Results (not via _assemble) get the same
        # defaults _assemble would derive.
        if self.status is None:
            self.status = np.zeros(len(self.hours), np.int32)
        if self.valid is None:
            self.valid = _valid_mask(self.nhp, self.status)

    @property
    def lowest_reliable_freq(self):
        """Per hour: the lowest REQUESTED frequency meeting the reliability.

        That is, the smallest of ``freqs`` whose ``reliab`` is at or above
        ``System.required_reliability`` (MHz; NaN if none does, and invalid
        cells never qualify).  It is NOT VOACAP's LUF: VOACAP's LUF (the LUF
        arm, IPFLAG 300/400, /MUFS/ XLUF) searches its own frequency
        complement below the MUF and interpolates between the bracketing
        frequencies, so it is generally not one of the requested frequencies
        at all.  This is the same array as the ``luf`` field, which is kept
        under that name for compatibility.
        """
        return self.luf

    def array(self, name):
        """Named table as a plain numpy array (see the class docstring)."""
        return getattr(self, name)

    def _index(self, hour, freq):
        hi = np.flatnonzero(self.hours == hour)
        if hi.size == 0:
            raise KeyError(
                f"hour {hour!r} is not in this result; hours are "
                f"{[int(h) for h in self.hours]}")
        fi = np.flatnonzero(np.isclose(self.freqs, freq, rtol=0.0,
                                       atol=_FREQ_ATOL))
        if fi.size == 0:
            raise KeyError(
                f"frequency {freq!r} MHz is not in this result (exact match "
                f"within {_FREQ_ATOL} MHz required); freqs are "
                f"{[round(float(f), 6) for f in self.freqs]}")
        return int(hi[0]), int(fi[0])

    def at(self, hour, freq):
        """A dict of every quantity at one (hour, frequency) pair.

        ``freq`` must match one of ``freqs`` to within 1e-4 MHz -- there is
        no snapping to the nearest frequency -- and ``hour`` one of ``hours``;
        otherwise ``KeyError``.
        """
        i, j = self._index(hour, freq)
        return {
            "hour": int(self.hours[i]), "freq": float(self.freqs[j]),
            "muf": float(self.muf[i]), "fot": float(self.fot[i]),
            "hpf": float(self.hpf[i]), "luf": float(self.luf[i]),
            "snr": float(self.snr[i, j]), "dbw": float(self.dbw[i, j]),
            "reliab": float(self.reliab[i, j]),
            "field_dbu": float(self.field_dbu[i, j]),
            "nhp": int(self.nhp[i, j]), "mode": str(self.mode[i, j]),
            "valid": bool(self.valid[i, j]), "status": int(self.status[i]),
        }

    def to_records(self):
        """Flat list of :meth:`at` dicts over every (hour, frequency)."""
        return [self.at(int(h), float(f))
                for h in self.hours for f in self.freqs]


# ---------------------------------------------------------------------------
# The public endpoint
# ---------------------------------------------------------------------------

def predict(tx, rx, *, freqs, month, ssn, hours=range(1, 25),
            tx_antenna=None, rx_antenna=None, tx_bearing=None, rx_bearing=None,
            system=None, method=30, model=None, coeff="CCIR", pole=POLE_DEFAULT,
            long_path=False, fast_math=False, tx_antenna_freq_range=(2, 30),
            rx_antenna_freq_range=(2, 30), tx_table_absolute=False,
            rx_table_absolute=False, backend="cuda") -> Results:
    """Predict an HF point-to-point circuit from raw inputs.

    The setup chain (geometry, ionosphere, MUFs) always runs on the host; only
    LUFFY, the per-hour system-performance calculation, goes to ``backend``.

    Parameters
    ----------
    tx, rx : (lat_deg, lon_deg)
        Endpoint coordinates, south and west negative; latitude in -90..90,
        longitude in -180..180.
    freqs : sequence of float
        Frequencies in MHz (1..11 of them), each in 1..30 MHz (the GAIN
        table's axis).
    month : int
        1..12.
    ssn : float
        Smoothed sunspot number, 0..300 (``validate.SSN_MAX``; the upper bound
        is a policy limit on extrapolation, not physics).
    hours : iterable of int, default 1..24
        1-based UTC hours, each an integer 1..24 (``12.0`` is accepted,
        ``12.7`` is refused rather than truncated).
    tx_antenna, rx_antenna : path or None
        A type-14 or type-13 ``.VOA`` file, or None for a 0 dBi isotropic
        pattern -- independently per end.  Type-13 patterns are
        azimuth-dependent; the slice at OFFAZIM (path azimuth minus
        ``*_bearing``) is installed, exactly as ANTCALC does.  Gain tables
        are RELATIVE to the file's header parameter 1 ("Max Gain dBi"),
        which is added to every tabulated gain as ANTCALC does -- see
        ``tx_table_absolute``.
    tx_bearing, rx_bearing : float or None
        Antenna main-beam heading in DEGREES TRUE AZIMUTH (clockwise from
        true north).  OFFAZIM is then the angle between that beam and the
        path azimuth at each end (ANTCALC's rule, via DAZEL0).  None (the
        default) points the beam along the path (OFFAZIM 0) -- the
        best-case gain on every path, so a type-13 file given with None
        emits a ``UserWarning``, as does a small non-zero bearing that
        looks like radians.  Inert for isotropic and type-14 antennas.
    system : System or None
        The Prop-SL parameter surface (SYSTEM/FPROB + gain offsets); see
        :class:`System` for units and accepted ranges.  Defaults to the golden
        decks' TEST values (RSN 73 dB-Hz, AMIND 0.1, 1 kW), not operating
        defaults.
    method : int, default 30
        30 smoothing, 20 system performance, 25 all modes.
    model : {'voacap', 'icepac', None}
        Prop-SL model toggle.  ``'voacap'`` selects CCIR coefficients + FPROB
        ``(1,1,1,0)`` (the golden-deck configuration).  ``'icepac'`` selects
        URSI + FPROB ``(1,1,1,0.70)`` (sporadic E on): Prop-SL's ICEPAC mode
        as voacapl runs it, i.e. the VOACAP engine with ICEPAC-like settings,
        NOT the ITS ICEPAC program (no ICED model; Q/Kp has no effect).  It
        is validated against voacapl on the 14 decks in
        ``golden/dumps/luffy_icepac/`` (see :func:`_resolve_model`).  A ``model`` that contradicts an explicit
        ``coeff`` or a non-default ``system.fprob`` raises ``ValueError``
        rather than overriding it.  ``None`` uses ``coeff`` / ``system`` as
        given.
    coeff : {'CCIR', 'URSI'}
        Coefficient set (exact upper-case spelling).  With ``model`` given,
        the default 'CCIR' counts as "not given" (the signature cannot tell
        it from an explicit 'CCIR'), so only an explicit 'URSI' with
        ``model='voacap'`` is detected as a conflict.  'URSI' with
        ``model=None`` keeps FPROB as given (Es off by default); its setup
        chain is validated by the ``*_ursi`` oracle decks.
    pole : (lat, lon)
        Geomagnetic north pole; see the module docstring.
    long_path : bool
        Take the long great-circle (through-antipode) geometry (NPSL=1).
    fast_math : bool
        Opt into the non-bit-exact native-f32 device libm (CUDA only;
        ``ValueError`` with ``backend='cpu'``).
    backend : {'cuda', 'cpu', 'auto'}, default 'cuda'
        Where LUFFY runs.  ``'cuda'`` is the production CUDA kernel (needs
        cupy and a GPU).  ``'cpu'`` runs the scalar Stage-A reference
        (``scalar.luffy.luffy``) on each circuit-hour; it never imports
        cupy, and it is bit-identical to the CUDA kernel on every captured
        oracle deck, but roughly 1000x slower (~70 ms per circuit-hour), so
        it suits a handful of predictions, not area grids.  ``'auto'`` uses
        CUDA when cupy imports and sees a device, otherwise falls back to
        CPU with a ``UserWarning``.  Where the kernel would return a
        non-zero status the scalar code raises; the CPU backend catches that
        per hour and maps it to the same status code (see
        :class:`Results`), so ``status`` and ``valid`` behave identically.
    tx_antenna_freq_range, rx_antenna_freq_range : (int, int), default (2, 30)
        The ANTENNA card's MINFREQ / MAXFREQ for a file antenna, whole MHz in
        1..30: only those table rows are installed (ANTCALC's rule).  VOACAP's
        GAIN silently substitutes 0 dBi outside the range, so a requested
        frequency outside it raises ``ValueError`` naming the end.  Ignored
        for an isotropic (None) end, which covers 0..30 MHz.
    tx_table_absolute, rx_table_absolute : bool, default False
        True when the file's gain table is already in absolute dBi: skips the
        header-parameter-1 add that ANTCALC applies (which would double-count
        such a table).  With False, an installed peak above 30 dBi emits a
        ``UserWarning`` suggesting the table may already be absolute.

    Returns
    -------
    Results

    Raises
    ------
    ValueError
        An input outside the ranges above, an unusable :class:`System` value,
        a ``model`` / ``coeff`` / ``fprob`` conflict (see
        :mod:`voacap_gpu.validate`), an unknown ``backend``, or
        ``fast_math=True`` with the CPU backend.
    NotImplementedError
        More than 11 (or no) frequencies, an unsupported METHOD, model or
        antenna type.
    """
    backend = _resolve_backend(backend, fast_math)
    system = system or System()
    _validate.check_coeff(coeff)
    coeff, system = _resolve_model(model, coeff, system)
    _validate.check_system(system)
    tx = _validate.check_point("tx", tx)         # longitude wrapped to ±180
    rx = _validate.check_point("rx", rx)
    month = _validate.check_month(month)
    _validate.check_ssn(ssn)
    hours = _validate.check_hours(hours)
    freqs_f = np.array(_validate.check_freqs(freqs), dtype=np.float32)

    def _parse(path):
        if path is None:
            return None
        try:
            return read_antenna(path)
        except UnsupportedAntennaType as e:
            raise NotImplementedError(str(e)) from e

    # ANTCALC per circuit: either end may independently be isotropic, a
    # type-14 table, or a type-13 azimuth-dependent table (which consumes
    # the path azimuth at that end, so the cant is built from tx/rx here).
    # freqs= makes a frequency outside a file antenna's MINFREQ..MAXFREQ
    # raise instead of GAIN silently returning 0 dBi for it.
    cant = antcalc_cant(_parse(tx_antenna), _parse(rx_antenna), tx, rx,
                        power_kw=system.power_kw, tx_gos=system.tx_gos,
                        rx_gos=system.rx_gos, tx_bearing=tx_bearing,
                        rx_bearing=rx_bearing, long_path=long_path,
                        tx_freq_range=tx_antenna_freq_range,
                        rx_freq_range=rx_antenna_freq_range,
                        tx_table_absolute=tx_table_absolute,
                        rx_table_absolute=rx_table_absolute,
                        freqs=freqs_f)

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
                              pole=pole, npsl=int(long_path),
                              tx_bearing=tx_bearing, rx_bearing=rx_bearing,
                              iono=st)
        it = c.time.it
        muf[k] = c.muf.allmuf[it]
        fot[k] = c.muf.fot[it]
        hpf[k] = c.muf.hpf[it]
        circuits.append(c)
        ipflags.append(ip)

    if backend == "cpu":
        _, status = _predict_circuits_cpu(circuits, ipflags)
    else:
        # called with exactly the pre-backend signature, so a wrapper of
        # predict_circuits (tests monkeypatch it) keeps working
        _, status = predict_circuits(circuits, ipflags, fast_math=fast_math)
    return _assemble(circuits, hours, freqs_f, muf, fot, hpf,
                     system.required_reliability, status=status)


def _assemble(circuits, hours, freqs_f, muf, fot, hpf, lufp, status=None):
    """Distil the mutated circuits' /SON/ into a :class:`Results`.

    ``status`` is the per-circuit (= per-hour) LUFFY status
    :func:`predict_circuits` returns; None means "all succeeded".  Cells that
    were not computed (see :class:`Results`, VALID) come out NaN with mode
    ''; every computed cell is copied from /SON/ unchanged.
    """
    H, Fn = len(hours), len(freqs_f)
    snr = np.full((H, Fn), np.nan, np.float32)
    dbw = np.full((H, Fn), np.nan, np.float32)
    rel = np.full((H, Fn), np.nan, np.float32)
    dbu = np.full((H, Fn), np.nan, np.float32)
    nhp = np.zeros((H, Fn), np.int32)
    mode = np.empty((H, Fn), object)
    luf = np.full(H, np.nan, np.float32)
    if status is None:
        status = np.zeros(H, np.int32)
    status = np.asarray(status, np.int32).reshape(-1)
    if status.shape != (H,):
        raise ValueError(f"status has shape {status.shape}, expected ({H},)"
                         " -- one per circuit-hour")
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
    valid = _valid_mask(nhp, status)
    for a in (snr, dbw, rel, dbu):
        a[~valid] = np.nan
    mode[~valid] = ""
    for k in range(H):
        # lowest REQUESTED frequency meeting the reliability (NaN compares
        # False, so an invalid cell never qualifies)
        thresh = lufp / 100.0
        ok = [freqs_f[j] for j in range(Fn) if rel[k, j] >= thresh]
        if ok:
            luf[k] = min(ok)
    return Results(hours=np.array(hours, np.int32), freqs=freqs_f, muf=muf,
                   fot=fot, hpf=hpf, luf=luf, snr=snr, dbw=dbw, reliab=rel,
                   field_dbu=dbu, nhp=nhp, mode=mode, status=status,
                   valid=valid)
