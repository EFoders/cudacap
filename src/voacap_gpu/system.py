"""Shared inputs: ``System``, the geomagnetic pole, the coefficient directory.

Split out of :mod:`voacap_gpu.engine` (which still re-exports every name
here).  :class:`System` is the SYSTEM/FPROB card plus the Prop-SL parameter
surface; ``POLE_DEFAULT`` is the geomagnetic north pole the oracle used (see
the engine module docstring, THE GEOMAGNETIC POLE); ``_COEFFS_DIR`` /
:func:`coeffs_dir` are the CCIR/URSI coefficient directory.  ``_COEFFS_DIR``
is bound HERE only: the setup chain, the host path and the device iono cache
all read it as ``voacap_gpu.system._COEFFS_DIR`` at call time.
"""

import pathlib
import warnings
from dataclasses import dataclass

import numpy as np

from voacap_gpu import validate as _validate
from voacap_gpu.scalar.f32math import F32


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


def coeffs_dir() -> pathlib.Path:
    """The CCIR/URSI coefficient directory ``predict`` reads (a ``Path``).

    The packaged copy (``voacap_gpu/data/coeffs``), or the vendored oracle
    tree in a source checkout without it.  This is the public spelling of
    the private ``_COEFFS_DIR`` (still there, same value): pass it to
    ``scalar.redmap.redmap`` when building an ionosphere yourself.
    """
    return _COEFFS_DIR


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
