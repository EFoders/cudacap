"""Input validation at the public boundary (``engine.predict`` and friends).

The ported chain is Fortran-faithful, which means it inherits the Fortran's
habit of computing *something* for any input: 0.5 MHz returns 0 dBW with a
success status, 60 MHz is extrapolated off the GAIN table, month 13 dies in a
file lookup, hour 12.7 is silently hour 12 and a man-made noise of -145 (dBW
passed with a sign) silently selects the 'remote' category.  These checks turn
each of those into a ``ValueError`` that names the argument, BEFORE any of it
reaches the chain.  They never alter a valid input, so they cannot move a
validated number.

Fortran-faithful truncations that are kept (``int()`` on ``man_made_noise``
and ``required_reliability``, exactly as the SYSTEM card's integer fields
read) are reported with a ``UserWarning`` instead.
"""

import math
import warnings

import numpy as np

# The GAIN table's frequency axis is 1..30 MHz (one row per integer MHz), so a
# frequency outside it is an extrapolation the program never validated.
FREQ_MIN_MHZ = 1.0
FREQ_MAX_MHZ = 30.0
MAX_FREQS = 11                  # the FREL card's capacity

# SSN upper bound.  POLICY, NOT PHYSICS: the coefficient maps are linear in
# SSN and accept anything, but the CCIR/URSI fits were made over observed
# solar cycles (smoothed SSN has peaked a little above 250 on the V2 scale),
# so beyond this the answer is an unvalidated extrapolation.  Raise it if a
# caller has a reason; negative SSN is meaningless and stays refused.
SSN_MAX = 300.0

# man_made_noise <= 0 is a CCIR category code (genois.py:167-180):
#   0 -> 4 (as the card's "-164 dBW"), -1 industrial, -2 residential,
#   -3 rural, -4 remote.  Anything more negative is clamped to 4 by MIN0(4,MA)
#   -- which is how "-145" silently became "remote".
NOISE_CATEGORIES = (0, -1, -2, -3, -4)

_COEFFS = ("CCIR", "URSI")


def _fail(msg):
    raise ValueError(msg)


def _real(name, v):
    """``v`` as a finite float, or a ValueError naming ``name``."""
    if isinstance(v, (bool, np.bool_)):
        _fail(f"{name} must be a real number, got {v!r}")
    try:
        f = float(v)
    except (TypeError, ValueError):
        _fail(f"{name} must be a real number, got {v!r}")
    if not math.isfinite(f):
        _fail(f"{name} must be finite, got {v!r}")
    return f


def _integer(name, v, lo, hi):
    """``v`` as an int in ``[lo, hi]``.

    Accepts Python/numpy integers and integral floats (``12.0``); refuses
    non-integral values (``12.7``) rather than truncating them.
    """
    f = _real(name, v)
    if f != math.floor(f):
        _fail(f"{name} must be an integer, got {v!r}")
    i = int(f)
    if not lo <= i <= hi:
        _fail(f"{name} must be in {lo}..{hi}, got {v!r}")
    return i


def check_freqs(freqs):
    """Frequency list: 1..11 entries (NotImplementedError, as before), each a
    finite MHz value in ``[1, 30]`` (ValueError).  Returns float list."""
    freqs = list(freqs)
    if not 1 <= len(freqs) <= MAX_FREQS:
        raise NotImplementedError(
            f"{len(freqs)} frequencies: 1..{MAX_FREQS} are supported per call")
    out = []
    for f in freqs:
        x = _real("frequency", f)
        if not FREQ_MIN_MHZ <= x <= FREQ_MAX_MHZ:
            _fail(f"frequency {f!r} MHz is outside {FREQ_MIN_MHZ:g}.."
                  f"{FREQ_MAX_MHZ:g} MHz (the GAIN table's axis)")
        out.append(x)
    return out


def check_month(month):
    """Month as an int in 1..12."""
    return _integer("month", month, 1, 12)


def check_hour(hour):
    """One UTC hour as an int in 1..24 (1-based, as LUFFY's IT)."""
    return _integer("hour", hour, 1, 24)


def check_hours(hours):
    """A non-empty list of UTC hours, each an int in 1..24."""
    out = [check_hour(h) for h in hours]
    if not out:
        _fail("hours is empty; pass at least one UTC hour in 1..24")
    return out


def check_ssn(ssn):
    """Smoothed sunspot number: finite, in ``[0, SSN_MAX]``."""
    s = _real("ssn", ssn)
    if not 0.0 <= s <= SSN_MAX:
        _fail(f"ssn {ssn!r} is outside 0..{SSN_MAX:g} (the upper bound is a "
              "policy limit on extrapolating the coefficient fits)")
    return s


def wrap_lon(lon):
    """Longitude into [-180, 180] by whole turns, as PropServer's
    ``normaliseLon`` and :func:`voacap_gpu.gridjob.normalise_lon` do: subtract
    or add 360 until inside, keeping BOTH endpoints (+180 stays +180).  A value
    already inside is returned unchanged, so in-range inputs stay bit-identical.
    Works on floats and on numpy arrays (returns a new array)."""
    if isinstance(lon, np.ndarray):
        out = np.array(lon, copy=True)
        hi = out > 180.0
        out[hi] -= 360.0 * np.ceil((out[hi] - 180.0) / 360.0)
        lo = out < -180.0
        out[lo] += 360.0 * np.ceil((-180.0 - out[lo]) / 360.0)
        return out
    if lon > 180.0:
        return lon - 360.0 * math.ceil((lon - 180.0) / 360.0)
    if lon < -180.0:
        return lon + 360.0 * math.ceil((-180.0 - lon) / 360.0)
    return lon


def check_point(name, p):
    """A ``(lat_deg, lon_deg)`` endpoint: lat in [-90, 90], both finite.

    Returns the endpoint to USE: ``p`` itself when the longitude is already in
    [-180, 180], otherwise ``(lat, wrapped_lon)``.  Longitude is circular, so an
    out-of-range value (a web map panned past the date line reports 190) is
    wrapped rather than refused; VOACAP's CIRCUIT card can only express
    -180..180 anyway, so the wrapped value is what the Fortran would be given.
    Latitude is not circular and stays a hard ``ValueError``."""
    try:
        lat, lon = p
    except (TypeError, ValueError):
        _fail(f"{name} must be a (lat_deg, lon_deg) pair, got {p!r}")
    la = _real(f"{name} latitude", lat)
    lo = _real(f"{name} longitude", lon)
    if not -90.0 <= la <= 90.0:
        _fail(f"{name} latitude {lat!r} is outside -90..90 degrees")
    if -180.0 <= lo <= 180.0:
        return p
    return (lat, wrap_lon(lo))


def check_coeff(coeff):
    """'CCIR' or 'URSI', exactly.  REDMAP reads anything that is not the
    exact string 'URSI' as CCIR, so a lower-case 'ursi' would silently run
    CCIR; refuse it instead."""
    if coeff not in _COEFFS:
        _fail(f"coeff={coeff!r}: must be 'CCIR' or 'URSI' (upper case; "
              "anything else would silently run CCIR)")
    return coeff


def check_multipath_aliases(system):
    """Refuse a System that sets a deprecated multipath name AND a new one.

    The deprecated names are crossed (``max_delay_ms`` feeds PMP, the power
    tolerance in dB; ``max_spread_db`` feeds DMP, the delay in ms), so a mix
    of old and new names is ambiguous under either reading of "the same
    quantity" -- by name or by slot.  Any mix is refused.
    """
    old = [n for n in ("max_delay_ms", "max_spread_db")
           if getattr(system, n) is not None]
    new = [n for n in ("multipath_delay_ms", "multipath_power_db")
           if getattr(system, n) is not None]
    if old and new:
        _fail(f"System sets both deprecated {', '.join(old)} and "
              f"{', '.join(new)}; use only multipath_delay_ms (DMP, ms) and "
              "multipath_power_db (PMP, dB).  The deprecated names are "
              "CROSSED: max_delay_ms feeds PMP (dB) and max_spread_db feeds "
              "DMP (ms)")
    return old


def _truncated(name, v, seen):
    """int(v), warning once per System state if that discards a fraction."""
    i = int(float(v))
    if i != float(v) and (name, v) not in seen:
        warnings.warn(
            f"System.{name}={v!r} is not an integer; it is truncated to {i} "
            "as the Fortran SYSTEM card reads it", UserWarning, stacklevel=4)
    return i


def check_system(system):
    """Validate a :class:`~voacap_gpu.engine.System`; return it unchanged.

    (The only thing recorded on it is a private warn-once marker.)

    Raises ``ValueError`` for an unusable value; warns (``UserWarning``) when
    a Fortran-faithful ``int()`` truncation discards a fraction.  Each
    distinct truncation warns once per System object, so ``predict`` (which
    re-checks inside ``build_circuit`` per hour) warns once, at the caller.
    """
    _real("System.required_snr", system.required_snr)
    _real("System.min_takeoff_deg", system.min_takeoff_deg)
    _real("System.tx_gos", system.tx_gos)
    _real("System.rx_gos", system.rx_gos)

    p = _real("System.power_kw", system.power_kw)
    if not p > 0.0:
        _fail(f"System.power_kw must be > 0 kW, got {system.power_kw!r}")

    r = _real("System.required_reliability", system.required_reliability)
    if not 1.0 <= r <= 99.0:
        _fail("System.required_reliability must be a percentile in 1..99, "
              f"got {system.required_reliability!r}")

    n = _real("System.man_made_noise", system.man_made_noise)
    if n <= 0.0:
        if n != math.floor(n) or int(n) not in NOISE_CATEGORIES:
            _fail(f"System.man_made_noise={system.man_made_noise!r}: values "
                  "<= 0 are CCIR category codes and only 0, -1 (industrial), "
                  "-2 (residential), -3 (rural) and -4 (remote) exist.  A "
                  "level is a POSITIVE -dBW/Hz at 3 MHz -- did you pass dBW "
                  "with a sign? pass 145, not -145")
    elif n < 1.0:
        _fail(f"System.man_made_noise={system.man_made_noise!r} would "
              "truncate to 0, which is a category code, not a level; a "
              "level is a positive -dBW/Hz at 3 MHz (e.g. 145)")

    fp = tuple(system.fprob)
    if len(fp) != 4:
        _fail(f"System.fprob must have 4 multipliers (E, F1, F2, Es), got {fp!r}")
    for k, x in enumerate(fp):
        _real(f"System.fprob[{k}]", x)

    check_multipath_aliases(system)
    for nm in ("multipath_delay_ms", "multipath_power_db", "max_delay_ms",
               "max_spread_db"):
        v = getattr(system, nm)
        if v is not None:
            _real(f"System.{nm}", v)

    seen = system.__dict__.get("_truncation_warned", frozenset())
    keys = set(seen)
    for nm in ("man_made_noise", "required_reliability"):
        v = getattr(system, nm)
        _truncated(nm, v, seen)
        keys.add((nm, v))
    system.__dict__["_truncation_warned"] = frozenset(keys)
    return system


def check_model_conflict(model_name, model_coeff, model_fprob, coeff, system):
    """Refuse a ``model`` toggle that contradicts an explicit coeff / FPROB.

    ``coeff`` equal to the default 'CCIR' (or None) counts as "not given",
    since ``predict``'s signature cannot tell an explicit CCIR from the
    default.  ``system.fprob`` at the default ``(1, 1, 1, 0)`` likewise counts
    as not given.
    """
    if coeff not in (None, "CCIR", model_coeff):
        _fail(f"model={model_name!r} uses coeff={model_coeff!r} but "
              f"coeff={coeff!r} was given; pass one or the other")
    if system is not None:
        fp = np.asarray(tuple(system.fprob), np.float32)
        default = np.asarray((1.0, 1.0, 1.0, 0.0), np.float32)
        want = np.asarray(model_fprob, np.float32)
        if fp.shape == (4,) and not (np.array_equal(fp, default)
                                     or np.array_equal(fp, want)):
            _fail(f"model={model_name!r} sets FPROB {tuple(model_fprob)} but "
                  f"System(fprob={tuple(system.fprob)}) was given; drop the "
                  "model or the fprob")
