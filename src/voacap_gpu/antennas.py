"""Antennas: the per-circuit /cantenna/ (ANTCALC) and OFFAZIM.

Split out of :mod:`voacap_gpu.engine` (which still re-exports every name
here): isotropic, type-14 and type-13 ``.VOA`` patterns installed exactly as
READANT + ANTCALC leave them, plus the power / gain-offset tail.
"""

import numpy as np

from voacap_gpu.scalar.antennastate import CantennaState
from voacap_gpu.scalar.dazel import dazel0
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.readant import UnsupportedAntennaType, read_antenna


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


def _parse_antenna(path):
    """A ``.VOA`` path -> parsed AntennaFile (None stays None)."""
    if path is None:
        return None
    try:
        return read_antenna(path)
    except UnsupportedAntennaType as e:
        raise NotImplementedError(str(e)) from e
