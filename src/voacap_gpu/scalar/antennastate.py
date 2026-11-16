"""COMMON /cantenna/ and /Careaant/ -- the loaded antenna tables.

Up to 20 antennas are held at once; IATS(iat) says whether entry `iat` is the
transmitter (1) or the receiver (2), and XFQS/XFQE bracket the frequency range
it is valid over.  GAIN picks the first entry matching both.

ARRAY(30,91,20) is the point-to-point gain table -- dB at [frequency 1..30
MHz][elevation 0..90 deg] -- with AEFF(30,20) the matching per-frequency
efficiency.  READANT/ANTCALC fill them from a type 14 .VOA file.

OFFAZIM(iat) == -999 flags an *area coverage* antenna, whose pattern lives in
/Careaant/ IARRAY360(360,91,2) instead: 360 azimuths x 91 elevations, held as
INTEGER*2 hundredths of a dB (ANTSAVE stores NINT(gain*100) clamped to +-300
dB, so GAIN divides by 100 on the way out).

Stored 1-based throughout: ARRAY as (31,92,21), IARRAY360 as (361,92,3).
"""

import warnings
from dataclasses import dataclass, field

import numpy as np

F32 = np.float32

NANT = 20           # /cantenna/ entries
NFREQ = 30          # frequency samples, 1..30 MHz
NELEV = 91          # elevation samples, 0..90 degrees
NAZIM = 360         # azimuth samples, 0..359 degrees

AREA_COVERAGE = F32(-999.0)     # OFFAZIM sentinel

# An installed peak above this is almost certainly a table already in absolute
# dBi with its peak repeated in header parameter 1 (ANTCALC adds the two).
PEAK_WARN_DBI = 30.0


def freq_rows(fqs, fqe):
    """The ANTENNA card's MINFREQ..MAXFREQ as 1-based ARRAY row bounds.

    ANTCALC's table loops run ``do ifreq = minfreq, maxfreq`` over whole MHz,
    so both ends must be integral and inside the 1..30 MHz table.
    """
    lo, hi = float(fqs), float(fqe)
    if lo != int(lo) or hi != int(hi):
        raise ValueError(
            f"antenna frequency range ({fqs}, {fqe}) must be whole MHz "
            "(ANTCALC installs one table row per MHz)")
    lo, hi = int(lo), int(hi)
    if not 1 <= lo <= hi <= NFREQ:
        raise ValueError(
            f"antenna frequency range ({fqs}, {fqe}) must satisfy "
            f"1 <= MINFREQ <= MAXFREQ <= {NFREQ}")
    return lo, hi


def _warn_peak(ant, absolute):
    """UserWarning when a relative table + PARM(1) installs > PEAK_WARN_DBI."""
    if absolute:
        return
    peak = ant.peak_gain_dbi(absolute=False)
    if peak > PEAK_WARN_DBI:
        warnings.warn(
            f"antenna {ant.description.strip()!r} (type {ant.jant}) installs a "
            f"peak gain of {peak:.2f} dBi: table max "
            f"{ant.peak_gain_dbi(absolute=True):.2f} + header parameter 1 "
            f"{float(ant.parm[0]):.2f}.  VOACAP tables are RELATIVE to "
            "parameter 1; if this table is already in absolute dBi, pass "
            "table_absolute=True so the offset is not added twice.",
            UserWarning, stacklevel=4)


def _a20():
    return np.zeros(NANT + 1, dtype=np.float32)


@dataclass
class CantennaState:
    numants: int = 0
    iats: np.ndarray = field(
        default_factory=lambda: np.zeros(NANT + 1, dtype=np.int32))
    xfqs: np.ndarray = field(default_factory=_a20)
    xfqe: np.ndarray = field(default_factory=_a20)
    beammain: np.ndarray = field(default_factory=_a20)
    offazim: np.ndarray = field(default_factory=_a20)
    array: np.ndarray = field(default_factory=lambda: np.zeros(
        (NFREQ + 1, NELEV + 1, NANT + 1), dtype=np.float32))
    aeff: np.ndarray = field(default_factory=lambda: np.zeros(
        (NFREQ + 1, NANT + 1), dtype=np.float32))
    iarray360: np.ndarray = field(default_factory=lambda: np.zeros(
        (NAZIM + 1, NELEV + 1, 3), dtype=np.int16))
    # /pantenna/ -- transmitter power per entry, read by PWRDB
    pwrkw: np.ndarray = field(default_factory=_a20)
    pwrdba: np.ndarray = field(default_factory=_a20)

    def load_p2p(self, iat: int, ant, itr: int, fqs=2.0, fqe=30.0,
                 absolute=False) -> None:
        """Install a parsed type 14 file as entry `iat`, for TX (1) or RX (2).

        Mirrors what READANT + ANTCALC leave in /cantenna/, so a Stage A run can
        be set up without the .VOA card-deck parsing that is out of scope.

        ANTCALC's type-14 arm sets ``antgains(ifreq,..) = table + parm(1)`` for
        ``ifreq = MINFREQ..MAXFREQ`` only (the ANTENNA card's range, normally
        2..30); rows outside stay at their zero preset, and XFQS/XFQE bracket
        the same range so GAIN never selects this entry outside it.  ``fqs`` /
        ``fqe`` are that MINFREQ/MAXFREQ, whole MHz.  (Before this mirrored
        the Fortran, all 30 rows were installed with XFQS 0.)

        The table is RELATIVE to header parameter 1 (PARM(1), "Max Gain dBi"),
        which is added to every gain; ``absolute=True`` skips that add for a
        table already in absolute dBi.  A UserWarning flags an installed peak
        above ``PEAK_WARN_DBI`` when the add is applied.
        """
        ifs, ife = freq_rows(fqs, fqe)
        arr, aeff = ant.p2p_tables(absolute=absolute)
        _warn_peak(ant, absolute)
        # The gainNN.dat round trip, exactly as load_p2p_t13 models it:
        # ANTCALC clamps at -99.99 and writes f7.3 / f6.2 (antcalc.for:342-347),
        # DECRED reads them back (decred.for:335-336).  A table + PARM(1) sum
        # leaves the 3-decimal grid whenever PARM(1) != 0 (sample.14x: 511 of
        # 2730 cells moved 1-2 ULP without this); for PARM(1) == 0 it is the
        # identity.  Validated against ANTCALC's captured /cantenna/ in
        # tests/test_antenna_oracle.py.
        arr = np.maximum(arr, F32(-99.99))
        arr = (np.round(np.float64(arr) * 1000.0) / 1000.0).astype(np.float32)
        aeff = (np.round(np.float64(aeff) * 100.0) / 100.0).astype(np.float32)
        self.array[1:NFREQ + 1, 1:NELEV + 1, iat] = F32(0.0)
        self.aeff[1:NFREQ + 1, iat] = F32(0.0)
        # table row k (0-based) is frequency k+1 MHz == ARRAY row k+1
        self.array[ifs:ife + 1, 1:NELEV + 1, iat] = arr[ifs - 1:ife]
        self.aeff[ifs:ife + 1, iat] = aeff[ifs - 1:ife]
        self.iats[iat] = itr
        self.xfqs[iat] = F32(fqs)
        self.xfqe[iat] = F32(fqe)
        self.offazim[iat] = F32(0.0)
        self.numants = max(self.numants, iat)

    def load_p2p_t13(self, iat: int, ant, itr: int, offazim_deg,
                     fqs=2.0, fqe=30.0, absolute=False) -> None:
        """Install a parsed type 13 (azimuth-dependent) file for one OFFAZIM.

        ANTCALC's point-to-point type-13 arm (antcalc.for:268-284): linearly
        interpolate between the two 1-degree azimuth slices bracketing
        OFFAZIM, add PARM(1) (the file's "Max Gain dBi"), and install that
        single elevation cut for every frequency row MINFREQ..MAXFREQ (the
        ANTENNA card's range -- Prop-SL decks use 2..30, so row 1 stays the
        zero preset exactly as the Fortran leaves it).  The per-frequency
        efficiency is PARM(3) over the same rows.

        Unlike type 14, the installed table depends on the circuit geometry
        (OFFAZIM = path azimuth at this end minus the beam heading, from
        DAZEL0), so each distinct path direction needs its own /cantenna/.

        The values then round-trip through the gainNN.dat text file --
        ANTCALC clamps at -99.99 and writes f7.3 (antcalc.for:342-347),
        DECRED reads it back with (2x,f6.2,(t10,10f7.3)) (decred.for:335-336)
        -- so gains are quantized to 3 decimals and the efficiency to 2.
        Type 14 skips this modelling because its values never leave the
        3-decimal grid; the azimuth interpolation here does.

        As for type 14, the table is RELATIVE to header parameter 1:
        ``absolute=True`` skips the PARM(1) add for a table already in
        absolute dBi, and a UserWarning flags an installed peak (over the
        whole 360x91 table, so independent of OFFAZIM) above
        ``PEAK_WARN_DBI`` when the add is applied.
        """
        if ant.jant != 13:
            raise ValueError(f"load_p2p_t13 needs a type 13 file, got {ant.jant}")
        ifs, ife = freq_rows(fqs, fqe)
        _warn_peak(ant, absolute)
        off = F32(offazim_deg)
        # OFFAZIM can land on exactly 360.0 in REAL*4 (a path azimuth of
        # 360-eps minus a 0 beam rounds up); the direction is circularly
        # identical to 0.  The Fortran would read direct-access record 361
        # here -- one past the table -- so wrap rather than reproduce that.
        if off >= F32(360.0):
            off = F32(off - F32(360.0))
        iazim = int(off)                      # REAL->INT truncates toward 0
        iazim2 = iazim + 1
        if iazim2 == 360:
            iazim2 = 0
        g1 = ant.gains360[iazim]              # rec iazim+1, (91,) float32
        g2 = ant.gains360[iazim2]
        fract = F32(off - F32(float(iazim)))
        # antcalc.for:282 evaluation order: (g1 + (g2-g1)*fract) + parm(1)
        if absolute:
            g = g1 + (g2 - g1) * fract
        else:
            g = (g1 + (g2 - g1) * fract) + ant.parm[0]
        g = np.maximum(g, F32(-99.99))        # antcalc.for:343 floor
        # gainNN.dat f7.3 / f6.2 round trip (round-half-even, as gfortran
        # formats), landing back in float32 like DECRED's formatted READ.
        g = np.float64(g)
        g = (np.round(g * 1000.0) / 1000.0).astype(np.float32)
        eff = F32(np.round(np.float64(ant.parm[2]) * 100.0) / 100.0)
        self.array[1:NFREQ + 1, 1:NELEV + 1, iat] = F32(0.0)
        self.aeff[1:NFREQ + 1, iat] = F32(0.0)
        self.array[ifs:ife + 1, 1:NELEV + 1, iat] = g[np.newaxis, :]
        self.aeff[ifs:ife + 1, iat] = eff
        self.iats[iat] = itr
        self.xfqs[iat] = F32(fqs)
        self.xfqe[iat] = F32(fqe)
        self.offazim[iat] = off
        self.numants = max(self.numants, iat)
