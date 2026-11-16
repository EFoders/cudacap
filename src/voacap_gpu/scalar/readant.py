"""READANT -- antenna file reader, table paths only (Module 5).

readant.for reads a .VOA antenna file: a description line, a parameter count
N, N free-form parameter lines, then a payload selected by
``jant = nint(parm(2))``:

===========  ==========================================================
jant         payload
===========  ==========================================================
1-10, 12     *parameters only* -- the pattern is synthesised downstream
             by ANTCALC/HARRIS.  **Out of scope** (method-of-moments).
11           gain table at 91 elevation angles (list-directed)
13           gain table 360 azimuths x 91 elevations
14           gain table 30 frequencies x (efficiency + 91 elevations)
===========  ==========================================================

Only the *table* types are ported: they are the runtime gain-lookup path that
:func:`voacap_gpu.scalar.gain.antenna_gain` consumes, and they need no
synthesis.  Type 14 is the point-to-point case and the primary target --
its 30x91 table plus per-frequency EFF is exactly GAIN's ARRAY/AEFF.
An unsupported type raises rather than returning a silently empty table.

The Fortran writes types 13/14 through direct-access scratch files
(``type13.tmp``/``type14.tmp``) that ANTCALC reads back.  That round trip is a
1990s memory-budget artifact with no effect on the values, so it is dropped;
the tables are built directly.

Records are fixed-column F-edit, not free-form -- see :func:`_f_edit`.
"""

import dataclasses

import numpy as np

from voacap_gpu.scalar.f32math import F32

NELEV = 91          # elevation samples, 0..90 degrees inclusive
NFREQ = 30          # frequency samples, 1..30 MHz
NAZIM = 360         # azimuth samples, 0..359 degrees
NPARM = 20          # /ccirparm/ parm(20)

TABLE_TYPES = (11, 13, 14)


class UnsupportedAntennaType(NotImplementedError):
    """A .VOA type whose pattern is synthesised, not tabulated.

    Types 1-10 and 12 carry only parameters; ANTCALC/HARRIS turn them into a
    pattern by method-of-moments, which is out of scope for this port.
    """


@dataclasses.dataclass
class AntennaFile:
    """One parsed .VOA antenna file."""

    description: str
    nparm: int
    # (20,) float32, /ccirparm/ parm.  For the table types, parm[0] (PARM(1),
    # "Max Gain dBi") is used for exactly one thing in this port: the offset
    # ANTCALC adds to every tabulated gain -- p2p_tables() (type 14) and
    # CantennaState.load_p2p_t13 (type 13), both skippable with
    # absolute=True.  Nothing else in src/ (scalar, batched or CUDA) reads it.
    # parm[1] is the type (JANT) and parm[2] the type-13 efficiency.
    parm: np.ndarray
    jant: int
    # exactly one of these is populated, per jant
    gain_type11: np.ndarray = None    # (91,)      float32
    gains360: np.ndarray = None       # (360, 91)  float32
    gains30: np.ndarray = None        # (30, 91)   float32
    eff30: np.ndarray = None          # (30,)      float32

    def p2p_tables(self, absolute=False):
        """ARRAY(30,91) and AEFF(30) as /cantenna/ holds them, for GAIN.

        **The table is RELATIVE to header parameter 1.**  ANTCALC adds parm(1)
        -- the file's "Max Gain dBi" -- to every tabulated gain
        (antcalc.for:291) and takes AEFF straight from the per-frequency
        efficiency.  Only that offset step belongs to ANTCALC; it lives here
        because it is inseparable from the table's meaning.

        ``absolute=True`` skips the parm(1) add, for tables that are already
        in absolute dBi (a common export convention that VOACAP would
        double-count: a flat 5 dB table with header 5 installs +10 dBi).

        All 30 rows are returned; which of them reach /cantenna/ is the
        ANTENNA card's MINFREQ..MAXFREQ (``CantennaState.load_p2p``).
        """
        if self.jant != 14:
            raise ValueError(f"p2p_tables needs a type 14 file, got {self.jant}")
        if absolute:
            return self.gains30.astype(np.float32), self.eff30
        return (self.gains30 + self.parm[0]).astype(np.float32), self.eff30

    def peak_gain_dbi(self, absolute=False) -> float:
        """The largest gain this file would install, dBi.

        max(table) + parm(1), or max(table) alone with ``absolute=True`` --
        over the whole table (every frequency row for type 14, every azimuth
        for type 13), so it does not depend on OFFAZIM or the frequency range.
        """
        table = {11: self.gain_type11, 13: self.gains360,
                 14: self.gains30}.get(self.jant)
        if table is None:
            raise ValueError(f"no gain table for antenna type {self.jant}")
        peak = float(np.max(table))
        return peak if absolute else peak + float(self.parm[0])


def _f_edit(record: str, col: int, width: int, decimals: int) -> F32:
    """One Fortran F-edit descriptor read at 1-based `col`.

    Two behaviours that a plain ``float()`` would get wrong:

    * **Short records pad with blanks** (PAD='YES', the default), so a field
      running past end-of-line reads as blank rather than raising.
    * **BLANK='NULL' is the default** for a formatted connection, so blanks
      *within* the field are ignored, not read as zeros: "  -1.7 " is -1.7 and
      an all-blank field is 0.

    A field with no '.' takes the format's implied decimal point, scaling by
    10**-decimals -- unused by the stock samples, which all carry an explicit
    point, but it is what the descriptor means.
    """
    field = record[col - 1:col - 1 + width]           # short slice == blank pad
    text = field.replace(" ", "")                     # BLANK='NULL'
    if not text:
        return F32(0.0)
    if "." in text:
        return F32(float(text))
    return F32(float(text) * 10.0 ** -decimals)


def _first_token(line: str) -> str:
    """Leading value of a list-directed read; the rest of the line is comment.

    The samples put a label after the value ("  14    [ 2] Antenna Type..:"),
    which list-directed input stops before once its item list is satisfied.
    """
    return line.replace(",", " ").split()[0]


def _read_gain_block(lines, start, ncols, col0, width, decimals, count):
    """Read `count` values in `ncols`-wide rows -- Fortran format reversion.

    ``(2x,f6.1,(t10,10f7.3))`` and ``(9x,10f7.3)`` both revert to a group
    holding 10 values, so a 91-value table spans ceil(91/10) = 10 records and
    the last record supplies a single value.  Returns (values, next_index).
    """
    out = np.empty(count, dtype=np.float32)
    i = 0
    while i < count:
        record = lines[start] if start < len(lines) else ""
        for j in range(min(ncols, count - i)):
            out[i + j] = _f_edit(record, col0 + j * width, width, decimals)
        i += min(ncols, count - i)
        start += 1
    return out, start


def parse_antenna(text: str) -> AntennaFile:
    """Parse .VOA antenna file content.  See :func:`read_antenna`."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    description = lines[0][:70]
    nparm = int(float(_first_token(lines[1])))
    parm = np.zeros(NPARM, dtype=np.float32)
    for i in range(nparm):
        parm[i] = F32(float(_first_token(lines[2 + i])))
    jant = int(np.rint(float(parm[1])))               # NINT(PARM(2))
    ant = AntennaFile(description=description, nparm=nparm, parm=parm, jant=jant)
    k = 2 + nparm

    if jant == 11:
        # read(lu,*) gain_type11 -- list-directed, 91 values across records
        vals = []
        while len(vals) < NELEV:
            vals += [float(t) for t in lines[k].replace(",", " ").split()]
            k += 1
        ant.gain_type11 = np.array(vals[:NELEV], dtype=np.float32)
    elif jant == 13:
        # read(lu,'(9x,10f7.3)') gains
        ant.gains360 = np.empty((NAZIM, NELEV), dtype=np.float32)
        for iang in range(NAZIM):
            ant.gains360[iang], k = _read_gain_block(
                lines, k, 10, 10, 7, 3, NELEV)
    elif jant == 14:
        # read(lu,'(2x,f6.1,(t10,10f7.3))') eff,gains
        ant.gains30 = np.empty((NFREQ, NELEV), dtype=np.float32)
        ant.eff30 = np.empty(NFREQ, dtype=np.float32)
        for ifreq in range(NFREQ):
            ant.eff30[ifreq] = _f_edit(lines[k], 3, 6, 1)   # 2x,f6.1
            ant.gains30[ifreq], k = _read_gain_block(
                lines, k, 10, 10, 7, 3, NELEV)
    else:
        raise UnsupportedAntennaType(
            f"antenna type {jant} is synthesised, not tabulated; only types "
            f"{TABLE_TYPES} are in scope")
    return ant


def read_antenna(path) -> AntennaFile:
    """Parse the .VOA antenna file at `path`.

    The stock sample files are CRLF; the parser accepts any line ending.
    """
    with open(path, "r", encoding="latin-1", newline="") as fh:
        return parse_antenna(fh.read())
