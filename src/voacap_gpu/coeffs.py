"""Module 0: ingestion of the CCIR/URSI ionospheric coefficient sources.

Parses the ASCII coefficient files that voacapl ships in
``reference/voacapl/voacapl/itshfbc/coeffs/`` — the same files its
``coeffbin``/``fof2bin`` converter programs read — directly into numpy
arrays.  The converters are the format specification; layout details and
array shapes are documented in docs/coeff_format.md and validated bit-exact
against the oracle-generated binaries in golden/coeffs/.

Arrays keep their Fortran shapes and are reshaped with ``order='F'``, so
0-based numpy indices map one-to-one onto 1-based Fortran subscripts
(``dud[i-1, j-1, k-1]`` == ``DUD(i,j,k)``).  Values are float32 (int32 for
IKIM), exactly as the REAL*4/INTEGER declarations in the consumers.
"""

import pathlib

import numpy as np

# The READ statements of coeffbin.f in file order: (name-record tag,
# [(array, shape), ...]).  Each entry is ONE Fortran READ — a group like
# ANEW..BCHI shares lines, so it must be parsed as a single value stream.
COEFF_READS = [
    ("IKIM", [("ikim", (10, 6))]),
    ("DUD", [("dud", (5, 12, 5))]),
    ("FAM", [("fam", (14, 12))]),
    ("SYS", [("sys", (9, 16, 6))]),
    ("FAKP", [("fakp", (29, 16, 6))]),
    ("FAKABP", [("fakabp", (2, 6))]),
    ("XFM3CF", [("xfm3cf", (9, 49, 2))]),
    ("ANEW(3),BNEW(3),ACHI(2),BCHI(2)",
     [("anew", (3,)), ("bnew", (3,)), ("achi", (2,)), ("bchi", (2,))]),
    ("FAKMAP", [("fakmap", (29, 16))]),
    ("ABMAP", [("abmap", (2, 3))]),
    ("F2D", [("f2d", (16, 6, 6))]),
    ("PERR", [("perr", (9, 4, 6))]),
    ("XESMCF", [("xesmcf", (7, 61, 2))]),
    ("XPMAP", [("xpmap", (29, 16, 2))]),
    ("XESLCF", [("xeslcf", (5, 55, 2))]),
    ("XESUCF", [("xesucf", (5, 55, 2))]),
    ("XERCOF", [("xercof", (9, 22, 2))]),
]

FOF2_SHAPE = (13, 76, 2)  # XF2COF: diurnal Fourier x geographic fn x SSN level
_FOF2_N = 13 * 76 * 2  # 1976


def _floats_fixed(lines, start: int, count: int):
    """Parse `count` values in 6E15.8 layout (fixed 15-char fields — negative
    values fill the field, so whitespace splitting is unsafe).  Returns
    (float32 array in file order, next line index)."""
    vals = np.empty(count, dtype=np.float32)
    i = start
    k = 0
    while k < count:
        line = lines[i]
        n = min(6, count - k)
        for j in range(n):
            vals[k] = np.float32(float(line[15 * j:15 * (j + 1)]))
            k += 1
        i += 1
    return vals, i


def _reshape(vals: np.ndarray, shape) -> np.ndarray:
    return np.reshape(vals, shape, order="F")


def load_coeff_month(path) -> dict[str, np.ndarray]:
    """Parse one coeffXX.asc file into {array name: ndarray}.

    Includes ``"month"`` (int) from the header record.  Name records are
    checked against COEFF_READS just as coeffbin.f checks them.
    """
    lines = pathlib.Path(path).read_text().splitlines()
    if not lines[0].startswith("month="):
        raise ValueError(f"{path}: bad header {lines[0]!r}")
    out: dict[str, np.ndarray] = {"month": int(lines[0][6:8])}
    i = 1
    for tag, arrays in COEFF_READS:
        name_rec = lines[i]
        if not name_rec.startswith(tag):
            raise ValueError(f"{path}: expected {tag!r} at line {i + 1}, "
                             f"got {name_rec!r}")
        i += 1
        if tag == "IKIM":
            count = 60
            vals = np.empty(count, dtype=np.int32)
            k = 0
            while k < count:  # 10I5: fixed 5-char integer fields
                line = lines[i]
                for j in range(10):
                    vals[k] = int(line[5 * j:5 * (j + 1)])
                    k += 1
                i += 1
            out["ikim"] = _reshape(vals, (10, 6))
            continue
        total = sum(int(np.prod(s)) for _, s in arrays)
        vals, i = _floats_fixed(lines, i, total)
        pos = 0
        for name, shape in arrays:
            n = int(np.prod(shape))
            out[name] = _reshape(vals[pos:pos + n], shape)
            pos += n
    return out


def load_fof2_atlas(path) -> np.ndarray:
    """Parse fof2CCIR.asc or fof2URSI.asc → float32 (12, 13, 76, 2);
    first axis is month-1."""
    lines = pathlib.Path(path).read_text().splitlines()
    i = 1  # skip title
    months = []
    for month in range(1, 13):
        mon = int(lines[i][6:8])  # format(6x,i2)
        if mon != month:
            raise ValueError(f"{path}: month record {mon} != {month}")
        vals, j = _floats_fixed(lines, i + 1, _FOF2_N)
        months.append(_reshape(vals, FOF2_SHAPE))
        i = j
    return np.stack(months)


def load_fof2_daily(path) -> np.ndarray:
    """Parse fof2daly.asc → float32 (9, 13, 76, 2); first axis is the
    coefficient-set index (KFOF2R-1).  Values are free-format (the Fortran
    reads them list-directed), so tokens are whitespace-split."""
    lines = pathlib.Path(path).read_text().splitlines()
    i = 1  # skip title
    sets = []
    for _ in range(9):
        i += 1  # per-set title line
        vals = np.empty(_FOF2_N, dtype=np.float32)
        k = 0
        while k < _FOF2_N:
            for tok in lines[i].split():
                vals[k] = np.float32(float(tok))
                k += 1
            i += 1
        sets.append(_reshape(vals, FOF2_SHAPE))
    return np.stack(sets)
