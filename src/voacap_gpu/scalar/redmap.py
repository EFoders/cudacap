"""Stage A of REDMAP (reference/voacapl/src/voacapw/redmap.for).

Loads the monthly ionospheric coefficient sets and linearly blends the
two solar-activity levels to the target sunspot number, in REAL*4 exactly
as the Fortran does.  The Fortran reads the converted binaries
(coeffXXw.bin / fof2*.daw); this port reads the vendored ASCII sources
through voacap_gpu.coeffs — proven byte-identical to those binaries by
tests/test_coeffs.py — so no binary files are needed at runtime.

Level pairs per set (the divisor is the level span):
  F2COF, FM3COF:            R12 =   0 / 100   -> /100
  ESLCOF, ESMCOF, ESUCOF,
  ERCOF:                    R12 =  10 / 150   -> /140
  HMYM (from XPMAP):        R12 =  25 / 125   -> /100

Also carries the /ONE/ bookkeeping constants IA/IB (COFION offsets and
strides, voacapw.for DATA IIA/IIB) and IC (AB offsets, virtim.for/
versy.for DATA) used by the downstream evaluations.
"""

import pathlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from voacap_gpu.coeffs import load_coeff_month, load_fof2_atlas

F32 = np.float32

# COFION start offset and first-dimension stride per set (1-based), and
# the AB(318) start offsets. Set order: ESL, ESM, ESU, F2, FM3, ER.
IA = (1, 276, 703, 978, 1966, 2407)
IB = (5, 7, 5, 13, 9, 9)
IC = (1, 56, 117, 172, 248, 297)


@dataclass
class IonoState:
    """The /ONE/, /FONE/ and /TWO/ contents after REDMAP (float32)."""

    ikim: np.ndarray      # (10,6) int32
    eslcof: np.ndarray    # (5,55)
    esmcof: np.ndarray    # (7,61)
    esucof: np.ndarray    # (5,55)
    f2cof: np.ndarray     # (13,76)
    fm3cof: np.ndarray    # (9,49)
    ercof: np.ndarray     # (9,22)
    cofion: np.ndarray    # (2604,) — /ONE/ overlay ESLCOF..ERCOF, F-order
    anew: np.ndarray
    bnew: np.ndarray
    achi: np.ndarray
    bchi: np.ndarray
    f2d: np.ndarray       # (16,6,6)
    fakp: np.ndarray      # (29,16,6)
    fakmap: np.ndarray    # (29,16)
    hmym: np.ndarray      # (29,16) — SSN-blended XPMAP
    fakabp: np.ndarray
    abmap: np.ndarray
    dud: np.ndarray
    fam: np.ndarray
    sys: np.ndarray
    perr: np.ndarray


@lru_cache(maxsize=8)
def _load_month(coeffs_dir: str, coeff: str, month: int):
    d = pathlib.Path(coeffs_dir)
    arrays = load_coeff_month(d / f"coeff{month:02d}.asc")
    atlas = load_fof2_atlas(d / f"fof2{coeff}.asc")
    return arrays, atlas[month - 1]


def redmap(coeffs_dir, coeff: str, month: int, ssn) -> IonoState:
    """coeff is 'CCIR' or 'URSI' (the Fortran defaults non-URSI to CCIR)."""
    if coeff != "URSI":
        coeff = "CCIR"
    a, xf2cof = _load_month(str(coeffs_dir), coeff, month)
    ssn = F32(ssn)

    ssn100 = F32(100.0) - ssn
    f2cof = (xf2cof[:, :, 0] * ssn100 + xf2cof[:, :, 1] * ssn) / F32(100.0)
    fm3cof = (a["xfm3cf"][:, :, 0] * ssn100
              + a["xfm3cf"][:, :, 1] * ssn) / F32(100.0)
    ssn150 = F32(150.0) - ssn
    ssn10 = ssn - F32(10.0)
    esmcof = (a["xesmcf"][:, :, 0] * ssn150
              + a["xesmcf"][:, :, 1] * ssn10) / F32(140.0)
    eslcof = (a["xeslcf"][:, :, 0] * ssn150
              + a["xeslcf"][:, :, 1] * ssn10) / F32(140.0)
    esucof = (a["xesucf"][:, :, 0] * ssn150
              + a["xesucf"][:, :, 1] * ssn10) / F32(140.0)
    ercof = (a["xercof"][:, :, 0] * ssn150
             + a["xercof"][:, :, 1] * ssn10) / F32(140.0)
    ssn125 = F32(125.0) - ssn
    ssn25 = ssn - F32(25.0)
    hmym = (a["xpmap"][:, :, 0] * ssn125
            + a["xpmap"][:, :, 1] * ssn25) / F32(100.0)

    cofion = np.concatenate([
        eslcof.ravel(order="F"), esmcof.ravel(order="F"),
        esucof.ravel(order="F"), f2cof.ravel(order="F"),
        fm3cof.ravel(order="F"), ercof.ravel(order="F"),
    ])
    assert cofion.shape == (2604,) and cofion.dtype == np.float32

    return IonoState(
        ikim=a["ikim"], eslcof=eslcof, esmcof=esmcof, esucof=esucof,
        f2cof=f2cof, fm3cof=fm3cof, ercof=ercof, cofion=cofion,
        anew=a["anew"], bnew=a["bnew"], achi=a["achi"], bchi=a["bchi"],
        f2d=a["f2d"], fakp=a["fakp"], fakmap=a["fakmap"], hmym=hmym,
        fakabp=a["fakabp"], abmap=a["abmap"], dud=a["dud"], fam=a["fam"],
        sys=a["sys"], perr=a["perr"],
    )
