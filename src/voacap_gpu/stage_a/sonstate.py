"""COMMON /SON/ and friends -- the per-FREQUENCY results table.

Where /ZON/ holds one row per mode at one frequency and /allMODE/ accumulates
those across hops, /SON/ holds one row per FREQUENCY: the summary RELBIL
distils from all the modes at that frequency, which the output routines then
print as a column.  Indexed by IF, the frequency index (up to 13 -- eleven
requested frequencies plus the MUF and FOT columns).

Bundled here because they are filled together and by the same routines:

    /SON/         the main table (below)
    /DUDL_NOIS/   DU_NOIS, DL_NOIS -- the noise deciles that were in force
    /sncom/       SNXX -- the required S/N at the requested percentile
    /cgains/      GAINT, GAINR -- the chosen mode's antenna gains
    /ALPHA/ MODE  the two-character layer label, from LAYTYP

D10R/D50R/D90R are a **deviation**: in the Fortran they live in /DON/ among
the path geometry, but only RELBIL writes them and only RELBIL and OUTALL's
diagnostic read them, so they are kept here with the outputs they feed rather
than dragging the whole of /DON/ into a state object.  They persist across
calls, and that is observable: when the only mode has a virtual height at or
below 70 km, RELBIL never assigns them and SNRLW/SNRUP/SNPR read whatever the
previous frequency left.  Faithfully preserved.

MDL, ANGLER, PROBMP and SPROB are filled by other routines (MPATH, SERPRB);
carried here for layout completeness.

Stored 1-based as (14,).
"""

from dataclasses import dataclass, field

import numpy as np

F32 = np.float32

NFREQ = 13

# blkdat.for:149 -- DATA LAYTYP /' E','F1','F2','ES',' N'/, indexed by NMODE
LAYTYP = (" E", "F1", "F2", "ES", " N")


def _z():
    return np.zeros(NFREQ + 1, dtype=np.float32)


def _labels():
    return [""] * (NFREQ + 1)


@dataclass
class SonState:
    # /SON/
    angle: np.ndarray = field(default_factory=_z)
    angler: np.ndarray = field(default_factory=_z)
    cprob: np.ndarray = field(default_factory=_z)
    dblos: np.ndarray = field(default_factory=_z)
    dblosl: np.ndarray = field(default_factory=_z)
    dblosu: np.ndarray = field(default_factory=_z)
    dbu: np.ndarray = field(default_factory=_z)
    delay: np.ndarray = field(default_factory=_z)
    dbw: np.ndarray = field(default_factory=_z)
    nhp: np.ndarray = field(
        default_factory=lambda: np.zeros(NFREQ + 1, dtype=np.int32))
    xnynois: np.ndarray = field(default_factory=_z)
    probmp: np.ndarray = field(default_factory=_z)
    reliab: np.ndarray = field(default_factory=_z)
    sndb: np.ndarray = field(default_factory=_z)
    snpr: np.ndarray = field(default_factory=_z)
    snrlw: np.ndarray = field(default_factory=_z)
    snrup: np.ndarray = field(default_factory=_z)
    sprob: np.ndarray = field(default_factory=_z)
    vhigh: np.ndarray = field(default_factory=_z)
    rneff: np.ndarray = field(default_factory=_z)
    mdl: list = field(default_factory=_labels)
    # /ALPHA/ MODE(13) -- the layer label for the most reliable mode
    mode: list = field(default_factory=_labels)
    # /DUDL_NOIS/
    du_nois: np.ndarray = field(default_factory=_z)
    dl_nois: np.ndarray = field(default_factory=_z)
    # /sncom/
    snxx: np.ndarray = field(default_factory=_z)
    # /cgains/
    gaint: np.ndarray = field(default_factory=_z)
    gainr: np.ndarray = field(default_factory=_z)
    # /DON/ -- see the module docstring; these PERSIST across calls
    d10r: F32 = F32(0.0)
    d50r: F32 = F32(0.0)
    d90r: F32 = F32(0.0)
