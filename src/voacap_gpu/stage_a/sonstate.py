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

The two decile triples are a **deviation**: in the Fortran D10R/D50R/D90R and
D10S/D50S/D90S live in /DON/ among the path geometry, but the R triple is
written only by RELBIL and the S triple only by SERPRB, and the only other
reader of either is OUTALL's diagnostic line.  They are kept here with the
outputs they feed rather than dragging the whole of /DON/ into a state
object.  They persist across calls, and for the R triple that is observable:
when the only mode has a virtual height at or below 70 km, RELBIL never
assigns them and SNRLW/SNRUP/SNPR read whatever the previous frequency left.
Faithfully preserved.

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
    # /ALPHA/ MODER(13) -- the same for the RECEIVER end, which only the long
    # path fills: the short path has one mode with one layer, so MODE says it
    # all, but a long-path mode arrives at a different layer than it left.
    moder: list = field(default_factory=_labels)
    # /DUDL_NOIS/
    du_nois: np.ndarray = field(default_factory=_z)
    dl_nois: np.ndarray = field(default_factory=_z)
    # /sncom/
    snxx: np.ndarray = field(default_factory=_z)
    # /cgains/
    gaint: np.ndarray = field(default_factory=_z)
    gainr: np.ndarray = field(default_factory=_z)
    # /DON/ -- see the module docstring; these PERSIST across calls.
    # The R triple is RELBIL's reliability spread, the S triple SERPRB's
    # service-probability spread (where D90S is simply set equal to D10S).
    d10r: F32 = F32(0.0)
    d50r: F32 = F32(0.0)
    d90r: F32 = F32(0.0)
    d10s: F32 = F32(0.0)
    d50s: F32 = F32(0.0)
    d90s: F32 = F32(0.0)
