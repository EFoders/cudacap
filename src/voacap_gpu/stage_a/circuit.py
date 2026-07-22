"""One circuit's worth of state, bundled.

Every routine below LUFFY takes its commons explicitly, which keeps each one
honest about what it reads.  LUFFY is not a kernel but a *driver*, and it
touches fifteen commons plus a scatter of loose scalars; passing thirty
arguments would obscure rather than document.  So the driver level gets one
object holding the state blocks and the per-circuit scalars, and the leaf
signatures stay as they are.

The scalars are grouped by the COMMON they come from.  Two deserve comment:

* ``jlong`` is set by LUFFY (-1 short path, +1 long) and is carried only for
  completeness -- **no engine routine reads it**.  Every use in the original
  is inside an output routine (OUTBOD, OUTBOD2, OUTAREA, OUTCIRAF) choosing
  which columns to print.
* ``want_service_prob`` is the Fortran's ``LINBOT(14)``, an OUTPUT-selection
  flag that gates an engine computation: SERPRB runs only when the service
  probability line is going to be printed.  SETOUT sets it from METHOD, and
  it is true for the methods this port targets (20 and 25 both fill
  LINBOT(1..NBOD) with NBOD >= 21).  METHOD 24 would clear it.
"""

from dataclasses import dataclass, field

import numpy as np

from .allmodestate import AllModeState
from .antennastate import CantennaState
from .frqstate import FrqState
from .layerstate import LayerState
from .modesstate import ModesState
from .mufstate import MufState
from .noisestate import AnoisState, NoiseDiagState
from .raysstate import RaysState
from .redmap import IonoState
from .reflxstate import ReflxState
from .signalstate import SigdState, TonState
from .sonstate import SonState
from .zonstate import ZonState

F32 = np.float32


@dataclass
class Circuit:
    # --- the commons, one field per block ---------------------------------
    # `iono` alone has no default: it holds the coefficient maps, which only
    # REDMAP can fill.  Everything else starts zeroed the way a fresh COMMON
    # block does.
    iono: IonoState = None
    layers: LayerState = field(default_factory=LayerState)
    muf: MufState = field(default_factory=MufState)
    rays: RaysState = field(default_factory=RaysState)
    reflx: ReflxState = field(default_factory=ReflxState)
    modes: ModesState = field(default_factory=ModesState)
    zon: ZonState = field(default_factory=ZonState)
    allm: AllModeState = field(default_factory=AllModeState)
    son: SonState = field(default_factory=SonState)
    anois: AnoisState = field(default_factory=AnoisState)
    ndiag: NoiseDiagState = field(default_factory=NoiseDiagState)
    sigd: SigdState = field(default_factory=SigdState)
    ton: TonState = field(default_factory=TonState)
    cant: CantennaState = field(default_factory=CantennaState)
    frq: FrqState = field(default_factory=FrqState)

    # --- /DON/ path geometry and circuit requirements ---------------------
    gcd: F32 = F32(0.0)             # great-circle distance, radians
    gcdkm: F32 = F32(0.0)           # ... and km
    amind: F32 = F32(0.0)           # SANG's reduced minimum take-off angle, deg
    rsn: F32 = F32(0.0)             # required signal-to-noise, dB
    dmp: F32 = F32(0.0)             # max tolerable multipath delay, ms
    pmp: F32 = F32(0.0)             # max tolerable multipath power spread, dB
    rlat: F32 = F32(0.0)            # receiver latitude, radians
    rlong: F32 = F32(0.0)           # receiver longitude, radians
    rlongd: F32 = F32(0.0)          # ... and degrees

    # --- /ION/ ------------------------------------------------------------
    lufp: int = 90                  # required reliability, percent
    method: int = 20
    noise: int = 0                  # man-made noise category

    # --- /CONTRL/ ---------------------------------------------------------
    mspec: int = 0                  # 121 selects long/short path smoothing
    jlong: int = -1                 # output-only; see the module docstring

    # --- /TIME/ -----------------------------------------------------------
    it: int = 1                     # hour index, 1..24
    gmtr: F32 = F32(0.0)            # receiver mean time, hours

    # --- /ALPHA/ ----------------------------------------------------------
    nyear: str = ""                 # " 2089" selects the pre-1989 noise model

    # --- /OUTPRT/ LINBOT(14) ---------------------------------------------
    want_service_prob: bool = True
