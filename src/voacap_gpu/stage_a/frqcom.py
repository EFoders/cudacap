"""FRQCOM — synthesise a frequency complement from 2 to 40 MHz.

Called when the user gave no FREQUENCY card.  The job is to lay eleven
frequencies across the usable band in a way that puts one of them *on* the
lower MUF, because that is where the prediction changes fastest: below the
E/F2 MUF the mode is absorbed, above it the mode exists.  Slot 12 always gets
the circuit MUF and slot 13 is zeroed to say "no LUF computed yet".

The band is bracketed by two numbers:

    XFL = max(2, min(EMUF, F2MUF))    the lower MUF -- the knee to straddle
    XFH = min(HPF, 40)                the highest usable frequency

and the six cases are ordered by how much room there is between them.  Only
five can run.  **CASE 6 IS DEAD**: it guards on ``(FHIGH-2.)-FLOW``, and FLOW
and FHIGH are the DATA constants 2 and 40, so that expression is the constant
36 and the arithmetic IF can only ever take its positive arm into case 5.
Transliterated as an unreachable branch rather than dropped, so the case
numbering still matches the source.

Two asymmetries in the original are load-bearing and preserved:

* **case 1 does not zero slot 13** where cases 2-5 all do.  It is the
  "not likely to occur" arm (XFH <= 2 MHz, i.e. no usable band at all) and it
  leaves whatever LUF was there from the previous hour.
* case 3 can compute an index past the end of the array.  See ``_CASE3_MAX``.
"""

import numpy as np

from .f32math import F32

_FLOW = F32(2.0)            # DATA FLOW/2./
_FHIGH = F32(40.0)          # DATA FHIGH/40./
_I_MUF = 12
_I_LUF = 13
_CASE3_MAX = 12             # largest NE whose NE+1 write still lands in FREA


class FrequencyIndexError(IndexError):
    """Case 3 addressed FREA past its 13 elements.

    NE is ``(XFL-FLOW)/DELF`` truncated, and nothing bounds it: DELF comes
    from XFH while NE's numerator comes from XFL, so a circuit whose lower
    MUF sits well ABOVE its highest usable frequency drives NE arbitrarily
    high and the Fortran writes off the end of the common block.  Refused
    here rather than reproduced.
    """


def frqcom(muf, frea, ifreq: int, it: int) -> None:
    """Fill `frea` slots 1..11 (and 12, 13) in place.

    muf: MufState (reads ALLMUF/FOT/EMUF/F2MUF/HPF at hour `it`); frea: the
    1-based array to fill -- LUFFY passes /FRQ/ FREA, the hour driver passes
    FREL, which is longer and otherwise identical; ifreq: DECRED's flag,
    where -10 means "slot 1 only"; it: hour index.
    """
    # .....INSERT MUF INTO FREQUENCY SLOT 12
    frea[_I_MUF] = muf.allmuf[it]
    if ifreq == -10:                        # IF(IFREQ + 10) 95, 90, 95
        # .....INSERT FOT INTO FREQUENCY SLOT 1
        frea[1] = muf.fot[it]
        return

    xfl = min(muf.emuf[it], muf.f2muf[it])
    xfh = muf.hpf[it]
    xfl = max(_FLOW, xfl)
    xfh = min(xfh, _FHIGH)

    if xfh <= _FLOW:
        # .....CASE 1.  NOT LIKELY TO OCCUR -- and note slot 13 is left alone
        frea[1] = _FLOW
        for i in range(2, 12):
            frea[i] = frea[i - 1] + F32(2.0)
        return

    if xfl <= _FLOW:
        # .....CASE 2.  NIGHTTIME -- the whole band is above the lower MUF
        delf = (xfh - _FLOW) / F32(11.0)
        frea[1] = _FLOW
        for i in range(2, 12):
            frea[i] = frea[i - 1] + delf
        frea[_I_LUF] = F32(0.0)
        return

    if xfh <= _FLOW + F32(20.0):
        # .....CASE 3.  INSERT SMALLER MUF, WITHIN EQUAL INCREMENTS
        # .....RANGE .LE. 20 MHZ
        delf = (xfh - _FLOW) / F32(9.0)
        ne = int((xfl - _FLOW) / delf)      # REAL->INT truncates toward zero
        ne = ne + 2
        if ne > _CASE3_MAX:
            raise FrequencyIndexError(
                f"FRQCOM case 3 wants FREA({ne + 1}): XFL={float(xfl)} MHz "
                f"is above XFH={float(xfh)} MHz, so the step {float(delf)} "
                "does not span the gap")
        frea[1] = _FLOW
        for i in range(2, ne + 1):
            frea[i] = frea[i - 1] + delf
        frea[ne] = xfl                      # the knee, planted on slot NE
        frea[ne + 1] = frea[ne - 1] + delf  # resume the ladder, ignoring XFL
        ne = ne + 2
        ne = min(ne, 11)
        for i in range(ne, 12):
            frea[i] = frea[i - 1] + delf
        frea[_I_LUF] = F32(0.0)
        return

    if xfl <= _FLOW + F32(2.0):
        # .....CASE 4.  INSERT LOWER MUF BETWEEN 2 AND 4 MHZ
        # .....THEN EQUAL INCREMENTS FROM 4 MHZ TO 40 MHZ
        delf = (xfh - (_FLOW + F32(2.0))) / F32(9.0)
        frea[1] = _FLOW
        frea[2] = xfl
        frea[3] = _FLOW + F32(2.0)
        for i in range(4, 12):
            frea[i] = frea[i - 1] + delf
        frea[_I_LUF] = F32(0.0)
        return

    # IF( (FHIGH-2.) -FLOW) 185,185,166 -- constant 36, so case 6 is dead.
    # .....CASE 5.  EQUAL INCREMENTS TO LOWER MUF, THEN EQUAL INCREMENTS TO
    # .....UPPER LIMIT
    ne = int((xfl - _FLOW) / F32(2.0))      # >= 1 here: case 5 needs XFL > 4
    ne = min(ne, 7)
    xne = F32(ne)
    delf = (xfl - _FLOW) / xne
    frea[1] = _FLOW
    for i in range(2, ne + 1):
        frea[i] = frea[i - 1] + delf
    xne = F32(ne)
    delf = (xfh - frea[ne]) / (F32(12.0) - xne - F32(1.0))
    ne = ne + 1
    for i in range(ne, 12):
        frea[i] = frea[i - 1] + delf
    frea[_I_LUF] = F32(0.0)
