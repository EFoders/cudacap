"""``Results``: the per-(hour, frequency) answer, and /SON/ -> Results.

Split out of :mod:`voacap_gpu.engine` (which still re-exports every name
here).  :func:`_assemble` distils host-path Circuits, :func:`_results_from_son`
a packed device /SON/ table; both finish in :func:`_finish_results`.
"""

from dataclasses import dataclass, field

import numpy as np

from voacap_gpu.scalar.sonstate import LAYTYP


# LUFFY's /SON/ MODE layer labels (blkdat LAYTYP), for reference.
_MODE_LABELS = (" E", "F1", "F2", "ES", " N")


# Frequency match tolerance for Results.at, MHz.  ``freqs`` is float32, so a
# requested 7.1 comes back as 7.0999999; 1e-4 MHz (100 Hz) absorbs that and
# nothing a caller would mean as a different frequency.
_FREQ_ATOL = 1e-4


def _valid_mask(nhp, status):
    """Which (hour, freq) cells LUFFY actually computed; see :class:`Results`."""
    nhp = np.asarray(nhp)
    ok_hour = np.asarray(status).reshape(-1, 1) == 0
    return (nhp >= 1) & ok_hour


@dataclass
class Results:
    """Per-(hour, frequency) HF-circuit predictions.

    Axes: ``hours`` (1-based UTC, shape ``(H,)``) and ``freqs`` (MHz, float32,
    shape ``(F,)``).  The 2-D tables are ``(H, F)``:

        snr       median signal-to-noise ratio in a 1 Hz bandwidth, dB-Hz
                  (dB re 1 Hz).  The noise is computed per hertz (GENOIS
                  references kTb at 290 K in 1 Hz, ``scalar/genois.py``
                  ``_KTB``), so this is NOT the S/N in the receiver's
                  bandwidth: subtract ``10*log10(bandwidth_hz)`` for that.
        dbw       median signal power at the receiver, dBW
        reliab    circuit reliability, a fraction 0..1 (not percent)
        field_dbu median field strength at the receiver, dB(uV/m)
        nhp       number of hops of the dominant mode (>= 1 where valid)
        mode      2-char reflecting-layer label ('F2', ' E', ...) or ''
        valid     bool: True where LUFFY computed this cell (see below)

    The per-hour 1-D vectors are ``muf`` (circuit MUF, MHz), ``fot`` and
    ``hpf`` (the 90 % / 10 % decile MUFs, MHz), ``luf`` (see
    :attr:`lowest_reliable_freq`) and ``status`` (int32, see below).

    VALID AND NOT-COMPUTED CELLS
    ----------------------------
    ``valid[h, f]`` is False where LUFFY did not produce an answer for that
    cell, and there ``snr``, ``dbw``, ``reliab`` and ``field_dbu`` are NaN,
    ``mode`` is ``''`` and ``nhp`` keeps its raw value (``-1`` or ``0``).  A
    cell is valid iff ``status[h] == 0`` AND ``nhp[h, f] >= 1``:

    * On the system-performance arms ``predict`` uses (IPFLAG 100 / 200),
      LUFFY presets ``NHP(IFX) = -1`` at the top of every frequency slot
      (``luffy.cuh`` "PRESET IN COMMON/SON/").  The ONLY writer of NHP after
      that is RELBIL's tail (or the smoothing blend copying a saved RELBIL
      column), which writes ``INT(HN)`` of the chosen mode, and every mode
      that reaches /allMODE/ has HN >= 1 (a hop count; LNGPAT clamps it with
      ``MAX(1., ...)``).  So ``nhp >= 1`` <=> RELBIL ran for that slot.  The
      skip paths -- "no frequency in this slot" (FREL <= 0) and RELBIL's own
      NMMOD <= 0 no-op -- leave -1 and the /SON/ presets (0 dB, 0 dBW, ...),
      which look like data but are not.
    * A non-zero ``status`` means the kernel stopped part-way through the
      hour.  Slots after the failing one are never reached (their NHP is the
      fresh circuit's 0), and the slots before it are NOT the final answer
      either: on a 7000-10000 km path the short/long blend runs only after
      the whole loop, so they hold one model's unblended pass.  The whole
      hour is therefore invalid.

    VERY NEGATIVE VALUES ARE REAL, NOT MISSING
    ------------------------------------------
    Above the MUF VOACAP still computes an over-the-MUF mode, whose loss
    grows steeply with the frequency ratio, so valid cells can hold e.g.
    ``dbw`` of -500 to -650 dBW, ``snr`` of -380 dB-Hz and ``reliab`` around
    5.8e-6.  These ARE what VOACAP computes and match the Fortran oracle; they
    mean "no usable signal".  Two of them are floors rather than physics:
    ``dbw``/``field_dbu`` of exactly -500 is RELBIL's sentinel when the
    multi-mode power sum underflows, and ``reliab`` ~5.84e-6 is FNORML's
    5-sigma clamp (``scalar/fnorml.py``).  They are deliberately left as is;
    use ``valid`` for "not computed", and a threshold of your own for "no
    signal".

    STATUS CODES (per hour)
    -----------------------
    ``status[h]`` is the LUFFY kernel's return for that hour's circuit
    (``cuda/driver.py``).  0 = success.  -1 = the kernel never wrote it (the
    driver's preset).  Anything else is the FIRST refusal a callee reported
    -- each one marks a place where the Stage-A Python raises because the
    Fortran would misbehave (read out of bounds, STOP, or hit an unported
    feature).  Every code is GLOBALLY unique (``cuda/kernels/status.cuh``,
    mirrored by the ``engine.STATUS_*`` constants): ``status // 100`` is the
    routine and ``status % 100`` that routine's own local code.

        101   ALLMODES: more than 20 modes
        201   FRQCOM: frequency-complement index out of range
        301   GENOIS: NYEAR ' 2089' selects the unported GENOIS_OLD
        302   GENOIS: receive-antenna GAIN lookup left the table
              (``gain.cuh`` oob_flag 1 -- e.g. a frequency below 1 MHz)
        303   GENOIS: receive antenna is AREA-COVERAGE (oob_flag 2,
              OFFAZIM == -999, not ported)
        401   REGMOD STOP: MUFday probability <= 0
        402   REGMOD STOP: frequency <= 0
        403   REGMOD: antenna GAIN lookup left the table (oob_flag 1)
        404   REGMOD: area-coverage antenna (oob_flag 2)
        503   ESMOD: antenna GAIN lookup left the table
        504   ESMOD: area-coverage antenna
        601   SETTXR: MODMUF outside 1..4 (DELMUF)
        602   SETTXR: MODMUF outside 1..4 (YMUF)
        603   SETTXR: antenna GAIN lookup left the table
        604   SETTXR: area-coverage antenna
        705   INMUF: /MODES/ ITMOD outside 1..3
        706   INMUF: zero-distance stage would write /MODES/ row 7
        801   LNGPAT: YMUF index out of range
        901   RELBIL: mode label outside LAYTYP
        1001  SELTXR: ray index out of range
        1101  LSMOOTH: smoothing store index (LUF-arm IPFLAG in the blend)
        1201  LUFFY: /allMODE/ NMODE(2) outside LAYTYP on the long path
        1301  SETUP: hour outside 1..24, the circuit was not built

        99    CPU backend only (``STATUS_CPU_UNMAPPED``): the scalar reference
              raised where the kernel has no identifiable status (e.g.
              SERPRB's NoModesError)

    The CPU backend (``predict(..., backend='cpu')``) runs the scalar
    reference, which RAISES at each of these places; the exception is caught
    per circuit-hour and mapped back to the kernel's code from where it was
    raised (``engine._scalar_status``), so ``status`` and ``valid`` read the
    same on either backend.

    Treat any non-zero status as "this hour has no answer"; ``valid`` already
    does.
    """

    hours: np.ndarray
    freqs: np.ndarray
    muf: np.ndarray
    fot: np.ndarray
    hpf: np.ndarray
    luf: np.ndarray
    snr: np.ndarray
    dbw: np.ndarray
    reliab: np.ndarray
    field_dbu: np.ndarray
    nhp: np.ndarray
    mode: np.ndarray
    status: np.ndarray = field(default=None)
    valid: np.ndarray = field(default=None)

    def __post_init__(self):
        # Directly-constructed Results (not via _assemble) get the same
        # defaults _assemble would derive.
        if self.status is None:
            self.status = np.zeros(len(self.hours), np.int32)
        if self.valid is None:
            self.valid = _valid_mask(self.nhp, self.status)

    @property
    def lowest_reliable_freq(self):
        """Per hour: the lowest REQUESTED frequency meeting the reliability.

        That is, the smallest of ``freqs`` whose ``reliab`` is at or above
        ``System.required_reliability`` (MHz; NaN if none does, and invalid
        cells never qualify).  It is NOT VOACAP's LUF: VOACAP's LUF (the LUF
        arm, IPFLAG 300/400, /MUFS/ XLUF) searches its own frequency
        complement below the MUF and interpolates between the bracketing
        frequencies, so it is generally not one of the requested frequencies
        at all.  This is the same array as the ``luf`` field, which is kept
        under that name for compatibility.
        """
        return self.luf

    def array(self, name):
        """Named table as a plain numpy array (see the class docstring)."""
        return getattr(self, name)

    def _index(self, hour, freq):
        hi = np.flatnonzero(self.hours == hour)
        if hi.size == 0:
            raise KeyError(
                f"hour {hour!r} is not in this result; hours are "
                f"{[int(h) for h in self.hours]}")
        fi = np.flatnonzero(np.isclose(self.freqs, freq, rtol=0.0,
                                       atol=_FREQ_ATOL))
        if fi.size == 0:
            raise KeyError(
                f"frequency {freq!r} MHz is not in this result (exact match "
                f"within {_FREQ_ATOL} MHz required); freqs are "
                f"{[round(float(f), 6) for f in self.freqs]}")
        return int(hi[0]), int(fi[0])

    def at(self, hour, freq):
        """A dict of every quantity at one (hour, frequency) pair.

        ``freq`` must match one of ``freqs`` to within 1e-4 MHz -- there is
        no snapping to the nearest frequency -- and ``hour`` one of ``hours``;
        otherwise ``KeyError``.
        """
        i, j = self._index(hour, freq)
        return {
            "hour": int(self.hours[i]), "freq": float(self.freqs[j]),
            "muf": float(self.muf[i]), "fot": float(self.fot[i]),
            "hpf": float(self.hpf[i]), "luf": float(self.luf[i]),
            "snr": float(self.snr[i, j]), "dbw": float(self.dbw[i, j]),
            "reliab": float(self.reliab[i, j]),
            "field_dbu": float(self.field_dbu[i, j]),
            "nhp": int(self.nhp[i, j]), "mode": str(self.mode[i, j]),
            "valid": bool(self.valid[i, j]), "status": int(self.status[i]),
        }

    def to_records(self):
        """Flat list of :meth:`at` dicts over every (hour, frequency)."""
        return [self.at(int(h), float(f))
                for h in self.hours for f in self.freqs]


# LAYTYP index -> label, with the packed "" (index -1) as the LAST entry, so a
# packed /SON/ MODE column indexes it directly.
_MODE_BY_CODE = np.array(list(LAYTYP) + [""], dtype=object)


def _results_from_son(son, status, hours, freqs_f, muf, fot, hpf, lufp):
    """A packed ``(H,) SON_DTYPE`` table (one row per hour) -> Results."""
    sl = slice(1, len(freqs_f) + 1)                # /SON/ slots 1..nf
    return _finish_results(
        hours, freqs_f, np.array(muf, np.float32), np.array(fot, np.float32),
        np.array(hpf, np.float32), lufp,
        np.array(son["sndb"][:, sl], np.float32),
        np.array(son["dbw"][:, sl], np.float32),
        np.array(son["reliab"][:, sl], np.float32),
        np.array(son["dbu"][:, sl], np.float32),
        np.array(son["nhp"][:, sl], np.int32),
        _MODE_BY_CODE[son["mode"][:, sl]], status)


def _assemble(circuits, hours, freqs_f, muf, fot, hpf, lufp, status=None):
    """Distil the mutated circuits' /SON/ into a :class:`Results`.

    ``status`` is the per-circuit (= per-hour) LUFFY status
    :func:`predict_circuits` returns; None means "all succeeded".  Cells that
    were not computed (see :class:`Results`, VALID) come out NaN with mode
    ''; every computed cell is copied from /SON/ unchanged.
    """
    H, Fn = len(hours), len(freqs_f)
    sl = slice(1, Fn + 1)                          # /SON/ slots 1..nf

    def col(name, dt):
        return np.array([np.asarray(getattr(c.son, name))[sl]
                         for c in circuits], dt).reshape(H, Fn)

    mode = np.empty((H, Fn), object)
    for k, c in enumerate(circuits):
        mode[k, :] = list(c.son.mode[sl])
    return _finish_results(hours, freqs_f, muf, fot, hpf, lufp,
                           col("sndb", np.float32), col("dbw", np.float32),
                           col("reliab", np.float32), col("dbu", np.float32),
                           col("nhp", np.int32), mode, status)


def _finish_results(hours, freqs_f, muf, fot, hpf, lufp, snr, dbw, rel, dbu,
                    nhp, mode, status):
    """The (H, F) tables -> Results: status, the valid mask, NaNs, the LUF.

    Mutates the given arrays, which are this call's own.
    """
    H = len(hours)
    if status is None:
        status = np.zeros(H, np.int32)
    status = np.asarray(status, np.int32).reshape(-1)
    if status.shape != (H,):
        raise ValueError(f"status has shape {status.shape}, expected ({H},)"
                         " -- one per circuit-hour")
    valid = _valid_mask(nhp, status)
    for a in (snr, dbw, rel, dbu):
        a[~valid] = np.nan
    mode[~valid] = ""
    # lowest REQUESTED frequency meeting the reliability (NaN compares False,
    # so an invalid cell never qualifies) -- the same float32-reliab against
    # `lufp / 100.0` comparison the per-cell loop made.
    thresh = lufp / 100.0
    ok = rel >= thresh
    cand = np.where(ok, np.asarray(freqs_f, np.float32)[None, :],
                    np.float32(np.inf))
    luf = np.where(ok.any(axis=1), cand.min(axis=1, initial=np.inf),
                   np.float32(np.nan)).astype(np.float32)
    return Results(hours=np.array(hours, np.int32), freqs=freqs_f, muf=muf,
                   fot=fot, hpf=hpf, luf=luf, snr=snr, dbw=dbw, reliab=rel,
                   field_dbu=dbu, nhp=nhp, mode=mode, status=status,
                   valid=valid)
