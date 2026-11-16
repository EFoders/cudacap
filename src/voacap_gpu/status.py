"""LUFFY status codes, and the CPU backend's exception -> code mapping.

Split out of :mod:`voacap_gpu.engine` (which still re-exports every name
here).  The ``STATUS_*`` values mirror ``cuda/kernels/status.cuh``;
:func:`_scalar_status` maps a refusal the scalar reference raised back to the
kernel's code (see :class:`voacap_gpu.results.Results`, STATUS CODES).
"""


# THE STATUS TABLE (see Results, STATUS CODES).  Every refusal code the CUDA
# kernel returns is GLOBALLY unique: ``code // 100`` is the routine and
# ``code % 100`` that routine's own local code.  These mirror
# ``cuda/kernels/status.cuh`` value for value, and
# ``tests/cuda/test_status_codes.py`` holds them to it.
STATUS_OK = 0
STATUS_UNWRITTEN = -1                   # the driver's preset
STATUS_ALLMODES_OVERFLOW = 101
STATUS_FRQCOM_INDEX = 201
STATUS_GENOIS_OLD = 301
STATUS_GENOIS_GAIN_OOB = 302
STATUS_GENOIS_AREA_COVERAGE = 303
STATUS_REGMOD_MUFDAY = 401
STATUS_REGMOD_FREQ = 402
STATUS_REGMOD_GAIN_OOB = 403
STATUS_REGMOD_AREA_COVERAGE = 404
STATUS_ESMOD_GAIN_OOB = 503
STATUS_ESMOD_AREA_COVERAGE = 504
STATUS_SETTXR_DELMUF = 601
STATUS_SETTXR_YMUF = 602
STATUS_SETTXR_GAIN_OOB = 603
STATUS_SETTXR_AREA_COVERAGE = 604
STATUS_INMUF_ITMOD = 705
STATUS_INMUF_MODES_ROW = 706
STATUS_LNGPAT_INDEX = 801
STATUS_RELBIL_INDEX = 901
STATUS_SELTXR_INDEX = 1001
STATUS_LSMOOTH_INDEX = 1101
STATUS_LUFFY_BAD_NMODE2 = 1201
STATUS_SETUP_HOUR = 1301

# The CPU backend's own status code: the scalar reference raised somewhere the
# CUDA kernel has no identifiable counterpart status for.  Below 100, so it
# cannot collide with any kernel code, and not the driver's -1 preset.
STATUS_CPU_UNMAPPED = 99

# routine base per scalar module, for the codes that are base + 1
_STATUS_INDEX_REFUSAL = {"allmodes": STATUS_ALLMODES_OVERFLOW,
                         "frqcom": STATUS_FRQCOM_INDEX,
                         "lngpat": STATUS_LNGPAT_INDEX,
                         "relbil": STATUS_RELBIL_INDEX,
                         "seltxr": STATUS_SELTXR_INDEX}
# GAIN's caller -> (table miss, area-coverage antenna)
_STATUS_GAIN = {"genois": (STATUS_GENOIS_GAIN_OOB,
                           STATUS_GENOIS_AREA_COVERAGE),
                "regmod": (STATUS_REGMOD_GAIN_OOB,
                           STATUS_REGMOD_AREA_COVERAGE),
                "esmod": (STATUS_ESMOD_GAIN_OOB, STATUS_ESMOD_AREA_COVERAGE),
                "settxr": (STATUS_SETTXR_GAIN_OOB,
                           STATUS_SETTXR_AREA_COVERAGE)}

# Exceptions the scalar LUFFY raises where the kernel returns a non-zero
# status: the explicit refusals (AllModesOverflow, ModmufError, RegmodError,
# OldNoiseModel, ... are IndexError / ValueError / NotImplementedError
# subclasses), numpy / list index errors at the Fortran's out-of-bounds reads,
# and FDIST/FINDF's RuntimeError.  Anything else -- TypeError, AttributeError,
# KeyError -- is a bug in the caller or the port and propagates.
_SCALAR_REFUSALS = (IndexError, ValueError, NotImplementedError, RuntimeError,
                    ArithmeticError)


def _scalar_status(exc) -> int:
    """The CUDA status code for a refusal the scalar LUFFY raised.

    The kernel's codes are GLOBAL (``cuda/kernels/status.cuh``, the table in
    :class:`Results`): routine base + that routine's local code.  The code is
    identified from WHERE the scalar reference raised -- the innermost
    ``voacap_gpu.scalar`` frame of the traceback, plus GAIN's caller and,
    where one callee has two codes, the message or source line that tells
    them apart.  Each arm mirrors a status the kernel returns:

        allmodes IndexError                      STATUS_ALLMODES_OVERFLOW 101
        frqcom   IndexError                      STATUS_FRQCOM_INDEX      201
        genois   OldNoiseModel (GENOIS_OLD)      STATUS_GENOIS_OLD        301
        gain     via genois: IndexError 302, NotImplementedError (area) 303
        gain     via regmod: 403 / 404, via esmod: 503 / 504,
                 via settxr: 603 / 604
        regmod   RegmodError: MUFday prob <= 0 -> 401, frequency <= 0 -> 402
        settxr   ModmufError: DELMUF -> 601, YMUF -> 602
        inmuf    IndexError on INM(ITMOD) -> 705, on a /MODES/ row -> 706
        lngpat   IndexError                      STATUS_LNGPAT_INDEX      801
        relbil   IndexError                      STATUS_RELBIL_INDEX      901
        seltxr   IndexError                      STATUS_SELTXR_INDEX     1001
        lsmooth  IndexError on the smoothing store -> 1101
        luffy    IndexError, /allMODE/ NMODE(2) outside LAYTYP -> 1201

    Anything else (e.g. SERPRB's NoModesError or FDIST/FINDF's convergence
    RuntimeError, which the kernel has no status for) is
    :data:`STATUS_CPU_UNMAPPED`.
    """
    import linecache

    frames = []                                  # (module, line text)
    tb = exc.__traceback__
    while tb is not None:
        name = tb.tb_frame.f_globals.get("__name__", "")
        if name.startswith("voacap_gpu.scalar."):
            line = linecache.getline(tb.tb_frame.f_code.co_filename,
                                     tb.tb_lineno)
            frames.append((name.rsplit(".", 1)[-1], line))
        tb = tb.tb_next
    if not frames:
        return STATUS_CPU_UNMAPPED
    where, line = frames[-1]
    msg = str(exc)
    is_index = isinstance(exc, IndexError)
    if where == "gain":
        caller = frames[-2][0] if len(frames) > 1 else ""
        area = isinstance(exc, NotImplementedError)      # oob_flag 2
        if not (area or is_index) or caller not in _STATUS_GAIN:
            return STATUS_CPU_UNMAPPED
        oob, area_code = _STATUS_GAIN[caller]
        return area_code if area else oob
    if where in _STATUS_INDEX_REFUSAL:
        return _STATUS_INDEX_REFUSAL[where] if is_index else \
            STATUS_CPU_UNMAPPED
    if where == "genois":
        return STATUS_GENOIS_OLD if isinstance(exc, NotImplementedError) \
            else STATUS_CPU_UNMAPPED
    if where == "regmod" and isinstance(exc, ValueError):
        return STATUS_REGMOD_FREQ if "frequency must be positive" in msg \
            else STATUS_REGMOD_MUFDAY
    if where == "settxr" and is_index:
        if "YMUF" in msg:
            return STATUS_SETTXR_YMUF
        if "DELMUF" in msg:
            return STATUS_SETTXR_DELMUF
        return STATUS_CPU_UNMAPPED
    if where == "inmuf" and is_index:
        return STATUS_INMUF_ITMOD if "inm[" in line else \
            STATUS_INMUF_MODES_ROW
    if where == "lsmooth" and is_index:
        return STATUS_LSMOOTH_INDEX
    if where == "luffy" and is_index and "NMODE(2)" in msg:
        return STATUS_LUFFY_BAD_NMODE2
    return STATUS_CPU_UNMAPPED
