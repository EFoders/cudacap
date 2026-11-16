"""The public VOACAP-GPU endpoint: raw circuit parameters in, results out.

Everything below this module is validated line-against-line with the Fortran;
nothing above it existed.  ``predict`` composes the Stage-A setup routines --
REDMAP, GEOM, SANG, and the per-hour GEOTIM / VIRTIM / TIMVAR / F2VAR / ESIND /
IONSET / CURMUF chain, in exactly the order ``voacapw/hfmufs.for`` drives them
-- into a fully-built Stage-A ``Circuit`` for each requested hour, then hands
those circuits to the production CUDA LUFFY kernel (``cuda/driver.py``) and
distils the /SON/ table into a structured :class:`Results`.

WHAT IS SUPPORTED
-----------------
* METHOD 30 (short/long-path smoothing -- the default), 20 (system
  performance, short path forced past the long-path threshold) and 25
  (all-modes, short path forced).  These are the three regimes the LUFFY
  replay validates.  Any other METHOD raises ``NotImplementedError``.
* Up to 11 frequencies per call (the FREL card's capacity).
* Antennas, independently per end: an isotropic 0 dBi pattern (the default),
  a *type 14* ``.VOA`` table (30 frequencies x 91 elevations), or a *type 13*
  ``.VOA`` table (360 azimuths x 91 elevations at one design frequency) --
  each read through ``scalar/readant.py`` and installed exactly as READANT +
  ANTCALC would leave it.  Type 13 is azimuth-dependent: ANTCALC resolves
  OFFAZIM (DAZEL0 path azimuth minus the beam heading) at setup and installs
  the interpolated azimuth slice, so the /cantenna/ is per-circuit (see
  :func:`antcalc_cant`).  Types 1-12 (method-of-moments synthesis) raise
  ``NotImplementedError`` rather than returning wrong gains.
* Antenna bearings: ``tx_bearing`` / ``rx_bearing`` (degrees true) set
  OFFAZIM = DAZEL0 path azimuth at that end minus the main-beam heading,
  ANTCALC's rule, computed once in :func:`antcalc_cant`; default
  beam-along-path (a type-13 antenna with no bearing warns).
* Prop-SL ``model`` toggle, BOTH validated against voacapl (the same binary
  Prop-SL runs): ``'voacap'`` (CCIR + FPROB (1,1,1,0)) and ``'icepac'`` (URSI +
  FPROB (1,1,1,0.70), sporadic-E on).  'icepac' is Prop-SL's ICEPAC mode on
  the VOACAP engine, NOT the ITS ICEPAC program (no ICED model; Q/Kp is
  ignored by voacapl too).  See ``_resolve_model``.
  A ``model`` that contradicts an explicit ``coeff`` / ``fprob`` raises.
* The Prop-SL parameter surface (:class:`System`): required S/N (dB-Hz),
  reliability percentile, man-made noise (a -dBW/Hz level at 3 MHz or a CCIR
  category code), transmit power (kW), the multipath delay / power
  tolerances (DMP / PMP; MPATH only), and the ``tx_gos`` / ``rx_gos`` gain
  offsets (consumed); ``bandwidth`` / ``modulation`` are advisory (the
  S/N-derivation front-end is not ported).
* UTC hours 1..24.  ``hours`` given in local mean time at the transmitter is
  not exposed here (the Fortran's ITIM<0 path); pass UTC.

INPUT VALIDATION
----------------
``predict``, ``build_circuit`` and ``setup_config`` check their inputs with
:mod:`voacap_gpu.validate` before any of the chain runs: frequencies 1..30
MHz, month 1..12, integral hours 1..24, SSN 0..300, lat/lon in range, and the
:class:`System` values (see its docstring).  A bad value raises
``ValueError``; the Fortran-faithful ``int()`` truncation of a fractional
noise level or reliability is kept but warned about.  The checks never alter
a valid input, so the validated numbers below are untouched.

THE GEOMAGNETIC POLE
--------------------
GEOM's geomagnetic latitude needs the north geomagnetic pole.  The vendored
``database/north_pole.txt`` carries 79.5N, but ``set_magnetic_pole.for``'s
*runtime fallback* -- used when that file is not on the run path, which is how
the oracle LUFFY decks were captured -- is 78.5N, -69.0.  The default here is
that fallback, because it is what the validation oracle used and what
reproduces the captured entry state bit-for-bit; pass ``pole=(79.5, -69.0)`` to
match a modern voacapl install that finds the file.

VALIDATION
----------
``tests/test_engine.py`` builds circuits from the raw inputs (the CIRCUIT
card's coordinates and S/L flag) of every captured LUFFY deck -- the 21
replay decks plus the ICEPAC one -- and shows the built geometry + per-area
ionosphere matches the captured LUFFY entry state at the documented libm
floor, then runs them through LUFFY on both backends in ``predict``'s
configuration and shows the /SON/ table matches the captured output: hop
counts exact, every numeric field within an absolute bound (0.01 dB on the
dB fields; the table is at the top of that file).

BACKENDS
--------
``predict(..., backend=)`` picks where LUFFY runs: ``'cuda'`` (default, the
production kernel), ``'cpu'`` (the scalar Stage-A reference, no cupy import,
bit-identical to the kernel on every oracle deck and ~1000x slower), or
``'auto'`` (CUDA if a device is usable, else CPU with a ``UserWarning``).
``tests/test_cpu_backend.py`` holds the two to bit-identity.

THE CUDA PATH
-------------
``predict(backend='cuda')`` does not run ``build_circuit`` at all.  The hours
of the request become one batch of 32-byte ``SETUP_PARAMS`` rows (plus one
:func:`setup_config` row and the per-circuit /cantenna/ from
:func:`antcalc_cant`), the device builds each circuit-hour's state
(``cuda/kernels/setup.cuh``, bit-identical to ``build_circuit``) and the
warp-per-circuit-hour LUFFY kernel runs on it in place; /SON/ comes back as
one packed table and is sliced into :class:`Results`.  The device rig and the
ionosphere tables are cached between calls, keyed by size and by
(coefficients, month, SSN) -- see ``_RigCache`` for the thread-safety rules
and :func:`clear_device_cache` to release them.  ~8 ms warm for one circuit
x 24 hours x 11 frequencies, against ~140 ms for the host-build path.
``tests/test_predict_fast.py`` holds it bit-identical to the host-build path
(``_predict_host``: ``build_circuit`` per hour, then ``predict_circuits``),
which the CPU backend always takes.  ``predict(backend='cuda')`` never calls
``build_circuit`` or ``predict_circuits``, so replacing either does not
affect it.

For many circuits, :func:`predict_batch` (``voacap_gpu.batch``) runs them all
in one chunked, pipelined device run.  :func:`coeffs_dir` is the public name
of the packaged coefficient directory.

LAYOUT
------
This module is a compatibility facade: every name it has always had still
resolves here (``tests/test_engine_facade.py`` pins the list), but the code
lives in

* :mod:`voacap_gpu.system`    -- ``System``, ``POLE_DEFAULT``,
  ``coeffs_dir`` / ``_COEFFS_DIR``, the DMP / PMP defaults;
* :mod:`voacap_gpu.status`    -- the ``STATUS_*`` codes, ``_scalar_status``;
* :mod:`voacap_gpu.antennas`  -- ``offazim``, ``isotropic_cant``,
  ``cant_from_files``, ``antcalc_cant``;
* :mod:`voacap_gpu.circuit`   -- ``build_circuit``, ``setup_config``, the
  METHOD / model / IPFLAG rules;
* :mod:`voacap_gpu.backends`  -- ``predict_circuits``, the backend choice;
* :mod:`voacap_gpu.results`   -- ``Results`` and /SON/ -> Results;
* :mod:`voacap_gpu.rigcache`  -- the cached device rigs,
  ``clear_device_cache``;
* :mod:`voacap_gpu.endpoint`  -- ``predict``, its host and device paths.

Names here are the SAME objects as there (the rig caches exist once), but
they are bindings, not aliases: to replace a function for the code that calls
it (``monkeypatch.setattr``), replace it in the module that looks it up, e.g.
``voacap_gpu.backends._cuda_usable`` or ``voacap_gpu.endpoint.build_circuit``.
"""

# Third-party / scalar names engine.py has always had in its namespace (it
# imported them for its own code); kept so ``engine.<name>`` keeps resolving.
from collections import OrderedDict  # noqa: F401
from dataclasses import dataclass, field, replace  # noqa: F401

from voacap_gpu.scalar.antennastate import CantennaState  # noqa: F401
from voacap_gpu.scalar.circuit import Circuit  # noqa: F401
from voacap_gpu.scalar.curmuf import curmuf  # noqa: F401
from voacap_gpu.scalar.dazel import dazel0  # noqa: F401
from voacap_gpu.scalar.esind import esind  # noqa: F401
from voacap_gpu.scalar.f2var import f2var  # noqa: F401
from voacap_gpu.scalar.f32math import F32  # noqa: F401
from voacap_gpu.scalar.geom import geom  # noqa: F401
from voacap_gpu.scalar.geotim import geotim  # noqa: F401
from voacap_gpu.scalar.ionset import ionset  # noqa: F401
from voacap_gpu.scalar.noisy import noisy  # noqa: F401
from voacap_gpu.scalar.readant import (UnsupportedAntennaType,  # noqa: F401
                                       read_antenna)
from voacap_gpu.scalar.redmap import redmap  # noqa: F401
from voacap_gpu.scalar.sang import sang  # noqa: F401
from voacap_gpu.scalar.sonstate import LAYTYP  # noqa: F401
from voacap_gpu.scalar.timvar import timvar  # noqa: F401
from voacap_gpu.scalar.virtim import virtim  # noqa: F401

from voacap_gpu.system import (  # noqa: F401
    _COEFFS_DIR, _PKG_COEFFS, _VENDORED_COEFFS, DMP_DEFAULT_MS,
    PMP_DEFAULT_DB, POLE_DEFAULT, System, coeffs_dir)
from voacap_gpu.status import (  # noqa: F401
    _SCALAR_REFUSALS, _STATUS_GAIN, _STATUS_INDEX_REFUSAL,
    STATUS_ALLMODES_OVERFLOW, STATUS_CPU_UNMAPPED,
    STATUS_ESMOD_AREA_COVERAGE, STATUS_ESMOD_GAIN_OOB, STATUS_FRQCOM_INDEX,
    STATUS_GENOIS_AREA_COVERAGE, STATUS_GENOIS_GAIN_OOB, STATUS_GENOIS_OLD,
    STATUS_INMUF_ITMOD, STATUS_INMUF_MODES_ROW, STATUS_LNGPAT_INDEX,
    STATUS_LSMOOTH_INDEX, STATUS_LUFFY_BAD_NMODE2, STATUS_OK,
    STATUS_REGMOD_AREA_COVERAGE, STATUS_REGMOD_FREQ, STATUS_REGMOD_GAIN_OOB,
    STATUS_REGMOD_MUFDAY, STATUS_RELBIL_INDEX, STATUS_SELTXR_INDEX,
    STATUS_SETTXR_AREA_COVERAGE, STATUS_SETTXR_DELMUF,
    STATUS_SETTXR_GAIN_OOB, STATUS_SETTXR_YMUF, STATUS_SETUP_HOUR,
    STATUS_UNWRITTEN, _scalar_status)
from voacap_gpu.antennas import (  # noqa: F401
    _antcalc_offazim, _check_bearing, _finish_cant, _parse_antenna,
    antcalc_cant, cant_from_files, isotropic_cant, offazim)
from voacap_gpu.circuit import (  # noqa: F401
    _GCDLNG, _install_geometry, _ipflag, _resolve_method, _resolve_model,
    _setup_hour, build_circuit, setup_config)
from voacap_gpu.backends import (  # noqa: F401
    BACKENDS, _cuda_usable, _predict_circuits_cpu, _resolve_backend,
    predict_circuits)
from voacap_gpu.results import (  # noqa: F401
    _FREQ_ATOL, _MODE_BY_CODE, _MODE_LABELS, Results, _assemble,
    _finish_results, _results_from_son, _valid_mask)
from voacap_gpu.rigcache import (  # noqa: F401
    _BATCH_RIGS, _IONO_TABLES, _SMALL_RIGS, _RigCache, _device_id,
    _device_iono, _small_rig, clear_device_cache)
from voacap_gpu.endpoint import (  # noqa: F401
    _PREDICT_ROWS, _device_path_ok, _predict_device, _predict_host, predict,
    predict_batch)

# Before the split, predict() compared engine.build_circuit /
# engine.predict_circuits against these to notice a monkeypatched hook and
# fall back to the host path.  That detection is gone (predict no longer
# consults them); the names stay, bound to the real functions, so code that
# read them keeps working.
_BUILD_CIRCUIT = build_circuit
_PREDICT_CIRCUITS = predict_circuits
