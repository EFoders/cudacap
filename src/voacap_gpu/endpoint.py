"""``predict``: the public point-to-point endpoint, and its two paths.

Split out of :mod:`voacap_gpu.engine` (which still re-exports every name
here; the engine module docstring is the user-facing description).  ``predict``
validates, builds the per-circuit /cantenna/, then takes the device-setup
path (:func:`_predict_device`, ``backend='cuda'``) or the host-build path
(:func:`_predict_host`, the CPU backend, and the CUDA reference the device
path is held bit-identical to).
"""

import numpy as np

from voacap_gpu import system as _system
from voacap_gpu import validate as _validate
from voacap_gpu.antennas import _parse_antenna, antcalc_cant
from voacap_gpu.backends import (_predict_circuits_cpu, _resolve_backend,
                                 predict_circuits)
from voacap_gpu.circuit import _resolve_model, build_circuit, setup_config
from voacap_gpu.results import Results, _assemble, _results_from_son
from voacap_gpu.rigcache import _small_rig
from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.system import POLE_DEFAULT, System


def predict(tx, rx, *, freqs, month, ssn, hours=range(1, 25),
            tx_antenna=None, rx_antenna=None, tx_bearing=None, rx_bearing=None,
            system=None, method=30, model=None, coeff="CCIR", pole=POLE_DEFAULT,
            long_path=False, fast_math=False, tx_antenna_freq_range=(2, 30),
            rx_antenna_freq_range=(2, 30), tx_table_absolute=False,
            rx_table_absolute=False, backend="cuda") -> Results:
    """Predict an HF point-to-point circuit from raw inputs.

    With ``backend='cuda'`` every requested hour runs as ONE device batch:
    the setup chain (geometry, ionosphere, MUFs) is built on the device by
    ``setup_batch`` and LUFFY runs on it in place (see the module docstring,
    THE CUDA PATH).  With ``backend='cpu'`` the setup chain runs on the host
    (``build_circuit`` per hour) and LUFFY is the scalar reference.  Both give
    the same Results the host-build CUDA path does, bit for bit.
    For many circuits use :func:`predict_batch`.

    Parameters
    ----------
    tx, rx : (lat_deg, lon_deg)
        Endpoint coordinates, south and west negative; latitude in -90..90,
        longitude in -180..180.
    freqs : sequence of float
        Frequencies in MHz (1..11 of them), each in 1..30 MHz (the GAIN
        table's axis).
    month : int
        1..12.
    ssn : float
        Smoothed sunspot number, 0..300 (``validate.SSN_MAX``; the upper bound
        is a policy limit on extrapolation, not physics).
    hours : iterable of int, default 1..24
        1-based UTC hours, each an integer 1..24 (``12.0`` is accepted,
        ``12.7`` is refused rather than truncated).
    tx_antenna, rx_antenna : path or None
        A type-14 or type-13 ``.VOA`` file, or None for a 0 dBi isotropic
        pattern -- independently per end.  Type-13 patterns are
        azimuth-dependent; the slice at OFFAZIM (path azimuth minus
        ``*_bearing``) is installed, exactly as ANTCALC does.  Gain tables
        are RELATIVE to the file's header parameter 1 ("Max Gain dBi"),
        which is added to every tabulated gain as ANTCALC does -- see
        ``tx_table_absolute``.
    tx_bearing, rx_bearing : float or None
        Antenna main-beam heading in DEGREES TRUE AZIMUTH (clockwise from
        true north).  OFFAZIM is then the angle between that beam and the
        path azimuth at each end (ANTCALC's rule, via DAZEL0).  None (the
        default) points the beam along the path (OFFAZIM 0) -- the
        best-case gain on every path, so a type-13 file given with None
        emits a ``UserWarning``, as does a small non-zero bearing that
        looks like radians.  Inert for isotropic and type-14 antennas.
    system : System or None
        The Prop-SL parameter surface (SYSTEM/FPROB + gain offsets); see
        :class:`System` for units and accepted ranges.  Defaults to the golden
        decks' TEST values (RSN 73 dB-Hz, AMIND 0.1, 1 kW), not operating
        defaults.
    method : int, default 30
        30 smoothing, 20 system performance, 25 all modes.
    model : {'voacap', 'icepac', None}
        Prop-SL model toggle.  ``'voacap'`` selects CCIR coefficients + FPROB
        ``(1,1,1,0)`` (the golden-deck configuration).  ``'icepac'`` selects
        URSI + FPROB ``(1,1,1,0.70)`` (sporadic E on): Prop-SL's ICEPAC mode
        as voacapl runs it, i.e. the VOACAP engine with ICEPAC-like settings,
        NOT the ITS ICEPAC program (no ICED model; Q/Kp has no effect).  It
        is validated against voacapl on the 14 decks in
        ``golden/dumps/luffy_icepac/`` (see :func:`_resolve_model`).  A ``model`` that contradicts an explicit
        ``coeff`` or a non-default ``system.fprob`` raises ``ValueError``
        rather than overriding it.  ``None`` uses ``coeff`` / ``system`` as
        given.
    coeff : {'CCIR', 'URSI'}
        Coefficient set (exact upper-case spelling).  With ``model`` given,
        the default 'CCIR' counts as "not given" (the signature cannot tell
        it from an explicit 'CCIR'), so only an explicit 'URSI' with
        ``model='voacap'`` is detected as a conflict.  'URSI' with
        ``model=None`` keeps FPROB as given (Es off by default); its setup
        chain is validated by the ``*_ursi`` oracle decks.
    pole : (lat, lon)
        Geomagnetic north pole; see the module docstring.
    long_path : bool
        Take the long great-circle (through-antipode) geometry (NPSL=1).
    fast_math : bool
        Opt into the non-bit-exact native-f32 device libm (CUDA only;
        ``ValueError`` with ``backend='cpu'``).  It covers the device-built
        setup chain (geometry, ionosphere, MUFs) as well as LUFFY, exactly
        as in :func:`predict_batch`, so the two agree with it on.
    backend : {'cuda', 'cpu', 'auto'}, default 'cuda'
        Where LUFFY runs.  ``'cuda'`` is the production CUDA kernel (needs
        cupy and a GPU).  ``'cpu'`` runs the scalar Stage-A reference
        (``scalar.luffy.luffy``) on each circuit-hour; it never imports
        cupy, and it is bit-identical to the CUDA kernel on every captured
        oracle deck, but roughly 1000x slower (~70 ms per circuit-hour), so
        it suits a handful of predictions, not area grids.  ``'auto'`` uses
        CUDA when cupy imports and sees a device, otherwise falls back to
        CPU with a ``UserWarning``.  Where the kernel would return a
        non-zero status the scalar code raises; the CPU backend catches that
        per hour and maps it to the same status code (see
        :class:`Results`), so ``status`` and ``valid`` behave identically.
    tx_antenna_freq_range, rx_antenna_freq_range : (int, int), default (2, 30)
        The ANTENNA card's MINFREQ / MAXFREQ for a file antenna, whole MHz in
        1..30: only those table rows are installed (ANTCALC's rule).  VOACAP's
        GAIN silently substitutes 0 dBi outside the range, so a requested
        frequency outside it raises ``ValueError`` naming the end.  Ignored
        for an isotropic (None) end, which covers 0..30 MHz.
    tx_table_absolute, rx_table_absolute : bool, default False
        True when the file's gain table is already in absolute dBi: skips the
        header-parameter-1 add that ANTCALC applies (which would double-count
        such a table).  With False, an installed peak above 30 dBi emits a
        ``UserWarning`` suggesting the table may already be absolute.

    Returns
    -------
    Results

    Raises
    ------
    ValueError
        An input outside the ranges above, an unusable :class:`System` value,
        a ``model`` / ``coeff`` / ``fprob`` conflict (see
        :mod:`voacap_gpu.validate`), an unknown ``backend``, or
        ``fast_math=True`` with the CPU backend.
    NotImplementedError
        More than 11 (or no) frequencies, an unsupported METHOD, model or
        antenna type.
    """
    backend = _resolve_backend(backend, fast_math)
    system = system or System()
    _validate.check_coeff(coeff)
    coeff, system = _resolve_model(model, coeff, system)
    _validate.check_system(system)
    tx = _validate.check_point("tx", tx)         # longitude wrapped to ±180
    rx = _validate.check_point("rx", rx)
    month = _validate.check_month(month)
    _validate.check_ssn(ssn)
    hours = _validate.check_hours(hours)
    freqs_f = np.array(_validate.check_freqs(freqs), dtype=np.float32)

    # ANTCALC per circuit: either end may independently be isotropic, a
    # type-14 table, or a type-13 azimuth-dependent table (which consumes
    # the path azimuth at that end, so the cant is built from tx/rx here).
    # freqs= makes a frequency outside a file antenna's MINFREQ..MAXFREQ
    # raise instead of GAIN silently returning 0 dBi for it.
    cant = antcalc_cant(_parse_antenna(tx_antenna), _parse_antenna(rx_antenna),
                        tx, rx,
                        power_kw=system.power_kw, tx_gos=system.tx_gos,
                        rx_gos=system.rx_gos, tx_bearing=tx_bearing,
                        rx_bearing=rx_bearing, long_path=long_path,
                        tx_freq_range=tx_antenna_freq_range,
                        rx_freq_range=rx_antenna_freq_range,
                        tx_table_absolute=tx_table_absolute,
                        rx_table_absolute=rx_table_absolute,
                        freqs=freqs_f)

    if backend == "cuda" and _device_path_ok(long_path):
        return _predict_device(tx, rx, hours, freqs_f, month, ssn, coeff,
                               system, method, pole, long_path, cant,
                               fast_math)
    return _predict_host(tx, rx, hours, freqs_f, month, ssn, coeff, system,
                         method, pole, long_path, cant, fast_math, backend,
                         tx_bearing, rx_bearing)


def _device_path_ok(long_path):
    """Whether ``predict(backend='cuda')`` takes the device-setup path.

    It does unless ``long_path`` is not 0/1 as an integer (``long_path=2``):
    that takes the host path, so its pre-existing behaviour is untouched (the
    device refuses NPSL outside 0/1).  Nothing else routes a CUDA request to
    the host path; to exercise that path, call :func:`_predict_host` (or, in a
    test, replace this predicate).
    """
    return int(long_path) in (0, 1)


def _predict_host(tx, rx, hours, freqs_f, month, ssn, coeff, system, method,
                  pole, long_path, cant, fast_math, backend, tx_bearing,
                  rx_bearing):
    """The host-build path: ``build_circuit`` per hour, then LUFFY.

    The CPU backend's only path, and the CUDA reference the device path is
    held bit-identical to (``tests/test_predict_fast.py``).  One coefficient
    map for the whole request; one INDEPENDENT circuit per hour (fresh
    /SON/), all sharing the iono and antenna tables.
    """
    st = redmap(_system._COEFFS_DIR, coeff, int(month), np.float32(ssn))
    circuits, ipflags = [], []
    muf = np.empty(len(hours), np.float32)
    fot = np.empty(len(hours), np.float32)
    hpf = np.empty(len(hours), np.float32)
    for k, h in enumerate(hours):
        c, ip = build_circuit(tx, rx, month, ssn, h, freqs_f, method=method,
                              system=system, cant=cant, coeff=coeff,
                              pole=pole, npsl=int(long_path),
                              tx_bearing=tx_bearing, rx_bearing=rx_bearing,
                              iono=st)
        it = c.time.it
        muf[k] = c.muf.allmuf[it]
        fot[k] = c.muf.fot[it]
        hpf[k] = c.muf.hpf[it]
        circuits.append(c)
        ipflags.append(ip)

    if backend == "cpu":
        _, status = _predict_circuits_cpu(circuits, ipflags)
    else:
        # called with exactly the pre-backend signature, so a wrapper of
        # predict_circuits (fault injection in tests) keeps working
        _, status = predict_circuits(circuits, ipflags, fast_math=fast_math)
    return _assemble(circuits, hours, freqs_f, muf, fot, hpf,
                     system.required_reliability, status=status)


# predict's rig width: every request of up to 24 hours shares one rig (the
# padding rows repeat a real row and cost a few idle warps, not latency).
_PREDICT_ROWS = 24


def _predict_device(tx, rx, hours, freqs_f, month, ssn, coeff, system, method,
                    pole, long_path, cant, fast_math):
    """``predict`` on the device-setup path: one batch, no host build.

    Every requested hour of the circuit becomes one SETUP_PARAMS row (the
    same tx/rx, NPSL from ``long_path``; the config's ITIM=1 makes the hour
    UTC), all sharing ONE SETUP_CONFIG row (:func:`setup_config`, which
    carries the pole) and the per-circuit /cantenna/ ``antcalc_cant`` already
    built.  ``setup_batch`` builds each CircuitState on the device --
    bit-identical to ``build_circuit`` (tests/cuda/test_setup_batch.py) --
    and LUFFY runs on it in place; only /SON/, the status and the MUF triple
    come back.
    """
    from voacap_gpu.cuda.driver import SETUP_PARAMS_DTYPE, shared_antenna_tables

    # METHOD / System / frequency checks, at the point build_circuit made
    # them (after the antenna checks predict has already run)
    cfg = setup_config(system, method=method, freqs=freqs_f, pole=pole)
    H = len(hours)
    params = np.zeros(H, SETUP_PARAMS_DTYPE)
    params["tlatd"], params["tlongd"] = tx[0], tx[1]
    params["rlatd"], params["rlongd"] = rx[0], rx[1]
    params["hour"] = hours
    params["npsl"] = int(long_path)
    # iono / cfg slot 0: one ionosphere and one SYSTEM card per request
    rig, lock = _small_rig(max(H, _PREDICT_ROWS), coeff, month, ssn,
                           fast_math)
    ant = shared_antenna_tables([cant])
    with lock:
        res, muf3 = rig.run(params, cfg.reshape(1), np.zeros(H, np.int32),
                            ant=ant)
    return _results_from_son(res.son, res.status, hours, freqs_f,
                             muf3[:, 0], muf3[:, 1], muf3[:, 2],
                             system.required_reliability)


def predict_batch(*args, **kwargs):
    """Many circuits in one device run; see :func:`voacap_gpu.batch.predict_batch`."""
    from voacap_gpu.batch import predict_batch as _pb
    return _pb(*args, **kwargs)
