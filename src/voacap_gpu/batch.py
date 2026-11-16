"""Many circuits, one device run: the public batch API.

:func:`predict_batch` is :func:`voacap_gpu.engine.predict` for N circuits at
once -- one transmitter and many receivers (an area map), or explicit TX/RX
pairs -- over the same hours, frequencies, month, SSN, System, model,
method and antennas.  It returns compact numpy arrays rather than N
``Results`` objects::

    from voacap_gpu.batch import predict_batch      # also engine.predict_batch

    r = predict_batch((45.42, -75.70), rx_points,  # rx_points: (N, 2) lat/lon
                      freqs=[3.5, 7.0, 14.0, 21.0], month=7, ssn=120,
                      hours=range(1, 25))
    r.snr            # (N, H, F) float32, NaN where not computed
    r.reliab[i]      # circuit i: (H, F)
    r.muf            # (N, H) float32
    r.circuit(i)     # circuit i as an engine.Results, equal to predict()

RESULT ARRAYS (:class:`BatchResults`)
-------------------------------------
``(N, H, F)``: ``snr`` (dB-Hz), ``dbw`` (dBW), ``field_dbu`` (dB(uV/m)),
``reliab`` (0..1) -- all float32, NaN where not computed -- ``nhp`` (int32,
raw where not computed), ``mode`` (int8 LAYTYP code: 0 ' E', 1 'F1', 2 'F2',
3 'ES', 4 ' N', -1 none; ``mode_labels()`` gives the strings) and ``valid``
(bool).  ``(N, H)``: ``muf``, ``fot``, ``hpf``, ``luf`` (float32 MHz; ``luf``
is :attr:`engine.Results.lowest_reliable_freq`) and ``status`` (int32).  The
validated inputs are echoed: ``tx`` / ``rx`` ``(N, 2)`` (longitudes wrapped
to +-180 as ``predict`` wraps them), ``long_path`` ``(N,)``, ``tx_bearing`` /
``rx_bearing`` ``(N,)`` (NaN = None), ``hours``, ``freqs``, ``month``,
``ssn``, ``method``, ``coeff``, ``model``, ``system`` (as resolved by
``model``) and ``pole``.  Every value means exactly what the same-named
:class:`engine.Results` field means -- see its docstring for VALID, the very
negative over-the-MUF values and the STATUS CODES -- and for any ``i``,
``r.circuit(i)`` equals ``predict(tx[i], rx[i], ...)`` bit for bit
(``tests/test_batch.py``).

INPUTS
------
``tx`` / ``rx``: one ``(lat, lon)`` point or an ``(N, 2)`` array each; a
single point is broadcast against the other side.  ``tx_bearing`` /
``rx_bearing``: None, one heading, or ``(N,)`` (None / NaN entries = beam
along the path).  ``long_path``: a bool or ``(N,)`` bools.  Everything else
is ``predict``'s argument of the same name, validated by the same
``voacap_gpu.validate`` checks (points vectorised; a bad one is reported by
index).  Up to 11 frequencies (the FREL card).  Antennas: isotropic, type 14
or type 13 ``.VOA`` files; type 13 is azimuth-dependent, so each circuit
gets its own /cantenna/ slice (built vectorised, and uploaded in blocks of
``CANT_BLOCK`` so a world grid never needs gigabytes of antenna tables).

HOW IT RUNS
-----------
``backend='cuda'`` (default): the device builds every circuit-hour from 32
bytes of parameters and runs LUFFY on it in place (``PipelinedSetupLuffy``,
chunked across ``nstreams`` CUDA streams; results are scattered back while
the next chunk computes).  Batches of at most ``SMALL_ROWS`` circuit-hours
use the lower-latency warp-per-circuit-hour kernel instead.  The rigs are
cached between calls (``engine.clear_device_cache()`` releases them).
``backend='cpu'`` loops ``predict(..., backend='cpu')`` -- correct, ~70 ms a
circuit-hour, for small jobs on machines without a GPU; ``'auto'`` picks.

ORDERING.  The LUFFY kernel runs one circuit-hour per thread, and threads of
a warp that take different branches serialise, so WHICH circuit-hours share a
warp matters: a randomly ordered batch runs at about half the speed of a
coherent one.  ``order='spatial'`` (default) therefore runs the rows sorted
by (path-length class, long-path flag, hour, Morton code of the transmitter,
Morton code of the receiver) and scatters the answers back, so results are
always in the CALLER's order.  The path class -- GEOM's 1 / 3 / 5
control-point bands (2,000 / 4,000 km), the METHOD-30 smoothing band
(7,000..10,000 km) and beyond -- is the key that matters most: measured on
an RTX 5080 it beat every purely spatial order, including a raster world
grid's natural row order, by ~1.8x (hour + Morton alone: ~1.0-1.6x;
distance sort: ~1.1x).  ``order='none'`` runs the rows as given (circuit-major,
``(N, H)`` row-major).
"""

from dataclasses import dataclass

import numpy as np

from voacap_gpu import engine as _engine
from voacap_gpu import validate as _validate
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.sonstate import LAYTYP

__all__ = ["predict_batch", "BatchResults", "MODE_LABELS"]

# mode code -> label; code -1 (no mode) is ''.
MODE_LABELS = tuple(LAYTYP)
_MODE_CODE = {lab: i for i, lab in enumerate(LAYTYP)}
_MODE_CODE[""] = -1
_MODE_OBJ = np.array(list(LAYTYP) + [""], dtype=object)   # code -1 -> last

# Batches of at most this many circuit-hours run on the warp kernel
# (``SetupLuffy(warp=True)``): lower latency for small jobs.
SMALL_ROWS = 1024
# Default pipelined chunk and stream count: ~55 KB of device state per row,
# so 4096 x 3 is ~0.7 GB resident while cached.
DEFAULT_CHUNK = 4096
DEFAULT_STREAMS = 3
# At most this many distinct /cantenna/ sets per device run (~34 KB each, so
# ~140 MB).  Type-13 batches with more distinct beam slices run in blocks.
CANT_BLOCK = 4096


@dataclass
class BatchResults:
    """Per-(circuit, hour, frequency) predictions for a batch; see the module
    docstring for the fields."""

    tx: np.ndarray
    rx: np.ndarray
    long_path: np.ndarray
    tx_bearing: np.ndarray
    rx_bearing: np.ndarray
    hours: np.ndarray
    freqs: np.ndarray
    month: int
    ssn: float
    method: int
    coeff: str
    model: object
    system: object
    pole: tuple
    snr: np.ndarray
    dbw: np.ndarray
    field_dbu: np.ndarray
    reliab: np.ndarray
    nhp: np.ndarray
    mode: np.ndarray
    valid: np.ndarray
    muf: np.ndarray
    fot: np.ndarray
    hpf: np.ndarray
    luf: np.ndarray
    status: np.ndarray

    def __len__(self):
        return int(self.snr.shape[0])

    def mode_labels(self):
        """``mode`` as the 2-character LAYTYP labels ('' for none), object."""
        return _MODE_OBJ[self.mode]

    def circuit(self, i) -> "_engine.Results":
        """Circuit ``i`` as an :class:`engine.Results` -- the very object
        ``predict`` returns for it (equal field for field, bit for bit)."""
        i = int(i)
        return _engine.Results(
            hours=self.hours.copy(), freqs=self.freqs.copy(),
            muf=self.muf[i].copy(), fot=self.fot[i].copy(),
            hpf=self.hpf[i].copy(), luf=self.luf[i].copy(),
            snr=self.snr[i].copy(), dbw=self.dbw[i].copy(),
            reliab=self.reliab[i].copy(),
            field_dbu=self.field_dbu[i].copy(), nhp=self.nhp[i].copy(),
            mode=_MODE_OBJ[self.mode[i]], status=self.status[i].copy(),
            valid=self.valid[i].copy())


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------

def _points(name, p):
    """``(lat, lon)`` or ``(N, 2)`` -> ``(array (n, 2) float64, single)``.

    Vectorised :func:`validate.check_point`: latitude finite in -90..90,
    longitude finite and WRAPPED into -180..180 (``validate.wrap_lon``, the
    same arithmetic).  A bad row is re-checked through ``check_point`` itself
    so the error names it exactly as ``predict`` would, with its index.
    """
    try:
        a = np.array(p, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a (lat_deg, lon_deg) pair or an "
                         f"(N, 2) array of them, got {p!r}") from None
    single = a.ndim == 1
    if single:
        a = a.reshape(1, -1)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] == 0:
        raise ValueError(f"{name} must be a (lat_deg, lon_deg) pair or an "
                         f"(N, 2) array of them, got shape {np.shape(p)}")
    bad = ~(np.isfinite(a).all(axis=1) & (np.abs(a[:, 0]) <= 90.0))
    if bad.any():
        k = int(np.flatnonzero(bad)[0])
        label = name if single else f"{name}[{k}]"
        _validate.check_point(label, tuple(a[k]))       # raises
        raise ValueError(f"{label} = {tuple(a[k])!r} is not a valid point")
    out = ~(np.abs(a[:, 1]) <= 180.0)
    if out.any():
        a[out, 1] = _validate.wrap_lon(a[out, 1])
    return a, single


def _per_circuit(name, v, n, kind):
    """None / scalar / length-n sequence -> (n,) array (NaN = None)."""
    if kind == "bearing":
        if v is None:
            return np.full(n, np.nan)
        if np.ndim(v) == 0:
            return np.full(n, float(v))
        a = np.array([np.nan if x is None else float(x) for x in v],
                     np.float64)
    else:                                            # long_path
        if np.ndim(v) == 0:
            iv = int(v)
            if iv not in (0, 1):
                raise ValueError(f"{name} must be a bool, got {v!r}")
            return np.full(n, bool(iv))
        a = np.asarray(v)
        if a.ndim == 1 and not np.isin(a, (0, 1)).all():
            raise ValueError(f"{name} entries must be bools (0/1)")
        a = a.astype(bool)
    if a.shape != (n,):
        raise ValueError(f"{name} must be a scalar or have shape ({n},), "
                         f"got {a.shape}")
    return a


def _morton(lat, lon):
    """A 32-bit Z-order code of (lat, lon): 16 bits each, interleaved."""
    def spread(v):
        v = v.astype(np.uint32) & np.uint32(0xFFFF)
        v = (v | (v << np.uint32(8))) & np.uint32(0x00FF00FF)
        v = (v | (v << np.uint32(4))) & np.uint32(0x0F0F0F0F)
        v = (v | (v << np.uint32(2))) & np.uint32(0x33333333)
        v = (v | (v << np.uint32(1))) & np.uint32(0x55555555)
        return v
    y = np.clip((lat + 90.0) / 180.0 * 65535.0, 0, 65535)
    x = np.clip((lon + 180.0) / 360.0 * 65535.0, 0, 65535)
    return (spread(y) << np.uint32(1)) | spread(x)


# GEOM's control-point classes (KM 1 / 3 / 5 at 2,000.01 / 4,000 km) and the
# METHOD-30 smoothing band (7,000..10,000 km, past which IPFLAG is 200):
# circuit-hours in one class run the same branches of the kernel.
_PATH_CLASS_KM = (2000.01, 4000.0, 7000.0, 10000.0)


def _path_class(txs, rxs, long_path):
    """Path-length class per circuit, from a float64 great circle.

    An ORDERING key only -- it never reaches the computation, so its
    precision is irrelevant to the answer.  The long path is the complement
    of the short one.
    """
    la1, lo1 = np.radians(txs[:, 0]), np.radians(txs[:, 1])
    la2, lo2 = np.radians(rxs[:, 0]), np.radians(rxs[:, 1])
    c = (np.sin(la1) * np.sin(la2)
         + np.cos(la1) * np.cos(la2) * np.cos(lo2 - lo1))
    d = np.arccos(np.clip(c, -1.0, 1.0)) * 6370.0
    d = np.where(long_path, 2.0 * np.pi * 6370.0 - d, d)
    return np.digitize(d, _PATH_CLASS_KM)


# ---------------------------------------------------------------------------
# antennas: ANTCALC for N circuits
# ---------------------------------------------------------------------------

def _offazims(txs, rxs, long_path, bearings, end):
    """ANTCALC's OFFAZIM at one end for every circuit (antcalc_cant's rule).

    DAZEL0 per circuit (it is float64 libm work, kept scalar so it is the
    very function ``antcalc_cant`` calls), the long-path 180-degree flip,
    then ``engine._antcalc_offazim``.
    """
    from voacap_gpu.scalar.dazel import dazel0

    out = np.empty(len(txs), np.float32)
    for k in range(len(txs)):
        t, r = txs[k], rxs[k]
        if end == "tx":
            ztaz, _ = dazel0(t[0], t[1], r[0], r[1])
        else:
            ztaz, _ = dazel0(r[0], r[1], t[0], t[1])
        if long_path[k]:                        # antcalc.for:110-115
            ztaz = F32(ztaz + F32(180.0))
            if ztaz >= 360.0:
                ztaz = F32(ztaz - F32(360.0))
        b = bearings[k]
        out[k] = _engine._antcalc_offazim(ztaz,
                                          None if np.isnan(b) else float(b))
    return out


def _t13_rows(ant, offs, absolute):
    """``CantennaState.load_p2p_t13``'s installed elevation cut, for many
    OFFAZIMs at once: ``(M, 91)`` float32, element for element the same
    float32 / float64 operations in the same order (held to it by
    ``tests/test_batch.py``)."""
    off = np.asarray(offs, np.float32).copy()
    hi = off >= F32(360.0)
    off[hi] = off[hi] - F32(360.0)
    iazim = off.astype(np.int64)                # REAL->INT truncates
    iazim2 = iazim + 1
    iazim2[iazim2 == 360] = 0
    g1 = ant.gains360[iazim]
    g2 = ant.gains360[iazim2]
    fract = (off - iazim.astype(np.float32)).astype(np.float32)[:, None]
    if absolute:
        g = g1 + (g2 - g1) * fract
    else:
        g = (g1 + (g2 - g1) * fract) + ant.parm[0]
    g = np.maximum(g, F32(-99.99))
    g = np.float64(g)
    return (np.round(g * 1000.0) / 1000.0).astype(np.float32)


class _Antennas:
    """The /cantenna/ sets of a batch, and which one each circuit uses.

    ``template`` is ``engine.antcalc_cant`` for circuit 0, so every check and
    warning ``predict`` makes on the antennas (frequency range, bearing
    guards, the absolute-table peak) happens here exactly once.  Without a
    type-13 end, that one set serves every circuit: an isotropic or type-14
    table does not depend on the path, and OFFAZIM is read by the kernel only
    as the area-coverage sentinel (``gain.cuh``), never as a value, so the
    template's OFFAZIM stands for all.  With a type-13 end, each distinct
    beam slice is its own set -- the template with that end's elevation cut
    (and OFFAZIM) replaced -- deduplicated on the OFFAZIM pair.
    """

    def __init__(self, tx_ant, rx_ant, txs, rxs, long_path, txb, rxb,
                 system, kw, freqs_f):
        self.n = len(txs)
        b0 = (None if np.isnan(txb[0]) else float(txb[0]),
              None if np.isnan(rxb[0]) else float(rxb[0]))
        self.template = _engine.antcalc_cant(
            tx_ant, rx_ant, tuple(txs[0]), tuple(rxs[0]),
            power_kw=system.power_kw, tx_gos=system.tx_gos,
            rx_gos=system.rx_gos, tx_bearing=b0[0], rx_bearing=b0[1],
            long_path=bool(long_path[0]), freqs=freqs_f, **kw)
        # the bearing guard rails, once per further distinct heading
        for end, ant, bs in (("tx", tx_ant, txb), ("rx", rx_ant, rxb)):
            for b in np.unique(bs[~np.isnan(bs)]):
                if (ant is not None and ant.jant == 13
                        and b not in [x for x in b0 if x is not None]):
                    _engine._check_bearing(end, ant, float(b))
        self.ends = []                      # (iat, ant, offs, absolute, gos)
        if tx_ant is not None and tx_ant.jant == 13:
            self.ends.append((1, tx_ant, _offazims(txs, rxs, long_path, txb,
                                                   "tx"),
                              kw["tx_table_absolute"], system.tx_gos,
                              kw["tx_freq_range"]))
        if rx_ant is not None and rx_ant.jant == 13:
            self.ends.append((2, rx_ant, _offazims(txs, rxs, long_path, rxb,
                                                   "rx"),
                              kw["rx_table_absolute"], system.rx_gos,
                              kw["rx_freq_range"]))
        if not self.ends:
            self.set_of = np.zeros(self.n, np.int64)
            self.n_sets = 1
            self._keys = None
        else:
            key = np.zeros((self.n, 2), np.uint32)
            for j, (iat, _a, offs, *_r) in enumerate(self.ends):
                key[:, iat - 1] = offs.view(np.uint32)
            uniq, inv = np.unique(key, axis=0, return_inverse=True)
            self.set_of = inv.reshape(-1).astype(np.int64)
            self.n_sets = len(uniq)
            self._keys = uniq

    def tables(self, sets):
        """``shared_antenna_tables`` layout for the given set ids, in order."""
        from voacap_gpu.cuda.driver import shared_antenna_tables
        t = self.template
        if not self.ends:
            return shared_antenna_tables([t])
        from voacap_gpu.scalar.antennastate import NELEV, freq_rows
        m = len(sets)
        na = int(t.numants) + 1
        array = np.repeat(np.ascontiguousarray(t.array[:, :, :na])[None], m,
                          axis=0)
        offazim = np.repeat(np.asarray(t.offazim[:na], np.float32)[None], m,
                            axis=0)
        for iat, ant, _offs, absolute, gos, rng in self.ends:
            offs = self._keys[sets, iat - 1].view(np.float32)
            rows = _t13_rows(ant, offs, absolute)
            if gos:                             # _finish_cant's offset
                rows = rows + F32(gos)
            ifs, ife = freq_rows(*rng)
            array[:, ifs:ife + 1, 1:NELEV + 1, iat] = rows[:, None, :]
            offazim[:, iat] = offs
        def rep(a, dt):
            return np.repeat(np.asarray(a, dt)[None], m, axis=0).ravel()
        return dict(array=array.ravel(),
                    aeff=rep(np.ascontiguousarray(t.aeff[:, :na]), np.float32),
                    iats=rep(t.iats[:na], np.int32),
                    xfqs=rep(t.xfqs[:na], np.float32),
                    xfqe=rep(t.xfqe[:na], np.float32),
                    offazim=offazim.ravel(),
                    pwrdba=rep(t.pwrdba[:na], np.float32),
                    numants=int(t.numants))


# ---------------------------------------------------------------------------
# the endpoint
# ---------------------------------------------------------------------------

def predict_batch(tx, rx, *, freqs, month, ssn, hours=range(1, 25),
                  tx_antenna=None, rx_antenna=None, tx_bearing=None,
                  rx_bearing=None, system=None, method=30, model=None,
                  coeff="CCIR", pole=_engine.POLE_DEFAULT, long_path=False,
                  fast_math=False, tx_antenna_freq_range=(2, 30),
                  rx_antenna_freq_range=(2, 30), tx_table_absolute=False,
                  rx_table_absolute=False, backend="cuda", order="spatial",
                  chunk=None, nstreams=None) -> BatchResults:
    """Predict N HF circuits at once; see the module docstring.

    Arguments are :func:`engine.predict`'s, with ``tx`` / ``rx`` /
    ``tx_bearing`` / ``rx_bearing`` / ``long_path`` per circuit, plus
    ``order`` ('spatial' or 'none', see ORDERING) and the pipelined rig's
    ``chunk`` (circuit-hours per launch, default ``DEFAULT_CHUNK``) and
    ``nstreams`` (default ``DEFAULT_STREAMS``).  Raises as ``predict`` does.
    """
    backend = _engine._resolve_backend(backend, fast_math)
    if order not in ("spatial", "none"):
        raise ValueError(f"order={order!r}: expected 'spatial' or 'none'")
    system = system or _engine.System()
    _validate.check_coeff(coeff)
    coeff, system = _engine._resolve_model(model, coeff, system)
    _validate.check_system(system)
    txs, tx1 = _points("tx", tx)
    rxs, rx1 = _points("rx", rx)
    if len(txs) != len(rxs):
        if len(txs) == 1 and tx1:
            txs = np.repeat(txs, len(rxs), axis=0)
        elif len(rxs) == 1 and rx1:
            rxs = np.repeat(rxs, len(txs), axis=0)
        else:
            raise ValueError(f"tx has {len(txs)} points and rx {len(rxs)}; "
                             "give one of them as a single point or both as "
                             "(N, 2) arrays of the same length")
    n = len(txs)
    month = _validate.check_month(month)
    _validate.check_ssn(ssn)
    hours = _validate.check_hours(hours)
    freqs_f = np.array(_validate.check_freqs(freqs), dtype=np.float32)
    txb = _per_circuit("tx_bearing", tx_bearing, n, "bearing")
    rxb = _per_circuit("rx_bearing", rx_bearing, n, "bearing")
    lp = _per_circuit("long_path", long_path, n, "long_path")
    hours_a = np.array(hours, np.int32)

    echo = dict(tx=txs, rx=rxs, long_path=lp, tx_bearing=txb, rx_bearing=rxb,
                hours=hours_a, freqs=freqs_f, month=month, ssn=ssn,
                method=method, coeff=coeff, model=model, system=system,
                pole=tuple(pole))
    kw = dict(tx_freq_range=tx_antenna_freq_range,
              rx_freq_range=rx_antenna_freq_range,
              tx_table_absolute=tx_table_absolute,
              rx_table_absolute=rx_table_absolute)
    tx_ant = _engine._parse_antenna(tx_antenna)
    rx_ant = _engine._parse_antenna(rx_antenna)

    if backend == "cpu":
        # the antenna checks / warnings, once, as the CUDA path makes them
        _Antennas(tx_ant, rx_ant, txs[:1], rxs[:1], lp[:1], txb[:1], rxb[:1],
                  system, kw, freqs_f)
        return _run_cpu(echo, tx_antenna, rx_antenna, kw, fast_math)

    ants = _Antennas(tx_ant, rx_ant, txs, rxs, lp, txb, rxb, system, kw,
                     freqs_f)
    cfg = _engine.setup_config(system, method=method, freqs=freqs_f,
                               pole=pole).reshape(1)
    return _run_cuda(echo, ants, cfg, order, fast_math,
                     chunk or DEFAULT_CHUNK, nstreams or DEFAULT_STREAMS)


def _alloc(n, h, nf):
    f = lambda: np.full((n * h, nf), np.nan, np.float32)
    return dict(snr=f(), dbw=f(), field_dbu=f(), reliab=f(),
                nhp=np.zeros((n * h, nf), np.int32),
                mode=np.full((n * h, nf), -1, np.int8),
                muf=np.empty(n * h, np.float32),
                fot=np.empty(n * h, np.float32),
                hpf=np.empty(n * h, np.float32),
                status=np.full(n * h, -1, np.int32))


def _finish(out, echo):
    """Mask, NaN, LUF and reshape -- ``engine._finish_results`` for N*H rows
    (the same operations, so ``circuit(i)`` equals ``predict``)."""
    n, h, nf = len(echo["tx"]), len(echo["hours"]), len(echo["freqs"])
    valid = (out["nhp"] >= 1) & (out["status"] == 0)[:, None]
    for k in ("snr", "dbw", "field_dbu", "reliab"):
        out[k][~valid] = np.nan
    out["mode"][~valid] = -1
    thresh = echo["system"].required_reliability / 100.0
    ok = out["reliab"] >= thresh
    cand = np.where(ok, echo["freqs"][None, :], np.float32(np.inf))
    luf = np.where(ok.any(axis=1), cand.min(axis=1, initial=np.inf),
                   np.float32(np.nan)).astype(np.float32)
    res = {k: v.reshape(n, h, nf) for k, v in out.items()
           if k in ("snr", "dbw", "field_dbu", "reliab", "nhp", "mode")}
    res["valid"] = valid.reshape(n, h, nf)
    for k in ("muf", "fot", "hpf", "status"):
        res[k] = out[k].reshape(n, h)
    res["luf"] = luf.reshape(n, h)
    return BatchResults(**echo, **res)


def _scatter(out, rows, son, status, muf3, nf):
    sl = slice(1, nf + 1)
    out["snr"][rows] = son["sndb"][:, sl]
    out["dbw"][rows] = son["dbw"][:, sl]
    out["field_dbu"][rows] = son["dbu"][:, sl]
    out["reliab"][rows] = son["reliab"][:, sl]
    out["nhp"][rows] = son["nhp"][:, sl]
    out["mode"][rows] = son["mode"][:, sl]
    out["status"][rows] = status
    out["muf"][rows] = muf3[:, 0]
    out["fot"][rows] = muf3[:, 1]
    out["hpf"][rows] = muf3[:, 2]


def _run_cuda(echo, ants, cfg, order, fast_math, chunk, nstreams):
    from voacap_gpu.cuda.driver import (SETUP_PARAMS_DTYPE,
                                         PipelinedSetupLuffy)

    txs, rxs, lp = echo["tx"], echo["rx"], echo["long_path"]
    hours = echo["hours"]
    n, h, nf = len(txs), len(hours), len(echo["freqs"])
    rows_total = n * h
    # caller row r = i * H + j  (circuit i, hour j)
    ci = np.repeat(np.arange(n), h)
    hj = np.tile(np.arange(h), n)

    # antenna blocks: one, unless type-13 slices exceed CANT_BLOCK
    if ants.n_sets <= CANT_BLOCK:
        circuit_block = np.zeros(n, np.int64)
    else:
        sp = np.lexsort((_morton(rxs[:, 0], rxs[:, 1]),
                         _morton(txs[:, 0], txs[:, 1])))
        circuit_block = np.empty(n, np.int64)
        blk, seen = 0, set()
        for i in sp:
            s = int(ants.set_of[i])
            if s not in seen and len(seen) >= CANT_BLOCK:
                blk, seen = blk + 1, set()
            seen.add(s)
            circuit_block[i] = blk
    blk_row = circuit_block[ci]
    if order == "spatial":
        perm = np.lexsort((_morton(rxs[:, 0], rxs[:, 1])[ci],
                           _morton(txs[:, 0], txs[:, 1])[ci],
                           hours[hj], lp[ci],
                           _path_class(txs, rxs, lp)[ci], blk_row))
    else:
        perm = np.lexsort((np.arange(rows_total), blk_row))

    params = np.zeros(rows_total, SETUP_PARAMS_DTYPE)
    params["tlatd"] = txs[ci, 0]
    params["tlongd"] = txs[ci, 1]
    params["rlatd"] = rxs[ci, 0]
    params["rlongd"] = rxs[ci, 1]
    params["hour"] = hours[hj]
    params["npsl"] = lp[ci].astype(np.int32)
    params = params[perm]
    blk_sorted = blk_row[perm]
    set_sorted = ants.set_of[ci[perm]]

    out = _alloc(n, h, nf)
    month, ssn, coeff = echo["month"], echo["ssn"], echo["coeff"]
    bounds = np.flatnonzero(np.diff(blk_sorted)) + 1
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, rows_total]):
        a, b = int(a), int(b)
        sets, local = np.unique(set_sorted[a:b], return_inverse=True)
        ant = ants.tables(sets)
        ant_idx = local.reshape(-1).astype(np.int32)
        rows = perm[a:b]
        p = params[a:b]
        if b - a <= SMALL_ROWS:
            width = max(32, 1 << int(np.ceil(np.log2(b - a))))
            rig, lock = _engine._small_rig(width, coeff, month, ssn,
                                           fast_math)
            with lock:
                res, muf3 = rig.run(p, cfg, ant_idx, ant=ant)
            _scatter(out, rows, res.son, res.status, muf3, nf)
            continue
        ikey, slot, tables = _engine._device_iono(coeff, month, ssn)
        key = (_engine._device_id(), int(chunk), int(nstreams),
               bool(fast_math))
        rig, lock = _engine._BATCH_RIGS.get(key, lambda: PipelinedSetupLuffy(
            int(chunk), int(nstreams), fast_math=bool(fast_math)))

        def on_chunk(lo, hi, res, muf3, rows=rows):
            _scatter(out, rows[lo:hi], res.son, res.status, muf3, nf)

        with lock:
            rig.set_ionosphere([slot], tables, key=ikey)
            rig.run(p, cfg, ant, ant_idx, on_chunk=on_chunk)
    return _finish(out, echo)


def _run_cpu(echo, tx_antenna, rx_antenna, kw, fast_math):
    """``predict(backend='cpu')`` per circuit, into the batch arrays."""
    import warnings

    n, h, nf = len(echo["tx"]), len(echo["hours"]), len(echo["freqs"])
    out = _alloc(n, h, nf)
    for i in range(n):
        tb, rb = echo["tx_bearing"][i], echo["rx_bearing"][i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # already warned once above
            r = _engine.predict(
                tuple(echo["tx"][i]), tuple(echo["rx"][i]),
                freqs=echo["freqs"], month=echo["month"], ssn=echo["ssn"],
                hours=echo["hours"], tx_antenna=tx_antenna,
                rx_antenna=rx_antenna,
                tx_bearing=None if np.isnan(tb) else float(tb),
                rx_bearing=None if np.isnan(rb) else float(rb),
                system=echo["system"], method=echo["method"],
                coeff=echo["coeff"], pole=echo["pole"],
                long_path=bool(echo["long_path"][i]), fast_math=fast_math,
                tx_antenna_freq_range=kw["tx_freq_range"],
                rx_antenna_freq_range=kw["rx_freq_range"],
                tx_table_absolute=kw["tx_table_absolute"],
                rx_table_absolute=kw["rx_table_absolute"], backend="cpu")
        rows = slice(i * h, (i + 1) * h)
        for k in ("snr", "dbw", "field_dbu", "reliab", "nhp"):
            src = "field_dbu" if k == "field_dbu" else k
            out[k][rows] = getattr(r, src)
        out["mode"][rows] = np.vectorize(_MODE_CODE.__getitem__,
                                         otypes=[np.int8])(r.mode)
        for k in ("muf", "fot", "hpf", "status"):
            out[k][rows] = getattr(r, k)
    return _finish(out, echo)
