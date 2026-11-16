"""The device rigs ``predict`` and ``predict_batch`` reuse between calls.

Split out of :mod:`voacap_gpu.engine` (which still re-exports every name
here).  The three caches below are this package's ONLY instances: the engine
facade and :mod:`voacap_gpu.batch` refer to these very objects.
"""

import threading
from collections import OrderedDict

import numpy as np

from voacap_gpu import system as _system
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.redmap import redmap


class _RigCache:
    """A small LRU of device rigs (or device tables), shared between threads.

    THREAD SAFETY.  A rig is mutable device state -- resident buffers that
    take one submission at a time -- so it must never run two requests at
    once.  Every entry is therefore a ``(value, lock)`` pair, and a caller
    holds the entry's OWN lock for the whole submit-and-fetch; the cache's
    lock guards only the dictionary (and construction, so two threads never
    build the same rig twice).  Two threads wanting the same key take turns
    on one rig; different keys can run at once.  Evicting an entry only drops
    the cache's reference, so a thread mid-run keeps its rig alive until it
    is done.

    NO REQUEST STATE SURVIVES A CALL.  Every submission uploads its own
    parameters, SYSTEM config and antenna tables, and the setup kernel
    rebuilds every input field (and resets /SON/) per row, so a cached rig
    cannot leak one caller's inputs into another's answer.  What is cached
    is only what is a pure function of the key: device buffers sized by the
    key, and the ionosphere tables, which depend only on (coefficients,
    month, SSN).  Compiled kernels are cached by ``cuda/compile.py`` itself.
    :func:`clear_device_cache` drops all of it and returns the memory.
    """

    def __init__(self, maxsize):
        self.maxsize = int(maxsize)
        self._lock = threading.Lock()
        self._d = OrderedDict()

    def get(self, key, factory):
        """``(value, lock)`` for ``key``, building it with ``factory()``."""
        with self._lock:
            hit = self._d.get(key)
            if hit is None:
                hit = (factory(), threading.Lock())
                self._d[key] = hit
                while len(self._d) > self.maxsize:
                    self._d.popitem(last=False)
            else:
                self._d.move_to_end(key)
            return hit

    def clear(self):
        with self._lock:
            self._d.clear()

    def __len__(self):
        return len(self._d)


# (device, n, iono key, fast_math) -> SetupLuffy(warp=True): predict's rigs,
# and predict_batch's for small batches.  ~55 KB of device state per row, so
# a full cache of the widest small rigs is ~100 MB.
_SMALL_RIGS = _RigCache(8)
# (device, chunk, nstreams, fast_math) -> PipelinedSetupLuffy for large
# batches.  These are big (chunk x nstreams x ~55 KB: ~0.7 GB at the default
# 4096 x 3), so only ONE is kept; clear_device_cache() releases it.
_BATCH_RIGS = _RigCache(1)
# iono key -> (IonoSlot, shared_iono_tables dict): host-side, ~100 KB each.
_IONO_TABLES = _RigCache(16)


def clear_device_cache():
    """Drop every cached device rig and return cupy's pooled memory.

    ``predict`` (CUDA) and ``predict_batch`` keep their device rigs between
    calls so that a warm call pays no allocation (see ``_RigCache``); the
    large-batch rig holds ~0.7 GB.  Call this when the GPU is wanted for
    something else.  Safe at any time: a call already running keeps its rig
    until it finishes, and the next call rebuilds what it needs.
    """
    _SMALL_RIGS.clear()
    _BATCH_RIGS.clear()
    _IONO_TABLES.clear()
    import sys
    cp = sys.modules.get("cupy")
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def _device_id():
    import cupy as cp
    return int(cp.cuda.Device().id)


def _device_iono(coeff, month, ssn):
    """``(key, IonoSlot, shared_iono_tables)`` for one (coeff, month, SSN).

    The same ``redmap`` call ``_predict_host`` makes (and the same F32(ssn)
    ``build_circuit`` puts on /DON/), so the device setup chain reads exactly
    the tables the host one would.
    """
    from voacap_gpu.cuda.driver import IonoSlot, shared_iono_tables

    key = (str(_system._COEFFS_DIR), coeff, int(month), float(F32(ssn)))

    def make():
        st = redmap(_system._COEFFS_DIR, coeff, int(month), np.float32(ssn))
        return IonoSlot(st, int(month), ssn), shared_iono_tables([st])

    (slot, tables), _lock = _IONO_TABLES.get(key, make)
    return key, slot, tables


def _small_rig(n, coeff, month, ssn, fast_math):
    """A cached warp-per-circuit-hour ``SetupLuffy`` with this ionosphere.

    ``luffy_batch_warp`` puts one circuit-hour on one warp (a lane per
    frequency) and is bit-identical to the thread-per-circuit kernel
    (``tests/cuda/test_luffy_warp.py``); for a few dozen circuit-hours it has
    ~4x lower latency, because the thread kernel's latency is a whole LUFFY
    run on one lane.  Returns ``(rig, lock)``.
    """
    from voacap_gpu.cuda.driver import SetupLuffy

    ikey, slot, tables = _device_iono(coeff, month, ssn)
    key = (_device_id(), int(n), ikey, bool(fast_math))
    return _SMALL_RIGS.get(key, lambda: SetupLuffy(
        int(n), [slot], tables, warp=True, fast_math=bool(fast_math)))
