"""Where LUFFY runs: the CUDA kernel, or the scalar Stage-A reference (CPU).

Split out of :mod:`voacap_gpu.engine` (which still re-exports every name
here).  :func:`predict_circuits` runs pre-built circuits on either backend;
:func:`_resolve_backend` is the ``backend=`` / ``'auto'`` rule ``predict`` and
``predict_batch`` share.
"""

import warnings

import numpy as np

from voacap_gpu.status import _SCALAR_REFUSALS, _scalar_status


BACKENDS = ("cuda", "cpu", "auto")


def _cuda_usable() -> bool:
    """True if cupy imports and sees at least one CUDA device.

    Does not compile anything: a device that is present but cannot build the
    kernels (no NVRTC, an unsupported architecture) still fails at launch.
    """
    try:
        import cupy
        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def _resolve_backend(backend, fast_math=False) -> str:
    """``backend`` -> ``'cuda'`` or ``'cpu'`` (see :func:`predict`)."""
    if backend not in BACKENDS:
        raise ValueError(
            f"backend={backend!r}: expected one of {', '.join(BACKENDS)}")
    if backend == "auto":
        if _cuda_usable():
            backend = "cuda"
        else:
            warnings.warn(
                "backend='auto': no usable CUDA device (cupy missing or no "
                "GPU visible), falling back to the CPU scalar reference, "
                "which is roughly 1000x slower than the GPU kernel (~70 ms "
                "per circuit-hour).  Pass backend='cpu' to silence this.",
                UserWarning, stacklevel=3)
            backend = "cpu"
    if backend == "cpu" and fast_math:
        raise ValueError(
            "fast_math=True selects the CUDA native-f32 libm and has no CPU "
            "equivalent; use backend='cuda' or fast_math=False")
    return backend


def _predict_circuits_cpu(circuits, ipflags):
    """The CPU backend: the scalar reference LUFFY, one circuit at a time.

    Same contract as the CUDA path -- mutates each Circuit in place and
    returns ``(circuits, status)`` -- with a refusal caught per circuit and
    turned into the kernel's status code by :func:`_scalar_status`.  Like the
    kernel, a refused circuit keeps whatever partial /SON/ it had reached;
    :class:`Results` masks the whole hour.
    """
    from voacap_gpu.scalar.luffy import luffy

    ipflags = np.asarray(ipflags, np.int32)
    assert len(ipflags) == len(circuits)
    status = np.zeros(len(circuits), np.int32)
    for k, (c, ip) in enumerate(zip(circuits, ipflags)):
        try:
            luffy(c, int(ip))
        except _SCALAR_REFUSALS as e:
            status[k] = _scalar_status(e)
    return circuits, status


# ---------------------------------------------------------------------------
# Running circuits on the GPU
# ---------------------------------------------------------------------------

def predict_circuits(circuits, ipflags, fast_math=False, *, backend="cuda"):
    """Run pre-built Stage-A circuits through LUFFY.

    With ``backend='cuda'`` (the default) this is a thin wrapper over
    ``cuda/driver.py``: shares each distinct ``iono`` / ``cant`` on the device
    once, launches one thread per circuit, and writes the mutated /SON/ (and
    the rest) back into the given Circuit objects in place.  With
    ``backend='cpu'`` the scalar reference ``scalar.luffy.luffy`` runs each
    circuit in turn instead (no cupy import); ``'auto'`` picks as
    :func:`predict` does.  Returns ``(circuits, status)`` -- ``status`` is
    per-circuit (0 = success; the codes are the same on both backends, see
    :class:`Results`).

    ``fast_math`` opts into the native-f32 device libm, which is NOT bit-exact
    against Stage A (see ``cuda/f32math.cuh``); off by default, and refused
    (``ValueError``) on the CPU backend.
    """
    if _resolve_backend(backend, fast_math) == "cpu":
        return _predict_circuits_cpu(circuits, ipflags)
    from voacap_gpu.cuda.driver import (launch_luffy, pack_circuit,
                                         shared_antenna_tables,
                                         shared_iono_tables, unpack_circuit)
    I32 = np.int32
    ipflags = np.asarray(ipflags, I32)
    assert len(ipflags) == len(circuits)

    ionos, iono_idx, seen_i = [], [], {}
    cants, ant_idx, seen_a = [], [], {}
    for c in circuits:
        ki = id(c.iono)
        if ki not in seen_i:
            seen_i[ki] = len(ionos)
            ionos.append(c.iono)
        iono_idx.append(seen_i[ki])
        ka = id(c.cant)
        if ka not in seen_a:
            seen_a[ka] = len(cants)
            cants.append(c.cant)
        ant_idx.append(seen_a[ka])

    packed = [pack_circuit(c) for c in circuits]
    bufs = np.stack([p[0] for p in packed])
    rays_bufs = np.stack([p[1] for p in packed])
    out, out_rays, status = launch_luffy(
        bufs, rays_bufs, ipflags, shared_iono_tables(ionos),
        np.array(iono_idx, I32), shared_antenna_tables(cants),
        np.array(ant_idx, I32), fast_math=fast_math)
    for c, b, r in zip(circuits, out, out_rays):
        unpack_circuit(b, r, c)
    return circuits, status
