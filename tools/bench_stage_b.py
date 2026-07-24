"""Benchmark the Stage B chain: CPU vs CUDA, few circuits vs many.

WHAT A "CIRCUIT" MEANS HERE.  A lane is one transmitter-to-point path at one
frequency and one hop order.  For the ionospheric front half that is the same
thing as a grid point; for the loss back half a single grid point contributes
one lane per frequency, so a 1,000-point area map at 11 frequencies is 11,000
back-half lanes.  The two halves are therefore timed separately and the
summary says which is which, because quoting one number for both would flatter
whichever is cheaper.

WHAT IS NOT MEASURED, and why the totals are a floor rather than a prediction:
there is no composed driver yet, so this times the pieces back to back without
the control flow that will eventually sit between them (the frequency sweep,
the hop loop, LUFFY's mode bookkeeping).  Real throughput will be lower.  What
these numbers ARE good for is the shape: where the crossover against scalar
sits, which stage dominates, and whether a stage is launch-bound or
compute-bound.

The scalar baseline is Stage A on ONE circuit, timed directly rather than
assumed -- it is the thing the GPU has to beat, and it is what the existing
program does today.
"""

import argparse
import pathlib
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
COEFFS = ROOT / "reference" / "voacapl" / "voacapl" / "itshfbc" / "coeffs"

MONTH, SSN, GMT = 4, 120.0, 12.0
PARAMS = ("fi1", "fi2", "fi3", "yi1", "yi2", "yi3", "hi1", "hi2", "hi3")
ANGLES = np.array([0.1, 1.0, 3.0, 6.0, 10.0, 16.0, 24.0, 35.0, 50.0, 74.0],
                  dtype=np.float32)
NFV = 30


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _time(fn, device, repeat=3, warmup=1):
    """Best-of-`repeat` wall clock in ms.  Best, not mean: we are after the
    machine's capability, and a slow sample is another process, not physics."""
    for _ in range(warmup):
        fn()
    _sync(device)
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def _grid(n):
    """`n` points spread over the globe, as an area run would be."""
    side = int(np.ceil(np.sqrt(n)))
    lat = np.linspace(-70.0, 70.0, side, dtype=np.float32)
    lon = np.linspace(-180.0, 180.0, side, dtype=np.float32)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    return la.ravel()[:n].copy(), lo.ravel()[:n].copy()


# --------------------------------------------------------------- front half

def bench_front(n, device):
    """Grid -> layer params -> profile -> ionogram -> reflectrix."""
    from voacap_gpu.batched.area import build_map
    from voacap_gpu.batched.fobby import fobby_batch
    from voacap_gpu.batched.genion import genion_batch
    from voacap_gpu.batched.profile import lecden_batch

    lat, lon = _grid(n)
    ang = torch.as_tensor(ANGLES, device=device)

    def run():
        _, p = build_map(COEFFS, "CCIR", MONTH, SSN, GMT, lat, lon,
                         device=device)
        htr, fnsq, _, _, _ = lecden_batch(*[p[k] for k in PARAMS])
        fv, _, ht = genion_batch(htr, fnsq, p["fi1"], p["fi2"], p["fi3"])
        fobby_batch(ang, fv, ht)

    return _time(run, device)


# ----------------------------------------------------------- the ray search

def _ray_inputs(n, device):
    """A realistic FINDF input set, built once and reused."""
    from voacap_gpu.scalar.sang import ANG as SANG_ANG
    from voacap_gpu.batched.area import build_map
    from voacap_gpu.batched.genion import genion_batch
    from voacap_gpu.batched.profile import lecden_batch

    lat, lon = _grid(n)
    _, p = build_map(COEFFS, "CCIR", MONTH, SSN, GMT, lat, lon, device=device)
    htr, fnsq, _, _, _ = lecden_batch(*[p[k] for k in PARAMS])
    fvert, hprim, htrue = genion_batch(htr, fnsq, p["fi1"], p["fi2"],
                                       p["fi3"])
    ih = torch.arange(1, NFV + 1, device=device, dtype=torch.float32)
    afac = (0.1 + 0.03 * ih).unsqueeze(1).expand(NFV, n).contiguous()
    nang = 40
    ang = torch.as_tensor(np.asarray(SANG_ANG, dtype=np.float32),
                          device=device).unsqueeze(1).expand(nang,
                                                             n).contiguous()
    from voacap_gpu.batched.fobby import fobby_batch
    ifob = fobby_batch(ang[:, 0].contiguous(), fvert, htrue)
    freq = torch.full((n,), 14.1, device=device)
    amind = torch.full((n,), 3.0, device=device)
    nangt = torch.full((n,), nang, dtype=torch.long, device=device)
    return (htrue, hprim, fvert, afac, p["fi2"], p["fi3"], ang, nangt, ifob,
            freq, amind)


def bench_findf(n, device, graph=True):
    from voacap_gpu.batched.findf import findf_batch
    args = _ray_inputs(n, device)
    return _time(lambda: findf_batch(*args, use_cuda_graph=graph), device,
                 repeat=2)


# ---------------------------------------------------------------- back half

def bench_back(n, device):
    """REGMOD, the per-mode loss budget -- the elementwise back half.

    Inputs are synthesised rather than run through the front half, because
    the front half produces one lane per GRID POINT and this consumes one per
    (grid point x frequency).  Synthesising keeps the lane count honest.
    """
    from voacap_gpu.scalar.antennastate import CantennaState
    from voacap_gpu.scalar.readant import read_antenna
    from voacap_gpu.batched.gain import BatchedCantenna
    from voacap_gpu.batched.regmod import regmod_batch

    rng = np.random.default_rng(7)
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32),  # noqa
                                  device=device)
    u = lambda lo, hi: t(rng.uniform(lo, hi, n))                    # noqa

    array, aeff = read_antenna(
        ROOT / "reference" / "voacapl" / "voacapl" / "itshfbc" / "antennas"
        / "samples" / "sample.14").p2p_tables()
    c = CantennaState()
    for iat in (1, 2):
        c.array[1:31, 1:92, iat] = array
        c.aeff[1:31, iat] = aeff
        c.offazim[iat] = np.float32(0.0)
    c.numants = 2
    c.iats[1], c.xfqs[1], c.xfqe[1] = 1, np.float32(2.0), np.float32(30.0)
    c.iats[2], c.xfqs[2], c.xfqe[2] = 2, np.float32(2.0), np.float32(30.0)
    c.pwrdba[1] = np.float32(40.0)
    cant = BatchedCantenna.from_stage_a(c, device)

    modes = {"delmod": torch.cat([torch.zeros(1, n, device=device),
                                  t(rng.uniform(3, 40, (6, n)))]),
             "hpmod": torch.cat([torch.zeros(1, n, device=device),
                                 t(rng.uniform(90, 400, (6, n)))]),
             "htmod": torch.cat([torch.zeros(1, n, device=device),
                                 t(rng.uniform(90, 350, (6, n)))]),
             "fvmod": torch.cat([torch.zeros(1, n, device=device),
                                 t(rng.uniform(2, 12, (6, n)))]),
             "afmod": torch.cat([torch.zeros(1, n, device=device),
                                 t(rng.uniform(0, 2, (6, n)))]),
             "itmod": torch.cat([torch.zeros(1, n, device=device),
                                 t(rng.integers(1, 4, (6, n)))]).to(
                                     torch.int32),
             "ghop": u(0.1, 0.4)}
    muf = {name: t(rng.uniform(5, 25, (4, n)))
           for name in ("ymuf", "yfot", "yhpf", "hpmuf", "htmuf", "fvmuf")}
    muf["sigl"] = t(rng.uniform(0.5, 2.0, (4, n)))
    muf["sigu"] = t(rng.uniform(0.5, 2.0, (4, n)))
    muf["htmuf"] = t(rng.uniform(90, 350, (4, n)))
    muf["hpmuf"] = t(rng.uniform(120, 400, (4, n)))
    muf["fvmuf"] = t(rng.uniform(2, 12, (4, n)))
    sigd = {name: u(0.1, 2.0) for name in
            ("acav", "htloss", "xnuz", "hnu", "xve", "afe", "bfe", "asm",
             "dsl", "dsu")}
    args = dict(
        modes=modes, muf=muf, sigd=sigd, cant=cant,
        sigpat=t(rng.uniform(0.001, 0.02, (5, n))),
        epspat=t(rng.uniform(4, 20, (5, n))),
        km=torch.full((n,), 3, dtype=torch.long, device=device),
        fi1=u(2, 5), gyz=u(0.8, 1.6), fs1=u(1, 5), fs2=u(1, 5), hs=u(95, 120),
        ifob13=u(1000, 9000), rcnse=u(-160, -140),
        freq=u(5, 25), gcd=u(0.05, 0.5))
    return _time(lambda: regmod_batch(**args), device)


# ------------------------------------------------------------ Stage A scalar

def bench_scalar_front():
    """Stage A: one circuit, grid params -> ionogram -> reflectrix."""
    from voacap_gpu.scalar.f32math import F32
    from voacap_gpu.scalar.fobby import fobby
    from voacap_gpu.scalar.genion import genion
    from voacap_gpu.scalar.layerstate import LayerState
    from voacap_gpu.scalar.lecden import lecden
    from voacap_gpu.scalar.raysstate import RaysState

    def run():
        s = LayerState(km=1)
        s.iedp = -1
        for lay, (fi, yi, hi) in enumerate(
                [(3.2, 20.7, 132.6), (4.0, 40.5, 112.4), (9.4, 186.5, 478.8)],
                start=1):
            s.fi[lay, 1], s.yi[lay, 1], s.hi[lay, 1] = (F32(fi), F32(yi),
                                                        F32(hi))
        lecden(s, 1)
        genion(s, 1)
        r = RaysState()
        r.nang = len(ANGLES)
        r.ang[1:len(ANGLES) + 1] = ANGLES
        fobby(s, r, 1)

    return _time(run, "cpu", repeat=5, warmup=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max", type=int, default=100_000,
                    help="largest lane count to try")
    ap.add_argument("--cpu-max", type=int, default=10_000,
                    help="largest lane count to run on CPU (it gets slow)")
    args = ap.parse_args()

    have_cuda = torch.cuda.is_available()
    counts = [c for c in (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000)
              if c <= args.max]

    scalar_front = bench_scalar_front()
    print(f"\nStage A scalar, one circuit, front half: "
          f"{scalar_front:.3f} ms/circuit")
    if have_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for title, fn, kw in (
            ("FRONT HALF  grid -> params -> profile -> ionogram -> reflectrix",
             bench_front, {}),
            ("RAY SEARCH  FINDF (lockstep + CUDA graph)", bench_findf, {}),
            ("BACK HALF   REGMOD, the per-mode loss budget", bench_back, {})):
        print(f"\n{title}")
        print(f"{'lanes':>9} {'CPU ms':>10} {'CUDA ms':>10} {'speedup':>8} "
              f"{'us/lane':>9}")
        for n in counts:
            try:
                cpu = fn(n, "cpu", **kw) if n <= args.cpu_max else None
            except (RuntimeError, MemoryError) as e:
                cpu, _ = None, print(f"  cpu n={n}: {e}")
            gpu = None
            if have_cuda:
                try:
                    gpu = fn(n, "cuda", **kw)
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    print(f"  cuda n={n}: {type(e).__name__}")
                    torch.cuda.empty_cache()
            sp = f"{cpu / gpu:.1f}x" if (cpu and gpu) else "-"
            per = f"{gpu * 1000 / n:.3f}" if gpu else "-"
            print(f"{n:>9} {cpu if cpu else float('nan'):>10.1f} "
                  f"{gpu if gpu else float('nan'):>10.1f} {sp:>8} {per:>9}")
            if have_cuda:
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
