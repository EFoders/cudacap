"""THE DEVICE-SETUP ACCEPTANCE TEST: raw parameters in, same answer out.

``tests/cuda/test_setup_chain.py`` proved the per-hour physics chain on the
device.  This proves the COMPOSITION around it -- ``setup_batch``
(cuda/kernels/setup.cuh) building a whole CircuitState from 32 bytes of
parameters -- against the host path it replaces, on three counts:

1. **The state.**  Every leaf of ``driver.UPLOAD_FIELDS`` (the measured
   read-before-written set plus the two carry fields) and every /RAYS/ input
   word, device-built vs ``engine.build_circuit`` + ``pack_circuit``,
   bit-exact.  Fields OUTSIDE that set are deliberately NOT compared: the
   setup kernel does not write them because the LUFFY kernel does not read
   them, which is the same claim ``LuffyArena`` relies on when it declines to
   re-zero between submissions.  Asserting on them would be asserting on
   scratch.

2. **The answer.**  ``SetupLuffy`` (setup on the device, then LUFFY) against
   ``luffy_results`` (build on the host, upload, then LUFFY) -- /SON/ entire,
   XLUF, JMODE and the status, bit-exact.  This is the claim that actually
   matters, and it is independent of (1): a field could be equal and unused,
   or unequal and unused, and only this test can tell the difference.

3. **The two things the host used to compute that nobody else can.**  IPFLAG
   (which LUFFY branches on) and the per-hour ALLMUF/FOT/HPF triple, which
   ``engine.predict`` reads off the built circuit and which therefore has to
   come back from the device now that there is no built circuit on the host.

The cases span short/medium/long paths, both hemispheres, a trans-Pacific
great circle and a near-antipodal one past the 10,000 km GCDLNG threshold (so
both IPFLAG arms are live), four seasons, quiet and active sun, all 24 hours,
and both the VOACAP and ICEPAC model configurations (FPROB PSC(4) 0 and 0.70,
CCIR and URSI), plus a ragged batch that exercises the padding path.
"""

import numpy as np
import pytest

from voacap_gpu import engine
from voacap_gpu.cuda.driver import (CIRCUIT_DTYPE, RAYS_DTYPE,
                                       RAYS_UPLOAD_LEAVES, SETUP_CONFIG_DTYPE,
                                       SETUP_PARAMS_DTYPE, UPLOAD_LEAVES,
                                       IonoSlot, SetupLuffy, fetch_luffy,
                                       luffy_results, pack_circuit,
                                       setup_iono_tables,
                                       setup_struct_sizeof,
                                       shared_antenna_tables,
                                       shared_iono_tables)
from voacap_gpu.scalar.redmap import redmap

from tests.cuda.harness import assert_bitexact, requires_cuda

pytestmark = requires_cuda

FREQS = (5.0, 7.0, 10.0, 14.0, 18.0, 21.0, 25.0)

# (tx, rx, npsl) — the last two clear GCDLNG (10,000 km), so IPFLAG 200 is live.
_PATHS = [
    ((51.50, -0.13), (48.85, 2.35), 0),         # short, KM=1
    ((40.00, -105.0), (40.00, -70.0), 0),       # medium, KM=3
    ((40.00, -105.0), (50.00, 0.0), 0),         # long, KM=5
    ((-23.00, 30.00), (-35.00, 138.00), 0),     # southern hemisphere
    ((34.05, -118.24), (35.68, 139.77), 0),     # trans-Pacific, > GCDLNG
    ((-33.87, 151.21), (64.80, -147.70), 0),    # southern -> arctic, > GCDLNG
]

# (label, coeff, fprob) — the two Prop-SL model toggles, as _resolve_model
# resolves them.
_MODELS = [("voacap", "CCIR", (1.0, 1.0, 1.0, 0.0)),
           ("icepac", "URSI", (1.0, 1.0, 1.0, 0.70))]


def _reference(cases, slots, cant, method):
    """The host path: build_circuit per case, then the production LUFFY trip.

    Returns the packed entry states as well as the results, so the state
    comparison and the answer comparison both come from the same run.
    """
    bufs, rays_bufs, ipflags, mufs = [], [], [], []
    for tx, rx, npsl, hour, iso, sys_ in cases:
        s = slots[iso]
        circ, ip = engine.build_circuit(
            tx, rx, s.month, s.ssn, hour, FREQS, method=method, system=sys_,
            cant=cant, pole=engine.POLE_DEFAULT, npsl=npsl, iono=s.iono)
        b, r = pack_circuit(circ)
        bufs.append(b)
        rays_bufs.append(r)
        ipflags.append(ip)
        it = circ.time.it
        mufs.append((circ.muf.allmuf[it], circ.muf.fot[it], circ.muf.hpf[it]))
    bufs = np.stack(bufs)
    rays_bufs = np.stack(rays_bufs)
    iono_idx = np.array([c[4] for c in cases], np.int32)
    res = luffy_results(bufs, rays_bufs, np.array(ipflags, np.int32),
                        shared_iono_tables([s.iono for s in slots]), iono_idx,
                        shared_antenna_tables([cant]),
                        np.zeros(len(cases), np.int32))
    return bufs, rays_bufs, np.array(ipflags, np.int32), \
        np.array(mufs, np.float32), res


def _params(cases):
    p = np.zeros(len(cases), SETUP_PARAMS_DTYPE)
    for i, (tx, rx, npsl, hour, iso, _sys) in enumerate(cases):
        p[i] = (tx[0], tx[1], rx[0], rx[1], hour, npsl, iso, iso)
    return p


def _leaf(buf, path):
    """A dotted leaf path out of a structured array, contiguous and flat."""
    v = buf
    for part in path.split("."):
        v = v[part]
    return np.ascontiguousarray(v).ravel()


def _slots_and_cases(models, hours, paths=None, system=None):
    """(slots, configs, cases) for one sweep.  One ionosphere per model."""
    paths = paths if paths is not None else _PATHS
    slots, cfgs, systems = [], [], []
    for name, coeff, fprob in models:
        base = engine.System(**(system or {}))
        _, sysr = engine._resolve_model(name, coeff, base)
        month, ssn = (7, 160.0) if name == "voacap" else (1, 30.0)
        slots.append(IonoSlot(
            redmap(engine._COEFFS_DIR, coeff, month, np.float32(ssn)),
            month, ssn))
        systems.append(sysr)
    cases = [(tx, rx, npsl, h, iso, systems[iso])
             for iso in range(len(models))
             for (tx, rx, npsl) in paths
             for h in hours]
    return slots, systems, cases


def _configs(systems, method):
    cfgs = np.zeros(len(systems), SETUP_CONFIG_DTYPE)
    for i, s in enumerate(systems):
        cfgs[i] = engine.setup_config(s, method=method, freqs=FREQS)
    return cfgs


def test_setup_struct_abi():
    """numpy's packing of the two parameter blocks equals the C layout.

    The same proof ``circuit_state_sizeof`` carries for CircuitState: every
    member is 4 bytes, so agreeing totals means agreeing offsets.
    """
    assert setup_struct_sizeof() == (SETUP_PARAMS_DTYPE.itemsize,
                                     SETUP_CONFIG_DTYPE.itemsize)
    assert SETUP_PARAMS_DTYPE.itemsize == 32, "32 bytes a circuit-hour"


def test_setup_params_is_a_hundredth_of_the_input_set():
    """The reduction this whole path exists for, stated as a number."""
    from voacap_gpu.cuda.driver import UPLOAD_BYTES_PER_THREAD
    assert UPLOAD_BYTES_PER_THREAD == 3252      # 3,228 + LAYERS.HS's 24
    assert UPLOAD_BYTES_PER_THREAD // SETUP_PARAMS_DTYPE.itemsize >= 100


@pytest.mark.parametrize("method", [30, 20, 25])
def test_device_setup_matches_host_build(method):
    """The built state, the answer, IPFLAG and the MUF triple, all at once."""
    hours = list(range(1, 25))
    slots, systems, cases = _slots_and_cases(_MODELS, hours)
    cant = engine.isotropic_cant(power_kw=1.0)
    cfgs = _configs(systems, method)

    bufs, rays_bufs, ip_exp, muf_exp, res_exp = _reference(
        cases, slots, cant, method)

    rig = SetupLuffy(len(cases), slots,
                     shared_iono_tables([s.iono for s in slots]),
                     shared_antenna_tables([cant]))
    params = _params(cases)
    rig.upload(params, cfgs, np.zeros(len(cases), np.int32))

    # --- (1) the state, BEFORE LUFFY mutates it ---------------------------
    rig.launch_setup()
    got, got_rays, _st = fetch_luffy(rig.state)
    got = got.view(CIRCUIT_DTYPE)
    got_rays = got_rays.view(RAYS_DTYPE)
    for path in UPLOAD_LEAVES:
        assert_bitexact(_leaf(got, path), _leaf(bufs, path),
                        f"circuit.{path} [method {method}]")
    for path in RAYS_UPLOAD_LEAVES:
        assert_bitexact(_leaf(got_rays, path), _leaf(rays_bufs, path),
                        f"rays.{path} [method {method}]")

    # --- (3) IPFLAG and the MUF triple the host can no longer compute -----
    assert np.array_equal(rig._cp.asnumpy(rig._d_ipflag), ip_exp), "IPFLAG"
    assert_bitexact(rig._cp.asnumpy(rig._d_muf3).ravel(), muf_exp.ravel(),
                    f"ALLMUF/FOT/HPF [method {method}]")
    # METHOD 25 (all modes) returns 100 unconditionally -- it is the first arm
    # of engine._ipflag and never reaches the GCDLNG test.  The other two do,
    # and the case list has paths on both sides of 10,000 km.
    want = {100} if method == 25 else {100, 200}
    assert set(ip_exp.tolist()) == want, (
        "the IPFLAG arms this method can reach", set(ip_exp.tolist()))

    # --- (2) the answer ----------------------------------------------------
    rig.launch_kernel()
    res, muf3 = rig.results()
    _assert_results_equal(res, res_exp, f"method {method}")
    assert_bitexact(muf3.ravel(), muf_exp.ravel(), "muf3 survives LUFFY")


def _assert_results_equal(got, exp, tag):
    for name in got.son.dtype.names:
        assert_bitexact(np.ascontiguousarray(got.son[name]).ravel(),
                        np.ascontiguousarray(exp.son[name]).ravel(),
                        f"son.{name} [{tag}]")
    assert_bitexact(got.xluf.ravel(), exp.xluf.ravel(), f"xluf [{tag}]")
    assert np.array_equal(got.jmode, exp.jmode), f"jmode [{tag}]"
    assert np.array_equal(got.status, exp.status), f"status [{tag}]"
    assert (exp.status == 0).all(), f"every circuit succeeded [{tag}]"


def test_ragged_batch_pads_without_changing_the_answer():
    """A batch shorter than the rig's fixed width.

    /RAYS/'s SoA column stride IS the launch's thread count (raysview.cuh), so
    the width cannot shrink per submission; ``run`` pads by REPEATING a real
    row -- never zeros, which would hang the sample-area loop -- and drops the
    padding on the way out.  The first `c` rows must be untouched by that.
    """
    hours = [3, 11, 19]
    slots, systems, cases = _slots_and_cases(_MODELS[:1], hours)
    cant = engine.isotropic_cant(power_kw=1.0)
    cfgs = _configs(systems, 30)
    _b, _r, ip_exp, muf_exp, res_exp = _reference(cases, slots, cant, 30)

    # a rig 7 lanes wider than the batch, so every submission is ragged
    rig = SetupLuffy(len(cases) + 7, slots,
                     shared_iono_tables([s.iono for s in slots]),
                     shared_antenna_tables([cant]))
    res, muf3 = rig.run(_params(cases), cfgs, np.zeros(len(cases), np.int32))
    assert len(res.son) == len(cases), "the padding is dropped"
    _assert_results_equal(res, res_exp, "ragged")
    assert_bitexact(muf3.ravel(), muf_exp.ravel(), "ragged muf3")


def test_rig_reuse_is_stateless():
    """Two different batches through ONE rig, back to back.

    The device buffers are zeroed once at construction and never again -- the
    same reuse ``LuffyArena`` relies on.  If ``setup_batch`` failed to write
    any field LUFFY reads, batch 2 would silently inherit batch 1's value and
    this is where it would show.  Run B, then A, then B again: B's two answers
    must agree, and A's must equal its own reference.
    """
    slots_a, systems_a, cases_a = _slots_and_cases(_MODELS[:1], [6, 18])
    slots_b, systems_b, cases_b = _slots_and_cases(
        _MODELS[1:], [1, 12], paths=_PATHS[:2])
    # one rig has to serve both, so it carries both ionospheres
    slots = slots_a + slots_b
    systems = systems_a + systems_b
    cases_b = [(tx, rx, npsl, h, 1, s) for (tx, rx, npsl, h, _i, s) in cases_b]
    cant = engine.isotropic_cant(power_kw=1.0)
    cfgs = _configs(systems, 30)

    _b, _r, _ip, muf_a, res_a = _reference(cases_a, slots, cant, 30)
    _b, _r, _ip, muf_b, res_b = _reference(cases_b, slots, cant, 30)

    n = max(len(cases_a), len(cases_b))
    rig = SetupLuffy(n, slots, shared_iono_tables([s.iono for s in slots]),
                     shared_antenna_tables([cant]))
    z = lambda c: np.zeros(len(c), np.int32)        # noqa: E731
    first, mf1 = rig.run(_params(cases_b), cfgs, z(cases_b))
    got_a, mfa = rig.run(_params(cases_a), cfgs, z(cases_a))
    again, mf2 = rig.run(_params(cases_b), cfgs, z(cases_b))

    _assert_results_equal(got_a, res_a, "A after B")
    _assert_results_equal(first, res_b, "B first")
    _assert_results_equal(again, res_b, "B again")
    assert_bitexact(mfa.ravel(), muf_a.ravel(), "A muf3")
    assert_bitexact(mf2.ravel(), mf1.ravel(), "B muf3 reproduces")


def test_virtim_rows_cover_every_hour():
    """The 25 precomputed VIRTIM rows ARE what the host would have computed.

    The whole per-hour saving rests on GMT being the requested hour on the
    ITIM >= 0 arm, so row `h` must equal ``virtim(iono, h)`` for every hour a
    caller can ask for.  Cheap to state, expensive to get wrong.
    """
    from voacap_gpu.cuda.driver import SETUP_NAB, SETUP_NHOUR
    from voacap_gpu.scalar.virtim import virtim

    slots = [IonoSlot(redmap(engine._COEFFS_DIR, "CCIR", 7,
                             np.float32(160.0)), 7, 160.0)]
    ab = setup_iono_tables(slots)["ab"].reshape(SETUP_NHOUR, SETUP_NAB)
    for h in range(SETUP_NHOUR):
        assert_bitexact(ab[h], virtim(slots[0].iono, np.float32(h)),
                        f"ab row {h}")
