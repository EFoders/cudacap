"""THE BIT-EXACT GATE: the LUFFY kernel's answers, frozen as SHA-256 digests.

Every performance change to ``cuda/kernels/*.cuh`` or to the LUFFY launch in
``cuda/driver.py`` has to leave the production answer BIT-IDENTICAL for valid
inputs (the fidelity contract in ``cuda/compile.py``).  The per-routine tests
prove each leaf against Stage A and ``test_luffy_replay.py`` proves the whole
kernel on the oracle decks, but neither covers the production shapes a
performance change can disturb: a big launch, a pipelined run with ragged
chunks and recycled slot buffers, a world grid in its natural order, a
random order that maximises warp divergence, every model arm, file antennas.

So this file runs a FIXED set of device workloads through the production rigs
and compares a SHA-256 digest of every output column -- each /SON/ field,
XLUF, JMODE, the status and the per-hour ALLMUF/FOT/HPF triple -- against
digests generated from the unmodified tree:

    world      2-degree world grid subset, natural (hour-major) cell order
    random     random endpoints and hours, randomly ordered
    m20/m25    METHOD 20 and 25 (METHOD 30 is every other workload)
    icepac     URSI coefficients + FPROB PSC(4) 0.70 (the sporadic-E arms)
    longpath   NPSL = 1, the long-way-round geometry
    ant13      type-13 (azimuth-dependent) file antenna, one /cantenna/ per
               circuit, and a type-14 receiver
    ant14      type-14 file antennas at both ends
    refusals   inputs the kernel refuses: a 0.5 MHz frequency (GENOIS's
               receive GAIN leaves its table), an area-coverage receive
               antenna, an area-coverage transmit antenna -- so the status
               column is NOT all zeros
    hourguard  hour 25 pushed past the host validation (SetupLuffy only):
               the setup kernel's refusal

Every workload runs through BOTH ``PipelinedSetupLuffy`` (ragged chunks, two
streams) and ``SetupLuffy``, each built fresh, on the one-thread-per-circuit
``luffy_batch`` kernel; the two must agree with each other before either is
compared with the stored digest.  ``fast_math=False`` is the gate.  The
``fast_math=True`` digests are recorded separately (``FAST_DIGESTS``): fast
math is not bit-exact against Stage A, so they are NOT required to equal the
strict ones -- only to stay what they were, which catches a change to the
fast path that nobody meant to make.

REGENERATING.  ``uv run python tools/kernel_perf/make_gate.py`` recomputes
every digest (three times, refusing to write unless all three runs agree) and
rewrites the block between the markers below.  Regenerate ONLY deliberately --
the point of the gate is that a kernel change does not move these.  The one
sanctioned regeneration so far is the status renumbering (unique status
codes, ``cuda/kernels/status.cuh``), which moved the ``status`` digests of
the two refusal workloads and nothing else.
"""

import hashlib
import pathlib

import numpy as np
import pytest

from tests.cuda.harness import requires_cuda

pytestmark = requires_cuda

REPO = pathlib.Path(__file__).resolve().parents[2]
ANTENNAS = REPO / "tests" / "data" / "antennas"

FREQS9 = (3.5, 5.0, 7.0, 10.0, 14.0, 18.0, 21.0, 24.9, 28.0)
FREQS7 = (5.0, 7.0, 10.0, 14.0, 18.0, 21.0, 25.0)
OTTAWA = (45.42, -75.70)


# ---------------------------------------------------------------------------
# Workload builders.  Each returns (slots, cfgs, ant, ant_idx, params, how),
# deterministic from fixed seeds.  `how` is "both" or "serial".
# ---------------------------------------------------------------------------

def _slot(coeff, month, ssn):
    from voacap_gpu import engine
    from voacap_gpu.cuda.driver import IonoSlot
    from voacap_gpu.scalar.redmap import redmap
    return IonoSlot(redmap(engine._COEFFS_DIR, coeff, month,
                           np.float32(ssn)), month, float(ssn))


def _cfg(method=30, freqs=FREQS9, system=None):
    from voacap_gpu import engine
    from voacap_gpu.cuda.driver import SETUP_CONFIG_DTYPE
    cfgs = np.zeros(1, SETUP_CONFIG_DTYPE)
    cfgs[0] = engine.setup_config(system or engine.System(), method=method,
                                  freqs=freqs)
    return cfgs


def _iso_ant():
    from voacap_gpu import engine
    from voacap_gpu.cuda.driver import shared_antenna_tables
    return shared_antenna_tables([engine.isotropic_cant(power_kw=1.0)])


def _random_params(n, seed, tx=None, npsl=0):
    from voacap_gpu.cuda.driver import SETUP_PARAMS_DTYPE
    rng = np.random.default_rng(seed)
    p = np.zeros(n, SETUP_PARAMS_DTYPE)
    if tx is None:
        p["tlatd"] = rng.uniform(-75.0, 75.0, n)
        p["tlongd"] = rng.uniform(-180.0, 180.0, n)
    else:
        p["tlatd"], p["tlongd"] = tx
    p["rlatd"] = rng.uniform(-85.0, 85.0, n)
    p["rlongd"] = rng.uniform(-180.0, 180.0, n)
    p["hour"] = rng.integers(1, 25, n)
    p["npsl"] = npsl
    return p


def wl_world():
    """Every 8th cell of the 2-degree world grid x 6 hours, hour-major --
    the production world-grid order (``tools/kernel_perf/bench_kernel.py``
    runs the whole of it)."""
    from voacap_gpu.cuda.driver import SETUP_PARAMS_DTYPE
    from voacap_gpu.gridjob import grid_cells
    lat, lon, *_ = grid_cells(-90, 90, -180, 180, 2.0)
    lat, lon = lat[::8], lon[::8]
    hours = (1, 5, 9, 13, 17, 21)
    nc = len(lat)
    p = np.zeros(nc * len(hours), SETUP_PARAMS_DTYPE)
    p["tlatd"], p["tlongd"] = OTTAWA
    p["rlatd"] = np.tile(lat, len(hours))
    p["rlongd"] = np.tile(lon, len(hours))
    p["hour"] = np.repeat(np.array(hours), nc)
    return ([_slot("CCIR", 7, 120.0)], _cfg(30), _iso_ant(),
            np.zeros(len(p), np.int32), p, "both")


def wl_random():
    p = _random_params(8192, 20261115)
    return ([_slot("CCIR", 7, 120.0)], _cfg(30), _iso_ant(),
            np.zeros(len(p), np.int32), p, "both")


def wl_m20():
    p = _random_params(2048, 2020, tx=OTTAWA)
    return ([_slot("CCIR", 3, 40.0)], _cfg(20), _iso_ant(),
            np.zeros(len(p), np.int32), p, "both")


def wl_m25():
    p = _random_params(2048, 2025, tx=OTTAWA)
    return ([_slot("CCIR", 11, 80.0)], _cfg(25, FREQS7), _iso_ant(),
            np.zeros(len(p), np.int32), p, "both")


def wl_icepac():
    from voacap_gpu import engine
    p = _random_params(4096, 7070)
    sysi = engine.System(fprob=(1.0, 1.0, 1.0, 0.70))
    return ([_slot("URSI", 1, 40.0)], _cfg(30, FREQS9, sysi), _iso_ant(),
            np.zeros(len(p), np.int32), p, "both")


def wl_longpath():
    p = _random_params(2048, 1001, npsl=1)
    return ([_slot("CCIR", 7, 120.0)], _cfg(30), _iso_ant(),
            np.zeros(len(p), np.int32), p, "both")


def _file_ant_workload(tx_file, rx_file, seed, ncells, tx_bearing, rx_bearing):
    """One /cantenna/ per CELL (a type-13 pattern is azimuth-dependent), four
    hours per cell, cells in random order."""
    from voacap_gpu import engine
    from voacap_gpu.cuda.driver import shared_antenna_tables
    from voacap_gpu.scalar.readant import read_antenna
    txa = read_antenna(str(ANTENNAS / tx_file))
    rxa = read_antenna(str(ANTENNAS / rx_file))
    rng = np.random.default_rng(seed)
    rlat = rng.uniform(-80.0, 80.0, ncells)
    rlon = rng.uniform(-180.0, 180.0, ncells)
    cants = []
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for la, lo in zip(rlat, rlon):
            cants.append(engine.antcalc_cant(
                txa, rxa, OTTAWA, (float(la), float(lo)), power_kw=10.0,
                tx_bearing=tx_bearing, rx_bearing=rx_bearing))
    ant = shared_antenna_tables(cants)
    hours = (2, 8, 14, 20)
    from voacap_gpu.cuda.driver import SETUP_PARAMS_DTYPE
    p = np.zeros(ncells * len(hours), SETUP_PARAMS_DTYPE)
    p["tlatd"], p["tlongd"] = OTTAWA
    p["rlatd"] = np.repeat(rlat, len(hours))
    p["rlongd"] = np.repeat(rlon, len(hours))
    p["hour"] = np.tile(np.array(hours), ncells)
    ant_idx = np.repeat(np.arange(ncells, dtype=np.int32), len(hours))
    order = rng.permutation(len(p))
    return p[order], ant_idx[order], ant


def wl_ant13():
    p, ai, ant = _file_ant_workload("sample.13", "sample.14", 1313, 384,
                                    tx_bearing=45.0, rx_bearing=None)
    return [_slot("CCIR", 4, 100.0)], _cfg(30), ant, ai, p, "both"


def wl_ant14():
    p, ai, ant = _file_ant_workload("sample.14x", "sample.14", 1414, 256,
                                    tx_bearing=None, rx_bearing=None)
    return [_slot("CCIR", 10, 150.0)], _cfg(30), ant, ai, p, "both"


def wl_refusals():
    """Three refusal kinds, interleaved with good rows."""
    from voacap_gpu import engine
    from voacap_gpu.cuda.driver import shared_antenna_tables
    from voacap_gpu.scalar.antennastate import AREA_COVERAGE
    good = engine.isotropic_cant(power_kw=1.0)
    rx_area = engine.isotropic_cant(power_kw=1.0)
    rx_area.offazim[2] = AREA_COVERAGE
    tx_area = engine.isotropic_cant(power_kw=1.0)
    tx_area.offazim[1] = AREA_COVERAGE
    ant = shared_antenna_tables([good, rx_area, tx_area])
    cfgs = np.concatenate([_cfg(30), _cfg(30)])
    cfgs[1]["frel"][3] = np.float32(0.5)     # GENOIS GAIN below 1 MHz
    p = _random_params(3072, 4242)
    rng = np.random.default_rng(4243)
    kind = rng.integers(0, 4, len(p))        # 0 good, 1 sub-1MHz, 2/3 area
    p["cfg"] = (kind == 1).astype(np.int32)
    ant_idx = np.where(kind >= 2, kind - 1, 0).astype(np.int32)
    return [_slot("CCIR", 7, 120.0)], cfgs, ant, ant_idx, p, "both"


def wl_hourguard():
    p = _random_params(512, 2525)
    p["hour"][::3] = 25                       # refused by setup.cuh
    return ([_slot("CCIR", 7, 120.0)], _cfg(30), _iso_ant(),
            np.zeros(len(p), np.int32), p, "serial")


WORKLOADS = {
    "world": wl_world, "random": wl_random, "m20": wl_m20, "m25": wl_m25,
    "icepac": wl_icepac, "longpath": wl_longpath, "ant13": wl_ant13,
    "ant14": wl_ant14, "refusals": wl_refusals, "hourguard": wl_hourguard,
}


# ---------------------------------------------------------------------------
# Running and digesting
# ---------------------------------------------------------------------------

def _run_serial(wl, fast_math):
    from voacap_gpu.cuda.driver import SetupLuffy, shared_iono_tables
    slots, cfgs, ant, ai, p, how = wl
    iono = shared_iono_tables([s.iono for s in slots])
    rig = SetupLuffy(len(p), slots, iono, ant, fast_math=fast_math)
    if how == "serial":                      # bypass the host hour check
        rig._upload(np.ascontiguousarray(p), np.ascontiguousarray(cfgs),
                    np.ascontiguousarray(ai, np.int32)).launch()
        out = rig.results()
    else:
        out = rig.run(p, cfgs, ai)
    del rig
    return out


def _run_pipe(wl, fast_math):
    from voacap_gpu.cuda.driver import PipelinedSetupLuffy, shared_iono_tables
    slots, cfgs, ant, ai, p, _ = wl
    iono = shared_iono_tables([s.iono for s in slots])
    chunk = max(256, (len(p) // 3) + 77)      # ragged: forces padding
    pipe = PipelinedSetupLuffy(chunk, 2, slots, iono, fast_math=fast_math)
    out = pipe.run(p, cfgs, ant, ai)
    del pipe
    return out


def digest_results(out):
    """{column: sha256 hex} over every result column of ``(LuffyResults,
    muf3)``."""
    res, muf3 = out
    d = {}

    def h(a):
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

    for name in res.son.dtype.names:
        d[f"son.{name}"] = h(res.son[name])
    d["xluf"] = h(res.xluf)
    d["jmode"] = h(res.jmode)
    d["status"] = h(res.status)
    d["muf3"] = h(muf3)
    return d


def run_workload(name, fast_math=False):
    """Digest one workload; raises if the two rigs disagree."""
    import cupy as cp
    wl = WORKLOADS[name]()
    ser = digest_results(_run_serial(wl, fast_math))
    cp.get_default_memory_pool().free_all_blocks()
    if wl[5] == "both":
        pip = digest_results(_run_pipe(wl, fast_math))
        cp.get_default_memory_pool().free_all_blocks()
        bad = [k for k in ser if ser[k] != pip[k]]
        if bad:
            raise AssertionError(
                f"{name}: PipelinedSetupLuffy != SetupLuffy on {bad}")
    return ser


def status_histogram(name, fast_math=False):
    """{code: count} of one workload's status column, for reports."""
    wl = WORKLOADS[name]()
    res, _ = _run_serial(wl, fast_math)
    u, c = np.unique(res.status, return_counts=True)
    return {int(a): int(b) for a, b in zip(u, c)}


# ---------------------------------------------------------------------------
# The stored digests -- generated by tools/kernel_perf/make_gate.py
# ---------------------------------------------------------------------------
# BEGIN GENERATED DIGESTS
DIGESTS = {'ant13': {'jmode': 'ce4dd332c63f06cb9963b618bfe1d1fdfa001d14af5125ae19edb66c168fc6eb',
           'muf3': 'f4a1290b640b5db030277c4a5558671d4abcd91d67957422cb2f2d72a83642f6',
           'son.angle': 'ce039918498e03e68825cb448398af62f04093c477132ad9dabb6a25d1d0c5ea',
           'son.angler': 'b9917657f491960b864c00ef3049a1c838a1f42d96cd812735c8f2da0c221faf',
           'son.cprob': 'e15a6b3f584114bffe82375b275eb9bb381099ad230f65955f74b4967082400f',
           'son.d10r': '964e5a7a1ce7f1e9ca7469cff5fa68fff6514c900f42546ea8785874a4b15912',
           'son.d10s': '076d4e0527be37999f54ad7df179b270096e3e12ff16609b3eb7464d6579ddf4',
           'son.d50r': '72479b67c6ca29960f798fc46f8cabd54df8ccfe524c75c8bf06dd814ed2ebd3',
           'son.d50s': 'f604afc748d55edba0b9efa980fa57d63634df0df6ba5a7086c7f81b216225a1',
           'son.d90r': '342be95fe9716e5caa9e11de67b0b660c197ed181adec3d3215d36976aa1ad6c',
           'son.d90s': '076d4e0527be37999f54ad7df179b270096e3e12ff16609b3eb7464d6579ddf4',
           'son.dblos': '5ac32944121f6313bb8c1957a378efdcd97989841cc2cb79057c6ab00d6fc8a6',
           'son.dblosl': '3439267e8527996f79840d5a06b4f5cab4666d1b6c1f2cd3c3c21f03dd1411a9',
           'son.dblosu': '7b10c10a1198b355c246ae9307e7350f2f71589a7a478a760b9a79b3168772be',
           'son.dbu': 'b0797a39e3dde29810e9d6e97bd58a4c814dd13c4dc23b33f5df75b25ca67054',
           'son.dbw': 'd6a428fa2eafc0be68902ca945396694ce1742d23c511f174cb32ece8bcb93f7',
           'son.delay': 'ab8c5af0b34e99d9b7789962cb42b2ba492712bda370eb4bd5abe7000521ac7d',
           'son.dl_nois': '3b4235e73e11efb8f4181115de69bc77e101b2324e428fe7ef138bd1a8aadd5f',
           'son.du_nois': '0af02ea9cf5f4b37ecbb9a35b64ad143530027a42bb958edb4873cc1a8eac1b7',
           'son.gainr': 'e4cc434a8de543d04664523daf7fa4244ffc48d44e0eb355af00789aa3d780ff',
           'son.gaint': '9af60e25ff216d4c2a8109f2837bd33a1e7033c4e61e7722c2db37153bbf11a2',
           'son.mdl': 'b547957c34e728bff027e2cd8f1044dff636e28b90efdf929808231090e02125',
           'son.mode': '5b1d0bcfc6f2c25d903e3a8180d83346f493e76cbc07470cb56ba9a4a19cf738',
           'son.moder': '0cc587bf9abe7d0628365a25f377c63b28173ecd2502719d723e87aa020a10c6',
           'son.nhp': '5fb7e6eafb7eda8e13f21f36d1934981098268c1dfd03a25d189ccd902a8f69f',
           'son.probmp': '0266fb27935088dd3b3f90bfa571cf2628a1951b0b2252bffbbeac5cadf4b4c8',
           'son.reliab': '382c702aef315145089278ae9aba13d7230fb52304007b26371f3277b51fcbc6',
           'son.rneff': '59f5754f640e5f758cc965dc48f180a5cd57d408c7abb327554f5129c0d38e66',
           'son.sndb': '15db60ad63b7787866d51202ee0a1670eeaeb08e7f450b6864373a7ce12781bd',
           'son.snpr': 'e80013d989110ffd8a7fc59e6d5faeaa7beb7cf5863356767337af7eaa2a0830',
           'son.snrlw': 'e5d4203b88d1b3602846b922dd834a063e58b8e54c369f30fe40c9ab68b9ad6d',
           'son.snrup': 'feb638b045bbca88b2afdc3297ea339287e92d77a18573609dfacdd5f91c30aa',
           'son.snxx': '2cd70d5a8766ba1aaf1137b0b0419ab4c07e13552b01ffb2bd0662fa1ea3fbfe',
           'son.sprob': 'a6ce1926e78246a6e3ded3c95bc03a2d63ee0d15bdb1e5c5ae5e4e3f11c118a2',
           'son.vhigh': '3731b63cc5b1929998eebd14dcd8b8313a013e08626c5a3bce2207311dc48aba',
           'son.xnynois': '9fda4c5ed5447462b7e2096a867cc7d9481f2b602f1a2be12ecf5c184d5873b0',
           'status': 'fd9243e1ba57263ed469c3bdbd7ade6ec5254e7ed924a9f5737fa44749933cc0',
           'xluf': '016e948dea7be3c1c43860fcc59f657c269bb9c41123dc745c8a92fe6fdf0b83'},
 'ant14': {'jmode': 'b3bb3d89643e06efef8d709b0e5e3541ee712410c7b04b2b835133fd958573fe',
           'muf3': '9b9bbf087fb5ff15fa2fb433f5aa1ec58b247434c6637230d2d3dd1cf76085bb',
           'son.angle': '2f386334761ba8879d0418e67987de5afeb1f32f21f04cbec16a3afa0862741f',
           'son.angler': 'f1afd7c3d2dc6988f7e16084f3393fc4c7b591bb567865fc26f3848dc6f7580e',
           'son.cprob': '4a52d9445f8fef8885b9e028c669fee1e59ea89b71b306e33bf27532b9bda789',
           'son.d10r': 'b44181f875cd6a1767abe646ca8a44f649c03a06075b41a9e5a678b744599d9e',
           'son.d10s': '63844e7da7fa30cb594aec9c6d0d2272c8543c5e178ddecc3ade58e9175ebc7d',
           'son.d50r': 'c79a9ff7df10b46dc81186b96edea3bc10b8848490a821956a148d65bdd43215',
           'son.d50s': 'c9de3941c9b8b8a8917ce315ee7686bf7b058c32abcb8697a4cca13bec4e7ec1',
           'son.d90r': '05dd4c3fb71cf104b39d97c605e20bf7406d7164c456c3a8b6fdf83186fc25ae',
           'son.d90s': '63844e7da7fa30cb594aec9c6d0d2272c8543c5e178ddecc3ade58e9175ebc7d',
           'son.dblos': 'e1a515cf3efb9dbd68e4b7de2092dd592f835ba3b430aaeea7a46e59d494d8cf',
           'son.dblosl': '95cd5017cd7e3207f90f659bec6e0c057f53294321aeb0cf7e6536ebc3d7f1ef',
           'son.dblosu': 'daa0e938ae3f4d1e0ce1f2a62234a981e918c55350682b4361babf47d285e85f',
           'son.dbu': 'af66ce0c99d18e71f8dd003e99b419cef745a9ce316f22906d125a86935120d3',
           'son.dbw': 'bda884751e47a655fd3c1e85f72268b2596e3ed4221fa0b98104106356bfca06',
           'son.delay': '9f169232a5225cc751c144ff55a5434550c585544e377d6e54c0a027cc4ab3a0',
           'son.dl_nois': '40ce32e0f8a47d3adb981c448eba585527b8de6b80c73db0d73b3935152c173d',
           'son.du_nois': 'f71c224a698e41cab6ecaa9291ef651054285ba575bafd6fc9354eeba28df4f8',
           'son.gainr': 'db9848b25e3380944d82abec5a724305f21a6a36a6cce1908039b917c9ef71b5',
           'son.gaint': '97749c3fb8901019c40f798e722414543ebc6c0629104ba7fdb510ba62f080d6',
           'son.mdl': '7c9c9d5d65c92f0d75ec277b8178296547d578f7a6c12f1c6eb1bc740ab7b746',
           'son.mode': '61736fab4a1392841cc9d5f41ad6269ef2741ac19ba3e7f40fbced082b6a7eb3',
           'son.moder': 'a062218520fff66db864728277229477977c96d31e5fb2ca0d1d1b579c2ffda8',
           'son.nhp': 'e0078ae6a51cbf97052ec223af87e02b56feaa47e58b0c5eb6994a7ef71d99e2',
           'son.probmp': 'f28254ff555c637d19a922050f155b36e3b34f1c3c5dec8519f64882ea21408b',
           'son.reliab': '33b5c208fc84dfe69470e5b88f7d55e15cebb793007216c435e6f015c091ffc3',
           'son.rneff': '7209b567395ac8645c660ca7064c5055d5b5c1cfe4c82fd1fddefa9737f9da58',
           'son.sndb': '8fc106ad15d640b00c7e1c2f4ae3be0bbd42fc4cfd9c985981d20cc60137982b',
           'son.snpr': '94fd33261841830fa22336a771034f66b1fc91c6181e114101c8a7afb1d4bcd4',
           'son.snrlw': '2abb039daab3367a0b4849d49035888191019608f8c8a9e85318160482669517',
           'son.snrup': 'b12e17fb011d8f849ea5ee64c6af17e5ec3f68af72c0a95efa1fc0efd2a369ce',
           'son.snxx': '6baea8fdc768027350d8a10d36d3d81e76595c619144719773f2859ab07eca0b',
           'son.sprob': '75efd4f6cb9094a50e0e309a8fb1260797c51e7a07b1e1117da52abd067fd4ab',
           'son.vhigh': '3e8351518ecb9064977a7c5a062f0516bf25ffeadabbe8d8ed924cb51ac356a6',
           'son.xnynois': '156efc9ce6a5e6d75c778d91cfef3a67ba47b98e15dae3d217922756284e1163',
           'status': 'ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7',
           'xluf': '99146e972721676422d37b8d0e7b33d81f033291be903d6fffdeba1977131215'},
 'hourguard': {'jmode': '771702f0f4e324d5a882815c3e3515478e578aff450bc9064341678399e5209f',
               'muf3': 'e583a9db6d0c0b0a41ea211e9f9d65ba9a85e11066f39b0131f6d444d5e88078',
               'son.angle': 'd74e6b947bfeeb62f824a7d28c882765b9f2beac9fd55fb9ec171c04b2cc89f0',
               'son.angler': 'd4f3ae402fc22099a0b2a68c2578b15582170ff1adfec6dd0f8038fb56c8499c',
               'son.cprob': '6dbfc10b5d492ebfae5054f3dc551995638988c93e63ac601354557ab3c3941f',
               'son.d10r': 'a9dcfb16acb43ff2e2bb9e992228c5a96a9c083151311c096bbbab8d70740494',
               'son.d10s': 'f484cad4c3cb2a5163f8c22fedbceb1cdb911d9e27ac42aff2dbd13a8541de9c',
               'son.d50r': '23376e279c693c3870be8b8df5276a4e2ad5d5c7db6e80c80df0627b6cc42692',
               'son.d50s': 'f6f8711dadd8697aa1c578eaea3c6fc3080eada0e75e5a10f97d0d4f70cf2419',
               'son.d90r': 'e243d027526d847c831dfa7bd2b9d253acdc3040c64ac0200d9816ae272da462',
               'son.d90s': 'f484cad4c3cb2a5163f8c22fedbceb1cdb911d9e27ac42aff2dbd13a8541de9c',
               'son.dblos': '6d953046ba67e9fa423887ef2def8ab9a8ed9f49b4cffe6da31d237728ad987d',
               'son.dblosl': 'd527a4a21d62284fe0c67490a2ea75fb9fc9ea2771fe33d311cf1d111904f357',
               'son.dblosu': '1c6e5a0a9fb774345d20fbadbd5b8d06ffe53a76c839e782dce6404a4575bc1f',
               'son.dbu': 'e2382260ebae571fe85b9237806ad567cbec6dad049a97929cc364c81af74716',
               'son.dbw': '31f5de46efc7881a781b3dd6f0961c192655cc97663f4b1fa269bcb9cbf78da1',
               'son.delay': '44fd896cd6d17f129e8f5481c01a2dd261cc599867b313df0cedc2672351f4fd',
               'son.dl_nois': 'd23d6824ecae33bbef7a04cc5d8b69efe38a92c764317ed523709a1bc4d4f0c5',
               'son.du_nois': '96bffefc2244cf79704292f344ca2b4f2acefc78583b0d67c71bfb0125c9e61e',
               'son.gainr': '18619b678a5c207a971a0aa931604f48162e307c57ecdec450d5f095fe9f32c7',
               'son.gaint': '18619b678a5c207a971a0aa931604f48162e307c57ecdec450d5f095fe9f32c7',
               'son.mdl': '22756872985e273a5f689e533d0f397bfef0f16709f5e1505215948ac8a8ddba',
               'son.mode': '59bbbd209077a5dde419d6ee80268a2d15b805a12eca343cb8c036572fbf131e',
               'son.moder': '3dad12d2b61680563d9606832aee892747f11fbe84e691562871fefd0e671267',
               'son.nhp': '1a303c152c53fbe1509bc25bcf1363762bb6797af380f315a52d05ec889df228',
               'son.probmp': '603c95291676a015ecc97568d992dfa72c0677031e036f543ed1c53701dd8a99',
               'son.reliab': '619d9ba7b6b1415ac19a439b3e19366cf6dd4a28a3469c54c90c709ba87a3127',
               'son.rneff': '18619b678a5c207a971a0aa931604f48162e307c57ecdec450d5f095fe9f32c7',
               'son.sndb': '0d15810f16b66ea0fb0eca00711edb70613ab56f9be4503dd7288dde1fdf8e87',
               'son.snpr': '4d50fd1e077d87d9c4a40552319e0975eedf23a4383f96988e28ba23dffe145e',
               'son.snrlw': '479befc1f8bf7eb0b4565d96859049411657d33e8cf78034dc8ee9855f2095a7',
               'son.snrup': '5836335739ad214e1261940758149f4da34a8fdabf3ebf5edbd9505f54840ed5',
               'son.snxx': '25c67ff39039a8a38bedf4609bdd8c360e041e5e1960f52f1f300147e6132ef7',
               'son.sprob': 'b2616150b528bfc5addabff570115f37e25b86c2c2e80435c9e6c40b22c926fc',
               'son.vhigh': '1bb34fc7672346e670899f6018861d2460fcdca1e7dacafc020a90879775e74a',
               'son.xnynois': '9dfeeec21e9c1568d518802acf80ca322022d0f74839174ae7e30373e5d8c959',
               'status': 'e074d9ebe0a7947e98308f86f8c3d25ae5d4595c370122cd4b3a9c46839765a2',
               'xluf': '37d8a97fb605af344650d4cfdc1660fb6fc4b9d73ee7bdd0aea3b8ba333ea0e5'},
 'icepac': {'jmode': '63f49cae896ebb66f7a781e25e2c5eaceb7f632198ac62474a7260e7be2bc0d8',
            'muf3': '9f3807fa092118e625dc048e86c650c56d7b1c1f33c9f1bb47a3266283bb47e8',
            'son.angle': '785befa459a6b3e1c2210131ac18e7deb8c1590564f950e109de71b551de7a09',
            'son.angler': 'd2bc95ef182d9ad6f39d51aed3536415fa008d9b40a5da0882bea7aebb374373',
            'son.cprob': '0335121e5cd0c87ee1063c82833ccdcf75ab8387965194ce80ea70f2156426ff',
            'son.d10r': '95a8844288e4b921275f378f457a8b92e1eb2dd9d8a8bb674c684bf556466607',
            'son.d10s': '7f6e59fed2bdbf49ca97ad222ebb25d2114549002ec4e90482deb225b58427d2',
            'son.d50r': '2c1bc136082664746c7e05cf5c4da68ed128388d1d95e8ae8f9ee36c5384197c',
            'son.d50s': 'eac8ecf99dba18d64cf01e229aa5994b076e96be0a13bbeb3840084bbbc51bad',
            'son.d90r': 'bad2821fff466677438c27a461f23e47e4584b2ed17bcdbd7c30b747759f529f',
            'son.d90s': '7f6e59fed2bdbf49ca97ad222ebb25d2114549002ec4e90482deb225b58427d2',
            'son.dblos': '8dad6399431b6a46e7a0343f5c2737d208800e61e95601933b87a72e825f0785',
            'son.dblosl': 'c075d99df7bc1dd1ad52b470243097ec4b23037fbfe69c61a5198a76fe63fb0e',
            'son.dblosu': '9a2a9493ed9c2b458b1af95fa64dfb9cf5f3f3ca028f3a8baf0c6391a1acffe9',
            'son.dbu': 'c901295bf9a76f4629eabb38065fe95cfb5194e2090fbc96a828a91b23d9dc00',
            'son.dbw': '50d12038be6545e5953a2a3ad11de6d7abde3f221e8db3061868e1077f273d64',
            'son.delay': '833905dd4804476f18722b69570815b6596e96d82585eeb0dcbdbc872d7a93f7',
            'son.dl_nois': 'b1c22ffafbb2c39fae2873b4a0b17a8649c595098bbf4857432951ed6a3019b3',
            'son.du_nois': '28eed443791da44f3a6c617909f8c0164291363abf545f648aec5c28969cb3e7',
            'son.gainr': 'd2a23a7de3af6bf70d164a0ce45201371aca7624e30079d30cccab1cb6cf52e3',
            'son.gaint': 'd2a23a7de3af6bf70d164a0ce45201371aca7624e30079d30cccab1cb6cf52e3',
            'son.mdl': '8330b25f96b379410fb2d553db01ba0f4b77ff938fc9a611e62a309946565269',
            'son.mode': '4724be65d71ac61d982de79611b03055b997bc873aadd635982bed43afc1eb64',
            'son.moder': 'a17204576eda9fca9172239b8e8cc1948c1fb9c98c33060304b238aac233e8b0',
            'son.nhp': '4c33b201c315bede133b02ccd10e88dbe3785609e21d7680d6ac1f0a5213217a',
            'son.probmp': 'a97428cd8db918217666ddb0cbaf19b3af4cecce97b339ceaf13739202104554',
            'son.reliab': 'ebd662a17f4a9769ddd3235a0c3149e5a5fd32de02389e9e16a3d15826e74fe4',
            'son.rneff': 'd2a23a7de3af6bf70d164a0ce45201371aca7624e30079d30cccab1cb6cf52e3',
            'son.sndb': 'a1282e6337c19164cfb0d3a4f98e7e038d764e0aa733a1aa7464e20beb2b56ac',
            'son.snpr': '0891126429bb64d5d153c37e62d60ca5e201f2de309103e4334b094109d1664e',
            'son.snrlw': '62ae88e5234a909ccc5cc490294fd57ea34982b4ba6fe5070a9a695e9d928c37',
            'son.snrup': '032c2a42e180627e8840dc7ce0bd629d47f1ceaf58415c1d015f218489eb8fa5',
            'son.snxx': '80f5e51626674bfd7a2bffd14c6dc9bf741a0183f60b50417c0abde288de23cb',
            'son.sprob': '286bb1cb6b26f250d94a274a275b44660555be2ea0f548ebf59452ce3107526d',
            'son.vhigh': '5115180a1167741551de2776eb4b04a049a83757ea7360a0da8d4d898239231a',
            'son.xnynois': '53dfbb7bb24b95405808014e962b7764c67922997c8b4c754df47249311351ab',
            'status': '4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe',
            'xluf': '92fe71732db478fc8306259b99670d2e6bffd0a932e9d5eb75ae16a555ef2a2f'},
 'longpath': {'jmode': '60af5db0dcfc35014218249528871639feb1ef95a62cfc7bd8955bced796c377',
              'muf3': 'b1fcc64fd0923cffa854656a4800b0dcaf6f792f4706c4308049ba7773b5d29c',
              'son.angle': 'f4c57e66fb25c68e97dd177a0b1b000807775684083e1a04287b4c21ed0d7fe0',
              'son.angler': '9bde147338feddf28d4a8d6ef7068915e1c71f51fa86b7ebc7cb7548f184418b',
              'son.cprob': '19daf3049c0403bba614b80069441df67e3b948fb8b0e55f4e46005c2077bab8',
              'son.d10r': '2880fabe963088bf8bf20a5b98d5deea1971a7e6a5dc2c77ad2f8b747dc77080',
              'son.d10s': '1434f0fbb24e922f63e8e5a35ca08145a76d2290523b244676fd4211a6d70887',
              'son.d50r': '17c500c4f576bbfc46c4feb94a3633c5566c8f9e6c36a1e871799b088376444a',
              'son.d50s': '84b51514d7db8c49cc4e85024bd43c18d31b40454b7467285b0ff4aedff94225',
              'son.d90r': '83b0c58b49a059826f465da624f04577df3d4430262f3ff985e4414d3fb8b850',
              'son.d90s': '1434f0fbb24e922f63e8e5a35ca08145a76d2290523b244676fd4211a6d70887',
              'son.dblos': '41d611bb2a326417d062a6f9e474e9d45d1320557cefa67b121d86df5d4427fb',
              'son.dblosl': 'dc28d1d13631573a86d56957d5d92ebd5979695da840eaac6d82c0b862f086af',
              'son.dblosu': '3d8586e3f30fde2c53787829d346aafcf7cf13aac166d84272e41098374dd004',
              'son.dbu': '87751b92eae1aa996e543de96541e1fd6a7ead63c675876749211466aaeef8c5',
              'son.dbw': '25a98eed0c23b45fc6cbe4158f62a8d757541a8c1f0bd7c714721413807ddfb3',
              'son.delay': '6ad0b43485ec85157949ad3d678d81d08bb3c94c86bb09fc358e0de0792c2480',
              'son.dl_nois': 'a8eb17dfdea67abdac23d36c893109d7f47d8598f56923b2dbda5c705a490807',
              'son.du_nois': '864d9a0e98f32063b9afd7b8358c3c0379a184125d8a08c8525da90a18430662',
              'son.gainr': '96c94d0826105fe47c587fd79e8869ce5edbfbacdddab9f4f30c5fecba2ca6a3',
              'son.gaint': '96c94d0826105fe47c587fd79e8869ce5edbfbacdddab9f4f30c5fecba2ca6a3',
              'son.mdl': '7da9ab0b4999ea2d9df517e3a96fd0c87c197c8146917301221c67a154c7bd7e',
              'son.mode': '38fae3dba900800f0cca01e7d75672f762446a421c0f7fa0c8fcbb83097847dc',
              'son.moder': 'c1ca6198692afbe383308c32a25b29324998828d253a0fda0e8891eb43874b6c',
              'son.nhp': 'aa5facea4cb99fd0d8324d6984abeb121e1e9335080a1fb70c8b5687d5319193',
              'son.probmp': '96c94d0826105fe47c587fd79e8869ce5edbfbacdddab9f4f30c5fecba2ca6a3',
              'son.reliab': 'bc50dc12d552438b43ffb5217bb0860265541a9df38729aa8578b5b575d80e67',
              'son.rneff': '96c94d0826105fe47c587fd79e8869ce5edbfbacdddab9f4f30c5fecba2ca6a3',
              'son.sndb': '95df9ce762593325e94441ba9175e1f44e2d77ddccf3529aab8a249a2c5e382d',
              'son.snpr': '1ad645594e09a4ee3eb4d849f7845515b228139b6e17ccc0ef4b9ea2b558a1df',
              'son.snrlw': '7e2ddbf02ecfc03c091914ed761bb370b8abc72c2782a3aaeeb634d1529c7c90',
              'son.snrup': '8d5162d5b39060d93b2f3998c07d60d971abb6169250182f0ebfec6214eaf125',
              'son.snxx': '2574efcc77fbcf2fe7a754e23bc0054ca47f8ae78c2e802dad4861ed47544a04',
              'son.sprob': 'e5e2f4ed07d85eed99ef08c9245b473c27b9d45e2a0825c13bc4e5ace2bddce5',
              'son.vhigh': 'd22a9d0b3dc91173da794d24a27f690c6bd912c5fb78d280f7a8f1a8d20d3b96',
              'son.xnynois': 'ed70d69adb51f4d3733cd494f1127ee7d01e474f5b77ea2cb5ac958aac0a1af6',
              'status': '9f1dcbc35c350d6027f98be0f5c8b43b42ca52b7604459c0c42be3aa88913d47',
              'xluf': '18774bfced33056fc695cce04eb09c4ee02d821f2f2ea9fd21ab3b8e33f76ac2'},
 'm20': {'jmode': '76e85d7f8ab67095807a80c903c0dd2968b661cf7c5997eef444734f395f0f02',
         'muf3': '590962335e5760d6e2a762b4e95f98c1f90bb22627e3cf84649428811e474d7c',
         'son.angle': '1e0a43ddc256e6b5410210384e8cbfc6398292a25b6ed8ac9a941519621efe16',
         'son.angler': '08b7ce43188bacf3c63543970607c3eb5840d52b624c3cdbfc9c4c0fd328c154',
         'son.cprob': '34f8ed9f5d28b58a4c426874f8d951d8a534350ebe04a72be25c6db1b43725d0',
         'son.d10r': 'e86d0c5dcb8cc55c8a87f2c139a35c52c6f021cbe4ead69935865d3b268b7403',
         'son.d10s': '3d4cf7eeae83d1d1bfd32bcc74ffbb11425f2c3b93a645aee22595e0263b10db',
         'son.d50r': '182b7d0693a83535e531e6520ebb51201049d8d08cd73a9898ee636c8a81ea46',
         'son.d50s': '018f416fa3b123e0c16aa0cfff60346458c8b4f86eb3125adc9eaeb09f3a72ad',
         'son.d90r': '5b81c9eb9508fe31bb09241d5a88daafad312022c05fd6dd4b09c2872e1a3e4e',
         'son.d90s': '3d4cf7eeae83d1d1bfd32bcc74ffbb11425f2c3b93a645aee22595e0263b10db',
         'son.dblos': '7d47b7b3a35ef4eeeeaf6c72ff76081113fcdcbcb685b57d3c8408425e3bf09c',
         'son.dblosl': '7ffae1cb116c619093c70079acec4a898879a48b3a787e1aa3819ed6dd033fa2',
         'son.dblosu': 'abfd166ce2ee869c1c797a5bfd2ca763cf245cc2d42855ef549e34a4726176e4',
         'son.dbu': '1aeba10a7eb3e15ba85fabdee8cff9e3d12bec1e66edf1fa78b17e06260bbace',
         'son.dbw': '3071fb696db263f78c3ab40427daff332880cf1388a69557ecb0126cc25b17b0',
         'son.delay': '68fbdbf5067c331a49a6d129acc80264fd676cbe45242c6d64237ded5ab88c99',
         'son.dl_nois': 'fcb7f49801a58915b8de782974401b08290e9d1841e9292558eb944db0e5dc9a',
         'son.du_nois': 'e3e604106f4fb8dd9ef85837cda1c2b11c44f6cc51c11f479386b9aa06098dcb',
         'son.gainr': '96c94d0826105fe47c587fd79e8869ce5edbfbacdddab9f4f30c5fecba2ca6a3',
         'son.gaint': '96c94d0826105fe47c587fd79e8869ce5edbfbacdddab9f4f30c5fecba2ca6a3',
         'son.mdl': 'ceba350da25201b97ba73435c1d659914557939af78c23155ca56f2a3a334d10',
         'son.mode': '5aebb81aeb01ee1c6dfd591ccf37d7b26e64b2f83a627c948ca50a4232c91717',
         'son.moder': 'c82ea7351b1b3069de93f7fc932fb9703e79e9d4e9b5cb512e2cf0144a634167',
         'son.nhp': '789e357bf1d4b87b5c3a9b0589c334f3bf3f0e65a2f03423305c924a1a30cd3d',
         'son.probmp': 'cd6cd6e43346818c09c713ee5e16f69e98dca865343fe39d968d9a00318e07d7',
         'son.reliab': '02b7a64f73ae66c2225dc5824add9e59fefebb26db122ce1422fe65fb51dab2d',
         'son.rneff': '96c94d0826105fe47c587fd79e8869ce5edbfbacdddab9f4f30c5fecba2ca6a3',
         'son.sndb': 'f64da608d2031cdafa374d7ef4c7bd1f4ef114a3996511f1cb2c2053db67744b',
         'son.snpr': '9dd374ff2b3a587883131e34eae4fe4941842bda68ca64174957d7e3082bc4b1',
         'son.snrlw': 'de75546e68daf1ad2bfccb5209792c4481788411942d3570d09e20b5927a36dc',
         'son.snrup': '020f14ae5fefc6f2145c68e6524b7fbcad77f8231a68f9cd465f9322cd1a0cb3',
         'son.snxx': '4d6b7578b0eb331efeca7ff6fee459ba6d43b533af81d9e08a73dcf624d22185',
         'son.sprob': '3851b50813bdc33bc2e87a82dadf4455825cd2ca5e356d338c3f13bfdab9436b',
         'son.vhigh': '8d50c3e2b7cfe93d933cca2b130e5643ca6880e246adb6377b51c1be671aa839',
         'son.xnynois': 'd4fbbc6b41ab9abd3f8f7afe0e0c9b0e74315520f14563d2084e901445f5a51d',
         'status': '9f1dcbc35c350d6027f98be0f5c8b43b42ca52b7604459c0c42be3aa88913d47',
         'xluf': '18774bfced33056fc695cce04eb09c4ee02d821f2f2ea9fd21ab3b8e33f76ac2'},
 'm25': {'jmode': '96a7cf9d82dc7fa53cb0d9e9ead7f797f5b1217fece1de3469266f0866aeb7e2',
         'muf3': '8bdab1b707c90a94273e40d6ab57b77c786ae7c35016bad3c00a3e0e05a01e13',
         'son.angle': 'e03745c1354ff12717f3385d9a9c6af5d8ac16a6d46df63d94d91920fb5f0400',
         'son.angler': '96c94d0826105fe47c587fd79e8869ce5edbfbacdddab9f4f30c5fecba2ca6a3',
         'son.cprob': '4b4b2caaba528e02f7099cf7769907c1a0ef0ca10667b47124483cb56e238dd1',
         'son.d10r': '6c05e8b7009e97ec41e49738b35d2ce46d5604b047934cee9f90519703ce6395',
         'son.d10s': 'b6207a00a3bd809fdd19e696b29d261e1b1fc8b35f90d63248fb645f125cb70d',
         'son.d50r': 'bbf8a379ef410fa05aefc2015428e7a806ee4af2742eba563d83af8ee946f76c',
         'son.d50s': '403039b63124e8e133695ca23a87c755f1fe7ef9860814244aee132cb7626402',
         'son.d90r': 'c915da0feb1e74caca11bef67c1cce7e63586913f0477f58379601b683ff8534',
         'son.d90s': 'b6207a00a3bd809fdd19e696b29d261e1b1fc8b35f90d63248fb645f125cb70d',
         'son.dblos': 'cb7ae4ac53a9937436e6322316e42a7a33750d97e07d0e6df5254e503a9294b0',
         'son.dblosl': '35bbf3dd96b1e05909c805f1a0bf73a5842075d4edbdb4270494e3832f5a5cc4',
         'son.dblosu': 'ac86290e1c70f4e668010809afd6a9a15b5429636b3ac828d809ede8b12cb9a4',
         'son.dbu': 'ff3c75aeb8ce82fe2f211cd2046e7bcacb73c47c722c2aac2cd4154064ad7e57',
         'son.dbw': '9fa1a18937d7c1d5b7b554144a908c9c7fc5ef9c406efab83dd2d5678b8babb1',
         'son.delay': '9c48f2bbaf14905d2486aaee190c90f3c20ce8d7ef3d314795e79001e8596454',
         'son.dl_nois': 'd1f7746208b259193b34c033055efa66254fead19eca26a1d2226f6d7b1b6504',
         'son.du_nois': '0f9e0ab7cd2c15cf3e50a5aa6e18a053f760bde29d3640e527bfcb80239f369c',
         'son.gainr': '96c94d0826105fe47c587fd79e8869ce5edbfbacdddab9f4f30c5fecba2ca6a3',
         'son.gaint': '96c94d0826105fe47c587fd79e8869ce5edbfbacdddab9f4f30c5fecba2ca6a3',
         'son.mdl': '00c2a75e90d2a711d81f7d160e41036a7a0fc9b8b533719d59b4f1a71ae91ddb',
         'son.mode': '93f5307e31ee89981860d0f711c0326b013856a8031e5e06467c538356bd41b0',
         'son.moder': '5d4747785a6c9bdfba475ed1f5a217249bef39a489d0fe07c9121d4aff3d96c8',
         'son.nhp': '17ebe030b16fa49c919e53e24cd03677f819e514fc7fa3f1602b5337ce4503f2',
         'son.probmp': 'd972eef9ca6ca7056b44ce2982f02d3847c2ad997d86a0c9fea8bf7e2c4737a4',
         'son.reliab': '3febc1f7870f9809e40fe7514ebd964a687b7020f94684acf3b4cc3662d2e73a',
         'son.rneff': '96c94d0826105fe47c587fd79e8869ce5edbfbacdddab9f4f30c5fecba2ca6a3',
         'son.sndb': 'e7645279aa4d0e7976bfdf81a7e71d809b96cc5e96756099b7e4395d4306307d',
         'son.snpr': '29a2d7aa4602924ccb8acb6ffdb0f45e37414841bb9d9032cda3a5381c4269ec',
         'son.snrlw': 'c160ebba3a7973233de9969fdb72842b03d632d531ae8ae0f5608d96ae4f38c4',
         'son.snrup': '3bf8933885a1d9fecc3f4d69e4e5c5a57c699225ff094f469edc68afd27e5adc',
         'son.snxx': '1340479c1fc2c27b71247ed5f29318ee74dad10b26db0e54948de5a5dfed32fc',
         'son.sprob': '366ed8443420c31e11640795e5a518cc893a28c2a745ba8d56ee104d0ec31e90',
         'son.vhigh': 'c47005da35d5e4d55260d6056c3fffded36f96fe46386adfab12b2740336ea24',
         'son.xnynois': '750cd46dc73a12a5005938b9093b5097f1bd39ee89accc952c483eca76d4e3e4',
         'status': '9f1dcbc35c350d6027f98be0f5c8b43b42ca52b7604459c0c42be3aa88913d47',
         'xluf': '18774bfced33056fc695cce04eb09c4ee02d821f2f2ea9fd21ab3b8e33f76ac2'},
 'random': {'jmode': '67a3c3b61921bf1c51cbffa38168f1c4e40ed9bfe7492693fa85c0a1c3a4fff2',
            'muf3': '9a92d8ad7a7263cf3e00ca48dbedb91367bc65636fab9072dd9378d22404da50',
            'son.angle': '5b5ec4b61d7e119578ecd1dc26eeb3d5ea4dfb9587da51c28c58cf31714bc28c',
            'son.angler': '5b0f002985e19012936d16b2564d53641755e134df2d235ac4f01b46fd25cf3f',
            'son.cprob': '6d5a6f1e802637672af84e5068ba93519b25cea1a2cbf205ffcee9a5fe22491e',
            'son.d10r': '1abf40afd3bfefb601959a5d31bab69f9904f8644f8de1136cccc9c170d1abb3',
            'son.d10s': '1ceeda5d172550e5c4c2bdba4171ca255fbb3a89d87d719dd4ea8be0e48e612d',
            'son.d50r': 'c7bc19742b61bd0d880aca63cc3f1e8ecfc4fc5ae08b33d32b6b70e370103416',
            'son.d50s': '4f9d8c2b79bc8b9480f5cee346b34e05487081b7d4f19b4a917d5004b7111cbc',
            'son.d90r': 'f73b5305e7a096ccebb8423b48ff05a5e2f98d73762ed11eb16d0f24f9586523',
            'son.d90s': '1ceeda5d172550e5c4c2bdba4171ca255fbb3a89d87d719dd4ea8be0e48e612d',
            'son.dblos': '64ed3aa46b309be2996cfa350ed5f8c2497ad9a82bac30e4cb2efbe3028ea2d2',
            'son.dblosl': '1866c61b4989b41c038d1791139bf11a171d0018d13668194c0fc7280db4e8e5',
            'son.dblosu': '1e560235de7ecd00d34a5f9d58bf66e9d2291ab5caf8ebcfd86f3fb8a9137bad',
            'son.dbu': '8f9563ace591acdcf54dadacb2755107cce814ea7e39e30e1c50dbaf9b406b0c',
            'son.dbw': 'd04c990b54cc5cc42e6d8e3465c9347a7986bb519f0641107ff2fe2fb1bc5ea8',
            'son.delay': '9e8b51e8778cc6d2a6cbe997e1e9c98399bfde8e3449e41a1839420a124afb94',
            'son.dl_nois': 'd7fcc6b6fcc1ef998e398ce0e3dec4504011a5beed39746225b08fe407adacd4',
            'son.du_nois': '18aa78687cc3448539794d48f587190bccb5592e220810fd3fdb6b810f2ad0b7',
            'son.gainr': '183facbc37f3a18e7fbdd26d4963c44885692139b16d1da7a41da17e5770f9e1',
            'son.gaint': '183facbc37f3a18e7fbdd26d4963c44885692139b16d1da7a41da17e5770f9e1',
            'son.mdl': '8edaf6ec67ce708f281d752fa248fe89845a022bbd7be93b32d1e391f91306ae',
            'son.mode': '96093ec911ce4ee95812c051fe367cb6fd387ddb397dea3a22846ffd07b94f6a',
            'son.moder': '8904f2565967be09536b6fc2054df7c426ac4f4d5fa4f87791ad611fd54ccd92',
            'son.nhp': '16aea089b3a85022a0a4321e6a5d04f31224aee8d28cb5c7a3ff99ab85f62855',
            'son.probmp': '5b5d79d2eadaf1520ca034fdb4b5dfc9d93ab7dbf25887663471b5ebb0087052',
            'son.reliab': '43eeaa87cb93931e592eafce5493d53341194a152e81332e85312bc99589200d',
            'son.rneff': '183facbc37f3a18e7fbdd26d4963c44885692139b16d1da7a41da17e5770f9e1',
            'son.sndb': '09bdf2feb0aac210b0a78155e2660ea101232b02649b7cd7806afc01d9e92145',
            'son.snpr': '2bda491a4832c9bb9035982eab0f390db1c69d39ac08691fe0ae709f6a647f6e',
            'son.snrlw': '1ad4dd2753cb097e0a22cc4ccb968d82f4347e616f67f47d0f0f4908f8707e2b',
            'son.snrup': '2903963f4b4c209b63365684beca6627bf89b6d8ce46739b58d818b34da990bf',
            'son.snxx': 'a3f23770d108e0f88a04d015685ef933fbc4e3d106dbaea8e5cada1688e25957',
            'son.sprob': '574d4d99c05933460cbbf210305c2322d0d743b86e0147d5aa1c9586c8d5a754',
            'son.vhigh': '7868b1065fa7daae8c3c8942565ef1abc490deb21df816a8bca4a23794af2b04',
            'son.xnynois': 'd7640290469016f0e7772d28f9466608bb105780869d641c498c6df64cf6852b',
            'status': 'c35020473aed1b4642cd726cad727b63fff2824ad68cedd7ffb73c7cbd890479',
            'xluf': '4e4440325b0a20383510cfd4c0ace4da561b832201638d10f87627156b58e8e2'},
 'refusals': {'jmode': '4e5794bf19b446a8f988218f2e3805e88ade26ebbcc8f6611b445cec9a92e4e2',
              'muf3': '123ac0319d9ffa660aed03a483c6968de615a0c58473d1a2d17765d5f3cee86f',
              'son.angle': 'f0d431ccb03a841cfe2e9e5025b3714eb0747bc77b2a3319429a936c6371a556',
              'son.angler': '1ad2e9ce1e386dbaa1ebec16d18b022b43c05dd37b71013848b016c7cb68f0e6',
              'son.cprob': '4202014ff0c90444a5bbd65087fd9bd8d6b64145cbd4e873ec1164b939fbde6f',
              'son.d10r': '9e2a0f8c76a5a45fc798025283597336ab2710dee18119320c7596a5ab50a94e',
              'son.d10s': 'ea00a23ec25cfe84ba11297813751078e55b43265c139b14a53e586417e8fe76',
              'son.d50r': '845a6b1177079fc68db041387996b87e16457bc2afbdd2eebd1aa6ddb292bcee',
              'son.d50s': '8ddb9a9c1780147c3023422df45d495f1a5282d4a18dc5959897ce0d04c56ce9',
              'son.d90r': 'dda4232861343601b86e10772a60c6a9719d01a2f1027d9856a912faf4adac01',
              'son.d90s': 'ea00a23ec25cfe84ba11297813751078e55b43265c139b14a53e586417e8fe76',
              'son.dblos': '44217e54aadcaa18ed88fc916e9751ed476feae082c4324264ab988e37e4b711',
              'son.dblosl': '572b0dc69bbbafcab344e2728219b29a04d5ea226ec03a7a0aae203ca3db0f9b',
              'son.dblosu': '4f00c4b4874d5f609569647e48a26ff6dc6a2c2e8ccbf1ba44e811bab991d1b5',
              'son.dbu': '441e154f9cf9386afeb72269c2bdb05c0af04bdf8cdfe6cca96a1ee423acb9d8',
              'son.dbw': 'f35a304afa016af4356e1242fc2d400bb8b300b90808c0b94c578ebd13e04a5f',
              'son.delay': '1c18556a9f02fb0295bd74a685df2805852deb6c7b2a239f13b1baed500cc67a',
              'son.dl_nois': 'bacaa54a631687c51babea7474c5a64b78e530634c8fd85ff919cad0b16bec03',
              'son.du_nois': '04b649a9078900bc5258bcb51b5ecac21a491dad1ca8b7bb2382e53ab2c9e485',
              'son.gainr': '2cbbeef1249170a43854962fa5b19fba628470c70beb9ce23e15a0f05cb891f2',
              'son.gaint': '2cbbeef1249170a43854962fa5b19fba628470c70beb9ce23e15a0f05cb891f2',
              'son.mdl': '8ccb8c3c9911a1b0e1d48ed5dc784791656702e34a304d442233bf1d2cc6b8d5',
              'son.mode': 'aab299b6aadd50def05cb5443d22cd270f99c1be7b3c9d1cf58a7eaaf6e191fe',
              'son.moder': 'c573c55f74c843064a2fd616202a6f8127f36fa4bc6e7cba275260f654df14b7',
              'son.nhp': '245ca33aed54ae3a625ac400cbb5cd2b7f8832b461c72436e8a808477cced8a6',
              'son.probmp': 'faa6b50d7ee83356ffd9b52316835d1ce3fad25d19387c3f54ee2a4f14925649',
              'son.reliab': '979148231cb7d69d699b21d7729d258fcb67869a1d80e0427aff0cfd153e7fd6',
              'son.rneff': '2cbbeef1249170a43854962fa5b19fba628470c70beb9ce23e15a0f05cb891f2',
              'son.sndb': '333339924ab341d616a1671593e46bbdaedc16552b8f0b54b2635dde22af9393',
              'son.snpr': '934e5aceffb7323a0a98e1ccf33dd312ade918911c53582f53d2dcf645002ed6',
              'son.snrlw': '28e1b69feb2c26fdb649309a91652ffc3c014772a8d9c083afa79a8c1200d4b0',
              'son.snrup': '3449e86ce3fc2c3bb3d79a18d006050e7871a945ec817742908337ea5a2d1036',
              'son.snxx': 'f38ececf46532840c0b80a6d80a3c5d26114f57b8a17c8a63641608a28ddae8d',
              'son.sprob': 'ad5b25faf1d2b9279eeaf6d5681fe55ad480d39b1c92bd2ad058956405f9a5dd',
              'son.vhigh': '77cab2575cb32a4857385cbac98824c9d238e6f4c619635dc20a58f8cde534b7',
              'son.xnynois': 'de286fc7fdbc40d6ed5cf5568c11a67f4482c488c8f944356b7f0985c9c42b1c',
              'status': 'eb84ed92c71fac21acd2d8673e480d57fd4884267f92f837f05ba55ee64beab4',
              'xluf': 'd0766c246f445a5a774ed158f47a10ebe2f83b56e9f8636ec655d11e93a66cfc'},
 'world': {'jmode': '0590eb879d9132e635e08b700219611764d6481b9bb186252814691f3ab5ab9a',
           'muf3': '70cfa0eb6322b9c605802f1156410361137daedcd0ab494ebd5846bbf292a6b1',
           'son.angle': '06768a2e8b6ea6f80f5ba5e43bae66a0b25649d3d936d6ba28b6f85e666763d8',
           'son.angler': '683fd5dcf97b498aaf2e048f59b2d980d2d9ce8dfa39b44c8377d03322dacfb3',
           'son.cprob': '80a4aa4cea76243e9c6df81f46aecc40855702365fbc3ed6bce80651ae41b4b3',
           'son.d10r': '972638863be7f6f260b0c3384cd213a8b045c6cf980bcdc5e91810e449a795fd',
           'son.d10s': '26e26b215c3f6b8e94879c761d1bb9ee9e6b438d6ca1b016e79215c99732c16a',
           'son.d50r': 'c451ccaa9a12e4d43c4393ad01130c6443d02e74ab37fd48b8247bc6d5b4dc4f',
           'son.d50s': '58e109c2b69c3caabd96ede8007d07feb77fbadc8f9cc225aaa52cbfd5867d1a',
           'son.d90r': 'e32fc4fb5e7ef58b7c148941e3b68e9fd3bc4c9f3f1a1aa49accf00f40bc8dfb',
           'son.d90s': '26e26b215c3f6b8e94879c761d1bb9ee9e6b438d6ca1b016e79215c99732c16a',
           'son.dblos': '98d09a11f985512e6334e2472b964fab7e461f0c2185518216c47ce98ae26594',
           'son.dblosl': '022be57f2244b9b773347d1b25beb8e6514e5aed1c2cd349987cd6b114de6e91',
           'son.dblosu': '56325064abec6fb6f43a10d0c9d4a7f09a901f48d0c94c5303cc25abe1d23ec7',
           'son.dbu': 'a64fa5d9edfa9f3a4e04e4e728c9c67e07613f74f7f1a10e8647e0647e5d6881',
           'son.dbw': '1c263d91b9b159d9df1aa6b2f61015ddac631c585f118a732b3f79d88300bd65',
           'son.delay': '36c1440db967f82d9b98a7d8eaa4acb6a7687539700808a89851de09dc843db6',
           'son.dl_nois': '3aeb86e9211a981e3ccac4c88a0d630f98b74adfa955368fc1cc19212c1a270b',
           'son.du_nois': 'f21944942758024636cd813db1691d147f1b4891445601fe12fe5cd18e1db61d',
           'son.gainr': 'a1cf17cc48edf34ef12135db9df48c7c4ad18e6fec8fe5d4052468f3a72187d4',
           'son.gaint': 'a1cf17cc48edf34ef12135db9df48c7c4ad18e6fec8fe5d4052468f3a72187d4',
           'son.mdl': 'f56f817e4a5fae47d4f7469af65f6374b5b81fa8f795ddd19616a56edf676cbc',
           'son.mode': '7eed04a4fca1e70b76cfa7d36d9ffb479de1579b39cf8695da9771ebe728cf88',
           'son.moder': 'd2ab4a2adfe5128e17b5453efb510b5a3d9c36d64a369f130712f8daf5f757cf',
           'son.nhp': 'b17acb6dc3974cc8f9c1216276177e6be5836bbd56a3786294987f124d95852c',
           'son.probmp': '55ecc0b226cea87fe20d6c0b13477450b21be82c33577464a32c75bed7095d15',
           'son.reliab': '6ac446d5f8ba0ddfd482102568b44c6cf9203e8929e466119a36bdb2b557afb4',
           'son.rneff': 'a1cf17cc48edf34ef12135db9df48c7c4ad18e6fec8fe5d4052468f3a72187d4',
           'son.sndb': '3424d157aad92c6470fb3b916f647ab6e9c8fcc6ec73f7cf34a4612b8c70b686',
           'son.snpr': 'f56d5285bd9f96cf1fa69f6e3e899bbbbc88ad8b654f3b7c04bc4c44ee6d9429',
           'son.snrlw': 'e9806eedc7975a59c2e48763fd7e26af53cbe435b891506967aa731834ee9bab',
           'son.snrup': '0499dd59ae84badb927478d94b0d3dca418d7b97f3308b0fdab9bac72724ea19',
           'son.snxx': 'd4317a88d9459119efd48d8d3adbd2882e004a96a4757d12dcb9aa9a27797983',
           'son.sprob': '47d3cfe7ecb32a0d526a5952db7726fc8ad54b2f1204a5f15b4f91bdc4c85176',
           'son.vhigh': '7c78052fdd713fb891cecc13c1a371437127c1d16c765fa3076df855f3628e18',
           'son.xnynois': '70a6d90d3de923c6637c7c605974a9cc7e464290fe6ec43116ea313aa5990b43',
           'status': 'e249ce5672237e4202ecabc1b43b9eb17bfface778c4fc8ec71a2070855febc8',
           'xluf': '7b83fe16cc005564d06696bddc56e43a656f2acf0c501cfc206cbbab2204c902'}}
FAST_DIGESTS = {'ant13': {'jmode': 'ce4dd332c63f06cb9963b618bfe1d1fdfa001d14af5125ae19edb66c168fc6eb',
           'muf3': '9ec444ed459975f878c4e8cdb7b1c0dece7953fd2364a63a155a54b4d10add92',
           'son.angle': '76a1fcf596a0556f469c36be88acb6ffa3278c7ed3233159a6a1bbe1383222b4',
           'son.angler': '725fc968f730f0119612727923bde7d28b756bbbfc88d6908043b224b290ded5',
           'son.cprob': 'f74e29097b72b8882c173139a012a8d16d3631ff1cfcfa6056854cfb4f05e654',
           'son.d10r': '768ac3f38362d94bb7c14b3bcd5c36c2ed0d126ba372f841082d5744e3c07be0',
           'son.d10s': '69db357fb3629f804d868c1c5913556f47041e1d862d3e9cd984a49e05c4d23a',
           'son.d50r': '53e27d95d7df4c28dc5254a8ba38190cc6de69d15fcff754e9f559ec5efde043',
           'son.d50s': 'c6b4ac47185ddfa0b93e250d37c51a50e9175ce37087d67970480e9c2054d225',
           'son.d90r': 'b5eca2f77d57439a689961a169cfdd7bdbbbf880a6b269ddfef9dc3f32793f9f',
           'son.d90s': '69db357fb3629f804d868c1c5913556f47041e1d862d3e9cd984a49e05c4d23a',
           'son.dblos': 'd548537f8995f9f0a5441fec6529c4c2b6ae4e5b65c92837c751661baee932cf',
           'son.dblosl': '080d45b2d7585162637bec47310a96228c54f5bc644aaeccf2394e7eb631d38f',
           'son.dblosu': 'fca1fcaba988c8e10900f7dcaa29ae8bc035c5f1fd7bc2ee67803bb6a2a460de',
           'son.dbu': '138c5753c8e36ddaa4fa956a82663a32f0b5de79e7df9095e72e60e51caebcb3',
           'son.dbw': 'c198291e9cd6cc627be84f94be099511d265da4c43764d6ede1e85bf12198126',
           'son.delay': 'bac28f209d870c50ea4427563513b7ac83f2adc8a2c5b05bede5d8cb93daadc5',
           'son.dl_nois': '21ca4f74b684a8e179826f434f673dfc3297bdba28e157c465959442695f12da',
           'son.du_nois': '71dd7700a4344f1a633955790414e7381b23d7c59d5631c9b5fe540788824399',
           'son.gainr': '8705cece3d2295b14d5456550eb4c1f7ac5d8cbdc9afb9f25e877a9894ac71fa',
           'son.gaint': 'a0a2eedfb4db9b76a840bbde097b1a57d254b14c426b1e6a661d30050c225d3e',
           'son.mdl': 'b547957c34e728bff027e2cd8f1044dff636e28b90efdf929808231090e02125',
           'son.mode': '5b1d0bcfc6f2c25d903e3a8180d83346f493e76cbc07470cb56ba9a4a19cf738',
           'son.moder': '0cc587bf9abe7d0628365a25f377c63b28173ecd2502719d723e87aa020a10c6',
           'son.nhp': '53bb93ae834fee62a9368b20369336b27ce1fe9c4a34904a077ffd55490c5c1b',
           'son.probmp': 'c0fcded156baa8de06b4a09bf11d50d6b9db3df038b552a15ab8466db1ecdc6f',
           'son.reliab': 'dcc6c5197336d458bde297d86b916284f73c91372d4f230e33cea767c2c0426f',
           'son.rneff': '59f5754f640e5f758cc965dc48f180a5cd57d408c7abb327554f5129c0d38e66',
           'son.sndb': '6fbb57065624a670398f2ee538e46f642e37d0df893213eff0ae8735ba3142d3',
           'son.snpr': '492b9a54d5bfed7af3a6bb24a1d9d7bfb6c4d178276add2fd69edb0bd2192f97',
           'son.snrlw': 'f454f7e38ab628993a135287c4a94e9e20745a6071e529c8df3202f8aa7cad4e',
           'son.snrup': '006dd7200a78a89586eb312cc818cfd9a1fbf6aaac296681c04111e66941d80e',
           'son.snxx': '94da27454fa9ef3a639ca5d8b64a03fb759707977b2451c910680eefde54c128',
           'son.sprob': '84b0cf0d17261b8436fd4ed9070ba62663eee4ae6504776b1dfd1a2b07a8e68b',
           'son.vhigh': '537d87900f14c18ec3d19d431cf5b78c3a8d6f77c421c2a556b19dd80cca251a',
           'son.xnynois': '7ebe96c726985f3aca8b07b40ae34c4420bc76543fc78a0182c18ce1b0f2f0ff',
           'status': 'fd9243e1ba57263ed469c3bdbd7ade6ec5254e7ed924a9f5737fa44749933cc0',
           'xluf': '016e948dea7be3c1c43860fcc59f657c269bb9c41123dc745c8a92fe6fdf0b83'},
 'icepac': {'jmode': '63f49cae896ebb66f7a781e25e2c5eaceb7f632198ac62474a7260e7be2bc0d8',
            'muf3': '99ad446f6f1b6ce689ff0359f4f3b5cd5058a818b4a2a4fb0f035048c8f26849',
            'son.angle': '2e9cad7aa36030a2f5186e04c49472912a4df2a7de2902849db73ed6b0072a52',
            'son.angler': '1c0031553e058dcea173fe441603af47e1a81919ba7d4d7be1327f48af33f520',
            'son.cprob': 'bb9ac3ec23e83746a0c2b3273a1f83d27eed8ca237a0dba421f367ba0e080455',
            'son.d10r': '4fad7edcbafaf1c0b4de648d76a69b7c2325f12fe2112533e225888292daa9dd',
            'son.d10s': '53ad37732587324cb50a43b7caa1cad51837b61e9c818b9770c1049c26a916e6',
            'son.d50r': '71bc8c8a7b95243bc80637c58fc0bd3f1ae328b0ca2fd938e8d78ac2608f7fc2',
            'son.d50s': '597f97cf48a2e76b0030fd1975e822cd582381d3ff05c2db61b1414df614d92f',
            'son.d90r': '80a06264267491dd2ef5c1163b5afc4ab84d08865605d657c3dde88f7891a5ee',
            'son.d90s': '53ad37732587324cb50a43b7caa1cad51837b61e9c818b9770c1049c26a916e6',
            'son.dblos': '931f8c9cd00ffab37b733c54bdc15dc2e3f86756638f3ce00a52ec81ce34071d',
            'son.dblosl': 'fba9ecb4d26433ce5dfd3f94438f138f4b587939ec6455c3df6ff0a227222920',
            'son.dblosu': '36db59dbd88032c5c5c99b7531975af4990fda4a53bde99c4a4b6bf2a87e7b86',
            'son.dbu': '1bd841ae2f2e8f7b232897a0719b5715c4c38a7500f579802c5fa576c5873944',
            'son.dbw': '0884455aedcd078bc1de0f754ab549df21c9e5b6eb30324897d49f51e8303b6b',
            'son.delay': '4f77e7185f0cfa06da1f6eea8cb67f59c2a684ea2efb6da45af1a12730cea54d',
            'son.dl_nois': '943b5bf2bf4d8bae122a3e14ff6ae0fd7982bb3ae850739f223d5d910a9b578c',
            'son.du_nois': '0ead2e0b6398c604ca9799159fdd145463936c82594744bdfee15710c6ea8ff4',
            'son.gainr': 'd2a23a7de3af6bf70d164a0ce45201371aca7624e30079d30cccab1cb6cf52e3',
            'son.gaint': 'd2a23a7de3af6bf70d164a0ce45201371aca7624e30079d30cccab1cb6cf52e3',
            'son.mdl': '8330b25f96b379410fb2d553db01ba0f4b77ff938fc9a611e62a309946565269',
            'son.mode': '4724be65d71ac61d982de79611b03055b997bc873aadd635982bed43afc1eb64',
            'son.moder': 'a17204576eda9fca9172239b8e8cc1948c1fb9c98c33060304b238aac233e8b0',
            'son.nhp': '195e5666293a2a398d6f31a7061ea804018413f21f0cc8c7fc3b4f9fa1522632',
            'son.probmp': '8710c8662d3fb1d31604e4360adb7beac80054e916785dba3cec1098a59f6ae2',
            'son.reliab': '3ef625054f1cd153e58c32e2d41cd7bfd359cb8afcdc3b0416ad5155e2d29b23',
            'son.rneff': 'd2a23a7de3af6bf70d164a0ce45201371aca7624e30079d30cccab1cb6cf52e3',
            'son.sndb': '5fde137e5d802f3072a0053e0e3e640e6a4abeddc831ce94ba866cf3124ed691',
            'son.snpr': 'e4d1536991150d7ba0ad378b890503ea1e46dc158d1b30842dff2db5074a96bb',
            'son.snrlw': '5af59637f32cef606559047e8a44ce2cb730ff4e56ecdb19857ed40333539bdf',
            'son.snrup': 'a3a6957f4f8f94f34c1291ada92a943fa670eedd250522ad70a1e35e4f1af419',
            'son.snxx': 'f18ae44a64ca0fc7f8271a459440ee27624c0d92df2b047d22876b483db605d8',
            'son.sprob': '08316304c777667aa67d30e89c27e4dacce8a03664644f1d801e6d4073812f98',
            'son.vhigh': '3f0014c4c64a50744a0410a49c220ef7604553d44d0078ea4c6abcda3d0f08cf',
            'son.xnynois': '86ac142a230bdd76c7a5f30783045d0ea1cbe8a47c9fe1233c5ed0cb9f027260',
            'status': '4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe',
            'xluf': '92fe71732db478fc8306259b99670d2e6bffd0a932e9d5eb75ae16a555ef2a2f'},
 'random': {'jmode': '67a3c3b61921bf1c51cbffa38168f1c4e40ed9bfe7492693fa85c0a1c3a4fff2',
            'muf3': '11ee70d8de889d92b0746a647acf6ce917240dc9da3a051cf96a191aa6889349',
            'son.angle': '855f4e48c38a3442ca9c49c332b4e7c2f0c0a58a0ab8c7a3d98591546c6bf16f',
            'son.angler': 'ae76adb1e222f369b1c1cf1f709d784fcd3bf4907002b96def593c5e499ef986',
            'son.cprob': '36af435af2b66ac19bd237a4764d9caf731046feddd9f9d7f0f56e71b808ee80',
            'son.d10r': '0cfa6ef0f38a9ca612ad2b8155073d6ae5deb7dabdec7f2920345c28cacdfe84',
            'son.d10s': 'a3d465942a9a2fea706fe51ca15d0125773160847d08051709c79e005240a03b',
            'son.d50r': '2e417de2ed50e1f44fcd0ed93b081d9570d113db85ebd6f17ac80e041053621d',
            'son.d50s': 'bf0b960da9b98297cf5f494a572952effb4d9955a0c09b4d8abcbe7e7c603f73',
            'son.d90r': '1fee1c3837e7aa959181a4724cc957a833267807a41231531f75221cd9a889f0',
            'son.d90s': 'a3d465942a9a2fea706fe51ca15d0125773160847d08051709c79e005240a03b',
            'son.dblos': '8615fd2654a40960892adcd5d25263ec2d45a662882cc3ff865c7dbebf0a35c8',
            'son.dblosl': 'da0961fbf7cb21ed9142dc5b1c0aee10f1e2a50a69d898d69be0d98a087e5bdb',
            'son.dblosu': '358ce06b3a0e4a30f003b722b14082aa833041fbd198a4b81970283fcccc57e3',
            'son.dbu': 'b35c83663d8303b6da8f00e26f595575fccb4a815b657f37f423faade2e6c3ca',
            'son.dbw': '4b2c5b9d9274f325d7bfd3fd538ad5a6874bab42bf96daa50db4ef747fbca2f7',
            'son.delay': '6d76d0e54860c653aa85f3a71e13c3d439d88cb0a63ac24157313b9a6b6c6e9b',
            'son.dl_nois': '35bf0bccbeae38cfdcbf01eec27f11295858998e4c0253e66a32acd3f630cc90',
            'son.du_nois': '8d5b9a96e50f993520771107abccfc2ab3f1ba080b1a6a7d5757eaa885eaeedf',
            'son.gainr': '183facbc37f3a18e7fbdd26d4963c44885692139b16d1da7a41da17e5770f9e1',
            'son.gaint': '183facbc37f3a18e7fbdd26d4963c44885692139b16d1da7a41da17e5770f9e1',
            'son.mdl': '8edaf6ec67ce708f281d752fa248fe89845a022bbd7be93b32d1e391f91306ae',
            'son.mode': '96093ec911ce4ee95812c051fe367cb6fd387ddb397dea3a22846ffd07b94f6a',
            'son.moder': '8904f2565967be09536b6fc2054df7c426ac4f4d5fa4f87791ad611fd54ccd92',
            'son.nhp': '9b55ce25c7bc301d802d9d1f6b46d48d1dbc157d450ba6391dce2720d024f32a',
            'son.probmp': '07baadcdfd00287e914adc3f0794d0edcb8533dfc1095c9fbe6e9c0ce852f36d',
            'son.reliab': '80ba468d202e3d266977b0d212946492700506953a3da074a5ef2c18c25ce617',
            'son.rneff': '183facbc37f3a18e7fbdd26d4963c44885692139b16d1da7a41da17e5770f9e1',
            'son.sndb': 'f5c8fc38614978b5a34acdc802012770aafc88ce90aad98d3b7e0e40fe29ff35',
            'son.snpr': '309c49561b346cc192e2affcb2f0aaacab93aed317ef4a90a9e904e59812ab15',
            'son.snrlw': '2a7b25f432cc01014093e4f5f9bf086bb45d249ef2ae0196f51d0aff54dce33a',
            'son.snrup': '5e06ef2bd7649c7aaeef5872db5c9c26eeddf02b61baef46fdd3899359535bc2',
            'son.snxx': '143bcd4fbde8f8219096308280feed0bd0dd7500c095bfddbb9c0689348a8546',
            'son.sprob': 'cc7c522699fdb25365795c906567f9ea5bf14d4c4a383f198c4335e89142a5cd',
            'son.vhigh': 'c92f17d3e074937b3b16200c3c0c59fd2d3f506df3205a9d1ea55a078f3eeaf4',
            'son.xnynois': '76f2bd81dfa6f603c96267dec33099a8a966372f1f99a1514de2548a0ee43ce2',
            'status': 'c35020473aed1b4642cd726cad727b63fff2824ad68cedd7ffb73c7cbd890479',
            'xluf': '4e4440325b0a20383510cfd4c0ace4da561b832201638d10f87627156b58e8e2'},
 'refusals': {'jmode': '4e5794bf19b446a8f988218f2e3805e88ade26ebbcc8f6611b445cec9a92e4e2',
              'muf3': '3f1ce85c120122c9b0d4ccb02cbaac5b713ed3f2487317848b18eb6206316ccf',
              'son.angle': '6fe312c6b0b928da149cb0128433834c5e128a1e6503c08e4db62a996d9ef0f6',
              'son.angler': '043943ecb24c42cfdbe99b8e93d66c18d60b9cfbd869e153b2a4551becd17f18',
              'son.cprob': 'a74373b812b0d12cf3e1c2837c515df340dd4b55c91fee2ea2dde4787b9f13ff',
              'son.d10r': 'dd3ec4142a87f4bae1c899ef519a001225ce580e26e2f17301cc415eb55c1fe8',
              'son.d10s': 'ac9fbedf1bfab417d1be4c0cac6fd9e898c447e967dfd463fd0bef4344f32433',
              'son.d50r': '9f3ba3dc4356c392f1d7e540acfa448ef2c5c3fba8555eea82a485e5f07760c7',
              'son.d50s': '31a62a34eb15883b708c852eb418eea152290356c7a19c290b59cde7f9397eba',
              'son.d90r': 'be5fc018619a0768f920b189c4ea5530892fcfe940e1ba2928bfde96fd8b3c67',
              'son.d90s': 'ac9fbedf1bfab417d1be4c0cac6fd9e898c447e967dfd463fd0bef4344f32433',
              'son.dblos': 'faef4cbe1c2496b6f5132187003adb0d4e7382db9fa385f6d48fc8ca08d7522a',
              'son.dblosl': 'eff9deefa15b540760218cb15688cf42bcc8e39ed33289db87443057da69d957',
              'son.dblosu': '3472a5b60eb4b1ee0303d152cb88931368da79395a0ec93d72e5c67aaaf949d6',
              'son.dbu': '7f31fda2fdbd8005d723cb47d9bda561e442b716275d20b12742f02211ca4914',
              'son.dbw': 'bed313213a50f158586f2263c1814459251d1fc3f3c1b48f0124d69a42069fc1',
              'son.delay': '8810152c9e3464215458ed9690c5add5626e5af3519892d73879f0a3edfdd451',
              'son.dl_nois': 'eab2c289a249fb201d0a343f0c9a0ec57e26bb99309283a1e92d782e02fc845f',
              'son.du_nois': '6f2b1e86bebbc83d8480e3a8a7adb088306932ecb749b0abde20b6d197bd3ad8',
              'son.gainr': '2cbbeef1249170a43854962fa5b19fba628470c70beb9ce23e15a0f05cb891f2',
              'son.gaint': '2cbbeef1249170a43854962fa5b19fba628470c70beb9ce23e15a0f05cb891f2',
              'son.mdl': '8ccb8c3c9911a1b0e1d48ed5dc784791656702e34a304d442233bf1d2cc6b8d5',
              'son.mode': 'aab299b6aadd50def05cb5443d22cd270f99c1be7b3c9d1cf58a7eaaf6e191fe',
              'son.moder': 'c573c55f74c843064a2fd616202a6f8127f36fa4bc6e7cba275260f654df14b7',
              'son.nhp': '19ce2b755d6576f3b87656b4d0e7daf35986c97fdd1332433b2a81004723918f',
              'son.probmp': '63ae28e5f44da6da64b49967ad0c65e367b38c8031b4bd1c686a8e83d48c2694',
              'son.reliab': '576fc5759e4c28b7636f64528263fab0d62e7a4eb242d8853eb87a2cc0468591',
              'son.rneff': '2cbbeef1249170a43854962fa5b19fba628470c70beb9ce23e15a0f05cb891f2',
              'son.sndb': '50ae6ad1d01214069dc8ce9f35adab8ce983e706acaa4f1b2ccd3fa217942c44',
              'son.snpr': '05badfb3ce513866e79313cd0d8adc6bc8f509fe6d5af2797c566b7ccf9e5b02',
              'son.snrlw': 'e7d3377178567215dce42b6baccff384e562692591ed5029fcc050d21a558601',
              'son.snrup': '1c8334e2b5b2eb97559ed194f37f49c8f01359d2e26d9ff2ca1d72882b471f6f',
              'son.snxx': '995cd8ba17d40bfdd05a13c0adf0ce9332d9458eb14b56e697fc7720cae04e87',
              'son.sprob': '7cb6788994c15ea8eeec2116bf4c15121c012c17d29b5abd68f1a68b3496985a',
              'son.vhigh': '97f6de1d356c71f7d1ebb7f7261113a97a4498622d3d1c9de7b5e40e24f89e23',
              'son.xnynois': '318bdfeb84428ac7bf89dffa33fd0f2b9bcd2e47f11e50589a34888c6ff8b13b',
              'status': 'eb84ed92c71fac21acd2d8673e480d57fd4884267f92f837f05ba55ee64beab4',
              'xluf': 'd0766c246f445a5a774ed158f47a10ebe2f83b56e9f8636ec655d11e93a66cfc'},
 'world': {'jmode': '0590eb879d9132e635e08b700219611764d6481b9bb186252814691f3ab5ab9a',
           'muf3': '12a1d635e8205a1f55c922e9210a394796b65fae55401c53bbc697f845df9bad',
           'son.angle': 'cfaab96a99563e8c5bf433f8a695b7496d1c98d669ac3a23780f1286b3b55186',
           'son.angler': '7b3a3c5606d8d04ac6eb3fcd9d16fe6ee7e8e6cab81db6ef53fa0ca165373d0c',
           'son.cprob': 'ff253bfe65e9008fb9083582c0d5c9375e0467410cf8d7f7f4925291adf5eefe',
           'son.d10r': '66ca4eed04641cb8b8204859dba8de0666c90a1e310ace4aa25d493393889ada',
           'son.d10s': '5c9efd59974878c182eeec343b4ab6b826a26997bd684db6355addc21501d9f7',
           'son.d50r': '8e4ac904ca10abde15ad9382aa8cd02d9c4271299f2ddc4bd3097cfd2720ee11',
           'son.d50s': '28e29787baffba073befbe90916eac1585cc05b203914333d03f5de002a53f26',
           'son.d90r': '087c4a0fea32b00b00708227fc35e2a7324b904cc22956c887f36c5e9da42a4b',
           'son.d90s': '5c9efd59974878c182eeec343b4ab6b826a26997bd684db6355addc21501d9f7',
           'son.dblos': '2f30ec46ca02b91ee5df298ccfcbed01fd58b294730b571c0add5406dfaca9d8',
           'son.dblosl': '03d11b3f4992adea896cceddad20c3983827e3e3b199882d4f081837774768a1',
           'son.dblosu': 'c99b99f400fe10edb8ebd62501ca5eab6b4f26ac658e79adb7eada806f6dfd14',
           'son.dbu': '857b9a0135cabd50bbdd240c8f64ad0ee8bb0cba2f1bce9d7111a1e3a1391a55',
           'son.dbw': 'ae71e7e52490545bad70906209cb458f3ed48106fdc9415704ec59f85778b01a',
           'son.delay': '35173dc644b78d76275b321a0e6d7c37a5a60c4bedc1b695c5d8433a84c79710',
           'son.dl_nois': 'abfbcb04b5feb9f9d218db155f926dd36229564e9d26eb18d1b0f18c19661401',
           'son.du_nois': '77c6546093452732f3aa449c41e7692137a3469345cfb3f93ba194034ad73135',
           'son.gainr': 'a1cf17cc48edf34ef12135db9df48c7c4ad18e6fec8fe5d4052468f3a72187d4',
           'son.gaint': 'a1cf17cc48edf34ef12135db9df48c7c4ad18e6fec8fe5d4052468f3a72187d4',
           'son.mdl': 'f56f817e4a5fae47d4f7469af65f6374b5b81fa8f795ddd19616a56edf676cbc',
           'son.mode': '7eed04a4fca1e70b76cfa7d36d9ffb479de1579b39cf8695da9771ebe728cf88',
           'son.moder': 'd2ab4a2adfe5128e17b5453efb510b5a3d9c36d64a369f130712f8daf5f757cf',
           'son.nhp': '0bcf0263e072784e56a026f1c62dcbe6881c70a0b42407fd4b2f7ce8280d7527',
           'son.probmp': '58972d3e9189debbea195473af65587fd5f852096ab8d7a5b5fc39db21696859',
           'son.reliab': 'fe1b02e1aeb469c7a16d05745979f840133f42626c7b1e4cf9848309a9d8ece7',
           'son.rneff': 'a1cf17cc48edf34ef12135db9df48c7c4ad18e6fec8fe5d4052468f3a72187d4',
           'son.sndb': '751669b0734774aeaf6f0943f908eeaa003d49af8720a9326179fee73d83cee9',
           'son.snpr': 'bbe96fd4f8000a5ff4133364d3139744b41e0d99525b465b236b2952f92e4df1',
           'son.snrlw': '8c72ee62891b5b32b78c02b69a8a8c1aed867948d9f66f14245cd5002b8d6920',
           'son.snrup': '7c7e43ad8004c39b69cc2564a6542e7ef6ec023d40f948d3ad5d988616ccd4b6',
           'son.snxx': '4fb75b34515999cd8cb200bd21b7e441576acea9cb443ecccc7f589ae3fa25f2',
           'son.sprob': '82a37ddee87e90beec85a3f113cfeb358b6c43f6f72e6c8c49fd04400a8d6ded',
           'son.vhigh': 'bba394cb4b2f4a9fc3281f14e9880ebf5b7344147ec959a4be24c2169c59f3f3',
           'son.xnynois': '611b01d9e78070f091676eec12b49dc5c9c804db852210a2b5e8488f1a4907f7',
           'status': 'e249ce5672237e4202ecabc1b43b9eb17bfface778c4fc8ec71a2070855febc8',
           'xluf': '7b83fe16cc005564d06696bddc56e43a656f2acf0c501cfc206cbbab2204c902'}}
# END GENERATED DIGESTS


def _check(name, stored, fast_math):
    if name not in stored:
        pytest.skip(f"no stored digest for {name!r}; run make_gate.py")
    got = run_workload(name, fast_math)
    want = stored[name]
    bad = sorted(k for k in want if got.get(k) != want[k])
    missing = sorted(set(got) - set(want))
    assert not bad and not missing, (
        f"{name} (fast_math={fast_math}): digests moved on {bad}"
        + (f"; new columns {missing}" if missing else ""))


@pytest.mark.parametrize("name", list(WORKLOADS))
def test_gate_strict(name):
    _check(name, DIGESTS, False)


@pytest.mark.parametrize("name", ["world", "random", "icepac", "ant13",
                                  "refusals"])
def test_gate_fast_math(name):
    _check(name, FAST_DIGESTS, True)
