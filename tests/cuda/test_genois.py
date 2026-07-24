"""GENOIS Stage C vs Stage A, on the genois oracle grid.

Atmospheric + galactic + man-made noise combined into one lognormal.  This is
the routine that mixes REAL*4 and REAL*8 at nearly every step -- the twelve
A/V accumulators, the two sigmas and Caruana's DXX are double and everything
feeding them is single -- so the bar being bit-exact is what proves the
double regions are drawn in exactly the right places.

Whole /ANOIS/, /TON/, /ZON/ and the noise-diagnostic struct are compared,
including the fields GENOIS never writes: /ZON/ EFF slot 7 belongs to ESMOD
and must survive, and MODE_N1/MODE_N3 are never assigned here.

Where Stage A raises OldNoiseModel, Stage C returns status 1 with nothing
written; that is tested separately, together with the status being 0 on every
dump case (no lane hits GAIN's refusals).
"""

import numpy as np
import pytest

from voacap_gpu.scalar.antennastate import CantennaState
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.genois import OldNoiseModel, genois
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.noisestate import AnoisState, NoiseDiagState
from voacap_gpu.scalar.readant import read_antenna
from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.scalar.signalstate import TonState
from voacap_gpu.scalar.zonstate import ZonState
from voacap_gpu.cuda.state import (pack_anois, pack_layer, pack_noisediag,
                                      pack_ton, pack_zon)

from tests.scalar.dumputil import ANTENNAS_DIR, COEFFS_DIR, load_dump
from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("genois")
SENTINEL = F32(-8888.0)
NA = 3                                  # numants + 1, the /cantenna/ extent

_SRC = r'''
#include "genois.cuh"

extern "C" __global__ void drive(const float* fam_all, const float* dud_all,
                                 const int* mi, const float* f2d111,
                                 const LayerState* state, AnoisState* anois,
                                 TonState* ton, ZonState* zon,
                                 NoiseDiagState* diag, const float* array,
                                 const float* aeff, const int* iats,
                                 const float* xfqs_all, const float* xfqe_all,
                                 const float* offazim, const float* freq,
                                 const int* noise, const float* rlat,
                                 const int* nyear2089, int* status, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int m = mi[i];
    status[i] = genois(fam_all + m * (14 * 12), dud_all + m * (5 * 12 * 5),
                       f2d111[i], &state[i], &anois[i], &ton[i], &zon[i],
                       &diag[i], array, aeff, iats, xfqs_all + i * 3,
                       xfqe_all + i * 3, offazim, 2, freq[i], noise[i],
                       rlat[i], nyear2089[i]);
}
'''


def _cantenna(rxlo, rxhi) -> CantennaState:
    """sample.14 as /cantenna/ entry 2, the receiver, over [rxlo, rxhi]."""
    ant = read_antenna(ANTENNAS_DIR / "sample.14")
    cant = CantennaState()
    cant.numants = 2
    cant.iats[1] = 1
    cant.iats[2] = 2
    cant.xfqs[1], cant.xfqe[1] = F32(2.0), F32(30.0)
    cant.xfqs[2], cant.xfqe[2] = F32(rxlo), F32(rxhi)
    cant.offazim[1] = cant.offazim[2] = F32(0.0)
    array, aeff = ant.p2p_tables()
    cant.array[1:31, 1:92, 2] = array
    cant.aeff[1:31, 2] = aeff
    return cant


def _entry(fi3):
    """The entry state for one dump row: sentinels everywhere GENOIS writes."""
    state = LayerState()
    state.kfx = 1
    state.fi[3, 1] = F32(fi3)
    ton = TonState(gnos=SENTINEL, xnoise=SENTINEL, znoise=SENTINEL)
    zon = ZonState()
    zon.eff[:] = SENTINEL                   # slot 7 must survive untouched
    diag = NoiseDiagState(mode_n1=-11, mode_n2=-22, mode_n3=-33,
                          xn_atnos=SENTINEL, xn_galactic=SENTINEL,
                          xn_mmn=SENTINEL, method_used=-44)
    return state, ton, zon, diag


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_genois_bit_exact_vs_stage_a():
    _, inputs, _ = DUMP
    n = len(inputs)
    rows = []
    for r in inputs:
        (month, izf, freq, noise, rlat, atnu, atny,
         cc, tm, kj, jk, fi3, rxlo, rxhi) = r
        rows.append(dict(month=int(month), izf=int(izf), freq=F32(freq),
                         noise=int(noise), rlat=F32(rlat), atnu=F32(atnu),
                         atny=F32(atny), cc=F32(cc), tm=F32(tm), kj=int(kj),
                         jk=int(jk), fi3=F32(fi3), rxlo=F32(rxlo),
                         rxhi=F32(rxhi)))

    months = sorted({v["month"] for v in rows})
    maps = {m: redmap(COEFFS_DIR, "CCIR", m, F32(100.0)) for m in months}
    midx = {m: i for i, m in enumerate(months)}
    fam_all = np.ascontiguousarray(
        np.stack([np.asarray(maps[m].fam, np.float32) for m in months])).ravel()
    dud_all = np.ascontiguousarray(
        np.stack([np.asarray(maps[m].dud, np.float32) for m in months])).ravel()

    # the shared antenna pattern (the same for every case); only the receiver
    # frequency range moves, so xfqs/xfqe are per-case
    base = _cantenna(2.0, 30.0)
    array = np.ascontiguousarray(base.array[:, :, :NA]).ravel()
    aeff = np.ascontiguousarray(base.aeff[:, :NA]).ravel()
    iats = np.ascontiguousarray(base.iats[:NA]).astype(np.int32)
    offazim = np.ascontiguousarray(base.offazim[:NA]).astype(np.float32)
    xfqs = np.zeros((n, NA), np.float32)
    xfqe = np.zeros((n, NA), np.float32)

    pre_state, pre_anois, pre_ton, pre_zon, pre_diag = [], [], [], [], []
    e_anois, e_ton, e_zon, e_diag, e_state = [], [], [], [], []
    for k, v in enumerate(rows):
        st = maps[v["month"]]
        if v["izf"]:        # force the no-data branch without touching cache
            f2d = st.f2d.copy()
            f2d[0, 0, 0] = F32(-1.0)
            st = type(st)(**{**st.__dict__, "f2d": f2d})
        state, ton, zon, diag = _entry(v["fi3"])
        anois = AnoisState(atnu=v["atnu"], atny=v["atny"], cc=v["cc"],
                           tm=v["tm"], kj=v["kj"], jk=v["jk"],
                           rcnse=SENTINEL, du=SENTINEL, dl=SENTINEL,
                           sigm=SENTINEL, sygu=SENTINEL, sygl=SENTINEL)
        cant = _cantenna(v["rxlo"], v["rxhi"])
        xfqs[k] = cant.xfqs[:NA]
        xfqe[k] = cant.xfqe[:NA]
        pre_state.append(pack_layer(state))
        pre_anois.append(pack_anois(anois))
        pre_ton.append(pack_ton(ton))
        pre_zon.append(pack_zon(zon))
        pre_diag.append(pack_noisediag(diag))
        genois(st, state, anois, ton, zon, diag, cant, v["freq"], v["noise"],
               v["rlat"])
        e_state.append(pack_layer(state))
        e_anois.append(pack_anois(anois))
        e_ton.append(pack_ton(ton))
        e_zon.append(pack_zon(zon))
        e_diag.append(pack_noisediag(diag))
    exp = {"anois": np.stack(e_anois), "ton": np.stack(e_ton),
           "zon": np.stack(e_zon), "diag": np.stack(e_diag)}

    mi = np.array([midx[v["month"]] for v in rows], np.int32)
    f2d111 = np.array([F32(-1.0) if v["izf"] else maps[v["month"]].f2d[0, 0, 0]
                       for v in rows], np.float32)
    freq = np.array([v["freq"] for v in rows], np.float32)
    noise = np.array([v["noise"] for v in rows], np.int32)
    rlat = np.array([v["rlat"] for v in rows], np.float32)
    nyear2089 = np.zeros(n, np.int32)
    status = np.full(n, -1, np.int32)

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n,
                      [fam_all, dud_all, mi, f2d111,
                       np.stack(pre_state).view(np.uint8),
                       np.stack(pre_anois).view(np.uint8),
                       np.stack(pre_ton).view(np.uint8),
                       np.stack(pre_zon).view(np.uint8),
                       np.stack(pre_diag).view(np.uint8),
                       array, aeff, iats, xfqs.ravel(), xfqe.ravel(), offazim,
                       freq, noise, rlat, nyear2089, status])
    got = {"anois": fetch(dev[5]).view(exp["anois"].dtype),
           "ton": fetch(dev[6]).view(exp["ton"].dtype),
           "zon": fetch(dev[7]).view(exp["zon"].dtype),
           "diag": fetch(dev[8]).view(exp["diag"].dtype)}
    assert (fetch(dev[19]) == 0).all(), "no dump lane may refuse"
    # GENOIS only READS /RON/: the device copy must equal the entry copy,
    # and Stage A must not have moved it either
    exp_state = np.stack(pre_state)
    got_state = fetch(dev[4]).view(exp_state.dtype)
    for name in exp_state.dtype.names:
        assert_bitexact(got_state[name], exp_state[name],
                        f"state.{name} (read-only)")
        assert_bitexact(np.stack(e_state)[name], exp_state[name],
                        f"stage-a state.{name} (read-only)")

    for tag, buf in exp.items():
        for name in buf.dtype.names:
            assert_bitexact(got[tag][name], buf[name], f"{tag}.{name}")

    # the untouched fields really are untouched
    assert (exp["zon"]["eff"][:, 7] == SENTINEL).all(), "EFF(7) is ESMOD's"
    assert (exp["diag"]["mode_n1"] == -11).all()
    assert (exp["diag"]["mode_n3"] == -33).all()

    # branch coverage, exactly the arms the Stage A test pins
    seen = {
        "nodata": sum(v["izf"] != 0 for v in rows),
        "man_level": sum(v["noise"] > 0 for v in rows),
        "man_default": sum(v["noise"] == 0 for v in rows),
        "man_index": sum(v["noise"] < 0 for v in rows),
        "caruana": int((exp["diag"]["method_used"] == 1).sum()),
        "spaulding": int((exp["diag"]["method_used"] == 2).sum()),
        "no_galactic": int((exp["diag"]["mode_n2"] == 0).sum()),
        "galactic": int((exp["diag"]["mode_n2"] == 1).sum()),
        "no_ant": int((exp["zon"]["eff"][:, 1] == F32(0.0)).sum()),
        "ant": int((exp["zon"]["eff"][:, 1] != F32(0.0)).sum()),
    }
    for arm, cnt in seen.items():
        assert cnt, f"branch never reached: {arm} ({seen})"
    # ASSERTED UNREACHABLE on this grid: the AMIN1(FREQ,55.) clamp feeding
    # GENFAM.  The dump tops out well below 55 MHz, so DUME == FREQ on every
    # case and the clamp arm is not covered here (it is covered where it
    # actually lives, in test_genfam.py's frequency sweep).
    assert not any(v["freq"] > F32(55.0) for v in rows)


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_old_noise_model_returns_status_1_untouched():
    """NYEAR ' 2089' selects GENOIS_OLD; Stage A raises, Stage C refuses."""
    st = redmap(COEFFS_DIR, "CCIR", 6, F32(100.0))
    with pytest.raises(OldNoiseModel):
        genois(st, LayerState(), AnoisState(), TonState(), ZonState(),
               NoiseDiagState(), CantennaState(), F32(10.0), -2, F32(0.5),
               nyear=" 2089")

    state, ton, zon, diag = _entry(F32(4.0))
    anois = AnoisState(atnu=F32(70.0), atny=F32(68.0), cc=F32(10.0),
                       tm=F32(10.0), kj=3, jk=3)
    base = _cantenna(2.0, 30.0)
    pre = (pack_layer(state), pack_anois(anois), pack_ton(ton),
           pack_zon(zon), pack_noisediag(diag))
    n = 2
    xfqs = np.tile(base.xfqs[:NA].astype(np.float32), (n, 1))
    xfqe = np.tile(base.xfqe[:NA].astype(np.float32), (n, 1))
    status = np.full(n, -1, np.int32)

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n,
                      [np.asarray(st.fam, np.float32).ravel(),
                       np.asarray(st.dud, np.float32).ravel(),
                       np.zeros(n, np.int32),
                       np.full(n, st.f2d[0, 0, 0], np.float32),
                       np.stack([pre[0]] * n).view(np.uint8),
                       np.stack([pre[1]] * n).view(np.uint8),
                       np.stack([pre[2]] * n).view(np.uint8),
                       np.stack([pre[3]] * n).view(np.uint8),
                       np.stack([pre[4]] * n).view(np.uint8),
                       np.ascontiguousarray(base.array[:, :, :NA]).ravel(),
                       np.ascontiguousarray(base.aeff[:, :NA]).ravel(),
                       np.ascontiguousarray(base.iats[:NA]).astype(np.int32),
                       xfqs.ravel(), xfqe.ravel(),
                       np.ascontiguousarray(base.offazim[:NA]).astype(
                           np.float32),
                       np.full(n, F32(10.0), np.float32),
                       np.full(n, -2, np.int32),
                       np.full(n, F32(0.5), np.float32),
                       np.array([1, 0], np.int32), status])
    got_status = fetch(dev[19])
    assert got_status.tolist() == [1, 0]
    # the refused lane wrote nothing; the control lane did
    for j, buf in ((5, pre[1]), (6, pre[2]), (7, pre[3]), (8, pre[4])):
        out = fetch(dev[j]).view(buf.dtype)
        for name in buf.dtype.names:
            assert_bitexact(out[name][:1], np.stack([buf[name]]),
                            f"refused lane moved {name}")
    assert fetch(dev[6]).view(pre[2].dtype)["gnos"][1] != SENTINEL
