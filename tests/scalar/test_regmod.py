"""Stage A REGMOD vs the tier-B oracle dump.

REGMOD is the deepest transcendental chain ported so far: per mode it runs
COS/SIN/SQRT for the geometry, two ``**1.98`` powers, ALOG/EXP for the
collision-frequency and E-bending terms, and up to six ALOG10s through PRBMUF,
then sums seven comparably sized terms into XTLOS.  So this sits above the
[[f32-libm-noise]] floor with real cancellation on top, and the bar is a ULP
budget rather than bit-exactness.

The 262 cases inject every input directly -- REGMOD interpolates no upstream
table, so the FINDF/IFOB rule does not apply and injection buys branch control
instead: D-E against F layer selection and its collision-frequency
sub-branch, sporadic-E obscuration present and absent, the low-MUFday extra
loss, the 1e-6 probability floors, 1-hop through 8-hop paths, unused mode
slots, and a frequency outside the transmitter's range.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.antennastate import CantennaState
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.modesstate import ModesState
from voacap_gpu.scalar.mufstate import MufState
from voacap_gpu.scalar.noisestate import AnoisState
from voacap_gpu.scalar.raysstate import RaysState
from voacap_gpu.scalar.readant import read_antenna
from voacap_gpu.scalar.regmod import RegmodError, regmod
from voacap_gpu.scalar.signalstate import SigdState
from voacap_gpu.scalar.zonstate import ZonState

from .dumputil import ANTENNAS_DIR, load_dump, ulp_diff_floored

DUMP = load_dump("regmod")
SAMPLE = ANTENNAS_DIR / "sample.14"

# column order, mirroring gen_regmod_cases.py COLUMNS
_SCALARS = ["jmode", "freq", "gcdkm", "ghopkm", "km", "fi1", "gyz", "fs1",
            "fs2", "hs", "ifob13", "acav", "htloss", "xnuz", "hnu", "xve",
            "afe", "bfe", "asm", "dsl", "dsu", "rcnse"]
_QUADS = ["sigl", "sigu", "ymuf", "yfot", "yhpf", "hpmuf", "htmuf", "fvmuf"]
_SIXES = ["delmod", "hpmod", "htmod", "fvmod", "itmod", "afmod"]
_FIVES = ["sigpat", "epspat"]

COLUMNS = (_SCALARS
           + [f"{n}{i}" for n in _QUADS for i in range(1, 5)]
           + [f"{n}{i}" for n in _SIXES for i in range(1, 7)]
           + [f"{n}{i}" for n in _FIVES for i in range(1, 6)]
           + ["pwrdba", "txlo", "txhi"])

# /ZON/ fields dumped, in order; each spans 7 mode slots
FIELDS = ["abps", "adv", "b", "eff", "fldst", "fslos", "grlos", "hn", "hp",
          "nmode", "obf", "prob", "rgain", "sigpow", "sn", "tgain", "timed",
          "tllow", "tlhgh", "tloss"]

_ANT = None


def _cantenna(txlo, txhi, pwrdba) -> CantennaState:
    """The two entries dump_regmod.f installs: TX over [txlo,txhi], RX 2-30."""
    global _ANT
    if _ANT is None:
        _ANT = read_antenna(SAMPLE).p2p_tables()
    array, aeff = _ANT
    c = CantennaState()
    for iat in (1, 2):
        c.array[1:31, 1:92, iat] = array
        c.aeff[1:31, iat] = aeff
        c.offazim[iat] = F32(0.0)
    c.numants = 2
    c.iats[1], c.xfqs[1], c.xfqe[1] = 1, F32(txlo), F32(txhi)
    c.iats[2], c.xfqs[2], c.xfqe[2] = 2, F32(2.0), F32(30.0)
    c.pwrdba[1] = F32(pwrdba)
    return c


def _build(row):
    """Reconstruct every state REGMOD reads from one case row."""
    v = dict(zip(COLUMNS, (float(x) for x in row)))
    k = int(v["jmode"])
    ll = 2 * k - 1                       # LX = 1, 3, 5

    st = LayerState()
    st.km = int(v["km"])
    st.kfx = 1
    st.fi[1, k] = F32(v["fi1"])
    st.gyz[ll] = F32(v["gyz"])
    st.fs[1, k] = F32(v["fs1"])
    st.fs[2, k] = F32(v["fs2"])
    st.hs[k] = F32(v["hs"])
    for i in range(1, 6):
        st.sigpat[i] = F32(v[f"sigpat{i}"])
        st.epspat[i] = F32(v[f"epspat{i}"])

    rays = RaysState()
    rays.ifob[1, 3, k] = int(v["ifob13"])

    modes = ModesState()
    modes.ghop = F32(v["ghopkm"] / 6370.0)
    for im in range(1, 7):
        modes.delmod[im, k] = F32(v[f"delmod{im}"])
        modes.hpmod[im, k] = F32(v[f"hpmod{im}"])
        modes.htmod[im, k] = F32(v[f"htmod{im}"])
        modes.fvmod[im, k] = F32(v[f"fvmod{im}"])
        modes.itmod[im, k] = int(v[f"itmod{im}"])
        modes.afmod[im, k] = F32(v[f"afmod{im}"])

    muf = MufState()
    for i in range(1, 5):
        muf.sigl[i] = F32(v[f"sigl{i}"])
        muf.sigu[i] = F32(v[f"sigu{i}"])
        muf.ymuf[i] = F32(v[f"ymuf{i}"])
        muf.yfot[i] = F32(v[f"yfot{i}"])
        muf.yhpf[i] = F32(v[f"yhpf{i}"])
        muf.hpmuf[i] = F32(v[f"hpmuf{i}"])
        muf.htmuf[i] = F32(v[f"htmuf{i}"])
        muf.fvmuf[i] = F32(v[f"fvmuf{i}"])

    sigd = SigdState()
    for name in ("acav", "htloss", "xnuz", "hnu", "xve", "afe", "bfe", "asm",
                 "dsl", "dsu"):
        setattr(sigd, name, F32(v[name]))

    anois = AnoisState()
    anois.rcnse = F32(v["rcnse"])

    cant = _cantenna(v["txlo"], v["txhi"], v["pwrdba"])
    gcd = F32(v["gcdkm"] / 6370.0)
    return st, modes, muf, sigd, anois, rays, cant, F32(v["freq"]), k, gcd


# Fields that only copy or count, so no arithmetic can drift: the radiation
# angle and virtual height come straight from /MODES/, the layer index and hop
# count are integers, and the gains/efficiency are GAIN's own table lookup.
EXACT_FIELDS = {"b", "hp", "hn", "nmode", "eff", "rgain", "tgain"}

# Measured worst absolute error is 3.05e-5 (== 2^-15, i.e. 1-2 ULP at the
# 128-256 dB scale these quantities live on).  The floored-ULP figures run much
# higher on FLDST/PROB only because those land near zero while being built from
# ~150 dB terms, so both bars are needed: the absolute one catches a real
# numerical regression, the relative one catches a scale blow-up.
ABS_BAR = 1.0e-4
ULP_BAR = 256.0


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_regmod_dumps.sh")
def test_regmod_vs_oracle():
    header, inputs, rows = DUMP
    assert header == [f"{n}{i}" for n in FIELDS for i in range(1, 8)]
    total = exact = 0
    worst = 0.0
    worst_abs = 0.0
    worst_where = ""
    filled = 0
    for kc, (inp, exp) in enumerate(zip(inputs, rows)):
        st, modes, muf, sigd, anois, rays, cant, freq, k, gcd = _build(inp)
        zon = ZonState()
        regmod(st, modes, muf, sigd, anois, rays, cant, zon, freq, k, gcd)
        for j, name in enumerate(FIELDS):
            got_all = getattr(zon, name)
            for im in range(1, 8):
                e = np.float32(exp[j * 7 + im - 1])
                got = np.float32(got_all[im])
                d = ulp_diff_floored(got, e)
                a = abs(float(got) - float(e))
                if d > worst:
                    worst = d
                    worst_where = (f"case {kc} {name}[{im}]: {got!r} vs {e!r}")
                worst_abs = max(worst_abs, a)
                where = f"case {kc} {name}[{im}]: got {got!r} expected {e!r}"
                if name in EXACT_FIELDS:
                    assert got == e, f"{where} — this field cannot drift"
                assert a <= ABS_BAR, f"{where} ({a:.3e} absolute)"
                assert d <= ULP_BAR, f"{where} ({d:.1f} floored ULP)"
                total += 1
                exact += d == 0.0
        filled += int(zon.hn[1] > 0 or zon.hn[2] > 0)
    assert filled > len(rows) // 2, f"only {filled} cases produced a mode"
    assert exact / total >= 0.99, (f"exact {exact}/{total} "
                                   f"(worst {worst:.1f} @ {worst_where})")
    print(f"\nregmod: {total} cells, {exact} exact "
          f"({100 * exact / total:.2f}%), worst {worst:.1f} floored ULP / "
          f"{worst_abs:.3e} absolute @ {worst_where}")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_slot7_and_unused_modes_keep_the_preset():
    """Slot 7 is ESMOD's; an unused mode keeps HN/HP = -1, as /MODES/ marks."""
    header, inputs, rows = DUMP
    st, modes, muf, sigd, anois, rays, cant, freq, k, gcd = _build(inputs[0])
    for im in range(1, 7):
        modes.hpmod[im, k] = F32(-1.0)          # every slot unused
    zon = ZonState()
    regmod(st, modes, muf, sigd, anois, rays, cant, zon, freq, k, gcd)
    assert np.all(zon.hn[1:8] == np.float32(-1.0))
    assert np.all(zon.hp[1:8] == np.float32(-1.0))
    assert np.all(zon.tloss[1:8] == np.float32(0.0))


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_zero_mufday_is_refused():
    """PRBMUF returns a hard 0 when no median is placed; the Fortran STOPs."""
    st, modes, muf, sigd, anois, rays, cant, freq, k, gcd = _build(DUMP[1][0])
    for i in range(1, 5):
        muf.ymuf[i] = F32(0.0)                  # FSET <= 0 -> step at FGO
        muf.fvmuf[i] = F32(1.0)                 # ... and FREQ above it -> 0
    zon = ZonState()
    with pytest.raises(RegmodError):
        regmod(st, modes, muf, sigd, anois, rays, cant, zon, freq, k, gcd)
