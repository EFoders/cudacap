"""Stage A GENOIS vs the tier-B oracle dump (combined receiver noise).

Every input but the /TWO/ tables is injected: GENOIS interpolates no upstream
quantized table, so the FINDF/IFOB replay rule does not apply and direct
injection buys control over the branches instead.  The tables come from
redmap() reading the vendored ASCII coefficients, and GENFAM -- the one thing
between those tables and this routine -- is pinned separately in
test_genfam.py, so a failure here is GENOIS's own arithmetic.

Measured: 570 cases x 21 columns = 11,970 cells, ALL BIT-EXACT.  That is worth
stating plainly, because it is not obvious in advance -- the routine is twelve
EXPs, three powers of ten, two natural logs and two square roots deep, and it
mixes REAL*4 and REAL*8 at nearly every step.  Getting zero drift out of it
means three things are modelled exactly right, and each would have shown up as
a nonzero residual otherwise:

* which quantities are REAL*8 (A, V and the sigmas) and which are not -- every
  EXP feeding the sums is REAL*4 and only its *result* widens;
* ``**2`` is an INTEGER exponent, so gfortran emits a multiply, while
  ``10.**x`` is a REAL one and emits powf.  Routing the first through pow32
  would have drifted;
* ``4.34294`` and ``4.34294d0`` are different constants, and GENOIS uses both
  -- the first promoted from binary32 in XRNSE, the second a true double in
  Caruana's test.

So unlike REGMOD and INMUF this needs no two-bar treatment: there is no
residual to characterise.  The ULP bar below exists only to keep a future
libm difference reporting as a small drift rather than a hard failure, and
the exactness floor is what actually guards the arithmetic.

Fields asserted BIT-EXACT unconditionally because they are copies or integers
rather than computed: ZNOISE (a function of the NOISE card alone), the seven
EFF slots (GAIN's output broadcast), METHOD_USED and MODE_N2.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.antennastate import CantennaState
from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.genois import OldNoiseModel, genois
from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.noisestate import AnoisState, NoiseDiagState
from voacap_gpu.stage_a.readant import read_antenna
from voacap_gpu.stage_a.redmap import redmap
from voacap_gpu.stage_a.signalstate import TonState
from voacap_gpu.stage_a.zonstate import ZonState

from .dumputil import (ANTENNAS_DIR, COEFFS_DIR, load_dump, ulp_diff_floored)

DUMP = load_dump("genois")

GENOIS_ULP_BAR = 4.0            # measured worst 0.0 -- a libm safety valve
GENOIS_EXACT = {"znoise", "method_used", "mode_n2",
                "eff1", "eff2", "eff3", "eff4", "eff5", "eff6", "eff7"}


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


def _run(inp):
    """One case: build the injected state, call GENOIS, return the outputs."""
    (month, izf, freq, noise, rlat, atnu, atny,
     cc, tm, kj, jk, fi3, rxlo, rxhi) = inp
    st = redmap(COEFFS_DIR, "CCIR", int(month), F32(100.0))
    if int(izf):
        # force the no-data-base branch without disturbing the redmap cache
        f2d = st.f2d.copy()
        f2d[0, 0, 0] = F32(-1.0)
        st = type(st)(**{**st.__dict__, "f2d": f2d})
    state = LayerState()
    state.kfx = 1
    state.fi[3, 1] = F32(fi3)
    anois = AnoisState(atnu=F32(atnu), atny=F32(atny), cc=F32(cc),
                       tm=F32(tm), kj=int(kj), jk=int(jk))
    ton = TonState()
    zon = ZonState()
    diag = NoiseDiagState()
    genois(st, state, anois, ton, zon, diag, _cantenna(rxlo, rxhi),
           F32(freq), int(noise), F32(rlat))
    return {
        "rcnse": anois.rcnse, "du": anois.du, "dl": anois.dl,
        "sigm": anois.sigm, "sygu": anois.sygu, "sygl": anois.sygl,
        "gnos": ton.gnos, "xnoise": ton.xnoise, "znoise": ton.znoise,
        **{f"eff{i}": zon.eff[i] for i in range(1, 8)},
        "xn_atnos": diag.xn_atnos, "xn_galactic": diag.xn_galactic,
        "xn_mmn": diag.xn_mmn,
        "method_used": F32(diag.method_used), "mode_n2": F32(diag.mode_n2),
    }


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_genois_dumps.sh")
def test_genois_vs_oracle():
    header, inputs, rows = DUMP
    total = exact = 0
    worst_ulp = 0.0
    worst_at = None
    seen = {"caruana": 0, "nodata": 0, "no_galactic": 0, "galactic": 0,
            "man_level": 0, "man_default": 0, "man_index": 0, "no_ant": 0}
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        got = _run(inp)
        seen["nodata"] += int(inp[1]) != 0
        noise = int(inp[3])
        seen["man_level"] += noise > 0
        seen["man_default"] += noise == 0
        seen["man_index"] += noise < 0
        seen["caruana"] += got["method_used"] == 1
        seen["no_galactic"] += got["mode_n2"] == 0
        seen["galactic"] += got["mode_n2"] == 1
        seen["no_ant"] += got["eff1"] == 0
        for name, e_str in zip(header, exp):
            g = got[name]
            e = np.float32(e_str)
            if name in GENOIS_EXACT:
                assert g == e, (f"case {k} {name}: got {g!r} expected {e!r} "
                                f"(must be bit-exact) in {inp}")
                total += 1
                exact += 1
                continue
            d = ulp_diff_floored(g, e)
            if d > worst_ulp:
                worst_ulp, worst_at = d, f"case {k} {name} {inp}"
            assert d <= GENOIS_ULP_BAR, (
                f"case {k} {name}: got {g!r} expected {e!r} "
                f"({abs(float(g) - float(e)):.3e} abs, {d:.1f} floored ULP) "
                f"in {inp}")
            total += 1
            exact += d == 0.0
    for arm, n in seen.items():
        assert n, f"branch never reached: {arm} ({seen})"
    assert exact / total >= 0.999, (
        f"exact {exact}/{total} (worst {worst_ulp:.1f} ULP at {worst_at})")


def test_old_noise_model_is_refused():
    """NYEAR ' 2089' selects GENOIS_OLD -- superseded physics, not ported."""
    st = redmap(COEFFS_DIR, "CCIR", 6, F32(100.0))
    with pytest.raises(OldNoiseModel):
        genois(st, LayerState(), AnoisState(), TonState(), ZonState(),
               NoiseDiagState(), CantennaState(), F32(10.0), -2, F32(0.5),
               nyear=" 2089")


def test_galactic_noise_is_dropped_below_the_f2_critical():
    """Galactic noise cannot penetrate at or below foF2, so it is zeroed."""
    st = redmap(COEFFS_DIR, "CCIR", 6, F32(100.0))
    state = LayerState()
    state.kfx = 1
    state.fi[3, 1] = F32(9.0)

    def run(freq):
        anois = AnoisState(atnu=F32(70.0), atny=F32(68.0), cc=F32(10.0),
                           tm=F32(10.0), kj=3, jk=3)
        ton, zon, diag = TonState(), ZonState(), NoiseDiagState()
        genois(st, state, anois, ton, zon, diag, CantennaState(),
               F32(freq), -2, F32(0.5))
        return ton, diag, anois

    ton, diag, _ = run(8.99)
    assert diag.mode_n2 == 0 and ton.gnos == F32(-204.0)
    ton, diag, _ = run(9.0)                     # the boundary is inclusive
    assert diag.mode_n2 == 0
    ton, diag, _ = run(9.01)
    assert diag.mode_n2 == 1 and ton.gnos != F32(-204.0)


def test_no_data_base_uses_the_fixed_deciles():
    """F2D(1,1,1) < 0 skips GENFAM: atmospheric noise vanishes but its
    deciles are still the fixed 9/7 dB, so the spread does not collapse."""
    st = redmap(COEFFS_DIR, "CCIR", 6, F32(100.0))
    f2d = st.f2d.copy()
    f2d[0, 0, 0] = F32(-1.0)
    st_nodata = type(st)(**{**st.__dict__, "f2d": f2d})
    state = LayerState()
    state.kfx = 1
    state.fi[3, 1] = F32(4.0)
    anois = AnoisState(atnu=F32(70.0), atny=F32(68.0), cc=F32(10.0),
                       tm=F32(10.0), kj=3, jk=3)
    diag = NoiseDiagState()
    genois(st_nodata, state, anois, TonState(), ZonState(), diag,
           CantennaState(), F32(10.0), -2, F32(0.5))
    assert diag.xn_atnos == F32(-204.0)          # 0 dB above kTb
    assert anois.du > 0 and anois.dl > 0
    assert diag.method_used == 2                 # 9 and 7 are both under 12


def test_receiver_efficiency_reaches_six_slots_not_seven():
    """GENOIS fills EFF(1..6); slot 7 belongs to ESMOD and must stay put."""
    st = redmap(COEFFS_DIR, "CCIR", 6, F32(100.0))
    state = LayerState()
    state.kfx = 1
    state.fi[3, 1] = F32(4.0)
    zon = ZonState()
    zon.eff[7] = F32(-42.0)
    genois(st, state, AnoisState(kj=3, jk=3), TonState(), zon,
           NoiseDiagState(), _cantenna(2.0, 30.0), F32(14.0), -2, F32(0.5))
    assert all(zon.eff[i] == zon.eff[1] for i in range(1, 7))
    assert zon.eff[1] != 0.0
    assert zon.eff[7] == F32(-42.0)
