"""Stage A INMUF vs the tier-B oracle dump.

INMUF inserts the modes FDIST could not find -- over the MUF, and at distances
the ray table does not reach -- then runs REGMOD on the result, so this test
covers both its own bookkeeping and the integration with everything beneath it.

The first 126 columns of the dump are *injected inputs*, not results: the
FVERT/HPRIM/HTRUE/AFAC ionogram, which INMUF's XLIN interpolates and which by
the FINDF/IFOB rule has to come from the oracle, plus FI and HI **as LECDEN and
GENION leave them**.  That last part was not obvious and cost a wrong answer
first time round: the chain adjusts the critical frequencies and layer heights
it is handed, and INMUF's layer tests read the adjusted values, so feeding back
the case file's originals put a mode in the wrong layer.  Injecting part of a
routine's upstream state is worse than injecting none of it.

Beyond that: NUMMOD and the /MODES/ table check the compaction and the two
insert paths, the /MUFS/ block checks that stage 3's overwrite of YMUF/SIGL/
YFOT/SIGU/YHPF is undone before returning, and /ZON/ checks REGMOD ran on the
modes INMUF actually produced.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.inmuf import inmuf
from voacap_gpu.scalar.zonstate import ZonState

from .dumputil import load_dump, ulp_diff_floored
from .test_regmod import FIELDS
from .test_regmod import COLUMNS as REGMOD_COLUMNS
from .test_regmod import _build as _build_regmod

DUMP = load_dump("inmuf")

COLUMNS = (list(REGMOD_COLUMNS) + ["ihop", "modmuf"]
           + [f"delmuf{i}" for i in range(1, 5)]
           + [f"afmuf{i}" for i in range(1, 5)]
           + [f"nhopmf{i}" for i in range(1, 5)]
           + ["fi2", "fi3", "yi1", "yi2", "yi3", "hi1", "hi2", "hi3", "iedp"])

IONO = ["fvert", "hprim", "htrue", "afac"]          # injected, 30 each
MODEF = ["delmod", "hpmod", "htmod", "fvmod", "itmod", "afmod"]
MUFF = ["ymuf", "sigl", "yfot", "sigu", "yhpf"]

# column offsets into one expected.csv row.  The leading block is INJECTED
# INPUT: the ionogram, and FI/HI as LECDEN+GENION leave them -- the chain
# adjusts the critical frequencies and layer heights it was handed, and INMUF's
# layer tests read the adjusted values, so the case-file originals are wrong to
# feed back in.
_O_IONO = 0
_O_FIHI = 120
_O_NUMMOD = 126
_O_GHOP = 127
_O_MODES = 128
_O_MUFS = _O_MODES + 36
_O_ZON = _O_MUFS + 20

# Looser than REGMOD's 1e-4 because INMUF reaches deeper losses -- FLDST gets
# to -206 dB here against REGMOD's tens -- and because the inserted modes carry
# XLIN plus a second Martyn correction ahead of everything REGMOD then does.
# Measured worst is 2.4e-4; the bar is set just above it, not rounded up.
INMUF_ABS_BAR = 4.0e-4

# REGMOD's EXACT_FIELDS does not carry over.  There, B/HP/EFF/RGAIN/TGAIN only
# copied injected inputs; here INMUF *computes* DELMOD (ATAN(TDEL)*R2D) and
# HPMOD (HPVER + the Martyn correction), so B and HP inherit that drift, and
# the gains inherit it through GAIN's elevation index.  What is left is the
# genuinely integral: the hop count and the two layer indices.
INMUF_EXACT = {"hn", "nmode", "itmod"}

# Headroom over REGMOD's 256 for the same reason as the absolute bar: SN lands
# at 0.6 while carrying a 3e-5 dB error inherited from a ~150 dB budget, which
# reads as exactly 256 floored ULP.  Sitting a bar on the observed maximum
# makes it fire on noise, so this one is doubled and the absolute bar above
# does the real work.
INMUF_ULP_BAR = 512.0


def _build(row, exp):
    """States for one case; the ionogram comes from `exp`, not from `row`."""
    v = dict(zip(COLUMNS, (float(x) for x in row)))
    k = int(v["jmode"])
    st, modes, muf, sigd, anois, rays, cant, freq, _k, gcd = _build_regmod(
        row[:len(REGMOD_COLUMNS)])
    # FI/HI as the oracle's LECDEN+GENION left them, not as the case set them
    for i in (1, 2, 3):
        st.fi[i, k] = np.float32(exp[_O_FIHI + i - 1])
        st.hi[i, k] = np.float32(exp[_O_FIHI + 3 + i - 1])
        st.yi[i, k] = F32(v[f"yi{i}"])
    muf.modmuf = int(v["modmuf"])
    for i in range(1, 5):
        muf.delmuf[i] = F32(v[f"delmuf{i}"])
        muf.afmuf[i] = F32(v[f"afmuf{i}"])
        muf.nhopmf[i] = int(v[f"nhopmf{i}"])
    # inject the oracle's ionogram
    for j, name in enumerate(IONO):
        col = getattr(st, name)
        for ih in range(1, 31):
            col[ih, k] = np.float32(exp[_O_IONO + j * 30 + ih - 1])
    return st, modes, muf, sigd, anois, rays, cant, freq, k, gcd, int(v["ihop"])


def _negative_takeoff(exp) -> bool:
    """Does the oracle's own /MODES/ hold a used slot at a negative angle?

    INMUF's stage-3 Martyn recompute sets DELMOD = ATAN(TDEL)*R2D, and TDEL
    goes negative when COS(PSI) < RZ/(RZ+HPNOW).  REGMOD then hands that
    straight to GAIN, whose elevation index J = INT(DELTD)+1 goes <= 0 -- and
    GAIN clamps only the top of the axis, so it reads ARRAY out of bounds,
    picking up a neighbouring antenna's table.  Deterministic, but meaningless.

    The port raises IndexError there instead (see gain.antenna_gain), so /ZON/
    cannot be compared on such a case; /MODES/ still can, since INMUF writes it
    in full before calling REGMOD.
    """
    delmod = [float(exp[_O_MODES + i]) for i in range(6)]
    hpmod = [float(exp[_O_MODES + 6 + i]) for i in range(6)]
    return any(h > 0.0 and d < 0.0 for d, h in zip(delmod, hpmod))


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_inmuf_dumps.sh")
def test_inmuf_vs_oracle():
    header, inputs, rows = DUMP
    assert header[_O_NUMMOD] == "nummod" and header[_O_GHOP] == "ghop"
    assert header[_O_ZON] == "abps1"
    total = exact = 0
    worst = 0.0
    worst_abs = 0.0
    worst_where = ""
    inserted = 0
    out_of_bounds = 0
    for kc, (inp, exp) in enumerate(zip(inputs, rows)):
        (st, modes, muf, sigd, anois, rays, cant,
         freq, k, gcd, ihop) = _build(inp, exp)
        zon = ZonState()
        ub = _negative_takeoff(exp)
        if ub:
            out_of_bounds += 1
            with pytest.raises(IndexError):
                inmuf(st, modes, muf, sigd, anois, rays, cant, zon,
                      freq, k, gcd, ihop)
            nummod = None
        else:
            nummod = inmuf(st, modes, muf, sigd, anois, rays, cant, zon,
                           freq, k, gcd, ihop)
            # NUMMOD and the layer index are counts -- they cannot drift
            assert nummod == int(float(exp[_O_NUMMOD])), (
                f"case {kc} nummod: got {nummod} "
                f"expected {int(float(exp[_O_NUMMOD]))}")

        # /MODES/ is written in full before REGMOD runs, so it is comparable
        # even on the out-of-bounds case
        checks = []
        for j, name in enumerate(MODEF):
            arr = getattr(modes, name)
            for im in range(1, 7):
                checks.append((f"{name}[{im}]", arr[im, k],
                               np.float32(exp[_O_MODES + j * 6 + im - 1])))
        if not ub:
            checks.append(("ghop", modes.ghop, np.float32(exp[_O_GHOP])))
            for j, name in enumerate(MUFF):
                arr = getattr(muf, name)
                for il in range(1, 5):
                    checks.append((f"{name}[{il}] (restored)", arr[il],
                                   np.float32(exp[_O_MUFS + j * 4 + il - 1])))
            for j, name in enumerate(FIELDS):
                arr = getattr(zon, name)
                for im in range(1, 8):
                    checks.append((f"{name}[{im}]", arr[im],
                                   np.float32(exp[_O_ZON + j * 7 + im - 1])))

        for label, got, e in checks:
            got = np.float32(got)
            d = ulp_diff_floored(got, e)
            a = abs(float(got) - float(e))
            where = f"case {kc} {label}: got {got!r} expected {e!r}"
            if d > worst:
                worst, worst_where = d, where
            worst_abs = max(worst_abs, a)
            if label.split("[")[0] in INMUF_EXACT:
                assert got == e, f"{where} — this field cannot drift"
            assert a <= INMUF_ABS_BAR, f"{where} ({a:.3e} absolute)"
            assert d <= INMUF_ULP_BAR, f"{where} ({d:.1f} floored ULP)"
            total += 1
            exact += d == 0.0
        inserted += bool(nummod)
    assert inserted > len(rows) // 3, f"only {inserted} cases produced a mode"
    assert out_of_bounds == 1, (
        f"{out_of_bounds} cases drive GAIN out of bounds; the grid pins this "
        f"at exactly 1 so a change in the count is noticed")
    assert exact / total >= 0.99, (f"exact {exact}/{total} "
                                   f"(worst {worst:.1f} @ {worst_where})")
    print(f"\ninmuf: {total} cells, {exact} exact "
          f"({100 * exact / total:.2f}%), worst {worst:.1f} floored ULP / "
          f"{worst_abs:.3e} absolute @ {worst_where}")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_ionogram_injection_matches_the_oracle():
    """The injected columns must be exactly what the oracle chain produced.

    Trivially true by construction -- which is the point: it documents that
    these 120 columns are an input, so a future change that recomputes them
    from LECDEN/GENION in Python fails here first, loudly, instead of showing
    up as unexplained drift in the outputs.
    """
    header, inputs, rows = DUMP
    for j, name in enumerate(IONO):
        assert header[_O_IONO + j * 30] == f"{name}01"
    st, _, _, _, _, _, _, _, k, _, _ = _build(inputs[0], rows[0])
    for j, name in enumerate(IONO):
        col = getattr(st, name)
        for ih in range(1, 31):
            assert col[ih, k] == np.float32(rows[0][_O_IONO + j * 30 + ih - 1])
