"""Stage A SIGDIS_orig vs the tier-B oracle dump (signal distribution +
absorption).

Each case loads SYS/PERR/F2D via redmap(), injects three areas' FI/YI/HI +
GLAT + CLCK, builds each active area's ionogram with LECDEN+GENION, injects the
F2/Es distribution medians/deciles into a MufState, and runs SIGDIS, comparing
the /SIGD/ and /TON/ path outputs plus the per-area ABIY/ARTIC (checked only
for the active areas k<=KFX).

Measured over 130 cases x (18 path outputs + active-area ABIY/ARTIC) = 2824
cells: every one is bit-exact, including the EXP-derived (ABIY/ACAV) and
ALOG10-derived (ASM/DSU/DSL) fields -- the evaluate-round-once f32math helpers
match this gfortran build's runtime expf/log10f on all these inputs.  The bar
still allows a small residual for the [[f32-libm-noise]] floor (the transcen-
dental paths could disagree by ~1 ULP on other arguments, amplified by the
ASM/DSU/DSL cancellations), but none is observed here.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.genion import genion
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.lecden import lecden
from voacap_gpu.scalar.mufstate import MufState
from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.scalar.signalstate import SigdState, TonState
from voacap_gpu.scalar.sigdis import sigdis

from .dumputil import COEFFS_DIR, load_dump, ulp_diff_floored

DUMP = load_dump("sigdis")


def _run_case(inp):
    month = int(inp[0])
    ssn = float(inp[1])
    kfx = int(inp[2])
    jmode = int(inp[3])
    gcdkm = float(inp[4])
    iedp = int(inp[5])
    s = LayerState(kfx=kfx)
    s.iedp = iedp
    for a in range(1, 4):
        off = 6 + (a - 1) * 11
        (fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3, glat, clck) = (
            float(x) for x in inp[off:off + 11])
        s.fi[1, a], s.fi[2, a], s.fi[3, a] = F32(fi1), F32(fi2), F32(fi3)
        s.yi[1, a], s.yi[2, a], s.yi[3, a] = F32(yi1), F32(yi2), F32(yi3)
        s.hi[1, a], s.hi[2, a], s.hi[3, a] = F32(hi1), F32(hi2), F32(hi3)
        s.glat[a], s.clck[a] = F32(glat), F32(clck)
    st = redmap(COEFFS_DIR, "CCIR", month, F32(ssn))
    for k in range(1, kfx + 1):
        lecden(s, k)
        genion(s, k)
    (yfot3, yfot4, yhpf3, yhpf4, ymuf3, ymuf4,
     sigl3, sigl4, sigu3, sigu4) = (float(x) for x in inp[39:49])
    m = MufState()
    m.yfot[3], m.yfot[4] = F32(yfot3), F32(yfot4)
    m.yhpf[3], m.yhpf[4] = F32(yhpf3), F32(yhpf4)
    m.ymuf[3], m.ymuf[4] = F32(ymuf3), F32(ymuf4)
    m.sigl[3], m.sigl[4] = F32(sigl3), F32(sigl4)
    m.sigu[3], m.sigu[4] = F32(sigu3), F32(sigu4)
    sigd, ton = SigdState(), TonState()
    sigdis(s, m, st, sigd, ton, gcdkm, jmode)
    return [
        ("aglat", sigd.aglat, True), ("acav", sigd.acav, True),
        ("feav", sigd.feav, True), ("htloss", sigd.htloss, True),
        ("xve", sigd.xve, True), ("xnuz", sigd.xnuz, True),
        ("hnu", sigd.hnu, True), ("afe", sigd.afe, True),
        ("bfe", sigd.bfe, True), ("asm", sigd.asm, True),
        ("dsu", sigd.dsu, True), ("dsl", sigd.dsl, True),
        ("adj", ton.adj, True), ("su", ton.su, True), ("sl", ton.sl, True),
        ("ads", ton.ads, True), ("sus", ton.sus, True), ("sls", ton.sls, True),
        ("abiy1", s.abiy[1], kfx >= 1), ("abiy2", s.abiy[2], kfx >= 2),
        ("abiy3", s.abiy[3], kfx >= 3), ("artic1", s.artic[1], kfx >= 1),
        ("artic2", s.artic[2], kfx >= 2), ("artic3", s.artic[3], kfx >= 3),
    ]


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_sigdis_dumps.sh")
def test_sigdis_vs_oracle():
    header, inputs, rows = DUMP
    assert len(header) == 24
    total = exact = 0
    worst = 0.0
    worst_where = ""
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        got = _run_case(inp)
        for (name, g, check), e_str in zip(got, exp):
            if not check:
                continue
            e = np.float32(e_str)
            d = ulp_diff_floored(g, e)
            if d > worst:
                worst, worst_where = d, f"case {k} {name}: {g!r} vs {e!r}"
            assert d <= 8, (f"case {k} {name}: got {g!r} expected {e!r} "
                            f"({d:.2f} floored ULP), in {inp}")
            total += 1
            exact += d == 0.0
    assert exact / total >= 0.99, (f"exact {exact}/{total} "
                                   f"(worst {worst:.2f} @ {worst_where})")
