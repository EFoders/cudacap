"""Stage A Module-2 layer chain vs. the tier-B oracle dump.

One case = the full REDMAP -> VIRTIM -> TIMVAR -> F2VAR -> ESIND -> IONSET
pipeline for one sample area (KM=1), matched field-by-field against the
oracle running the identical Fortran routines.

Measured at generation (168 cases, 21 fields): every transcendental-free
field is bit-exact, and every residual divergence is the already-diagnosed
[[f32-libm-noise]] VERSY cosf floor propagated linearly into the outputs
that are direct functions of GAMMA:

  fs1/fs2/fs3 = GAMMA(1..3)*PSC(4)     (Es sets, ~350x cancellation)
  fi3 = (GAMMA(4)+0.5*GYZ)*PSC(3)      (foF2 term)
  f2m3 = GAMMA(5); hpf2 = 1490/GAMMA(5)-176

so their bars mirror tests/stage_a/test_versy.py's GAMMA bars.  Worst
observed: fs2 252 floored ULP (~3e-5 absolute, in the Es 5e-5 band).  The
branch-driven E/F1 slots (fi2/hi2/yi2) and KFX are exact — no case sits on
a zenith branch boundary where a 1-ULP wobble would flip F1 present/absent.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.esind import esind
from voacap_gpu.scalar.f2var import f2var
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.ionset import ionset
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.scalar.timvar import timvar
from voacap_gpu.scalar.virtim import virtim

from .dumputil import COEFFS_DIR, load_dump, ulp_diff_floored

DUMP = load_dump("layers")

FIELDS = ["fi1", "fi2", "fi3", "yi1", "yi2", "yi3", "hi1", "hi2", "hi3",
          "fs1", "fs2", "fs3", "hs", "f2m3", "hpf2", "rat", "abiy", "clck",
          "zenang", "zenmax", "kfx"]

# scale-floored ULP bars.  512 for the Es deciles and 256 for fi3 encode
# the VERSY cancellation floor (see test_versy.py); the rest are near-exact.
BARS = {
    "fi1": 16, "fi2": 32, "fi3": 256,
    "yi1": 4, "yi2": 32, "yi3": 32,
    "hi1": 2, "hi2": 32, "hi3": 32,
    "fs1": 512, "fs2": 512, "fs3": 512,
    "hs": 2, "f2m3": 16, "hpf2": 16, "rat": 16,
    "abiy": 16, "clck": 2, "zenang": 16, "zenmax": 16, "kfx": 0,
}

# Transcendental-free outputs: any divergence here is a real transliteration
# bug, not the libm floor, so they must be exactly bit-exact.
EXACT_FIELDS = ["hi1", "hs", "yi1", "clck", "kfx"]


def _run_case(inp):
    month = int(inp[0])
    ssn = F32(inp[1])
    gmt = F32(inp[2])
    clat, clong, glat, gmdip, gyz = (F32(x) for x in inp[3:8])
    st = redmap(COEFFS_DIR, "CCIR", month, ssn)
    ab = virtim(st, gmt)
    s = LayerState(km=1, month=month, ssn=ssn)
    s.clat[1], s.clong[1], s.glat[1] = clat, clong, glat
    s.gmdip[1], s.gyz[1] = gmdip, gyz
    timvar(s, st, ab, gmt)
    f2var(s, st, ab)
    esind(s, st, ab)
    ionset(s)
    return {
        "fi1": s.fi[1, 1], "fi2": s.fi[2, 1], "fi3": s.fi[3, 1],
        "yi1": s.yi[1, 1], "yi2": s.yi[2, 1], "yi3": s.yi[3, 1],
        "hi1": s.hi[1, 1], "hi2": s.hi[2, 1], "hi3": s.hi[3, 1],
        "fs1": s.fs[1, 1], "fs2": s.fs[2, 1], "fs3": s.fs[3, 1],
        "hs": s.hs[1], "f2m3": s.f2m3[1], "hpf2": s.hpf2[1],
        "rat": s.rat[1], "abiy": s.abiy[1], "clck": s.clck[1],
        "zenang": s.zenang[1], "zenmax": s.zenmax[1], "kfx": F32(s.kfx),
    }


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_layer_dumps.sh")
def test_layers_vs_oracle():
    header, inputs, rows = DUMP
    assert header == FIELDS
    total = 0
    exact = 0
    exact_by_field = {k: 0 for k in FIELDS}
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        got = _run_case(inp)
        for j, name in enumerate(FIELDS):
            e = np.float32(exp[j])
            d = ulp_diff_floored(got[name], e)
            assert d <= BARS[name], (
                f"case {k} {name}: got {got[name]!r} expected {e!r} "
                f"({d:.2f} floored ULP > {BARS[name]}), in {inp}")
            total += 1
            hit = d == 0.0
            exact += hit
            exact_by_field[name] += hit

    n = len(inputs)
    for name in EXACT_FIELDS:
        assert exact_by_field[name] == n, (
            f"{name} must be bit-exact (transcendental-free): "
            f"{exact_by_field[name]}/{n}")
    assert exact / total >= 0.96, f"bit-exact rate {exact}/{total}"
