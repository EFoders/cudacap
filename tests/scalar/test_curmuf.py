"""Stage A CURMUF_orig vs. the tier-B oracle dump (Module-3 MUF/FOT).

Each case injects controlled FI/YI/HI/FS + path geometry (isolating CURMUF
from the VERSY floor) and loads a real F2D table via REDMAP, then runs the
oracle's own CURMUF_orig.  Measured at generation (145 cases, 53 fields):
every circuit-level output is bit-exact — EMUF/F1MUF/F2MUF/ESMUF/ALLMUF
(the MUFs), FOT/HPF (the decile envelopes), MODMUF, and all the SIG/decile
statistics.  The only residual divergence is the [[f32-libm-noise]] floor
on the takeoff angles (DELMUF) and virtual heights (HPMUF), whose
ATAN/ASIN/COS chains amplify near grazing incidence on the longest paths
(worst 14 floored ULP ~= 1.7e-6 deg at 10 559 km) — far below the printed
0.01-deg / 0.1-MHz resolution.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.curmuf import curmuf
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.mufstate import MufState
from voacap_gpu.scalar.redmap import redmap

from .dumputil import COEFFS_DIR, load_dump, ulp_diff_floored

DUMP = load_dump("curmuf")

# takeoff-angle / virtual-height fields carry the trig floor; everything
# else (the MUFs, deciles, statistics) is bit-exact.
TRIG_FLOOR = {"delmuf1", "delmuf2", "delmuf3", "delmuf4",
              "hpmuf1", "hpmuf2", "hpmuf3", "hpmuf4", "angmuf"}
# the printed circuit quantities — must match the oracle exactly
EXACT_FIELDS = ["emuf", "f1muf", "f2muf", "esmuf", "allmuf", "fot", "hpf",
                "modmuf"]


def _run_case(inp):
    (month, ssn, gcdkm, amind, clat, clck, fi1, fi2, fi3, yi1, yi2, yi3,
     hi1, hi2, hi3, fs1, fs2, fs3) = (float(x) for x in inp)
    st = redmap(COEFFS_DIR, "CCIR", int(round(month)), F32(ssn))
    s = LayerState(km=1, ssn=F32(ssn))
    s.kfx = 1
    s.fi[1, 1], s.fi[2, 1], s.fi[3, 1] = F32(fi1), F32(fi2), F32(fi3)
    s.yi[1, 1], s.yi[2, 1], s.yi[3, 1] = F32(yi1), F32(yi2), F32(yi3)
    s.hi[1, 1], s.hi[2, 1], s.hi[3, 1] = F32(hi1), F32(hi2), F32(hi3)
    s.fs[1, 1], s.fs[2, 1], s.fs[3, 1] = F32(fs1), F32(fs2), F32(fs3)
    s.hs[1] = F32(110.0)
    s.clat[1], s.clck[1] = F32(clat), F32(clck)
    muf = MufState()
    curmuf(s, muf, st, F32(gcdkm), F32(gcdkm) / F32(6370.0),
           F32(amind), F32(ssn), it=1)
    out = {"emuf": muf.emuf[1], "f1muf": muf.f1muf[1], "f2muf": muf.f2muf[1],
           "esmuf": muf.esmuf[1], "allmuf": muf.allmuf[1], "fot": muf.fot[1],
           "hpf": muf.hpf[1], "angmuf": muf.angmuf[1],
           "modmuf": F32(muf.modmuf)}
    for base, arr in [("sigl", muf.sigl), ("sigu", muf.sigu),
                      ("delmuf", muf.delmuf), ("hpmuf", muf.hpmuf),
                      ("htmuf", muf.htmuf), ("fvmuf", muf.fvmuf),
                      ("afmuf", muf.afmuf), ("nhopmf", muf.nhopmf),
                      ("yfot", muf.yfot), ("yhpf", muf.yhpf),
                      ("ymuf", muf.ymuf)]:
        for L in range(1, 5):
            out[f"{base}{L}"] = F32(arr[L])
    return out


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_curmuf_dumps.sh")
def test_curmuf_vs_oracle():
    header, inputs, rows = DUMP
    assert len(header) == 53
    total = exact = 0
    exact_by_field = {h: 0 for h in header}
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        got = _run_case(inp)
        for name, e_str in zip(header, exp):
            e = np.float32(e_str)
            bar = 64 if name in TRIG_FLOOR else 2
            d = ulp_diff_floored(got[name], e)
            assert d <= bar, (
                f"case {k} {name}: got {got[name]!r} expected {e!r} "
                f"({d:.2f} floored ULP > {bar}), in {inp}")
            total += 1
            hit = d == 0.0
            exact += hit
            exact_by_field[name] += hit

    n = len(inputs)
    for name in EXACT_FIELDS:
        assert exact_by_field[name] == n, (
            f"{name} (a printed MUF/FOT quantity) must be bit-exact: "
            f"{exact_by_field[name]}/{n}")
    assert exact / total >= 0.98, f"bit-exact rate {exact}/{total}"
