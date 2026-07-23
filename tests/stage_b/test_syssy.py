"""Stage B batched SYSSY vs the SAME tier-B oracle dump Stage A uses.

The whole dump is one batch -- 470 lanes, all twelve months -- so every case
runs side by side and the per-lane index arithmetic is what is under test.
That means a different SYS/PERR table per lane, which the batched routine
takes as (9, 16, 6, N); the shared (9, 16, 6) spelling a real grid would use
is checked separately against the same answers, so neither path is assumed.

MEASURED: 2820/2820 cells bit-exact on CPU AND on CUDA (470 cases x 6
outputs), worst 0.00 ULP.  That is the RIGHT answer and the bar is set at it:
SYSSY has no transcendental in it, only a table read and a divide by 1.28, so
there is no libm floor to hide behind and any residual at all would be an
indexing bug.  Stage A measured the same 2820/2820 -- two independent
transliterations agreeing bit-for-bit with the Fortran.

WHAT THE BATCH COVERS (all asserted below, so the claim stays true):
  * the no-data ITSA-1 fallback, 21 lanes -- including its SL/1.29 quirk
  * both hemispheres (238 south, 232 north), i.e. the KK/ND table shift
  * both KJ clamps: 162 lanes below the 40-degree band, 27 above the 80
  * the LJ==0 and LD==0 wraps to the last local-time block (20 and 43 lanes)
  * both distance codes NN (246 short, 224 long)
  * every one of the 16 LOL columns and all 9 KJ rows
"""

import numpy as np
import pytest
import torch

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.redmap import redmap
from voacap_gpu.stage_b.syssy import syssy_batch

from tests.stage_a.dumputil import COEFFS_DIR, load_dump, ulp_diff_floored

DUMP = load_dump("syssy")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
# No libm, so no floor: the measured answer is 1.000 on both devices and the
# bar says so.  A 0.999 bar here would be hiding a real indexing bug.
EXACT_FLOOR = {"cpu": 1.0, "cuda": 1.0}


def _tables():
    """Per-lane SYS/PERR, (9,16,6,N) and (9,4,6,N).

    SYS and PERR are month-dependent only -- REDMAP copies them straight out
    of the coefficient set without blending on SSN -- so twelve loads cover
    the whole dump and the stack is just a fan-out.
    """
    _, inputs, _ = DUMP
    tab = {m: redmap(COEFFS_DIR, "CCIR", m, F32(100.0))
           for m in {int(r[0]) for r in inputs}}
    sysl = np.stack([np.ascontiguousarray(tab[int(r[0])].sys)
                     for r in inputs], axis=-1)
    perrl = np.stack([np.ascontiguousarray(tab[int(r[0])].perr)
                      for r in inputs], axis=-1)
    return sysl, perrl


def _args(device):
    _, inputs, rows = DUMP
    sysl, perrl = _tables()
    col = lambda i, dt: np.array(  # noqa: E731
        [dt(r[i]) for r in inputs],
        dtype=np.float32 if dt is float else np.int64)
    izf = np.array([int(r[4]) for r in inputs], dtype=bool)
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    # IZF forces the "no data tape" arm without disturbing the tables, exactly
    # as the Stage A test does by zeroing F2D(1,1,1).
    return (t(sysl), t(perrl), t(~izf), t(col(1, float)), t(col(2, float)),
            t(col(3, int))), rows


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_syssy_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_syssy_batch_vs_oracle(device):
    header, inputs, _ = DUMP
    assert header == ["fm", "su", "sl", "fmp", "sup", "slp"]
    args, rows = _args(device)
    got = [x.cpu().numpy() for x in syssy_batch(*args)]

    for f, name in enumerate(header):
        for i, exp in enumerate(rows):
            e = np.float32(exp[f])
            d = ulp_diff_floored(got[f][i], e)
            # BITWISE. Every output is either a copied table cell or a table
            # cell over 1.28; there is no libm and no cancellation, so a
            # nonzero residual means the wrong cell, not rounding.
            assert d == 0.0, (
                f"[{device}] lane {i} {name}: got {got[f][i]!r} expected "
                f"{e!r} ({d:.2f} floored ULP), in {inputs[i]}")

    exact = sum(
        int((got[f] == np.array([np.float32(r[f]) for r in rows])).sum())
        for f in range(len(header)))
    total = len(header) * len(rows)
    assert exact / total >= EXACT_FLOOR[device], f"exact {exact}/{total}"


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_branch_coverage():
    """The batch really does reach every arm the docstring claims.

    Recomputed here from the raw inputs rather than read out of the routine,
    so this stays an independent statement about the DUMP.
    """
    _, inputs, _ = DUMP
    g = np.array([float(r[1]) for r in inputs], dtype=np.float32)
    pt = np.array([float(r[2]) for r in inputs], dtype=np.float32)
    nn = np.array([int(r[3]) for r in inputs])
    izf = np.array([int(r[4]) for r in inputs], dtype=bool)

    gg = np.abs(g) * np.float32(5.729577e1)
    kj = ((gg - np.float32(40.0)) / np.float32(5.0)
          + np.float32(1.5)).astype(np.int64)
    j = (pt + np.float32(0.5)).astype(np.int64)
    jf = j.astype(np.float32)
    lj = (jf / np.float32(3.0) + np.float32(0.67)).astype(np.int64)
    ld = (jf / np.float32(6.0) + np.float32(0.67)).astype(np.int64)

    assert izf.sum() >= 10 and (~izf).sum() >= 100, "no-data arm unbalanced"
    assert (g < 0).sum() >= 50 and (g >= 0).sum() >= 50, "one hemisphere only"
    assert (kj <= 0).sum() >= 20, "KJ never clamps up to 1"
    assert (kj > 9).sum() >= 5, "KJ never clamps down to 9"
    assert ((kj > 0) & (kj <= 9)).sum() >= 50, "KJ is always clamped"
    assert (lj < 1).sum() >= 5, "the LJ==0 -> 8 wrap is never taken"
    assert (ld < 1).sum() >= 5, "the LD==0 -> 4 wrap is never taken"
    assert (nn == 2).sum() >= 50 and (nn == 5).sum() >= 50, "one NN only"
    lol = np.where(g < 0, 8, 0) + np.where(lj < 1, 8, lj)
    assert set(lol.tolist()) == set(range(1, 17)), (
        f"LOL columns reached: {sorted(set(lol.tolist()))}, want all 16")
    assert set(np.clip(kj, 1, 9).tolist()) == set(range(1, 10))
    # The LOL/LD index guards in syssy_batch are safety only; assert the dump
    # cannot reach them so the comment saying so stays honest.
    assert lol.max() <= 16 and lol.min() >= 1
    assert np.where(ld < 1, 4, ld).max() <= 4


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_shared_table_matches_per_lane():
    """The (9,16,6) spelling a real grid uses is the same lookup.

    Every lane on a real grid shares one month, so production passes a 3-D
    table and takes the advanced-index path rather than the gather.  Restrict
    the dump to a single month and the two must agree bitwise.
    """
    _, inputs, _ = DUMP
    month = int(inputs[0][0])
    keep = [i for i, r in enumerate(inputs) if int(r[0]) == month]
    assert len(keep) >= 10
    st = redmap(COEFFS_DIR, "CCIR", month, F32(100.0))
    sys3 = torch.as_tensor(np.ascontiguousarray(st.sys))
    perr3 = torch.as_tensor(np.ascontiguousarray(st.perr))
    n = len(keep)
    sys4 = sys3.unsqueeze(-1).expand(9, 16, 6, n).contiguous()
    perr4 = perr3.unsqueeze(-1).expand(9, 4, 6, n).contiguous()

    g = torch.as_tensor(np.array([float(inputs[i][1]) for i in keep],
                                 dtype=np.float32))
    pt = torch.as_tensor(np.array([float(inputs[i][2]) for i in keep],
                                  dtype=np.float32))
    nn = torch.as_tensor(np.array([int(inputs[i][3]) for i in keep]))
    hd = torch.as_tensor(np.array([not int(inputs[i][4]) for i in keep]))

    a = syssy_batch(sys3, perr3, hd, g, pt, nn)
    b = syssy_batch(sys4, perr4, hd, g, pt, nn)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@pytest.mark.skipif(not torch.cuda.is_available() or DUMP is None,
                    reason="no CUDA")
def test_cuda_matches_cpu():
    """SYSSY alone, from ONE CPU-built input set moved across.

    Nothing upstream to inherit -- the tables come from REDMAP on the host --
    so this is purely the index arithmetic, and it has to be BITWISE.  The
    float32 divides by 5.0, 3.0, 6.0 and 1.28 all go through ``_k`` for
    exactly this reason: a CUDA divide by a Python float rounds like a double
    divide and would move a boundary case a whole table cell.
    """
    args, _ = _args("cpu")
    a = syssy_batch(*args)
    b = syssy_batch(*[x.cuda() for x in args])
    for name, x, y in zip(("fm", "su", "sl", "fmp", "sup", "slp"), a, b):
        assert torch.equal(x, y.cpu()), f"{name} differs across devices"
