"""The status table: one set of globally unique refusal codes, three mirrors.

``cuda/kernels/status.cuh`` is the authority -- a per-routine base plus the
routine's local code -- and documents every code in one table.  Python
mirrors it as ``engine.STATUS_*`` (what ``_scalar_status`` maps the CPU
backend's exceptions onto, and what ``Results`` documents).  This holds the
three together, checks uniqueness, and runs the kernel on inputs that reach
several of the codes, so a renumbering on one side only cannot pass.
"""

import pathlib
import re

import numpy as np
import pytest

from voacap_gpu import engine

KERNELS = (pathlib.Path(engine.__file__).resolve().parent / "cuda"
           / "kernels")


def _header():
    return (KERNELS / "status.cuh").read_text(encoding="utf-8")


def _defines():
    """Every ``#define VC_ST_*`` in status.cuh, evaluated."""
    vals = {}
    for name, expr in re.findall(r"^#define\s+(VC_ST_\w+)\s+(.+?)\s*$",
                                 _header(), re.M):
        expr = re.sub(r"VC_ST_\w+", lambda m: str(vals[m.group(0)]), expr)
        vals[name] = int(eval(expr, {"__builtins__": {}}))   # noqa: S307
    return vals


def _documented():
    """{code: routine} from the comment table in status.cuh."""
    rows = re.findall(r"^//\s+(-?\d+)\s+([A-Z]+)\s", _header(), re.M)
    return {int(c): r for c, r in rows}


def _engine_codes():
    return {k: v for k, v in vars(engine).items()
            if k.startswith("STATUS_") and isinstance(v, int)}


def test_codes_are_unique():
    codes = _engine_codes()
    vals = list(codes.values())
    assert len(vals) == len(set(vals)), codes


def test_engine_mirrors_the_header():
    doc = _documented()
    eng = {v for k, v in _engine_codes().items()
           if k not in ("STATUS_OK", "STATUS_UNWRITTEN",
                        "STATUS_CPU_UNMAPPED")}
    assert eng == set(doc), (sorted(eng ^ set(doc)))
    # every documented code is base + local with local in 1..99, and its
    # routine's base is the header's
    d = _defines()
    for code, routine in doc.items():
        base = d[f"VC_ST_{routine}"]
        assert base % 100 == 0 and 0 < code - base < 100, (code, routine)
    assert d["VC_ST_OK"] == engine.STATUS_OK == 0
    assert d["VC_ST_LUFFY_BAD_NMODE2"] == engine.STATUS_LUFFY_BAD_NMODE2
    assert d["VC_ST_LSMOOTH_BAD_IDX"] == engine.STATUS_LSMOOTH_INDEX
    assert d["VC_ST_SETUP_HOUR_OOB"] == engine.STATUS_SETUP_HOUR
    from voacap_gpu.cuda import driver
    assert driver.SETUP_STATUS_BAD_HOUR == engine.STATUS_SETUP_HOUR
    assert engine.STATUS_CPU_UNMAPPED < 100     # cannot collide


def test_old_local_codes_are_gone_from_the_kernel_surface():
    """Nothing the batch kernels return is a bare 1..99 any more."""
    doc = _documented()
    assert all(c >= 100 for c in doc if c > 0)


@pytest.mark.skipif(not __import__("tests.cuda.harness",
                                   fromlist=["_HAVE_CUDA"])._HAVE_CUDA,
                    reason="no CUDA device")
def test_kernel_returns_the_table_codes():
    """The gate's refusal workloads, through the real kernels: every code
    that comes back is in the table, and the expected ones do."""
    from tests.cuda import test_bitexact_gate as G
    seen = {}
    for name in ("refusals", "hourguard"):
        seen.update(G.status_histogram(name))
    doc = set(_documented()) | {0}
    assert set(seen) <= doc, seen
    for want in (engine.STATUS_GENOIS_GAIN_OOB,
                 engine.STATUS_GENOIS_AREA_COVERAGE,
                 engine.STATUS_REGMOD_AREA_COVERAGE,
                 engine.STATUS_SETTXR_AREA_COVERAGE,
                 engine.STATUS_SETUP_HOUR):
        assert seen.get(want, 0) > 0, (want, seen)
    assert not np.isin(list(seen), np.arange(1, 100)).any()
