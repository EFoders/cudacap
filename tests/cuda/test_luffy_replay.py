"""THE PAYOFF: Stage C's LUFFY kernel vs Stage A's ``luffy()``, 504 threads.

Twenty-one oracle decks x 24 hours = 504 circuit-hours, every one of them a
whole hour of system performance -- SELMOD, the ionosphere build, the twelve
frequencies, the hop sweep or the two-ended long path, ESMOD, RELBIL, SERPRB,
MPATH, and where the distance calls for it the short/long blend -- running as
ONE THREAD EACH, in one launch.  The bar is Stage C's usual one: bit-identity
with Stage A on EVERY FIELD of the whole CircuitState, not just /SON/.

**Hours are not independent, so the entry states have to be manufactured
sequentially.**  Stage A reuses ONE ``Circuit`` per deck across the 24 hours
because Fortran COMMON persists: every deck gives 9 frequencies, so FREL(10),
FREL(11) and FREL(13) are zero and LUFFY skips those columns, leaving the
previous hour's numbers in place; the /DON/ decile triple D10R/D50R/D90R
carries over the same way.  A fresh Circuit per hour would zero them and
disagree with the oracle (``tests/stage_a/test_luffy_replay.py`` documents
this).  So the host walks each deck's 24 hours in order with Stage A --
applying the captured entry state, SNAPSHOTTING the Circuit as the device's
entry state, then running ``luffy`` to get the expected exit state -- which
turns a 24-step chain into 24 INDEPENDENT (entry, exit) pairs.  Those 504
pairs are what the device runs in parallel.

What is being tested here is COMPOSITION.  Every leaf is already proved
bit-exact on its own dump by the other hundred tests in this directory; a
mismatch here is an argument order, a missed state carry or a control-flow
arm, not arithmetic.

The three regimes the replay spans are asserted present, because the driver's
shape differs completely between them and a subset would validate a third of
it: SHORT (IPFLAG 100 with MSPEC 0, the hop sweep), LONG (past 10000 km with
IPFLAG 200, the two-ended model), SMOOTH (MSPEC 121 inside 7000-10000 km, the
label-83 recycle that runs BOTH models and blends them per frequency).
"""

import numpy as np
import pytest

from voacap_gpu.scalar.circuit import Circuit
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.luffy import luffy
from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.cuda.state import RAYS_DTYPE
from voacap_gpu.cuda.driver import (CIRCUIT_DTYPE, circuit_state_sizeof,
                                       launch_luffy, pack_circuit,
                                       shared_antenna_tables,
                                       shared_iono_tables)

from tests.scalar.dumputil import COEFFS_DIR
from tests.scalar.test_luffy_replay import DECKS, _antennas, _apply, _load
from tests.cuda.harness import assert_bitexact, requires_cuda

pytestmark = requires_cuda

NHOUR = 24


def _deck_cases(deck):
    """One deck -> its 24 (entry, expected exit, ipflag) rows.

    ``pack_circuit`` returns a PAIR -- the CircuitState buffer and the /RAYS/
    buffer -- because /RAYS/ travels to the device structure-of-arrays rather
    than inside CircuitState (see ``stage_c/driver.py``).  Both halves are
    carried, and both are compared: the two together are still the whole
    Stage A ``Circuit``, cell for cell.

    The Circuit is built once and reused, exactly as Stage A's replay does, so
    the COMMON persistence between hours is reproduced; the snapshot taken
    after ``_apply`` and before ``luffy`` is what the device gets.
    """
    loaded = _load(deck)
    if loaded is None:
        return None
    ins, outs, ants, _header = loaded
    months = {int(v["month"]) for v in ins}
    ssns = {float(v["ssn"]) for v in ins}
    assert len(months) == 1 and len(ssns) == 1, "one map set per deck"
    coeff = "URSI" if "_ursi" in deck else "CCIR"
    iono = redmap(COEFFS_DIR, coeff, months.pop(), F32(ssns.pop()))
    c = Circuit(iono=iono)
    c.cant = _antennas(ants)
    # SETOUT's preset, once before the hour loop (setout.for:29).
    c.muf.xluf[:] = F32(-1.0)

    rows = []
    for vin in ins:
        _apply(c, vin)
        ipflag = int(vin["ipflag"])
        entry, entry_rays = pack_circuit(c)
        luffy(c, ipflag)                 # Stage A raising here is a hard fail
        exit_, exit_rays = pack_circuit(c)
        rows.append((entry, entry_rays, exit_, exit_rays, ipflag,
                     int(vin["mspec"]), float(vin["gcdkm"])))
    return iono, c.cant, rows


_CACHE = {}


def _replay():
    """Every deck's cases, built once for the whole module."""
    if "v" in _CACHE:
        return _CACHE["v"]
    ionos, cants, entries, exits = [], [], [], []
    entries_rays, exits_rays = [], []
    iono_idx, ant_idx, ipflags, tags, deckof = [], [], [], [], []
    for deck in DECKS:
        got = _deck_cases(deck)
        if got is None:
            continue
        iono, cant, rows = got
        ionos.append(iono)
        cants.append(cant)
        ii, ai = len(ionos) - 1, len(cants) - 1
        for entry, entry_rays, exit_, exit_rays, ipflag, mspec, gcdkm in rows:
            entries.append(entry)
            entries_rays.append(entry_rays)
            exits.append(exit_)
            exits_rays.append(exit_rays)
            ipflags.append(ipflag)
            iono_idx.append(ii)
            ant_idx.append(ai)
            deckof.append(deck)
            if mspec == 121 and gcdkm > 10000.0:
                tags.append("LONG")
            elif mspec == 121 and 7000.0 <= gcdkm <= 10000.0:
                tags.append("SMOOTH")
            elif ipflag in (200, 400):
                tags.append("LONG")
            else:
                tags.append("SHORT")
    v = dict(ionos=ionos, cants=cants,
             entries=np.stack(entries), exits=np.stack(exits),
             entries_rays=np.stack(entries_rays),
             exits_rays=np.stack(exits_rays),
             ipflags=np.array(ipflags, np.int32),
             iono_idx=np.array(iono_idx, np.int32),
             ant_idx=np.array(ant_idx, np.int32),
             tags=tags, decks=deckof)
    _CACHE["v"] = v
    return v


def _fields(dtype, prefix=""):
    """(label, getter path) for every leaf field of the nested dtype."""
    out = []
    for name in dtype.names:
        sub = dtype.fields[name][0]
        if sub.names:
            out.extend(_fields(sub, f"{prefix}{name}."))
        else:
            out.append((f"{prefix}{name}", (prefix + name).split(".")))
    return out


def _get(buf, path):
    for p in path:
        buf = buf[p]
    return buf


@pytest.mark.skipif(not DECKS, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_luffy_dumps.sh")
def test_circuit_state_abi_matches_the_device():
    """The same offset proof the per-struct state ABI carries."""
    assert circuit_state_sizeof() == CIRCUIT_DTYPE.itemsize, (
        f"sizeof(CircuitState) {circuit_state_sizeof()} != "
        f"CIRCUIT_DTYPE.itemsize {CIRCUIT_DTYPE.itemsize}")


@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
def test_luffy_kernel_bit_exact_vs_stage_a():
    v = _replay()
    n = v["entries"].shape[0]
    assert n == len(DECKS) * NHOUR, f"{n} circuit-hours, expected 24 per deck"

    got, got_rays, status = launch_luffy(
        v["entries"], v["entries_rays"], v["ipflags"],
        shared_iono_tables(v["ionos"]),
        v["iono_idx"], shared_antenna_tables(v["cants"]), v["ant_idx"])

    bad = np.flatnonzero(status != 0)
    assert bad.size == 0, (
        "lanes refused: "
        + ", ".join(f"{v['decks'][i]} row {i % NHOUR} status {status[i]}"
                    for i in bad[:10]))

    # CircuitState and the separate SoA /RAYS/ buffer -- together the whole
    # Stage A Circuit, so both are walked field by field.
    checks = [(CIRCUIT_DTYPE, got, v["exits"], ""),
              (RAYS_DTYPE, got_rays, v["exits_rays"], "rays.")]
    # per-struct and overall exact fractions, reported before the assert so a
    # failure says WHERE it starts rather than only that it happened
    per_struct = {}
    failures = []
    for dt, g_all, e_all, prefix in checks:
        for label, path in _fields(dt):
            label = prefix + label
            g = np.asarray(_get(g_all, path))
            e = np.asarray(_get(e_all, path))
            if g.dtype.kind == "f":
                same = g.view(np.uint32) == e.view(np.uint32)
            else:
                same = g == e
            block = label.split(".")[0]
            hit, tot = per_struct.get(block, (0, 0))
            per_struct[block] = (hit + int(same.sum()), tot + int(same.size))
            if not same.all():
                failures.append((label, int((~same).sum()), int(same.size)))
    hits = sum(h for h, _ in per_struct.values())
    tots = sum(t for _, t in per_struct.values())
    print(f"\nStage C LUFFY vs Stage A: {n} circuit-hours, "
          f"{hits}/{tots} cells bit-exact ({100.0 * hits / tots:.4f}%)")
    for block, (h, t) in per_struct.items():
        print(f"  {block:<10s} {h:>10d}/{t:<10d} {100.0 * h / t:8.4f}%")
    if failures:
        head = "; ".join(f"{lab} {bad_}/{tot}" for lab, bad_, tot
                         in failures[:12])
        raise AssertionError(
            f"{len(failures)} field(s) differ from Stage A: {head}")

    # and the unforgiving per-field form, for the identical report shape the
    # rest of Stage C uses
    for dt, g_all, e_all, prefix in checks:
        for label, path in _fields(dt):
            assert_bitexact(np.asarray(_get(g_all, path)),
                            np.asarray(_get(e_all, path)), prefix + label)


@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
def test_the_replay_covers_all_three_regimes():
    """The driver is three different programs; a subset would test one."""
    v = _replay()
    seen = {t: v["tags"].count(t) for t in ("SHORT", "LONG", "SMOOTH")}
    for arm, cnt in seen.items():
        assert cnt, f"no circuit-hour exercises {arm} ({seen})"
    print(f"\nregimes: {seen}")

    # the entry states really do span the arms the driver branches on
    ip = v["ipflags"]
    assert set(ip.tolist()) & {100}, "no short-path system-performance hour"
    assert (ip == 200).any() or "LONG" in v["tags"], "no long-path hour"
