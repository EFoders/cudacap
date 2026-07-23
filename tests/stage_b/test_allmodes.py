"""Stage B batched ALLMODES vs the SAME tier-B oracle dump Stage A uses.

MEASURED: ALL BIT-EXACT, on both devices, over the whole dump run eight ways
at once -- 249 lockstep calls x 8 lanes x (22 arrays x 20 slots + NMODE x 20 +
/ZON/ HP x 7 + NMMOD + FREQ), 0 mismatches.  That is the only acceptable
result: ALLMODES contains no arithmetic at all, only assignment and index
selection, so any drift would be a bookkeeping error and not a numerical one.

WHAT "ONE BATCH" MEANS HERE.  The dump is a CALL SEQUENCE, not 228 independent
cases: /allMODE/ accumulates across calls, NMMOD is both the running count and
the write cursor, and the driver never clears the common between records.
Loading the records as 228 parallel lanes would therefore test nothing.  So
the batch is EIGHT COPIES OF THE WHOLE SEQUENCE, STAGGERED: lane j starts the
228-record sequence ``OFFSETS[j]`` calls late and is inactive until then.  At
any one call the eight lanes are consequently at eight different records --
different IFLG arms, different cursors, different /ZON/ contents, some of them
not making the call at all -- and every lane is checked against the record it
is on.  A batched port that quietly shared a cursor passes an in-phase batch
and fails this one.

That staggering is not artificial, either: it is what a real batched driver
looks like.  LUFFY calls ALLMODES once per hop over ``ihsrt..ihstp``, bounds
that come out of the path geometry, so no two grid points make the same number
of calls.
"""

import numpy as np
import pytest
import torch

from voacap_gpu.stage_a.allmodestate import FIELDS, NSLOT
from voacap_gpu.stage_b.allmodes import (AllModesOverflow, allmodes_batch,
                                         new_allmode_state, new_zon_state)

from tests.stage_a.dumputil import load_dump

DUMP = load_dump("allmodes")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# /ZON/ fields the case file carries, in order (gen_allmodes_cases.py)
ZON_FIELDS = ("abps", "crel", "fldst", "hn", "hp", "prob", "rely", "rgain",
              "sigpow", "sn", "spro", "tgain", "timed", "tloss", "b", "fslos",
              "grlos", "adv", "obf", "tllow", "tlhgh", "eff", "nmode")

_O_ZNMODE = 22 * NSLOT          # 22 arrays x 20 come first
_O_ZONHP = _O_ZNMODE + NSLOT    # the /ZON/ HP the IFLG=1 arm writes back
_O_NMMOD = _O_ZONHP + 7
_O_FREQ = _O_NMMOD + 1

# Coprime-ish gaps so the lanes stay out of phase for the whole run rather
# than falling back into step after the first reset.
OFFSETS = (0, 1, 2, 3, 5, 8, 13, 21)


def _dump_arrays():
    """The dump as numpy: IFLG, FVAL, the /ZON/ inputs and the expectations."""
    _, inputs, rows = DUMP
    nrec = len(rows)
    iflg = np.array([int(float(r[0])) for r in inputs])
    fval = np.array([np.float32(r[1]) for r in inputs], dtype=np.float32)
    zin = np.array([[np.float32(v) for v in r[2:2 + 23 * 7]] for r in inputs],
                   dtype=np.float32).reshape(nrec, 23, 7)
    exp = np.array([[np.float32(v) for v in r] for r in rows],
                   dtype=np.float32)
    return nrec, iflg, fval, zin, exp


def _replay(offsets, device):
    """Run the whole dump on len(offsets) staggered lanes.

    Yields ``(step, record_index, active, allm, zon)`` after every call, with
    the state as numpy so the caller can compare it without holding a device
    reference that the next call mutates.
    """
    nrec, iflg, fval, zin, _ = _dump_arrays()
    n = len(offsets)
    allm = new_allmode_state(n, device)
    zon = new_zon_state(n, device)
    for step in range(nrec + max(offsets)):
        rec = np.array([step - o for o in offsets])
        act = (rec >= 0) & (rec < nrec)
        safe = np.clip(rec, 0, nrec - 1)        # inactive lanes read anything
        for j, name in enumerate(ZON_FIELDS):
            col = torch.as_tensor(np.ascontiguousarray(zin[safe, j, :].T))
            arr = zon[name]
            arr[1:8] = (col.round().to(torch.int32) if name == "nmode"
                        else col).to(device)
        allmodes_batch(
            zon, allm,
            torch.as_tensor(iflg[safe], dtype=torch.long, device=device),
            torch.as_tensor(fval[safe], device=device),
            torch.as_tensor(act, device=device))
        snap = {k: v.cpu().numpy().copy() for k, v in allm.items()}
        yield step, rec, act, snap, zon["hp"].cpu().numpy().copy()


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_allmodes_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_allmodes_batch_vs_oracle(device):
    """Every lane, every call, every slot -- bit-exact or it is a bug."""
    nrec, iflg, _, _, exp = _dump_arrays()
    seen = {0: 0, 1: 0, 2: 0, 3: 0}
    peak = 0
    mixed = idle = 0
    total = 0
    for step, rec, act, allm, zonhp in _replay(OFFSETS, device):
        safe = np.clip(rec, 0, nrec - 1)
        e = exp[safe]                            # (N, 469)
        live = act[None, :]
        for j, name in enumerate(FIELDS):
            got = allm[name][1:NSLOT + 1]
            want = e[:, j * NSLOT:(j + 1) * NSLOT].T
            bad = live & (got != want)
            assert not bad.any(), (
                f"[{device}] step {step} z{name}: "
                f"{np.argwhere(bad)[:4].tolist()} got "
                f"{got[bad][:4]} want {want[bad][:4]}")
            total += int(live.sum()) * NSLOT
        want = np.round(e[:, _O_ZNMODE:_O_ZNMODE + NSLOT]).astype(np.int32).T
        bad = live & (allm["nmode"][1:NSLOT + 1] != want)
        assert not bad.any(), f"[{device}] step {step} zNMODE"
        # the IFLG=1 arm writes HP(2) = HP(1) back into /ZON/
        bad = act[None, :] & (zonhp[1:8] != e[:, _O_ZONHP:_O_ZONHP + 7].T)
        assert not bad.any(), f"[{device}] step {step} /ZON/ HP write-back"
        want_n = np.round(e[:, _O_NMMOD]).astype(np.int64)
        bad = act & (allm["nmmod"] != want_n)
        assert not bad.any(), (
            f"[{device}] step {step} NMMOD got {allm['nmmod'][bad]} "
            f"want {want_n[bad]}")
        bad = act & (allm["freq"] != e[:, _O_FREQ])
        assert not bad.any(), f"[{device}] step {step} FREQ"

        arms = {int(iflg[r]) if 0 <= r < nrec else None
                for r, a in zip(rec, act) if a}
        for arm in arms:
            seen[arm if arm in seen else 3] += 1
        mixed += len(arms) > 1
        idle += int((~act).sum())
        peak = max(peak, int(allm["nmmod"].max()))

    assert all(seen[i] for i in (0, 1, 2, 3)), f"unreached IFLG arm: {seen}"
    assert peak >= 18, f"accumulation only reached slot {peak} of {NSLOT}"
    assert mixed > 100, (
        f"[{device}] only {mixed} calls had lanes on different IFLG arms; "
        "the batch is not diverging and proves nothing about the port")
    assert idle > 100, (
        f"[{device}] only {idle} lane-calls were inactive; the `active` mask "
        "is not being exercised")
    assert total > 800_000


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_a_lane_must_not_depend_on_its_batch_neighbours():
    """The test that matters for a scatter-with-a-cursor routine.

    Every lane's write goes to a row chosen per lane, and lanes with nothing
    to append go to row 0.  If that sink were left dirty -- or if a cursor
    leaked -- a lane's answer would depend on how long the OTHER lanes in the
    batch kept writing.  Running a lane alone has to give the same bits.
    """
    whole = list(_replay(OFFSETS, "cpu"))
    for j, off in enumerate(OFFSETS):
        alone = list(_replay((off,), "cpu"))
        assert len(alone) >= len(whole) - max(OFFSETS)
        for (_, _, _, wa, wh), (_, _, _, sa, sh) in zip(whole, alone):
            for name in list(FIELDS) + ["nmode", "nmmod", "freq"]:
                assert np.array_equal(wa[name][..., j:j + 1], sa[name]), (
                    f"lane {j} (offset {off}) {name} depends on its batch "
                    "neighbours")
            assert np.array_equal(wh[..., j:j + 1], sh)


@pytest.mark.skipif(not torch.cuda.is_available() or DUMP is None,
                    reason="no CUDA")
def test_cuda_matches_cpu():
    """No arithmetic means no libm, so this is bitwise on every field."""
    for a, b in zip(_replay(OFFSETS, "cpu"), _replay(OFFSETS, "cuda")):
        for name in list(FIELDS) + ["nmode", "nmmod", "freq"]:
            assert np.array_equal(a[3][name], b[3][name]), (
                f"step {a[0]} {name} differs between CPU and CUDA")
        assert np.array_equal(a[4], b[4]), f"step {a[0]} /ZON/ HP differs"


@pytest.mark.parametrize("device", DEVICES)
def test_reset_clears_to_the_no_mode_markers(device):
    """A reset must restore all 20 slots, not just the ones in use."""
    allm = new_allmode_state(3, device)
    zon = new_zon_state(3, device)
    zon["hp"][1:8] = 300.0
    allmodes_batch(zon, allm, 0)         # NMMOD starts at 1 (blkdat.for:181)
    allmodes_batch(zon, allm, 3)
    assert torch.equal(allm["nmmod"], torch.full((3,), 6, dtype=torch.long,
                                                 device=device))
    allmodes_batch(zon, allm, 0, 21.5)
    assert bool((allm["nmmod"] == 0).all())
    assert bool((allm["freq"] == 21.5).all())
    assert bool((allm["tloss"][1:] == 99999.0).all())
    assert bool((allm["tllow"][1:] == 999.0).all())
    assert bool((allm["tlhgh"][1:] == 999.0).all())
    assert bool((allm["hp"][1:] == -1.0).all())


@pytest.mark.parametrize("device", DEVICES)
def test_a_per_lane_iflg_picks_a_different_arm_on_every_lane(device):
    """IST/LST are per lane here, unlike Stage A where IFLG is a constant.

    Lane 0 takes slot 1 only, lane 1 the Es pair 4-5, lane 2 the six regular
    slots -- from one call, with one /ZON/.
    """
    allm = new_allmode_state(3, device)
    zon = new_zon_state(3, device)
    zon["hp"][1:8] = 300.0
    for s in range(1, 8):
        zon["tloss"][s] = float(s)
    allmodes_batch(zon, allm, 0)
    allmodes_batch(zon, allm, torch.tensor([1, 2, 3], device=device))
    assert allm["nmmod"].tolist() == [1, 2, 6]
    tl = allm["tloss"].cpu().numpy()
    assert tl[1, 0] == 1.0                       # IFLG=1: slot 1 alone
    assert tl[1, 1] == 4.0 and tl[2, 1] == 5.0   # IFLG=2: the Es pair
    assert list(tl[1:7, 2]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


@pytest.mark.parametrize("device", DEVICES)
def test_iflg_1_reserves_the_next_but_one_slot_and_parks_the_marker(device):
    """The three things in the IFLG=1 arm that look like bugs and are not."""
    allm = new_allmode_state(1, device)
    zon = new_zon_state(1, device)
    zon["hp"][1] = 300.0
    zon["hp"][2] = -7.0                          # about to be overwritten
    zon["b"][2] = 42.0
    zon["nmode"][2] = 3
    allmodes_batch(zon, allm, 0)
    allmodes_batch(zon, allm, 1)
    assert float(zon["hp"][2]) == 300.0, "HP(2) = HP(1) writes back into /ZON/"
    assert float(allm["tloss"][2]) == 32000.0, "the marker goes in slot 2"
    assert int(allm["nmmod"]) == 1, "only slot 1 is appended"
    # NMMOD was 0, so NMODE/B land in slot 2 -- the cursor is at slot 1
    assert float(allm["b"][2]) == 42.0 and int(allm["nmode"][2]) == 3
    assert float(allm["hp"][2]) == -1.0, "the reserved slot keeps its marker"


@pytest.mark.parametrize("device", DEVICES)
def test_an_inactive_lane_is_untouched(device):
    """A lane past its own last hop sits the call out and must not move."""
    allm = new_allmode_state(2, device)
    zon = new_zon_state(2, device)
    zon["hp"][1:8] = 300.0
    zon["tloss"][1:8] = 5.0
    allmodes_batch(zon, allm, 0)
    active = torch.tensor([True, False], device=device)
    allmodes_batch(zon, allm, 3, active=active)
    assert allm["nmmod"].tolist() == [6, 0]
    assert float(allm["tloss"][1, 0]) == 5.0
    assert float(allm["tloss"][1, 1]) == 99999.0, "the idle lane moved"
    # and the reset itself honours the mask
    allmodes_batch(zon, allm, 0, 9.0, active=active)
    assert allm["freq"].tolist() == [9.0, 0.0]


@pytest.mark.parametrize("device", DEVICES)
def test_the_sink_row_is_cleared(device):
    """Row 0 takes every write that must not happen, so it has to be wiped.

    Nothing reads it -- the slots are 1..20 -- but leaving it makes the
    returned arrays depend on batch composition, which is the FDIST bug this
    check exists to rule out.
    """
    allm = new_allmode_state(2, device)
    zon = new_zon_state(2, device)
    zon["hp"][1:8] = -1.0                        # nothing is live anywhere
    zon["tloss"][1:8] = 1234.0
    allmodes_batch(zon, allm, 0)
    allmodes_batch(zon, allm, 3)
    assert int(allm["nmmod"].max()) == 0
    for name in list(FIELDS) + ["nmode"]:
        assert bool((allm[name][0] == 0).all()), f"the sink kept {name}"


@pytest.mark.parametrize("device", DEVICES)
def test_overflow_is_refused_not_wrapped(device):
    """Past 20 slots the Fortran writes into the next common-block variable.

    The batch is planned before it is written, so a refused call must also
    leave the accumulator exactly as it was.
    """
    allm = new_allmode_state(2, device)
    zon = new_zon_state(2, device)
    zon["hp"][1:8] = 300.0
    allmodes_batch(zon, allm, 0)         # NMMOD starts at 1 (blkdat.for:181)
    for _ in range(3):
        allmodes_batch(zon, allm, 3)     # 6 each -> 18
    assert allm["nmmod"].tolist() == [18, 18]
    before = {k: v.clone() for k, v in allm.items()}
    with pytest.raises(AllModesOverflow):
        allmodes_batch(zon, allm, 3)     # would reach 24
    for name, v in before.items():
        assert torch.equal(allm[name], v), f"{name} moved on a refused call"
    # one overflowing lane refuses the whole call -- there is no partial write
    with pytest.raises(AllModesOverflow):
        allmodes_batch(zon, allm, 3,
                       active=torch.tensor([True, False], device=device))
