"""The ICEPAC-mode oracle decks: configuration, sporadic E exercised, LUFFY replay.

"ICEPAC" in cudacap is Prop-SL's ICEPAC mode as voacapl runs it: the stock
VOACAP engine with COEFFS URSI and FPROB 1.00 1.00 1.00 0.70 (sporadic E
on).  It is not the real ICEPAC program (the ICED profile model with SSN +
Q index, the closed-source Icepacw.exe); voacapl carries no ICED code.  So
these decks validate cudacap against voacapl in that mode only.

The end-to-end and setup-chain checks for these decks are the shared
``test_engine`` tests (``ICEPAC_DECKS`` is in ``ALL_DECKS``).  This file adds:

* ``test_icepac_deck_config`` -- every deck really is URSI + FPROB(4)=0.70,
  on its card and in the captured entry state (PSC(4)).
* ``test_icepac_es_exercised`` -- the oracle found sporadic E (ESMUF > 0) in
  at least one hour of every deck, and cudacap's setup chain finds it in
  exactly the same hours.  Without this an ICEPAC deck could pass while
  never touching the Es path.  (With FPROB(4)=0.70 the oracle has Es in all
  24 hours of every deck, so this is a floor, not a discriminator.)
* ``test_icepac_es_reaches_son`` -- turning Es off (FPROB(4)=0) moves
  cudacap's /SON/ dbw by far more than the 0.01 dB end-to-end bound
  somewhere on every deck, so the oracle match above is sensitive to the Es
  path rather than passing around it.
* ``test_icepac_replay_vs_oracle`` -- LUFFY fed the CAPTURED entry state
  (the tier-A replay, carry-over circuit as voacapl runs it) reproduces the
  /SON/ within the end-to-end absolute bounds.  This isolates LUFFY's Es
  handling from the host setup chain: on ushrot_m01_s160_v30_ice it passes
  although the end-to-end run is out of bounds (a GEOM conditioning issue,
  see test_engine._OUT_OF_BOUNDS).
"""

import dataclasses

import numpy as np
import pytest

from tests.scalar import test_luffy_replay as tlr
from tests.test_engine import (ICEPAC_DECKDIR, ICEPAC_DECKS, ICEPAC_DUMPS,
                               POLE_DEFAULT, SON_ABS_BOUND, _SON_NUM, F32,
                               _antennas, _deck_inputs, _engine, _rows,
                               build_circuit, predict_circuits, redmap,
                               run_deck)

# Es off must move dbw by at least this much somewhere (10x the e2e bound).
ES_EFFECT_FLOOR_DB = 10 * SON_ABS_BOUND["dbw"]

pytestmark = pytest.mark.skipif(not ICEPAC_DECKS,
                                reason="ICEPAC oracle dumps missing")


def _cards(deck):
    """{card keyword: rest of line} off the deck source, if it is kept."""
    src = ICEPAC_DECKDIR / f"{deck}.dat"
    if not src.exists():
        return None
    out = {}
    for line in src.read_text().splitlines():
        out.setdefault(line[:10].strip(), line[10:])
    return out


@pytest.mark.parametrize("deck", ICEPAC_DECKS)
def test_icepac_deck_config(deck):
    ins = tlr._rows(ICEPAC_DUMPS / f"{deck}_in.csv")
    assert len(ins) == 24
    for v in ins:
        assert (F32(v["psc01"]), F32(v["psc02"]), F32(v["psc03"]),
                F32(v["psc04"])) == (F32(1.0), F32(1.0), F32(1.0), F32(0.70))
    cards = _cards(deck)
    if cards is None:        # tanbel_m07_s160_icepac: deck source not kept
        return
    assert cards["COEFFS"].strip() == "URSI"
    assert cards["FPROB"].split() == ["1.00", "1.00", "1.00", "0.70"]


@pytest.mark.parametrize("deck", ICEPAC_DECKS)
def test_icepac_es_exercised(deck):
    coords, system, method, coeff, freqs, ins = _deck_inputs(deck)
    assert coeff == "URSI"
    v0 = ins[0]
    st = redmap(_engine._COEFFS_DIR, coeff, int(v0["month"]), F32(v0["ssn"]))
    oracle_es, ours_es = set(), set()
    for v in ins:
        it = int(v["it"])
        if F32(v["esmuf"]) > 0:
            oracle_es.add(it)
        circ, _ip = build_circuit(coords[:2], coords[2:4], int(v["month"]),
                                  F32(v["ssn"]), it, freqs, method=method,
                                  system=system, coeff=coeff,
                                  pole=POLE_DEFAULT, npsl=coords[4], iono=st)
        if F32(circ.muf.esmuf[it]) > 0:
            ours_es.add(it)
    print(f"\n{deck}: ESMUF>0 in {len(oracle_es)}/24 h (oracle)")
    assert oracle_es, f"{deck}: the oracle never saw sporadic E"
    assert ours_es == oracle_es, (
        f"{deck}: Es hours differ, only-oracle {sorted(oracle_es - ours_es)} "
        f"only-ours {sorted(ours_es - oracle_es)}")


@pytest.mark.parametrize("deck", ICEPAC_DECKS)
def test_icepac_es_reaches_son(deck):
    """Es on (the deck) vs Es off (FPROB(4)=0): the /SON/ must move."""
    coords, system, method, coeff, freqs, ins = _deck_inputs(deck)
    no_es = dataclasses.replace(system, fprob=system.fprob[:3] + (0.0,))
    cant = _antennas(_rows(ICEPAC_DUMPS / f"{deck}_ant.csv"))
    st = redmap(_engine._COEFFS_DIR, coeff, int(ins[0]["month"]),
                F32(ins[0]["ssn"]))
    circuits, ipflags = [], []
    for v in ins:
        c, ip = build_circuit(coords[:2], coords[2:4], int(v["month"]),
                              F32(v["ssn"]), int(v["it"]), freqs,
                              method=method, system=no_es, cant=cant,
                              coeff=coeff, pole=POLE_DEFAULT, npsl=coords[4],
                              iono=st)
        circuits.append(c)
        ipflags.append(ip)
    predict_circuits(circuits, ipflags, backend="cpu")
    nf = len(freqs)
    off = np.array([np.asarray(c.son.dbw[1:nf + 1], np.float32)
                    for c in circuits])
    on = run_deck(deck, "cpu")[0]["dbw"]       # cached from test_engine
    moved = np.abs(on - off)
    print(f"\n{deck}: Es moves dbw in "
          f"{int((moved > SON_ABS_BOUND['dbw']).sum())}/{moved.size} cells, "
          f"max {moved.max():.2f} dB")
    assert moved.max() > ES_EFFECT_FLOOR_DB, (
        f"{deck}: Es off moves dbw by only {moved.max():.3e} dB")


@pytest.mark.parametrize("deck", ICEPAC_DECKS)
def test_icepac_replay_vs_oracle(deck):
    ins = tlr._rows(ICEPAC_DUMPS / f"{deck}_in.csv")
    outs = tlr._rows(ICEPAC_DUMPS / f"{deck}_out.csv")
    ants = tlr._rows(ICEPAC_DUMPS / f"{deck}_ant.csv")
    c = tlr.Circuit(iono=redmap(tlr.COEFFS_DIR, "URSI", int(ins[0]["month"]),
                                F32(ins[0]["ssn"])))
    c.cant = tlr._antennas(ants)
    c.muf.xluf[:] = F32(-1.0)             # SETOUT's preset, as the replay
    nf = sum(1 for i in range(1, 12) if ins[0][f"frel{i:02d}"] > 0)
    worst = {nm: 0.0 for nm in _SON_NUM}
    for vin, vout in zip(ins, outs):
        tlr._apply(c, vin)
        tlr.luffy(c, int(vin["ipflag"]))
        got = tlr._got(c)
        it = int(vin["it"])
        for j in range(1, nf + 1):
            key = f"nhp{j:02d}"
            assert int(got[key]) == int(vout[key]), f"{deck} {key} h{it}"
            for nm in _SON_NUM:
                key = f"{nm}{j:02d}"
                a = abs(float(got[key]) - float(F32(vout[key])))
                worst[nm] = max(worst[nm], a)
                assert a <= SON_ABS_BOUND[nm], (
                    f"{deck} {key} h{it}: {float(got[key])!r} vs "
                    f"{vout[key]!r}, |d| {a:.3e}")
    print(f"\n{deck} replay worst |d| "
          + ", ".join(f"{nm} {worst[nm]:.1e}" for nm in _SON_NUM))
