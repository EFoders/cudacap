"""Generate the randomised accuracy-sweep decks (docs/accuracy_sweep.md).

Deterministic: the same ``--seed`` and ``--n`` always write the same decks,
byte for byte, on any CPython 3 (only ``random.Random.random()`` is used, and
every derived value goes through fixed-width card formatting).  Stdlib only,
so it runs in WSL's bare python3 as well as the project venv.

Card layouts are those of tools/oracle/golden_decks/gen_matrix.py (the reader
formats in decred.for): CIRCUIT FORMAT(10X,F5.2,A1,3(F9.2,A1),4X,I5) with the
trailing I5 as NPSL; SYSTEM FORMAT(10X,F5.2,F5.0,F5.2,F5.0,3F5.2) = PWX, NOISE,
AMIND, LUFP, RSN, PMP, DMP; FPROB / FREQUENCY / SUNSPOT 13F5.2; ANTENNA
FORMAT(10X,4I5,F10.3,1X,A21,1X,F5.1,F10.4).

What varies, per deck (see ``draw``):

* TX anywhere on the globe, a third of them deliberately beyond +-60 deg and a
  quarter within +-15 deg; RX at a random bearing and a great-circle distance
  drawn from six equal-weight buckets spanning 100 .. 19,900 km;
* 10% long path (NPSL 1, ``L``);
* month 1..12, SSN 0..200 (integer), METHOD 20 / 25 / 30;
* coefficients CCIR or URSI, FPROB Es multiplier 0.00 (off) or 0.70;
* 1..11 frequencies, 2.00..30.00 MHz (distinct, mostly ascending);
* SYSTEM: noise 140..165 (-dBW/Hz at 3 MHz), min take-off 0.10..10.00 deg,
  required reliability 50..99 %, required SNR 30.0..90.0 dB-Hz, PMP
  0.50..10.00 dB, DMP 0.05..5.00 ms;
* antennas: the isotrope samples/sample.00 at both ends, as in the golden
  decks.  ANTCALC takes the isotrope gain from the TX card's design-frequency
  field and from the RX card's power field (antcalc.for:119,136), so those
  carry TX gain -10..+20 dBi and RX gain -10..+20 dBi; TX power
  0.01..1000 kW (log-uniform); beam headings 0..359.9 (inert for an isotrope,
  but they move the captured OFFAZIM).

Every value stays inside cudacap's validated ranges (voacap_gpu.validate):
frequencies 1..30 MHz, SSN 0..300, hours 1..24, power > 0.

Usage::

    python tools/oracle/sweep/gen_sweep.py --out DIR [--n 2200] [--seed S]

writes DIR/<name>.dat and DIR/MANIFEST.tsv.
"""

import argparse
import math
import pathlib
import random

SEED = 20261115
N_DECKS = 2200
R_EARTH_KM = 6370.0

# great-circle distance buckets (km); each is drawn with equal weight
DIST_BUCKETS = ((100.0, 1000.0), (1000.0, 3000.0), (3000.0, 7000.0),
                (7000.0, 10000.0), (10000.0, 15000.0), (15000.0, 19900.0))

MANIFEST_COLS = ("name", "tlat", "tlon", "rlat", "rlon", "npsl", "dist_km",
                 "month", "ssn", "method", "coeffs", "es", "nf", "freqs",
                 "noise", "amind", "lufp", "rsn", "pmp", "dmp", "tx_gain",
                 "rx_gain", "pwrkw", "tx_beam", "rx_beam")


class Draw:
    """Every draw goes through ``random()`` -- stable across CPython 3.x."""

    def __init__(self, seed):
        self.r = random.Random(seed)

    def u(self, lo, hi):
        return lo + (hi - lo) * self.r.random()

    def i(self, lo, hi):                     # inclusive
        return lo + min(int(self.r.random() * (hi - lo + 1)), hi - lo)

    def pick(self, seq):
        return seq[self.i(0, len(seq) - 1)]

    def p(self, prob):
        return self.r.random() < prob


def destination(lat, lon, bearing, dist_km):
    """Spherical direct problem: the point ``dist_km`` along ``bearing``."""
    p1, l1, th = map(math.radians, (lat, lon, bearing))
    d = dist_km / R_EARTH_KM
    p2 = math.asin(math.sin(p1) * math.cos(d)
                   + math.cos(p1) * math.sin(d) * math.cos(th))
    l2 = l1 + math.atan2(math.sin(th) * math.sin(d) * math.cos(p1),
                         math.cos(d) - math.sin(p1) * math.sin(p2))
    lon2 = (math.degrees(l2) + 180.0) % 360.0 - 180.0
    return math.degrees(p2), lon2


def gc_km(lat1, lon1, lat2, lon2):
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    c = (math.sin(p1) * math.sin(p2)
         + math.cos(p1) * math.cos(p2) * math.cos(l2 - l1))
    return R_EARTH_KM * math.acos(max(-1.0, min(1.0, c)))


def _clip_lat(x):
    return max(-89.5, min(89.5, x))


def draw(g, k):
    """One deck's parameters (a dict of card values, already quantised)."""
    # TX latitude: a third high-latitude, a quarter equatorial, rest area-
    # uniform over the globe
    sel = g.r.random()
    sign = -1.0 if g.p(0.5) else 1.0
    if sel < 1.0 / 3.0:
        tlat = sign * g.u(60.0, 89.5)
    elif sel < 1.0 / 3.0 + 0.25:
        tlat = g.u(-15.0, 15.0)
    else:
        tlat = math.degrees(math.asin(g.u(-1.0, 1.0)))
    tlon = g.u(-180.0, 180.0)
    lo, hi = DIST_BUCKETS[g.i(0, len(DIST_BUCKETS) - 1)]
    dist = (math.exp(g.u(math.log(lo), math.log(hi))) if lo < 1000.0
            else g.u(lo, hi))
    rlat, rlon = destination(tlat, tlon, g.u(0.0, 360.0), dist)
    tlat, rlat = round(_clip_lat(tlat), 2), round(_clip_lat(rlat), 2)
    tlon, rlon = round(tlon, 2), round(rlon, 2)
    tlon = 180.0 if tlon == -180.0 else tlon
    rlon = 180.0 if rlon == -180.0 else rlon
    npsl = 1 if g.p(0.10) else 0

    nf = g.i(1, 11)
    freqs = []
    while len(freqs) < nf:
        f = round(g.u(2.0, 30.0), 2)
        if f not in freqs:
            freqs.append(f)
    if g.p(0.8):
        freqs.sort()

    return dict(
        name=f"sw{k:04d}", tlat=tlat, tlon=tlon, rlat=rlat, rlon=rlon,
        npsl=npsl, dist_km=round(gc_km(tlat, tlon, rlat, rlon), 1),
        month=g.i(1, 12), ssn=g.i(0, 200), method=g.pick((20, 25, 30)),
        coeffs=g.pick(("CCIR", "URSI")), es=g.pick((0.0, 0.70)),
        nf=nf, freqs=freqs,
        noise=g.i(140, 165), amind=round(g.u(0.1, 10.0), 2),
        lufp=g.i(50, 99), rsn=round(g.u(30.0, 90.0), 1),
        pmp=round(g.u(0.5, 10.0), 2), dmp=round(g.u(0.05, 5.0), 2),
        tx_gain=round(g.u(-10.0, 20.0), 3), rx_gain=round(g.u(-10.0, 20.0), 3),
        pwrkw=round(math.exp(g.u(math.log(0.01), math.log(1000.0))), 4),
        tx_beam=round(g.u(0.0, 359.9), 1), rx_beam=round(g.u(0.0, 359.9), 1),
    )


def _ns(x):
    return f"{abs(x):5.2f}", "S" if x < 0 else "N"


def _ew(x):
    return f"{abs(x):9.2f}", "W" if x < 0 else "E"


def deck(p):
    """The deck text, card for card in gen_matrix.py's fixed columns."""
    tla, tns = _ns(p["tlat"])
    tlo, tew = _ew(p["tlon"])
    rla, rns = _ns(p["rlat"])
    rlo, rew = _ew(p["rlon"])
    rla = f"{rla:>9s}"
    sl = "L" if p["npsl"] else "S"
    freqs = "".join(f"{f:5.2f}" for f in p["freqs"])
    freqs += " 0.00" * (11 - len(p["freqs"]))
    lines = [
        "COMMENT   generated by tools/oracle/sweep/gen_sweep.py",
        "LINEMAX      55       number of lines-per-page",
        f"COEFFS    {p['coeffs']}",
        "TIME          1   24    1    1",
        f"MONTH      1994{p['month']:5.2f}",
        f"SUNSPOT   {p['ssn']:4.0f}.",
        f"LABEL     {p['name'] + ' TX':<20.20s}{p['name'] + ' RX':<20.20s}",
        f"CIRCUIT   {tla}{tns}{tlo}{tew}{rla}{rns}{rlo}{rew}  {sl} {p['npsl']:5d}",
        (f"SYSTEM       1.{p['noise']:4d}.{p['amind']:5.2f}{p['lufp']:4d}."
         f"{p['rsn']:5.1f}{p['pmp']:5.2f}{p['dmp']:5.2f}"),
        f"FPROB      1.00 1.00 1.00{p['es']:5.2f}",
        (f"ANTENNA       1    1    2   30{p['tx_gain']:10.3f}"
         f"[samples/sample.00    ]{p['tx_beam']:5.1f}{p['pwrkw']:10.4f}"),
        (f"ANTENNA       2    2    2   30{0.0:10.3f}"
         f"[samples/sample.00    ]{p['rx_beam']:5.1f}{p['rx_gain']:10.4f}"),
        f"FREQUENCY {freqs}",
        f"METHOD    {p['method']:5d}    0",
        "EXECUTE",
        "QUIT",
    ]
    return "\n".join(line.rstrip() for line in lines) + "\n"


def generate(n=N_DECKS, seed=SEED):
    g = Draw(seed)
    return [draw(g, k) for k in range(n)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--n", type=int, default=N_DECKS)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    decks = generate(a.n, a.seed)
    with open(a.out / "MANIFEST.tsv", "w", newline="\n") as f:
        f.write(f"# gen_sweep.py seed={a.seed} n={a.n}\n")
        f.write("\t".join(MANIFEST_COLS) + "\n")
        for p in decks:
            (a.out / f"{p['name']}.dat").write_text(deck(p), newline="\n")
            row = dict(p, freqs=" ".join(f"{x:.2f}" for x in p["freqs"]))
            f.write("\t".join(str(row[c]) for c in MANIFEST_COLS) + "\n")
    print(f"wrote {len(decks)} decks (seed {a.seed}) to {a.out}")


if __name__ == "__main__":
    main()
