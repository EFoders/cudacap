#!/usr/bin/env python3
"""Deterministic case grid for the REGMOD dump.

REGMOD reads scalars out of six commons and never interpolates an upstream
table, so unlike FINDF/FDIST there is nothing here that needs the real chain
run ahead of it -- every input is injected directly, which buys precise
control over the branches instead.  What has to be covered:

* **D-E vs F layer**, selected by the sign of FI(1,K) - FVMOD: a mode below
  the E critical frequency takes the D-E path with its collision-frequency
  term and E-bending correction, above it the F path.
* **The collision-frequency sub-branch** inside D-E, on HTLOSS - HTMOD.
* **Sporadic-E obscuration**, skipped entirely when FS(2,K) <= 0.
* **The low-MUFday extra loss**, which only fires below MUFday 1e-4 and is
  itself clamped to the 1e-4..1e-7 decade range.
* **The 1e-6 probability floors** on P and PF.
* **Hop count**: 1 hop (where the ground-loss term vanishes) and more than 2
  (where the obscuration hop count clamps but the others do not).
* **Unused mode slots** (HPMOD <= 0) and slot 7, which REGMOD never fills.
* **No transmitter antenna in range**, which makes TGAIN 0 and sends PWRDB to
  its 30 dBW default.

Column order is fixed and shared with dump_regmod.f and test_regmod.py.

stdlib-only.  Usage: gen_regmod_cases.py <outdir>
"""

import pathlib
import sys

SEED = 20260722
RZ = 6370.0

# --- column order, mirrored in dump_regmod.f and test_regmod.py -------------
COLUMNS = (
    ["jmode", "freq", "gcdkm", "ghopkm", "km",
     "fi1", "gyz", "fs1", "fs2", "hs", "ifob13",
     "acav", "htloss", "xnuz", "hnu", "xve", "afe", "bfe", "asm", "dsl",
     "dsu", "rcnse"]
    + [f"sigl{i}" for i in range(1, 5)]
    + [f"sigu{i}" for i in range(1, 5)]
    + [f"ymuf{i}" for i in range(1, 5)]
    + [f"yfot{i}" for i in range(1, 5)]
    + [f"yhpf{i}" for i in range(1, 5)]
    + [f"hpmuf{i}" for i in range(1, 5)]
    + [f"htmuf{i}" for i in range(1, 5)]
    + [f"fvmuf{i}" for i in range(1, 5)]
    + [f"delmod{i}" for i in range(1, 7)]
    + [f"hpmod{i}" for i in range(1, 7)]
    + [f"htmod{i}" for i in range(1, 7)]
    + [f"fvmod{i}" for i in range(1, 7)]
    + [f"itmod{i}" for i in range(1, 7)]
    + [f"afmod{i}" for i in range(1, 7)]
    + [f"sigpat{i}" for i in range(1, 6)]
    + [f"epspat{i}" for i in range(1, 6)]
    + ["pwrdba", "txlo", "txhi"]
)

GROUNDS = [(5.0, 80.0), (0.01, 30.0), (0.03, 30.0), (0.005, 15.0),
           (0.002, 10.0)]


def lcg(seed):
    state = seed

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (state >> 11) / 2**53 * (hi - lo)

    return rnd


def f(v) -> str:
    return f"{float(v):.9g}"


def case(**kw) -> str:
    """Build one row from defaults overridden by keyword."""
    d = dict(DEFAULTS)
    d.update(kw)
    return ",".join(f(d[c]) for c in COLUMNS)


DEFAULTS = {
    "jmode": 1, "freq": 10.0, "gcdkm": 3000.0, "ghopkm": 1500.0, "km": 1,
    "fi1": 3.0, "gyz": 1.2, "fs1": 4.0, "fs2": 5.0, "hs": 105.0,
    "ifob13": 2500, "acav": 0.12, "htloss": 90.0, "xnuz": 12.0, "hnu": 8.0,
    "xve": 1.0, "afe": 0.5, "bfe": 1.5, "asm": 3.0, "dsl": 8.0, "dsu": 9.0,
    "rcnse": -145.0, "pwrdba": 40.0, "txlo": 2.0, "txhi": 30.0,
}
for _i in range(1, 5):
    DEFAULTS[f"sigl{_i}"] = 0.15
    DEFAULTS[f"sigu{_i}"] = 0.20
    DEFAULTS[f"ymuf{_i}"] = 14.0
    DEFAULTS[f"yfot{_i}"] = 11.0
    DEFAULTS[f"yhpf{_i}"] = 17.0
    DEFAULTS[f"hpmuf{_i}"] = 300.0
    DEFAULTS[f"htmuf{_i}"] = 250.0
    DEFAULTS[f"fvmuf{_i}"] = 8.0
for _i in range(1, 7):
    DEFAULTS[f"delmod{_i}"] = 5.0 + 4.0 * _i
    DEFAULTS[f"hpmod{_i}"] = 250.0 + 20.0 * _i
    DEFAULTS[f"htmod{_i}"] = 200.0 + 15.0 * _i
    DEFAULTS[f"fvmod{_i}"] = 6.0
    DEFAULTS[f"itmod{_i}"] = 3
    DEFAULTS[f"afmod{_i}"] = 0.4
for _i in range(1, 6):
    DEFAULTS[f"sigpat{_i}"] = GROUNDS[_i - 1][0]
    DEFAULTS[f"epspat{_i}"] = GROUNDS[_i - 1][1]


def cases():
    rows = []
    # --- layer branch: every mode D-E (fvmod < fi1), every mode F, and mixed
    rows.append(case(fi1=8.0, **{f"fvmod{i}": 4.0 for i in range(1, 7)}))
    rows.append(case(fi1=2.0, **{f"fvmod{i}": 9.0 for i in range(1, 7)}))
    rows.append(case(fi1=5.0, **{f"fvmod{i}": 3.0 + 1.0 * i
                                 for i in range(1, 7)}))
    # exactly on the boundary -> the arithmetic IF takes the D-E arm
    rows.append(case(fi1=6.0, **{f"fvmod{i}": 6.0 for i in range(1, 7)}))
    # --- D-E collision-frequency sub-branch, both arms and the seam
    for htl in (60.0, 90.0, 120.0, 215.0):
        rows.append(case(fi1=8.0, htloss=htl,
                         **{f"fvmod{i}": 4.0 for i in range(1, 7)}))
    # --- sporadic-E obscuration on and off (F modes only reach it)
    for fs2 in (0.0, -1.0, 2.0, 9.0, 25.0):
        rows.append(case(fi1=2.0, fs2=fs2,
                         **{f"fvmod{i}": 9.0 for i in range(1, 7)}))
    # FSDEAD clamps at 3 MHz; straddle it
    for ifob in (500, 2999, 3000, 3001, 7000):
        rows.append(case(fi1=2.0, ifob13=ifob, freq=2.5,
                         **{f"fvmod{i}": 9.0 for i in range(1, 7)}))
    # --- hop count: 1 (ground term vanishes), 2, and past the obscuration clamp
    for nhop in (1, 2, 3, 5, 8):
        rows.append(case(gcdkm=1200.0 * nhop, ghopkm=1200.0))
    # --- unused mode slots
    rows.append(case(**{f"hpmod{i}": -1.0 for i in range(1, 7)}))
    rows.append(case(hpmod1=-1.0, hpmod3=0.0, hpmod5=-250.0))
    # --- low MUFday: frequency far above the MUF drives the extra-loss branch
    for freq in (18.0, 22.0, 26.0, 30.0):
        rows.append(case(freq=freq, **{f"ymuf{i}": 6.0 for i in range(1, 5)},
                         **{f"fvmuf{i}": 4.0 for i in range(1, 5)}))
    # --- sample-area count for the ground-loss average
    for km in (1, 2, 3, 4, 5):
        rows.append(case(km=km))
    # --- ionosphere selection K -> sample area L = 1, 3, 5
    for jmode in (1, 2, 3):
        rows.append(case(jmode=jmode, km=5))
    # --- layer index ISMOD picks a different /MUFS/ entry per mode
    rows.append(case(**{f"itmod{i}": 1 + (i - 1) % 4 for i in range(1, 7)}))
    # --- no transmitter in range: TGAIN 0 and PWRDB back to its 30 dBW default
    rows.append(case(freq=25.0, txlo=2.0, txhi=20.0))
    rows.append(case(freq=3.0, txlo=8.0, txhi=30.0))
    # --- take-off angle at and past the 89.99 degree clamp
    rows.append(case(**{f"delmod{i}": 89.0 + 0.5 * i for i in range(1, 7)}))
    rows.append(case(**{f"delmod{i}": 0.0 for i in range(1, 7)}))
    # --- random fill
    rnd = lcg(SEED)
    for _ in range(220):
        nhop = 1 + int(rnd(0.0, 6.0)) % 6
        gcdkm = rnd(600.0, 3200.0) * nhop
        fi1 = rnd(1.0, 6.0)
        kw = {
            "jmode": 1 + int(rnd(0.0, 3.0)) % 3,
            "freq": rnd(2.0, 30.0),
            "gcdkm": gcdkm,
            "ghopkm": gcdkm / nhop,
            "km": 1 + int(rnd(0.0, 5.0)) % 5,
            "fi1": fi1,
            "gyz": rnd(0.5, 1.6),
            "fs1": rnd(1.0, 8.0),
            "fs2": rnd(-1.0, 12.0),
            "hs": rnd(95.0, 115.0),
            "ifob13": int(rnd(300.0, 8000.0)),
            "acav": rnd(0.01, 0.6),
            "htloss": rnd(60.0, 130.0),
            "xnuz": rnd(5.0, 20.0),
            "hnu": rnd(4.0, 14.0),
            "xve": rnd(0.5, 1.5),
            "afe": rnd(-1.0, 2.0),
            "bfe": rnd(0.2, 3.0),
            "asm": rnd(0.0, 8.0),
            "dsl": rnd(4.0, 14.0),
            "dsu": rnd(4.0, 14.0),
            "rcnse": rnd(-165.0, -120.0),
            "pwrdba": rnd(20.0, 60.0),
        }
        for i in range(1, 5):
            ymuf = rnd(6.0, 26.0)
            kw[f"sigl{i}"] = rnd(0.08, 0.30)
            kw[f"sigu{i}"] = rnd(0.08, 0.30)
            kw[f"ymuf{i}"] = ymuf
            kw[f"yfot{i}"] = ymuf * rnd(0.70, 0.95)
            kw[f"yhpf{i}"] = ymuf * rnd(1.05, 1.35)
            kw[f"hpmuf{i}"] = rnd(180.0, 420.0)
            kw[f"htmuf{i}"] = rnd(100.0, 360.0)
            kw[f"fvmuf{i}"] = rnd(2.0, 16.0)
        for i in range(1, 7):
            # a third of the slots unused, so the skip path stays exercised
            unused = rnd(0.0, 1.0) < 0.33
            kw[f"hpmod{i}"] = -1.0 if unused else rnd(100.0, 450.0)
            kw[f"delmod{i}"] = rnd(0.0, 45.0)
            kw[f"htmod{i}"] = rnd(90.0, 400.0)
            kw[f"fvmod{i}"] = fi1 * rnd(0.4, 2.5)
            kw[f"itmod{i}"] = 1 + int(rnd(0.0, 4.0)) % 4
            kw[f"afmod{i}"] = rnd(0.0, 2.0)
        for i in range(1, 6):
            sigma, er = GROUNDS[int(rnd(0.0, 5.0)) % 5]
            kw[f"sigpat{i}"] = sigma * rnd(0.5, 2.0)
            kw[f"epspat{i}"] = er * rnd(0.8, 1.2)
        rows.append(case(**kw))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "regmod_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"regmod: {len(rows)} cases, {len(COLUMNS)} columns")


if __name__ == "__main__":
    main()
