"""Stage A READANT table parsing vs the tier-B oracle dump.

Every value the parser produces from the three stock sample antenna files is
compared against what gfortran's own F-editing produced from readant.for's
FORMATs: 60 parameters, sample.11's 91-value table, sample.13's 360x91 and
sample.14's 30x(EFF + 91).

These are decimal literals read through fixed-column F edit descriptors, so
there is no transcendental in the path and nothing to interpolate -- the bar
is bit-exact, and anything less means the record layout is being read wrong.
The things that could go wrong are structural: format reversion splitting the
91-value tables across 10 records, T-editing to column 10, BLANK='NULL'
turning "   .000" into 0.0, and blank padding of short records.
"""

import collections
import csv

import numpy as np
import pytest

from voacap_gpu.scalar.readant import read_antenna

from .dumputil import ANTENNAS_DIR, GOLDEN

DUMP = GOLDEN / "readant" / "expected.csv"
# the inputs are the vendored distribution samples, byte-identical to the ones
# the oracle read.  sample.11, sample.13 and sample.14 are copied byte-exact
# into tests/data/antennas (see tests/data/README.md), so none of these tests
# needs the vendored reference/ tree.
SAMPLES = ANTENNAS_DIR
SAMPLES13 = ANTENNAS_DIR

pytestmark = pytest.mark.skipif(
    not DUMP.exists(), reason="oracle dump missing — run "
    "tools/oracle/dumps/run_readant_dumps.sh")

# tag -> sample file, matching run_readant_dumps.sh
FILES = {"s11": "sample.11", "s13": "sample.13", "s14": "sample.14"}


def _expected():
    """Oracle rows grouped as tag -> kind -> {(i, j): value}."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    with open(DUMP, newline="") as f:
        r = csv.reader(f)
        next(r)
        for tag, kind, i, j, value in r:
            out[tag][kind][(int(i), int(j))] = np.float32(value)
    return out


EXPECTED = _expected() if DUMP.exists() else {}


def _check(tag, kind, lookup):
    """Compare every oracle cell of one kind against `lookup(i, j)`."""
    cells = EXPECTED[tag][kind]
    assert cells, f"{tag}/{kind} missing from the dump"
    for (i, j), exp in cells.items():
        got = np.float32(lookup(i, j))
        assert got == exp, (f"{tag} {kind}[{i},{j}]: got {got!r} "
                            f"expected {exp!r}")
    return len(cells)


def _sample(tag):
    return (SAMPLES13 if tag == "s13" else SAMPLES) / FILES[tag]


@pytest.mark.parametrize("tag", [
    t
    for t in sorted(FILES)])
def test_parm_vs_oracle(tag):
    ant = read_antenna(_sample(tag))
    assert _check(tag, "parm", lambda i, j: ant.parm[i - 1]) == 20


def test_type11_table_vs_oracle():
    ant = read_antenna(SAMPLES / FILES["s11"])
    assert ant.jant == 11
    assert _check("s11", "t11", lambda i, j: ant.gain_type11[j - 1]) == 91


def test_type13_table_vs_oracle():
    ant = read_antenna(_sample("s13"))
    assert ant.jant == 13
    n = _check("s13", "t13", lambda i, j: ant.gains360[i - 1, j - 1])
    assert n == 360 * 91


def test_type14_table_vs_oracle():
    ant = read_antenna(SAMPLES / FILES["s14"])
    assert ant.jant == 14
    assert _check("s14", "t14e", lambda i, j: ant.eff30[i - 1]) == 30
    n = _check("s14", "t14g", lambda i, j: ant.gains30[i - 1, j - 1])
    assert n == 30 * 91


def test_p2p_tables_apply_the_max_gain_offset():
    """ANTCALC adds parm(1) to every gain and takes AEFF straight through."""
    ant = read_antenna(SAMPLES / FILES["s14"])
    array, aeff = ant.p2p_tables()
    assert array.shape == (30, 91) and aeff.shape == (30,)
    assert np.array_equal(array, (ant.gains30 + ant.parm[0]).astype(np.float32))
    assert np.array_equal(aeff, ant.eff30)


def test_synthesised_types_are_refused():
    """Types 1-10/12 carry no table; the parser must say so, not return zeros."""
    from voacap_gpu.scalar.readant import UnsupportedAntennaType, parse_antenna
    text = "desc\n 3\n 0.0\n 2\n 10.0\n"
    with pytest.raises(UnsupportedAntennaType):
        parse_antenna(text)
