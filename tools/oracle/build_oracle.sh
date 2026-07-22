#!/bin/bash
# build_oracle.sh — build the voacapl reference oracle inside WSL.
#
# Builds the vendored reference/voacapl source with gfortran in the WSL
# filesystem (never in-tree: reference/ stays pristine), installs it to
# ~/voacapl-oracle/prefix, populates ~/itshfbc, and verifies the build by
# reproducing tests/p2p/test01.out from test01.dat.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/build_oracle.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC="$REPO_ROOT/reference/voacapl"

WORK="$HOME/voacapl-oracle"
BUILD="$WORK/build"
PREFIX="$WORK/prefix"

echo "== ensuring toolchain (dnf) =="
sudo dnf install -y -q gcc-gfortran make automake autoconf rsync diffutils >/dev/null

echo "== copying source off /mnt/c for build speed =="
rm -rf "$BUILD"
mkdir -p "$BUILD"
rsync -a --exclude='.git' "$SRC/" "$BUILD/"

cd "$BUILD"
echo "== autoreconf =="
autoreconf -fi >autoreconf.log 2>&1 || { tail -30 autoreconf.log; exit 1; }

# -std=legacy / -fallow-argument-mismatch: required to compile 1970s-90s
# Fortran with modern gfortran (>=10). These relax diagnostics only; they do
# not enable value-changing optimizations. No -ffast-math, ever: the oracle
# must stay IEEE-conformant.
LEGACY_FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch"

echo "== configure =="
./configure --prefix="$PREFIX" FFLAGS="$LEGACY_FLAGS" FCFLAGS="$LEGACY_FLAGS" \
    >configure.log 2>&1 || { tail -30 configure.log; exit 1; }

echo "== make =="
make -j"$(nproc)" >make.log 2>&1 || { tail -50 make.log; exit 1; }

echo "== install (also generates coefficient .bin files from .asc) =="
rm -rf "$PREFIX"
make install >install.log 2>&1 || { tail -50 install.log; exit 1; }

echo "== populating ~/itshfbc =="
rm -rf "$HOME/itshfbc"
"$PREFIX/bin/makeitshfbc"

{ gfortran --version | head -1
  echo "flags: $LEGACY_FLAGS"
  date -u +"built: %Y-%m-%dT%H:%M:%SZ"
} > "$WORK/BUILD_INFO.txt"

echo "== verification 1: determinism (two runs must be byte-identical) =="
cp "$SRC/tests/p2p/test01.dat" "$HOME/itshfbc/run/voacapx.dat"
"$PREFIX/bin/voacapl" "$HOME/itshfbc" voacapx.dat voacapx.out
OUT="$HOME/itshfbc/run/voacapx.out"
cp "$OUT" "$WORK/run1.out"
"$PREFIX/bin/voacapl" "$HOME/itshfbc" voacapx.dat voacapx.out
if ! cmp -s "$WORK/run1.out" "$OUT"; then
    echo "FAIL: oracle is non-deterministic between identical runs"
    exit 1
fi
echo "PASS: deterministic"

echo "== verification 2: pinned golden fixture (toolchain drift detector) =="
GOLD="$REPO_ROOT/golden/p2p/test01/expected.out"
if [ -f "$GOLD" ]; then
    if cmp -s "$OUT" "$GOLD"; then
        echo "PASS: byte-identical to pinned golden fixture"
    else
        echo "FAIL: oracle output drifted from the pinned golden fixture"
        echo "(toolchain change? see golden/p2p/test01/META.txt) — first diffs:"
        diff "$GOLD" "$OUT" | head -20
        exit 1
    fi
else
    echo "(no pinned fixture yet — seed it by copying \$HOME/itshfbc/run/voacapx.out"
    echo " to golden/p2p/test01/expected.out and recording BUILD_INFO in META.txt)"
fi

echo "== verification 3 (advisory): upstream tests/p2p/test01.out =="
# That fixture came from a different (Windows) compiler lineage: expect a
# nonzero count, confined to +/-1 in the last printed digit of ~1% of cells
# plus cosmetic path/banner text. See PLAN.md "On bit-exact" and
# tools/oracle/compare_test01.sh for the detailed breakdown.
REF="$SRC/tests/p2p/test01.out"
NDIFF=$(diff <(tr -d '\r' <"$OUT" | sed 's/[ \t]*$//') \
             <(tr -d '\r' <"$REF" | sed 's/[ \t]*$//') | grep -c '^[<>]' || true)
echo "INFO: $NDIFF whitespace-normalized differing lines vs upstream fixture"

echo "Oracle ready: $PREFIX/bin/voacapl   (data: ~/itshfbc)"
