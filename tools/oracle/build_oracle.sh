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

echo "== verification: reproduce tests/p2p/test01.out =="
cp "$SRC/tests/p2p/test01.dat" "$HOME/itshfbc/run/voacapx.dat"
"$PREFIX/bin/voacapl" "$HOME/itshfbc" voacapx.dat voacapx.out
OUT="$HOME/itshfbc/run/voacapx.out"
REF="$SRC/tests/p2p/test01.out"

if cmp -s "$OUT" "$REF"; then
    echo "PASS: byte-identical to tests/p2p/test01.out"
elif diff -q <(tr -d '\r' <"$OUT") <(tr -d '\r' <"$REF") >/dev/null; then
    echo "PASS (modulo CRLF/LF): line-ending-normalized outputs identical"
else
    echo "MISMATCH vs tests/p2p/test01.out — first differences:"
    diff <(tr -d '\r' <"$OUT") <(tr -d '\r' <"$REF") | head -40
    exit 1
fi

echo "Oracle ready: $PREFIX/bin/voacapl   (data: ~/itshfbc)"
