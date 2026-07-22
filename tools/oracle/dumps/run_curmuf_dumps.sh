#!/bin/bash
# run_curmuf_dumps.sh — tier-B dump for CURMUF_orig (Module-3 MUF/FOT).
# Links CURMUF + its profile/bending callees + F2DIS + a real REDMAP (for
# the F2D table).  REDMAP needs the modules and production FFLAGS; CURMUF's
# dispatcher `use`s cversn, so that module is linked too.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_curmuf_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"
MODS="$REPO_ROOT/reference/voacapl/src/modules"

WORK="$HOME/voacapl-oracle/dumps/curmuf"
mkdir -p "$WORK"

[[ -d "$HOME/itshfbc/coeffs" ]] || { echo "~/itshfbc missing — run build_oracle.sh" >&2; exit 1; }

python3 "$SCRIPT_DIR/gen_curmuf_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -cpp -w -fno-backslash -ffixed-line-length-none -fno-sign-zero"

srcs=("$VSRC/curmuf.for" "$VSRC/lecden.for" "$VSRC/gethp.for" \
      "$VSRC/bendy.for" "$VSRC/pen.for" "$VSRC/f2dis.for" \
      "$VSRC/redmap.for")

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_curmuf \
    "$MODS/crun_directory.f95" "$MODS/voacapl_defs.for" "$MODS/cversn.f95" \
    "$SCRIPT_DIR/dump_curmuf.f" "${srcs[@]}"
./dump_curmuf

DEST="$REPO_ROOT/golden/dumps/curmuf"
mkdir -p "$DEST"
cp curmuf_cases.csv "$DEST/cases.csv"
cp curmuf_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_curmuf.f + curmuf(orig)/lecden/gethp/bendy/pen/f2dis/redmap + modules"
  echo "inputs: controlled FI/YI/HI/FS + path geometry; REDMAP loads F2D"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "curmuf: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/curmuf/"
