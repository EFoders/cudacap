#!/bin/bash
# run_profile_dumps.sh — tier-B dump for the profile/bending primitives
# LECDEN + GETHP + BENDY + PEN.  These are plain COMMON-block F77 routines
# (no modules, no runtime coeffs), fed controlled FI/YI/HI to isolate them
# from the coefficient chain.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_profile_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"

WORK="$HOME/voacapl-oracle/dumps/profile"
mkdir -p "$WORK"

python3 "$SCRIPT_DIR/gen_profile_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -w -ffixed-line-length-none"

srcs=("$VSRC/lecden.for" "$VSRC/gethp.for" "$VSRC/bendy.for" "$VSRC/pen.for")

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_profile "$SCRIPT_DIR/dump_profile.f" "${srcs[@]}"
./dump_profile

DEST="$REPO_ROOT/golden/dumps/profile"
mkdir -p "$DEST"
cp profile_cases.csv "$DEST/cases.csv"
cp profile_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_profile.f + lecden/gethp/bendy/pen"
  echo "inputs: controlled FI/YI/HI (no coefficient chain)"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "profile: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/profile/"
