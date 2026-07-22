#!/bin/bash
# run_sigdis_dumps.sh — tier-B dump for SIGDIS_orig (signal distribution +
# absorption).  Links SIGDIS + its callees SYSSY/PRBMUF/XLIN, the ionogram
# chain LECDEN/GENION/GETHP/BENDY/PEN, a real REDMAP (SYS/PERR/F2D) and the
# modules; requires the converted coefficients under ~/itshfbc.
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_sigdis_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"
MODS="$REPO_ROOT/reference/voacapl/src/modules"

WORK="$HOME/voacapl-oracle/dumps/sigdis"
mkdir -p "$WORK"

[[ -d "$HOME/itshfbc/coeffs" ]] || { echo "~/itshfbc missing — run build_oracle.sh" >&2; exit 1; }

python3 "$SCRIPT_DIR/gen_sigdis_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -cpp -w -fno-backslash -ffixed-line-length-none -fno-sign-zero"

srcs=("$VSRC/sigdis.for" "$VSRC/syssy.for" "$VSRC/prbmuf.for" \
      "$VSRC/xlin.for" "$VSRC/lecden.for" "$VSRC/genion.for" \
      "$VSRC/gethp.for" "$VSRC/bendy.for" "$VSRC/pen.for" "$VSRC/redmap.for")

cd "$WORK"
# shellcheck disable=SC2086
gfortran $FLAGS -o dump_sigdis \
    "$MODS/crun_directory.f95" "$MODS/voacapl_defs.for" "$MODS/cversn.f95" \
    "$SCRIPT_DIR/dump_sigdis.f" "${srcs[@]}"
./dump_sigdis

DEST="$REPO_ROOT/golden/dumps/sigdis"
mkdir -p "$DEST"
cp sigdis_cases.csv "$DEST/cases.csv"
cp sigdis_dump.csv "$DEST/expected.csv"
{ gfortran --version | head -1
  echo "flags: $FLAGS"
  echo "sources: dump_sigdis.f + sigdis(orig)/syssy/prbmuf/xlin/lecden/genion/gethp/bendy/pen/redmap + modules"
  echo "inputs: month,ssn,kfx,jmode,gcdkm,iedp + 3 areas FI/YI/HI/GLAT/CLCK + F2/Es YFOT/YHPF/YMUF/SIGL/SIGU; VERSN='...W'"
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "sigdis: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/sigdis/"
