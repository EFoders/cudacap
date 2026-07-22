#!/bin/bash
# run_ionomap_dumps.sh — generate the Module-2 tier-B dumps in WSL:
# VIRTIM (REDMAP load + SSN blend + diurnal Fourier -> AB(318)) and
# VERSY (geographic series -> GAMMA(6)).
#
# Unlike the Module-1 dumps, these link real voacapw routines that need
# the module files and the full production FFLAGS from
# src/voacapw/Makefile.am (-cpp, -ffixed-line-length-none,
# -fno-backslash, -fno-sign-zero) on top of the oracle configure flags.
# REDMAP reads $HOME/itshfbc/coeffs at runtime (build_oracle.sh output).
#
# Usage (from Windows, repo root):  wsl bash tools/oracle/dumps/run_ionomap_dumps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VSRC="$REPO_ROOT/reference/voacapl/src/voacapw"
MODS="$REPO_ROOT/reference/voacapl/src/modules"

WORK="$HOME/voacapl-oracle/dumps/ionomap"
mkdir -p "$WORK"

[[ -d "$HOME/itshfbc/coeffs" ]] || { echo "~/itshfbc missing — run build_oracle.sh" >&2; exit 1; }

python3 "$SCRIPT_DIR/gen_ionomap_cases.py" "$WORK"

FLAGS="-g -O2 -std=legacy -fallow-argument-mismatch -cpp -w -fno-backslash -ffixed-line-length-none -fno-sign-zero"

cd "$WORK"
for name in virtim versy; do
    case $name in
        virtim) srcs=("$VSRC/redmap.for" "$VSRC/virtim.for");;
        versy)  srcs=("$VSRC/redmap.for" "$VSRC/virtim.for" "$VSRC/versy.for");;
    esac
    # module sources must precede their users
    # shellcheck disable=SC2086
    gfortran $FLAGS -o "dump_$name" \
        "$MODS/crun_directory.f95" "$MODS/voacapl_defs.for" \
        "$SCRIPT_DIR/dump_$name.f" "${srcs[@]}"
    "./dump_$name"
    DEST="$REPO_ROOT/golden/dumps/$name"
    mkdir -p "$DEST"
    cp "${name}_cases.csv" "$DEST/cases.csv"
    cp "${name}_dump.csv" "$DEST/expected.csv"
    { gfortran --version | head -1
      echo "flags: $FLAGS"
      echo "sources: dump_$name.f ${srcs[*]#"$REPO_ROOT/"} + modules"
      echo "coeffs: \$HOME/itshfbc/coeffs (w-twins; byte-identical to golden/coeffs)"
      date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
    } > "$DEST/META.txt"
    echo "$name: $(($(wc -l <"$DEST/expected.csv") - 1)) cases -> golden/dumps/$name/"
done
