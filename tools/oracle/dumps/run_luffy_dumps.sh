#!/bin/bash
# run_luffy_dumps.sh — tier-A replay capture for LUFFY.
#
# Unlike every other dump here, this one does not drive a routine in
# isolation: LUFFY builds its own ionosphere, so injecting its inputs would
# mean injecting most of the program.  Instead LUFFY is SHIMMED into the real
# voacapl: dump_luffy.f provides a LUFFY that records the entry state, calls
# the real routine under the name LUFFYR, and records the exit state.  Card
# reading, map interpolation, CURMUF and the hour loop all run untouched.
#
# reference/ is never modified.  LUFFYR is made by copying luffy.for into the
# WSL work directory and renaming its SUBROUTINE statement there.
#
# Usage (from Windows, repo root):
#   wsl bash tools/oracle/dumps/run_luffy_dumps.sh [deck ...]
# with deck names (no .dat) from golden/p2p/matrix/decks; default is a
# short-path set the port can actually replay.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD="$HOME/voacapl-oracle/build"
DECKS="$REPO_ROOT/golden/p2p/matrix/decks"

[[ -d "$BUILD" ]] || { echo "oracle build tree missing — run tools/oracle/build_oracle.sh" >&2; exit 1; }

# All three regimes, since all three are now ported.
#
# latok_m01_s100_v20 at 8809 km is in the SHORT list deliberately: METHOD 20
# leaves MSPEC at 0, so it stays short-path despite being past the smoothing
# distance.  The same circuit on METHOD 30 is in the SMOOTH list, so the two
# sit side by side and the method is the only difference between them.
#
# The run CLEARS every CSV in the destination first, so this list has to name
# every deck the replay test expects -- capturing a subset deletes the rest.
SHORT_DECKS="lonpar_m01_s100_v30 lonpar_m07_s15_v30 lonpar_m10_s160_v30
             tanbel_m01_s100_v30 tanbel_m07_s160_v30 tanbel_m01_s100_v25
             fbahel_m01_s100_v30 fbahel_m07_s15_v30 fbahel_m01_s100_v20
             latok_m01_s100_v20"
# Past 10000 km, where HFMUFS forces IPFG 200 outright.
LONG_DECKS="sydstg_m01_s100_v30 sydstg_m07_s160_v30 sydstg_m10_s15_v30
            lonakl_m01_s100_v30 lonakl_m07_s15_v30 lonakl_m01_s100_v30_long"
# Between 7000 and 10000 km on METHOD 30, where LUFFY runs BOTH models and
# blends them.  Still 24 rows: the recycle is INSIDE LUFFY (luffy.for's GO TO
# 83), so the shim sees one entry and one exit per hour and what it captures
# at exit is the blended answer, not either pass on its own.
SMOOTH_DECKS="latok_m01_s100_v30 latok_m07_s15_v30 latok_m10_s160_v30
              toksyd_m01_s100_v30 toksyd_m07_s160_v30"
DEFAULT_DECKS="$SHORT_DECKS $LONG_DECKS $SMOOTH_DECKS"
DECK_LIST=("$@")
# shellcheck disable=SC2206
[[ ${#DECK_LIST[@]} -eq 0 ]] && DECK_LIST=($DEFAULT_DECKS)

WORK="$HOME/voacapl-oracle/dumps/luffy"
rm -rf "$WORK"; mkdir -p "$WORK"

# LUFFYR: the real routine under a name the shim can call.
sed 's/^      SUBROUTINE LUFFY(IPFLAG)/      SUBROUTINE LUFFYR(IPFLAG)/' \
    "$REPO_ROOT/reference/voacapl/src/voacapw/luffy.for" > "$WORK/luffyr.f"
grep -q 'SUBROUTINE LUFFYR' "$WORK/luffyr.f" || { echo "rename failed" >&2; exit 1; }

# Reuse the real link line rather than guessing at it: the program is built
# with -fno-sign-zero, -fno-backslash and -cpp on top of the legacy flags, and
# its object list is explicit (112 objects plus four convenience archives).
LINK=$(grep -m1 -- ' -o voacapl ' "$BUILD/make.log") \
    || { echo "no voacapl link line in make.log" >&2; exit 1; }
FFLAGS_REAL=$(printf '%s' "${LINK%% -o voacapl *}" | sed 's/^gfortran //')

cd "$BUILD/src/voacapw"        # the -I and ../lib paths are relative to here
# shellcheck disable=SC2086
gfortran $FFLAGS_REAL -c -o "$WORK/dump_luffy.o" "$SCRIPT_DIR/dump_luffy.f"
# shellcheck disable=SC2086
gfortran $FFLAGS_REAL -c -o "$WORK/luffyr.o" "$WORK/luffyr.f"
SHIM=${LINK/ luffy.o / $WORK/dump_luffy.o $WORK/luffyr.o }
SHIM=${SHIM/ -o voacapl / -o $WORK/voacapl_shim }
[[ "$SHIM" != "$LINK" ]] || { echo "link line substitution failed" >&2; exit 1; }
eval "$SHIM"
cd "$WORK"

# DEST defaults to the main replay dir, but an ICEPAC (or any other isolated)
# capture can redirect it with LUFFY_DEST so it does NOT clear or overwrite the
# committed 21-deck replay set.  e.g.
#   LUFFY_DEST=golden/dumps/luffy_icepac wsl bash ... tanbel_m07_s160_icepac
DEST="${LUFFY_DEST:+$REPO_ROOT/$LUFFY_DEST}"
DEST="${DEST:-$REPO_ROOT/golden/dumps/luffy}"
# Clear the CSVs but not the directory: OneDrive-backed paths refuse rmdir
# from WSL even where file deletion is allowed.
mkdir -p "$DEST"; rm -f "$DEST"/*.csv
RUN="$HOME/itshfbc/run"
n=0
for name in "${DECK_LIST[@]}"; do
    [[ -f "$DECKS/$name.dat" ]] || { echo "no such deck: $name" >&2; exit 1; }
    rm -f luffy_in.csv luffy_out.csv luffy_ant.csv
    cp "$DECKS/$name.dat" "$RUN/voacapx.dat"
    ./voacapl_shim "$HOME/itshfbc" voacapx.dat voacapx.out >/dev/null
    [[ -s luffy_in.csv ]] || { echo "no capture for $name" >&2; exit 1; }
    cp luffy_in.csv "$DEST/${name}_in.csv"
    cp luffy_out.csv "$DEST/${name}_out.csv"
    cp luffy_ant.csv "$DEST/${name}_ant.csv"
    echo "  $name: $(($(wc -l <luffy_in.csv) - 1)) LUFFY calls"
    n=$((n + 1))
done

{ gfortran --version | head -1
  echo "flags: $FFLAGS_REAL   (taken from the real voacapl link line)"
  echo "method: LUFFY shimmed into the real voacapl (dump_luffy.f provides"
  echo "        LUFFY, the real one is relinked as LUFFYR).  Everything else"
  echo "        -- DECRED, REDMAP, GEOM, SANG, CURMUF, the hour loop -- is"
  echo "        the stock program, so the captured state is what a real deck"
  echo "        produces rather than what a case generator imagines."
  echo "decks: ${DECK_LIST[*]}"
  echo "files: <deck>_in.csv  = LUFFY's entry state, one row per call"
  echo "       <deck>_out.csv = /SON/ + XLUF + JMODE at exit, same rows"
  echo "       <deck>_ant.csv = /cantenna/ + /pantenna/, one row per antenna"
  echo "                        and frequency; captured because OFFAZIM comes"
  echo "                        from ANTCALC, not from the antenna file."
  echo "NOT captured: the coefficient maps (loaded"
  echo "  Python-side by REDMAP, already validated), and /SON/"
  echo "  MODE and MDL, which are CHARACTER and feed only the output tables."
  date -u +"generated: %Y-%m-%dT%H:%M:%SZ"
} > "$DEST/META.txt"
echo "luffy: $n decks -> golden/dumps/luffy/"
