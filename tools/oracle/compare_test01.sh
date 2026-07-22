#!/bin/bash
# compare_test01.sh — detailed comparison of the WSL oracle's voacapx.out
# against the upstream tests/p2p/test01.out fixture (which provenance shows
# was generated on Windows: backslash antenna paths, no "L" version tag).
# Advisory only: the WSL-built voacapl is the project oracle; this fixture
# is a cross-check from a different build lineage.
set -euo pipefail

OUT="$HOME/itshfbc/run/voacapx.out"
REF="$HOME/voacapl-oracle/build/tests/p2p/test01.out"

# normalize CRLF and trailing whitespace — both vary by platform/compiler
# runtime, neither is a value difference
norm() { tr -d '\r' <"$1" | sed 's/[ \t]*$//'; }

echo "total output lines: $(norm "$OUT" | wc -l)"
echo "differing lines (whitespace-normalized): $(diff <(norm "$OUT") <(norm "$REF") | grep -c '^[<>]' || true)"
echo
echo "=== differences excluding known-cosmetic lines ==="
diff <(norm "$OUT") <(norm "$REF") \
  | grep '^[<>]' \
  | grep -vE 'sample|VOACAP|TANGIER|PAGE' \
  || echo "(none — all remaining differences are cosmetic)"
