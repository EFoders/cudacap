# Provenance of vendored reference sources

Both trees below are vendored **read-only**: nothing in `reference/` is ever
modified by this project (this file is the one addition, at the `reference/`
root, outside both trees).

## voacapl/

- Upstream: https://github.com/jawatson/voacapl.git
- Vendored at commit: `c12a98b348c39f36f2659b8b3aa97b86284c8a1d`
  (2026-03-29, "Merge branch 'master' of https://github.com/jawatson/voacapl")
- The clone's inner `.git/` directory was removed so the tree vendors as plain
  files in this repository; to compare against upstream, re-clone and check
  out the commit above.
- Byte fidelity note: the original clone was made on Windows with
  `core.autocrlf=true`, which CRLF-mangled the working tree (breaking
  autotools). The vendored tree was therefore re-extracted **byte-exact** from
  the clone's object database (`git -c core.autocrlf=false archive <commit>`).
  Upstream stores build/source files as LF; the `tests/p2p` fixtures and most
  ITSHFBC-derived data files are CRLF in upstream's own storage. Because a
  Linux Fortran run writes LF, oracle output comparison against the CRLF
  fixtures is done line-ending-normalized; fixtures *we* generate are stored
  exactly as the oracle wrote them.
- Licensing (per its `LICENSE`/`COPYING`): original VOACAP code is a US
  Government (NTIA/ITS) work, not subject to copyright in the US; J.A.
  Watson's port modifications are CC0; `dst2csv.f90`/`dst2ascii.f90` only are
  GPLv3.

## itshfbc/

- The Windows ITSHFBC distribution (VOACAP/ICEPAC/REC533), originally
  distributed by Greg Hand (NTIA/ITS) via http://www.greg-hand.com/hfwin32.html
  — an existing local install copied here for cross-reference (Windows
  binaries, Windows-format coefficient `.BIN` files, antenna patterns, data).
- Same NTIA/ITS public-domain status as above.
- Note: the runtime coefficient binaries here (`coeffs/COEFF*W.BIN`) were
  written by a Windows Fortran compiler and are **not** used by this project;
  the oracle regenerates its own from the ASCII sources shipped inside
  `voacapl/voacapl/itshfbc/coeffs/*.asc` at install time.
