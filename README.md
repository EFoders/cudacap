# voacap-gpu

A GPU rewrite of VOACAP (Voice of America Coverage Analysis Program) HF
propagation prediction in Python/PyTorch, validated module-by-module against
the original Fortran.

**Status: early development.** See [PLAN.md](PLAN.md) for the full plan,
testing methodology, and module port order.

## Layout

- `reference/` — vendored, read-only upstream sources (never modified):
  - `reference/voacapl/` — Linux/gfortran port of VOACAP (the *oracle* we
    validate against). See `reference/PROVENANCE.md`.
  - `reference/itshfbc/` — the Windows ITSHFBC distribution (data + binaries),
    kept for cross-reference.
- `tools/oracle/` — scripts to build and run the reference oracle in WSL.
- `src/voacap_gpu/` — the new implementation.
- `tests/` — three-tier test suite (golden end-to-end decks, per-module oracle
  dumps, property tests).
- `docs/` — reverse-engineered format notes, precision map.

## Oracle

The ground truth is `voacapl` built with gfortran in WSL:

    wsl bash tools/oracle/build_oracle.sh

This builds the vendored source out-of-tree (in `~/voacapl-oracle` inside
WSL), populates `~/itshfbc`, and verifies the build reproduces the upstream
`tests/p2p/test01.out` fixture.

## License

No license has been chosen yet — all rights reserved for now. The vendored
reference code retains its own notices: original VOACAP is a US Government
(NTIA/ITS) public-domain work; voacapl port modifications are CC0.
