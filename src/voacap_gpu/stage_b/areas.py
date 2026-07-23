"""Stage B (PyTorch, batched) sample-area bookkeeping — SETLNG and SELMOD.

A circuit has up to five SAMPLE AREAS along the great-circle path -- the
ionospheres the ray passes through -- and how many were actually computed
depends on the path length, which the Fortran carries as KFX.  The downstream
frequency x hop loop indexes slots the short-path cases never filled, so
SETLNG fills them by copying, and SELMOD picks the one area that controls the
ray-path geometry.

THE AREA AXIS.  Everything here is (5, N) or (layer, 5, N) with the area in
its own axis, rows 0..4 for the Fortran's areas 1..5, following the rest of
Stage B in using 0-based rows for 1-based Fortran subscripts.  N is the lane
count -- grid points -- so a lane is a circuit and each lane carries its own
KFX.  That is the whole reason SETLNG cannot be a slice assignment here: in
the scalar code KFX picks one of three copy patterns for the whole call, and
in a batch all three patterns are live at once.

SO SETLNG BECOMES AN INDEX MAP RATHER THAN A COPY.  Each pattern says, for
every destination area, which area it should read from; expressed that way,
the three patterns are three columns of a lookup and applying one is a gather
along the area axis.  ``setlng_sources`` returns those per-lane indices and
``replicate`` applies them, which is both faster than a masked copy and easier
to check -- the maps ARE the routine, and they are small enough to read.

The two maps differ, and the asymmetry is the Fortran's, not a simplification:
the per-area scalars and layer parameters propagate to all five slots, while
the ionogram (and the reflectrix IFOB with it) only ever reaches area 3.

No arithmetic anywhere, so bit-exactness is by construction; what the tests
pin is the control flow and those index ranges.
"""

import torch

_F32 = torch.float32
NAREA = 5

# [destination area][KFX pattern] -> source area, all 0-based.
# Columns are KFX <= 1, KFX == 2, KFX >= 3.
_SCALAR_MAP = ((0, 0, 0),       # area 1 always reads itself
               (0, 1, 1),
               (0, 2, 2),
               (0, 2, 3),
               (0, 2, 4))
# The ionogram stops at area 3: areas 4 and 5 are never written by SETLNG,
# which is why rows 3 and 4 read themselves in every column.
_IONO_MAP = ((0, 0, 0),
             (0, 1, 1),
             (0, 1, 2),
             (3, 3, 3),
             (4, 4, 4))


def _pattern(kfx):
    """KFX -> 0 (one area), 1 (three areas), 2 (already full)."""
    return torch.where(kfx <= 1, torch.zeros_like(kfx),
                       torch.where(kfx == 2, torch.ones_like(kfx),
                                   torch.full_like(kfx, 2)))


def setlng_sources(kfx):
    """Batched SETLNG, as the two source-index maps it amounts to.

    kfx: (N,) integer sample-area count code.  Returns ``(scalar_src,
    iono_src)``, each a (5, N) long tensor giving, for every destination area,
    the area it takes its value from.  Feed either to ``replicate``.
    """
    col = _pattern(kfx.to(torch.long))
    lut = lambda m: torch.stack(  # noqa: E731 - a table lookup, not a policy
        [torch.tensor(row, dtype=torch.long, device=kfx.device)[col]
         for row in m])
    return lut(_SCALAR_MAP), lut(_IONO_MAP)


def replicate(arr, src):
    """Gather along the area axis: ``out[..., a, n] = arr[..., src[a, n], n]``.

    `arr` is (5, N) or (rows, 5, N); `src` is the (5, N) map.  A destination
    that reads itself is a copy, so an untouched slot stays untouched without
    needing a mask.
    """
    if arr.dim() == 2:
        return torch.gather(arr, 0, src)
    if arr.dim() == 3:
        return torch.gather(arr, 1, src.unsqueeze(0).expand(arr.shape[0], -1,
                                                            -1))
    raise ValueError(
        f"replicate wants (5, N) or (rows, 5, N); got {tuple(arr.shape)}")


def selmod_batch(kfx, fi1_1, fi1_2, fi1_3, fi3_1, fi3_3):
    """Batched SELMOD: which sample area controls the ray path.

    Takes exactly the five ionosphere values the Fortran reads -- the E
    critical at areas 1, 2 and 3 and the F2 critical at areas 1 and 3 -- so
    the caller can hand over slices without materialising the whole table.
    Returns JMODE as a (N,) long tensor holding the FORTRAN 1-based area
    index, because that is what /FRQ/ carries and what the dump compares.

    One area means area 1.  Two means the ends share an F layer (IONSET forced
    them equal), so the higher E critical decides.  Three or more compares
    only the two ENDS, on F2 unless they match within 0.01 MHz.
    """
    kfx = kfx.to(torch.long)
    one = torch.ones_like(kfx)
    delfi = fi3_1 - fi3_3
    tied = delfi.abs() < 0.01
    ends = torch.where(tied,
                       torch.where(fi1_1 > fi1_3, 3 * one, one),
                       torch.where(delfi > 0.0, 3 * one, one))
    return torch.where(
        kfx == 1, one,
        torch.where(kfx == 2, torch.where(fi1_1 > fi1_2, 2 * one, one), ends))
