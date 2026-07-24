"""Compute once per unique key, then gather back to lanes.

A Stage B batch is a flat list of LANES, one per (circuit, hour, frequency,
...) combination the driver wants an answer for.  Some routines do not depend
on every axis of that product.  ANOIS1 is the extreme case: it reads the
receiver's latitude, longitude and mean time and nothing else, so on a
100-transmitter area run the same receiver site is asked the same question 100
times and 99 of the answers are thrown away by being recomputed identically.

This module is the whole fix, and it is deliberately three functions long.
The transformation is:

    keys = compose(grid_point_index, hour_index)   # integers, see below
    uniq, inv = unique_inverse(keys)               # U lanes instead of N
    compact  = anois1_batch(..., gmtr[uniq_lane], ...)
    per_lane = expand(compact, inv)                # back to N lanes

``expand`` is a pure index read.  It performs NO ARITHMETIC, so the result is
bit-identical to evaluating every lane -- not "within a ULP", identical -- and
``tests/batched/test_dedupe.py`` asserts exactly that on real ANOIS1 oracle
inputs with lanes duplicated on purpose.  That is the only property that makes
this safe to drop into a chain whose whole value is bit-exactness.

WHY THE KEYS MUST BE INTEGERS.  The correctness of the whole scheme rests on
one implication: *equal key implies equal inputs*.  If that holds, taking one
lane's answer for its whole group is not an approximation at all.  Integer
keys keep it true by construction, because they are the driver's own loop
counters -- grid point index, hour index, frequency index -- and two lanes
carry the same counter only when the driver meant them to be the same
evaluation.

Float keys break it from both ends.  Keyed on a raw latitude, two lanes that
ought to share may differ in the last bit because the value reached them by
different arithmetic, and they silently stop sharing -- harmless, but it
quietly removes the speedup you thought you had.  Keyed on a *rounded*
latitude, which is the natural next thing to try, genuinely different inputs
collide and one lane's answer is substituted for another's -- and nothing
downstream can detect it, because the output is still perfectly
self-consistent.  So this module refuses non-integer keys rather than
documenting a rule someone has to remember.

WHAT IS ACTUALLY SHAREABLE is a property of each routine, not of this module;
see the "KEYED BY" sections in ``anois1.py`` and ``genois.py``.  The short
version: ANOIS1 is keyed by (receiver site, receiver mean time) and is fully
transmitter-independent, while GENOIS is NOT shareable across transmitters --
its galactic-noise test reads the F2 critical at a path control point, and
control points move with the bearing.

No float32 arithmetic happens here at all, so the precision rules in
``stage_b.profile`` have nothing to bite on; ``_k`` is deliberately unused.
"""

import torch

_I = (torch.int8, torch.int16, torch.int32, torch.int64,
      torch.uint8, torch.bool)


def _check_int(t, what):
    if t.dtype not in _I:
        raise TypeError(
            f"{what} has dtype {t.dtype}; dedupe keys must be INTEGER.  A "
            "float key either fails to match lanes that differ in the last "
            "bit, or -- if rounded to make it match -- merges lanes whose "
            "inputs are genuinely different and silently substitutes one "
            "lane's answer for another's.  Key on the driver's own indices "
            "(grid point, hour, frequency), not on the physical values.")


def compose(*keys):
    """Several per-lane integer keys as ONE key column, without packing.

    Returns an (N, K) int64 tensor for :func:`unique_inverse`.  Each argument
    is an (N,) integer tensor -- e.g. ``compose(grid_index, hour_index)``.

    THIS DOES NOT PACK THE KEYS INTO A SINGLE INTEGER, on purpose.  Packing
    (``g * 24 + h``) is faster and is the usual trick, but it needs a bound on
    every field, silently aliases if the bound is wrong, and a wrong alias
    here is the exact failure mode this module exists to prevent -- two
    unrelated lanes merged, with a self-consistent and wrong answer.  A
    lexicographic unique over K columns costs a sort either way; the bound is
    not worth it.
    """
    if not keys:
        raise ValueError("compose() needs at least one key column")
    cols = []
    for i, k in enumerate(keys):
        t = torch.as_tensor(k)
        _check_int(t, f"key column {i}")
        if t.dim() != 1:
            raise ValueError(
                f"key column {i} has shape {tuple(t.shape)}; each column must "
                "be (N,), one entry per lane")
        cols.append(t.to(torch.int64))
    n = cols[0].shape[0]
    if any(c.shape[0] != n for c in cols):
        raise ValueError(
            f"key columns disagree on the lane count: "
            f"{[int(c.shape[0]) for c in cols]}")
    return torch.stack(cols, dim=1)


def unique_inverse(keys):
    """The distinct keys, and the lane -> distinct-key map.

    keys: an (N,) integer tensor, or an (N, K) one from :func:`compose`.

    Returns ``(unique_keys, inverse)`` where ``unique_keys`` has the same
    trailing shape as `keys` with U rows, and ``inverse`` is (N,) int64 with
    ``unique_keys[inverse] == keys``.  U <= N, and N/U is the redundancy the
    caller is about to stop paying for.

    The rows come back SORTED, which is torch's own guarantee and is worth
    leaning on: the compact batch a driver builds from this is a deterministic
    function of the key set alone, so two runs that ask for the same work
    evaluate it in the same order and produce the same bits.  A hash-ordered
    unique would not promise that.

    ``unique_keys`` is the keys themselves, not lane indices.  To pick the
    representative LANE for each group -- which is what you index the inputs
    with -- use :func:`representatives`.
    """
    t = torch.as_tensor(keys)
    _check_int(t, "keys")
    if t.dim() == 1:
        uniq, inv = torch.unique(t, sorted=True, return_inverse=True)
        return uniq, inv.to(torch.int64)
    if t.dim() == 2:
        uniq, inv = torch.unique(t, sorted=True, return_inverse=True, dim=0)
        return uniq, inv.to(torch.int64)
    raise ValueError(
        f"keys has shape {tuple(t.shape)}; expected (N,) or (N, K)")


def representatives(inverse, n_unique=None):
    """One lane index per distinct key: the LOWEST-numbered lane in each group.

    inverse: the (N,) map from :func:`unique_inverse`.  Returns a (U,) int64
    tensor `rep` with ``inverse[rep] == arange(U)``.

    "Lowest-numbered" rather than "whichever the scatter happened to land on".
    Every lane in a group carries identical inputs, so any of them gives the
    same answer -- but only if that premise really holds, and if it ever stops
    holding you want the failure to be reproducible rather than to depend on
    the order a parallel scatter finished in.  ``scatter_reduce`` with ``amin``
    is deterministic; a plain scatter of ``arange`` is not.
    """
    inv = torch.as_tensor(inverse).to(torch.int64)
    u = int(inv.max()) + 1 if n_unique is None else int(n_unique)
    lanes = torch.arange(inv.numel(), dtype=torch.int64, device=inv.device)
    out = torch.full((u,), inv.numel(), dtype=torch.int64, device=inv.device)
    return out.scatter_reduce(0, inv, lanes, reduce="amin")


def take(values, index, dim=0, n_lanes=None):
    """Compact a per-lane input down to the representative lanes.

    values: a tensor, or a dict/tuple/list of tensors, whose lane axis is
    `dim`.  index: the (U,) representative lanes.  Non-tensors and 0-dim
    tensors are returned untouched -- neither can carry a lane axis -- so a
    call site can hand over a mixed bag of arguments.

    `dim` defaults to 0 because Stage B's per-lane INPUTS are (N,).  The
    coefficient maps that accept an optional trailing lane axis -- ANOIS1's
    ``fakp`` is (29, 16, 6, N) in that form -- need ``dim=-1``.  A real area
    run shares one map and never builds that form, so the default is the
    common case, not the general one.

    `n_lanes` PASSES SHARED TENSORS THROUGH.  Give it the batch's lane count
    and any tensor whose extent along `dim` is not N is left alone, which is
    what lets one call handle both of ANOIS1's map forms -- (29,16,6) shared
    and (29,16,6,N) per-lane -- without the caller sorting them out first.
    The obvious hole is a shared tensor that happens to be N long on `dim`;
    with dim=-1 and N=6 a shared ``fakp`` would be compacted as if it were
    per-lane.  Pass shared arrays separately if your N can be that small.
    Without `n_lanes` every tensor is indexed and a mismatch raises.
    """
    def one(t):
        if t.dim() == 0:
            return t
        if n_lanes is not None and t.shape[dim] != int(n_lanes):
            return t
        return torch.index_select(t, dim, index)
    return _map(values, one)


def expand(values, inverse, dim=0):
    """Broadcast a compact result back to every lane.

    values: a tensor, or a dict/tuple/list of tensors, as returned by a
    ``*_batch`` routine run on the compact set.  inverse: the (N,) map from
    :func:`unique_inverse`.

    THIS IS A PURE INDEX READ.  No arithmetic, no accumulation, no dtype
    change -- every returned element is a bit-for-bit copy of an element the
    routine computed.  That is what makes the whole dedupe a work-sharing
    transformation rather than an approximation, and it is why the test file
    can assert equality with ``torch.equal`` instead of a ULP bar.
    """
    idx = torch.as_tensor(inverse).to(torch.int64)

    def one(t):
        if t.dim() == 0:
            return t                    # a scalar was never per-lane
        return torch.index_select(t, dim, idx)
    return _map(values, one)


def _map(values, fn):
    """Apply `fn` to every tensor in a tensor / dict / tuple / list, in place
    of the tensor, leaving non-tensors alone."""
    if isinstance(values, torch.Tensor):
        return fn(values)
    if isinstance(values, dict):
        return {k: _map(v, fn) for k, v in values.items()}
    if isinstance(values, (tuple, list)):
        return type(values)(_map(v, fn) for v in values)
    return values
