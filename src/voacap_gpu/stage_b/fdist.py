"""Stage B (PyTorch, batched) mode search — FDIST.

Everything batched so far has been elementwise or a table lookup.  This is the
first routine with genuinely irregular control flow: a state machine that
walks the ray table looking for the low- and high-angle ray of each layer
whose ground distance brackets the hop, emitting up to six modes.  Different
grid points take different numbers of steps, stop at different places, and
emit different numbers of modes.

THE TECHNIQUE IS LOCKSTEP EXECUTION.  Every lane carries its own program
counter, table cursor and mode cursor, and one iteration of the host loop
advances every lane by one Fortran statement.  A lane that has finished sits
at the terminal label and its state stops changing; the loop runs until no
lane is live.  That trades wasted work -- a lane that finished early still
rides along -- for the branch divergence disappearing entirely, which is the
right trade when the lane count is large and the step count is bounded.

It IS bounded, and by construction rather than by hope: IH only ever
increases, and the scan gives up once it passes the end of the table, so the
number of times any lane can visit label 145 is at most the table length.
``_MAX_STEPS`` is that bound with room, and the loop raises rather than
silently truncating if it is ever hit -- the same tripwire Stage A carries.

WRITES GO THROUGH ROW 0 AS A SINK.  Only labels 165 and 185 write a mode, and
they write it at the lane's own mode cursor, so the store is a scatter to a
different row per lane.  Lanes that are not writing this step scatter to row
0 instead, which is never read: the mode slots are 1..6.  That keeps the store
unconditional and avoids a masked scatter.

PERFORMANCE: THE STEP IS TINY AND THERE ARE HUNDREDS OF THEM, so the naive
version is bound by kernel launches rather than by arithmetic.  The batch
converges in 130-256 lockstep steps and each step issues about forty small
kernels; eagerly, that pins CUDA at ~300 ms REGARDLESS OF LANE COUNT.

``_run_lockstep`` therefore captures one step into a CUDA GRAPH and replays
it, turning forty launches per step into one.  Against the 30 us/lane scalar
Stage A:

                    eager    graph          graph vs scalar
        180 lanes   299 ms    46 ms   6.6x    8x slower
      3,600 lanes   324 ms    54 ms   6.0x    2x faster
     36,000 lanes   305 ms    54 ms   5.7x   20x faster
    360,000 lanes   310 ms   112 ms   2.8x   96x faster

The graph path is BIT-FOR-BIT identical to the eager one, which is asserted in
the tests -- it is the same kernels in the same order, just launched once.

Note the shape change as well as the speed: eager is flat in the lane count
(launch-bound), graph is flat until 36,000 and then starts to grow (finally
compute-bound, which is the regime it should be in).  The crossover against
scalar drops from ~40,000 lanes to under 2,000, so this is now the right
choice for anything bigger than a handful of circuits rather than for area
grids only.

torch.compile over the step body would be the other lever and is NOT available
here: it needs Triton, which has no Windows build.  Amortising the completion
check was tried before either and moved the number by 1% -- synchronisation
was never the cost.

Precision follows Stage A: float32, with SIN/COS/ATAN carrying the documented
libm floor -- so unlike the profile chain this one cannot be bit-exact against
a different libm, and the test bars accordingly.
"""

import torch

from voacap_gpu.stage_a.constants import D2R, PIO2, R2D, RZ
from voacap_gpu.stage_b.profile import _k

_F32 = torch.float32
NMODE = 6                       # mode slots per area
NRAY = 45                       # the ray table FINDF fills, 1-based
_DONE = 230                     # the terminal label
_MAX_STEPS = 600                # see the module docstring
_EPS = 0.001
_CHECK_EVERY = 16               # graph replays between completion checks


def _run_lockstep(step, pc, buffers, use_cuda_graph):
    """Drive `step` until every lane parks at the terminal label.

    The eager path calls it in a Python loop, which is correct but issues
    ~40 kernel launches per step and is what makes the routine launch-bound.
    The CUDA-graph path records those launches once and replays the lot with
    a single one.

    CAPTURE HAS TO BE FED A COLD STATE.  CUDA graph capture requires a warmup
    on a side stream, and the warmup genuinely RUNS the step -- so it advances
    the very state machine being captured.  Every buffer is therefore cloned
    beforehand and restored both after the warmup and after the capture; skip
    either restore and the answer is silently the result of starting a few
    steps in.
    """
    if not (use_cuda_graph and pc.is_cuda):
        for _ in range(_MAX_STEPS):
            if not bool((pc != _DONE).any()):
                return
            step()
        raise RuntimeError(
            f"FDIST lockstep did not converge in {_MAX_STEPS} steps; IH only "
            "increases, so this means a transition is wrong")

    saved = [b.clone() for b in buffers]

    def _restore():
        for buf, cold in zip(buffers, saved):
            buf.copy_(cold)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    _restore()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    _restore()

    # A lane parked at the terminal label is a no-op apart from touching the
    # scratch sink, so over-replaying is harmless and the check can be rare.
    for _ in range(0, _MAX_STEPS, _CHECK_EVERY):
        for _ in range(_CHECK_EVERY):
            graph.replay()
        if not bool((pc != _DONE).any()):
            return
    raise RuntimeError(
        f"FDIST lockstep did not converge in {_MAX_STEPS} steps; IH only "
        "increases, so this means a transition is wrong")


def _g(arr, idx):
    """arr[idx[lane], lane] for every lane."""
    return torch.gather(arr, 0, idx.unsqueeze(0)).squeeze(0)


def fdist_batch(delfx, hpflx, htflx, gdflx, fvflx, afflx, imode, dmaxkm,
                ghop, freq, amind, use_cuda_graph=True):
    """Batched FDIST.

    The /REFLX/ arrays are (46, N) with row 0 unused, matching the Fortran's
    1-based ray index; ``imode`` is int.  dmaxkm/ghop/freq/amind are (N,).
    Returns ``(delmod, hpmod, htmod, fvmod, afmod, itmod, nummod)`` with the
    mode arrays (7, N) -- again 1-based, row 0 unused -- and nummod (N,) int.
    """
    n = ghop.shape[0]
    dev = ghop.device
    lanes = torch.arange(n, device=dev)
    zero_i = torch.zeros(n, dtype=torch.long, device=dev)


    # Hoisted out of the step: `_k` builds its scalar on the host, and a
    # host-to-device copy is not capturable into a CUDA graph.  They are
    # loop-invariant anyway.
    k_rz, k_half = _k(RZ, ghop), _k(0.5, ghop)
    k_pio2, k_r2d, k_d2r = _k(PIO2, ghop), _k(R2D, ghop), _k(D2R, ghop)

    dhopkm = ghop * k_rz
    delmod = torch.zeros((NMODE + 1, n), dtype=_F32, device=dev)
    hpmod = torch.full((NMODE + 1, n), -1.0, dtype=_F32, device=dev)
    htmod = torch.zeros((NMODE + 1, n), dtype=_F32, device=dev)
    fvmod = torch.zeros((NMODE + 1, n), dtype=_F32, device=dev)
    afmod = torch.zeros((NMODE + 1, n), dtype=_F32, device=dev)
    itmod = torch.full((NMODE + 1, n), 5, dtype=torch.int32, device=dev)
    hpmod[0] = 0.0                                  # the sink row

    pc = torch.where(dhopkm >= dmaxkm,
                     torch.full((n,), _DONE, dtype=torch.long, device=dev),
                     torch.full((n,), 140, dtype=torch.long, device=dev))
    ih = zero_i.clone()
    il = zero_i.clone()
    dth2 = torch.zeros(n, dtype=_F32, device=dev)


    def _step():
        """Advance every lane by one Fortran statement, IN PLACE.

        In place because this is what gets captured into a CUDA graph: a
        replay re-runs the captured kernels against the captured buffer
        addresses, so the state has to live in buffers that never move.
        Rebinding ``pc = torch.where(...)`` would leave the graph updating a
        tensor the Python name no longer refers to.
        """
        live = pc != _DONE
        at = {p: live & (pc == p) for p in
              (140, 145, 160, 165, 170, 175, 176, 180, 185, 215, 220, 305,
               315)}
        nxt = pc.clone()

        # 140: take the next mode slot
        il.copy_(torch.where(at[140], il + 1, il))
        nxt = torch.where(at[140], torch.full_like(pc, 145), nxt)

        # 145: take the next ray; give up past the end or at a dead entry
        ih.copy_(torch.where(at[145], ih + 1, ih))
        ihc = ih.clamp(0, NRAY)                     # keep the gathers in range
        ih1 = (ih + 1).clamp(0, NRAY)
        past_end = at[145] & ((ih > NRAY - 1) | (_g(hpflx, ih1) <= 0))
        d145 = _g(gdflx, ihc) - _g(gdflx, ih1)
        nxt = torch.where(at[145] & past_end, torch.full_like(pc, _DONE), nxt)
        step145 = at[145] & ~past_end
        nxt = torch.where(step145 & (d145 < 0), torch.full_like(pc, 170), nxt)
        nxt = torch.where(step145 & (d145 == 0), torch.full_like(pc, 160), nxt)
        nxt = torch.where(step145 & (d145 > 0), torch.full_like(pc, 215), nxt)

        gd_ih = _g(gdflx, ihc)
        gd_ih1 = _g(gdflx, ih1)

        # 160 / 170 / 175 / 215 / 220: bracket the hop distance
        nxt = torch.where(
            at[160],
            torch.where((dhopkm - gd_ih).abs() <= _EPS,
                        torch.full_like(pc, 145), torch.full_like(pc, 165)),
            nxt)
        d170 = gd_ih - dhopkm
        nxt = torch.where(at[170], torch.where(
            d170 < 0, torch.full_like(pc, 175),
            torch.where(d170 == 0, torch.full_like(pc, 165),
                        torch.full_like(pc, 145))), nxt)
        d175 = dhopkm - gd_ih1
        nxt = torch.where(at[175], torch.where(
            d175 < 0, torch.full_like(pc, 180),
            torch.where(d175 == 0, torch.full_like(pc, 176),
                        torch.full_like(pc, 145))), nxt)
        nxt = torch.where(at[215], torch.where(
            d170 < 0, torch.full_like(pc, 145),
            torch.where(d170 == 0, torch.full_like(pc, 165),
                        torch.full_like(pc, 220))), nxt)
        nxt = torch.where(at[220], torch.where(
            d175 < 0, torch.full_like(pc, 145),
            torch.where(d175 == 0, torch.full_like(pc, 176),
                        torch.full_like(pc, 180))), nxt)

        # 176: the hop lands exactly on the next ray -- step onto it
        ih.copy_(torch.where(at[176], ih + 1, ih))
        nxt = torch.where(at[176], torch.full_like(pc, 165), nxt)

        # 180: too narrow a bracket to interpolate across
        dth2.copy_(torch.where(at[180], gd_ih1 - gd_ih, dth2))
        nxt = torch.where(at[180], torch.where(
            dth2.abs() - _EPS <= 0, torch.full_like(pc, 165),
            torch.full_like(pc, 185)), nxt)

        # ---- the two writes ------------------------------------------------
        ihc = ih.clamp(0, NRAY)                     # 176 may have moved it
        ih1 = (ih + 1).clamp(0, NRAY)
        w165, w185 = at[165], at[185]
        # 165: take the table entry as it stands
        v_del, v_hp = _g(delfx, ihc), _g(hpflx, ihc)
        v_ht, v_af, v_fv = _g(htflx, ihc), _g(afflx, ihc), _g(fvflx, ihc)
        v_it = _g(imode, ihc)
        # 185: interpolate, then re-derive the angle from exact geometry so
        # the mode is self-consistent rather than table-rounded
        gd_ih, gd_ih1 = _g(gdflx, ihc), _g(gdflx, ih1)
        dth = dhopkm - gd_ih
        safe = torch.where(dth2 == 0, torch.ones_like(dth2), dth2)
        thet = k_half * dhopkm / k_rz
        hp1, hp2 = _g(hpflx, ihc), _g(hpflx, ih1)
        ht1, ht2 = _g(htflx, ihc), _g(htflx, ih1)
        af1, af2 = _g(afflx, ihc), _g(afflx, ih1)
        hp_i = hp1 + (hp2 - hp1) * dth / safe
        ht_i = ht1 + (ht2 - ht1) * dth / safe
        af_i = af1 + (af2 - af1) * dth / safe
        st = torch.sin(thet)
        tanp = st / (1.0 - torch.cos(thet) + hp_i / k_rz)
        phe = torch.atan(tanp)
        del_i = (k_pio2 - phe - thet) * k_r2d
        sphi = k_rz * torch.cos(del_i * k_d2r) / (
            k_rz + ht_i)
        sphi = torch.clamp(1.0 - sphi * sphi, min=0.000001)
        fv_i = freq * torch.sqrt(sphi)

        write = w165 | w185
        row = torch.where(write, il, zero_i)        # row 0 is the sink
        delmod[row, lanes] = torch.where(w185, del_i, v_del)
        hpmod[row, lanes] = torch.where(w185, hp_i, v_hp)
        htmod[row, lanes] = torch.where(w185, ht_i, v_ht)
        afmod[row, lanes] = torch.where(w185, af_i, v_af)
        fvmod[row, lanes] = torch.where(w185, fv_i, v_fv)
        itmod[row, lanes] = v_it.to(torch.int32)    # both labels take IMODE
        nxt = torch.where(write, torch.full_like(pc, 305), nxt)

        # 305: reject a mode below the minimum take-off angle by negating HP
        ilc = il.clamp(0, NMODE)
        cur_del = delmod[ilc, lanes]
        reject = at[305] & (cur_del < amind)
        rrow = torch.where(reject, ilc, zero_i)
        hpmod[rrow, lanes] = torch.where(reject, -hpmod[rrow, lanes].abs(),
                                         hpmod[rrow, lanes])
        nxt = torch.where(at[305], torch.where(
            cur_del >= amind, torch.full_like(pc, 315),
            torch.full_like(pc, 145)), nxt)

        # 315: six modes is all there is room for
        nxt = torch.where(at[315], torch.where(
            il - NMODE < 0, torch.full_like(pc, 140),
            torch.full_like(pc, _DONE)), nxt)

        pc.copy_(nxt)

    _run_lockstep(_step, pc,
                  (pc, ih, il, dth2, delmod, hpmod, htmod, fvmod, afmod,
                   itmod), use_cuda_graph)

    # Clear the sink.  A lane that finished early keeps riding along and
    # keeps scattering into row 0, so how many times row 0 was written
    # depends on how long the SLOWEST lane in the batch ran.  Nothing reads
    # it -- the mode slots are 1..6 -- but leaving it would make the returned
    # arrays depend on batch composition, which is exactly the property the
    # lockstep tests exist to rule out.
    for arr in (delmod, hpmod, htmod, fvmod, afmod):
        arr[0] = 0.0
    itmod[0] = 5

    nummod = (hpmod[1:NMODE + 1] >= 0).sum(dim=0).to(torch.int32)
    return delmod, hpmod, htmod, fvmod, afmod, itmod, nummod
