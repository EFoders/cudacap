"""The lockstep driver shared by the batched GO TO state machines.

FDIST proved the pattern and FINDF is the second user, so it lives here rather
than in either.  The idea, in one line: every lane carries its own program
counter, one call to ``step`` advances every lane by one Fortran statement, and
a lane that has finished parks at the terminal label where its state stops
changing.  Branch divergence disappears; the cost is that a lane which
finished early rides along until the slowest lane in the batch is done.

CUDA GRAPHS ARE WHY THIS IS WORTH DOING.  These steps are tiny -- tens of
elementwise kernels over a vector -- and there are hundreds or thousands of
them, so the eager loop is bound by kernel LAUNCHES and runs at the same speed
regardless of how many lanes it is carrying.  Capturing one step and replaying
it collapses those launches into one and is bit-for-bit identical, because it
is the same kernels in the same order.

Two rules the step has to obey, both of them consequences of capture:

  * **Update state in place** (``buf.copy_(...)``), never by rebinding.  A
    replay re-runs the captured kernels against the captured buffer
    ADDRESSES; ``pc = torch.where(...)`` would leave the graph faithfully
    updating a tensor the Python name no longer points at.
  * **Hoist host-built constants out of the step.**  ``_k`` allocates on the
    host and a host-to-device copy is not capturable.

CAPTURE HAS TO BE FED A COLD STATE.  Capture requires a warmup on a side
stream, and the warmup genuinely RUNS the step -- so it advances the very
state machine being captured.  Every buffer is cloned beforehand and restored
after both the warmup and the capture; skip either restore and the answer is
silently the result of starting a few steps in, which is invisible on a batch
whose lanes all finish early.

torch.compile over the step body would be the other lever and is NOT available
here: it needs Triton, which has no Windows build.
"""

import torch

CHECK_EVERY = 16                # graph replays between completion checks


def run_lockstep(step, pc, buffers, done, max_steps, name,
                 use_cuda_graph=True, check_every=CHECK_EVERY):
    """Drive `step` until every lane parks at label `done`.

    `buffers` is every tensor the step mutates, `name` names the routine in
    the non-convergence error, and `max_steps` is the structural bound on the
    step count -- exceeding it means a transition is wrong, so it raises
    rather than silently truncating.  This is the same tripwire Stage A
    carries; a lockstep port needs it more, because a lane stuck in a cycle
    costs the whole batch.
    """
    if not (use_cuda_graph and pc.is_cuda):
        for _ in range(max_steps):
            if not bool((pc != done).any()):
                return
            step()
        raise RuntimeError(_stuck(name, max_steps))

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
    for _ in range(0, max_steps, check_every):
        for _ in range(check_every):
            graph.replay()
        if not bool((pc != done).any()):
            return
    raise RuntimeError(_stuck(name, max_steps))


def _stuck(name, max_steps):
    return (f"{name} lockstep did not converge in {max_steps} steps.  The "
            "cursors only ever advance, so this is a wrong transition rather "
            "than a batch that needs more room")


def plan_buckets(key, nbucket):
    """Group lanes by `key` so a lockstep run pays each group's OWN length.

    Returns ``(order, bounds)``: `order` is a permutation of the lane indices
    sorted by key, and `bounds` are the split points, so
    ``order[bounds[i]:bounds[i+1]]`` is one bucket.  Reordering is exact -- it
    is a permutation, and `restore_order` undoes it -- so nothing about the
    answer changes, only how long the batch waits for its slowest member.

    WHETHER THIS PAYS DEPENDS ENTIRELY ON THE KEY, and for FINDF the honest
    answer today is "not yet".  Its step count is determined EXACTLY by the
    number of rays a lane finds (measured correlation 1.000), and that is not
    known until the lane has run.  The best a-priori predictor to hand,
    whether an F1 ledge exists, correlates only 0.403.  Perfect 8-way
    bucketing would save 36% of lane-steps on the benchmark batch; a 0.4
    correlation delivers a fraction of that, against the cost of a sort, a
    gather and scatter of every state buffer, and a separate CUDA-graph
    capture per bucket.

    It becomes worth using the moment a caller HAS a good key, and a driver
    will: consecutive operating frequencies at the same circuit and hour find
    almost the same number of rays, so the previous frequency's IAFTXR is a
    strong predictor of this one's.  That is why this lives here now rather
    than being written when the driver arrives -- and why it is opt-in.
    """
    if nbucket < 1:
        raise ValueError(f"nbucket must be >= 1, got {nbucket}")
    order = torch.argsort(key)
    n = key.shape[0]
    edges = [round(i * n / nbucket) for i in range(nbucket + 1)]
    return order, edges


def restore_order(values, order):
    """Undo `plan_buckets`' permutation along the LAST axis."""
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.shape[0], device=order.device)
    return values[..., inv]


def gather_rows(arr, idx):
    """``arr[idx[lane], lane]`` for every lane -- the per-lane row pick.

    Every cursor in these state machines indexes the first axis of a
    (rows, lanes) table, and each lane wants a different row, so this one
    gather is most of what the ports do.
    """
    return torch.gather(arr, 0, idx.unsqueeze(0).expand(1, -1)).squeeze(0)
