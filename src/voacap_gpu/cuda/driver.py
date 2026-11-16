"""Host side of Stage C's LUFFY kernel — one thread per circuit-hour.

``cuda/luffy.cuh`` composes the thirty-odd ported leaves into one
``__device__`` function that runs a whole hour of system performance for one
circuit.  This module is the other half: it lays a Stage A ``Circuit`` out in
the byte order ``cuda/circuit.cuh`` declares, moves the SHARED tables (the
REDMAP coefficient maps and the antenna patterns) once, launches N threads,
and reads the mutated states back into Stage A objects.

THE ABI, and how it is proved
-----------------------------
``CIRCUIT_DTYPE`` is a numpy structured dtype built from the per-struct
dtypes in ``cuda/state.py`` — the same ones ``tests/cuda/test_state.py``
already pins field-for-field against ``state.cuh`` — followed by the loose
scalars in Stage A ``Circuit`` order.  Every member is 4 bytes, so numpy's
unaligned packing equals the C layout.  ``circuit_state_sizeof()`` compiles a
one-line kernel that writes ``sizeof(CircuitState)``; the test asserts it
equals ``CIRCUIT_DTYPE.itemsize``, which is the same offset proof the state
ABI already carries.

``iono`` and ``cant`` are NOT in the dtype.  They are big, read-only and
shared between threads, so they move once as flat device arrays and each
thread gets an index into them.

/RAYS/ IS NOT IN THE DTYPE EITHER, and for a completely different reason
--------------------------------------------------------------------------
``RaysState`` is 30,672 of the old 55,652 bytes per thread.  Carried inside
CircuitState it made adjacent lanes of a warp 55,652 bytes apart, so every
warp-wide read of one float cost 32 memory transactions -- measured at that
stride, 34 GB/s of useful bandwidth against 2,892 GB/s for the same data laid
out structure-of-arrays.  So /RAYS/ now lives in its OWN device buffers,
transposed to SoA: element ``o`` of thread ``t`` is at ``o * n + t``.  The
device sees it through ``cuda/raysview.cuh``'s ``RaysView``; the host still
packs and unpacks an ordinary ``(N,) RAYS_DTYPE`` array, and this module does
the transpose on the device (a couple of milliseconds at 32k threads) so the
callers' buffers keep their obvious per-circuit shape.

``pack_circuit`` therefore returns a PAIR -- ``(circuit_buffer, rays_buffer)``
-- and ``unpack_circuit`` takes the same pair back.  Together they are still
the whole Stage A ``Circuit``, field for field.

/SON/ MDL
---------
``pack_son``/``unpack_son`` encode ``mdl``/``mode``/``moder`` as LAYTYP
indices.  That is right for MODE and MODER, but MDL is not a layer label: it
is LUFFY's one-character model tag, "" / " " / "S" / "L" / "M".  Rather than
duplicate ``state.py``'s logic this module calls the same packers with MDL
blanked and then overwrites that one column with ``MDL_CODE`` — which is what
``cuda/circuit.cuh``'s ``VC_MDL_*`` constants hold.

MOST OF THE 55,652 BYTES NEVER NEEDED TO CROSS THE BUS
------------------------------------------------------
Measured at 32,256 circuit-hours, the round trip was ~618 ms of which the
kernel is ~127 ms; the other ~490 ms was 1.8 GB of per-thread state going up
and 1.8 GB coming back.  Almost none of it is INPUT.  ``UPLOAD_FIELDS`` below
is the set that is, established by experiment rather than by reading the
source: zero one field in all 504 replay circuit-hours, launch, and diff the
WHOLE output against the clean run.  A field whose zeroing changes nothing
else is not read.  ``tests/cuda/test_input_set.py`` pins the result, and
``tools/bench_stage_c.py --trip`` measures what it buys.

The three buckets that came out of that sweep, for the whole CircuitState and
the whole of /RAYS/:

* **INPUT** — read before it is written.  ``INPUT_FIELDS``.
* **ECHO** — never read, but not fully overwritten either, so its exit value
  still depends on its entry value (``reflx``, ``modes``, ``zon``, ``allm``,
  most of ``son``, ``losx``, the ``layers`` profile arrays, and /RAYS/ IFOB).
  Scratch as far as the kernel is concerned.
* **DEAD** — never read AND fully rewritten (``anois``, ``ndiag``, ``sigd``,
  ``ton``, ``frq.jmode`` ...).

**/RAYS/ IS NOT AN INPUT except for its first 168 bytes.**  NANG and ANG are
read before FOBBY rewrites them; IFOB — 30,504 of the 30,672 bytes, 55% of
the whole per-thread state — is pure scratch.

THREE THINGS THE EXPERIMENT CANNOT SEE, AND WHY THEY ARE UPLOADED ANYWAY
-----------------------------------------------------------------------
``layers.fs``, ``layers.ielect`` and ``muf.afmuf`` are zero in every one of
the 21 replay decks, so zeroing them is a no-op and the sweep calls them
dead.  They are read (esmod.cuh/regmod.cuh, lecden.cuh, inmuf.cuh/settxr.cuh)
and NOTHING in the port ever writes them, so for a caller who sets them they
are inputs.  132 bytes; they are carried.  The same argument covers the four
``muf`` MUF ladders and ``time.it`` that ``frqcom`` reads on the LUF arms
(IPFLAG 300/400), which the replay does not exercise at all.

``CARRY_FIELDS`` is a different thing again: fields the kernel does NOT read,
but which appear in ``RESULT_FIELDS`` and are only partly written, so the
answer the host reads back still contains their entry values.  /SON/ is the
whole result table and LUFFY skips every frequency slot whose FREL is zero,
leaving the previous hour's numbers in place; /MUFS/ XLUF is untouched
altogether on the system-performance arms, where SETOUT's ``xluf[:] = -1``
preset has to survive.  Carrying them is not about fidelity inside the
kernel; it is about the result table being complete.  Note that a ZERO probe
misses this entirely -- the skipped /SON/ columns happen to hold zeros in the
replay decks -- which is why the pinning test poisons with NaN.

TWO UPLOAD MODES, TWO FETCH PATHS
---------------------------------
``stage_luffy(..., upload="full")`` is the old behaviour and stays the
default: every byte up, so ``fetch_luffy`` can return every byte back and the
replay can compare all 7,012,152 cells.  ``upload="inputs"`` sends 3,228
bytes per thread and ``cp.zeros`` the rest on the device (a memset, not a
copy).  ``fetch_results`` brings back 1,700 bytes -- /SON/, XLUF, JMODE and
the status -- which is what a production caller actually wants.  The two
modes agree bit-for-bit on those results across all 504 replay circuit-hours;
that is the test that makes the reduction safe.

``LuffyArena`` is the third piece: the device buffers and the shared tables
allocated once and reused, so a caller sweeping batches pays neither the
allocation nor the 1.8 GB memset per batch.  It never re-zeroes between
submissions, which is the same claim as the input set made once more --
nothing outside ``UPLOAD_FIELDS`` is read, so the previous batch's leftovers
are as good as zeros.  Measured at 32,256 circuit-hours the two ends together
go from 515 ms to 98 ms, **5.2x**; the round trip from 901 to 472 ms.

MODULE LAYOUT
-------------
This module is now a COMPATIBILITY FACADE: everything above still holds, and
every name it ever exported is still importable from here, but the code
lives in cohesive modules beside it (the facade imports from them, never the
reverse):

* ``abi.py``        -- dtypes, field sets, word runs, byte budgets, constants
* ``checks.py``     -- input validation and the stream-ordering helpers
* ``tables.py``     -- shared iono / antenna / setup tables, circuit pack/unpack
* ``kernels.py``    -- the NVRTC sources, their loaders, the transfer helpers
* ``results.py``    -- result fetches: full, compact, slim, grid-reduce offsets
* ``luffy_rigs.py`` -- stage/launch, ``LuffyArena``, ``PipelinedLuffy``,
  ``run_luffy``
* ``setup_rigs.py`` -- ``SetupLuffy``, ``setup_and_run``,
  ``PipelinedSetupLuffy``

Code that MONKEYPATCHES a helper must patch the module that looks it up (for
example ``setup_rigs._check_setup_params`` or ``luffy_rigs.load_kernel``);
rebinding the name here changes only this facade's attribute.
"""

# ruff: noqa: F401 -- every import below is a re-export.
import numpy as np

from voacap_gpu.cuda.abi import (CARRY_FIELDS, CIRCUIT_DTYPE, F32, I32,
                                 INPUT_FIELDS, MDL_CODE, MDL_LABEL,
                                 RAYS_UPLOAD_FIELDS, RAYS_UPLOAD_LEAVES,
                                 RESULT_BYTES_PER_THREAD, RESULT_FIELDS,
                                 SETUP_BYTES_PER_THREAD, SETUP_CONFIG_DTYPE,
                                 SETUP_IPFLAG_REFUSED, SETUP_NAB, SETUP_NHOUR,
                                 SETUP_PARAMS_DTYPE, SETUP_STATUS_BAD_HOUR,
                                 STATE_BYTES_PER_THREAD,
                                 UPLOAD_BYTES_PER_THREAD, UPLOAD_FIELDS,
                                 UPLOAD_LEAVES, _BLOCKS, _CIRC_WORDS,
                                 _IONO_TABLES, _RAYS_CHUNK, _RAYS_LEN,
                                 _RAYS_OFF, _RAYS_UPLOAD_RUNS, _RAYS_WORDS,
                                 _RESULT_COLS, _RESULT_RUNS, _RESULT_WORDS,
                                 _SCALARS, _SETUP_TABLES, _UPLOAD_IDX,
                                 _UPLOAD_RUNS, _columns, _merge, _span,
                                 expand_fields, leaves_of)
from voacap_gpu.cuda.checks import (_check_luffy_inputs, _check_setup_params,
                                    _drain_default_stream, _n_antenna_sets,
                                    _offending, _quiesce, _require_dtype,
                                    _require_rows)
from voacap_gpu.cuda.compile import load
from voacap_gpu.cuda.kernels import (SMOOTHSTORE_BYTES, _GRIDREDUCE_SRC,
                                     _KERNEL_SRC, _MAP_CACHE, _RESULT_MAP,
                                     _SETUP_SRC, _UPLOAD_MAP, _XFER_SRC,
                                     _circuit_inputs_to_device, _device_map,
                                     _rays_from_soa, _rays_inputs_to_soa,
                                     _rays_to_soa, _rays_views, _word_map,
                                     _write_circuit_inputs, _write_rays_inputs,
                                     _xfer, circuit_state_sizeof,
                                     load_gridreduce_kernel, load_kernel,
                                     load_setup_kernel, setup_struct_sizeof,
                                     smoothstore_sizeof)
from voacap_gpu.cuda.luffy_rigs import (LuffyArena, PipelinedLuffy,
                                        launch_luffy, luffy_results, run_luffy,
                                        run_luffy_pipelined, stage_luffy)
from voacap_gpu.cuda.results import (CIRCUIT_WORDS, GRID_SON_FIELDS,
                                     GRID_SON_OFFSETS, SLIM_ALWAYS, SLIM_EXTRA,
                                     _SON_BASE, _SON_SLOTS, _USER_FREQ_SLOTS,
                                     LuffyResults, SlimResults, SlimSpec,
                                     _gather_slim, _split_results, fetch_luffy,
                                     fetch_results, fetch_slim_results,
                                     slim_spec, slots_in_use)
from voacap_gpu.cuda.setup_rigs import (PipelinedSetupLuffy, SetupLuffy,
                                        setup_and_run)
from voacap_gpu.cuda.state import (ALLMODE_DTYPE, ANOIS_DTYPE, FRQ_DTYPE,
                                   LAYER_DTYPE, LOSX_DTYPE, LPATH_DTYPE,
                                   MODES_DTYPE, MUF_DTYPE, NOISEDIAG_DTYPE,
                                   RAYS_DTYPE, REFLX_DTYPE, SIGD_DTYPE,
                                   SON_DTYPE, TIME_DTYPE, TON_DTYPE, ZON_DTYPE,
                                   pack_allmode, pack_anois, pack_frq,
                                   pack_layer, pack_losx, pack_lpath,
                                   pack_modes, pack_muf, pack_noisediag,
                                   pack_rays, pack_reflx, pack_sigd, pack_son,
                                   pack_time, pack_ton, pack_zon,
                                   unpack_allmode, unpack_anois, unpack_frq,
                                   unpack_layer, unpack_losx, unpack_lpath,
                                   unpack_modes, unpack_muf, unpack_noisediag,
                                   unpack_rays, unpack_reflx, unpack_sigd,
                                   unpack_son, unpack_time, unpack_ton,
                                   unpack_zon)
from voacap_gpu.cuda.tables import (IonoSlot, _pack_son_mdl, _unpack_son_mdl,
                                    pack_circuit, setup_iono_tables,
                                    shared_antenna_tables, shared_iono_tables,
                                    unpack_circuit)
