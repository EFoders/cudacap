"""``cuda/driver.py`` is a compatibility facade: every name it had still resolves.

``driver.py`` used to hold the whole LUFFY host side; it now re-exports from
``abi`` / ``checks`` / ``tables`` / ``kernels`` / ``results`` /
``luffy_rigs`` / ``setup_rigs``.  Code outside the package (Prop-SL, the
tools, these tests) imports through ``voacap_gpu.cuda.driver``, so this pins:

* every module-level name ``driver`` exported before the split (snapshotted
  below from the base commit with ``sorted(vars(driver))``, dunders and
  module objects excluded) still resolves on the facade, as the same kind of
  object, and IS the object of the module that now defines it -- so the
  mutable state (``_MAP_CACHE``) exists exactly once;
* the NVRTC source strings are byte-identical to the base commit's, so the
  compiled kernels cannot have moved;
* the split modules never import the facade (no cycle);
* where each monkeypatch-sensitive helper is LOOKED UP, which is where a test
  or bench has to patch it.

Import only: nothing here touches a device, so it needs CuPy importable but
no GPU.
"""

import ast
import hashlib
import inspect
import pathlib
import types

import numpy as np
import pytest

pytest.importorskip("cupy")

from voacap_gpu.cuda import (abi, checks, driver, kernels,  # noqa: E402
                             luffy_rigs, results, setup_rigs, tables)
from voacap_gpu.cuda import compile as cuda_compile  # noqa: E402
from voacap_gpu.cuda import state  # noqa: E402

SPLIT = (abi, checks, tables, kernels, results, luffy_rigs, setup_rigs)

# sorted(vars(driver)) at a091fb0 (pre-split), name -> kind (see _kind).
BASE_NAMES = {
    'ALLMODE_DTYPE': 'dtype',
    'ANOIS_DTYPE': 'dtype',
    'CARRY_FIELDS': 'tuple',
    'CIRCUIT_DTYPE': 'dtype',
    'CIRCUIT_WORDS': 'int',
    'F32': 'class',
    'FRQ_DTYPE': 'dtype',
    'GRID_SON_FIELDS': 'tuple',
    'GRID_SON_OFFSETS': 'tuple',
    'I32': 'class',
    'INPUT_FIELDS': 'tuple',
    'IonoSlot': 'class',
    'LAYER_DTYPE': 'dtype',
    'LOSX_DTYPE': 'dtype',
    'LPATH_DTYPE': 'dtype',
    'LuffyArena': 'class',
    'LuffyResults': 'class',
    'MDL_CODE': 'dict',
    'MDL_LABEL': 'dict',
    'MODES_DTYPE': 'dtype',
    'MUF_DTYPE': 'dtype',
    'NOISEDIAG_DTYPE': 'dtype',
    'PipelinedLuffy': 'class',
    'PipelinedSetupLuffy': 'class',
    'RAYS_DTYPE': 'dtype',
    'RAYS_UPLOAD_FIELDS': 'tuple',
    'RAYS_UPLOAD_LEAVES': 'tuple',
    'REFLX_DTYPE': 'dtype',
    'RESULT_BYTES_PER_THREAD': 'int',
    'RESULT_FIELDS': 'tuple',
    'SETUP_BYTES_PER_THREAD': 'int',
    'SETUP_CONFIG_DTYPE': 'dtype',
    'SETUP_IPFLAG_REFUSED': 'int',
    'SETUP_NAB': 'int',
    'SETUP_NHOUR': 'int',
    'SETUP_PARAMS_DTYPE': 'dtype',
    'SETUP_STATUS_BAD_HOUR': 'int',
    'SIGD_DTYPE': 'dtype',
    'SLIM_ALWAYS': 'tuple',
    'SLIM_EXTRA': 'tuple',
    'SMOOTHSTORE_BYTES': 'int',
    'SON_DTYPE': 'dtype',
    'STATE_BYTES_PER_THREAD': 'int',
    'SetupLuffy': 'class',
    'SlimResults': 'class',
    'SlimSpec': 'class',
    'TIME_DTYPE': 'dtype',
    'TON_DTYPE': 'dtype',
    'UPLOAD_BYTES_PER_THREAD': 'int',
    'UPLOAD_FIELDS': 'tuple',
    'UPLOAD_LEAVES': 'tuple',
    'ZON_DTYPE': 'dtype',
    '_BLOCKS': 'tuple',
    '_CIRC_WORDS': 'int',
    '_GRIDREDUCE_SRC': 'str',
    '_IONO_TABLES': 'tuple',
    '_KERNEL_SRC': 'str',
    '_MAP_CACHE': 'dict',
    '_RAYS_CHUNK': 'int',
    '_RAYS_LEN': 'dict',
    '_RAYS_OFF': 'dict',
    '_RAYS_UPLOAD_RUNS': 'tuple',
    '_RAYS_WORDS': 'int',
    '_RESULT_COLS': 'dict',
    '_RESULT_MAP': 'ndarray',
    '_RESULT_RUNS': 'tuple',
    '_RESULT_WORDS': 'int',
    '_SCALARS': 'tuple',
    '_SETUP_SRC': 'str',
    '_SETUP_TABLES': 'tuple',
    '_SON_BASE': 'int',
    '_SON_SLOTS': 'int',
    '_UPLOAD_IDX': 'ndarray',
    '_UPLOAD_MAP': 'ndarray',
    '_UPLOAD_RUNS': 'tuple',
    '_USER_FREQ_SLOTS': 'int',
    '_XFER_SRC': 'str',
    '_check_luffy_inputs': 'function',
    '_check_setup_params': 'function',
    '_circuit_inputs_to_device': 'function',
    '_columns': 'function',
    '_device_map': 'function',
    '_drain_default_stream': 'function',
    '_gather_slim': 'function',
    '_merge': 'function',
    '_n_antenna_sets': 'function',
    '_offending': 'function',
    '_pack_son_mdl': 'function',
    '_quiesce': 'function',
    '_rays_from_soa': 'function',
    '_rays_inputs_to_soa': 'function',
    '_rays_to_soa': 'function',
    '_rays_views': 'function',
    '_require_dtype': 'function',
    '_require_rows': 'function',
    '_span': 'function',
    '_split_results': 'function',
    '_unpack_son_mdl': 'function',
    '_word_map': 'function',
    '_write_circuit_inputs': 'function',
    '_write_rays_inputs': 'function',
    '_xfer': 'function',
    'circuit_state_sizeof': 'function',
    'expand_fields': 'function',
    'fetch_luffy': 'function',
    'fetch_results': 'function',
    'fetch_slim_results': 'function',
    'launch_luffy': 'function',
    'leaves_of': 'function',
    'load': 'function',
    'load_gridreduce_kernel': 'function',
    'load_kernel': 'function',
    'load_setup_kernel': 'function',
    'luffy_results': 'function',
    'pack_allmode': 'function',
    'pack_anois': 'function',
    'pack_circuit': 'function',
    'pack_frq': 'function',
    'pack_layer': 'function',
    'pack_losx': 'function',
    'pack_lpath': 'function',
    'pack_modes': 'function',
    'pack_muf': 'function',
    'pack_noisediag': 'function',
    'pack_rays': 'function',
    'pack_reflx': 'function',
    'pack_sigd': 'function',
    'pack_son': 'function',
    'pack_time': 'function',
    'pack_ton': 'function',
    'pack_zon': 'function',
    'run_luffy': 'function',
    'run_luffy_pipelined': 'function',
    'setup_and_run': 'function',
    'setup_iono_tables': 'function',
    'setup_struct_sizeof': 'function',
    'shared_antenna_tables': 'function',
    'shared_iono_tables': 'function',
    'slim_spec': 'function',
    'slots_in_use': 'function',
    'smoothstore_sizeof': 'function',
    'stage_luffy': 'function',
    'unpack_allmode': 'function',
    'unpack_anois': 'function',
    'unpack_circuit': 'function',
    'unpack_frq': 'function',
    'unpack_layer': 'function',
    'unpack_losx': 'function',
    'unpack_lpath': 'function',
    'unpack_modes': 'function',
    'unpack_muf': 'function',
    'unpack_noisediag': 'function',
    'unpack_rays': 'function',
    'unpack_reflx': 'function',
    'unpack_sigd': 'function',
    'unpack_son': 'function',
    'unpack_time': 'function',
    'unpack_ton': 'function',
    'unpack_zon': 'function',
}

# SHA-256 of each NVRTC translation unit at a091fb0.
BASE_SRC_SHA256 = {
    "_KERNEL_SRC":
        "fad76500cd05703e12d8747780a6470336d9b97d79f0f00c66307b484f6c1307",
    "_SETUP_SRC":
        "48b10218ecca1d734ba4f6f9d87b55440704300e9e03ee2772dd7e3d94f6f650",
    "_XFER_SRC":
        "a20e41d04201fecc8c5d53b1aa884abc9544dd632eb290a655508d124424430c",
    "_GRIDREDUCE_SRC":
        "53b852e8d7caa8692c289a5e477f3e82313fa62dca865e482210190f56c45404",
}


def _kind(v):
    if inspect.isclass(v):
        return "class"
    if inspect.isfunction(v) or inspect.isbuiltin(v):
        return "function"
    if isinstance(v, np.dtype):
        return "dtype"
    return type(v).__name__


def _exported(mod):
    return {k: v for k, v in vars(mod).items()
            if not (k.startswith("__") and k.endswith("__"))
            and not isinstance(v, types.ModuleType)}


def test_snapshot_is_complete():
    assert len(BASE_NAMES) == 159


def test_every_name_resolves_as_before():
    gone = [n for n in BASE_NAMES if not hasattr(driver, n)]
    assert not gone, f"gone from driver: {gone}"
    moved = {n: (k, _kind(getattr(driver, n)))
             for n, k in BASE_NAMES.items() if _kind(getattr(driver, n)) != k}
    assert not moved, f"kind changed (was, now): {moved}"


def test_no_names_added_to_the_facade():
    """The facade re-exports exactly the old surface, nothing new."""
    assert sorted(_exported(driver)) == sorted(BASE_NAMES)


def _defined_in(mod):
    """Names a module DEFINES at top level (def / class / assignment)."""
    tree = ast.parse(pathlib.Path(mod.__file__).read_text(encoding="utf-8"))
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            out.add(node.target.id)
    return out


def test_facade_objects_are_the_defining_modules_objects():
    """Re-exports, not copies: one object per name, one cache per process."""
    homes = {}
    for mod in SPLIT:
        for k in _defined_in(mod):
            assert k not in homes, f"{k} defined in {homes[k].__name__} "                                    f"and {mod.__name__}"
            homes[k] = mod
    for name in BASE_NAMES:
        if name in homes:
            home = homes[name]
        elif name == "load":
            home = cuda_compile
        else:
            home = state                    # the per-struct dtypes / packers
        assert getattr(driver, name) is getattr(home, name), name
    assert driver._MAP_CACHE is kernels._MAP_CACHE
    assert isinstance(kernels._MAP_CACHE, dict)
    for mod in SPLIT:
        if hasattr(mod, "_MAP_CACHE"):
            assert mod._MAP_CACHE is kernels._MAP_CACHE, mod.__name__


def test_functions_and_classes_are_defined_once():
    """No helper is duplicated across the split modules."""
    seen = {}
    for mod in SPLIT:
        for k, v in vars(mod).items():
            if (inspect.isfunction(v) or inspect.isclass(v))                     and v.__module__ == mod.__name__:
                assert k not in seen, f"{k} defined in {seen[k]} and "                                       f"{mod.__name__}"
                seen[k] = mod.__name__


@pytest.mark.parametrize("name", sorted(BASE_SRC_SHA256))
def test_kernel_sources_byte_identical(name):
    src = getattr(driver, name)
    assert hashlib.sha256(src.encode()).hexdigest() == BASE_SRC_SHA256[name]
    assert getattr(kernels, name) is src


def test_split_modules_never_import_the_facade():
    for mod in SPLIT:
        tree = ast.parse(pathlib.Path(mod.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "voacap_gpu.cuda.driver", mod.__name__
                assert not (node.module == "voacap_gpu.cuda" and any(
                    a.name == "driver" for a in node.names)), mod.__name__
                assert not (node.level and node.module in (None, "driver")
                            ), mod.__name__
            elif isinstance(node, ast.Import):
                assert all(a.name != "voacap_gpu.cuda.driver"
                           for a in node.names), mod.__name__


def test_monkeypatch_lookup_sites():
    """Where the helpers tests / benches swap out are looked up.

    Patching ``driver.X`` rebinds only the facade attribute; the callers read
    their own module globals.  test_rig_race patches
    ``setup_rigs._check_setup_params``; test_fastmath (and
    tools/bench_fastmath.py through it) patches ``luffy_rigs.load_kernel``.
    """
    g = luffy_rigs.stage_luffy.__globals__
    assert g is vars(luffy_rigs) and g["load_kernel"] is kernels.load_kernel
    for fn in (setup_rigs.SetupLuffy._validate,
               setup_rigs.PipelinedSetupLuffy._run):
        g = fn.__globals__
        assert g is vars(setup_rigs)
        assert g["_check_setup_params"] is checks._check_setup_params
