"""``voacap_gpu.engine`` is a compatibility facade over the split modules.

``engine.py`` used to hold the whole endpoint (~1,700 lines); its code now
lives in ``system`` / ``status`` / ``antennas`` / ``circuit`` / ``backends`` /
``results`` / ``rigcache`` / ``endpoint``.  Prop-SL and the test suite import
names from ``voacap_gpu.engine`` -- public and private -- so every name it had
must still resolve there, and the shared mutable state (the rig caches, the
coefficient directory) must be the very objects the new modules use, not
copies.  Import-only: no GPU, no cupy.
"""

import sys
import types

import pytest

from voacap_gpu import engine

# ``sorted(n for n in vars(voacap_gpu.engine) if not n.startswith('__'))`` at
# a091fb0 (before the split), minus the imported modules (``copy``,
# ``pathlib``, ``threading``, ``warnings``, ``np``, ``_validate``, ``_K``).
ENGINE_NAMES_AT_SPLIT = [
    'BACKENDS', 'CantennaState', 'Circuit', 'DMP_DEFAULT_MS', 'F32', 'LAYTYP',
    'OrderedDict', 'PMP_DEFAULT_DB', 'POLE_DEFAULT', 'Results',
    'STATUS_ALLMODES_OVERFLOW', 'STATUS_CPU_UNMAPPED',
    'STATUS_ESMOD_AREA_COVERAGE', 'STATUS_ESMOD_GAIN_OOB',
    'STATUS_FRQCOM_INDEX', 'STATUS_GENOIS_AREA_COVERAGE',
    'STATUS_GENOIS_GAIN_OOB', 'STATUS_GENOIS_OLD', 'STATUS_INMUF_ITMOD',
    'STATUS_INMUF_MODES_ROW', 'STATUS_LNGPAT_INDEX', 'STATUS_LSMOOTH_INDEX',
    'STATUS_LUFFY_BAD_NMODE2', 'STATUS_OK', 'STATUS_REGMOD_AREA_COVERAGE',
    'STATUS_REGMOD_FREQ', 'STATUS_REGMOD_GAIN_OOB', 'STATUS_REGMOD_MUFDAY',
    'STATUS_RELBIL_INDEX', 'STATUS_SELTXR_INDEX',
    'STATUS_SETTXR_AREA_COVERAGE', 'STATUS_SETTXR_DELMUF',
    'STATUS_SETTXR_GAIN_OOB', 'STATUS_SETTXR_YMUF', 'STATUS_SETUP_HOUR',
    'STATUS_UNWRITTEN', 'System', 'UnsupportedAntennaType', '_BATCH_RIGS',
    '_BUILD_CIRCUIT', '_COEFFS_DIR', '_FREQ_ATOL', '_GCDLNG', '_IONO_TABLES',
    '_MODE_BY_CODE', '_MODE_LABELS', '_PKG_COEFFS', '_PREDICT_CIRCUITS',
    '_PREDICT_ROWS', '_RigCache', '_SCALAR_REFUSALS', '_SMALL_RIGS',
    '_STATUS_GAIN', '_STATUS_INDEX_REFUSAL', '_VENDORED_COEFFS',
    '_antcalc_offazim', '_assemble', '_check_bearing', '_cuda_usable',
    '_device_id', '_device_iono', '_device_path_ok', '_finish_cant',
    '_finish_results', '_install_geometry', '_ipflag', '_parse_antenna',
    '_predict_circuits_cpu', '_predict_device', '_predict_host',
    '_resolve_backend', '_resolve_method', '_resolve_model',
    '_results_from_son', '_scalar_status', '_setup_hour', '_small_rig',
    '_valid_mask', 'antcalc_cant', 'build_circuit', 'cant_from_files',
    'clear_device_cache', 'coeffs_dir', 'curmuf', 'dataclass', 'dazel0',
    'esind', 'f2var', 'field', 'geom', 'geotim', 'ionset', 'isotropic_cant',
    'noisy', 'offazim', 'predict', 'predict_batch', 'predict_circuits',
    'read_antenna', 'redmap', 'replace', 'sang', 'setup_config', 'timvar',
    'virtim',
]

# Where each split-out name now lives (engine's binding must be that object).
HOMES = {
    "system": ["System", "POLE_DEFAULT", "DMP_DEFAULT_MS", "PMP_DEFAULT_DB",
               "coeffs_dir", "_COEFFS_DIR", "_PKG_COEFFS",
               "_VENDORED_COEFFS"],
    "status": [n for n in ENGINE_NAMES_AT_SPLIT if n.startswith("STATUS_")]
              + ["_scalar_status", "_SCALAR_REFUSALS", "_STATUS_GAIN",
                 "_STATUS_INDEX_REFUSAL"],
    "antennas": ["offazim", "_finish_cant", "isotropic_cant",
                 "cant_from_files", "_antcalc_offazim", "_check_bearing",
                 "antcalc_cant", "_parse_antenna"],
    "circuit": ["_GCDLNG", "_resolve_method", "_resolve_model", "_ipflag",
                "_install_geometry", "_setup_hour", "build_circuit",
                "setup_config"],
    "backends": ["BACKENDS", "_cuda_usable", "_resolve_backend",
                 "_predict_circuits_cpu", "predict_circuits"],
    "results": ["Results", "_valid_mask", "_MODE_LABELS", "_FREQ_ATOL",
                "_MODE_BY_CODE", "_results_from_son", "_assemble",
                "_finish_results"],
    "rigcache": ["_RigCache", "_SMALL_RIGS", "_BATCH_RIGS", "_IONO_TABLES",
                 "clear_device_cache", "_device_id", "_device_iono",
                 "_small_rig"],
    "endpoint": ["predict", "_device_path_ok", "_predict_host",
                 "_predict_device", "_PREDICT_ROWS", "predict_batch"],
}


@pytest.mark.parametrize("name", ENGINE_NAMES_AT_SPLIT)
def test_every_pre_split_name_resolves(name):
    assert hasattr(engine, name), f"engine.{name} no longer resolves"
    assert not isinstance(getattr(engine, name), types.ModuleType)


def test_from_import_names():
    """The spellings Prop-SL and the suite use, as ``from ... import``."""
    from voacap_gpu.engine import (  # noqa: F401
        _COEFFS_DIR, STATUS_CPU_UNMAPPED, STATUS_GENOIS_AREA_COVERAGE,
        STATUS_OK, Results, System, antcalc_cant, build_circuit,
        cant_from_files, coeffs_dir, isotropic_cant, offazim, predict,
        predict_circuits, setup_config)


@pytest.mark.parametrize("module", sorted(HOMES))
def test_engine_names_are_the_new_modules_objects(module):
    mod = __import__(f"voacap_gpu.{module}", fromlist=["_"])
    for name in HOMES[module]:
        assert getattr(engine, name) is getattr(mod, name), (module, name)
    # module-level state exists once: the home module is the defining one
    for name in HOMES[module]:
        obj = getattr(mod, name)
        if isinstance(obj, (types.FunctionType, type)):
            assert obj.__module__ == f"voacap_gpu.{module}", (name,
                                                              obj.__module__)


def test_split_covers_every_engine_definition():
    """Each name engine.py used to define is in exactly one HOMES entry,
    except the two pre-split hook sentinels (kept in the facade only)."""
    placed = [n for names in HOMES.values() for n in names]
    assert len(placed) == len(set(placed))
    facade_only = {"_BUILD_CIRCUIT", "_PREDICT_CIRCUITS"}
    # names engine.py imported rather than defined
    scalar_or_stdlib = {
        "CantennaState", "Circuit", "F32", "LAYTYP", "OrderedDict",
        "UnsupportedAntennaType", "curmuf", "dataclass", "dazel0", "esind",
        "f2var", "field", "geom", "geotim", "ionset", "noisy",
        "read_antenna", "redmap", "replace", "sang", "timvar", "virtim"}
    missing = (set(ENGINE_NAMES_AT_SPLIT) - scalar_or_stdlib - facade_only
               - set(placed))
    assert not missing, sorted(missing)
    assert set(placed) <= set(ENGINE_NAMES_AT_SPLIT)


def test_rig_caches_are_shared_with_batch():
    from voacap_gpu import batch, rigcache
    assert batch._rigcache is rigcache
    assert engine._SMALL_RIGS is rigcache._SMALL_RIGS
    assert engine._BATCH_RIGS is rigcache._BATCH_RIGS
    assert engine._IONO_TABLES is rigcache._IONO_TABLES


def test_coeffs_dir_unchanged():
    from voacap_gpu import system
    assert engine._COEFFS_DIR == system._COEFFS_DIR == engine.coeffs_dir()
    assert engine._COEFFS_DIR.is_dir()


def test_hook_sentinels_are_the_real_functions():
    assert engine._BUILD_CIRCUIT is engine.build_circuit
    assert engine._PREDICT_CIRCUITS is engine.predict_circuits


def test_facade_import_does_not_need_cupy():
    """Importing the facade (and every split module) never imports cupy, so
    the CPU-only CI runner can import it."""
    import subprocess
    code = ("import sys, voacap_gpu.engine, voacap_gpu.batch; "
            "sys.exit(1 if 'cupy' in sys.modules else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr
