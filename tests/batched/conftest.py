"""Stage B (the batched PyTorch port) is an optional extra.

torch is only installed with ``uv sync --extra batched``.  Every module in
this directory imports torch at top level, so without it collection would
error.  When torch is not importable, ignore the whole directory at
collection time and say so once in the terminal summary.

(``pytest.importorskip`` at conftest level is not an option: a ``Skipped``
raised while importing a non-root conftest is reported as a collection
error, not a skip.)
"""

try:
    import torch  # noqa: F401  (every test module here imports it anyway)
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

if not HAVE_TORCH:
    # Patterns are relative to this conftest's directory.
    collect_ignore_glob = ["test_*.py", "*/test_*.py"]

    def pytest_terminal_summary(terminalreporter):
        terminalreporter.write_line(
            "tests/batched/ skipped: torch not installed "
            "(install the optional extra: uv sync --extra batched)")
