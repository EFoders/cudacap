"""Suite-wide pytest plumbing: the ``oracle`` marker.

Tests marked ``@pytest.mark.oracle`` genuinely need the vendored Fortran tree
under ``reference/`` (voacapl sources and data that are not shipped with the
package or copied into ``tests/data``).  ``reference/`` was removed from the
distributed tree, so on a clean clone those tests are skipped with a reason
that says how to restore it; with ``reference/`` present they run as normal.

    pytest -m "not oracle"     # only what a clean clone can run
    pytest -m oracle           # only the reference/-dependent tests

The marker is registered here rather than in pyproject.toml.
"""

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
REFERENCE = REPO / "reference"
RESTORE = "git checkout 477dad7^ -- reference"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "oracle: needs the vendored Fortran tree under reference/ (skipped "
        f"when absent; restore it with `{RESTORE}`)")


def pytest_collection_modifyitems(config, items):
    if REFERENCE.is_dir():
        return
    skip = pytest.mark.skip(
        reason=f"oracle: vendored reference/ tree absent -- restore it with "
        f"`{RESTORE}`")
    for item in items:
        if item.get_closest_marker("oracle") is not None:
            item.add_marker(skip)
