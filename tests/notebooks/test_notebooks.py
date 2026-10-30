from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat", reason="install the notebooks extra")
nbclient = pytest.importorskip("nbclient", reason="install the notebooks extra")

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIRECTORY = ROOT / "examples" / "notebooks"
EXPECTED_NOTEBOOKS = (
    "00_getting_started.ipynb",
    "01_signal_domains_and_noise.ipynb",
    "02_3gpp_tdl_cdl_and_antennas.ipynb",
    "03_system_scenarios_and_blockage.ipynb",
    "04_persistence_replay_and_metrics.ipynb",
    "05_ieee80211_wlan.ipynb",
    "06_tr38811_ntn.ipynb",
    "07_tr37885_v2x.ipynb",
    "08_geometric_cdl_evolution.ipynb",
)


def test_notebook_inventory_is_complete() -> None:
    assert tuple(path.name for path in sorted(NOTEBOOK_DIRECTORY.glob("*.ipynb"))) == (
        EXPECTED_NOTEBOOKS
    )


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_is_valid_has_saved_key_outputs_and_executes(name: str) -> None:
    path = NOTEBOOK_DIRECTORY / name
    document = nbformat.read(path, as_version=4)
    nbformat.validate(document)

    setup_cell = next(cell for cell in document.cells if cell.cell_type == "code")
    assert 'SOURCE_DIRECTORY = str(ROOT / "src")' in setup_cell.source
    assert "sys.path.insert(0, SOURCE_DIRECTORY)" in setup_cell.source

    key_cells = [
        cell
        for cell in document.cells
        if cell.cell_type == "code" and "keep-output" in cell.metadata.get("tags", ())
    ]
    assert key_cells, f"{name} has no keep-output cells"
    assert all(cell.outputs for cell in key_cells), f"{name} has an empty committed key output"

    executed = deepcopy(document)
    for cell in executed.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    nbclient.NotebookClient(
        executed,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()

    errors = [
        output
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if output.output_type == "error"
    ]
    assert not errors
