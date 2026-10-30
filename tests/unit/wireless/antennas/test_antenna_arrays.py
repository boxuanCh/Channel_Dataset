import numpy as np
import pytest
from jsonschema.validators import validator_for

from channel_dataset.channels.wireless.antennas import (
    AntennaPanel,
    MultiPanelArray,
    load_antenna_layout,
    parse_antenna_layout,
)
from channel_dataset.config import load_resource_document


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_multi_panel_positions_and_indexing(execution_context) -> None:
    array = MultiPanelArray(
        AntennaPanel(2, 2),
        panel_rows=2,
        panel_columns=2,
        vertical_panel_spacing_wavelengths=2.0,
        horizontal_panel_spacing_wavelengths=3.0,
    )
    positions = _numpy(array.positions_wavelengths(execution_context))
    assert positions.shape == (16, 3)
    assert array.element_indices(0) == (0, 0, 0, 0, 0)
    assert array.element_indices(15) == (1, 1, 1, 1, 0)
    assert np.allclose(positions.mean(axis=0), 0.0, atol=1.0e-12)
    assert np.allclose(positions[0], [0.0, -1.75, -1.25])
    assert np.allclose(positions[-1], [0.0, 1.75, 1.25])


def test_packaged_three_gpp_layout_loads() -> None:
    layout = load_antenna_layout(
        "package://profiles/wireless/antenna_3gpp_8x8_dual.yaml"
    )
    assert layout.name == "3gpp-8x8-dual-polarized"
    assert layout.array.num_elements == 128
    assert layout.metadata["review_status"] == "pending_independent_review"


def test_packaged_layout_satisfies_schema() -> None:
    schema = load_resource_document("package://schema/antenna-layout-v1.schema.json")
    document = load_resource_document(
        "package://profiles/wireless/antenna_3gpp_8x8_dual.yaml"
    )
    validator_for(schema).check_schema(schema)
    validator_for(schema)(schema).validate(document)


def test_layout_parser_rejects_unknown_fields() -> None:
    document = {
        "schema_version": 1,
        "name": "bad",
        "element": {"kind": "isotropic"},
        "panel": {
            "rows": 1,
            "columns": 1,
            "vertical_spacing_wavelengths": 0.5,
            "horizontal_spacing_wavelengths": 0.5,
            "polarization_slant_angles_degrees": [0.0],
        },
        "array": {"panel_rows": 1, "panel_columns": 1},
        "metadata": {"source": "test", "review_status": "reviewed"},
        "surprise": True,
    }
    with pytest.raises(ValueError, match="unknown layout fields"):
        parse_antenna_layout(document)
