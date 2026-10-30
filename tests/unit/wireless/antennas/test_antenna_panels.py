from math import pi

import numpy as np
import pytest

from channel_dataset.channels.wireless.antennas import AntennaPanel


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_panel_positions_and_indexing(execution_context) -> None:
    panel = AntennaPanel(
        rows=2,
        columns=3,
        vertical_spacing_wavelengths=0.5,
        horizontal_spacing_wavelengths=0.25,
        polarization_slant_angles_rad=(-pi / 4.0, pi / 4.0),
    )
    positions = _numpy(panel.positions_wavelengths(execution_context))
    assert positions.shape == (12, 3)
    assert panel.element_indices(0) == (0, 0, 0)
    assert panel.element_indices(1) == (0, 0, 1)
    assert panel.element_indices(11) == (1, 2, 1)
    assert np.allclose(positions[0], [0.0, -0.25, -0.25])
    assert np.allclose(positions[-1], [0.0, 0.25, 0.25])
    assert np.array_equal(positions[0], positions[1])


def test_panel_metric_positions_scale_by_wavelength(execution_context) -> None:
    panel = AntennaPanel(1, 2, horizontal_spacing_wavelengths=0.5)
    normalized = _numpy(panel.positions_wavelengths(execution_context))
    metric = _numpy(panel.positions_m(3.0e9, execution_context))
    assert np.allclose(metric, normalized * 299792458.0 / 3.0e9)


def test_invalid_dual_polarization_is_rejected() -> None:
    with pytest.raises(ValueError, match="orthogonal"):
        AntennaPanel(1, 1, polarization_slant_angles_rad=(0.0, pi / 4.0))
