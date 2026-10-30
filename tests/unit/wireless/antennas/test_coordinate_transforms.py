from math import pi

import numpy as np

from channel_dataset.channels.wireless.antennas import (
    Orientation,
    global_to_local_angles,
    local_to_global_angles,
    rotation_matrix,
    spherical_basis,
)


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_spherical_basis_is_orthonormal(execution_context) -> None:
    radial, theta, phi = spherical_basis(
        [0.3, 1.2, 2.8], [-2.0, 0.4, 2.7], execution_context
    )
    basis = np.stack((_numpy(radial), _numpy(theta), _numpy(phi)), axis=-2)
    products = np.einsum("...ik,...jk->...ij", basis, basis)
    assert np.allclose(products, np.eye(3), atol=1.0e-12)
    assert np.allclose(np.cross(basis[:, 1], basis[:, 2]), basis[:, 0], atol=1.0e-12)


def test_rotation_matrix_is_orthogonal_and_has_expected_bearing(execution_context) -> None:
    matrix = _numpy(rotation_matrix(Orientation(bearing_rad=pi / 2.0), execution_context))
    assert np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-12)
    assert np.allclose(matrix @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1.0e-12)


def test_global_local_angle_round_trip(execution_context) -> None:
    orientation = Orientation(0.4, -0.3, 0.2)
    theta = [0.5, 1.0, 2.2]
    phi = [-2.4, 0.3, 2.5]
    local_theta, local_phi = global_to_local_angles(theta, phi, orientation, execution_context)
    recovered_theta, recovered_phi = local_to_global_angles(
        local_theta, local_phi, orientation, execution_context
    )
    assert np.allclose(_numpy(recovered_theta), theta, atol=1.0e-12)
    angle_error = np.angle(np.exp(1j * (_numpy(recovered_phi) - np.asarray(phi))))
    assert np.allclose(angle_error, 0.0, atol=1.0e-12)


def test_bearing_changes_global_boresight(execution_context) -> None:
    local_theta, local_phi = global_to_local_angles(
        pi / 2.0,
        pi / 2.0,
        Orientation(bearing_rad=pi / 2.0),
        execution_context,
    )
    assert np.allclose(_numpy(local_theta), pi / 2.0, atol=1.0e-12)
    assert np.allclose(_numpy(local_phi), 0.0, atol=1.0e-12)
