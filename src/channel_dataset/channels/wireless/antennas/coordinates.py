"""3GPP global/local coordinate transforms and spherical basis vectors."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import Any

from channel_dataset.backends import backend_name_for_array
from channel_dataset.channels.wireless.math._utils import as_real_array
from channel_dataset.core import ExecutionContext


@dataclass(frozen=True, slots=True)
class Orientation:
    """Intrinsic orientation of a local coordinate system.

    Parameters
    ----------
    bearing_rad : float, default=0
        First rotation about the global positive z-axis, denoted ``alpha`` by
        3GPP TR 38.901.
    downtilt_rad : float, default=0
        Second intrinsic rotation about the already-rotated y-axis, denoted
        ``beta``.
    slant_rad : float, default=0
        Third intrinsic rotation about the twice-rotated x-axis, denoted
        ``gamma``. This is mechanical array orientation, not element
        polarization slant.

    Notes
    -----
    The resulting matrix maps local Cartesian vectors into the global frame as
    ``Rz(bearing) @ Ry(downtilt) @ Rx(slant)``. Angles use radians.
    """

    bearing_rad: float = 0.0
    downtilt_rad: float = 0.0
    slant_rad: float = 0.0

    def __post_init__(self) -> None:
        for name in ("bearing_rad", "downtilt_rad", "slant_rad"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real scalar")
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")


DEFAULT_ORIENTATION = Orientation()


@dataclass(frozen=True, slots=True)
class PolarizedFieldPattern:
    """Far-field components in the spherical zenith/azimuth basis.

    Parameters
    ----------
    theta : backend array
        Field component along increasing zenith angle.
    phi : backend array
        Field component along increasing azimuth angle.

    Notes
    -----
    ``theta`` and ``phi`` have the broadcast shape of the requested angles.
    Their squared magnitudes sum to the element power gain in linear units.
    """

    theta: Any
    phi: Any

    def __post_init__(self) -> None:
        theta_backend = backend_name_for_array(self.theta)
        phi_backend = backend_name_for_array(self.phi)
        if theta_backend != phi_backend:
            raise ValueError("polarized field components must use the same backend")
        if self.theta.shape != self.phi.shape:
            raise ValueError("polarized field components must have equal shapes")


def _validate_zenith(zenith: Any, execution: ExecutionContext) -> None:
    backend = execution.array_backend
    shape = backend.shape(zenith)
    size = 1
    for dimension in shape:
        size *= dimension
    if size == 0:
        raise ValueError("zenith_rad must not be empty")
    lower = backend.to_scalar(backend.min(zenith))
    upper = backend.to_scalar(backend.max(zenith))
    tolerance = 1.0e-6 if execution.precision == "single" else 1.0e-12
    if lower < -tolerance or upper > pi + tolerance:
        raise ValueError("zenith_rad must lie in [0, pi]")


def spherical_basis(
    zenith_rad: Any,
    azimuth_rad: Any,
    execution: ExecutionContext,
) -> tuple[Any, Any, Any]:
    """Return radial, zenith, and azimuth unit vectors.

    Parameters
    ----------
    zenith_rad, azimuth_rad : scalar or array-like
        3GPP spherical angles. Zenith is measured down from positive z and
        azimuth counter-clockwise from positive x toward positive y.
    execution : ExecutionContext
        Backend, precision, and device for the returned arrays.

    Returns
    -------
    radial, theta, phi : tuple of backend arrays
        Orthonormal vectors with a final Cartesian ``[x, y, z]`` axis.
    """

    backend = execution.array_backend
    theta = as_real_array(zenith_rad, execution, "zenith_rad")
    phi = as_real_array(azimuth_rad, execution, "azimuth_rad")
    _validate_zenith(theta, execution)
    zero = theta * 0.0 + phi * 0.0
    sin_theta = backend.sin(theta)
    cos_theta = backend.cos(theta)
    sin_phi = backend.sin(phi)
    cos_phi = backend.cos(phi)
    radial = backend.stack(
        (sin_theta * cos_phi + zero, sin_theta * sin_phi + zero, cos_theta + zero),
        axis=-1,
    )
    theta_basis = backend.stack(
        (cos_theta * cos_phi + zero, cos_theta * sin_phi + zero, -sin_theta + zero),
        axis=-1,
    )
    phi_basis = backend.stack((-sin_phi + zero, cos_phi + zero, zero), axis=-1)
    return radial, theta_basis, phi_basis


def rotation_matrix(
    orientation: Orientation,
    execution: ExecutionContext,
) -> Any:
    """Return the local-to-global 3GPP orientation matrix.

    Parameters
    ----------
    orientation : Orientation
        Bearing, downtilt, and mechanical slant angles.
    execution : ExecutionContext
        Backend, precision, and device for the returned matrix.

    Returns
    -------
    backend array
        Orthogonal matrix of shape ``[3, 3]`` mapping LCS vectors to GCS.
    """

    if not isinstance(orientation, Orientation):
        raise TypeError("orientation must be an Orientation")
    backend = execution.array_backend
    a = backend.array(orientation.bearing_rad, dtype=execution.float_dtype, device=execution.device)
    b = backend.array(orientation.downtilt_rad, dtype=execution.float_dtype, device=execution.device)
    g = backend.array(orientation.slant_rad, dtype=execution.float_dtype, device=execution.device)
    ca, sa = backend.cos(a), backend.sin(a)
    cb, sb = backend.cos(b), backend.sin(b)
    cg, sg = backend.cos(g), backend.sin(g)
    zero = a * 0.0
    one = zero + 1.0
    rz = backend.stack(
        (
            backend.stack((ca, -sa, zero)),
            backend.stack((sa, ca, zero)),
            backend.stack((zero, zero, one)),
        )
    )
    ry = backend.stack(
        (
            backend.stack((cb, zero, sb)),
            backend.stack((zero, one, zero)),
            backend.stack((-sb, zero, cb)),
        )
    )
    rx = backend.stack(
        (
            backend.stack((one, zero, zero)),
            backend.stack((zero, cg, -sg)),
            backend.stack((zero, sg, cg)),
        )
    )
    return backend.matmul(backend.matmul(rz, ry), rx)


def global_to_local_angles(
    zenith_rad: Any,
    azimuth_rad: Any,
    orientation: Orientation,
    execution: ExecutionContext,
) -> tuple[Any, Any]:
    """Transform global spherical directions into a local coordinate system.

    Parameters
    ----------
    zenith_rad, azimuth_rad : scalar or array-like
        Directions in the global coordinate system.
    orientation : Orientation
        Orientation of the local frame within the global frame.
    execution : ExecutionContext
        Numerical execution controls.

    Returns
    -------
    local_zenith_rad, local_azimuth_rad : tuple of backend arrays
        Directions in the local coordinate system.
    """

    backend = execution.array_backend
    global_radial, _, _ = spherical_basis(zenith_rad, azimuth_rad, execution)
    inverse = backend.transpose(rotation_matrix(orientation, execution), (1, 0))
    local = backend.einsum("ij,...j->...i", inverse, global_radial)
    horizontal = backend.sqrt(local[..., 0] * local[..., 0] + local[..., 1] * local[..., 1])
    return backend.atan2(horizontal, local[..., 2]), backend.atan2(local[..., 1], local[..., 0])


def local_to_global_angles(
    zenith_rad: Any,
    azimuth_rad: Any,
    orientation: Orientation,
    execution: ExecutionContext,
) -> tuple[Any, Any]:
    """Transform local spherical directions into the global coordinate system.

    Parameters
    ----------
    zenith_rad, azimuth_rad : scalar or array-like
        Directions in the local coordinate system.
    orientation : Orientation
        Orientation of the local frame within the global frame.
    execution : ExecutionContext
        Numerical execution controls.

    Returns
    -------
    global_zenith_rad, global_azimuth_rad : tuple of backend arrays
        Directions in the global coordinate system.
    """

    backend = execution.array_backend
    local_radial, _, _ = spherical_basis(zenith_rad, azimuth_rad, execution)
    global_vector = backend.einsum(
        "ij,...j->...i", rotation_matrix(orientation, execution), local_radial
    )
    horizontal = backend.sqrt(
        global_vector[..., 0] * global_vector[..., 0]
        + global_vector[..., 1] * global_vector[..., 1]
    )
    return (
        backend.atan2(horizontal, global_vector[..., 2]),
        backend.atan2(global_vector[..., 1], global_vector[..., 0]),
    )


def local_field_to_global(
    local_field: PolarizedFieldPattern,
    global_zenith_rad: Any,
    global_azimuth_rad: Any,
    orientation: Orientation,
    execution: ExecutionContext,
) -> PolarizedFieldPattern:
    """Rotate polarized local field components into the global spherical basis.

    Parameters
    ----------
    local_field : PolarizedFieldPattern
        Components expressed in the LCS basis at the corresponding local
        directions.
    global_zenith_rad, global_azimuth_rad : scalar or array-like
        Corresponding directions in the GCS.
    orientation : Orientation
        LCS orientation in the GCS.
    execution : ExecutionContext
        Numerical execution controls.

    Returns
    -------
    PolarizedFieldPattern
        Field components projected onto the GCS spherical basis.
    """

    backend = execution.array_backend
    local_theta, local_phi = global_to_local_angles(
        global_zenith_rad, global_azimuth_rad, orientation, execution
    )
    _, local_theta_basis, local_phi_basis = spherical_basis(local_theta, local_phi, execution)
    local_vector = (
        local_field.theta[..., None] * local_theta_basis
        + local_field.phi[..., None] * local_phi_basis
    )
    global_vector = backend.einsum(
        "ij,...j->...i", rotation_matrix(orientation, execution), local_vector
    )
    _, global_theta_basis, global_phi_basis = spherical_basis(
        global_zenith_rad, global_azimuth_rad, execution
    )
    return PolarizedFieldPattern(
        theta=backend.sum(global_vector * global_theta_basis, axis=-1),
        phi=backend.sum(global_vector * global_phi_basis, axis=-1),
    )


__all__ = [
    "Orientation",
    "PolarizedFieldPattern",
    "global_to_local_angles",
    "local_field_to_global",
    "local_to_global_angles",
    "rotation_matrix",
    "spherical_basis",
]
