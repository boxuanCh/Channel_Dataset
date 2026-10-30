"""Spatial correlation matrix construction."""

from __future__ import annotations

from cmath import exp
from math import isfinite, pi, sin
from typing import Any

from channel_dataset.core import ExecutionContext


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def exponential_correlation(
    num_antennas: int,
    coefficient: complex,
    execution: ExecutionContext,
) -> Any:
    r"""Construct a Hermitian exponential antenna-correlation matrix.

    Parameters
    ----------
    num_antennas : int
        Positive matrix dimension.
    coefficient : complex
        Adjacent-element correlation :math:`\rho`, constrained by
        :math:`|\rho| \leq 1`.
    execution : ExecutionContext
        Backend, complex precision, and device for the result.

    Returns
    -------
    backend array
        Matrix with ``R[i, j] = rho**(j-i)`` above the diagonal and its
        Hermitian conjugate below the diagonal.

    Raises
    ------
    TypeError
        If ``num_antennas`` is not an integer or ``coefficient`` is not numeric.
    ValueError
        If the dimension is nonpositive or the coefficient is non-finite or
        has magnitude greater than one.
    """

    _positive_integer(num_antennas, "num_antennas")
    if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float, complex)):
        raise TypeError("coefficient must be numeric")
    rho = complex(coefficient)
    if not isfinite(rho.real) or not isfinite(rho.imag) or abs(rho) > 1.0:
        raise ValueError("coefficient must be finite with magnitude at most one")
    matrix = [
        [rho ** (column - row) if column >= row else rho.conjugate() ** (row - column)
         for column in range(num_antennas)]
        for row in range(num_antennas)
    ]
    return execution.array_backend.array(
        matrix, dtype=execution.complex_dtype, device=execution.device
    )


def one_ring_correlation(
    num_antennas: int,
    mean_azimuth_rad: float,
    angular_spread_rad: float,
    execution: ExecutionContext,
    *,
    element_spacing_wavelengths: float = 0.5,
    quadrature_points: int = 256,
) -> Any:
    r"""Approximate uniform one-ring correlation for a linear array.

    Parameters
    ----------
    num_antennas : int
        Positive number of uniformly spaced elements.
    mean_azimuth_rad : float
        Mean arrival azimuth in radians, measured from array broadside.
    angular_spread_rad : float
        Nonnegative half-width of the uniform angular distribution. Values up
        to :math:`\pi` are supported.
    execution : ExecutionContext
        Backend, precision, and device for the result.
    element_spacing_wavelengths : float, default=0.5
        Positive inter-element spacing normalized by wavelength.
    quadrature_points : int, default=256
        Number of deterministic midpoint samples used for angular integration.

    Returns
    -------
    backend array
        Hermitian positive-semidefinite correlation matrix with unit diagonal.

    Notes
    -----
    The matrix is the midpoint-quadrature average of steering-vector outer
    products. This construction preserves positive semidefiniteness for every
    supported number of quadrature points.
    """

    _positive_integer(num_antennas, "num_antennas")
    _positive_integer(quadrature_points, "quadrature_points")
    values = (mean_azimuth_rad, angular_spread_rad, element_spacing_wavelengths)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise TypeError("angles and element spacing must be real numbers")
    mean = float(mean_azimuth_rad)
    spread = float(angular_spread_rad)
    spacing = float(element_spacing_wavelengths)
    if not all(isfinite(value) for value in (mean, spread, spacing)):
        raise ValueError("angles and element spacing must be finite")
    if not 0.0 <= spread <= pi:
        raise ValueError("angular_spread_rad must be between zero and pi")
    if spacing <= 0.0:
        raise ValueError("element_spacing_wavelengths must be positive")

    if spread == 0.0:
        angles = (mean,)
    else:
        width = 2.0 * spread
        angles = tuple(
            mean - spread + width * (sample + 0.5) / quadrature_points
            for sample in range(quadrature_points)
        )
    steering = [
        [exp(1j * 2.0 * pi * spacing * element * sin(angle)) for element in range(num_antennas)]
        for angle in angles
    ]
    normalization = float(len(steering))
    matrix = [
        [
            sum(vector[row] * vector[column].conjugate() for vector in steering)
            / normalization
            for column in range(num_antennas)
        ]
        for row in range(num_antennas)
    ]
    return execution.array_backend.array(
        matrix, dtype=execution.complex_dtype, device=execution.device
    )


__all__ = ["exponential_correlation", "one_ring_correlation"]
