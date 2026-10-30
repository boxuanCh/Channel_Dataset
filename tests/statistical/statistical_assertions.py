"""Small dependency-free statistical assertions used by certification tests.

The helpers deliberately derive tolerances from sample count and a requested
normal-approximation confidence multiplier.  They avoid SciPy so statistical
validation remains part of the normal development installation.
"""

from __future__ import annotations

from math import sqrt

import numpy as np


def as_numpy(value: object) -> np.ndarray:
    """Return a detached CPU NumPy view of a supported backend array."""

    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().resolve_conj().cpu().numpy()  # type: ignore[union-attr]
    return np.asarray(value)


def assert_mean(
    samples: object,
    expected_mean: float,
    expected_std: float,
    *,
    z: float = 5.0,
    absolute_floor: float = 1.0e-12,
) -> None:
    """Assert a sample mean using its theoretical standard error."""

    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    tolerance = max(absolute_floor, z * expected_std / sqrt(values.size))
    error = abs(float(np.mean(values)) - expected_mean)
    assert error <= tolerance, (
        f"mean error {error:.6g} exceeds {z:g}-sigma tolerance {tolerance:.6g}"
    )


def assert_standard_deviation(
    samples: object,
    expected_std: float,
    *,
    z: float = 5.0,
    relative_floor: float = 0.01,
) -> None:
    """Assert a Gaussian sample standard deviation with a finite-sample bound."""

    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    # For Gaussian samples, sd(s) is approximately sigma/sqrt(2(n-1)).
    relative = max(relative_floor, z / sqrt(2.0 * (values.size - 1)))
    error = abs(float(np.std(values, ddof=1)) - expected_std)
    assert error <= relative * expected_std, (
        f"standard-deviation error {error:.6g} exceeds tolerance "
        f"{relative * expected_std:.6g}"
    )


def assert_bernoulli_rate(
    samples: object,
    probability: float,
    *,
    z: float = 5.0,
    absolute_floor: float = 0.005,
) -> None:
    """Assert a Bernoulli frequency using a conservative normal interval."""

    values = np.asarray(samples, dtype=bool).reshape(-1)
    standard_error = sqrt(probability * (1.0 - probability) / values.size)
    tolerance = max(absolute_floor, z * standard_error)
    error = abs(float(np.mean(values)) - probability)
    assert error <= tolerance, (
        f"rate error {error:.6g} exceeds tolerance {tolerance:.6g}"
    )


def assert_correlation_matrix(
    samples: object,
    expected: object,
    *,
    z: float = 5.0,
    absolute_floor: float = 0.035,
) -> None:
    """Assert Pearson correlations using Fisher-transform sampling error."""

    values = np.asarray(samples, dtype=np.float64)
    target = np.asarray(expected, dtype=np.float64)
    if values.ndim != 2 or target.shape != (values.shape[1], values.shape[1]):
        raise ValueError("samples and expected correlation matrix have incompatible shapes")
    observed = np.corrcoef(values, rowvar=False)
    tolerance = max(absolute_floor, z / sqrt(values.shape[0] - 3.0))
    mask = ~np.eye(target.shape[0], dtype=bool)
    maximum_error = float(np.max(np.abs(observed[mask] - target[mask])))
    assert maximum_error <= tolerance, (
        f"maximum correlation error {maximum_error:.6g} exceeds tolerance {tolerance:.6g}"
    )


def assert_exponential_power(
    complex_samples: object,
    expected_mean_power: float,
    *,
    z: float = 5.0,
) -> None:
    """Assert moments and quantiles of circular complex Gaussian power."""

    values = np.asarray(complex_samples).reshape(-1)
    power = np.abs(values) ** 2
    assert_mean(power, expected_mean_power, expected_mean_power, z=z)
    assert_standard_deviation(power, expected_mean_power, z=z, relative_floor=0.02)
    # Exponential quantiles provide a distribution-shape check beyond moments.
    for probability in (0.1, 0.5, 0.9):
        expected = -expected_mean_power * np.log1p(-probability)
        observed = float(np.quantile(power, probability))
        # The asymptotic quantile SE simplifies to sigma*p-dependent factors.
        density = np.exp(-expected / expected_mean_power) / expected_mean_power
        standard_error = sqrt(probability * (1.0 - probability) / power.size) / density
        tolerance = max(0.025 * expected_mean_power, z * standard_error)
        assert abs(observed - expected) <= tolerance


def assert_unit_circularity(complex_samples: object, *, tolerance: float = 0.035) -> None:
    """Assert zero mean, equal I/Q variance, and negligible pseudo-covariance."""

    values = np.asarray(complex_samples).reshape(-1)
    power = float(np.mean(np.abs(values) ** 2))
    assert abs(np.mean(values)) <= tolerance * sqrt(power)
    assert abs(float(np.var(values.real) - np.var(values.imag))) <= tolerance * power
    assert abs(np.mean(values**2)) <= tolerance * power
