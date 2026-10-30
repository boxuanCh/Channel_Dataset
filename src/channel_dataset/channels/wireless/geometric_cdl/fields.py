"""Persistent sum-of-sinusoids spatial random fields."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, log, pi, sin, sqrt
from typing import Any

from channel_dataset.backends import RandomStream, get_backend


def _radical_inverse(index: int, base: int) -> float:
    """Return one deterministic Halton coordinate in the open unit interval."""

    value = 0.0
    factor = 1.0 / base
    while index:
        value += factor * (index % base)
        index //= base
        factor /= base
    return min(max(value, 1.0e-12), 1.0 - 1.0e-12)


def _normal_quad(index: int) -> tuple[float, float, float, float]:
    """Map four low-discrepancy coordinates to four standard normals."""

    first_radius = sqrt(-2.0 * log(_radical_inverse(index, 2)))
    first_angle = 2.0 * pi * _radical_inverse(index, 3)
    second_radius = sqrt(-2.0 * log(_radical_inverse(index, 5)))
    second_angle = 2.0 * pi * _radical_inverse(index, 7)
    return (
        first_radius * cos(first_angle),
        first_radius * sin(first_angle),
        second_radius * cos(second_angle),
        second_radius * sin(second_angle),
    )


def _spectral_wave_vector(
    index: int,
    correlation_distance_m: float,
    acf: str,
) -> tuple[float, float, float]:
    """Sample the 3-D spectrum whose Fourier transform is the target ACF."""

    z0, z1, z2, z3 = _normal_quad(index)
    if acf == "gaussian":
        # If k ~ N(0, sigma^2 I), E exp(i 2 pi k.x) is
        # exp(-2 pi^2 sigma^2 |x|^2).
        scale = 1.0 / (sqrt(2.0) * pi * correlation_distance_m)
        return scale * z0, scale * z1, scale * z2
    # A multivariate Cauchy vector k = gamma Z / |G| has characteristic
    # function exp(-gamma |t|).  Evaluated at t=2*pi*x this gives the
    # requested exp(-|x|/d) covariance.
    scale = 1.0 / (2.0 * pi * correlation_distance_m * max(abs(z3), 1.0e-12))
    return scale * z0, scale * z1, scale * z2


@dataclass(frozen=True, slots=True)
class SpatialFieldBank:
    """Immutable, query-order-independent Gaussian spatial fields."""

    directions_per_m: Any
    phases_rad: Any
    correlation_distance_m: float
    acf: str = "gaussian_exponential"

    @classmethod
    def sample(
        cls,
        num_fields: int,
        num_sinusoids: int,
        correlation_distance_m: float,
        random_stream: RandomStream[Any],
        *,
        dtype: str,
        acf: str = "gaussian_exponential",
    ) -> SpatialFieldBank:
        """Create deterministic wave vectors and random realization phases."""

        if num_fields <= 0 or num_sinusoids <= 0 or correlation_distance_m <= 0:
            raise ValueError("field dimensions and correlation distance must be positive")
        if acf not in {"exponential", "gaussian", "gaussian_exponential"}:
            raise ValueError("unsupported spatial autocorrelation function")
        backend = get_backend(random_stream.backend_name)
        # Wiener-Khinchin construction: deterministic low-discrepancy wave
        # vectors sample the exact 3-D spectrum of the requested isotropic
        # ACF. Only phases are randomized per realization. Sharing the basis
        # between fields prevents LSP-specific finite-basis calibration drift;
        # independent phases keep the fields uncorrelated before LSP mixing.
        basis_one = []
        exponential_index = gaussian_index = 0
        for index in range(num_sinusoids):
            component = acf
            if acf == "gaussian_exponential":
                component = "exponential" if index % 2 == 0 else "gaussian"
            if component == "exponential":
                exponential_index += 1
                spectral_index = exponential_index
            else:
                gaussian_index += 1
                spectral_index = gaussian_index
            basis_one.append(
                _spectral_wave_vector(
                    spectral_index,
                    correlation_distance_m,
                    component,
                )
            )
        # Each LSP field needs its own deterministic spectral basis.  Reusing
        # the exact same wave vectors and changing only phases makes fields
        # independent only across an ensemble of phase draws; over space in a
        # single persistent realization it leaves a finite-basis covariance
        # proportional to cos(phi_i-phi_j).  Deterministic 3-D rotations keep
        # the isotropic spectrum (and therefore the requested ACF) while
        # removing those coincident spectral lines.  Field zero is unchanged
        # so the canonical one-field basis remains stable.
        basis = []
        golden_angle = pi * (3.0 - sqrt(5.0))
        for field in range(num_fields):
            yaw = field * golden_angle
            pitch = field * golden_angle / sqrt(2.0)
            cy, sy = cos(yaw), sin(yaw)
            cp, sp = cos(pitch), sin(pitch)
            rotated = []
            for x, y, z in basis_one:
                yaw_x, yaw_y = cy * x - sy * y, sy * x + cy * y
                rotated.append((cp * yaw_x + sp * z, yaw_y, -sp * yaw_x + cp * z))
            basis.append(rotated)
        directions = backend.array(basis, dtype=dtype, device=random_stream.device)
        phases = random_stream.fork("wireless.geometric_cdl.fields.phases").uniform(
            (num_fields, num_sinusoids), -pi, pi, dtype=dtype
        )
        return cls(directions, phases, float(correlation_distance_m), acf)

    @property
    def num_fields(self) -> int:
        return int(self.phases_rad.shape[0])

    def evaluate(self, positions_m: Any) -> Any:
        """Evaluate standard-normal-like fields at ``[..., xyz]`` positions."""

        backend = get_backend(
            "torch" if type(positions_m).__module__.startswith("torch") else "numpy"
        )
        if backend.ndim(positions_m) < 1 or backend.shape(positions_m)[-1] != 3:
            raise ValueError("positions_m must append an xyz axis")
        argument = 2.0 * pi * backend.einsum(
            "...j,flj->...fl", positions_m, self.directions_per_m
        ) + self.phases_rad
        return sqrt(2.0 / self.phases_rad.shape[1]) * backend.sum(
            backend.cos(argument), axis=-1
        )

    def evaluate_dual(self, tx_positions_m: Any, rx_positions_m: Any) -> Any:
        """Evaluate reciprocal dual-mobility fields for paired link positions."""

        backend = get_backend(
            "torch" if type(tx_positions_m).__module__.startswith("torch") else "numpy"
        )
        tx = self.evaluate(tx_positions_m)
        rx = self.evaluate(rx_positions_m)
        separation = backend.sqrt(
            backend.sum((tx_positions_m - rx_positions_m) ** 2, axis=-1)
        )
        scaled = separation / self.correlation_distance_m
        exponential = backend.exp(-scaled)
        gaussian = backend.exp(-(scaled * scaled))
        if self.acf == "exponential":
            correlation = exponential
        elif self.acf == "gaussian":
            correlation = gaussian
        else:
            correlation = 0.5 * (exponential + gaussian)
        normalization = backend.sqrt(2.0 + 2.0 * correlation)[..., None]
        return (tx + rx) / (normalization + 1.0e-30)

    @staticmethod
    def normal_to_uniform(values: Any) -> Any:
        """Map approximately standard-normal values to stable open uniforms."""

        backend = get_backend(
            "torch" if type(values).__module__.startswith("torch") else "numpy"
        )
        # Logistic normal-CDF approximation; clipping avoids log(0) in the
        # cluster initializer while remaining backend native.
        uniform = 1.0 / (1.0 + backend.exp(-1.702 * values))
        return backend.clip(uniform, 1.0e-7, 1.0 - 1.0e-7)


__all__ = ["SpatialFieldBank"]
