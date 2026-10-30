"""Backend-neutral polarized antenna element patterns."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, pi
from typing import Any, Protocol, runtime_checkable

from channel_dataset.channels.wireless.math._utils import as_real_array
from channel_dataset.core import ExecutionContext

from .coordinates import (
    DEFAULT_ORIENTATION,
    Orientation,
    PolarizedFieldPattern,
    global_to_local_angles,
    local_field_to_global,
)


def _elementwise_min(left: Any, right: float, backend: Any) -> Any:
    return (left + right - backend.abs(left - right)) * 0.5


@runtime_checkable
class AntennaElement(Protocol):
    """Interface for a polarized antenna element far-field pattern."""

    def field_pattern(
        self,
        zenith_rad: Any,
        azimuth_rad: Any,
        execution: ExecutionContext,
        *,
        orientation: Orientation = DEFAULT_ORIENTATION,
        slant_angle_rad: float | None = None,
    ) -> PolarizedFieldPattern:
        """Evaluate polarized field components in the global spherical basis.

        Parameters
        ----------
        zenith_rad, azimuth_rad : scalar or array-like
            GCS directions using the 3GPP zenith-angle convention.
        execution : ExecutionContext
            Backend, precision, and device controls.
        orientation : Orientation, optional
            Orientation of the element LCS in the GCS.
        slant_angle_rad : float, optional
            Polarization slant in the LCS. If omitted, the element default is
            used.

        Returns
        -------
        PolarizedFieldPattern
            Global theta- and phi-directed field amplitudes.
        """

        ...


@dataclass(frozen=True, slots=True)
class IsotropicAntennaElement:
    """Ideal angle-independent linearly polarized antenna element.

    Parameters
    ----------
    gain_dbi : float, default=0
        Direction-independent power gain in dBi.
    slant_angle_rad : float, default=0
        Default LCS polarization slant; zero is theta-polarized.
    """

    gain_dbi: float = 0.0
    slant_angle_rad: float = 0.0

    def __post_init__(self) -> None:
        for name in ("gain_dbi", "slant_angle_rad"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real scalar")
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

    def field_pattern(
        self,
        zenith_rad: Any,
        azimuth_rad: Any,
        execution: ExecutionContext,
        *,
        orientation: Orientation = DEFAULT_ORIENTATION,
        slant_angle_rad: float | None = None,
    ) -> PolarizedFieldPattern:
        """Evaluate the isotropic polarized field pattern.

        Parameters
        ----------
        zenith_rad, azimuth_rad : scalar or array-like
            GCS directions using the 3GPP zenith-angle convention.
        execution : ExecutionContext
            Backend, precision, and device controls.
        orientation : Orientation, optional
            Orientation of the element LCS in the GCS.
        slant_angle_rad : float, optional
            LCS polarization slant overriding the element default.

        Returns
        -------
        PolarizedFieldPattern
            Global theta- and phi-directed field amplitudes.
        """

        backend = execution.array_backend
        local_theta, local_phi = global_to_local_angles(
            zenith_rad, azimuth_rad, orientation, execution
        )
        zero = local_theta * 0.0 + local_phi * 0.0
        amplitude = zero + float(10.0 ** (self.gain_dbi / 20.0))
        slant = self.slant_angle_rad if slant_angle_rad is None else slant_angle_rad
        if not isfinite(float(slant)):
            raise ValueError("slant_angle_rad must be finite")
        local = PolarizedFieldPattern(
            theta=amplitude * backend.cos(zero + float(slant)),
            phi=amplitude * backend.sin(zero + float(slant)),
        )
        return local_field_to_global(local, zenith_rad, azimuth_rad, orientation, execution)


@dataclass(frozen=True, slots=True)
class ThreeGPPAntennaElement:
    """Single-element pattern from 3GPP TR 38.901 Table 7.3-1.

    Parameters
    ----------
    maximum_gain_dbi : float, default=8
        Boresight directional power gain.
    vertical_beamwidth_rad, horizontal_beamwidth_rad : float, default=65 degrees
        Full 3 dB beamwidths in the vertical and horizontal cuts.
    vertical_sidelobe_attenuation_db : float, default=30
        Maximum attenuation in the vertical cut.
    front_back_attenuation_db : float, default=30
        Maximum combined element attenuation.
    slant_angle_rad : float, default=0
        Default LCS polarization slant for angle-independent polarization
        model 2 in clause 7.3.2.

    Notes
    -----
    ``|F_theta|**2 + |F_phi|**2`` equals the linear power gain, including the
    8 dBi default maximum gain.
    """

    maximum_gain_dbi: float = 8.0
    vertical_beamwidth_rad: float = 65.0 * pi / 180.0
    horizontal_beamwidth_rad: float = 65.0 * pi / 180.0
    vertical_sidelobe_attenuation_db: float = 30.0
    front_back_attenuation_db: float = 30.0
    slant_angle_rad: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "maximum_gain_dbi",
            "vertical_beamwidth_rad",
            "horizontal_beamwidth_rad",
            "vertical_sidelobe_attenuation_db",
            "front_back_attenuation_db",
            "slant_angle_rad",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real scalar")
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.vertical_beamwidth_rad <= 0.0 or self.horizontal_beamwidth_rad <= 0.0:
            raise ValueError("beamwidths must be positive")
        if self.vertical_sidelobe_attenuation_db < 0.0:
            raise ValueError("vertical_sidelobe_attenuation_db must be nonnegative")
        if self.front_back_attenuation_db < 0.0:
            raise ValueError("front_back_attenuation_db must be nonnegative")

    def power_gain_db(
        self,
        local_zenith_rad: Any,
        local_azimuth_rad: Any,
        execution: ExecutionContext,
    ) -> Any:
        """Evaluate the co-polar power gain in dBi in the element LCS.

        Parameters
        ----------
        local_zenith_rad, local_azimuth_rad : scalar or array-like
            LCS zenith and azimuth angles.
        execution : ExecutionContext
            Backend, precision, and device controls.

        Returns
        -------
        backend array
            Element power gain in dBi.
        """

        backend = execution.array_backend
        theta = as_real_array(local_zenith_rad, execution, "local_zenith_rad")
        phi = as_real_array(local_azimuth_rad, execution, "local_azimuth_rad")
        if backend.to_scalar(backend.min(theta)) < 0.0 or backend.to_scalar(backend.max(theta)) > pi:
            raise ValueError("local_zenith_rad must lie in [0, pi]")
        # atan2 already supplies azimuth in [-pi, pi]; direct callers receive
        # the periodic principal angle here without backend-specific remainder.
        phi = backend.atan2(backend.sin(phi), backend.cos(phi))
        vertical = 12.0 * ((theta - pi / 2.0) / self.vertical_beamwidth_rad) ** 2
        horizontal = 12.0 * (phi / self.horizontal_beamwidth_rad) ** 2
        vertical = _elementwise_min(vertical, self.vertical_sidelobe_attenuation_db, backend)
        horizontal = _elementwise_min(horizontal, self.front_back_attenuation_db, backend)
        attenuation = _elementwise_min(
            vertical + horizontal, self.front_back_attenuation_db, backend
        )
        return self.maximum_gain_dbi - attenuation

    def field_pattern(
        self,
        zenith_rad: Any,
        azimuth_rad: Any,
        execution: ExecutionContext,
        *,
        orientation: Orientation = DEFAULT_ORIENTATION,
        slant_angle_rad: float | None = None,
    ) -> PolarizedFieldPattern:
        """Evaluate the 3GPP polarized field pattern in the GCS.

        Parameters
        ----------
        zenith_rad, azimuth_rad : scalar or array-like
            GCS directions using the 3GPP zenith-angle convention.
        execution : ExecutionContext
            Backend, precision, and device controls.
        orientation : Orientation, optional
            Orientation of the element LCS in the GCS.
        slant_angle_rad : float, optional
            LCS polarization slant overriding the element default.

        Returns
        -------
        PolarizedFieldPattern
            Global theta- and phi-directed field amplitudes.
        """

        backend = execution.array_backend
        local_theta, local_phi = global_to_local_angles(
            zenith_rad, azimuth_rad, orientation, execution
        )
        gain_db = self.power_gain_db(local_theta, local_phi, execution)
        amplitude = backend.exp(gain_db * (log(10.0) / 20.0))
        slant = self.slant_angle_rad if slant_angle_rad is None else slant_angle_rad
        if not isfinite(float(slant)):
            raise ValueError("slant_angle_rad must be finite")
        local = PolarizedFieldPattern(
            theta=amplitude * backend.cos(gain_db * 0.0 + float(slant)),
            phi=amplitude * backend.sin(gain_db * 0.0 + float(slant)),
        )
        return local_field_to_global(local, zenith_rad, azimuth_rad, orientation, execution)


__all__ = ["AntennaElement", "IsotropicAntennaElement", "ThreeGPPAntennaElement"]
