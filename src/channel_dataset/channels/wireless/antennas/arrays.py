"""Panel and multi-panel antenna-array geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, isfinite, pi
from typing import Any

from channel_dataset.channels.wireless.math import wavelength_m
from channel_dataset.core import ExecutionContext

from .coordinates import DEFAULT_ORIENTATION, Orientation, PolarizedFieldPattern
from .elements import AntennaElement, IsotropicAntennaElement


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _positive_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    if not isfinite(float(value)) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_polarizations(angles: tuple[float, ...]) -> None:
    if not isinstance(angles, tuple):
        raise TypeError("polarization_slant_angles_rad must be a tuple")
    if len(angles) not in {1, 2}:
        raise ValueError("an antenna panel must be single or dual polarized")
    for angle in angles:
        if isinstance(angle, bool) or not isinstance(angle, (int, float)):
            raise TypeError("polarization slant angles must be real scalars")
        if not isfinite(float(angle)):
            raise ValueError("polarization slant angles must be finite")
    if len(angles) == 2:
        separation = abs((angles[1] - angles[0]) % pi)
        separation = min(separation, pi - separation)
        if not isclose(separation, pi / 2.0, abs_tol=1.0e-12):
            raise ValueError("dual-polarized slant angles must be orthogonal")


@dataclass(frozen=True, slots=True)
class AntennaPanel:
    """Uniform rectangular panel with one or two polarizations per location.

    Parameters
    ----------
    rows, columns : int
        Positive number of physical element locations along local z and y.
    vertical_spacing_wavelengths, horizontal_spacing_wavelengths : float, default=0.5
        Adjacent location spacing normalized by carrier wavelength.
    polarization_slant_angles_rad : tuple of float, default=(0,)
        One angle for single polarization or two orthogonal angles for dual
        polarization. ``(-pi/4, pi/4)`` is the standard cross-slant pair.
    element : AntennaElement, optional
        Element radiation model shared by all panel locations.

    Notes
    -----
    The panel lies in the local yz-plane with boresight along positive x.
    Index order is ``row, column, polarization`` with polarization fastest.
    Coordinates are centered about the panel origin; increasing row and column
    indices increase z and y respectively.
    """

    rows: int
    columns: int
    vertical_spacing_wavelengths: float = 0.5
    horizontal_spacing_wavelengths: float = 0.5
    polarization_slant_angles_rad: tuple[float, ...] = (0.0,)
    element: AntennaElement = field(default_factory=IsotropicAntennaElement)

    def __post_init__(self) -> None:
        _positive_integer(self.rows, "rows")
        _positive_integer(self.columns, "columns")
        _positive_finite(self.vertical_spacing_wavelengths, "vertical_spacing_wavelengths")
        _positive_finite(self.horizontal_spacing_wavelengths, "horizontal_spacing_wavelengths")
        _validate_polarizations(self.polarization_slant_angles_rad)
        if not isinstance(self.element, AntennaElement):
            raise TypeError("element must implement AntennaElement")

    @property
    def num_polarizations(self) -> int:
        """int: Number of polarization ports at each physical location."""

        return len(self.polarization_slant_angles_rad)

    @property
    def num_elements(self) -> int:
        """int: Total number of independently indexed antenna elements."""

        return self.rows * self.columns * self.num_polarizations

    def element_indices(self, index: int) -> tuple[int, int, int]:
        """Map a flat element index to row, column, and polarization indices.

        Parameters
        ----------
        index : int
            Flat index in ``[0, num_elements)``.

        Returns
        -------
        row, column, polarization : tuple of int
            Location and polarization indices.
        """

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if index < 0 or index >= self.num_elements:
            raise IndexError("element index is out of range")
        location, polarization = divmod(index, self.num_polarizations)
        row, column = divmod(location, self.columns)
        return row, column, polarization

    def positions_wavelengths(self, execution: ExecutionContext) -> Any:
        """Return element positions normalized by wavelength.

        Parameters
        ----------
        execution : ExecutionContext
            Backend, precision, and device for the returned array.

        Returns
        -------
        backend array
            Shape ``[num_elements, 3]`` with Cartesian ``[x, y, z]`` values.
            Dual-polarized elements at one location have equal positions.
        """

        positions: list[tuple[float, float, float]] = []
        for row in range(self.rows):
            z = (row - (self.rows - 1.0) / 2.0) * self.vertical_spacing_wavelengths
            for column in range(self.columns):
                y = (column - (self.columns - 1.0) / 2.0) * self.horizontal_spacing_wavelengths
                for _ in self.polarization_slant_angles_rad:
                    positions.append((0.0, y, z))
        return execution.array_backend.array(
            positions, dtype=execution.float_dtype, device=execution.device
        )

    def positions_m(self, carrier_frequency_hz: float, execution: ExecutionContext) -> Any:
        """Return element positions in metres at a carrier frequency.

        Parameters
        ----------
        carrier_frequency_hz : float
            Positive carrier frequency in hertz.
        execution : ExecutionContext
            Backend, precision, and device for the returned array.

        Returns
        -------
        backend array
            Shape ``[num_elements, 3]`` in metres.
        """

        return self.positions_wavelengths(execution) * wavelength_m(carrier_frequency_hz)

    def field_pattern(
        self,
        index: int,
        zenith_rad: Any,
        azimuth_rad: Any,
        execution: ExecutionContext,
        *,
        orientation: Orientation = DEFAULT_ORIENTATION,
    ) -> PolarizedFieldPattern:
        """Evaluate one indexed element's polarized field pattern.

        Parameters
        ----------
        index : int
            Flat element index.
        zenith_rad, azimuth_rad : scalar or array-like
            GCS directions.
        execution : ExecutionContext
            Numerical execution controls.
        orientation : Orientation, optional
            Orientation of the panel LCS in the GCS.

        Returns
        -------
        PolarizedFieldPattern
            Global-basis field components.
        """

        _, _, polarization = self.element_indices(index)
        return self.element.field_pattern(
            zenith_rad,
            azimuth_rad,
            execution,
            orientation=orientation,
            slant_angle_rad=self.polarization_slant_angles_rad[polarization],
        )


@dataclass(frozen=True, slots=True)
class MultiPanelArray:
    """Uniform rectangular arrangement of identical antenna panels.

    Parameters
    ----------
    panel : AntennaPanel
        Panel replicated over the array.
    panel_rows, panel_columns : int, default=1
        Number of panels along local z and y.
    vertical_panel_spacing_wavelengths, horizontal_panel_spacing_wavelengths : float, optional
        Centre-to-centre panel spacings. Defaults place adjacent panels one
        element spacing beyond the panel extent.

    Notes
    -----
    Index order is ``panel_row, panel_column, panel_element``. The complete
    geometry is centered at the LCS origin and lies in its yz-plane.
    """

    panel: AntennaPanel
    panel_rows: int = 1
    panel_columns: int = 1
    vertical_panel_spacing_wavelengths: float | None = None
    horizontal_panel_spacing_wavelengths: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.panel, AntennaPanel):
            raise TypeError("panel must be an AntennaPanel")
        _positive_integer(self.panel_rows, "panel_rows")
        _positive_integer(self.panel_columns, "panel_columns")
        if self.vertical_panel_spacing_wavelengths is not None:
            _positive_finite(
                self.vertical_panel_spacing_wavelengths,
                "vertical_panel_spacing_wavelengths",
            )
        if self.horizontal_panel_spacing_wavelengths is not None:
            _positive_finite(
                self.horizontal_panel_spacing_wavelengths,
                "horizontal_panel_spacing_wavelengths",
            )

    @property
    def num_elements(self) -> int:
        """int: Total number of antenna elements across every panel."""

        return self.panel_rows * self.panel_columns * self.panel.num_elements

    @property
    def vertical_panel_spacing(self) -> float:
        """float: Resolved vertical centre spacing in wavelengths."""

        if self.vertical_panel_spacing_wavelengths is not None:
            return self.vertical_panel_spacing_wavelengths
        return self.panel.rows * self.panel.vertical_spacing_wavelengths

    @property
    def horizontal_panel_spacing(self) -> float:
        """float: Resolved horizontal centre spacing in wavelengths."""

        if self.horizontal_panel_spacing_wavelengths is not None:
            return self.horizontal_panel_spacing_wavelengths
        return self.panel.columns * self.panel.horizontal_spacing_wavelengths

    def element_indices(self, index: int) -> tuple[int, int, int, int, int]:
        """Map a flat index to panel row/column and panel element indices.

        Parameters
        ----------
        index : int
            Flat array-element index.

        Returns
        -------
        panel_row, panel_column, row, column, polarization : tuple of int
            Hierarchical indices following the documented ordering.
        """

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if index < 0 or index >= self.num_elements:
            raise IndexError("array element index is out of range")
        panel_index, element_index = divmod(index, self.panel.num_elements)
        panel_row, panel_column = divmod(panel_index, self.panel_columns)
        row, column, polarization = self.panel.element_indices(element_index)
        return panel_row, panel_column, row, column, polarization

    def positions_wavelengths(self, execution: ExecutionContext) -> Any:
        """Return all element positions normalized by wavelength.

        Parameters
        ----------
        execution : ExecutionContext
            Backend, precision, and device for the returned array.

        Returns
        -------
        backend array
            Shape ``[num_elements, 3]`` ordered by panel then element.
        """

        base = self.panel.positions_wavelengths(execution)
        blocks = []
        backend = execution.array_backend
        for panel_row in range(self.panel_rows):
            z = (panel_row - (self.panel_rows - 1.0) / 2.0) * self.vertical_panel_spacing
            for panel_column in range(self.panel_columns):
                y = (
                    panel_column - (self.panel_columns - 1.0) / 2.0
                ) * self.horizontal_panel_spacing
                offset = backend.array(
                    [0.0, y, z], dtype=execution.float_dtype, device=execution.device
                )
                blocks.append(base + offset)
        return backend.concatenate(tuple(blocks), axis=0)

    def positions_m(self, carrier_frequency_hz: float, execution: ExecutionContext) -> Any:
        """Return all element positions in metres at a carrier frequency.

        Parameters
        ----------
        carrier_frequency_hz : float
            Positive carrier frequency in hertz.
        execution : ExecutionContext
            Backend, precision, and device for the returned array.

        Returns
        -------
        backend array
            Shape ``[num_elements, 3]`` in metres.
        """

        return self.positions_wavelengths(execution) * wavelength_m(carrier_frequency_hz)

    def field_pattern(
        self,
        index: int,
        zenith_rad: Any,
        azimuth_rad: Any,
        execution: ExecutionContext,
        *,
        orientation: Orientation = DEFAULT_ORIENTATION,
    ) -> PolarizedFieldPattern:
        """Evaluate one array element's polarized field pattern.

        Parameters
        ----------
        index : int
            Flat array-element index.
        zenith_rad, azimuth_rad : scalar or array-like
            GCS directions.
        execution : ExecutionContext
            Numerical execution controls.
        orientation : Orientation, optional
            Array orientation in the GCS.

        Returns
        -------
        PolarizedFieldPattern
            Global-basis field components.
        """

        *_, row, column, polarization = self.element_indices(index)
        panel_index = ((row * self.panel.columns) + column) * self.panel.num_polarizations
        panel_index += polarization
        return self.panel.field_pattern(
            panel_index,
            zenith_rad,
            azimuth_rad,
            execution,
            orientation=orientation,
        )


__all__ = ["AntennaPanel", "MultiPanelArray"]
