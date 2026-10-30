"""Validated batched wireless response value objects."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from typing import Any

from channel_dataset.backends import backend_name_for_array, get_backend
from channel_dataset.core import Direction, PathCluster, PropagationPath, RealizationMetadata
from channel_dataset.core.axes import (
    CIR_COEFFICIENT_AXES,
    CIR_DELAY_AXES,
    CLUSTER_AXES,
    FLAT_RESPONSE_AXES,
    LINK_AXES,
    OFDM_RESPONSE_AXES,
    RAY_AXES,
    TIME_RESPONSE_AXES,
)


def _validate_metadata_array(value: Any, metadata: RealizationMetadata, field: str) -> Any:
    backend_name = backend_name_for_array(value)
    if backend_name != metadata.backend:
        raise ValueError(f"{field} backend does not match realization metadata")
    backend = get_backend(backend_name)
    if backend.device(value) != metadata.device:
        raise ValueError(f"{field} device does not match realization metadata")
    return backend


def _expected_dtype(backend: Any, metadata: RealizationMetadata, *, complex_: bool) -> Any:
    if complex_:
        name = "complex64" if metadata.precision == "single" else "complex128"
    else:
        name = "float32" if metadata.precision == "single" else "float64"
    return backend.resolve_dtype(name)


def _validate_real_array(
    value: Any,
    metadata: RealizationMetadata,
    field: str,
    *,
    rank: int,
    nonnegative: bool = False,
) -> Any:
    backend = _validate_metadata_array(value, metadata, field)
    if backend.ndim(value) != rank:
        raise ValueError(f"{field} must have rank {rank}")
    if not backend.is_floating(value):
        raise TypeError(f"{field} must have a real floating dtype")
    if backend.dtype(value) != _expected_dtype(backend, metadata, complex_=False):
        raise TypeError(f"{field} dtype does not match realization precision")
    if not backend.all_finite(value):
        raise ValueError(f"{field} must contain only finite values")
    if nonnegative and not backend.all_nonnegative(value):
        raise ValueError(f"{field} must be nonnegative")
    return backend


def _validate_complex_array(
    value: Any, metadata: RealizationMetadata, field: str, *, rank: int
) -> Any:
    backend = _validate_metadata_array(value, metadata, field)
    if backend.ndim(value) != rank:
        raise ValueError(f"{field} must have rank {rank}")
    if not backend.is_complex(value):
        raise TypeError(f"{field} must have a complex dtype")
    if backend.dtype(value) != _expected_dtype(backend, metadata, complex_=True):
        raise TypeError(f"{field} dtype does not match realization precision")
    if not backend.all_finite(value):
        raise ValueError(f"{field} must contain only finite values")
    return backend


def _validate_bool_array(
    value: Any, metadata: RealizationMetadata, field: str, *, rank: int
) -> Any:
    backend = _validate_metadata_array(value, metadata, field)
    if backend.ndim(value) != rank:
        raise ValueError(f"{field} must have rank {rank}")
    if not backend.is_bool(value):
        raise TypeError(f"{field} must have a boolean dtype")
    return backend


def _validate_integer_array(
    value: Any, metadata: RealizationMetadata, field: str, *, rank: int
) -> Any:
    backend = _validate_metadata_array(value, metadata, field)
    if backend.ndim(value) != rank:
        raise ValueError(f"{field} must have rank {rank}")
    if not backend.is_integer(value):
        raise TypeError(f"{field} must have an integer dtype")
    return backend


def _checked_index(index: int, size: int, name: str) -> slice:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= index < size:
        raise IndexError(f"{name} {index} is outside [0, {size})")
    return slice(index, index + 1)


def _ensure_positive_shape(shape: tuple[int, ...], field: str) -> None:
    if any(dimension <= 0 for dimension in shape):
        raise ValueError(f"all {field} dimensions must be positive")


@dataclass(frozen=True, slots=True)
class CIRBatch:
    """Batched channel impulse response in canonical wireless axis order.

    Parameters
    ----------
    coefficients : backend array
        Complex array with axes
        ``[batch, rx, rx_antenna, tx, tx_antenna, path, time]``.
    delays_s : backend array
        Nonnegative real delays in seconds with axes
        ``[batch, rx, tx, path]``.
    metadata : RealizationMetadata
        Backend, precision, device, model identity, and provenance.
    path_mask : backend array, optional
        Boolean validity mask with the same shape as ``delays_s``.

    Raises
    ------
    TypeError
        If an array dtype is incompatible with its semantic field or declared
        precision.
    ValueError
        If ranks, dimensions, backend/device ownership, finiteness, or delays
        are invalid.

    Notes
    -----
    The dataclass is frozen, but wrapped arrays retain their native mutability.
    Use :meth:`select_link` to obtain array views with semantic axes preserved.
    """

    coefficients: Any
    delays_s: Any
    metadata: RealizationMetadata
    path_mask: Any | None = None

    def __post_init__(self) -> None:
        backend = _validate_metadata_array(self.coefficients, self.metadata, "coefficients")
        delays_backend = _validate_metadata_array(self.delays_s, self.metadata, "delays_s")
        if backend.name != delays_backend.name:
            raise ValueError("coefficients and delays_s must use the same backend")
        if backend.ndim(self.coefficients) != len(CIR_COEFFICIENT_AXES):
            raise ValueError("coefficients must have rank 7")
        if backend.ndim(self.delays_s) != len(CIR_DELAY_AXES):
            raise ValueError("delays_s must have rank 4")
        if not backend.is_complex(self.coefficients):
            raise TypeError("coefficients must have a complex dtype")
        if not backend.is_floating(self.delays_s):
            raise TypeError("delays_s must have a real floating dtype")
        expected_complex = backend.resolve_dtype(
            "complex64" if self.metadata.precision == "single" else "complex128"
        )
        expected_float = backend.resolve_dtype(
            "float32" if self.metadata.precision == "single" else "float64"
        )
        if backend.dtype(self.coefficients) != expected_complex:
            raise TypeError("coefficients dtype does not match realization precision")
        if backend.dtype(self.delays_s) != expected_float:
            raise TypeError("delays_s dtype does not match realization precision")
        if not backend.all_finite(self.coefficients):
            raise ValueError("coefficients must contain only finite values")
        if not backend.all_finite(self.delays_s):
            raise ValueError("delays_s must contain only finite values")
        if not backend.all_nonnegative(self.delays_s):
            raise ValueError("delays_s must be nonnegative")

        coefficient_shape = backend.shape(self.coefficients)
        delay_shape = backend.shape(self.delays_s)
        expected_delay_shape = (
            coefficient_shape[0], coefficient_shape[1], coefficient_shape[3],
            coefficient_shape[5],
        )
        if delay_shape != expected_delay_shape:
            raise ValueError(
                f"delays_s shape {delay_shape} does not match coefficients {expected_delay_shape}"
            )
        if any(dimension <= 0 for dimension in coefficient_shape):
            raise ValueError("all CIR dimensions must be positive")

        if self.path_mask is not None:
            mask_backend = _validate_metadata_array(self.path_mask, self.metadata, "path_mask")
            if mask_backend.name != backend.name:
                raise ValueError("path_mask must use the same backend as coefficients")
            if not backend.is_bool(self.path_mask):
                raise TypeError("path_mask must have a boolean dtype")
            if backend.shape(self.path_mask) != delay_shape:
                raise ValueError("path_mask shape must match delays_s")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the canonical coefficient shape.

        Returns
        -------
        tuple of int
            Seven coefficient dimensions in canonical axis order.
        """

        return get_backend(self.metadata.backend).shape(self.coefficients)

    def select_link(self, batch_index: int, rx: int, tx: int) -> CIRBatch:
        """Select one batch entry and transmitter/receiver link.

        Parameters
        ----------
        batch_index, rx, tx : int
            Zero-based indices into the corresponding semantic axes.

        Returns
        -------
        CIRBatch
            View with batch, receiver, and transmitter dimensions equal to one.

        Raises
        ------
        TypeError
            If an index is not an integer.
        IndexError
            If an index is outside its axis bounds.

        Notes
        -----
        NumPy and PyTorch basic slicing is used, so array storage is shared.
        """

        shape = self.shape
        batch_slice = _checked_index(batch_index, shape[0], "batch_index")
        rx_slice = _checked_index(rx, shape[1], "rx")
        tx_slice = _checked_index(tx, shape[3], "tx")
        coefficients = self.coefficients[batch_slice, rx_slice, :, tx_slice, :, :, :]
        delays = self.delays_s[batch_slice, rx_slice, tx_slice, :]
        mask = (
            None
            if self.path_mask is None
            else self.path_mask[batch_slice, rx_slice, tx_slice, :]
        )
        return CIRBatch(coefficients, delays, self.metadata, mask)


@dataclass(frozen=True, slots=True)
class OFDMChannelResponse:
    """Batched OFDM channel frequency response.

    Parameters
    ----------
    coefficients : backend array
        Complex coefficients with axes
        ``[batch, rx, rx_ant, tx, tx_ant, symbol, subcarrier]``.
    metadata : RealizationMetadata
        Backend, precision, device, model identity, and provenance.
    """

    coefficients: Any
    metadata: RealizationMetadata

    def __post_init__(self) -> None:
        backend = _validate_metadata_array(self.coefficients, self.metadata, "coefficients")
        if backend.ndim(self.coefficients) != len(OFDM_RESPONSE_AXES):
            raise ValueError("OFDM response coefficients must have rank 7")
        if not backend.is_complex(self.coefficients):
            raise TypeError("OFDM response coefficients must have a complex dtype")
        expected_complex = backend.resolve_dtype(
            "complex64" if self.metadata.precision == "single" else "complex128"
        )
        if backend.dtype(self.coefficients) != expected_complex:
            raise TypeError("OFDM response dtype does not match realization precision")
        if not backend.all_finite(self.coefficients):
            raise ValueError("OFDM response coefficients must contain only finite values")
        if any(dimension <= 0 for dimension in backend.shape(self.coefficients)):
            raise ValueError("all OFDM response dimensions must be positive")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the seven-dimensional coefficient shape."""

        return get_backend(self.metadata.backend).shape(self.coefficients)

    def select_link(self, batch_index: int, rx: int, tx: int) -> OFDMChannelResponse:
        """Return a storage-sharing view for one batch/link.

        Parameters
        ----------
        batch_index, rx, tx : int
            Zero-based semantic-axis indices.

        Returns
        -------
        OFDMChannelResponse
            Response with singleton batch, receiver, and transmitter axes.
        """

        shape = self.shape
        batch_slice = _checked_index(batch_index, shape[0], "batch_index")
        rx_slice = _checked_index(rx, shape[1], "rx")
        tx_slice = _checked_index(tx, shape[3], "tx")
        values = self.coefficients[batch_slice, rx_slice, :, tx_slice, :, :, :]
        return OFDMChannelResponse(values, self.metadata)


@dataclass(frozen=True, slots=True)
class TimeChannelResponse:
    """Batched discrete-time channel response.

    Parameters
    ----------
    coefficients : backend array
        Complex response with axes
        ``[batch, rx, rx_ant, tx, tx_ant, output_time, lag]``.
    metadata : RealizationMetadata
        Backend, precision, device, model identity, and provenance.
    lag_min : int, default=0
        Integer sample lag represented by the final-axis element at index zero.
    """

    coefficients: Any
    metadata: RealizationMetadata
    lag_min: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.lag_min, bool) or not isinstance(self.lag_min, int):
            raise TypeError("lag_min must be an integer")
        backend = _validate_complex_array(
            self.coefficients,
            self.metadata,
            "coefficients",
            rank=len(TIME_RESPONSE_AXES),
        )
        _ensure_positive_shape(backend.shape(self.coefficients), "time response")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the seven-dimensional coefficient shape."""

        return get_backend(self.metadata.backend).shape(self.coefficients)

    @property
    def lag_max(self) -> int:
        """Return the inclusive largest integer sample lag."""

        return self.lag_min + self.shape[-1] - 1

    def select_link(self, batch_index: int, rx: int, tx: int) -> TimeChannelResponse:
        """Return a storage-sharing view for one batch/link.

        Parameters
        ----------
        batch_index, rx, tx : int
            Zero-based semantic-axis indices.

        Returns
        -------
        TimeChannelResponse
            Response with singleton batch, receiver, and transmitter axes.
        """

        shape = self.shape
        batch_slice = _checked_index(batch_index, shape[0], "batch_index")
        rx_slice = _checked_index(rx, shape[1], "rx")
        tx_slice = _checked_index(tx, shape[3], "tx")
        values = self.coefficients[batch_slice, rx_slice, :, tx_slice, :, :, :]
        return TimeChannelResponse(values, self.metadata, self.lag_min)


@dataclass(frozen=True, slots=True)
class FlatChannelResponse:
    """Batched flat-fading multi-link channel response.

    Parameters
    ----------
    coefficients : backend array
        Complex response with axes
        ``[batch, rx, rx_ant, tx, tx_ant]``.
    metadata : RealizationMetadata
        Backend, precision, device, model identity, and provenance.
    """

    coefficients: Any
    metadata: RealizationMetadata

    def __post_init__(self) -> None:
        backend = _validate_complex_array(
            self.coefficients,
            self.metadata,
            "coefficients",
            rank=len(FLAT_RESPONSE_AXES),
        )
        _ensure_positive_shape(backend.shape(self.coefficients), "flat response")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the five-dimensional coefficient shape."""

        return get_backend(self.metadata.backend).shape(self.coefficients)

    def select_link(self, batch_index: int, rx: int, tx: int) -> FlatChannelResponse:
        """Return a storage-sharing view for one batch/link.

        Parameters
        ----------
        batch_index, rx, tx : int
            Zero-based semantic-axis indices.

        Returns
        -------
        FlatChannelResponse
            Response with singleton batch, receiver, and transmitter axes.
        """

        shape = self.shape
        batch_slice = _checked_index(batch_index, shape[0], "batch_index")
        rx_slice = _checked_index(rx, shape[1], "rx")
        tx_slice = _checked_index(tx, shape[3], "tx")
        values = self.coefficients[batch_slice, rx_slice, :, tx_slice, :]
        return FlatChannelResponse(values, self.metadata)


@dataclass(frozen=True, slots=True)
class TopologyBatch:
    """Batched transmitter/receiver geometry in a shared Cartesian frame.

    Parameters
    ----------
    tx_positions_m, rx_positions_m : backend array
        Cartesian endpoint positions in metres with axes
        ``[batch, endpoint, xyz]``.
    tx_velocities_mps, rx_velocities_mps : backend array
        Cartesian velocities in metres per second with the corresponding
        endpoint shape.
    tx_orientations_rad, rx_orientations_rad : backend array
        Yaw, pitch, and roll in radians with axes
        ``[batch, endpoint, orientation]``.
    rx_indoor : backend array
        Boolean receiver indoor state with axes ``[batch, rx]``.
    tx_site_ids : backend array
        Integer site identifiers with axes ``[batch, tx]``.
    metadata : RealizationMetadata
        Backend, precision, device, generator identity, and provenance.
    los_mask : backend array, optional
        Fixed Boolean line-of-sight state with axes ``[batch, rx, tx]``.
    link_mask : backend array, optional
        Boolean validity mask with axes ``[batch, rx, tx]``.

    Raises
    ------
    TypeError
        If a field has an incompatible dtype.
    ValueError
        If ranks, endpoint dimensions, backend/device ownership, or finite
        values are inconsistent.
    """

    tx_positions_m: Any
    rx_positions_m: Any
    tx_velocities_mps: Any
    rx_velocities_mps: Any
    tx_orientations_rad: Any
    rx_orientations_rad: Any
    rx_indoor: Any
    tx_site_ids: Any
    metadata: RealizationMetadata
    los_mask: Any | None = None
    link_mask: Any | None = None

    def __post_init__(self) -> None:
        vector_fields = (
            ("tx_positions_m", self.tx_positions_m),
            ("rx_positions_m", self.rx_positions_m),
            ("tx_velocities_mps", self.tx_velocities_mps),
            ("rx_velocities_mps", self.rx_velocities_mps),
            ("tx_orientations_rad", self.tx_orientations_rad),
            ("rx_orientations_rad", self.rx_orientations_rad),
        )
        backend = None
        for name, value in vector_fields:
            backend = _validate_real_array(value, self.metadata, name, rank=3)
            shape = backend.shape(value)
            if shape[-1] != 3:
                raise ValueError(f"{name} final dimension must be xyz or yaw/pitch/roll")
        assert backend is not None
        tx_shape = backend.shape(self.tx_positions_m)
        rx_shape = backend.shape(self.rx_positions_m)
        _ensure_positive_shape(tx_shape, "topology")
        _ensure_positive_shape(rx_shape, "topology")
        for name, value in vector_fields[2:]:
            expected = tx_shape if name.startswith("tx_") else rx_shape
            if backend.shape(value) != expected:
                raise ValueError(f"{name} shape must match its endpoint positions")
        if tx_shape[0] != rx_shape[0]:
            raise ValueError("transmitter and receiver topology batch dimensions must match")
        _validate_bool_array(self.rx_indoor, self.metadata, "rx_indoor", rank=2)
        if backend.shape(self.rx_indoor) != rx_shape[:2]:
            raise ValueError("rx_indoor shape must match receiver endpoints")
        _validate_integer_array(self.tx_site_ids, self.metadata, "tx_site_ids", rank=2)
        if backend.shape(self.tx_site_ids) != tx_shape[:2]:
            raise ValueError("tx_site_ids shape must match transmitter endpoints")
        link_shape = (tx_shape[0], rx_shape[1], tx_shape[1])
        for name, value in (("los_mask", self.los_mask), ("link_mask", self.link_mask)):
            if value is not None:
                _validate_bool_array(value, self.metadata, name, rank=len(LINK_AXES))
                if backend.shape(value) != link_shape:
                    raise ValueError(f"{name} shape must be {link_shape}")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``(batch, rx, tx)`` topology dimensions."""

        backend = get_backend(self.metadata.backend)
        tx_shape = backend.shape(self.tx_positions_m)
        rx_shape = backend.shape(self.rx_positions_m)
        return (tx_shape[0], rx_shape[1], tx_shape[1])

    def select_link(self, batch_index: int, rx: int, tx: int) -> TopologyBatch:
        """Select one batch entry and transmitter/receiver endpoint pair.

        Parameters
        ----------
        batch_index, rx, tx : int
            Zero-based semantic-axis indices.

        Returns
        -------
        TopologyBatch
            Storage-sharing endpoint views with shape ``(1, 1, 1)``.
        """

        batch_size, num_rx, num_tx = self.shape
        batch_slice = _checked_index(batch_index, batch_size, "batch_index")
        rx_slice = _checked_index(rx, num_rx, "rx")
        tx_slice = _checked_index(tx, num_tx, "tx")
        los = (
            None
            if self.los_mask is None
            else self.los_mask[batch_slice, rx_slice, tx_slice]
        )
        links = (
            None
            if self.link_mask is None
            else self.link_mask[batch_slice, rx_slice, tx_slice]
        )
        return TopologyBatch(
            tx_positions_m=self.tx_positions_m[batch_slice, tx_slice, :],
            rx_positions_m=self.rx_positions_m[batch_slice, rx_slice, :],
            tx_velocities_mps=self.tx_velocities_mps[batch_slice, tx_slice, :],
            rx_velocities_mps=self.rx_velocities_mps[batch_slice, rx_slice, :],
            tx_orientations_rad=self.tx_orientations_rad[batch_slice, tx_slice, :],
            rx_orientations_rad=self.rx_orientations_rad[batch_slice, rx_slice, :],
            rx_indoor=self.rx_indoor[batch_slice, rx_slice],
            tx_site_ids=self.tx_site_ids[batch_slice, tx_slice],
            metadata=self.metadata,
            los_mask=los,
            link_mask=links,
        )


@dataclass(frozen=True, slots=True)
class LargeScaleParameters:
    """Batched large-scale propagation parameters.

    Parameters
    ----------
    delay_spread_s : backend array
        Nonnegative RMS delay spread in seconds.
    azimuth_spread_arrival_rad, azimuth_spread_departure_rad : backend array
        Nonnegative azimuth spreads in radians.
    zenith_spread_arrival_rad, zenith_spread_departure_rad : backend array
        Nonnegative zenith spreads in radians.
    shadow_fading_db : backend array
        Signed shadow-fading value in decibels.
    k_factor_db : backend array
        Signed Ricean K-factor in decibels.
    path_loss_db : backend array
        Path loss in decibels.
    metadata : RealizationMetadata
        Backend, precision, device, model identity, and provenance.
    link_mask : backend array, optional
        Boolean link validity mask.

    Notes
    -----
    Every quantity and mask uses axes ``[batch, rx, tx]``.
    """

    delay_spread_s: Any
    azimuth_spread_arrival_rad: Any
    azimuth_spread_departure_rad: Any
    zenith_spread_arrival_rad: Any
    zenith_spread_departure_rad: Any
    shadow_fading_db: Any
    k_factor_db: Any
    path_loss_db: Any
    metadata: RealizationMetadata
    link_mask: Any | None = None

    def __post_init__(self) -> None:
        nonnegative_fields = (
            ("delay_spread_s", self.delay_spread_s),
            ("azimuth_spread_arrival_rad", self.azimuth_spread_arrival_rad),
            ("azimuth_spread_departure_rad", self.azimuth_spread_departure_rad),
            ("zenith_spread_arrival_rad", self.zenith_spread_arrival_rad),
            ("zenith_spread_departure_rad", self.zenith_spread_departure_rad),
        )
        signed_fields = (
            ("shadow_fading_db", self.shadow_fading_db),
            ("k_factor_db", self.k_factor_db),
            ("path_loss_db", self.path_loss_db),
        )
        backend = None
        for name, value in nonnegative_fields:
            backend = _validate_real_array(
                value, self.metadata, name, rank=len(LINK_AXES), nonnegative=True
            )
        for name, value in signed_fields:
            backend = _validate_real_array(value, self.metadata, name, rank=len(LINK_AXES))
        assert backend is not None
        shape = backend.shape(self.delay_spread_s)
        _ensure_positive_shape(shape, "large-scale parameter")
        for name, value in (*nonnegative_fields[1:], *signed_fields):
            if backend.shape(value) != shape:
                raise ValueError(f"{name} shape must match delay_spread_s")
        if self.link_mask is not None:
            _validate_bool_array(self.link_mask, self.metadata, "link_mask", rank=len(LINK_AXES))
            if backend.shape(self.link_mask) != shape:
                raise ValueError("link_mask shape must match large-scale parameters")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return ``(batch, rx, tx)`` dimensions."""

        return get_backend(self.metadata.backend).shape(self.delay_spread_s)

    def select_link(self, batch_index: int, rx: int, tx: int) -> LargeScaleParameters:
        """Return storage-sharing parameter views for one link.

        Parameters
        ----------
        batch_index, rx, tx : int
            Zero-based semantic-axis indices.

        Returns
        -------
        LargeScaleParameters
            Selected parameters with shape ``(1, 1, 1)``.
        """

        shape = self.shape
        slices = (
            _checked_index(batch_index, shape[0], "batch_index"),
            _checked_index(rx, shape[1], "rx"),
            _checked_index(tx, shape[2], "tx"),
        )
        mask = None if self.link_mask is None else self.link_mask[slices]
        return LargeScaleParameters(
            *(getattr(self, name)[slices] for name in (
                "delay_spread_s",
                "azimuth_spread_arrival_rad",
                "azimuth_spread_departure_rad",
                "zenith_spread_arrival_rad",
                "zenith_spread_departure_rad",
                "shadow_fading_db",
                "k_factor_db",
                "path_loss_db",
            )),
            metadata=self.metadata,
            link_mask=mask,
        )


@dataclass(frozen=True, slots=True)
class ClusterBatch:
    """Batched cluster-level propagation quantities.

    Parameters
    ----------
    delays_s : backend array
        Nonnegative cluster delays in seconds.
    powers_linear : backend array
        Nonnegative cluster powers in linear units. Model generators are
        responsible for normalization across valid clusters.
    azimuth_arrival_rad, azimuth_departure_rad : backend array
        Mean cluster azimuth angles in radians.
    zenith_arrival_rad, zenith_departure_rad : backend array
        Mean cluster zenith angles in radians.
    metadata : RealizationMetadata
        Backend, precision, device, model identity, and provenance.
    cluster_mask : backend array, optional
        Boolean cluster validity mask.

    Notes
    -----
    Every quantity and mask uses axes ``[batch, rx, tx, cluster]``. No scalar
    cluster objects are created in the simulation path.
    """

    delays_s: Any
    powers_linear: Any
    azimuth_arrival_rad: Any
    zenith_arrival_rad: Any
    azimuth_departure_rad: Any
    zenith_departure_rad: Any
    metadata: RealizationMetadata
    cluster_mask: Any | None = None

    def __post_init__(self) -> None:
        fields = (
            ("delays_s", self.delays_s, True),
            ("powers_linear", self.powers_linear, True),
            ("azimuth_arrival_rad", self.azimuth_arrival_rad, False),
            ("zenith_arrival_rad", self.zenith_arrival_rad, False),
            ("azimuth_departure_rad", self.azimuth_departure_rad, False),
            ("zenith_departure_rad", self.zenith_departure_rad, False),
        )
        backend = None
        for name, value, nonnegative in fields:
            backend = _validate_real_array(
                value,
                self.metadata,
                name,
                rank=len(CLUSTER_AXES),
                nonnegative=nonnegative,
            )
        assert backend is not None
        shape = backend.shape(self.delays_s)
        _ensure_positive_shape(shape, "cluster")
        for name, value, _ in fields[1:]:
            if backend.shape(value) != shape:
                raise ValueError(f"{name} shape must match delays_s")
        if self.cluster_mask is not None:
            _validate_bool_array(
                self.cluster_mask, self.metadata, "cluster_mask", rank=len(CLUSTER_AXES)
            )
            if backend.shape(self.cluster_mask) != shape:
                raise ValueError("cluster_mask shape must match cluster fields")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return ``(batch, rx, tx, cluster)`` dimensions."""

        return get_backend(self.metadata.backend).shape(self.delays_s)

    def select_link(self, batch_index: int, rx: int, tx: int) -> ClusterBatch:
        """Return storage-sharing cluster views for one link.

        Parameters
        ----------
        batch_index, rx, tx : int
            Zero-based semantic-axis indices.

        Returns
        -------
        ClusterBatch
            Selected clusters with singleton batch/receiver/transmitter axes.
        """

        shape = self.shape
        slices = (
            _checked_index(batch_index, shape[0], "batch_index"),
            _checked_index(rx, shape[1], "rx"),
            _checked_index(tx, shape[2], "tx"),
            slice(None),
        )
        mask = None if self.cluster_mask is None else self.cluster_mask[slices]
        return ClusterBatch(
            *(getattr(self, name)[slices] for name in (
                "delays_s",
                "powers_linear",
                "azimuth_arrival_rad",
                "zenith_arrival_rad",
                "azimuth_departure_rad",
                "zenith_departure_rad",
            )),
            metadata=self.metadata,
            cluster_mask=mask,
        )


@dataclass(frozen=True, slots=True)
class RayBatch:
    """Batched per-ray propagation quantities.

    Parameters
    ----------
    clusters : ClusterBatch
        Cluster delays, powers, mean angles, validity, and provenance.
    azimuth_arrival_rad, azimuth_departure_rad : backend array
        Per-ray azimuth angles in radians.
    zenith_arrival_rad, zenith_departure_rad : backend array
        Per-ray zenith angles in radians.
    xpr_linear : backend array
        Positive cross-polarization power ratios in linear units.
    initial_phase_rad : backend array
        Initial scalar ray phases in radians.
    ray_mask : backend array, optional
        Boolean ray validity mask.
    blockage_loss_db : backend array, optional
        Nonnegative ray attenuation in decibels.
    sort_indices : backend array, optional
        Integer cluster ordering with axes ``[batch, rx, tx, cluster]``.
    strongest_cluster_indices : backend array, optional
        Integer indices with axes ``[batch, rx, tx, strongest_cluster]``.

    Notes
    -----
    Per-ray fields use axes ``[batch, rx, tx, cluster, ray]``. Cluster
    dimensions must exactly match ``clusters``.
    """

    clusters: ClusterBatch
    azimuth_arrival_rad: Any
    zenith_arrival_rad: Any
    azimuth_departure_rad: Any
    zenith_departure_rad: Any
    xpr_linear: Any
    initial_phase_rad: Any
    ray_mask: Any | None = None
    blockage_loss_db: Any | None = None
    sort_indices: Any | None = None
    strongest_cluster_indices: Any | None = None

    def __post_init__(self) -> None:
        fields = (
            ("azimuth_arrival_rad", self.azimuth_arrival_rad, False),
            ("zenith_arrival_rad", self.zenith_arrival_rad, False),
            ("azimuth_departure_rad", self.azimuth_departure_rad, False),
            ("zenith_departure_rad", self.zenith_departure_rad, False),
            ("xpr_linear", self.xpr_linear, True),
            ("initial_phase_rad", self.initial_phase_rad, False),
        )
        backend = None
        for name, value, nonnegative in fields:
            backend = _validate_real_array(
                value,
                self.metadata,
                name,
                rank=len(RAY_AXES),
                nonnegative=nonnegative,
            )
        assert backend is not None
        shape = backend.shape(self.azimuth_arrival_rad)
        _ensure_positive_shape(shape, "ray")
        if shape[:4] != self.clusters.shape:
            raise ValueError("ray link/cluster dimensions must match clusters")
        for name, value, _ in fields[1:]:
            if backend.shape(value) != shape:
                raise ValueError(f"{name} shape must match ray angles")
        if backend.to_scalar(backend.min(self.xpr_linear)) <= 0:
            raise ValueError("xpr_linear must be positive")
        if self.ray_mask is not None:
            _validate_bool_array(self.ray_mask, self.metadata, "ray_mask", rank=len(RAY_AXES))
            if backend.shape(self.ray_mask) != shape:
                raise ValueError("ray_mask shape must match ray fields")
        if self.blockage_loss_db is not None:
            _validate_real_array(
                self.blockage_loss_db,
                self.metadata,
                "blockage_loss_db",
                rank=len(RAY_AXES),
                nonnegative=True,
            )
            if backend.shape(self.blockage_loss_db) != shape:
                raise ValueError("blockage_loss_db shape must match ray fields")
        cluster_count = shape[3]
        if self.sort_indices is not None:
            _validate_integer_array(
                self.sort_indices, self.metadata, "sort_indices", rank=len(CLUSTER_AXES)
            )
            if backend.shape(self.sort_indices) != self.clusters.shape:
                raise ValueError("sort_indices shape must match clusters")
            self._validate_cluster_indices(self.sort_indices, cluster_count, "sort_indices")
        if self.strongest_cluster_indices is not None:
            _validate_integer_array(
                self.strongest_cluster_indices,
                self.metadata,
                "strongest_cluster_indices",
                rank=4,
            )
            strongest_shape = backend.shape(self.strongest_cluster_indices)
            if strongest_shape[:3] != shape[:3] or not 0 < strongest_shape[3] <= cluster_count:
                raise ValueError("strongest_cluster_indices has incompatible dimensions")
            self._validate_cluster_indices(
                self.strongest_cluster_indices, cluster_count, "strongest_cluster_indices"
            )

    def _validate_cluster_indices(self, value: Any, cluster_count: int, field: str) -> None:
        backend = get_backend(self.metadata.backend)
        if not backend.all_nonnegative(value) or backend.to_scalar(backend.max(value)) >= cluster_count:
            raise ValueError(f"{field} values must identify existing clusters")

    @property
    def metadata(self) -> RealizationMetadata:
        """RealizationMetadata: Provenance shared with the cluster batch."""

        return self.clusters.metadata

    @property
    def shape(self) -> tuple[int, ...]:
        """Return ``(batch, rx, tx, cluster, ray)`` dimensions."""

        return get_backend(self.metadata.backend).shape(self.azimuth_arrival_rad)

    @property
    def delays_s(self) -> Any:
        """backend array: Cluster delays forwarded from :attr:`clusters`."""

        return self.clusters.delays_s

    @property
    def powers_linear(self) -> Any:
        """backend array: Cluster powers forwarded from :attr:`clusters`."""

        return self.clusters.powers_linear

    @property
    def cluster_mask(self) -> Any | None:
        """backend array or None: Cluster validity forwarded from ``clusters``."""

        return self.clusters.cluster_mask

    def select_link(self, batch_index: int, rx: int, tx: int) -> RayBatch:
        """Return storage-sharing cluster and ray views for one link.

        Parameters
        ----------
        batch_index, rx, tx : int
            Zero-based semantic-axis indices.

        Returns
        -------
        RayBatch
            Selected rays with singleton batch/receiver/transmitter axes.
        """

        shape = self.shape
        slices = (
            _checked_index(batch_index, shape[0], "batch_index"),
            _checked_index(rx, shape[1], "rx"),
            _checked_index(tx, shape[2], "tx"),
            slice(None),
            slice(None),
        )
        cluster_slices = slices[:4]
        ray_mask = None if self.ray_mask is None else self.ray_mask[slices]
        blockage = (
            None if self.blockage_loss_db is None else self.blockage_loss_db[slices]
        )
        sort_indices = None if self.sort_indices is None else self.sort_indices[cluster_slices]
        strongest = (
            None
            if self.strongest_cluster_indices is None
            else self.strongest_cluster_indices[slices[:3] + (slice(None),)]
        )
        return RayBatch(
            clusters=self.clusters.select_link(batch_index, rx, tx),
            azimuth_arrival_rad=self.azimuth_arrival_rad[slices],
            zenith_arrival_rad=self.zenith_arrival_rad[slices],
            azimuth_departure_rad=self.azimuth_departure_rad[slices],
            zenith_departure_rad=self.zenith_departure_rad[slices],
            xpr_linear=self.xpr_linear[slices],
            initial_phase_rad=self.initial_phase_rad[slices],
            ray_mask=ray_mask,
            blockage_loss_db=blockage,
            sort_indices=sort_indices,
            strongest_cluster_indices=strongest,
        )


def select_paths(
    batch: RayBatch, batch_index: int, rx: int, tx: int
) -> tuple[PathCluster, ...]:
    """Convert one batched ray link to scalar semantic path objects.

    Parameters
    ----------
    batch : RayBatch
        Batched cluster and ray quantities.
    batch_index, rx, tx : int
        Zero-based indices selecting exactly one link.

    Returns
    -------
    tuple of PathCluster
        Valid clusters in stored cluster order. Each contains valid
        :class:`PropagationPath` instances.

    Raises
    ------
    TypeError
        If an index is not an integer.
    IndexError
        If an index is outside its axis bounds.
    ValueError
        If a valid cluster has no valid rays.

    Notes
    -----
    The diagnostic scalar gain allocates cluster power equally across valid
    rays and applies ``blockage_loss_db`` as voltage attenuation. Zenith angles
    are converted to elevation by ``pi / 2 - zenith``. XPR is retained in path
    metadata rather than folded into the scalar gain.
    """

    shape = batch.shape
    _checked_index(batch_index, shape[0], "batch_index")
    _checked_index(rx, shape[1], "rx")
    _checked_index(tx, shape[2], "tx")
    backend = get_backend(batch.metadata.backend)
    clusters: list[PathCluster] = []
    strongest: set[int] = set()
    if batch.strongest_cluster_indices is not None:
        for index in range(backend.shape(batch.strongest_cluster_indices)[3]):
            strongest.add(
                int(
                    backend.to_scalar(
                        batch.strongest_cluster_indices[batch_index, rx, tx, index]
                    )
                )
            )
    for cluster_index in range(shape[3]):
        if batch.cluster_mask is not None and not bool(
            backend.to_scalar(batch.cluster_mask[batch_index, rx, tx, cluster_index])
        ):
            continue
        valid_rays = [
            ray_index
            for ray_index in range(shape[4])
            if batch.ray_mask is None
            or bool(
                backend.to_scalar(
                    batch.ray_mask[batch_index, rx, tx, cluster_index, ray_index]
                )
            )
        ]
        if not valid_rays:
            raise ValueError("a valid cluster must contain at least one valid ray")
        cluster_power = float(
            backend.to_scalar(batch.powers_linear[batch_index, rx, tx, cluster_index])
        )
        delay_s = float(backend.to_scalar(batch.delays_s[batch_index, rx, tx, cluster_index]))
        ray_power = cluster_power / len(valid_rays)
        paths: list[PropagationPath] = []
        for ray_index in valid_rays:
            indices = (batch_index, rx, tx, cluster_index, ray_index)
            phase = float(backend.to_scalar(batch.initial_phase_rad[indices]))
            blockage_db = (
                0.0
                if batch.blockage_loss_db is None
                else float(backend.to_scalar(batch.blockage_loss_db[indices]))
            )
            amplitude = sqrt(ray_power) * 10.0 ** (-blockage_db / 20.0)
            arrival = Direction(
                azimuth_rad=float(backend.to_scalar(batch.azimuth_arrival_rad[indices])),
                elevation_rad=pi / 2.0
                - float(backend.to_scalar(batch.zenith_arrival_rad[indices])),
            )
            departure = Direction(
                azimuth_rad=float(backend.to_scalar(batch.azimuth_departure_rad[indices])),
                elevation_rad=pi / 2.0
                - float(backend.to_scalar(batch.zenith_departure_rad[indices])),
            )
            paths.append(
                PropagationPath(
                    path_id=f"b{batch_index}:rx{rx}:tx{tx}:c{cluster_index}:r{ray_index}",
                    complex_gain=amplitude * complex(cos(phase), sin(phase)),
                    delay_s=delay_s,
                    departure=departure,
                    arrival=arrival,
                    metadata={
                        "cluster_index": cluster_index,
                        "ray_index": ray_index,
                        "xpr_linear": float(backend.to_scalar(batch.xpr_linear[indices])),
                        "blockage_loss_db": blockage_db,
                    },
                )
            )
        clusters.append(
            PathCluster(
                cluster_id=f"b{batch_index}:rx{rx}:tx{tx}:c{cluster_index}",
                paths=tuple(paths),
                metadata={
                    "cluster_index": cluster_index,
                    "power_linear": cluster_power,
                    "strongest": cluster_index in strongest,
                },
            )
        )
    return tuple(clusters)
