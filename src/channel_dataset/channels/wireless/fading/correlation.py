"""Spatial-correlation transforms for flat MIMO coefficients."""

from __future__ import annotations

from typing import Any

from channel_dataset.backends import backend_name_for_array, get_backend
from channel_dataset.channels.wireless.math import matrix_sqrt_psd


def _validate_array(value: Any, backend_name: str, device: str, field: str) -> Any:
    if backend_name_for_array(value) != backend_name:
        raise ValueError(f"{field} backend must match coefficients")
    backend = get_backend(backend_name)
    if backend.device(value) != device:
        raise ValueError(f"{field} device must match coefficients")
    if not (backend.is_floating(value) or backend.is_complex(value)):
        raise TypeError(f"{field} must have a floating or complex dtype")
    if not backend.all_finite(value):
        raise ValueError(f"{field} must contain only finite values")
    return backend


def apply_kronecker_correlation(
    coefficients: Any,
    receive_correlation: Any,
    transmit_correlation: Any,
) -> Any:
    r"""Apply separable receive/transmit correlation to MIMO coefficients.

    Parameters
    ----------
    coefficients : backend array
        Complex coefficients with final axes ``[rx_antenna, tx_antenna]``.
    receive_correlation : backend array
        Hermitian PSD matrix ``[rx_antenna, rx_antenna]``.
    transmit_correlation : backend array
        Hermitian PSD matrix ``[tx_antenna, tx_antenna]``.

    Returns
    -------
    backend array
        Correlated coefficients
        :math:`R_{rx}^{1/2} H (R_{tx}^{1/2})^H`.

    Raises
    ------
    TypeError
        If coefficients are not complex or dtypes are incompatible.
    ValueError
        If backend, device, rank, shape, finiteness, Hermitian, or PSD
        validation fails.
    """

    backend_name = backend_name_for_array(coefficients)
    backend = get_backend(backend_name)
    if backend.ndim(coefficients) < 2:
        raise ValueError("coefficients must have at least two dimensions")
    if not backend.is_complex(coefficients):
        raise TypeError("coefficients must have a complex dtype")
    if not backend.all_finite(coefficients):
        raise ValueError("coefficients must contain only finite values")
    device = backend.device(coefficients)
    for matrix, field in (
        (receive_correlation, "receive_correlation"),
        (transmit_correlation, "transmit_correlation"),
    ):
        _validate_array(matrix, backend_name, device, field)

    coefficient_shape = backend.shape(coefficients)
    receive_shape = backend.shape(receive_correlation)
    transmit_shape = backend.shape(transmit_correlation)
    if receive_shape != (coefficient_shape[-2], coefficient_shape[-2]):
        raise ValueError("receive_correlation shape does not match receive antennas")
    if transmit_shape != (coefficient_shape[-1], coefficient_shape[-1]):
        raise ValueError("transmit_correlation shape does not match transmit antennas")

    coefficient_dtype = backend.dtype(coefficients)
    receive_root = backend.astype(matrix_sqrt_psd(receive_correlation), coefficient_dtype)
    transmit_root = backend.astype(matrix_sqrt_psd(transmit_correlation), coefficient_dtype)
    return backend.einsum(
        "ij,...jk,lk->...il",
        receive_root,
        coefficients,
        backend.conj(transmit_root),
    )


def apply_per_column_receive_correlation(
    coefficients: Any,
    receive_correlations: Any,
) -> Any:
    r"""Apply a distinct receive correlation to each transmit column.

    Parameters
    ----------
    coefficients : backend array
        Complex coefficients with final axes ``[rx_antenna, tx_antenna]``.
    receive_correlations : backend array
        Hermitian PSD matrices with axes
        ``[tx_antenna, rx_antenna, rx_antenna]``.

    Returns
    -------
    backend array
        Coefficients whose transmit column ``k`` is transformed by
        :math:`R_{rx,k}^{1/2}`.
    """

    backend_name = backend_name_for_array(coefficients)
    backend = get_backend(backend_name)
    if backend.ndim(coefficients) < 2:
        raise ValueError("coefficients must have at least two dimensions")
    if not backend.is_complex(coefficients):
        raise TypeError("coefficients must have a complex dtype")
    if not backend.all_finite(coefficients):
        raise ValueError("coefficients must contain only finite values")
    _validate_array(
        receive_correlations,
        backend_name,
        backend.device(coefficients),
        "receive_correlations",
    )
    receive_antennas, transmit_antennas = backend.shape(coefficients)[-2:]
    expected = (transmit_antennas, receive_antennas, receive_antennas)
    if backend.shape(receive_correlations) != expected:
        raise ValueError(f"receive_correlations shape must be {expected}")
    roots = backend.astype(matrix_sqrt_psd(receive_correlations), backend.dtype(coefficients))
    return backend.einsum("kij,...jk->...ik", roots, coefficients)


__all__ = ["apply_kronecker_correlation", "apply_per_column_receive_correlation"]
