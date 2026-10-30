"""Stable linear-algebra helpers for wireless covariance matrices."""

from __future__ import annotations

from math import isfinite
from typing import Any

from channel_dataset.backends import backend_name_for_array, get_backend


def _default_tolerance(dtype: Any) -> float:
    dtype_name = str(dtype)
    return 1.0e-5 if "float32" in dtype_name or "complex64" in dtype_name else 1.0e-12


def matrix_sqrt_psd(matrix: Any, *, tolerance: float | None = None) -> Any:
    r"""Compute the principal square root of a Hermitian PSD matrix.

    Parameters
    ----------
    matrix : backend array
        Real symmetric or complex Hermitian matrix, optionally with leading
        batch dimensions.
    tolerance : float, optional
        Relative tolerance used for Hermitian and eigenvalue validation.
        Defaults to ``1e-5`` for single precision and ``1e-12`` for double.

    Returns
    -------
    backend array
        Principal Hermitian square root with the same backend, dtype, device,
        and batch dimensions as ``matrix``.

    Raises
    ------
    TypeError
        If the input is not a real or complex backend array.
    ValueError
        If the matrix is empty, nonsquare, non-finite, non-Hermitian, or has
        an eigenvalue below the allowed numerical tolerance.

    Notes
    -----
    Small negative eigenvalues caused by roundoff are clamped to zero. The
    eigendecomposition remains native, preserving PyTorch autograd.
    """

    backend = get_backend(backend_name_for_array(matrix))
    if not (backend.is_floating(matrix) or backend.is_complex(matrix)):
        raise TypeError("matrix must have a real or complex floating dtype")
    shape = backend.shape(matrix)
    if len(shape) < 2 or shape[-1] == 0 or shape[-2] != shape[-1]:
        raise ValueError("matrix must be nonempty and square on its final two axes")
    if not backend.all_finite(matrix):
        raise ValueError("matrix must contain only finite values")
    resolved_tolerance = _default_tolerance(backend.dtype(matrix)) if tolerance is None else tolerance
    if isinstance(resolved_tolerance, bool) or not isinstance(resolved_tolerance, (int, float)):
        raise TypeError("tolerance must be a real number")
    resolved_tolerance = float(resolved_tolerance)
    if not isfinite(resolved_tolerance) or resolved_tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")

    axes = list(range(len(shape)))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    adjoint = backend.conj(backend.transpose(matrix, tuple(axes)))
    scale = max(1.0, float(backend.to_scalar(backend.max(backend.abs(matrix)))))
    hermitian_error = float(backend.to_scalar(backend.max(backend.abs(matrix - adjoint))))
    threshold = resolved_tolerance * scale
    if hermitian_error > threshold:
        raise ValueError("matrix must be Hermitian within tolerance")

    hermitian = (matrix + adjoint) * 0.5
    eigenvalues, eigenvectors = backend.eigh(hermitian)
    minimum = float(backend.to_scalar(backend.min(eigenvalues)))
    if minimum < -threshold:
        raise ValueError("matrix must be positive semidefinite within tolerance")
    clipped = (eigenvalues + backend.abs(eigenvalues)) * 0.5
    roots = backend.sqrt(clipped)
    return backend.einsum(
        "...ik,...k,...jk->...ij",
        eigenvectors,
        roots,
        backend.conj(eigenvectors),
    )


__all__ = ["matrix_sqrt_psd"]
