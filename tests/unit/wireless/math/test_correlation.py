from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless.math import (
    exponential_correlation,
    one_ring_correlation,
)


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().resolve_conj().cpu().numpy()
    return np.asarray(value)


def test_exponential_correlation_matches_analytic_matrix(execution_context) -> None:
    matrix = exponential_correlation(3, 0.5, execution_context)
    expected = [[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 1.0]]
    assert np.allclose(_as_numpy(matrix), expected)


def test_complex_exponential_correlation_is_hermitian_psd(execution_context) -> None:
    matrix = _as_numpy(exponential_correlation(5, 0.4 + 0.3j, execution_context))
    assert np.allclose(matrix, matrix.conj().T)
    assert np.linalg.eigvalsh(matrix).min() >= -1.0e-12


@pytest.mark.parametrize("coefficient", [1.01, complex(float("nan"), 0.0)])
def test_invalid_exponential_coefficient_is_rejected(
    execution_context, coefficient: complex
) -> None:
    with pytest.raises(ValueError):
        exponential_correlation(2, coefficient, execution_context)


def test_one_ring_correlation_is_hermitian_psd_with_unit_diagonal(
    execution_context,
) -> None:
    matrix = _as_numpy(
        one_ring_correlation(
            6, mean_azimuth_rad=0.3, angular_spread_rad=0.2,
            execution=execution_context, quadrature_points=128,
        )
    )
    assert np.allclose(matrix, matrix.conj().T, atol=1.0e-12)
    assert np.allclose(np.diag(matrix), 1.0, atol=1.0e-12)
    assert np.linalg.eigvalsh(matrix).min() >= -1.0e-10


def test_zero_spread_one_ring_is_rank_one(execution_context) -> None:
    matrix = _as_numpy(
        one_ring_correlation(4, 0.4, 0.0, execution_context)
    )
    assert np.count_nonzero(np.linalg.eigvalsh(matrix) > 1.0e-10) == 1


def test_one_ring_validates_spread(execution_context) -> None:
    with pytest.raises(ValueError, match="between zero and pi"):
        one_ring_correlation(2, 0.0, -0.1, execution_context)
