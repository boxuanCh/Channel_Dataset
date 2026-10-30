from math import sqrt

from channel_dataset.channels.wireless import (
    CIRBatch,
    OFDMChannelResponse,
    frequency_selectivity,
    mimo_singular_values,
    rms_delay_spread,
)


def test_rms_delay_spread_matches_two_equal_paths(backend, realization_metadata):
    coefficients = backend.ones((1, 1, 1, 1, 1, 2, 1), dtype="complex128")
    delays = backend.array([[[[0.0, 2.0]]]], dtype="float64")
    value = rms_delay_spread(CIRBatch(coefficients, delays, realization_metadata))
    assert abs(float(backend.to_scalar(value)) - 1.0) < 1e-12


def test_mimo_singular_values_match_diagonal_matrix(backend, realization_metadata):
    values = backend.reshape(
        backend.array([3.0 + 0j, 0.0 + 0j, 0.0 + 0j, 2.0 + 0j], dtype="complex128"),
        (1, 1, 2, 1, 2, 1, 1),
    )
    response = OFDMChannelResponse(values, realization_metadata)
    singular = mimo_singular_values(response)
    assert backend.shape(singular) == (1, 1, 1, 1, 1, 2)
    assert abs(float(backend.to_scalar(singular[0, 0, 0, 0, 0, 0])) - 3.0) < 1e-12


def test_frequency_selectivity_is_zero_for_flat_response(backend, realization_metadata):
    values = backend.ones((1, 1, 1, 1, 1, 2, 8), dtype="complex128") * sqrt(2.0)
    response = OFDMChannelResponse(values, realization_metadata)
    selectivity = frequency_selectivity(response)
    assert backend.to_scalar(backend.max(selectivity)) == 0.0
