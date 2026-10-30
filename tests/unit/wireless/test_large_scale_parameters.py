import pytest

from channel_dataset.channels.wireless import LargeScaleParameters


def _parameters(backend, metadata, shape=(2, 3, 2)) -> LargeScaleParameters:
    positive = backend.ones(shape, dtype="float64")
    signed = backend.zeros(shape, dtype="float64")
    return LargeScaleParameters(
        positive,
        positive,
        positive,
        positive,
        positive,
        signed,
        signed,
        positive,
        metadata,
        backend.ones(shape, dtype="bool"),
    )


def test_large_scale_parameters_select_link(backend, realization_metadata) -> None:
    parameters = _parameters(backend, realization_metadata)
    selected = parameters.select_link(1, 2, 1)
    assert parameters.shape == (2, 3, 2)
    assert selected.shape == (1, 1, 1)


def test_large_scale_parameters_reject_negative_spread(backend, realization_metadata) -> None:
    shape = (1, 1, 1)
    negative = backend.full(shape, -1.0, dtype="float64")
    positive = backend.ones(shape, dtype="float64")
    with pytest.raises(ValueError, match="delay_spread_s"):
        LargeScaleParameters(
            negative,
            positive,
            positive,
            positive,
            positive,
            positive,
            positive,
            positive,
            realization_metadata,
        )


def test_large_scale_parameters_reject_shape_mismatch(backend, realization_metadata) -> None:
    parameters = _parameters(backend, realization_metadata)
    with pytest.raises(ValueError, match="shadow_fading_db"):
        LargeScaleParameters(
            parameters.delay_spread_s,
            parameters.azimuth_spread_arrival_rad,
            parameters.azimuth_spread_departure_rad,
            parameters.zenith_spread_arrival_rad,
            parameters.zenith_spread_departure_rad,
            backend.zeros((1, 1, 1), dtype="float64"),
            parameters.k_factor_db,
            parameters.path_loss_db,
            realization_metadata,
        )
