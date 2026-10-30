def test_spatial_field_is_seeded_and_has_expected_shape(backend):
    from channel_dataset.channels.wireless import ExponentialSpatialField

    positions = backend.array(
        [[[0.0, 0.0], [1.0, 0.0], [1000.0, 0.0]]], dtype="float64"
    )
    field = ExponentialSpatialField(10.0)
    first = field.sample(positions, backend.random_stream(99))
    second = field.sample(positions, backend.random_stream(99))
    assert backend.shape(first) == (1, 3)
    assert backend.to_scalar(backend.max(backend.abs(first - second))) == 0.0


def test_spatial_field_rejects_invalid_distance():
    from channel_dataset.channels.wireless import ExponentialSpatialField

    try:
        ExponentialSpatialField(0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero correlation distance was accepted")
