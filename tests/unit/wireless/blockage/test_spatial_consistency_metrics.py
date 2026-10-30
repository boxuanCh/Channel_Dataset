from channel_dataset.channels.wireless import SnapshotSpatialConsistency


def test_nearby_points_are_more_correlated_than_distant_points(backend):
    positions = backend.array(
        [[[0.0, 0.0], [1.0, 0.0], [100.0, 0.0]]] * 1000, dtype="float64"
    )
    values = SnapshotSpatialConsistency(10.0).sample(
        positions, backend.random_stream(33), num_lsp_fields=1, num_small_scale_fields=1
    ).lsp_gaussian[..., 0]
    near = backend.mean(values[:, 0] * values[:, 1])
    far = backend.mean(values[:, 0] * values[:, 2])
    assert backend.to_scalar(near) > backend.to_scalar(far) + 0.5
