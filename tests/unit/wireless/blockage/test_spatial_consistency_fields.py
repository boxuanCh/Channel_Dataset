from channel_dataset.channels.wireless import SnapshotSpatialConsistency


def test_snapshot_fields_have_documented_shapes_and_replay(backend):
    positions = backend.array(
        [[[0.0, 0.0], [1.0, 0.0], [100.0, 0.0]]], dtype="float64"
    )
    model = SnapshotSpatialConsistency(10.0)
    first = model.sample(positions, backend.random_stream(8))
    second = model.sample(positions, backend.random_stream(8))
    assert backend.shape(first.los_uniform) == (1, 3)
    assert backend.shape(first.lsp_gaussian) == (1, 3, 7)
    assert backend.shape(first.small_scale_gaussian) == (1, 3, 4)
    assert backend.to_scalar(
        backend.max(backend.abs(first.lsp_gaussian - second.lsp_gaussian))
    ) == 0.0
