from channel_dataset.channels.wireless import CAPABILITIES, NonSelfBlockingModelA


def test_unrelated_draws_do_not_reorder_blockage(backend, blockage_case):
    rays, topology = blockage_case
    model = NonSelfBlockingModelA()
    expected = model.loss_db(rays, topology, backend.random_stream(71))
    stream = backend.random_stream(71)
    stream.uniform((50,), dtype="float64")
    actual = model.loss_db(rays, topology, stream)
    assert backend.to_scalar(backend.max(backend.abs(expected - actual))) == 0.0


def test_dynamic_behavior_is_explicitly_unsupported():
    assert "stateful mobility evolution" in CAPABILITIES.omissions
    assert "temporal blocker evolution" in CAPABILITIES.omissions
