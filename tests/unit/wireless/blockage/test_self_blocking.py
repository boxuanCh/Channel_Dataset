from channel_dataset.channels.wireless import SelfBlockingModel


def test_self_blocking_includes_boundary_and_excludes_outside(backend, blockage_case):
    rays, topology = blockage_case
    loss = SelfBlockingModel("portrait").loss_db(rays, topology)
    assert backend.to_scalar(loss[0, 0, 0, 0, 0]) == 30.0
    assert backend.to_scalar(loss[0, 0, 0, 0, 1]) == 30.0
    assert backend.to_scalar(loss[0, 0, 0, 0, 2]) == 0.0
