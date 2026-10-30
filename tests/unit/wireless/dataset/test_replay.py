import pytest

from channel_dataset.channels.wireless import ChannelSamplingRequest, ReplayChannel


def test_replay_exhaustion_and_repeat(dataset_cir, execution_context):
    values = [dataset_cir.select_link(index, 0, 0) for index in range(3)]
    request = ChannelSamplingRequest(1, 2, 1.0, execution_context, execution_context.random_stream(1))
    finite = ReplayChannel(values)
    assert finite.realize(request) is values[0]
    finite.realize(request)
    finite.realize(request)
    with pytest.raises(StopIteration):
        finite.realize(request)
    cyclic = ReplayChannel(values, repeat=True)
    assert cyclic.realize(request) is values[0]
