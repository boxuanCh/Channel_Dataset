import pytest

from channel_dataset.channels.wireless import (
    ChannelSamplingRequest,
    ScenarioConfig,
    SystemLevelWirelessChannel,
    TopologyBatch,
)
from channel_dataset.channels.wireless.antennas import load_antenna_layout


@pytest.fixture
def system_case(backend, execution_context, realization_metadata):
    zero = backend.zeros((1, 1, 3), dtype="float64")
    topology = TopologyBatch(
        backend.array([[[0.0, 0.0, 10.0]]], dtype="float64"),
        backend.array([[[100.0, 0.0, 1.5]]], dtype="float64"),
        zero,
        zero,
        zero,
        zero,
        backend.zeros((1, 1), dtype="bool"),
        backend.zeros((1, 1), dtype="int64"),
        realization_metadata,
        los_mask=backend.ones((1, 1, 1), dtype="bool"),
    )
    layout = load_antenna_layout("profiles/wireless/antenna_isotropic_siso.yaml")
    model = SystemLevelWirelessChannel(
        ScenarioConfig("UMi", 3.5e9), topology, layout, layout
    )
    request = ChannelSamplingRequest(
        1, 4, 1000.0, execution_context, execution_context.random_stream(700)
    )
    return model, request
