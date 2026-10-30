from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38901 import register_cdl_model
from channel_dataset.config import load_config
from channel_dataset.registry import ModelRegistry


def test_configured_cdl_uses_dual_polarized_arrays(execution_context):
    envelope = load_config("configs/wireless/example_cdl_polarized.yaml")
    registry = ModelRegistry()
    register_cdl_model(registry)
    model = registry.create(envelope)
    request = ChannelSamplingRequest(
        1, 2, 15_000.0, execution_context, execution_context.random_stream(2026)
    )
    cir = model.realize(request)
    assert cir.shape[:5] == (1, 1, 2, 1, 2)
    assert cir.shape[-1] == 2
