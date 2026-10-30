import numpy as np

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.operators import apply_ofdm
from channel_dataset.channels.wireless.tr38901 import register_cdl_model
from channel_dataset.channels.wireless.transforms import cir_to_ofdm, subcarrier_frequencies
from channel_dataset.config import load_config
from channel_dataset.registry import ModelRegistry


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_seeded_cdl_to_ofdm_replay(execution_context) -> None:
    envelope = load_config("configs/wireless/example_cdl.yaml")
    registry = ModelRegistry()
    register_cdl_model(registry)
    model = registry.create(envelope)

    def run():
        request = ChannelSamplingRequest(
            2, 3, 14_000.0, execution_context, execution_context.random_stream(2026)
        )
        cir = model.realize(request)
        frequencies = subcarrier_frequencies(16, 15_000.0, execution_context)
        response = cir_to_ofdm(cir, frequencies)
        signal = execution_context.array_backend.ones(
            (2, 1, 1, 3, 16),
            dtype=execution_context.complex_dtype,
            device=execution_context.device,
        )
        return _numpy(cir.coefficients), _numpy(apply_ofdm(response, signal))

    first_cir, first_output = run()
    second_cir, second_output = run()
    assert np.array_equal(first_cir, second_cir)
    assert np.array_equal(first_output, second_output)
