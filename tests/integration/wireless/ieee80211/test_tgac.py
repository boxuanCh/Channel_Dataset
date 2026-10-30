from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless import (
    ChannelSamplingRequest,
    TopologyBatch,
)
from channel_dataset.channels.wireless.ieee80211 import (
    TGacChannel,
    TGacConfig,
    WLANChannelSamplingRequest,
)
from channel_dataset.channels.wireless.operators import apply_ofdm
from channel_dataset.channels.wireless.transforms import cir_to_ofdm, subcarrier_frequencies
from channel_dataset.core import RealizationMetadata


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


def _config(**overrides) -> TGacConfig:
    values = {
        "profile": "B", "standard_version": "09/0308r12",
        "carrier_frequency_hz": 5.25e9, "bandwidth_hz": 80e6,
        "num_rx": 1, "num_rx_antennas": 1, "num_tx": 1,
        "num_tx_antennas": 1, "environmental_speed_mps": 0.0,
        "num_sinusoids": 32, "spatial_quadrature_order": 64,
        "rx_antenna_spacing_wavelengths": 0.5,
        "tx_antenna_spacing_wavelengths": 0.5, "line_of_sight": False,
        "large_scale_mode": "none", "fluorescent_effect": False,
        "moving_vehicle_effect": False, "normalize_profile_power": True,
        "num_users": 1, "direction": "downlink",
    }
    values.update(overrides)
    return TGacConfig(**values)


def _topology(
    execution,
    rx_x: tuple[float, ...],
    tx_x: tuple[float, ...] = (0.0,),
) -> TopologyBatch:
    backend = execution.array_backend
    num_rx = len(rx_x)
    num_tx = len(tx_x)
    tx_positions = backend.array(
        [[[x, 0.0, 0.0] for x in tx_x]], dtype=execution.float_dtype
    )
    rx_positions = backend.array([[[x, 0.0, 0.0] for x in rx_x]], dtype=execution.float_dtype)
    tx_zeros = backend.zeros((1, num_tx, 3), dtype=execution.float_dtype)
    rx_zeros = backend.zeros((1, num_rx, 3), dtype=execution.float_dtype)
    metadata = RealizationMetadata(
        "wireless", "test_topology", 1, execution.backend, execution.precision,
        execution.device, "topology", "0" * 64, {}, "test",
    )
    return TopologyBatch(
        tx_positions, rx_positions, tx_zeros, rx_zeros, tx_zeros, rx_zeros,
        backend.full((1, num_rx), True, dtype="bool"),
        backend.array([list(range(num_tx))], dtype="int64"), metadata,
        link_mask=backend.full((1, num_rx, num_tx), True, dtype="bool"),
    )


@pytest.mark.ieee80211
@pytest.mark.tgac
def test_tgac_8_by_8_and_ofdm_application(execution_context) -> None:
    config = _config(profile="A", bandwidth_hz=160e6, num_rx_antennas=8, num_tx_antennas=8)
    sampling = ChannelSamplingRequest(
        1, 2, 1e3, execution_context, execution_context.random_stream(71)
    )
    cir = TGacChannel(config).realize(WLANChannelSamplingRequest(sampling))
    assert cir.shape == (1, 1, 8, 1, 8, 1, 2)
    frequencies = subcarrier_frequencies(16, 312.5e3, execution_context)
    response = cir_to_ofdm(cir, frequencies)
    signal = execution_context.array_backend.ones(
        (1, 1, 8, 2, 16), dtype=execution_context.complex_dtype
    )
    output = apply_ofdm(response, signal)
    assert output.shape == (1, 1, 8, 2, 16)


@pytest.mark.ieee80211
@pytest.mark.tgac
def test_tgac_multiuser_streams_follow_user_identity_not_order(execution_context) -> None:
    config = _config(num_rx=2, num_users=2)
    model = TGacChannel(config)
    first_sampling = ChannelSamplingRequest(
        1, 1, 1e3, execution_context, execution_context.random_stream(55)
    )
    first = model.realize(WLANChannelSamplingRequest(first_sampling, _topology(execution_context, (10.0, 20.0))))
    second_sampling = ChannelSamplingRequest(
        1, 1, 1e3, execution_context, execution_context.random_stream(55)
    )
    second = model.realize(WLANChannelSamplingRequest(second_sampling, _topology(execution_context, (20.0, 10.0))))
    assert np.array_equal(
        _as_numpy(first.coefficients)[:, ::-1], _as_numpy(second.coefficients)
    )


@pytest.mark.ieee80211
@pytest.mark.tgac
def test_tgac_multiuser_requires_matching_topology(execution_context) -> None:
    model = TGacChannel(_config(num_rx=2, num_users=2))
    sampling = ChannelSamplingRequest(
        1, 1, 1e3, execution_context, execution_context.random_stream(1)
    )
    with pytest.raises(ValueError, match="requires topology"):
        model.realize(WLANChannelSamplingRequest(sampling))


@pytest.mark.ieee80211
@pytest.mark.tgac
def test_tgac_uplink_streams_follow_user_identity_not_order(execution_context) -> None:
    model = TGacChannel(_config(num_tx=2, num_users=2, direction="uplink"))
    sampling = ChannelSamplingRequest(
        1, 1, 1e3, execution_context, execution_context.random_stream(56)
    )
    first = model.realize(
        WLANChannelSamplingRequest(sampling, _topology(execution_context, (0.0,), (10.0, 20.0)))
    )
    replay_sampling = ChannelSamplingRequest(
        1, 1, 1e3, execution_context, execution_context.random_stream(56)
    )
    second = model.realize(
        WLANChannelSamplingRequest(
            replay_sampling, _topology(execution_context, (0.0,), (20.0, 10.0))
        )
    )
    assert np.array_equal(
        _as_numpy(first.coefficients)[:, :, :, ::-1], _as_numpy(second.coefficients)
    )
