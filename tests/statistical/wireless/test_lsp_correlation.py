import numpy as np
import pytest

from channel_dataset.channels.wireless import ScenarioConfig, TopologyBatch, create_scenario
from channel_dataset.core import RealizationMetadata


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _topology(execution, batch_size, rx_x):
    backend = execution.array_backend
    tx = backend.zeros((batch_size, 1, 3), dtype=execution.float_dtype)
    tx[..., 2] = 10.0
    rx_one = backend.array(
        [[x, 0.0, 1.5] for x in rx_x], dtype=execution.float_dtype
    )
    rx = backend.broadcast_to(rx_one[None, ...], (batch_size, len(rx_x), 3))
    metadata = RealizationMetadata(
        "wireless", "lsp_test", 1, execution.backend, execution.precision,
        execution.device, f"{execution.backend}:1:root", "0" * 64, {}, "test",
    )
    return TopologyBatch(
        tx, rx,
        backend.zeros(tx.shape, dtype=execution.float_dtype),
        backend.zeros(rx.shape, dtype=execution.float_dtype),
        backend.zeros(tx.shape, dtype=execution.float_dtype),
        backend.zeros(rx.shape, dtype=execution.float_dtype),
        backend.zeros((batch_size, len(rx_x)), dtype="bool"),
        backend.zeros((batch_size, 1), dtype="int64"),
        metadata,
        backend.ones((batch_size, len(rx_x), 1), dtype="bool"),
    )


@pytest.mark.statistical
def test_lsp_log_domain_moments_and_cross_correlation(execution_context) -> None:
    scenario = create_scenario(ScenarioConfig("UMi", 3.5e9))
    topology = _topology(execution_context, 5000, [100.0])
    result = scenario.sample_large_scale(topology, execution_context.random_stream(100))
    lgds = np.log10(_numpy(result.delay_spread_s)[:, 0, 0])
    lgasd = np.log10(np.rad2deg(_numpy(result.azimuth_spread_departure_rad)[:, 0, 0]))
    assert np.mean(lgds) == pytest.approx(-7.14, abs=0.02)
    assert np.std(lgds) == pytest.approx(0.38, abs=0.02)
    assert np.corrcoef(lgds, lgasd)[0, 1] == pytest.approx(0.1, abs=0.05)


@pytest.mark.statistical
def test_lsp_spatial_filtering_decays_with_distance(execution_context) -> None:
    scenario = create_scenario(ScenarioConfig("UMi", 3.5e9))
    topology = _topology(execution_context, 3000, [100.0, 101.0, 150.0])
    result = scenario.sample_large_scale(topology, execution_context.random_stream(101))
    lgds = np.log10(_numpy(result.delay_spread_s)[..., 0])
    near = np.corrcoef(lgds[:, 0], lgds[:, 1])[0, 1]
    far = np.corrcoef(lgds[:, 0], lgds[:, 2])[0, 1]
    assert near > far
    assert near > 0.7
    assert far < 0.2
