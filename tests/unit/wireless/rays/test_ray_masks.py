import pytest

from channel_dataset.channels.wireless import ScenarioRayGenerator


@pytest.mark.parametrize(
    ("scenario", "los_count", "nlos_count"),
    [("RMa", 11, 10), ("UMa", 12, 20), ("UMi", 12, 19), ("InH", 15, 19), ("InF", 25, 25)],
)
def test_every_scenario_uses_configured_cluster_masks(scenario, los_count, nlos_count):
    parameters = ScenarioRayGenerator(scenario).parameters
    assert parameters["los"].num_clusters == los_count
    assert parameters["nlos"].num_clusters == nlos_count


def test_invalid_clusters_are_zeroed(backend, ray_case):
    _, _, _, rays = ray_case
    invalid_clusters = ~rays.cluster_mask
    invalid_rays = ~rays.ray_mask
    assert backend.to_scalar(backend.max(rays.powers_linear[invalid_clusters])) == 0.0
    assert backend.to_scalar(backend.max(backend.abs(rays.initial_phase_rad[invalid_rays]))) == 0.0
    assert backend.to_scalar(backend.min(rays.xpr_linear[invalid_rays])) == 1.0
