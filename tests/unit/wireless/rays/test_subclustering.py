def test_strongest_indices_match_power_order(backend, ray_case):
    _, _, generator, rays = ray_case
    expected = backend.argsort(-rays.powers_linear, axis=-1)[..., :2]
    difference = rays.strongest_cluster_indices - expected
    assert backend.to_scalar(backend.max(backend.abs(difference))) == 0
    plan = generator.subcluster_plan(rays)
    assert backend.shape(plan.ray_group_indices) == (20,)
    assert {int(backend.to_scalar(value)) for value in plan.ray_group_indices} == {0, 1, 2}
