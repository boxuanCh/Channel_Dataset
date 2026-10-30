def test_valid_cluster_powers_are_normalized(backend, ray_case):
    _, _, _, rays = ray_case
    total = backend.sum(rays.powers_linear, axis=-1)
    assert abs(float(backend.to_scalar(total)) - 1.0) < 1.0e-12
    assert backend.all_nonnegative(rays.powers_linear)
    assert backend.to_scalar(backend.max(rays.powers_linear[~rays.cluster_mask])) == 0.0
