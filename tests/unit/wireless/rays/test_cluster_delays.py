def test_cluster_delays_are_sorted_and_nonnegative(backend, ray_case):
    _, _, _, rays = ray_case
    differences = rays.delays_s[..., 1:] - rays.delays_s[..., :-1]
    valid_pairs = rays.cluster_mask[..., 1:]
    selected = differences[valid_pairs]
    assert backend.all_nonnegative(selected)
    assert backend.to_scalar(rays.delays_s[0, 0, 0, 0]) == 0.0


def test_generation_is_reproducible(backend, ray_case):
    topology, large_scale, generator, first = ray_case
    second = generator.generate(topology, large_scale, backend.random_stream(200))
    assert backend.to_scalar(backend.max(backend.abs(first.delays_s - second.delays_s))) == 0.0
    assert backend.to_scalar(
        backend.max(backend.abs(first.initial_phase_rad - second.initial_phase_rad))
    ) == 0.0
