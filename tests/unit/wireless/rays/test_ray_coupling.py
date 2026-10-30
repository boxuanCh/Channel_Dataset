def test_each_angle_uses_a_permutation_of_standard_offsets(backend, ray_case):
    _, _, _, rays = ray_case
    # Independent coupling should almost surely give different ray ordering.
    arrival = rays.azimuth_arrival_rad[0, 0, 0, 0]
    departure = rays.azimuth_departure_rad[0, 0, 0, 0]
    assert backend.to_scalar(backend.max(backend.abs(arrival - departure))) > 0.0


def test_unrelated_parent_draws_do_not_change_rays(backend, ray_case):
    topology, large_scale, generator, first = ray_case
    stream = backend.random_stream(200)
    stream.normal((100,), dtype="float64")
    second = generator.generate(topology, large_scale, stream)
    assert backend.to_scalar(
        backend.max(backend.abs(first.initial_phase_rad - second.initial_phase_rad))
    ) == 0.0
