from math import pi


def test_xpr_is_positive_and_phases_are_bounded(backend, ray_case):
    _, _, _, rays = ray_case
    assert backend.to_scalar(backend.min(rays.xpr_linear)) > 0.0
    assert backend.to_scalar(backend.min(rays.initial_phase_rad)) >= -pi
    assert backend.to_scalar(backend.max(rays.initial_phase_rad)) <= pi


def test_xpr_has_nonzero_variation(backend, ray_case):
    _, _, _, rays = ray_case
    valid = rays.xpr_linear[rays.ray_mask]
    assert backend.to_scalar(backend.max(valid) - backend.min(valid)) > 0.0
