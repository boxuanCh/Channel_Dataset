def test_los_component_is_deterministic_for_seed(system_case, backend):
    model, request = system_case
    first = model.realize(request)
    second = model.realize(request)
    difference = backend.max(backend.abs(first.coefficients - second.coefficients))
    assert backend.to_scalar(difference) == 0.0
