def test_system_assembly_has_canonical_axes(system_case):
    model, request = system_case
    result = model.realize_with_diagnostics(request)
    assert result.cir.shape[:5] == (1, 1, 1, 1, 1)
    assert result.cir.shape[-1] == request.num_time_samples
    assert result.coefficients is not None


def test_diagnostics_do_not_change_channel(system_case, backend):
    model, request = system_case
    with_details = model.realize_with_diagnostics(request, diagnostics=True).cir
    without_details = model.realize_with_diagnostics(request, diagnostics=False).cir
    difference = backend.max(backend.abs(with_details.coefficients - without_details.coefficients))
    assert backend.to_scalar(difference) == 0.0
