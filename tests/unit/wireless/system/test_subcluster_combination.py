def test_strongest_clusters_expand_to_three_subclusters(system_case):
    model, request = system_case
    result = model.realize_with_diagnostics(request)
    assert result.cir.shape[5] == result.rays.shape[3] + 4
    assert result.cir.path_mask.shape == result.cir.delays_s.shape
