from channel_dataset.channels.wireless import load_large_scale_parameters


def test_umi_large_scale_resource() -> None:
    resource = load_large_scale_parameters("UMi")
    assert resource["review_status"] == "pending_independent_review"
    assert len(resource["parameter_order"]) == 7
    assert len(resource["cross_correlation_matrix"]) == 7
    assert len(resource["spatial_correlation_distances_m"]) == 7
