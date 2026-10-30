from channel_dataset.channels.wireless import ScreenBlocker, ScreenBlockingModelB


def test_screen_blocks_ray_inside_projection_only(backend, blockage_case):
    rays, topology = blockage_case
    # Rotate the test rays so the first points toward the +x screen.
    rays = type(rays)(
        rays.clusters,
        rays.azimuth_arrival_rad * 0.0,
        rays.zenith_arrival_rad * 0.0 + 1.5707963267948966,
        rays.azimuth_departure_rad,
        rays.zenith_departure_rad,
        rays.xpr_linear,
        rays.initial_phase_rad,
        ray_mask=rays.ray_mask,
        strongest_cluster_indices=rays.strongest_cluster_indices,
    )
    model = ScreenBlockingModelB((ScreenBlocker((10.0, 0.0, 0.0), 2.0, 2.0, 25.0),))
    loss = model.loss_db(rays, topology)
    assert backend.to_scalar(backend.min(loss)) == 25.0


def test_screen_outside_direction_has_zero_loss(backend, blockage_case):
    rays, topology = blockage_case
    model = ScreenBlockingModelB((ScreenBlocker((10.0, 0.0, 0.0), 1.0, 1.0),))
    loss = model.loss_db(rays, topology)
    assert backend.to_scalar(backend.max(loss)) == 0.0
