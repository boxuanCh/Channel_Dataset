"""Shared assembly for TR 37.885 urban system and fixed-CDL channels."""

from __future__ import annotations

from dataclasses import asdict
from math import cos, exp, log, log10, pi, sin, sqrt
from typing import Any

from channel_dataset.channels.wireless.math import SPEED_OF_LIGHT_MPS
from channel_dataset.channels.wireless.topology import compute_link_geometry
from channel_dataset.channels.wireless.tr37885.antennas import (
    load_vehicle_antenna_profile,
)
from channel_dataset.channels.wireless.tr37885.config import TR37885UrbanGridConfig
from channel_dataset.channels.wireless.tr37885.grid import TR37885UrbanGrid
from channel_dataset.channels.wireless.tr37885.large_scale import (
    vehicle_blockage_parameters_db,
)
from channel_dataset.channels.wireless.tr37885.link_state import classify_urban_links
from channel_dataset.channels.wireless.tr37885.profiles import V2XCDLProfile, load_cdl_profile
from channel_dataset.channels.wireless.tr37885.topology import topology_at_snapshot
from channel_dataset.channels.wireless.tr37885.traffic import _software_version
from channel_dataset.channels.wireless.tr37885.types import (
    INVALID,
    LOS,
    NLOS,
    NLOSV,
    V2XChannelRealization,
    V2XChannelSamplingRequest,
    V2XChannelState,
    V2XLargeScaleComponents,
    V2XRealizationDiagnostics,
)
from channel_dataset.channels.wireless.types import CIRBatch, TopologyBatch
from channel_dataset.config import ConfigEnvelope
from channel_dataset.core import RealizationMetadata


class UrbanChannelAssembler:
    """Internal three-state channel engine shared by public model classes."""

    def __init__(self, config: Any, name: str, *, fixed_cdl: bool) -> None:
        self.config = config
        self.name = name
        self.fixed_cdl = fixed_cdl
        self.grid = TR37885UrbanGrid()
        self._resolved = asdict(config)
        self._digest = ConfigEnvelope(1, "wireless", name, self._resolved).model_digest
        self._profiles = {
            LOS: load_cdl_profile("los"),
            NLOSV: load_cdl_profile("nlosv"),
            NLOS: load_cdl_profile("nlos"),
        }

    def metadata(self, request: V2XChannelSamplingRequest) -> RealizationMetadata:
        """Create realization provenance for one sampling request."""

        sampling = request.sampling
        return RealizationMetadata(
            "wireless", self.name, 1, sampling.execution.backend,
            sampling.execution.precision, sampling.execution.device,
            sampling.random_stream.stream_id, self._digest, self._resolved,
            _software_version(), sampling.request_id,
            root_seed=sampling.random_stream.root_seed, model_version="1",
            standard_version="15.3.0",
        )

    def realize(
        self,
        request: V2XChannelSamplingRequest,
        *,
        diagnostics: bool = False,
    ) -> V2XChannelRealization:
        """Assemble one snapshot realization."""

        if not isinstance(request, V2XChannelSamplingRequest):
            raise TypeError("request must be V2XChannelSamplingRequest")
        sampling, scene = request.sampling, request.scene
        metadata = self.metadata(request)
        if request.state is not None:
            if request.state.metadata.stream_id != metadata.stream_id:
                raise ValueError("carried V2X state must use the same random stream")
            if request.state.metadata.config_digest != metadata.config_digest:
                raise ValueError(
                    "carried V2X state was produced by another channel configuration"
                )
        scene_grid = self._grid_for_scene(scene)
        conditions = classify_urban_links(
            scene, scene_grid, request.tx_vehicle_indices, request.rx_vehicle_indices,
            sampling.random_stream.fork("tr37885.link_conditions"), metadata=metadata,
        )
        topology = topology_at_snapshot(
            scene, request.snapshot_index, request.tx_vehicle_indices,
            request.rx_vehicle_indices, metadata,
        )
        codes = conditions.condition_codes[..., request.snapshot_index]
        topology = TopologyBatch(
            topology.tx_positions_m, topology.rx_positions_m,
            topology.tx_velocities_mps, topology.rx_velocities_mps,
            topology.tx_orientations_rad, topology.rx_orientations_rad,
            topology.rx_indoor, topology.tx_site_ids, metadata,
            codes == LOS, topology.link_mask,
        )
        large_scale, state = self._large_scale(request, topology, conditions, codes, metadata)
        coefficients, delays, path_mask, doppler = self._coefficients(
            request, topology, conditions, codes, large_scale, metadata
        )
        cir = CIRBatch(coefficients, delays, metadata, path_mask)
        values = {
            "standard_baseline": (
                scene_grid.config.mode == "standard"
                and self.config.carrier_frequency_hz == 6.0e9
            )
        }
        if diagnostics:
            values.update({"doppler_hz": doppler, "condition_codes": codes})
        details = V2XRealizationDiagnostics(values)
        return V2XChannelRealization(
            cir, topology, conditions, large_scale, state, scene, details
        )

    @staticmethod
    def _grid_for_scene(scene) -> TR37885UrbanGrid:
        """Reconstruct the authoritative grid recorded with a generated scene."""

        resolved = scene.metadata.resolved_configuration
        grid_values = resolved.get("grid") if hasattr(resolved, "get") else None
        if grid_values is None:
            return TR37885UrbanGrid()
        return TR37885UrbanGrid(TR37885UrbanGridConfig(**dict(grid_values)))

    def _large_scale(self, request, topology, conditions, codes, metadata):
        sampling, scene = request.sampling, request.scene
        backend, execution = sampling.execution.array_backend, sampling.execution
        geometry = compute_link_geometry(topology)
        b_count = topology.shape[0]
        distance = backend.clip(geometry.distance_3d_m, 1.0e-9, float("inf"))
        frequency_ghz = self.config.carrier_frequency_hz / 1.0e9
        los_path_loss = (
            38.77
            + 16.7 * backend.log10(distance)
            + 18.2 * log10(frequency_ghz)
        )
        nlos_path_loss = (
            36.85
            + 30.0 * backend.log10(distance)
            + 18.9 * log10(frequency_ghz)
        )
        valid_mask = topology.link_mask * (codes != INVALID)
        zero = backend.zeros(
            topology.shape, dtype=execution.float_dtype, device=execution.device
        )
        path_loss_values = backend.where(
            valid_mask,
            backend.where(codes == NLOS, nlos_path_loss, los_path_loss),
            zero,
        )
        shadow_values, blockage_values = [], []
        prior = request.state
        for batch in range(b_count):
            shadow_receivers, blockage_receivers = [], []
            for rx_axis, rx_vehicle in enumerate(request.rx_vehicle_indices):
                shadow_transmitters, blockage_transmitters = [], []
                for tx_axis, tx_vehicle in enumerate(request.tx_vehicle_indices):
                    condition = int(backend.to_scalar(codes[batch, rx_axis, tx_axis]))
                    valid = condition != INVALID and bool(backend.to_scalar(topology.link_mask[batch, rx_axis, tx_axis]))
                    if not valid:
                        shadow = blockage = 0.0
                    else:
                        link_distance = float(backend.to_scalar(
                            geometry.distance_3d_m[batch, rx_axis, tx_axis]
                        ))
                        shadow_std = 4.0 if condition == NLOS else 3.0
                        fresh = float(backend.to_scalar(sampling.random_stream.fork(
                            f"tr37885.link.{min(rx_vehicle, tx_vehicle)}.{max(rx_vehicle, tx_vehicle)}.shadow.{batch}.{request.snapshot_index}"
                        ).normal((1,), stddev=shadow_std, dtype=execution.float_dtype)[0]))
                        shadow = fresh
                        current_episode = int(backend.to_scalar(
                            conditions.episode_indices[
                                batch, rx_axis, tx_axis, request.snapshot_index
                            ]
                        ))
                        same_episode = False
                        if prior is not None and prior.snapshot_index + 1 == request.snapshot_index:
                            prior_condition = int(backend.to_scalar(
                                prior.condition_codes[batch, rx_axis, tx_axis]
                            ))
                            prior_episode = int(backend.to_scalar(
                                prior.episode_indices[batch, rx_axis, tx_axis]
                            ))
                            same_episode = (
                                prior_condition == condition
                                and prior_episode == current_episode
                            )
                        if same_episode:
                            old = float(backend.to_scalar(prior.shadow_fading_db[batch, rx_axis, tx_axis]))
                            tx_speed = _norm(topology.tx_velocities_mps[batch, tx_axis], backend)
                            rx_speed = _norm(topology.rx_velocities_mps[batch, rx_axis], backend)
                            times = scene.mobility_times_s
                            delta_time = float(backend.to_scalar(
                                times[request.snapshot_index]
                                - times[request.snapshot_index - 1]
                            ))
                            delta = 0.5 * (tx_speed + rx_speed) * delta_time
                            correlation_distance = 13.0 if condition != LOS else 10.0
                            rho = exp(-delta / correlation_distance)
                            shadow = rho * old + sqrt(max(0.0, 1.0 - rho * rho)) * fresh
                        blockage = 0.0
                        if condition == NLOSV:
                            blocker_height = float(backend.to_scalar(
                                conditions.blocker_heights_m[
                                    batch, rx_axis, tx_axis, request.snapshot_index
                                ]
                            ))
                            tx_height = float(backend.to_scalar(
                                scene.antenna_heights_m[batch, tx_vehicle]
                            ))
                            rx_height = float(backend.to_scalar(
                                scene.antenna_heights_m[batch, rx_vehicle]
                            ))
                            mean, stddev = vehicle_blockage_parameters_db(
                                link_distance, tx_height, rx_height, blocker_height
                            )
                            if stddev:
                                blockage = max(0.0, float(backend.to_scalar(
                                    sampling.random_stream.fork(
                                        f"tr37885.link.{min(rx_vehicle, tx_vehicle)}.{max(rx_vehicle, tx_vehicle)}.blockage.{batch}.{request.snapshot_index}"
                                    ).normal((1,), mean=mean, stddev=stddev, dtype=execution.float_dtype)[0]
                                )))
                    shadow_transmitters.append(shadow)
                    blockage_transmitters.append(blockage)
                shadow_receivers.append(shadow_transmitters)
                blockage_receivers.append(blockage_transmitters)
            shadow_values.append(shadow_receivers)
            blockage_values.append(blockage_receivers)
        dtype, device = execution.float_dtype, execution.device
        shadow_array = backend.array(shadow_values, dtype=dtype, device=device)
        blockage_array = backend.array(blockage_values, dtype=dtype, device=device)
        total = (
            (path_loss_values if self.config.apply_path_loss else zero)
            + (shadow_array if self.config.apply_shadow_fading else zero)
            + (blockage_array if self.config.apply_vehicle_blockage else zero)
        )
        large_scale = V2XLargeScaleComponents(
            path_loss_values,
            shadow_array,
            blockage_array,
            total,
            metadata,
        )
        state = V2XChannelState(
            backend.astype(codes, "int64"), large_scale.shadow_fading_db,
            backend.astype(conditions.episode_indices[..., request.snapshot_index], "int64"),
            request.snapshot_index, metadata,
        )
        return large_scale, state

    def _coefficients(self, request, topology, conditions, codes, large_scale, metadata):
        sampling, execution = request.sampling, request.sampling.execution
        backend = execution.array_backend
        profiles = self._profiles if self.fixed_cdl else None
        max_paths = max(len(profile.delays_s) for profile in self._profiles.values()) if profiles else 19
        batch_values, batch_delays, batch_masks, batch_doppler = [], [], [], [],
        for batch in range(topology.shape[0]):
            receivers, delay_receivers, mask_receivers, doppler_receivers = [], [], [], []
            for rx_axis, rx_vehicle in enumerate(request.rx_vehicle_indices):
                transmitters, delay_transmitters, mask_transmitters, doppler_transmitters = [], [], [], []
                for tx_axis, tx_vehicle in enumerate(request.tx_vehicle_indices):
                    condition = int(backend.to_scalar(codes[batch, rx_axis, tx_axis]))
                    valid = condition != INVALID and bool(backend.to_scalar(topology.link_mask[batch, rx_axis, tx_axis]))
                    episode = int(backend.to_scalar(
                        conditions.episode_indices[
                            batch, rx_axis, tx_axis, request.snapshot_index
                        ]
                    ))
                    profile = self._profile_for_link(
                        request, condition, batch, rx_vehicle, tx_vehicle, episode
                    ) if valid else ([], [], [])
                    delays_list, powers, angles = profile
                    path_values, dopplers = [], []
                    for path in range(max_paths):
                        if path < len(delays_list):
                            coefficient, doppler = self._one_path(
                                request, topology, batch, rx_axis, tx_axis,
                                rx_vehicle, tx_vehicle, condition, episode, path,
                                powers[path], angles[path],
                            )
                            attenuation = backend.exp(
                                -log(10.0)
                                * large_scale.total_loss_db[batch, rx_axis, tx_axis]
                                / 20.0
                            )
                            path_values.append(coefficient * attenuation)
                            dopplers.append(doppler)
                        else:
                            path_values.append(backend.zeros(
                                (self.config.num_rx_antennas, self.config.num_tx_antennas,
                                 sampling.num_time_samples),
                                dtype=execution.complex_dtype, device=execution.device,
                            ))
                            dopplers.append(0.0)
                    transmitters.append(backend.stack(tuple(path_values), axis=2))
                    padded_delays = delays_list + [0.0] * (max_paths - len(delays_list))
                    delay_transmitters.append(backend.array(
                        padded_delays, dtype=execution.float_dtype, device=execution.device
                    ))
                    mask_transmitters.append(backend.array(
                        [valid and index < len(delays_list) for index in range(max_paths)],
                        dtype="bool", device=execution.device,
                    ))
                    doppler_transmitters.append(backend.array(
                        dopplers, dtype=execution.float_dtype, device=execution.device
                    ))
                receivers.append(backend.stack(tuple(transmitters), axis=1))
                delay_receivers.append(backend.stack(tuple(delay_transmitters), axis=0))
                mask_receivers.append(backend.stack(tuple(mask_transmitters), axis=0))
                doppler_receivers.append(backend.stack(tuple(doppler_transmitters), axis=0))
            batch_values.append(backend.stack(tuple(receivers), axis=0))
            batch_delays.append(backend.stack(tuple(delay_receivers), axis=0))
            batch_masks.append(backend.stack(tuple(mask_receivers), axis=0))
            batch_doppler.append(backend.stack(tuple(doppler_receivers), axis=0))
        return (
            backend.stack(tuple(batch_values), axis=0),
            backend.stack(tuple(batch_delays), axis=0),
            backend.stack(tuple(batch_masks), axis=0),
            backend.stack(tuple(batch_doppler), axis=0),
        )

    def _profile_for_link(
        self, request, condition, batch, rx_vehicle, tx_vehicle, episode
    ):
        if self.fixed_cdl:
            profile: V2XCDLProfile = self._profiles[condition]
            raw = [10.0 ** (value / 10.0) for value in profile.powers_db]
            total = sum(raw)
            angles = list(zip(
                profile.aod_degrees,
                profile.aoa_degrees,
                profile.zod_degrees,
                profile.zoa_degrees,
                strict=True,
            ))
            return list(profile.delays_s), [value / total for value in raw], angles
        sampling = request.sampling
        carrier_ghz = self.config.carrier_frequency_hz / 1.0e9
        settings = {
            LOS: (-0.2 * log10(1.0 + carrier_ghz) - 7.5, 0.1, 12, 3.0, 3.48, 2.0),
            NLOSV: (-0.4 * log10(1.0 + carrier_ghz) - 7.0, 0.1, 19, 2.1, 0.0, 4.5),
            NLOS: (-0.3 * log10(1.0 + carrier_ghz) - 7.0, 0.28, 19, 2.1, None, None),
        }[condition]
        mean_log, std_log, count, scaling, k_mean, k_std = settings
        label = (
            f"tr37885.link.{min(rx_vehicle, tx_vehicle)}."
            f"{max(rx_vehicle, tx_vehicle)}.system.{batch}.episode.{episode}"
        )
        ds_normal = float(sampling.execution.array_backend.to_scalar(
            sampling.random_stream.fork(f"{label}.delay_spread").normal((1,))[0]
        ))
        delay_spread = 10.0 ** (mean_log + std_log * ds_normal)
        uniforms = sampling.random_stream.fork(f"{label}.delays").uniform(
            (count,), 1.0e-9, 1.0
        )
        delays = sorted(
            -scaling * delay_spread * log(float(sampling.execution.array_backend.to_scalar(value)))
            for value in uniforms
        )
        origin = delays[0]
        delays = [value - origin for value in delays]
        powers = [exp(-delay * (scaling - 1.0) / max(scaling * delay_spread, 1.0e-30)) for delay in delays]
        if k_mean is not None:
            k_draw = float(sampling.execution.array_backend.to_scalar(
                sampling.random_stream.fork(f"{label}.k_factor").normal((1,))[0]
            ))
            k_linear = 10.0 ** ((k_mean + k_std * k_draw) / 10.0)
            diffuse = sum(powers[1:])
            powers = [k_linear / (k_linear + 1.0)] + [
                value / diffuse / (k_linear + 1.0) for value in powers[1:]
            ]
        else:
            total = sum(powers)
            powers = [value / total for value in powers]
        angle_stream = sampling.random_stream.fork(f"{label}.angles")
        angle_values = angle_stream.uniform(
            (count, 2), -pi, pi, dtype=sampling.execution.float_dtype
        )
        backend = sampling.execution.array_backend
        angles = [
            (
                float(backend.to_scalar(angle_values[index, 0])) * 180.0 / pi,
                float(backend.to_scalar(angle_values[index, 1])) * 180.0 / pi,
                90.0,
                90.0,
            )
            for index in range(count)
        ]
        return delays, powers, angles

    def _one_path(
        self, request, topology, batch, rx_axis, tx_axis, rx_vehicle, tx_vehicle,
        condition, episode, path, power, angles_degrees,
    ):
        sampling, execution = request.sampling, request.sampling.execution
        backend = execution.array_backend
        aod = angles_degrees[0] * pi / 180.0
        aoa = angles_degrees[1] * pi / 180.0
        zod = angles_degrees[2] * pi / 180.0
        zoa = angles_degrees[3] * pi / 180.0
        tx_direction = backend.array(
            (
                sin(zod) * cos(aod),
                sin(zod) * sin(aod),
                cos(zod),
            ),
            dtype=execution.float_dtype,
            device=execution.device,
        )
        rx_direction = backend.array(
            (
                sin(zoa) * cos(aoa),
                sin(zoa) * sin(aoa),
                cos(zoa),
            ),
            dtype=execution.float_dtype,
            device=execution.device,
        )
        radial = float(backend.to_scalar(
            backend.sum(rx_direction * topology.rx_velocities_mps[batch, rx_axis])
            + backend.sum(tx_direction * topology.tx_velocities_mps[batch, tx_axis])
        ))
        wavelength = SPEED_OF_LIGHT_MPS / self.config.carrier_frequency_hz
        doppler = radial / wavelength
        label = (
            f"tr37885.link.{min(rx_vehicle, tx_vehicle)}."
            f"{max(rx_vehicle, tx_vehicle)}.path.{path}.{batch}.episode.{episode}"
        )
        if self.config.moving_scatterers and path > 0:
            alpha = float(backend.to_scalar(sampling.random_stream.fork(f"{label}.alpha").uniform((1,))[0]))
            maximum_speed = _maximum_speed(topology, backend)
            scatterer = float(backend.to_scalar(
                sampling.random_stream.fork(f"{label}.scatterer").uniform(
                    (1,), -maximum_speed, maximum_speed
                )[0]
            ))
            doppler += 2.0 * alpha * scatterer / wavelength
        base = (
            backend.array(1.0 + 0.0j, dtype=execution.complex_dtype, device=execution.device)
            if condition in {LOS, NLOSV} and path == 0
            else sampling.random_stream.fork(f"{label}.coefficient").complex_normal(
                (1,), dtype=execution.complex_dtype
            )[0]
        )
        tx_yaw = float(backend.to_scalar(topology.tx_orientations_rad[batch, tx_axis, 0]))
        rx_yaw = float(backend.to_scalar(topology.rx_orientations_rad[batch, rx_axis, 0]))
        tx_type = int(backend.to_scalar(
            request.scene.vehicle_type_codes[batch, tx_vehicle]
        ))
        rx_type = int(backend.to_scalar(
            request.scene.vehicle_type_codes[batch, rx_vehicle]
        ))
        rx_steer = _vehicle_steering(
            self.config.num_rx_antennas, rx_type, aoa, rx_yaw, backend, execution
        )
        tx_steer = _vehicle_steering(
            self.config.num_tx_antennas, tx_type, aod, tx_yaw, backend, execution
        )
        matrix = base * rx_steer[:, None] * tx_steer[None, :] * sqrt(power)
        time = backend.arange(
            sampling.num_time_samples, dtype=execution.float_dtype, device=execution.device
        ) / sampling.sampling_frequency_hz
        time = time + request.scene.mobility_times_s[request.snapshot_index]
        evolution = backend.exp(1j * (2.0 * pi * doppler) * time)
        return matrix[..., None] * evolution, doppler


def _norm(vector, backend) -> float:
    return sqrt(float(backend.to_scalar(backend.sum(vector * vector))))


def _maximum_speed(topology, backend) -> float:
    maximum = 0.0
    for values in (topology.tx_velocities_mps, topology.rx_velocities_mps):
        shape = backend.shape(values)
        for batch in range(shape[0]):
            for endpoint in range(shape[1]):
                maximum = max(maximum, _norm(values[batch, endpoint], backend))
    return maximum


def _vehicle_steering(
    num_ports, vehicle_type, azimuth, yaw, backend, execution
):
    profile = load_vehicle_antenna_profile(vehicle_type)
    bearings = profile.port_bearings_rad(num_ports, yaw)
    location_count = len(profile.locations)
    values = []
    for index, bearing in enumerate(bearings):
        local_index = index // location_count
        phase = (
            2.0
            * pi
            * profile.element_spacing_wavelengths
            * local_index
            * cos(azimuth - bearing)
        )
        values.append(complex(cos(phase), sin(phase)))
    return backend.array(
        values, dtype=execution.complex_dtype, device=execution.device
    )


__all__ = ["UrbanChannelAssembler"]
