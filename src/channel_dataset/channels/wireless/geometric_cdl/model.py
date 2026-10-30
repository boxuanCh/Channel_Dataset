"""Dedicated geometrically consistent multi-carrier CDL model."""

from __future__ import annotations

from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from math import log, pi
from typing import Any

from channel_dataset.channels.wireless.antennas import load_antenna_layout
from channel_dataset.channels.wireless.large_scale import load_large_scale_parameters
from channel_dataset.channels.wireless.math import matrix_sqrt_psd
from channel_dataset.channels.wireless.math.constants import SPEED_OF_LIGHT_MPS
from channel_dataset.channels.wireless.rays import (
    load_cdl_ray_offsets,
    load_ray_generation_parameters,
)
from channel_dataset.channels.wireless.tr38901 import load_cdl_profile
from channel_dataset.channels.wireless.types import LargeScaleParameters
from channel_dataset.config import ConfigEnvelope
from channel_dataset.core import ChannelModel, RealizationMetadata
from channel_dataset.registry import ModelRegistry

from .config import GeometricCDLConfig, parse_geometric_cdl_config
from .fields import SpatialFieldBank
from .geometry import (
    GROUND_COMPONENT,
    LOS_COMPONENT,
    NLOS_COMPONENT,
    construct_scatterers,
    directions_from_angles,
    drift_paths,
)
from .resources import load_geometric_resource
from .transitions import segment_amplitude_weights
from .types import (
    CarrierClusterState,
    CarrierGeometricRealization,
    DynamicCIRBatch,
    GeometricCDLRealization,
    GeometricCDLSamplingRequest,
    GeometricCDLState,
    LargeScaleTrajectoryBatch,
    RayTrajectoryBatch,
)


def _software_version() -> str:
    try:
        return version("channel-dataset")
    except PackageNotFoundError:
        return "0+unknown"


def _profile_info(identifier: str) -> tuple[str, str]:
    value = identifier.replace("3GPP_38.901_", "").replace("-", "_")
    upper = value.upper()
    if upper.startswith("RMA"):
        family = "RMa"
    elif upper.startswith("UMA"):
        family = "UMa"
    elif upper.startswith("UMI"):
        family = "UMi"
    elif upper.startswith(("INH", "INDOOR")):
        family = "InH"
    elif upper.startswith("INF"):
        family = "InF"
    else:
        raise ValueError(f"unsupported geometric CDL scenario profile: {identifier!r}")
    if "NLOS" in upper:
        state = "nlos"
    elif "LOS" in upper:
        state = "los"
    else:
        state = "sampled"
    if "O2I" in upper:
        state = "o2i_" + state
    return family, state


def _dtype(metadata: RealizationMetadata) -> str:
    return "float32" if metadata.precision == "single" else "float64"


def _broadcast_link_endpoints(request: GeometricCDLSamplingRequest) -> tuple[Any, Any]:
    tx = request.trajectory.tx_positions_m[:, None, :, :, :]
    rx = request.trajectory.rx_positions_m[:, :, None, :, :]
    return tx + rx * 0.0, rx + tx * 0.0


def _rotation_matrices(orientations_rad: Any, backend: Any) -> Any:
    """Return batched local-to-global 3GPP bearing/downtilt/slant matrices."""

    a, b, g = (
        orientations_rad[..., 0],
        orientations_rad[..., 1],
        orientations_rad[..., 2],
    )
    ca, sa = backend.cos(a), backend.sin(a)
    cb, sb = backend.cos(b), backend.sin(b)
    cg, sg = backend.cos(g), backend.sin(g)
    zero, one = a * 0.0, a * 0.0 + 1.0
    rz = backend.stack(
        (
            backend.stack((ca, -sa, zero), axis=-1),
            backend.stack((sa, ca, zero), axis=-1),
            backend.stack((zero, zero, one), axis=-1),
        ),
        axis=-2,
    )
    ry = backend.stack(
        (
            backend.stack((cb, zero, sb), axis=-1),
            backend.stack((zero, one, zero), axis=-1),
            backend.stack((-sb, zero, cb), axis=-1),
        ),
        axis=-2,
    )
    rx = backend.stack(
        (
            backend.stack((one, zero, zero), axis=-1),
            backend.stack((zero, cg, -sg), axis=-1),
            backend.stack((zero, sg, cg), axis=-1),
        ),
        axis=-2,
    )
    return backend.matmul(backend.matmul(rz, ry), rx)


class GeometricCDLModel(ChannelModel[GeometricCDLSamplingRequest, GeometricCDLRealization]):
    """QuaDRiGa-like persistent-scatterer CDL evolution engine."""

    def __init__(self, config: GeometricCDLConfig) -> None:
        self.config = config
        resolved = asdict(config)
        self._digest = ConfigEnvelope(1, "wireless", "geometric_cdl", resolved).model_digest
        self._resolved = resolved
        self.evolution_resource = load_geometric_resource(
            config.standard_version, "geometric_evolution.yaml"
        )
        self.tx_layout = load_antenna_layout(config.tx_array_profile)
        self.rx_layout = load_antenna_layout(config.rx_array_profile)

    @property
    def family(self) -> str:
        return "wireless"

    @property
    def name(self) -> str:
        return "geometric_cdl"

    def _metadata(self, request: GeometricCDLSamplingRequest) -> RealizationMetadata:
        sampling = request.sampling
        return RealizationMetadata(
            "wireless", self.name, 1, sampling.execution.backend,
            sampling.execution.precision, sampling.execution.device,
            sampling.random_stream.stream_id, self._digest, self._resolved,
            _software_version(), sampling.request_id, sampling.random_stream.root_seed,
            "1", self.config.standard_version,
        )

    def _validate_request(self, request: GeometricCDLSamplingRequest) -> None:
        if not isinstance(request, GeometricCDLSamplingRequest):
            raise TypeError("request must be GeometricCDLSamplingRequest")
        backend = request.sampling.execution.array_backend
        shortest = SPEED_OF_LIGHT_MPS / max(self.config.carrier_frequencies_hz)
        maximum = 0.25 * shortest
        for positions in (request.trajectory.tx_positions_m, request.trajectory.rx_positions_m):
            if request.trajectory.shape[-1] > 1:
                delta = positions[:, :, 1:, :] - positions[:, :, :-1, :]
                displacement = backend.sqrt(backend.sum(delta * delta, axis=-1))
                if float(backend.to_scalar(backend.max(displacement))) > maximum * (1.0 + 1.0e-6):
                    raise ValueError(
                        "trajectory displacement exceeds one quarter of the shortest wavelength"
                    )
        prior = request.prior_state
        if prior is not None:
            if prior.model_digest != self._digest:
                raise ValueError("prior state was produced by another configuration")
            if prior.root_seed != request.sampling.random_stream.root_seed:
                raise ValueError("prior state and request use different root seeds")
            if prior.stream_id != request.sampling.random_stream.stream_id:
                raise ValueError("prior state and request use different random stream lineages")
            for previous, current, name in (
                (prior.tx_endpoint_ids, request.trajectory.tx_endpoint_ids, "transmitter"),
                (prior.rx_endpoint_ids, request.trajectory.rx_endpoint_ids, "receiver"),
            ):
                if backend.shape(previous) != backend.shape(current) or backend.to_scalar(
                    backend.max(backend.abs(previous - current))
                ) != 0:
                    raise ValueError(f"prior state {name} identities do not match the trajectory")
            if len(prior.carrier_states) != len(self.config.carrier_frequencies_hz):
                raise ValueError("prior state carrier count does not match configuration")
            for current, expected in zip(prior.carrier_states, self.config.carrier_frequencies_hz, strict=True):
                if current.carrier_frequency_hz != expected:
                    raise ValueError("prior state carrier order does not match configuration")

    def _field_bank(
        self, request: GeometricCDLSamplingRequest, metadata: RealizationMetadata
    ) -> SpatialFieldBank:
        if request.prior_state is not None:
            return SpatialFieldBank(
                request.prior_state.field_directions_per_m,
                request.prior_state.field_phases_rad,
                self.config.stationarity_distance_m or 15.0,
                self.config.spatial_acf,
            )
        return SpatialFieldBank.sample(
            8, self.config.num_sinusoids, self.config.stationarity_distance_m or 15.0,
            request.sampling.random_stream.fork("wireless.geometric_cdl.spatial_fields"),
            dtype=_dtype(metadata), acf=self.config.spatial_acf,
        )

    def _field_values(
        self, request: GeometricCDLSamplingRequest, bank: SpatialFieldBank
    ) -> Any:
        tx, rx = _broadcast_link_endpoints(request)
        return bank.evaluate_dual(tx, rx)

    def _segment_boundaries(
        self, request: GeometricCDLSamplingRequest, los_mask: Any
    ) -> tuple[int, ...]:
        backend = request.sampling.execution.array_backend
        time_count = request.trajectory.shape[-1]
        boundaries = set(request.explicit_segment_boundaries or ())
        if request.prior_state is not None:
            scenario_changed = backend.to_scalar(backend.max(backend.abs(
                request.scenarios.scenario_codes[..., 0]
                - request.prior_state.last_scenario_codes
            ))) > 0
            los_changed = backend.to_scalar(backend.max(backend.astype(
                los_mask[..., 0] != request.prior_state.last_los_mask, "int64"
            ))) > 0
            if scenario_changed or los_changed:
                boundaries.add(0)
        for time in range(1, time_count):
            scenario_changed = backend.to_scalar(backend.max(backend.abs(
                request.scenarios.scenario_codes[..., time]
                - request.scenarios.scenario_codes[..., time - 1]
            ))) > 0
            los_changed = backend.to_scalar(backend.max(backend.astype(
                los_mask[..., time] != los_mask[..., time - 1], "int64"
            ))) > 0
            if scenario_changed or los_changed:
                boundaries.add(time)
        if request.explicit_segment_boundaries is None and self.config.automatic_segmentation:
            threshold = self.config.stationarity_distance_m or 15.0
            anchor = 0
            for time in range(1, time_count):
                tx_delta = (
                    request.trajectory.tx_positions_m[:, :, time, :]
                    - request.trajectory.tx_positions_m[:, :, anchor, :]
                )
                rx_delta = (
                    request.trajectory.rx_positions_m[:, :, time, :]
                    - request.trajectory.rx_positions_m[:, :, anchor, :]
                )
                displacement = max(
                    float(backend.to_scalar(backend.max(backend.sqrt(backend.sum(tx_delta * tx_delta, axis=-1))))),
                    float(backend.to_scalar(backend.max(backend.sqrt(backend.sum(rx_delta * rx_delta, axis=-1))))),
                )
                if displacement >= threshold:
                    boundaries.add(time)
                    anchor = time
        return tuple(sorted(boundaries))

    def _carrier_generations(
        self,
        carrier: CarrierClusterState,
        boundaries: tuple[int, ...],
        request: GeometricCDLSamplingRequest,
    ) -> tuple[CarrierClusterState, ...]:
        if not boundaries:
            return (carrier,)
        backend, execution = (
            request.sampling.execution.array_backend,
            request.sampling.execution,
        )
        generations = [carrier]
        base_generation = int(backend.to_scalar(backend.max(carrier.generation_ids)))
        for offset, boundary in enumerate(boundaries, start=1):
            generation = base_generation + offset
            carrier_key = (
                "shared" if self.config.multiband_correlation == "shared_latent"
                else str(carrier.carrier_frequency_hz)
            )
            absolute_boundary = boundary + (
                request.prior_state.last_snapshot_index + 1
                if request.prior_state is not None else 0
            )
            label = (
                "wireless.geometric_cdl.transition."
                f"{carrier_key}.{generation}.{absolute_boundary}"
            )
            initial = request.sampling.random_stream.fork(label + ".phase").uniform(
                carrier.initial_phase_rad.shape, -pi, pi, dtype=execution.float_dtype
            )
            polarization = request.sampling.random_stream.fork(label + ".polarization").uniform(
                carrier.polarization_phases_rad.shape, -pi, pi, dtype=execution.float_dtype
            )
            generation_ids = backend.full(
                carrier.generation_ids.shape, generation, dtype="int64", device=execution.device
            )
            cluster_ids = carrier.cluster_ids + offset * carrier.cluster_ids.shape[-1]
            # Persistent fields keep replacement clusters spatially correlated;
            # a small deterministic displacement prevents identical geometry.
            direction = backend.stack((
                carrier.fbs_positions_m[..., 1] * 0.0 + 1.0,
                carrier.fbs_positions_m[..., 1] * 0.0,
                carrier.fbs_positions_m[..., 1] * 0.0,
            ), axis=-1)
            displacement = 0.05 * generation * direction
            generations.append(replace(
                carrier,
                fbs_positions_m=carrier.fbs_positions_m + displacement,
                lbs_positions_m=carrier.lbs_positions_m - displacement,
                cluster_ids=cluster_ids,
                generation_ids=generation_ids,
                initial_phase_rad=initial,
                polarization_phases_rad=polarization,
            ))
        return tuple(generations)

    def _los_mask(self, request: GeometricCDLSamplingRequest, fields: Any) -> Any:
        backend = request.sampling.execution.array_backend
        if request.trajectory.los_mask is not None:
            return request.trajectory.los_mask
        if request.scenarios.propagation_state_codes is not None:
            return request.scenarios.propagation_state_codes == 1
        tx, rx = _broadcast_link_endpoints(request)
        distance = backend.sqrt(backend.sum((rx - tx) ** 2, axis=-1))
        probability = backend.exp(-distance / 100.0)
        sampled = SpatialFieldBank.normal_to_uniform(fields[..., 0]) < probability
        result = sampled
        for code, profile in request.scenarios.profiles.items():
            _, state = _profile_info(profile)
            selected = request.scenarios.scenario_codes == code
            if state.endswith("nlos"):
                result = backend.where(selected, selected & False, result)
            elif state.endswith("los"):
                result = backend.where(selected, selected | True, result)
        return result

    def _large_scale(
        self,
        request: GeometricCDLSamplingRequest,
        metadata: RealizationMetadata,
        fields: Any,
        los_mask: Any,
        carrier_frequency_hz: float,
    ) -> LargeScaleTrajectoryBatch:
        backend = request.sampling.execution.array_backend
        dtype, device = request.sampling.execution.float_dtype, request.sampling.execution.device
        shape = request.trajectory.shape
        values = backend.zeros((*shape, 7), dtype=dtype, device=device)
        xpr = fields[..., 7] * 3.0 + 8.0
        o2i = backend.zeros(shape, dtype=dtype, device=device)
        for code, profile in request.scenarios.profiles.items():
            family, state = _profile_info(profile)
            parameters = load_large_scale_parameters(family)
            matrix = backend.array(
                parameters["cross_correlation_matrix"], dtype=dtype, device=device
            )
            correlated = backend.einsum(
                "ij,...j->...i", matrix_sqrt_psd(matrix), fields[..., :7]
            )
            los_stats, nlos_stats = parameters["los"], parameters["nlos"]
            condition_los = los_mask[..., None]
            mean_los = backend.array(los_stats["means"], dtype=dtype, device=device)
            mean_nlos = backend.array(nlos_stats["means"], dtype=dtype, device=device)
            std_los = backend.array(los_stats["stddevs"], dtype=dtype, device=device)
            std_nlos = backend.array(nlos_stats["stddevs"], dtype=dtype, device=device)
            candidate = backend.where(
                condition_los,
                mean_los + std_los * correlated,
                mean_nlos + std_nlos * correlated,
            )
            selection = request.scenarios.scenario_codes == code
            values = backend.where(selection[..., None], candidate, values)
            if state.startswith("o2i"):
                o2i = backend.where(selection, o2i * 0.0 + float(parameters["o2i_loss_db"]), o2i)
        ln10 = log(10.0)
        delay = backend.exp(values[..., 0] * ln10)
        asd = backend.exp(values[..., 1] * ln10) * pi / 180.0
        asa = backend.exp(values[..., 2] * ln10) * pi / 180.0
        zsa = backend.exp(values[..., 5] * ln10) * pi / 180.0
        zsd = backend.exp(values[..., 6] * ln10) * pi / 180.0
        tx, rx = _broadcast_link_endpoints(request)
        distance = backend.sqrt(backend.sum((rx - tx) ** 2, axis=-1)) + 1.0e-9
        # Free-space reference plus a conservative NLOS excess term. Versioned
        # resources own the stochastic LSP calibration; this deterministic
        # transform remains independently replaceable.
        path_loss = (
            32.4 + 20.0 * backend.log10(distance / 1000.0)
            + 20.0 * backend.log10(distance * 0.0 + carrier_frequency_hz / 1.0e6)
        )
        path_loss = path_loss + backend.where(los_mask, path_loss * 0.0, path_loss * 0.0 + 20.0) + o2i
        shadow = values[..., 3]
        if not self.config.apply_path_loss:
            path_loss = path_loss * 0.0
        if not self.config.apply_shadow_fading:
            shadow = shadow * 0.0
        return LargeScaleTrajectoryBatch(
            delay, asa, asd, zsa, zsd, shadow, values[..., 4], xpr,
            path_loss, los_mask, metadata,
        )

    def _snapshot_lsp(
        self, trajectory: LargeScaleTrajectoryBatch, time_index: int
    ) -> LargeScaleParameters:
        return LargeScaleParameters(
            trajectory.delay_spread_s[..., time_index],
            trajectory.azimuth_spread_arrival_rad[..., time_index],
            trajectory.azimuth_spread_departure_rad[..., time_index],
            trajectory.zenith_spread_arrival_rad[..., time_index],
            trajectory.zenith_spread_departure_rad[..., time_index],
            trajectory.shadow_fading_db[..., time_index],
            trajectory.k_factor_db[..., time_index],
            trajectory.path_loss_db[..., time_index],
            trajectory.metadata,
        )

    def _scenario_anchor(
        self,
        request: GeometricCDLSamplingRequest,
        metadata: RealizationMetadata,
        large_scale: LargeScaleTrajectoryBatch,
        carrier_frequency_hz: float,
    ) -> CarrierClusterState:
        backend = request.sampling.execution.array_backend
        execution, stream = request.sampling.execution, request.sampling.random_stream
        stream_key = (
            "shared" if self.config.multiband_correlation == "shared_latent"
            else str(carrier_frequency_hz)
        )
        b, nr, nt, _ = request.trajectory.shape
        max_clusters, rays = 25, self.config.rays_per_cluster
        shape = (b, nr, nt)
        lsp = self._snapshot_lsp(large_scale, 0)
        los = large_scale.los_mask[..., 0]
        counts = backend.zeros(shape, dtype=execution.float_dtype, device=execution.device)
        delay_scaling = counts + 2.1
        shadow_std = counts + 3.0
        xpr_mean = counts + 8.0
        xpr_std = counts + 3.0
        for code, profile in request.scenarios.profiles.items():
            family, _ = _profile_info(profile)
            parameters = load_ray_generation_parameters(family)
            selected = request.scenarios.scenario_codes[..., 0] == code
            los_values, nlos_values = parameters["los"], parameters["nlos"]
            count_value = backend.where(los, counts + los_values.num_clusters, counts + nlos_values.num_clusters)
            scale_value = backend.where(los, counts + los_values.delay_scaling, counts + nlos_values.delay_scaling)
            shadow_value = backend.where(los, counts + los_values.cluster_shadow_std_db, counts + nlos_values.cluster_shadow_std_db)
            xpr_mean_value = backend.where(los, counts + los_values.xpr_mean_db, counts + nlos_values.xpr_mean_db)
            xpr_std_value = backend.where(los, counts + los_values.xpr_std_db, counts + nlos_values.xpr_std_db)
            counts = backend.where(selected, count_value, counts)
            delay_scaling = backend.where(selected, scale_value, delay_scaling)
            shadow_std = backend.where(selected, shadow_value, shadow_std)
            xpr_mean = backend.where(selected, xpr_mean_value, xpr_mean)
            xpr_std = backend.where(selected, xpr_std_value, xpr_std)
        cluster_index = backend.reshape(
            backend.arange(max_clusters, dtype="int64", device=execution.device),
            (1, 1, 1, max_clusters),
        )
        cluster_mask = cluster_index < counts[..., None]
        uniforms = stream.fork(f"wireless.geometric_cdl.anchor.{stream_key}.delays").uniform(
            (*shape, max_clusters), 1.0e-7, 1.0, dtype=execution.float_dtype
        )
        delays = -delay_scaling[..., None] * lsp.delay_spread_s[..., None] * backend.log(uniforms)
        minimum = backend.min(backend.where(cluster_mask, delays, delays * 0.0 + 1.0e30), axis=-1)
        delays = backend.where(cluster_mask, delays - minimum[..., None], delays * 0.0)
        cluster_shadow = stream.fork(f"wireless.geometric_cdl.anchor.{stream_key}.shadow").normal(
            (*shape, max_clusters), dtype=execution.float_dtype
        )
        power = backend.exp(
            -delays * (delay_scaling[..., None] - 1.0)
            / (delay_scaling[..., None] * lsp.delay_spread_s[..., None] + 1.0e-30)
            - cluster_shadow * shadow_std[..., None] * log(10.0) / 10.0
        ) * backend.astype(cluster_mask, execution.float_dtype)
        power = power / (backend.sum(power, axis=-1)[..., None] + 1.0e-30)
        k_linear = backend.exp(lsp.k_factor_db * log(10.0) / 10.0)
        diffuse_scale = backend.where(los, 1.0 / (1.0 + k_linear), k_linear * 0.0 + 1.0)
        power = power * diffuse_scale[..., None]
        first = backend.where(los, k_linear / (1.0 + k_linear), k_linear * 0.0)
        power = backend.concatenate(((power[..., :1] + first[..., None]), power[..., 1:]), axis=-1)
        topology = request.trajectory.snapshot(0, metadata=metadata)
        direct = topology.rx_positions_m[:, :, None, :] - topology.tx_positions_m[:, None, :, :]
        direct_azimuth = backend.atan2(direct[..., 1], direct[..., 0])
        direct_zenith = backend.atan2(
            backend.sqrt(direct[..., 0] ** 2 + direct[..., 1] ** 2 + 1.0e-30), direct[..., 2]
        )
        specifications = (
            ("aoa", direct_azimuth, lsp.azimuth_spread_arrival_rad),
            ("zoa", direct_zenith, lsp.zenith_spread_arrival_rad),
            ("aod", direct_azimuth, lsp.azimuth_spread_departure_rad),
            ("zod", direct_zenith, lsp.zenith_spread_departure_rad),
        )
        angles = []
        for label, center, spread in specifications:
            cluster_normal = stream.fork(
                f"wireless.geometric_cdl.anchor.{stream_key}.{label}.cluster"
            ).normal((*shape, max_clusters), dtype=execution.float_dtype)
            means = center[..., None] + spread[..., None] * cluster_normal
            if self.config.subpath_method == "laplacian":
                uniform_offsets = stream.fork(
                    f"wireless.geometric_cdl.anchor.{stream_key}.{label}.laplacian"
                ).uniform(
                    (*shape, max_clusters, rays),
                    -0.5 + 1.0e-7,
                    0.5 - 1.0e-7,
                    dtype=execution.float_dtype,
                )
                signs = backend.where(
                    uniform_offsets >= 0.0,
                    uniform_offsets * 0.0 + 1.0,
                    uniform_offsets * 0.0 - 1.0,
                )
                offset_grid = -signs * backend.log(
                    1.0 - 2.0 * backend.abs(uniform_offsets)
                )
                offset_grid = offset_grid / (
                    backend.sqrt(backend.mean(offset_grid**2, axis=-1))[..., None]
                    + 1.0e-30
                )
            elif rays == 20:
                offsets = backend.array(load_cdl_ray_offsets(), dtype=execution.float_dtype, device=execution.device)
            else:
                offsets = (
                    backend.arange(rays, dtype=execution.float_dtype, device=execution.device)
                    - 0.5 * (rays - 1)
                ) / max(1.0, 0.25 * rays)
            if self.config.subpath_method == "legacy":
                offset_grid = backend.broadcast_to(
                    backend.reshape(offsets, (1, 1, 1, 1, rays)),
                    (*shape, max_clusters, rays),
                )
            keys = stream.fork(
                f"wireless.geometric_cdl.anchor.{stream_key}.{label}.coupling"
            ).uniform((*shape, max_clusters, rays), dtype=execution.float_dtype)
            coupled = backend.take_along_axis(offset_grid, backend.argsort(keys, axis=-1), axis=-1)
            values = means[..., None] + spread[..., None, None] * coupled
            if label.startswith("z"):
                values = backend.abs(backend.atan2(backend.sin(values), backend.cos(values)))
            else:
                values = backend.atan2(backend.sin(values), backend.cos(values))
            angles.append(values)
        ray_mask = backend.broadcast_to(cluster_mask[..., None], (*shape, max_clusters, rays))
        xpr_normal = stream.fork(f"wireless.geometric_cdl.anchor.{stream_key}.xpr").normal(
            (*shape, max_clusters, rays), dtype=execution.float_dtype
        )
        xpr = backend.exp(
            (xpr_mean[..., None, None] + xpr_std[..., None, None] * xpr_normal)
            * log(10.0) / 10.0
        )
        initial = stream.fork(f"wireless.geometric_cdl.anchor.{stream_key}.phase").uniform(
            (*shape, max_clusters, rays), -pi, pi, dtype=execution.float_dtype
        )
        polarization = stream.fork(
            f"wireless.geometric_cdl.anchor.{stream_key}.polarization"
        ).uniform((*shape, max_clusters, rays, 4), -pi, pi, dtype=execution.float_dtype)
        component = backend.full((*shape, max_clusters), NLOS_COMPONENT, dtype="int64", device=execution.device)
        first_component = backend.where(los, component[..., 0] * 0 + LOS_COMPONENT, component[..., 0])
        component = backend.concatenate((first_component[..., None], component[..., 1:]), axis=-1)
        if self.config.ground_reflection and max_clusters > 1:
            ground = backend.where(los, component[..., 1] * 0 + GROUND_COMPONENT, component[..., 1])
            component = backend.concatenate((component[..., :1], ground[..., None], component[..., 2:]), axis=-1)
        tx = topology.tx_positions_m[:, None, :, None, None, :] + angles[0][..., None] * 0.0
        rx = topology.rx_positions_m[:, :, None, None, None, :] + angles[0][..., None] * 0.0
        fbs, lbs, residual, fallback_mask = construct_scatterers(
            tx, rx, angles[2], angles[3], angles[0], angles[1], delays[..., None], component[..., None],
            tolerance_m=self.config.solver_tolerance,
            max_iterations=self.config.solver_max_iterations,
            fallback=self.config.solver_fallback,
            minimum_bounce_separation_m=self.config.minimum_bounce_separation_m,
        )
        if self.config.solver_fallback == "mask":
            ray_mask = ray_mask & (~fallback_mask)
        cluster_ids = backend.broadcast_to(cluster_index, (*shape, max_clusters))
        ray_index = backend.reshape(
            backend.arange(rays, dtype="int64", device=execution.device), (1, 1, 1, 1, rays)
        )
        ray_ids = backend.broadcast_to(ray_index, (*shape, max_clusters, rays))
        generation = backend.zeros((*shape, max_clusters), dtype="int64", device=execution.device)
        ray_fraction = backend.full(
            (*shape, max_clusters, rays), 1.0 / rays,
            dtype=execution.float_dtype, device=execution.device,
        )
        return CarrierClusterState(
            carrier_frequency_hz, fbs, lbs, cluster_ids, ray_ids, generation, component,
            initial, polarization, xpr, power, ray_fraction, ray_mask, residual, fallback_mask,
        )

    def _fixed_anchor(
        self,
        request: GeometricCDLSamplingRequest,
        metadata: RealizationMetadata,
        carrier_frequency_hz: float,
    ) -> CarrierClusterState:
        backend, execution = request.sampling.execution.array_backend, request.sampling.execution
        stream_key = (
            "shared" if self.config.multiband_correlation == "shared_latent"
            else str(carrier_frequency_hz)
        )
        profile = load_cdl_profile(self.config.fixed_profile or "CDL-A")
        b, nr, nt, _ = request.trajectory.shape
        c, rays = profile.num_clusters, self.config.rays_per_cluster
        link_shape = (b, nr, nt)
        def clusters(values: tuple[float, ...]) -> Any:
            array = backend.array(values, dtype=execution.float_dtype, device=execution.device)
            return backend.broadcast_to(backend.reshape(array, (1, 1, 1, c)), (*link_shape, c))
        delays = clusters(tuple(value * 100.0e-9 for value in profile.delays))
        raw_power = backend.exp(clusters(profile.powers_db) * log(10.0) / 10.0)
        power = raw_power / backend.sum(raw_power, axis=-1)[..., None]
        offsets = backend.array(load_cdl_ray_offsets(), dtype=execution.float_dtype, device=execution.device)
        if rays != 20:
            offsets = (
                backend.arange(rays, dtype=execution.float_dtype, device=execution.device)
                - 0.5 * (rays - 1)
            ) / max(1.0, 0.25 * rays)
        offsets = backend.reshape(offsets, (1, 1, 1, 1, rays))
        specs = (
            (profile.aoa_degrees, profile.asa_degrees), (profile.zoa_degrees, profile.zsa_degrees),
            (profile.aod_degrees, profile.asd_degrees), (profile.zod_degrees, profile.zsd_degrees),
        )
        angles = []
        for means, spread in specs:
            mean = clusters(tuple(value * pi / 180.0 for value in means))[..., None]
            angles.append(mean + offsets * (spread * pi / 180.0))
        shape = (*link_shape, c, rays)
        angles = [backend.broadcast_to(value, shape) for value in angles]
        stream = request.sampling.random_stream
        initial = stream.fork(f"wireless.geometric_cdl.fixed.{stream_key}.phase").uniform(
            shape, -pi, pi, dtype=execution.float_dtype
        )
        polarization = stream.fork(f"wireless.geometric_cdl.fixed.{stream_key}.polarization").uniform(
            (*shape, 4), -pi, pi, dtype=execution.float_dtype
        )
        xpr = backend.full(shape, 10.0 ** (profile.xpr_db / 10.0), dtype=execution.float_dtype, device=execution.device)
        topology = request.trajectory.snapshot(0, metadata=metadata)
        tx = topology.tx_positions_m[:, None, :, None, None, :] + angles[0][..., None] * 0.0
        rx = topology.rx_positions_m[:, :, None, None, None, :] + angles[0][..., None] * 0.0
        component = backend.full((*link_shape, c), NLOS_COMPONENT, dtype="int64", device=execution.device)
        if profile.has_los:
            component = backend.concatenate((component[..., :1] * 0, component[..., 1:]), axis=-1)
        fbs, lbs, residual, fallback_mask = construct_scatterers(
            tx, rx, angles[2], angles[3], angles[0], angles[1], delays[..., None], component[..., None],
            tolerance_m=self.config.solver_tolerance, max_iterations=self.config.solver_max_iterations,
            fallback=self.config.solver_fallback,
            minimum_bounce_separation_m=self.config.minimum_bounce_separation_m,
        )
        cluster_ids = backend.broadcast_to(
            backend.reshape(backend.arange(c, dtype="int64", device=execution.device), (1, 1, 1, c)),
            (*link_shape, c),
        )
        ray_ids = backend.broadcast_to(
            backend.reshape(backend.arange(rays, dtype="int64", device=execution.device), (1, 1, 1, 1, rays)),
            shape,
        )
        return CarrierClusterState(
            carrier_frequency_hz, fbs, lbs, cluster_ids, ray_ids,
            backend.zeros((*link_shape, c), dtype="int64", device=execution.device), component,
            initial, polarization, xpr, power,
            backend.full(shape, 1.0 / rays, dtype=execution.float_dtype, device=execution.device),
            backend.full(shape, True, dtype="bool", device=execution.device), residual, fallback_mask,
        )

    def initialize_anchors(
        self, request: GeometricCDLSamplingRequest
    ) -> GeometricCDLState:
        """Initialize persistent spatial fields and carrier scatterers."""

        self._validate_request(request)
        if request.prior_state is not None:
            return request.prior_state
        metadata = self._metadata(request)
        bank = self._field_bank(request, metadata)
        fields = self._field_values(request, bank)
        los = self._los_mask(request, fields)
        carrier_states = []
        for frequency in self.config.carrier_frequencies_hz:
            large_scale = self._large_scale(request, metadata, fields, los, frequency)
            carrier_states.append(
                self._fixed_anchor(request, metadata, frequency)
                if self.config.initialization_mode == "fixed_profile"
                else self._scenario_anchor(request, metadata, large_scale, frequency)
            )
        backend = request.sampling.execution.array_backend
        b, nr, nt, _ = request.trajectory.shape
        return GeometricCDLState(
            bank.directions_per_m, bank.phases_rad,
            request.trajectory.tx_positions_m[:, :, 0, :],
            request.trajectory.rx_positions_m[:, :, 0, :],
            tuple(carrier_states), request.trajectory.tx_endpoint_ids,
            request.trajectory.rx_endpoint_ids,
            backend.zeros((b, nr, nt), dtype="int64", device=request.sampling.execution.device),
            request.scenarios.scenario_codes[..., 0], los[..., 0],
            tuple(state.fbs_positions_m[..., 0] * 0.0 for state in carrier_states),
            float(backend.to_scalar(request.trajectory.times_s[0])), 0,
            self._digest, request.sampling.random_stream.stream_id,
            request.sampling.random_stream.root_seed, metadata,
        )

    def _coefficient_cir(
        self,
        request: GeometricCDLSamplingRequest,
        rays: RayTrajectoryBatch,
        large_scale: LargeScaleTrajectoryBatch,
        frequency: float,
        phase_reference_path_m: Any,
    ) -> DynamicCIRBatch:
        backend, execution = request.sampling.execution.array_backend, request.sampling.execution
        rx_direction = directions_from_angles(rays.azimuth_arrival_rad, rays.zenith_arrival_rad, backend)
        tx_direction = directions_from_angles(rays.azimuth_departure_rad, rays.zenith_departure_rad, backend)
        rx_positions = self.rx_layout.array.positions_wavelengths(execution)
        tx_positions = self.tx_layout.array.positions_wavelengths(execution)
        rx_rotation = _rotation_matrices(
            request.trajectory.rx_orientations_rad, backend
        )
        tx_rotation = _rotation_matrices(
            request.trajectory.tx_orientations_rad, backend
        )
        rx_positions_global = backend.einsum(
            "brnij,aj->brnai", rx_rotation, rx_positions
        )
        tx_positions_global = backend.einsum(
            "bxnij,zj->bxnzi", tx_rotation, tx_positions
        )
        rx_spatial = backend.einsum(
            "brxcmnj,brnaj->brxcmna", rx_direction, rx_positions_global
        )
        tx_spatial = backend.einsum(
            "brxcmnj,bxnzj->brxcmnz", tx_direction, tx_positions_global
        )
        path_phase = (
            rays.initial_phase_rad[..., None]
            - 2.0 * pi * frequency / SPEED_OF_LIGHT_MPS
            * (rays.absolute_path_length_m - phase_reference_path_m[..., None])
        )
        phases = rays.polarization_phases_rad
        cross = 1.0 / backend.sqrt(rays.xpr_linear[..., 0] + 1.0e-30)
        polarization = 0.5 * (
            backend.exp(1j * phases[..., 0]) + backend.exp(1j * phases[..., 3])
            + cross * (backend.exp(1j * phases[..., 1]) + backend.exp(1j * phases[..., 2]))
        )
        amplitude = backend.sqrt(
            rays.cluster_power_linear[..., None, :]
            * rays.ray_power_fraction + 1.0e-30
        ) * rays.amplitude_weight
        amplitude = amplitude * backend.astype(rays.ray_mask, execution.float_dtype)
        amplitude = amplitude * backend.exp(-rays.blockage_loss_db * log(10.0) / 20.0)
        scalar = amplitude * polarization[..., None] * backend.exp(1j * path_phase)
        spatial = backend.exp(
            1j * 2.0 * pi * (rx_spatial[..., :, None] + tx_spatial[..., None, :])
        )
        ray_values = scalar[..., None, None] * spatial
        cluster_values = backend.sum(ray_values, axis=4)
        coefficients = backend.transpose(cluster_values, (0, 1, 5, 2, 6, 3, 4))
        total_loss = large_scale.path_loss_db + large_scale.shadow_fading_db
        coefficients = coefficients * backend.exp(-total_loss * log(10.0) / 20.0)[:, :, None, :, None, None, :]
        delays = backend.mean(rays.excess_delay_s, axis=4)
        path_mask = backend.max(backend.astype(rays.ray_mask, "int64"), axis=4) > 0
        return DynamicCIRBatch(coefficients, delays, rays.metadata, path_mask)

    def evolve(
        self,
        state: GeometricCDLState,
        request: GeometricCDLSamplingRequest,
    ) -> tuple[CarrierGeometricRealization, ...]:
        """Drift persistent scatterers and assemble every configured carrier."""

        metadata = self._metadata(request)
        bank = SpatialFieldBank(
            state.field_directions_per_m, state.field_phases_rad,
            self.config.stationarity_distance_m or 15.0, self.config.spatial_acf,
        )
        fields = self._field_values(request, bank)
        los = self._los_mask(request, fields)
        boundaries = self._segment_boundaries(request, los)
        weights = segment_amplitude_weights(
            boundaries, request.trajectory.shape[-1], self.config.transition_overlap_fraction,
            request.sampling.execution.array_backend,
            dtype=request.sampling.execution.float_dtype,
            device=request.sampling.execution.device,
        )
        backend = request.sampling.execution.array_backend
        outputs = []
        for carrier_index, carrier in enumerate(state.carrier_states):
            large_scale = self._large_scale(
                request, metadata, fields, los, carrier.carrier_frequency_hz
            )
            tx, rx = _broadcast_link_endpoints(request)
            direct = backend.sqrt(backend.sum((rx - tx) ** 2, axis=-1))
            time_count = request.trajectory.shape[-1]
            generations = self._carrier_generations(carrier, boundaries, request)
            path_values, aoa_values, zoa_values, aod_values, zod_values = [], [], [], [], []
            absolute_delay_values, excess_values = [], []
            cluster_power_values, fraction_values, amplitude_values = [], [], []
            xpr_values, blockage_values, mask_values = [], [], []
            cluster_ids, ray_ids, generation_ids, component_types = [], [], [], []
            initial_phases, polarization_phases, fbs_values, lbs_values = [], [], [], []
            phase_references = []
            for generation_index, generation in enumerate(generations):
                path, aoa, zoa, aod, zod = drift_paths(
                    request.trajectory.tx_positions_m, request.trajectory.rx_positions_m,
                    generation.fbs_positions_m, generation.lbs_positions_m,
                    generation.component_type_codes,
                )
                excess = backend.where(
                    path > direct[..., None, None, :],
                    path - direct[..., None, None, :], path * 0.0,
                ) / SPEED_OF_LIGHT_MPS
                weight = weights[generation_index]
                cluster_power = backend.broadcast_to(
                    generation.cluster_power_linear[..., None],
                    (*generation.cluster_power_linear.shape, time_count),
                )
                fraction = backend.broadcast_to(
                    generation.ray_power_fraction[..., None],
                    (*generation.ray_power_fraction.shape, time_count),
                )
                active = backend.broadcast_to(
                    generation.ray_mask[..., None], (*generation.ray_mask.shape, time_count)
                ) & (weight > 0.0)
                path_values.append(path)
                absolute_delay_values.append(path / SPEED_OF_LIGHT_MPS)
                excess_values.append(excess)
                aoa_values.append(aoa)
                zoa_values.append(zoa)
                aod_values.append(aod)
                zod_values.append(zod)
                cluster_power_values.append(cluster_power)
                fraction_values.append(fraction)
                amplitude_values.append(path * 0.0 + weight)
                xpr_values.append(backend.broadcast_to(generation.xpr_linear[..., None], path.shape))
                blockage_values.append(path * 0.0)
                mask_values.append(active)
                cluster_ids.append(generation.cluster_ids)
                ray_ids.append(generation.ray_ids)
                generation_ids.append(generation.generation_ids)
                component_types.append(generation.component_type_codes)
                initial_phases.append(generation.initial_phase_rad)
                polarization_phases.append(generation.polarization_phases_rad)
                fbs_values.append(generation.fbs_positions_m)
                lbs_values.append(generation.lbs_positions_m)
                if generation_index == 0 and request.prior_state is not None:
                    phase_references.append(state.final_path_lengths_m[carrier_index])
                else:
                    anchor_time = 0 if generation_index == 0 else boundaries[generation_index - 1]
                    phase_references.append(path[..., anchor_time])
            concatenate = lambda values: backend.concatenate(tuple(values), axis=3)
            rays = RayTrajectoryBatch(
                concatenate(path_values), concatenate(absolute_delay_values), concatenate(excess_values),
                concatenate(aoa_values), concatenate(zoa_values), concatenate(aod_values), concatenate(zod_values),
                concatenate(cluster_power_values), concatenate(fraction_values),
                concatenate(amplitude_values), concatenate(xpr_values), concatenate(blockage_values),
                concatenate(mask_values), concatenate(cluster_ids), concatenate(ray_ids),
                concatenate(generation_ids), concatenate(component_types), concatenate(initial_phases),
                concatenate(polarization_phases), concatenate(fbs_values), concatenate(lbs_values), metadata,
            )
            phase_reference = concatenate(phase_references)
            cir = self._coefficient_cir(
                request, rays, large_scale, carrier.carrier_frequency_hz, phase_reference
            )
            diagnostics: dict[str, Any] = {}
            if request.diagnostics or self.config.diagnostics == "full":
                diagnostics = {
                    "solver_residual_m": concatenate(
                        [generation.solver_residual_m for generation in generations]
                    ),
                    "fallback_mask": concatenate(
                        [generation.fallback_mask for generation in generations]
                    ),
                    "absolute_delay_s": rays.absolute_delay_s,
                    "segment_boundaries": boundaries,
                }
            outputs.append(CarrierGeometricRealization(
                carrier.carrier_frequency_hz, cir, rays, large_scale, diagnostics
            ))
        return tuple(outputs)

    def realize(self, request: GeometricCDLSamplingRequest) -> GeometricCDLRealization:
        """Generate a reusable, state-carrying multi-carrier realization."""

        self._validate_request(request)
        initial = self.initialize_anchors(request)
        carriers = self.evolve(initial, request)
        backend = request.sampling.execution.array_backend
        boundaries = self._segment_boundaries(request, carriers[0].large_scale.los_mask)
        updated_carriers = []
        final_paths = []
        for carrier_state, realization in zip(initial.carrier_states, carriers, strict=True):
            generations = self._carrier_generations(carrier_state, boundaries, request)
            final_generation = generations[-1]
            count = final_generation.cluster_ids.shape[-1]
            final_path = realization.rays.absolute_path_length_m[..., -count:, :, -1]
            if boundaries:
                phase_reference = realization.rays.absolute_path_length_m[
                    ..., -count:, :, boundaries[-1]
                ]
            elif request.prior_state is not None:
                phase_reference = initial.final_path_lengths_m[len(final_paths)]
            else:
                phase_reference = realization.rays.absolute_path_length_m[..., -count:, :, 0]
            final_phase = (
                final_generation.initial_phase_rad
                - 2.0 * pi * final_generation.carrier_frequency_hz / SPEED_OF_LIGHT_MPS
                * (final_path - phase_reference)
            )
            updated_carriers.append(replace(final_generation, initial_phase_rad=final_phase))
            final_paths.append(final_path)
        final_state = replace(
            initial,
            carrier_states=tuple(updated_carriers),
            final_path_lengths_m=tuple(final_paths),
            segment_ids=initial.segment_ids + len(boundaries),
            last_scenario_codes=request.scenarios.scenario_codes[..., -1],
            last_los_mask=carriers[0].large_scale.los_mask[..., -1],
            last_time_s=float(backend.to_scalar(request.trajectory.times_s[-1])),
            last_snapshot_index=(
                (request.prior_state.last_snapshot_index + request.trajectory.shape[-1])
                if request.prior_state is not None else request.trajectory.shape[-1] - 1
            ),
            metadata=self._metadata(request),
        )
        details: dict[str, Any] = {
            "carrier_frequencies_hz": self.config.carrier_frequencies_hz,
            "delay_reference": self.config.delay_reference,
            "segment_boundaries": boundaries,
        }
        return GeometricCDLRealization(carriers, final_state, self._metadata(request), details)


def _build(config: object) -> GeometricCDLModel:
    if not isinstance(config, GeometricCDLConfig):
        raise TypeError("geometric CDL builder requires GeometricCDLConfig")
    return GeometricCDLModel(config)


def register_geometric_cdl_model(registry: ModelRegistry) -> None:
    """Register ``wireless/geometric_cdl``."""

    registry.register("wireless", "geometric_cdl", parse_geometric_cdl_config, _build)


__all__ = ["GeometricCDLModel", "register_geometric_cdl_model"]
