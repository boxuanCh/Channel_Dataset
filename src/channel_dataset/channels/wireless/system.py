"""Antenna-aware system-level coefficient and channel assembly."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any

from channel_dataset.backends import RandomStream, get_backend
from channel_dataset.channels.wireless.antennas import (
    AntennaLayout,
    Orientation,
    spherical_basis,
)
from channel_dataset.channels.wireless.blockage import (
    NonSelfBlockingModelA,
    ScreenBlockingModelB,
    SelfBlockingModel,
    apply_blockage,
)
from channel_dataset.channels.wireless.math import SPEED_OF_LIGHT_MPS
from channel_dataset.channels.wireless.rays import ScenarioRayGenerator
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.scenarios import ScenarioConfig, create_scenario
from channel_dataset.channels.wireless.types import (
    CIRBatch,
    LargeScaleParameters,
    RayBatch,
    TopologyBatch,
)
from channel_dataset.core import ChannelModel, ExecutionContext


@dataclass(frozen=True, slots=True)
class CoefficientDiagnostics:
    """Optional intermediate arrays from coefficient generation.

    Parameters
    ----------
    polarization : backend array
        Polarization-coupled response before array and Doppler phases.
    array_phase : backend array
        Phase response caused by antenna element positions.
    doppler_hz : backend array
        Per-ray Doppler frequencies.
    ray_coefficients : backend array
        Complete per-ray coefficients before cluster combination.
    """

    polarization: Any
    array_phase: Any
    doppler_hz: Any
    ray_coefficients: Any


@dataclass(frozen=True, slots=True)
class SystemLevelResult:
    """System-level realization and optional sampled diagnostics.

    Parameters
    ----------
    cir : CIRBatch
        Antenna-aware channel impulse response including large-scale loss.
    large_scale : LargeScaleParameters
        Sampled LSPs used by the realization.
    rays : RayBatch
        Small-scale ray state used by the coefficient engine.
    coefficients : CoefficientDiagnostics, optional
        Intermediate coefficient terms when requested.
    """

    cir: CIRBatch
    large_scale: LargeScaleParameters
    rays: RayBatch
    coefficients: CoefficientDiagnostics | None = None


def evaluate_array_field(
    layout: AntennaLayout,
    zenith_rad: Any,
    azimuth_rad: Any,
    execution: ExecutionContext,
    *,
    orientation: Orientation | None = None,
) -> tuple[Any, Any]:
    """Evaluate every element field at every supplied ray direction.

    Returns
    -------
    theta, phi : tuple of backend arrays
        Polarized field components with an appended antenna axis.
    """

    if orientation is None:
        orientation = Orientation()
    theta_fields, phi_fields = [], []
    for index in range(layout.array.num_elements):
        field = layout.array.field_pattern(
            index, zenith_rad, azimuth_rad, execution, orientation=orientation
        )
        theta_fields.append(field.theta)
        phi_fields.append(field.phi)
    backend = execution.array_backend
    return backend.stack(tuple(theta_fields), axis=-1), backend.stack(
        tuple(phi_fields), axis=-1
    )


def polarization_coupling(
    rx_field: tuple[Any, Any],
    tx_field: tuple[Any, Any],
    xpr_linear: Any,
    phases_rad: Any,
    execution: ExecutionContext,
) -> Any:
    """Apply a four-phase dual-polarization coupling matrix.

    Parameters
    ----------
    rx_field, tx_field : tuple of backend arrays
        Theta and phi field components with a final antenna axis.
    xpr_linear : backend array
        Positive per-ray cross-polarization power ratio.
    phases_rad : backend array
        Four independent phases on the final axis.
    execution : ExecutionContext
        Backend, precision, and device controls.

    Returns
    -------
    backend array
        Coupled response with final ``[rx_antenna, tx_antenna]`` axes.
    """

    backend = execution.array_backend
    if backend.shape(phases_rad)[-1] != 4:
        raise ValueError("phases_rad must have four polarization phases")
    phase = backend.exp(1j * phases_rad)
    cross = 1.0 / backend.sqrt(xpr_linear)
    p00, p01 = phase[..., 0], cross * phase[..., 1]
    p10, p11 = cross * phase[..., 2], phase[..., 3]
    rx_theta, rx_phi = rx_field
    tx_theta, tx_phi = tx_field
    return (
        rx_theta[..., :, None]
        * (
            p00[..., None, None] * tx_theta[..., None, :]
            + p01[..., None, None] * tx_phi[..., None, :]
        )
        + rx_phi[..., :, None]
        * (
            p10[..., None, None] * tx_theta[..., None, :]
            + p11[..., None, None] * tx_phi[..., None, :]
        )
    )


def array_phase_offsets(
    rx_direction: Any,
    tx_direction: Any,
    rx_layout: AntennaLayout,
    tx_layout: AntennaLayout,
    execution: ExecutionContext,
) -> Any:
    """Return per-ray phase offsets for receiver/transmitter element pairs."""

    backend = execution.array_backend
    rx_positions = rx_layout.array.positions_wavelengths(execution)
    tx_positions = tx_layout.array.positions_wavelengths(execution)
    spatial = (
        backend.einsum("...j,aj->...a", rx_direction, rx_positions)[..., :, None]
        + backend.einsum("...j,aj->...a", tx_direction, tx_positions)[..., None, :]
    )
    return backend.exp(1j * 2.0 * pi * spatial)


def doppler_phase_evolution(
    rx_direction: Any,
    tx_direction: Any,
    rx_velocity_mps: Any,
    tx_velocity_mps: Any,
    carrier_frequency_hz: float,
    request: ChannelSamplingRequest,
) -> tuple[Any, Any]:
    """Return ray Doppler frequencies and their sampled complex evolution."""

    backend = request.execution.array_backend
    doppler = (
        backend.einsum("...j,j->...", rx_direction, rx_velocity_mps)
        + backend.einsum("...j,j->...", tx_direction, tx_velocity_mps)
    ) * carrier_frequency_hz / SPEED_OF_LIGHT_MPS
    time = backend.arange(
        request.num_time_samples,
        dtype=request.execution.float_dtype,
        device=request.execution.device,
    ) / request.sampling_frequency_hz
    return doppler, backend.exp(1j * 2.0 * pi * doppler[..., None] * time)


class SystemLevelCoefficientGenerator:
    """Generate and combine antenna, polarization, array, and Doppler terms."""

    def __init__(
        self,
        carrier_frequency_hz: float,
        tx_layout: AntennaLayout,
        rx_layout: AntennaLayout,
    ) -> None:
        if carrier_frequency_hz <= 0.0:
            raise ValueError("carrier_frequency_hz must be positive")
        self.carrier_frequency_hz = float(carrier_frequency_hz)
        self.tx_layout = tx_layout
        self.rx_layout = rx_layout

    def _link_coefficients(
        self,
        rays: RayBatch,
        topology: TopologyBatch,
        request: ChannelSamplingRequest,
        random_stream: RandomStream[Any],
        batch: int,
        rx: int,
        tx: int,
    ) -> tuple[Any, Any, Any, Any]:
        backend = request.execution.array_backend
        execution = request.execution
        link = rays.select_link(batch, rx, tx)
        aoa = link.azimuth_arrival_rad[0, 0, 0]
        zoa = link.zenith_arrival_rad[0, 0, 0]
        aod = link.azimuth_departure_rad[0, 0, 0]
        zod = link.zenith_departure_rad[0, 0, 0]
        rx_direction, _, _ = spherical_basis(zoa, aoa, execution)
        tx_direction, _, _ = spherical_basis(zod, aod, execution)
        rx_orientation = Orientation(
            *(float(backend.to_scalar(topology.rx_orientations_rad[batch, rx, i])) for i in range(3))
        )
        tx_orientation = Orientation(
            *(float(backend.to_scalar(topology.tx_orientations_rad[batch, tx, i])) for i in range(3))
        )
        rx_field = evaluate_array_field(
            self.rx_layout, zoa, aoa, execution, orientation=rx_orientation
        )
        tx_field = evaluate_array_field(
            self.tx_layout, zod, aod, execution, orientation=tx_orientation
        )
        phase_shape = (*link.shape[3:], 4)
        phases = random_stream.fork(
            f"wireless.coefficients.polarization.{batch}.{rx}.{tx}"
        ).uniform(phase_shape, -pi, pi, dtype=execution.float_dtype)
        polarization = polarization_coupling(
            rx_field, tx_field, link.xpr_linear[0, 0, 0], phases, execution
        )
        array_phase = array_phase_offsets(
            rx_direction,
            tx_direction,
            self.rx_layout,
            self.tx_layout,
            execution,
        )
        doppler, evolution = doppler_phase_evolution(
            rx_direction,
            tx_direction,
            topology.rx_velocities_mps[batch, rx],
            topology.tx_velocities_mps[batch, tx],
            self.carrier_frequency_hz,
            request,
        )
        ray_count = link.shape[-1]
        scale = backend.sqrt(link.powers_linear[0, 0, 0] / ray_count)[..., None]
        initial = backend.exp(1j * link.initial_phase_rad[0, 0, 0])
        coefficient = (
            scale[..., None, None, None]
            * polarization[..., None]
            * array_phase[..., None]
            * initial[..., None, None, None]
            * evolution[..., None, None, :]
        )
        if link.ray_mask is not None:
            coefficient = coefficient * backend.astype(
                link.ray_mask[0, 0, 0], execution.float_dtype
            )[..., None, None, None]
        if link.blockage_loss_db is not None:
            blockage = backend.exp(
                -link.blockage_loss_db[0, 0, 0] * (0.05 * 2.302585092994046)
            )
            coefficient = coefficient * blockage[..., None, None, None]
        return coefficient, polarization, array_phase, doppler

    def generate(
        self,
        rays: RayBatch,
        topology: TopologyBatch,
        request: ChannelSamplingRequest,
        random_stream: RandomStream[Any],
        *,
        diagnostics: bool = False,
    ) -> tuple[Any, CoefficientDiagnostics | None]:
        """Generate per-ray MIMO coefficients for all topology links.

        Returns
        -------
        coefficients, diagnostics : tuple
            Coefficients use axes ``[batch, rx, tx, cluster, ray,
            rx_antenna, tx_antenna, time]``. Diagnostics are returned only
            when requested and do not alter stochastic draws.
        """

        if rays.shape[:3] != topology.shape or request.batch_size != topology.shape[0]:
            raise ValueError("request, topology, and ray batch dimensions must match")
        backend = request.execution.array_backend
        batch_values, polarization_values, phase_values, doppler_values = [], [], [], []
        for batch in range(topology.shape[0]):
            receiver_values, receiver_pol, receiver_phase, receiver_doppler = [], [], [], []
            for rx in range(topology.shape[1]):
                transmitter_values, transmitter_pol, transmitter_phase, transmitter_doppler = [], [], [], []
                for tx in range(topology.shape[2]):
                    value, polarization, phase, doppler = self._link_coefficients(
                        rays, topology, request, random_stream, batch, rx, tx
                    )
                    transmitter_values.append(value)
                    transmitter_pol.append(polarization)
                    transmitter_phase.append(phase)
                    transmitter_doppler.append(doppler)
                receiver_values.append(backend.stack(tuple(transmitter_values)))
                receiver_pol.append(backend.stack(tuple(transmitter_pol)))
                receiver_phase.append(backend.stack(tuple(transmitter_phase)))
                receiver_doppler.append(backend.stack(tuple(transmitter_doppler)))
            batch_values.append(backend.stack(tuple(receiver_values)))
            polarization_values.append(backend.stack(tuple(receiver_pol)))
            phase_values.append(backend.stack(tuple(receiver_phase)))
            doppler_values.append(backend.stack(tuple(receiver_doppler)))
        coefficients = backend.stack(tuple(batch_values))
        details = None
        if diagnostics:
            details = CoefficientDiagnostics(
                backend.stack(tuple(polarization_values)),
                backend.stack(tuple(phase_values)),
                backend.stack(tuple(doppler_values)),
                coefficients,
            )
        return coefficients, details

    def combine(
        self,
        coefficients: Any,
        rays: RayBatch,
        large_scale: LargeScaleParameters,
    ) -> CIRBatch:
        """Combine rays, including three-way strongest-cluster splitting."""

        backend = get_backend(rays.metadata.backend)
        b, nr, nt, clusters, *_ = backend.shape(coefficients)
        outputs, delays, masks = [], [], []
        groups = ((0, 1, 2, 3, 4, 5, 6, 7, 18, 19), (8, 9, 10, 11, 16, 17), (12, 13, 14, 15))
        for batch in range(b):
            rx_outputs, rx_delays, rx_masks = [], [], []
            for rx in range(nr):
                tx_outputs, tx_delays, tx_masks = [], [], []
                for tx in range(nt):
                    strongest = [
                        int(backend.to_scalar(rays.strongest_cluster_indices[batch, rx, tx, i]))
                        for i in range(2)
                    ]
                    link_paths, link_delays, link_masks = [], [], []
                    for cluster in range(clusters):
                        values = coefficients[batch, rx, tx, cluster]
                        if cluster in strongest:
                            indices = backend.array(groups[0], dtype="int64", device=rays.metadata.device)
                            value = backend.sum(values[indices], axis=0)
                        else:
                            value = backend.sum(values, axis=0)
                        link_paths.append(value)
                        link_delays.append(rays.delays_s[batch, rx, tx, cluster])
                        link_masks.append(rays.cluster_mask[batch, rx, tx, cluster])
                    for cluster in strongest:
                        for group, offset in zip(groups[1:], (1.28, 2.56), strict=True):
                            indices = backend.array(group, dtype="int64", device=rays.metadata.device)
                            link_paths.append(backend.sum(coefficients[batch, rx, tx, cluster, indices], axis=0))
                            link_delays.append(
                                rays.delays_s[batch, rx, tx, cluster]
                                + offset * large_scale.delay_spread_s[batch, rx, tx]
                            )
                            link_masks.append(rays.cluster_mask[batch, rx, tx, cluster])
                    tx_outputs.append(backend.stack(tuple(link_paths), axis=2))
                    tx_delays.append(backend.stack(tuple(link_delays)))
                    tx_masks.append(backend.stack(tuple(link_masks)))
                rx_outputs.append(backend.stack(tuple(tx_outputs), axis=1))
                rx_delays.append(backend.stack(tuple(tx_delays)))
                rx_masks.append(backend.stack(tuple(tx_masks)))
            outputs.append(backend.stack(tuple(rx_outputs), axis=0))
            delays.append(backend.stack(tuple(rx_delays), axis=0))
            masks.append(backend.stack(tuple(rx_masks), axis=0))
        # Stacked path values are [b, rx, rx_ant, tx, tx_ant, path, time].
        return CIRBatch(
            backend.stack(tuple(outputs), axis=0),
            backend.stack(tuple(delays), axis=0),
            rays.metadata,
            backend.stack(tuple(masks), axis=0),
        )


def apply_large_scale_loss(cir: CIRBatch, large_scale: LargeScaleParameters) -> CIRBatch:
    """Apply path loss and shadow fading to CIR voltage coefficients."""

    if cir.delays_s.shape[:3] != large_scale.shape:
        raise ValueError("CIR and large-scale link dimensions must match")
    backend = get_backend(cir.metadata.backend)
    attenuation = backend.exp(
        -(large_scale.path_loss_db + large_scale.shadow_fading_db)
        * (0.05 * 2.302585092994046)
    )
    coefficients = cir.coefficients * attenuation[:, :, None, :, None, None, None]
    return CIRBatch(coefficients, cir.delays_s, cir.metadata, cir.path_mask)


class SystemLevelWirelessChannel(ChannelModel[ChannelSamplingRequest, CIRBatch]):
    """End-to-end snapshot channel for one fixed topology and antenna pair."""

    def __init__(
        self,
        scenario_config: ScenarioConfig,
        topology: TopologyBatch,
        tx_layout: AntennaLayout,
        rx_layout: AntennaLayout,
        blockage_models: tuple[
            SelfBlockingModel | NonSelfBlockingModelA | ScreenBlockingModelB, ...
        ] = (),
    ) -> None:
        self.scenario = create_scenario(scenario_config)
        self.topology = topology
        self.ray_generator = ScenarioRayGenerator(scenario_config.name)
        self.blockage_models = blockage_models
        self.coefficient_generator = SystemLevelCoefficientGenerator(
            scenario_config.carrier_frequency_hz, tx_layout, rx_layout
        )

    @property
    def family(self) -> str:
        """str: Registry family identifier."""

        return "wireless"

    @property
    def name(self) -> str:
        """str: Model identifier including the configured scenario."""

        return f"system_level_{self.scenario.name.lower()}"

    def realize_with_diagnostics(
        self, request: ChannelSamplingRequest, *, diagnostics: bool = True
    ) -> SystemLevelResult:
        """Sample a CIR and optionally return every intermediate realization."""

        if request.execution.backend != self.topology.metadata.backend:
            raise ValueError("request backend must match topology")
        root = request.random_stream
        large_scale = self.scenario.sample_large_scale(
            self.topology, root.fork("wireless.system.large_scale")
        )
        rays = self.ray_generator.generate(
            self.topology,
            large_scale,
            root.fork("wireless.system.rays"),
        )
        if self.blockage_models:
            backend = request.execution.array_backend
            loss = backend.zeros(
                rays.shape,
                dtype=request.execution.float_dtype,
                device=request.execution.device,
            )
            for index, model in enumerate(self.blockage_models):
                if isinstance(model, NonSelfBlockingModelA):
                    component = model.loss_db(
                        rays,
                        self.topology,
                        root.fork(f"wireless.system.blockage.{index}"),
                    )
                else:
                    component = model.loss_db(rays, self.topology)
                loss = loss + component
            rays = apply_blockage(rays, loss)
        ray_coefficients, details = self.coefficient_generator.generate(
            rays,
            self.topology,
            request,
            root.fork("wireless.system.coefficients"),
            diagnostics=diagnostics,
        )
        cir = self.coefficient_generator.combine(ray_coefficients, rays, large_scale)
        cir = self._add_los(cir, rays, large_scale, request)
        cir = apply_large_scale_loss(cir, large_scale)
        return SystemLevelResult(cir, large_scale, rays, details)

    def _add_los(
        self,
        cir: CIRBatch,
        rays: RayBatch,
        large_scale: LargeScaleParameters,
        request: ChannelSamplingRequest,
    ) -> CIRBatch:
        backend = request.execution.array_backend
        los_mask = self.topology.los_mask
        if los_mask is None:
            los_mask = large_scale.k_factor_db != 0.0
        selection = backend.astype(los_mask, request.execution.float_dtype)
        k_linear = backend.exp(large_scale.k_factor_db * 0.1 * 2.302585092994046)
        diffuse_scale = backend.sqrt(1.0 / (1.0 + selection * k_linear))
        values = cir.coefficients * diffuse_scale[:, :, None, :, None, None, None]
        # The deterministic component uses the first ray's coherent array
        # response and geometric carrier phase, normalized by its magnitude.
        first = values[..., 0, :]
        magnitude = backend.sqrt(first.real * first.real + first.imag * first.imag + 1.0e-30)
        unit = first / magnitude
        # Keep the reused response in quadrature with the diffuse first path.
        # ``unit`` is derived from that path, so adding it in phase would create
        # a strictly positive cross term and bias the Ricean link power above
        # unity.  The quadrature rotation preserves the configured K-factor
        # split without changing the response magnitude or Doppler evolution.
        direct = (
            1j
            * unit
            * backend.sqrt(k_linear / (1.0 + k_linear))[:, :, None, :, None, None]
        )
        direct = direct * selection[:, :, None, :, None, None]
        first = first + direct
        values = backend.concatenate((first[..., None, :], values[..., 1:, :]), axis=5)
        return CIRBatch(values, cir.delays_s, cir.metadata, cir.path_mask)

    def realize(self, request: ChannelSamplingRequest) -> CIRBatch:
        """Sample an antenna-aware, loss-scaled reusable CIR."""

        return self.realize_with_diagnostics(request, diagnostics=False).cir


__all__ = [
    "CoefficientDiagnostics",
    "SystemLevelCoefficientGenerator",
    "SystemLevelResult",
    "SystemLevelWirelessChannel",
    "apply_large_scale_loss",
    "array_phase_offsets",
    "doppler_phase_evolution",
    "evaluate_array_field",
    "polarization_coupling",
]
