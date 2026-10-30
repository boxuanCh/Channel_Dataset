"""3GPP path loss, LoS probability, and large-scale parameter sampling."""

from __future__ import annotations

from dataclasses import asdict
from math import log, log10, pi
from typing import Any

from channel_dataset.backends import RandomStream, get_backend
from channel_dataset.channels.wireless.math import SPEED_OF_LIGHT_MPS, matrix_sqrt_psd
from channel_dataset.channels.wireless.scenarios import GeometryScenario, ScenarioConfig
from channel_dataset.channels.wireless.topology import LinkGeometry
from channel_dataset.channels.wireless.types import LargeScaleParameters, TopologyBatch
from channel_dataset.config import ConfigEnvelope, load_resource_document
from channel_dataset.core import RealizationMetadata

_RESOURCE = "standards/wireless/tr_38_901/v16_1/scenario_large_scale.yaml"


def load_large_scale_parameters(name: str) -> dict[str, Any]:
    """Load versioned path-loss and LSP parameters for one scenario.

    Parameters
    ----------
    name : str
        Scenario identifier.

    Returns
    -------
    dict
        Independent copy of the scenario table plus shared correlation data.
    """

    document = load_resource_document(_RESOURCE)
    expected = {
        "schema_version", "standard", "standard_version", "tables",
        "parameter_order", "scenarios", "cross_correlation_matrix",
        "source_url", "transcription_date", "review_status",
    }
    if set(document) != expected or document["schema_version"] != 1:
        raise ValueError("invalid large-scale scenario resource")
    try:
        scenario = dict(document["scenarios"][name])
    except KeyError as error:
        raise ValueError(f"unsupported scenario: {name!r}") from error
    scenario["parameter_order"] = list(document["parameter_order"])
    scenario["cross_correlation_matrix"] = [
        list(row) for row in document["cross_correlation_matrix"]
    ]
    scenario["review_status"] = document["review_status"]
    return scenario


def _minimum(left: Any, right: float | Any, backend: Any) -> Any:
    return (left + right - backend.abs(left - right)) * 0.5


def _maximum(left: Any, right: float | Any, backend: Any) -> Any:
    return (left + right + backend.abs(left - right)) * 0.5


def _log10(value: Any, backend: Any) -> Any:
    return backend.log10(value)


class LargeScaleScenario(GeometryScenario):
    """3GPP scenario with probability, path loss, and LSP sampling.

    Parameters
    ----------
    config : ScenarioConfig
        Validated scenario and carrier-frequency configuration.
    """

    def __init__(self, config: ScenarioConfig) -> None:
        super().__init__(config)
        self.large_scale = load_large_scale_parameters(config.name)

    def los_probability(self, geometry: LinkGeometry) -> Any:
        """Evaluate the scenario LoS probability equation.

        Parameters
        ----------
        geometry : LinkGeometry
            Validated pairwise geometry.

        Returns
        -------
        backend array
            Probability on ``[batch, rx, tx]`` in ``[0, 1]``.
        """

        backend = get_backend(geometry.metadata.backend)
        d = geometry.distance_2d_m
        model = self.large_scale["los_probability_model"]
        if model == "rma":
            beyond = backend.exp(-(d - 10.0) / 1000.0)
            indicator = backend.astype(d <= 10.0, geometry.metadata.precision.replace("single", "float32").replace("double", "float64"))
            return indicator + (1.0 - indicator) * beyond
        if model == "umi":
            ratio = _minimum(18.0 / _maximum(d, 1.0e-12, backend), 1.0, backend)
            decay = backend.exp(-d / 36.0)
            return ratio * (1.0 - decay) + decay
        if model == "uma":
            ratio = _minimum(18.0 / _maximum(d, 1.0e-12, backend), 1.0, backend)
            decay = backend.exp(-d / 63.0)
            height = geometry.rx_height_m
            dtype = "float32" if geometry.metadata.precision == "single" else "float64"
            high = backend.astype(height > 13.0, dtype)
            correction = high * (_maximum(height - 13.0, 0.0, backend) / 10.0) ** 1.5
            return (
                ratio * (1.0 - decay) + decay
            ) * (1.0 + correction * 1.25 * (d / 100.0) ** 3 * backend.exp(-d / 150.0))
        if model == "inh_mixed":
            near = backend.exp(-(d - 1.2) / 4.7)
            far = 0.32 * backend.exp(-(d - 6.5) / 32.6)
            dtype = "float32" if geometry.metadata.precision == "single" else "float64"
            first = backend.astype(d <= 1.2, dtype)
            middle = backend.astype((d > 1.2) * (d < 6.5), dtype)
            return first + middle * near + (1.0 - first - middle) * far
        # InF-SL supported subset: clutter controls an exponential decay
        # distance while preserving the standard limiting behavior.
        density = float(self.config.clutter_density)
        clutter_size = (
            float(self.config.room_width_m) * float(self.config.room_length_m)
        ) ** 0.5
        decay_distance = max(1.0, clutter_size * (1.0 - density) / max(density, 1.0e-6))
        return backend.exp(-d / decay_distance)

    def path_loss_components_db(self, topology: TopologyBatch) -> tuple[Any, Any, Any]:
        """Return basic LoS, NLoS, and O2I loss components.

        Parameters
        ----------
        topology : TopologyBatch
            Validated endpoint geometry and receiver indoor state.

        Returns
        -------
        los_db, nlos_db, o2i_db : tuple of backend arrays
            Components on ``[batch, rx, tx]``.
        """

        geometry = self.validate_topology(topology)
        backend = get_backend(topology.metadata.backend)
        d2, d3 = geometry.distance_2d_m, geometry.distance_3d_m
        fc = self.config.carrier_frequency_hz / 1.0e9
        hbs = topology.tx_positions_m[:, None, :, 2]
        hut = topology.rx_positions_m[:, :, None, 2]
        model = self.large_scale["pathloss_model"]
        if model in {"umi", "uma"}:
            if model == "umi":
                intercept, slope1, correction = 32.4, 21.0, 9.5
                nlos_raw = 22.4 + 35.3 * _log10(d3, backend) + 21.3 * log10(fc) - 0.3 * (hut - 1.5)
            else:
                intercept, slope1, correction = 28.0, 22.0, 9.0
                nlos_raw = 13.54 + 39.08 * _log10(d3, backend) + 20.0 * log10(fc) - 0.6 * (hut - 1.5)
            breakpoint = 4.0 * (hbs - 1.0) * (hut - 1.0) * self.config.carrier_frequency_hz / SPEED_OF_LIGHT_MPS
            first = intercept + slope1 * _log10(d3, backend) + 20.0 * log10(fc)
            second = (
                intercept + 40.0 * _log10(d3, backend) + 20.0 * log10(fc)
                - correction * _log10(breakpoint * breakpoint + (hbs - hut) ** 2, backend)
            )
            dtype = "float32" if topology.metadata.precision == "single" else "float64"
            before = backend.astype(d2 <= breakpoint, dtype)
            los = before * first + (1.0 - before) * second
            nlos = _maximum(los, nlos_raw, backend)
        elif model == "rma":
            average_building_height = 5.0
            los_first = (
                20.0 * _log10(40.0 * pi * d3 * fc / 3.0, backend)
                + min(0.03 * average_building_height**1.72, 10.0) * _log10(d3, backend)
                - min(0.044 * average_building_height**1.72, 14.77)
                + 0.002 * log10(average_building_height) * d3
            )
            breakpoint_2d = (
                2.0 * pi * hbs * hut * self.config.carrier_frequency_hz
                / SPEED_OF_LIGHT_MPS
            )
            breakpoint_3d = backend.sqrt(breakpoint_2d**2 + (hbs - hut) ** 2)
            first_at_breakpoint = (
                20.0 * _log10(40.0 * pi * breakpoint_3d * fc / 3.0, backend)
                + min(0.03 * average_building_height**1.72, 10.0)
                * _log10(breakpoint_3d, backend)
                - min(0.044 * average_building_height**1.72, 14.77)
                + 0.002 * log10(average_building_height) * breakpoint_3d
            )
            los_second = first_at_breakpoint + 40.0 * _log10(
                d3 / breakpoint_3d, backend
            )
            dtype = "float32" if topology.metadata.precision == "single" else "float64"
            before = backend.astype(d2 <= breakpoint_2d, dtype)
            los = before * los_first + (1.0 - before) * los_second
            nlos_raw = (
                161.04 - 7.1 * log10(20.0)
                + 7.5 * log10(average_building_height)
                - (24.37 - 3.7 * (average_building_height / hbs) ** 2) * _log10(hbs, backend)
                + (43.42 - 3.1 * _log10(hbs, backend)) * (_log10(d3, backend) - 3.0)
                + 20.0 * log10(fc)
                - (3.2 * (_log10(11.75 * hut, backend)) ** 2 - 4.97)
            )
            nlos = _maximum(los, nlos_raw, backend)
        elif model == "inh":
            los = 32.4 + 17.3 * _log10(d3, backend) + 20.0 * log10(fc)
            nlos = _maximum(los, 17.3 + 38.3 * _log10(d3, backend) + 24.9 * log10(fc), backend)
        else:
            los = 31.84 + 21.5 * _log10(d3, backend) + 19.0 * log10(fc)
            nlos = _maximum(los, 33.0 + 25.5 * _log10(d3, backend) + 20.0 * log10(fc), backend)
        indoor = backend.astype(
            topology.rx_indoor,
            "float32" if topology.metadata.precision == "single" else "float64",
        )[:, :, None]
        o2i = indoor * float(self.large_scale["o2i_loss_db"])
        return los, nlos, backend.broadcast_to(o2i, geometry.shape)

    def path_loss_db(self, topology: TopologyBatch, los_mask: Any) -> Any:
        """Evaluate basic path loss including configured O2I loss.

        Parameters
        ----------
        topology : TopologyBatch
            Endpoint geometry and indoor state.
        los_mask : backend array
            Boolean LoS state on ``[batch, rx, tx]``.

        Returns
        -------
        backend array
            Path loss in decibels.
        """

        backend = get_backend(topology.metadata.backend)
        if backend.shape(los_mask) != topology.shape or not backend.is_bool(los_mask):
            raise ValueError("los_mask must be Boolean with topology link shape")
        los, nlos, o2i = self.path_loss_components_db(topology)
        selection = backend.astype(
            los_mask, "float32" if topology.metadata.precision == "single" else "float64"
        )
        return selection * los + (1.0 - selection) * nlos + o2i

    def sample_large_scale(
        self,
        topology: TopologyBatch,
        random_stream: RandomStream[Any],
        *,
        path_loss_enabled: bool = True,
        shadow_fading_enabled: bool = True,
    ) -> LargeScaleParameters:
        """Sample correlated large-scale parameters for a topology.

        Parameters
        ----------
        topology : TopologyBatch
            Validated endpoint state.
        random_stream : RandomStream
            Injected stream matching the topology backend/device.
        path_loss_enabled, shadow_fading_enabled : bool, default=True
            Independent output switches. All random variables are drawn in the
            same order regardless of either switch.

        Returns
        -------
        LargeScaleParameters
            Correlated lognormal spreads, shadowing, K-factor, and path loss.
        """

        geometry = self.validate_topology(topology)
        backend = get_backend(topology.metadata.backend)
        if random_stream.backend_name != topology.metadata.backend or random_stream.device != topology.metadata.device:
            raise ValueError("random stream backend/device must match topology")
        shape = geometry.shape
        probability = self.los_probability(geometry)
        uniform = random_stream.fork("wireless.lsp.los_state").uniform(
            shape, dtype="float32" if topology.metadata.precision == "single" else "float64"
        )
        sampled_los = uniform < probability
        los_mask = topology.los_mask if topology.los_mask is not None else sampled_los
        normal = random_stream.fork("wireless.lsp.gaussian").normal(
            (*shape, 7), dtype="float32" if topology.metadata.precision == "single" else "float64"
        )
        matrix = backend.array(
            self.large_scale["cross_correlation_matrix"],
            dtype="float32" if topology.metadata.precision == "single" else "float64",
            device=topology.metadata.device,
        )
        cross = backend.einsum("ij,...j->...i", matrix_sqrt_psd(matrix), normal)
        # Exponential spatial filtering across receivers, independently for
        # every batch, transmitter, and LSP dimension.
        rx_xy = topology.rx_positions_m[..., :2]
        if shape[1] == 1:
            filtered = cross
        else:
            filtered_parameters = []
            for parameter in range(7):
                correlation_distance = float(
                    self.large_scale["spatial_correlation_distances_m"][parameter]
                )
                difference = rx_xy[:, :, None, :] - rx_xy[:, None, :, :]
                distance = backend.sqrt(backend.sum(difference * difference, axis=-1))
                correlation = backend.exp(-distance / correlation_distance)
                root = matrix_sqrt_psd(correlation)
                filtered_parameters.append(
                    backend.einsum("bij,bjt->bit", root, cross[..., parameter])
                )
            filtered = backend.stack(tuple(filtered_parameters), axis=-1)
        dtype = "float32" if topology.metadata.precision == "single" else "float64"
        selection = backend.astype(los_mask, dtype)[..., None]
        los_stats, nlos_stats = self.large_scale["los"], self.large_scale["nlos"]
        los_mean = backend.array(los_stats["means"], dtype=dtype, device=topology.metadata.device)
        nlos_mean = backend.array(nlos_stats["means"], dtype=dtype, device=topology.metadata.device)
        los_std = backend.array(los_stats["stddevs"], dtype=dtype, device=topology.metadata.device)
        nlos_std = backend.array(nlos_stats["stddevs"], dtype=dtype, device=topology.metadata.device)
        values = selection * (los_mean + los_std * filtered) + (1.0 - selection) * (nlos_mean + nlos_std * filtered)
        ln10 = log(10.0)
        delay = backend.exp(values[..., 0] * ln10)
        asd = backend.exp(values[..., 1] * ln10) * pi / 180.0
        asa = backend.exp(values[..., 2] * ln10) * pi / 180.0
        zsa = backend.exp(values[..., 5] * ln10) * pi / 180.0
        zsd = backend.exp(values[..., 6] * ln10) * pi / 180.0
        shadow = values[..., 3]
        if not shadow_fading_enabled:
            shadow = shadow * 0.0
        path_loss = self.path_loss_db(topology, los_mask)
        if not path_loss_enabled:
            path_loss = path_loss * 0.0
        mask = topology.link_mask
        if mask is not None:
            valid = backend.astype(mask, dtype)
            delay, asa, asd, zsa, zsd = (value * valid for value in (delay, asa, asd, zsa, zsd))
            shadow, path_loss = shadow * valid, path_loss * valid
        digest = ConfigEnvelope(1, "wireless", f"{self.name.lower()}_large_scale", asdict(self.config)).model_digest
        metadata = RealizationMetadata(
            "wireless", f"{self.name.lower()}_large_scale", 1,
            topology.metadata.backend, topology.metadata.precision, topology.metadata.device,
            random_stream.stream_id, digest, asdict(self.config), topology.metadata.software_version,
            topology.metadata.request_id,
            random_stream.root_seed,
            "1",
            "16.1.0",
        )
        return LargeScaleParameters(
            delay, asa, asd, zsa, zsd, shadow, values[..., 4], path_loss,
            metadata, topology.link_mask,
        )


__all__ = ["LargeScaleScenario", "load_large_scale_parameters"]
