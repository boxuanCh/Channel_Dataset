"""3GPP TR 38.811 NTN, satellite, and HAPS channel models."""

from channel_dataset.channels.wireless.tr38811.antennas import (
    bessel_j1,
    circular_aperture_gain_db,
    circular_aperture_power_gain,
)
from channel_dataset.channels.wireless.tr38811.atmosphere import (
    NTNAtmosphereProvider,
    should_compute_gaseous_attenuation,
    slant_gaseous_attenuation_db,
)
from channel_dataset.channels.wireless.tr38811.cdl import (
    TR38811CDLChannel,
    register_cdl_model,
)
from channel_dataset.channels.wireless.tr38811.config import (
    NTNAntennaConfig,
    NTNAtmosphereConfig,
    NTNScintillationConfig,
    TR38811CDLConfig,
    TR38811DeploymentConfig,
    TR38811FlatFadingConfig,
    TR38811SystemConfig,
    TR38811TDLConfig,
)
from channel_dataset.channels.wireless.tr38811.dataset import (
    load_ntn_realization,
    save_ntn_realization,
)
from channel_dataset.channels.wireless.tr38811.flat_fading import (
    TR38811FlatFadingChannel,
    register_flat_model,
)
from channel_dataset.channels.wireless.tr38811.geometry import (
    EARTH_RADIUS_M,
    compute_ntn_geometry,
    doppler_phase,
    ecef_to_geodetic,
    geodetic_to_ecef,
)
from channel_dataset.channels.wireless.tr38811.large_scale import (
    free_space_path_loss_db,
    los_probability,
    sample_large_scale,
    sample_los,
)
from channel_dataset.channels.wireless.tr38811.polarization import faraday_rotation_matrix
from channel_dataset.channels.wireless.tr38811.profiles import (
    load_cdl_profile,
    load_resource,
    load_tdl_profile,
)
from channel_dataset.channels.wireless.tr38811.scintillation import (
    sample_scintillation_db,
    scale_ionospheric_phase_std,
    scale_ionospheric_s4,
)
from channel_dataset.channels.wireless.tr38811.system import (
    TR38811SystemChannel,
    register_system_model,
)
from channel_dataset.channels.wireless.tr38811.tdl import (
    TR38811TDLChannel,
    register_tdl_model,
)
from channel_dataset.channels.wireless.tr38811.trajectories import (
    circular_equatorial_orbit,
    constant_velocity_trajectory,
    geostationary_state,
)
from channel_dataset.channels.wireless.tr38811.transforms import ntn_to_ofdm
from channel_dataset.channels.wireless.tr38811.types import (
    NTNChannelRealization,
    NTNChannelSamplingRequest,
    NTNLargeScaleComponents,
    NTNLinkGeometry,
    NTNRealizationDiagnostics,
    NTNTopologyBatch,
)


def register_tr38811_models(registry) -> None:
    """Register every implemented TR 38.811 model."""

    register_tdl_model(registry)
    register_cdl_model(registry)
    register_flat_model(registry)
    register_system_model(registry)


__all__ = [
    "EARTH_RADIUS_M", "NTNAntennaConfig", "NTNAtmosphereConfig",
    "NTNAtmosphereProvider",
    "NTNChannelRealization", "NTNChannelSamplingRequest", "NTNLargeScaleComponents",
    "NTNLinkGeometry", "NTNRealizationDiagnostics", "NTNScintillationConfig",
    "NTNTopologyBatch", "TR38811CDLChannel", "TR38811CDLConfig",
    "TR38811DeploymentConfig",
    "TR38811FlatFadingChannel", "TR38811FlatFadingConfig", "TR38811SystemChannel",
    "TR38811SystemConfig", "TR38811TDLChannel",
    "TR38811TDLConfig", "bessel_j1", "circular_aperture_gain_db",
    "circular_aperture_power_gain", "circular_equatorial_orbit",
    "compute_ntn_geometry", "constant_velocity_trajectory", "doppler_phase",
    "ecef_to_geodetic", "faraday_rotation_matrix", "free_space_path_loss_db",
    "geodetic_to_ecef", "geostationary_state", "load_cdl_profile",
    "load_ntn_realization", "load_resource", "load_tdl_profile",
    "los_probability", "ntn_to_ofdm", "register_tr38811_models",
    "sample_large_scale", "sample_los", "sample_scintillation_db",
    "save_ntn_realization",
    "scale_ionospheric_phase_std", "scale_ionospheric_s4",
    "should_compute_gaseous_attenuation", "slant_gaseous_attenuation_db",
]
