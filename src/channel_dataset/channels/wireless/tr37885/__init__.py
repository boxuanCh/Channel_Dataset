"""3GPP TR 37.885 urban Vehicle-to-Everything channel models."""

from channel_dataset.channels.wireless.tr37885.antennas import (
    V2XVehicleAntennaProfile,
    load_vehicle_antenna_profile,
)
from channel_dataset.channels.wireless.tr37885.cdl import (
    TR37885UrbanCDLChannel,
    register_urban_cdl_model,
)
from channel_dataset.channels.wireless.tr37885.config import (
    TR37885TrafficConfig,
    TR37885UrbanCDLConfig,
    TR37885UrbanGridConfig,
    TR37885UrbanSystemConfig,
)
from channel_dataset.channels.wireless.tr37885.dataset import (
    load_v2x_realization,
    save_v2x_realization,
)
from channel_dataset.channels.wireless.tr37885.grid import (
    TR37885UrbanGrid,
    V2XBuildingBlock,
    V2XIntersection,
    V2XLane,
    V2XStreet,
)
from channel_dataset.channels.wireless.tr37885.large_scale import (
    urban_path_loss_db,
    vehicle_blockage_parameters_db,
)
from channel_dataset.channels.wireless.tr37885.link_state import (
    classify_urban_links,
    urban_los_probability,
)
from channel_dataset.channels.wireless.tr37885.profiles import (
    V2XCDLProfile,
    load_cdl_profile,
    load_urban_parameters,
)
from channel_dataset.channels.wireless.tr37885.system import (
    TR37885UrbanSystemChannel,
    register_urban_system_model,
)
from channel_dataset.channels.wireless.tr37885.topology import topology_at_snapshot
from channel_dataset.channels.wireless.tr37885.traffic import TR37885UrbanGridSimulator
from channel_dataset.channels.wireless.tr37885.types import (
    INVALID,
    LOS,
    NLOS,
    NLOSV,
    V2XChannelRealization,
    V2XChannelSamplingRequest,
    V2XChannelState,
    V2XLargeScaleComponents,
    V2XLinkConditionBatch,
    V2XRealizationDiagnostics,
    V2XSceneSamplingRequest,
    V2XSceneTrajectory,
)


def register_tr37885_models(registry) -> None:
    """Register all implemented TR 37.885 urban channel models."""

    register_urban_system_model(registry)
    register_urban_cdl_model(registry)


__all__ = [
    "INVALID",
    "LOS",
    "NLOS",
    "NLOSV",
    "TR37885TrafficConfig",
    "TR37885UrbanCDLChannel",
    "TR37885UrbanCDLConfig",
    "TR37885UrbanGrid",
    "TR37885UrbanGridConfig",
    "TR37885UrbanGridSimulator",
    "TR37885UrbanSystemChannel",
    "TR37885UrbanSystemConfig",
    "V2XBuildingBlock",
    "V2XCDLProfile",
    "V2XChannelRealization",
    "V2XChannelSamplingRequest",
    "V2XChannelState",
    "V2XIntersection",
    "V2XLane",
    "V2XLargeScaleComponents",
    "V2XLinkConditionBatch",
    "V2XRealizationDiagnostics",
    "V2XSceneSamplingRequest",
    "V2XSceneTrajectory",
    "V2XStreet",
    "V2XVehicleAntennaProfile",
    "classify_urban_links",
    "load_cdl_profile",
    "load_urban_parameters",
    "load_v2x_realization",
    "load_vehicle_antenna_profile",
    "register_tr37885_models",
    "save_v2x_realization",
    "topology_at_snapshot",
    "urban_los_probability",
    "urban_path_loss_db",
    "vehicle_blockage_parameters_db",
]
