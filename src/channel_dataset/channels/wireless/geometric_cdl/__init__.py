"""Geometrically consistent, stateful CDL evolution."""

from .config import GeometricCDLConfig, parse_geometric_cdl_config
from .dataset import load_geometric_cdl_realization, save_geometric_cdl_realization
from .fields import SpatialFieldBank
from .geometry import (
    GROUND_COMPONENT,
    LOS_COMPONENT,
    NLOS_COMPONENT,
    construct_scatterers,
    drift_paths,
)
from .model import GeometricCDLModel, register_geometric_cdl_model
from .resources import load_geometric_resource
from .types import (
    CarrierClusterState,
    CarrierGeometricRealization,
    DynamicCIRBatch,
    GeometricCDLRealization,
    GeometricCDLSamplingRequest,
    GeometricCDLState,
    LargeScaleTrajectoryBatch,
    RayTrajectoryBatch,
    ScenarioTimelineBatch,
    TopologyTrajectoryBatch,
)

__all__ = [
    "GROUND_COMPONENT",
    "LOS_COMPONENT",
    "NLOS_COMPONENT",
    "CarrierClusterState",
    "CarrierGeometricRealization",
    "DynamicCIRBatch",
    "GeometricCDLConfig",
    "GeometricCDLModel",
    "GeometricCDLRealization",
    "GeometricCDLSamplingRequest",
    "GeometricCDLState",
    "LargeScaleTrajectoryBatch",
    "RayTrajectoryBatch",
    "ScenarioTimelineBatch",
    "SpatialFieldBank",
    "TopologyTrajectoryBatch",
    "construct_scatterers",
    "drift_paths",
    "load_geometric_cdl_realization",
    "load_geometric_resource",
    "parse_geometric_cdl_config",
    "register_geometric_cdl_model",
    "save_geometric_cdl_realization",
]
