API reference
=============

The pages below are generated from the public Python interfaces and their
NumPy-style docstrings. Array dimensions are named explicitly in each wireless
value object's documentation.

Backends and random streams
---------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_dataset.backends.ArrayBackend
   channel_dataset.backends.RandomStream
   channel_dataset.backends.get_backend
   channel_dataset.backends.create_random_stream
   channel_dataset.backends.backend_name_for_array
   channel_dataset.backends.numpy_backend.NumpyBackend
   channel_dataset.backends.numpy_backend.NumpyRandomStream
   channel_dataset.backends.torch_backend.TorchBackend
   channel_dataset.backends.torch_backend.TorchRandomStream

Core domain objects
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_dataset.core.Axis
   channel_dataset.core.ExecutionContext
   channel_dataset.core.Vector3
   channel_dataset.core.Endpoint
   channel_dataset.core.SimulationContext
   channel_dataset.core.ChannelModel
   channel_dataset.core.Direction
   channel_dataset.core.PropagationPath
   channel_dataset.core.PathCluster
   channel_dataset.core.RealizationMetadata
   channel_dataset.core.ChannelRealization

Configuration and model registry
--------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_dataset.config.ConfigEnvelope
   channel_dataset.config.ExecutionConfig
   channel_dataset.config.ModelProfile
   channel_dataset.config.load_config
   channel_dataset.config.load_resource_document
   channel_dataset.config.parse_config_document
   channel_dataset.config.resolve_model_profile
   channel_dataset.config.resource_checksum
   channel_dataset.registry.ModelRegistration
   channel_dataset.registry.ModelRegistry
   channel_dataset.channels.wireless.register_wireless_models

Wireless requests and value objects
-----------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_dataset.channels.wireless.ChannelSamplingRequest
   channel_dataset.channels.wireless.CIRBatch
   channel_dataset.channels.wireless.TimeChannelResponse
   channel_dataset.channels.wireless.OFDMChannelResponse
   channel_dataset.channels.wireless.FlatChannelResponse
   channel_dataset.channels.wireless.TopologyBatch
   channel_dataset.channels.wireless.LargeScaleParameters
   channel_dataset.channels.wireless.ClusterBatch
   channel_dataset.channels.wireless.RayBatch
   channel_dataset.channels.wireless.TopologyTrajectoryBatch
   channel_dataset.channels.wireless.ScenarioTimelineBatch
   channel_dataset.channels.wireless.DynamicCIRBatch
   channel_dataset.channels.wireless.RayTrajectoryBatch
   channel_dataset.channels.wireless.GeometricCDLState
   channel_dataset.channels.wireless.GeometricCDLSamplingRequest
   channel_dataset.channels.wireless.GeometricCDLRealization
   channel_dataset.channels.wireless.select_paths

Geometrically consistent CDL
----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_dataset.channels.wireless.GeometricCDLConfig
   channel_dataset.channels.wireless.GeometricCDLModel
   channel_dataset.channels.wireless.register_geometric_cdl_model
   channel_dataset.channels.wireless.save_geometric_cdl_realization
   channel_dataset.channels.wireless.load_geometric_cdl_realization
   channel_dataset.channels.wireless.load_geometric_resource

IEEE 802.11 WLAN models
-----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_dataset.channels.wireless.WLANChannelSamplingRequest
   channel_dataset.channels.wireless.WLANRealizationDiagnostics
   channel_dataset.channels.wireless.IEEE80211Profile
   channel_dataset.channels.wireless.IEEE80211TapProfile
   channel_dataset.channels.wireless.IEEE80211ClusterProfile
   channel_dataset.channels.wireless.TGacBandwidthRule
   channel_dataset.channels.wireless.TGacMultiUserRule
   channel_dataset.channels.wireless.TGahBandwidthRule
   channel_dataset.channels.wireless.TGahPathLossProfile
   channel_dataset.channels.wireless.TGaxIndoorProfile
   channel_dataset.channels.wireless.TGaxOutdoorProfile
   channel_dataset.channels.wireless.TGnConfig
   channel_dataset.channels.wireless.TGnChannel
   channel_dataset.channels.wireless.TGacConfig
   channel_dataset.channels.wireless.TGacChannel
   channel_dataset.channels.wireless.TGahConfig
   channel_dataset.channels.wireless.TGahChannel
   channel_dataset.channels.wireless.TGaxConfig
   channel_dataset.channels.wireless.TGaxChannel
   channel_dataset.channels.wireless.load_ieee80211_profile
   channel_dataset.channels.wireless.register_ieee80211_models

3GPP TR 38.811 NTN models
-------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_dataset.channels.wireless.NTNChannelSamplingRequest
   channel_dataset.channels.wireless.NTNTopologyBatch
   channel_dataset.channels.wireless.NTNLinkGeometry
   channel_dataset.channels.wireless.NTNLargeScaleComponents
   channel_dataset.channels.wireless.NTNChannelRealization
   channel_dataset.channels.wireless.NTNRealizationDiagnostics
   channel_dataset.channels.wireless.TR38811DeploymentConfig
   channel_dataset.channels.wireless.TR38811FlatFadingConfig
   channel_dataset.channels.wireless.TR38811TDLConfig
   channel_dataset.channels.wireless.TR38811CDLConfig
   channel_dataset.channels.wireless.TR38811SystemConfig
   channel_dataset.channels.wireless.NTNAtmosphereConfig
   channel_dataset.channels.wireless.NTNScintillationConfig
   channel_dataset.channels.wireless.NTNAntennaConfig
   channel_dataset.channels.wireless.TR38811FlatFadingChannel
   channel_dataset.channels.wireless.TR38811TDLChannel
   channel_dataset.channels.wireless.TR38811CDLChannel
   channel_dataset.channels.wireless.TR38811SystemChannel
   channel_dataset.channels.wireless.ntn_to_ofdm
   channel_dataset.channels.wireless.save_ntn_realization
   channel_dataset.channels.wireless.load_ntn_realization
   channel_dataset.channels.wireless.register_tr38811_models

3GPP TR 37.885 urban V2X models
--------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_dataset.channels.wireless.TR37885UrbanGridConfig
   channel_dataset.channels.wireless.TR37885TrafficConfig
   channel_dataset.channels.wireless.TR37885UrbanSystemConfig
   channel_dataset.channels.wireless.TR37885UrbanCDLConfig
   channel_dataset.channels.wireless.TR37885UrbanGrid
   channel_dataset.channels.wireless.TR37885UrbanGridSimulator
   channel_dataset.channels.wireless.V2XSceneSamplingRequest
   channel_dataset.channels.wireless.V2XSceneTrajectory
   channel_dataset.channels.wireless.V2XVehicleAntennaProfile
   channel_dataset.channels.wireless.V2XLinkConditionBatch
   channel_dataset.channels.wireless.V2XChannelState
   channel_dataset.channels.wireless.V2XChannelSamplingRequest
   channel_dataset.channels.wireless.V2XLargeScaleComponents
   channel_dataset.channels.wireless.V2XChannelRealization
   channel_dataset.channels.wireless.TR37885UrbanSystemChannel
   channel_dataset.channels.wireless.TR37885UrbanCDLChannel
   channel_dataset.channels.wireless.save_v2x_realization
   channel_dataset.channels.wireless.load_v2x_realization
   channel_dataset.channels.wireless.load_vehicle_antenna_profile
   channel_dataset.channels.wireless.register_tr37885_models

Topology and scenario geometry
------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_dataset.channels.wireless.LinkGeometry
   channel_dataset.channels.wireless.RectangularTopologyConfig
   channel_dataset.channels.wireless.RectangularTopologyGenerator
   channel_dataset.channels.wireless.Scenario
   channel_dataset.channels.wireless.ScenarioConfig
   channel_dataset.channels.wireless.GeometryScenario
   channel_dataset.channels.wireless.LargeScaleScenario
   channel_dataset.channels.wireless.compute_link_geometry
   channel_dataset.channels.wireless.reverse_ray_batch
   channel_dataset.channels.wireless.load_scenario_geometry
   channel_dataset.channels.wireless.create_scenario
   channel_dataset.channels.wireless.load_large_scale_parameters

Cluster and ray generation
--------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_dataset.channels.wireless.RayGenerationConfig
   channel_dataset.channels.wireless.ScenarioRayGenerator
   channel_dataset.channels.wireless.SubclusterPlan
   channel_dataset.channels.wireless.ExponentialSpatialField
   channel_dataset.channels.wireless.load_ray_generation_parameters

System-level channel assembly
-----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_dataset.channels.wireless.CoefficientDiagnostics
   channel_dataset.channels.wireless.SystemLevelCoefficientGenerator
   channel_dataset.channels.wireless.SystemLevelResult
   channel_dataset.channels.wireless.SystemLevelWirelessChannel
   channel_dataset.channels.wireless.evaluate_array_field
   channel_dataset.channels.wireless.polarization_coupling
   channel_dataset.channels.wireless.array_phase_offsets
   channel_dataset.channels.wireless.doppler_phase_evolution
   channel_dataset.channels.wireless.apply_large_scale_loss

Blockage and spatial consistency
--------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_dataset.channels.wireless.WirelessCapabilityMetadata
   channel_dataset.channels.wireless.SelfBlockingModel
   channel_dataset.channels.wireless.NonSelfBlockingModelA
   channel_dataset.channels.wireless.ScreenBlocker
   channel_dataset.channels.wireless.ScreenBlockingModelB
   channel_dataset.channels.wireless.SnapshotSpatialConsistency
   channel_dataset.channels.wireless.SpatialConsistencySnapshot
   channel_dataset.channels.wireless.apply_blockage
   channel_dataset.channels.wireless.apply_blockage_to_coefficients
   channel_dataset.channels.wireless.load_blockage_parameters

Datasets, replay, and metrics
-----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_dataset.channels.wireless.StreamingCIRDataset
   channel_dataset.channels.wireless.ReplayChannel
   channel_dataset.channels.wireless.save_cir
   channel_dataset.channels.wireless.load_cir
   channel_dataset.channels.wireless.shuffle_stream
   channel_dataset.channels.wireless.rms_delay_spread
   channel_dataset.channels.wireless.circular_angular_spread
   channel_dataset.channels.wireless.mimo_singular_values
   channel_dataset.channels.wireless.frequency_selectivity

Shared wireless mathematics
---------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_dataset.channels.wireless.math.power_db_to_linear
   channel_dataset.channels.wireless.math.linear_power_to_db
   channel_dataset.channels.wireless.math.amplitude_db_to_linear
   channel_dataset.channels.wireless.math.linear_amplitude_to_db
   channel_dataset.channels.wireless.math.degrees_to_radians
   channel_dataset.channels.wireless.math.radians_to_degrees
   channel_dataset.channels.wireless.math.wrap_angle_radians
   channel_dataset.channels.wireless.math.wrap_angle_degrees
   channel_dataset.channels.wireless.math.spherical_to_cartesian
   channel_dataset.channels.wireless.math.cartesian_to_spherical
   channel_dataset.channels.wireless.math.exponential_correlation
   channel_dataset.channels.wireless.math.one_ring_correlation
   channel_dataset.channels.wireless.math.matrix_sqrt_psd
   channel_dataset.channels.wireless.math.wavelength_m
   channel_dataset.channels.wireless.math.doppler_shift_hz
   channel_dataset.channels.wireless.math.euclidean_distance
   channel_dataset.channels.wireless.math.line_of_sight_unit_vector
   channel_dataset.channels.wireless.math.closing_speed_mps
   channel_dataset.channels.wireless.math.complex_normal

Fading and random sampling
--------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_dataset.channels.wireless.fading.RayleighBlockFadingConfig
   channel_dataset.channels.wireless.fading.RayleighBlockFading
   channel_dataset.channels.wireless.fading.FlatMIMOFadingConfig
   channel_dataset.channels.wireless.fading.FlatMIMOFading
   channel_dataset.channels.wireless.fading.parse_rayleigh_config
   channel_dataset.channels.wireless.fading.register_rayleigh_model
   channel_dataset.channels.wireless.fading.parse_flat_mimo_config
   channel_dataset.channels.wireless.fading.register_flat_mimo_model
   channel_dataset.channels.wireless.fading.apply_kronecker_correlation
   channel_dataset.channels.wireless.fading.apply_per_column_receive_correlation
   channel_dataset.channels.wireless.noise.AWGNConfig
   channel_dataset.channels.wireless.noise.AdditiveWhiteGaussianNoise
   channel_dataset.channels.wireless.noise.parse_awgn_config

3GPP tapped- and clustered-delay-line models
---------------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_dataset.channels.wireless.tr38901.TDLConfig
   channel_dataset.channels.wireless.tr38901.TDLProfile
   channel_dataset.channels.wireless.tr38901.TappedDelayLine
   channel_dataset.channels.wireless.tr38901.load_tdl_profile
   channel_dataset.channels.wireless.tr38901.parse_tdl_config
   channel_dataset.channels.wireless.tr38901.register_tdl_model
   channel_dataset.channels.wireless.tr38901.CDLConfig
   channel_dataset.channels.wireless.tr38901.CDLProfile
   channel_dataset.channels.wireless.tr38901.ClusteredDelayLine
   channel_dataset.channels.wireless.tr38901.load_cdl_profile
   channel_dataset.channels.wireless.tr38901.load_cdl_ray_offsets
   channel_dataset.channels.wireless.tr38901.parse_cdl_config
   channel_dataset.channels.wireless.tr38901.register_cdl_model

Antenna coordinates, elements, and arrays
-----------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_dataset.channels.wireless.antennas.Orientation
   channel_dataset.channels.wireless.antennas.PolarizedFieldPattern
   channel_dataset.channels.wireless.antennas.AntennaElement
   channel_dataset.channels.wireless.antennas.IsotropicAntennaElement
   channel_dataset.channels.wireless.antennas.ThreeGPPAntennaElement
   channel_dataset.channels.wireless.antennas.AntennaPanel
   channel_dataset.channels.wireless.antennas.MultiPanelArray
   channel_dataset.channels.wireless.antennas.AntennaLayout
   channel_dataset.channels.wireless.antennas.spherical_basis
   channel_dataset.channels.wireless.antennas.rotation_matrix
   channel_dataset.channels.wireless.antennas.global_to_local_angles
   channel_dataset.channels.wireless.antennas.local_to_global_angles
   channel_dataset.channels.wireless.antennas.local_field_to_global
   channel_dataset.channels.wireless.antennas.load_antenna_layout
   channel_dataset.channels.wireless.antennas.parse_antenna_layout

Time and OFDM transforms and operators
--------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   channel_dataset.channels.wireless.transforms.subcarrier_frequencies
   channel_dataset.channels.wireless.transforms.time_lag_bounds
   channel_dataset.channels.wireless.transforms.time_lags
   channel_dataset.channels.wireless.transforms.cir_to_time
   channel_dataset.channels.wireless.transforms.cir_to_ofdm
   channel_dataset.channels.wireless.transforms.normalize_response
   channel_dataset.channels.wireless.operators.apply_flat
   channel_dataset.channels.wireless.operators.apply_time
   channel_dataset.channels.wireless.operators.apply_ofdm
