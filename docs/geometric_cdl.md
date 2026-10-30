# Geometrically consistent CDL evolution

The `wireless/geometric_cdl` model evolves persistent cluster geometry along a
sampled transmitter/receiver trajectory. It is separate from the snapshot
`ClusteredDelayLine` API, so existing CDL, TDL, NTN, V2X, and system-level
models retain their previous behavior.

The public model accepts a `GeometricCDLSamplingRequest` containing:

- a normal `ChannelSamplingRequest`;
- a time-last `TopologyTrajectoryBatch`;
- a `ScenarioTimelineBatch`; and
- optionally, the `GeometricCDLState` returned by the preceding contiguous
  request.

`GeometricCDLModel.realize()` returns one `CarrierGeometricRealization` for
each configured carrier, in configuration order. Each carrier contains a
`DynamicCIRBatch`, a diagnostic `RayTrajectoryBatch`, and its large-scale
parameter trajectory. Absolute path length drives phase; the CIR exposes
excess delay by default.

## Configuration

`GeometricCDLConfig` supports the versioned `16.1.0` and `19.2.0` resource
namespaces, scenario or CDL-A--E initialization, correlated multiband output,
automatic or explicit segmentation, transition overlap, ground reflection,
and strict/single-bounce/masked geometry failure policies. The checked example
is in `configs/wireless/geometric_cdl_multiband.yaml` and its strict schema is
`configs/schema/geometric-cdl-v1.schema.json`.

Configuration documents can construct the model through the registry after
calling `register_geometric_cdl_model(registry)`.

## Dynamic response processing

`cir_to_ofdm`, `cir_to_time`, and `rms_delay_spread` accept both static
`CIRBatch` and `DynamicCIRBatch` values. Dynamic conversion uses the delay at
the matching time sample. `DynamicCIRBatch.at_time(index)` produces a normal
static CIR snapshot without changing the static CIR schema.

## Continuation and persistence

Continuation is explicit. A prior state must have the same model digest,
carrier order, random-stream root and lineage, endpoint identities, execution
contract, and an exactly contiguous first sample. No incompatible state is
silently reset. The per-sample displacement may not exceed one quarter of the
shortest configured wavelength.

Use `save_geometric_cdl_realization()` and
`load_geometric_cdl_realization()` for checksummed realization/state storage.
The stored arrays can be loaded into either supported backend when precision
and device requirements are satisfied.

## Differentiability and validation boundary

The PyTorch path preserves gradients through deterministic geometry, drifting,
endpoint orientation, delay, path-length phase, array phase, and transition
ramps. Discrete LOS, segment, matching, fallback, and activity decisions are
not differentiable.

QuaDRiGa is not a production dependency. The packaged standard-resource rows
remain marked `pending_independent_review`; this status must be cleared only
after source-by-source review and the external v16.1 statistical oracle suite
have been completed.

The executable
[geometric CDL tutorial](../examples/notebooks/08_geometric_cdl_evolution.ipynb)
demonstrates the complete public workflow, including a segment boundary shared
between one-shot and resumed generation.
