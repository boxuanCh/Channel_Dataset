# Supported capabilities and validation status

This matrix describes the schema-version-1 release boundary. “Supported”
means the feature has deterministic tests on native NumPy and PyTorch CPU
backends unless the row says otherwise.

| Area | Supported subset | Validation |
| --- | --- | --- |
| Backends | NumPy 1.26–2.x on CPU; PyTorch 2.x on CPU; CUDA-compatible array paths | Python 3.11–3.14 CI; CUDA tests are marked and require an external CUDA runner |
| Basic channels | Rayleigh block fading, configurable flat MIMO, AWGN | Seed replay, moments, gradients, analytic application |
| 3GPP TDL | TDL-A through TDL-E, TR 38.901 V16.1.0; TS 38.101-4 fixed conformance profiles | Resource, Doppler, K-factor, correlation, statistical tests |
| 3GPP CDL | CDL-A through CDL-E, TR 38.901 V16.1.0 | Angles, coupling, polarization, arrays, Doppler, statistical tests |
| Geometric CDL evolution | Persistent scatterers, path-length phase, multiband dynamic CIR, explicit state continuation, and cluster transitions for versioned TR 38.901 v16.1/v19.2 namespaces | NumPy/PyTorch contracts, geometry, replay, chunk continuation, transforms, persistence; packaged evolution resources pending independent review |
| IEEE TGn | 03/940r4 A-F, SISO/MIMO, Bell Doppler, LoS, breakpoint loss, optional environmental effects | Exact resources, moments, covariance PSD/statistics, replay, OFDM/time integration |
| IEEE TGac | 09/0308r12 20/40/80/160 MHz, up to 8x8, topology-bound SU/MU | Delay vectors, 8x8, user-order invariance, OFDM integration |
| IEEE TGah | 11/0968r4 sub-1-GHz 1/2/4/8/16 MHz and four deployment modes | Delay, path-loss, shadowing, replay, OFDM integration |
| IEEE TGax | 14/0882r4 indoor composition; outdoor UMi/UMa loss/shadowing and fast-object subset | Penetration, reference loss equations, topology and spectral behavior |
| 3GPP NTN | TR 38.811 V15.4.0 geometry, D1-D5 applicability, TDL/CDL profiles, fixed-state LMS fading, LOS and large-scale decomposition, absolute-delay OFDM, antennas, Faraday rotation, injected atmospheric state, scintillation, and checksummed persistence | Exact resource rows, analytical geometry/loss/phase, statistics, replay, gradients, parity, registry, persistence |
| 3GPP V2X | TR 37.885 V15.3.0 urban V2V at 6 GHz: reference grid, mobility Options A/B, LOS/NLOSv/NLOS, loss decomposition, stochastic/fixed CDL, Doppler, snapshot state, and persistence | Resource checksums, geometry, mobility, analytical loss, statistics, replay, parity, OFDM, registry, and persistence |
| System scenarios | RMa, UMa, UMi, InH mixed office, InF sparse-clutter/low-BS snapshot subset | Geometry, LoS, path loss, LSP, rays, antenna-aware CIR and OFDM tests |
| Blockage | Self-blocking portrait/landscape, snapshot Model A, stationary-screen Model B | Boundary, attenuation, repeatability, stream-isolation tests |
| Spatial consistency | Snapshot LoS, LSP, and small-scale exponentially correlated fields | Seed replay and correlation-versus-distance tests |
| Persistence | CIR schema v1, checksummed NPY batch chunks, cross-backend loading and replay | Schema, chunking, corruption, NumPy↔PyTorch tests |
| Metrics | RMS delay spread, circular angular spread, MIMO singular values, frequency selectivity | Analytic fixtures on both backends |

## Known limitations

- Temporal blocker evolution and dynamic blocker birth/death are not
  implemented. The dedicated geometric CDL model covers trajectory-based
  TR 38.901 cluster evolution; TR 37.885 retains its own lane-bound snapshot
  sequence and condition/shadowing state contract.
- IEEE WLAN realizations are continuous only within one requested fixed
  window. Cross-call fading state is not implemented.
- TGax outdoor full ITU-R LSP/cluster calibration is not claimed; the current
  outdoor subset uses an explicitly selected A-F small-scale kernel. TGad and
  TGay are deferred.
- The TR 38.811 system model currently composes the standardized NTN CDL and
  LOS/large-scale tables. It does not yet claim the complete clause-6.7.2 LSP
  cross-correlation and spatial-consistency calibration pipeline. Exact
  P.676/P.618/P.840 weather engines and a packaged P.681 LMS parameter atlas
  are also not claimed; site-specific attenuation and LMS triplets are typed,
  caller-supplied inputs. The shipped 3GPP tables remain pending independent
  review.
- TR 37.885 support is urban V2V only. Highway, V2R/R2R, pedestrians, base
  stations, real maps, traffic lights, lane changes, 3-D ray tracing, MAC, and
  network metrics are deferred. Controlled grid/traffic variants are labelled
  non-baseline. The shipped 3GPP tables remain pending independent review.
- Scenario support is snapshot-based. It does not claim every optional clause
  or calibration case in TR 38.901.
- The InF implementation is the sparse-clutter/low-base-station subset; the
  InH implementation is mixed office.
- CUDA code paths are defined and marked, but this repository does not claim a
  local CUDA result unless CI is attached to a suitable runner.
- `torch.compile` is not enabled by default. The current coefficient assembly
  includes small Python loops over topology links, and no measured benefit has
  justified freezing a compiled contract.

## Reproducibility boundary

Random components use SHA-256-derived child streams. A seed reproduces results
within one backend; deterministic calculations are compared across backends at
double precision. NumPy and PyTorch random generators are intentionally native,
so their stochastic samples are not required to be bit-identical.

The external 3GPP tables currently carry `pending_independent_review` metadata.
They are covered by transcription and physical-sanity tests but should receive
an independent standards review before regulatory or conformance use.

The IEEE 802.11 transcriptions also carry `pending_independent_review`. Tests
and canonical checksums do not replace a second-person standards review.
