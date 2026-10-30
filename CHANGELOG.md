# Changelog

This project follows [Semantic Versioning](https://semver.org/). User-visible
changes are recorded here; internal implementation plans remain in `docs/`.

## [0.1.0] - 2026-10-29

First wireless-focused release.

### Added

- Native NumPy and PyTorch backends with injected, reproducible random streams.
- Typed batched CIR, flat, time, OFDM, topology, cluster, ray, and scenario APIs.
- Rayleigh, flat MIMO, AWGN, 3GPP TR 38.901 TDL/CDL and system-level models.
- IEEE 802.11 TGn/TGac/TGah and documented TGax indoor/outdoor subsets.
- 3GPP TR 38.811 NTN and TR 37.885 urban V2X model families.
- Geometric CDL evolution, checksummed persistence/replay, and channel metrics.
- A convenience function to register all configuration-backed wireless models.

### Known limitations

- Several standards-derived resource tables remain pending independent review.
- TGad/TGay, full dynamic blocker evolution, and the deferred scenarios listed
  in `docs/capabilities.md` are outside this release boundary.
