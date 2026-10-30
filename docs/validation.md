# Validation record

The v0.1.0 release-readiness baseline was completed on 2026-09-23 with Python
3.14.3, NumPy 2.4.2, and PyTorch 2.10.0+cu128. The available CUDA device was
an NVIDIA GeForce RTX 5080.

The mandatory fast suite, the statistical/reference selection, Ruff, the
warning-as-error Sphinx build, the CUDA smoke/gradient test, and wheel creation
all passed. GitHub Actions repeats the NumPy-only suite on Python 3.11–3.14,
the PyTorch CPU suite, combined installations on the oldest and newest
supported Python versions, documentation, lint, and package construction.

## Representative benchmark

The opt-in benchmark uses batch size 32, 4x4 transmit/receive links, 4x4
antenna dimensions, 14 time samples, and 256 OFDM subcarriers. Mean local
times over ten iterations were:

| Backend/device | Seconds per iteration |
| --- | ---: |
| PyTorch CPU | 0.0699 |
| PyTorch CUDA | 0.0609 |
| NumPy CPU | 0.1919 |

These values are environment records, not pass/fail thresholds. Performance
tests remain non-blocking until stable multi-runner baselines exist. The
results confirm PyTorch as the optimization-priority backend while NumPy
continues to execute independently.

## Schema status

Configuration envelope version 1 and CIR realization schema version 1 are
frozen for this release boundary. Future incompatible changes require a new
schema version and an explicit migration path. Replay and cross-backend tests
cover the current format.
