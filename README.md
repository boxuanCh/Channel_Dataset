# Channel Dataset

An extensible, object-oriented framework for reproducible physical-layer
communication channel simulation. Version 0.1.0 is the first wireless-focused
release. It provides typed configuration, injected random streams, reusable
channel realizations, NumPy and PyTorch backends, deterministic transforms, and
portable persistence.

## Release scope

- Basic Rayleigh block fading, correlated flat MIMO fading, and independent AWGN.
- 3GPP TR 38.901 TDL/CDL profiles, antenna arrays, system scenarios, blockage,
  spatial consistency, and geometric CDL evolution.
- IEEE 802.11 TGn, TGac, TGah, and the documented TGax subset.
- 3GPP TR 38.811 non-terrestrial and TR 37.885 urban V2X models.
- Time-domain, flat, and OFDM application; checksummed persistence and replay;
  propagation and MIMO metrics.

The exact supported subsets and known limitations are listed in the
[capability matrix](docs/capabilities.md). Several standards-derived parameter
tables are still marked `pending_independent_review`; this release is a
research and simulation implementation, not a formal conformance claim.

## Quick start

Install the NumPy-only development environment and run a seeded example:

```powershell
python -m pip install -e ".[test,wireless-numpy]"
python examples/wireless_ieee80211_tgn_ofdm.py
```

Configuration-backed models share one construction path:

```python
from channel_dataset.channels.wireless import register_wireless_models
from channel_dataset.config import load_config
from channel_dataset.registry import ModelRegistry

config = load_config("configs/wireless/ieee80211/tgn_model_b.yaml")
registry = ModelRegistry()
register_wireless_models(registry)
model = registry.create(config)
```

Continue with the [getting-started guide](docs/getting_started.md) for a full
realize-transform-apply workflow. The [Python API guide](docs/api.md) defines
tensor axes, backend semantics, persistence, and the lower-level APIs.

## Design principles

- Shared physical concepts—paths, rays, clusters, channel responses, scenarios, and realizations—have medium-independent representations.
- Each channel family owns its medium-specific physics.
- Model parameters live in versioned YAML/JSON files, not in implementation code.
- A model first samples a reusable realization and then applies it to signals.
- Random generators are injected for reproducibility.

## Repository map

```text
configs/                 Runnable, versioned model configurations
docs/                    User guides, architecture, and validation notes
examples/                Executable scripts and tutorial notebooks
src/channel_dataset/     Installable package and bundled model resources
tests/                   Unit, integration, statistical, and package tests
```

## Development setup

Supported versions are Python 3.11 through 3.14, NumPy 1.26 through 2.x, and
PyTorch 2.x. PyTorch is the default backend; select NumPy explicitly for a
PyTorch-free runtime.

```powershell
python -m pip install -e ".[dev,wireless]"
python -m pytest
```

The test configuration automatically gives every pytest process a unique
temporary directory. This avoids Windows permission conflicts when sandboxed
and interactive runs use different security identities. Do not add a shared
``--basetemp=.pytest-tmp`` option. If system policy blocks the standard Windows
temporary directory itself, select a writable parent explicitly:

```powershell
$env:CHANNEL_DATASET_PYTEST_TMPDIR = Join-Path $env:LOCALAPPDATA "ChannelDatasetTests"
python -m pytest
```

For a NumPy-only installation:

```powershell
python -m pip install -e ".[test,wireless-numpy]"
```

Build and view the documentation website with:

```powershell
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

Then open <http://localhost:8000/>. The strict Sphinx build treats warnings as
errors, matching the documentation job in GitHub Actions.

Run the configured seeded example with:

```powershell
python examples/wireless_rayleigh_ofdm.py
python examples/wireless_ieee80211_tgn_ofdm.py
python examples/wireless_tr38811_ntn_ofdm.py
python examples/wireless_tr37885_v2x_ofdm.py
python examples/wireless_geometric_cdl_evolution.py
python examples/wireless_dataset_replay.py
```

For interactive, plotted tutorials covering the core lifecycle, signal-domain
processing, 3GPP TDL/CDL and system scenarios, persistence, IEEE 802.11, NTN,
V2X, and geometric CDL evolution, use the
[Jupyter notebook suite](examples/notebooks/README.md):

```powershell
python -m pip install -e ".[notebooks]"
python -m jupyter lab examples/notebooks
```

`configs/wireless/rayleigh_block_fading.yaml` is a runnable configuration.
`configs/wireless/rayleigh_block_fading_profile.yaml` demonstrates packaged
profile inheritance and a deterministic envelope override.
`configs/wireless/flat_mimo_fading.yaml` and `configs/wireless/awgn.yaml`
form a replayable flat-MIMO-plus-noise example configuration pair.
`configs/wireless/example_tdl.yaml` is a runnable standards-backed TDL-C
configuration.
`configs/wireless/example_cdl.yaml` is a runnable single-link CDL-C
configuration.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the change workflow and
[RELEASING.md](RELEASING.md) for the reproducible release checklist. The
project is licensed under Apache-2.0.
