# Getting started

This guide uses the NumPy backend so the first run does not require PyTorch.
Run all commands from the repository root; example configuration paths are
intentionally relative to that directory.

## Install

Create and activate a Python 3.11-3.14 virtual environment, then install the
package, tests, and NumPy wireless support:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test,wireless-numpy]"
```

On macOS or Linux, activate with `source .venv/bin/activate`. Install
`.[dev,wireless]` instead when developing both the NumPy and PyTorch paths.

## Run a complete model lifecycle

The following uses the checked-in TGn Model B configuration. It loads and
validates YAML, constructs the model through the registry, samples a fixed CIR,
converts that CIR to an OFDM response, and applies the response to a grid.

```python
from channel_dataset.channels.wireless import (
    ChannelSamplingRequest,
    register_wireless_models,
)
from channel_dataset.channels.wireless.ieee80211 import WLANChannelSamplingRequest
from channel_dataset.channels.wireless.operators import apply_ofdm
from channel_dataset.channels.wireless.transforms import (
    cir_to_ofdm,
    subcarrier_frequencies,
)
from channel_dataset.config import load_config
from channel_dataset.registry import ModelRegistry

config = load_config("configs/wireless/ieee80211/tgn_model_b.yaml")
if config.execution is None:
    raise ValueError("the example configuration needs an execution section")

registry = ModelRegistry()
register_wireless_models(registry)
model = registry.create(config)

execution = config.execution.context()
sampling = ChannelSamplingRequest(
    batch_size=1,
    num_time_samples=4,
    sampling_frequency_hz=20_000.0,
    execution=execution,
    random_stream=execution.random_stream(config.execution.seed),
    request_id="getting-started",
)
cir = model.realize(WLANChannelSamplingRequest(sampling))

frequencies_hz = subcarrier_frequencies(64, 312_500.0, execution)
response = cir_to_ofdm(cir, frequencies_hz)
transmitted = execution.array_backend.ones(
    (1, 1, 2, 4, 64),
    dtype=execution.complex_dtype,
    device=execution.device,
)
received = apply_ofdm(response, transmitted)

print("CIR:", cir.shape)
print("OFDM response:", response.shape)
print("received grid:", received.shape)
print("seed:", cir.metadata.root_seed)
```

The canonical axes are:

- CIR coefficients: `[batch, rx, rx_antenna, tx, tx_antenna, path, time]`.
- OFDM response: `[batch, rx, rx_antenna, tx, tx_antenna, symbol, subcarrier]`.
- Transmit grid: `[batch, tx, tx_antenna, symbol, subcarrier]`.
- Receive grid: `[batch, rx, rx_antenna, symbol, subcarrier]`.

Recreating the request stream from the same seed replays a realization within
the selected backend. NumPy and PyTorch use their native random generators, so
random samples are not promised to be bit-identical across backends.

## Choose the next guide

| Goal | Guide or example |
| --- | --- |
| Learn all tensor and operator contracts | [Python API](api.md) |
| Check the supported release boundary | [Capabilities](capabilities.md) |
| Use 3GPP TDL/CDL and system scenarios | [Python API](api.md#3gpp-tapped-delay-line-models) |
| Use IEEE 802.11 WLAN models | [WLAN models](ieee80211_models.md) |
| Use satellite or HAPS channels | [TR 38.811 NTN](tr38811_models.md) |
| Use urban vehicle-to-vehicle channels | [TR 37.885 V2X](tr37885_models.md) |
| Model evolving persistent scatterers | [Geometric CDL](geometric_cdl.md) |
| Store and replay CIR batches | `examples/wireless_dataset_replay.py` |
| Work interactively | [Tutorial notebooks](tutorial_notebooks.md) |

Family-specific registration functions remain available for applications that
want to expose only selected models. `register_wireless_models` is the simplest
default for tools and agents that need to resolve arbitrary checked-in wireless
configurations.

## Common failures

- `KeyError: unknown channel model` means the matching registration function
  was not called before `registry.create(...)`.
- Backend or device mismatch errors mean the request's random stream was not
  created from the same `ExecutionContext`.
- Shape errors should be checked against the canonical axes above; dimensions
  of size one are still semantic axes and must not be removed.
- Standards resources marked `pending_independent_review` are tested
  transcriptions, not a formal standards-conformance certification.
