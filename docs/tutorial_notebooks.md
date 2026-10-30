# Interactive tutorial notebooks

The nine standalone notebooks in `examples/notebooks/` turn the public API
guides into executable, plotted workflows. They use PyTorch on CPU, fixed seeds,
small tensors, and assertions that are useful for both learning and manual
regression checks.

The setup cell discovers the repository root and adds its `src/` directory to
the kernel import path. Direct VS Code execution and a Jupyter server launched
from the repository root therefore use the same source checkout; the selected
kernel still needs the `notebooks` dependencies installed.

Install and start JupyterLab from the repository root:

```powershell
python -m pip install -e ".[notebooks]"
python -m jupyter lab examples/notebooks
```

The recommended order is:

1. [Getting started](../examples/notebooks/00_getting_started.ipynb): configuration, registry, CIR, and OFDM.
2. [Signal domains and noise](../examples/notebooks/01_signal_domains_and_noise.ipynb): flat MIMO, time-domain processing, and AWGN.
3. [3GPP TDL, CDL, and antennas](../examples/notebooks/02_3gpp_tdl_cdl_and_antennas.ipynb): standardized profiles and diagnostics.
4. [System scenarios and blockage](../examples/notebooks/03_system_scenarios_and_blockage.ipynb): topology through system-level coefficients.
5. [Persistence, replay, and metrics](../examples/notebooks/04_persistence_replay_and_metrics.ipynb): portable stored realizations and analysis.
6. [IEEE 802.11 WLAN](../examples/notebooks/05_ieee80211_wlan.ipynb): TGn, TGac, TGah, and TGax.
7. [TR 38.811 NTN](../examples/notebooks/06_tr38811_ntn.ipynb): satellite geometry, losses, timing, and persistence.
8. [TR 37.885 V2X](../examples/notebooks/07_tr37885_v2x.ipynb): urban traffic, link state, channel snapshots, and persistence.
9. [Geometrically consistent CDL evolution](../examples/notebooks/08_geometric_cdl_evolution.ipynb): persistent scatterers, dynamic multicarrier CIRs, segment transitions, exact chunk continuation, and persistence.

Every notebook ends with a manual-check list and suggested parameter changes.
The saved plots make the material readable without execution, while CI executes
fresh in-memory copies to detect broken imports, stale APIs, and failed sanity
assertions.
