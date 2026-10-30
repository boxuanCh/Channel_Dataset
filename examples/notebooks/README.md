# Interactive tutorials

These notebooks form an ordered learning path through the implemented wireless
channel APIs. Each notebook is standalone: it repeats the small amount of setup
needed to locate the repository, select the PyTorch CPU backend, and convert
backend arrays for plotting.

Each notebook adds the checkout's `src/` directory to its kernel import path.
It can therefore be opened directly in VS Code or launched from JupyterLab at
the repository root. In either case, select a Python kernel that has the
`notebooks` dependencies installed.

Install and launch the notebook environment from the repository root:

```powershell
python -m pip install -e ".[notebooks]"
python -m jupyter lab examples/notebooks
```

The committed outputs were produced with deterministic seeds. Re-running a
notebook should preserve the asserted numerical behavior, although PNG bytes
and minor plot rendering details can vary between Matplotlib versions.

| Notebook | Purpose | Typical CPU time |
| --- | --- | ---: |
| `00_getting_started.ipynb` | Configuration, registry, seeded CIR, and OFDM lifecycle | < 10 s |
| `01_signal_domains_and_noise.ipynb` | Flat MIMO, time/OFDM application, normalization, and AWGN | < 10 s |
| `02_3gpp_tdl_cdl_and_antennas.ipynb` | TR 38.901 TDL/CDL profiles, rays, polarization, and antennas | < 15 s |
| `03_system_scenarios_and_blockage.ipynb` | Topology, scenarios, system coefficients, blockage, and consistency | < 15 s |
| `04_persistence_replay_and_metrics.ipynb` | Checksummed storage, cross-backend replay, streams, and metrics | < 10 s |
| `05_ieee80211_wlan.ipynb` | TGn, TGac, TGah, and TGax fixed-window workflows | < 15 s |
| `06_tr38811_ntn.ipynb` | NTN geometry, loss, OFDM timing, and persistence | < 10 s |
| `07_tr37885_v2x.ipynb` | Urban grid, mobility, link state, OFDM, and persistence | < 15 s |
| `08_geometric_cdl_evolution.ipynb` | Persistent-scatterer multicarrier CDL evolution, transitions, continuation, and persistence | < 15 s |

The examples use small tensors so the whole suite remains suitable for routine
CPU validation. They are tutorials and physical sanity checks, not formal
standards-conformance results. Standards resources that are pending independent
review are identified inside the relevant notebooks.
