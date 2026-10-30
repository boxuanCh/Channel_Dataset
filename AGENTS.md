# Channel Dataset Contributor Guide

## Project purpose

This repository provides an extensible, object-oriented framework for physical-layer communication channel models. It is intended to grow across wireless/RF, acoustic, underwater acoustic, optical fiber, free-space optical, and other commonly used channel families.

The framework separates:

- reusable physical concepts such as paths/rays, clusters, delays, angles, Doppler shifts, and impulse responses;
- channel-family behavior such as propagation, loss, dispersion, fading, and noise;
- model-specific configuration and validation;
- simulation orchestration, reproducibility, and result serialization.

## Architecture rules

1. Put stable, domain-wide value objects in `src/channel_dataset/core/`.
2. Put channel-family implementations in `src/channel_dataset/channels/<family>/`.
3. Implement every usable channel model through the `ChannelModel` interface.
4. Keep model parameters out of source code. Store reusable defaults and named
   profiles under `src/channel_dataset/resources/`; keep runnable envelopes in
   `configs/`. Callers may provide alternate files at runtime.
5. Parse configuration files at the boundary, validate them into typed objects, and pass those objects into domain code. Do not pass unstructured dictionaries through the simulation core.
6. Prefer immutable dataclasses for physical value objects. State units in field names or documentation and use SI units internally.
7. Inject random-number generators. Do not use module-global random state.
8. Keep deterministic propagation effects separate from stochastic noise and fading when practical, so either can be tested or replaced independently.
9. Do not make one channel family depend directly on another. Cross-family concepts belong in `core` or `components`.
10. Add a schema version to persistent configurations and serialized outputs.

## Expected model lifecycle

Each model follows this flow:

1. Load a YAML/JSON document through the configuration layer.
2. Validate it into a family/model-specific configuration object.
3. Register the relevant models (or call `register_wireless_models`) and
   construct a `ChannelModel` via `ModelRegistry.create(envelope)`.
4. Create the model-specific request around a `ChannelSamplingRequest`; the
   latter owns execution controls and the injected random stream.
5. Call `model.realize(request)` to sample a fixed channel realization.
6. Transform that realization to a flat, time, or OFDM response as needed and
   apply it to a signal with a separate operator.
7. Serialize results with provenance: model name, config version, seed,
   environment, and software version.

Do not combine random realization generation with signal application. A sampled realization must be reusable for repeatable comparisons.

## Directory conventions

```text
configs/                       Runnable external configuration envelopes
docs/                          Architecture and design decisions
examples/                      Executable scripts and tutorial notebooks
src/channel_dataset/
  core/                        Shared interfaces and value objects
  config/                      File loading, schemas, and validation
  channels/                    Channel-family implementations
  components/                  Reusable propagation/noise components
  resources/                   Packaged profiles, schemas, and standards tables
  registry.py                  Model discovery and construction
tests/                         Unit and integration tests
```

The current release is wireless-focused. Acoustic, underwater, fiber, and
optical packages are namespace placeholders; do not imply that they contain
usable models. The supported wireless boundary and explicit deferrals live in
`docs/capabilities.md`.

## Quality bar

- Public APIs require type hints and concise docstrings.
- New models require configuration examples, validation tests, deterministic seeded tests, and at least one physical sanity test.
- Tests must compare quantities with suitable tolerances rather than exact floating-point equality.
- Dependencies in the core should remain small. Family-specific heavy dependencies should be optional extras.
- Preserve backward compatibility of configuration schemas when possible; otherwise add an explicit migration.

## Commands

After installing development dependencies:

```powershell
python -m pip install -e ".[dev,wireless]"
python -m ruff check src tests examples benchmarks
python -m pytest -m "not cuda"
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m build
python -m twine check dist/*
```

Use `.[dev,wireless-numpy]` when PyTorch is unavailable. A machine-specific
PyTorch import or CUDA failure is not evidence of a source failure; report it
separately and run only the backend selection the environment can support.
