# Contributing

Channel Dataset 0.1.x is wireless-focused. Preserve the split between typed
configuration, random realization sampling, deterministic signal transforms,
and persistence. The deeper architecture rules are in [AGENTS.md](AGENTS.md)
and [docs/architecture.md](docs/architecture.md).

## Development setup

Use Python 3.11-3.14 in an isolated virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,wireless]"
```

Use `.[dev,wireless-numpy]` if PyTorch is unavailable. The NumPy-only CI job
does not collect a claim about PyTorch behavior; PyTorch has separate CPU and
combined-install jobs.

## Change workflow

1. Read the relevant model guide and capability limitations.
2. Put physical defaults in a versioned YAML/JSON resource, not Python source.
3. Parse configuration into a frozen typed object at the model boundary.
4. Inject or fork a labeled random stream; never use module-global randomness.
5. Keep realization generation separate from transforms and signal application.
6. Add validation, seeded replay, and physical-sanity tests.
7. Update the capability matrix and the relevant user guide when behavior changes.

Run the same core gates as CI:

```powershell
python -m ruff check src tests examples benchmarks
python -m pytest -m "not cuda"
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m build
python -m twine check dist/*
```

Tests tagged `cuda`, `performance`, or `slow` require an explicit suitable
environment. Do not weaken tolerances or skip markers merely to accommodate a
machine-specific dependency failure.

## Adding a model

A usable model needs all of the following:

- a `ChannelModel` implementation and typed request/realization contract;
- strict typed configuration parsing and a model-registry entry;
- an example configuration and, where useful, an executable script;
- deterministic seeded tests and at least one physical-sanity test;
- schema-versioned serialization when it introduces a persistent format;
- documentation of applicability, units, source revision, and limitations.

Use SI units internally and include units in public field names. Public APIs
need type hints and concise NumPy-style docstrings. Preserve configuration and
serialization compatibility within the 0.1.x line unless a documented migration
is included.
