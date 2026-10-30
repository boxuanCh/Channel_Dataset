# Statistical certification tests

The tests in this directory validate ensemble behavior rather than seeded
sample equality. They are selected with:

```powershell
python -m pytest -m statistical
```

## Confidence policy

- Means use a five-standard-error limit derived from the theoretical variance.
- Gaussian standard deviations use the asymptotic sampling error
  `sigma / sqrt(2 * (n - 1))`, with a small numerical floor.
- Bernoulli rates use five binomial standard errors, with a 0.005 absolute
  floor near probabilities zero and one.
- Correlations use a five-standard-error Fisher-transform approximation, with
  a 0.035 floor for finite sum-of-sinusoids approximations.
- Complex Gaussian fading is checked with power mean, power variance,
  exponential quantiles, I/Q balance, mean, and pseudo-covariance.
- Profile power tests use five or six theoretical standard errors plus a small
  absolute floor for very weak taps.

These limits are deliberately wider than a conventional 95% interval so a
correct stochastic implementation is stable in CI while still rejecting
material model errors. A failing statistical test must not be repaired by
loosening its margin without a documented derivation and an investigation of
the underlying model.

## Coverage tiers

The normal suite uses thousands of realizations and runs on every pull request.
Very large Monte Carlo calibration against external tools belongs in the
`slow` and `reference` tiers and is not a substitute for these self-contained
checks. Standard expectations should come from reviewed resources or explicit
equations; model diagnostics are not authoritative reference data.

The normal certification matrix includes:

- AWGN, Rayleigh block fading, and uncorrelated/correlated flat MIMO fading;
- every terrestrial TDL/CDL profile and every RMa, UMa, UMi, InH, and InF
  LOS/NLOS scenario;
- terrestrial system-level ray generation, blockage, and snapshot spatial
  consistency;
- every NTN TDL/CDL profile, flat-fading state, environment, elevation,
  frequency band, and D1--D5 deployment kernel;
- every IEEE 802.11 TGn profile plus TGac, TGah, and TGax scenario, link,
  deployment, LOS/NLOS, and topology-aware large-scale variant;
- TR 37.885 mobility, link state, fixed CDL, and stochastic LOS/NLOSv/NLOS
  system channels; and
- geometric-CDL spatial fields, both standard namespaces, all exposed
  scenario variants, and CDL-A--E fixed-profile initialization.
