# fincurve

[![tests](https://github.com/yz12993-ux/fincurve/actions/workflows/tests.yml/badge.svg)](https://github.com/yz12993-ux/fincurve/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![demo](https://img.shields.io/badge/demo-GitHub%20Pages-2a78d6.svg)](https://yz12993-ux.github.io/fincurve/)

**Which functional form does your financial data actually follow?** fincurve profiles the data, chooses the noise model and validation scheme, fits 32 candidate forms — from Nelson–Siegel and SVI to hazard curves and power laws — and ranks them by **held-out likelihood**. It also knows when *not* to fit a curve.

**[Live demo →](https://yz12993-ux.github.io/fincurve/)** Four case studies with known ground truth, the full scoreboard, benchmarks and the method, all generated from real runs of the library.

```bash
pip install git+https://github.com/yz12993-ux/fincurve
```

## Quick start

```python
from fincurve import analyze

report = analyze(df["tenor"], df["yield"])                              # one x    → curve mode
report = analyze(df[["leverage", "maturity", "rating"]], df["spread"])  # features → additive shapes
report = analyze(df["daily_return"])                                    # sample   → distribution mode

report.summary()          # plain-language report
report.plot("out.png")    # data, top curves, residuals, ranking
report.ranking            # every form: ΔCV-NLL, paired SE, tie flag, AIC
report.recommended        # the chosen model object
report.predict(new_x)
report.warnings           # what to be careful about
```

Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`. A small C extension compiles during installation when a compiler is available; without one, fincurve falls back to NumPy automatically (set `FINCURVE_NO_C=1` to force the fallback).

## What it decides for you

| Decision | Evidence | Outcome |
|---|---|---|
| Target type | value range, integers, {0, 1} | continuous · positive · proportion · binary · count |
| Likelihood | target type; noise vs level; count dispersion | Gaussian, log-normal, binomial or Poisson — additive and multiplicative noise compete when unclear |
| Time ordering | datetime x, time-like name, or an equally spaced random-walk-like series | rolling-origin cross-validation (never train on the future) |
| Random-walk paths | AR(1), lag-1 of differences, variance ratios at q = 2, 4, 8, residual autocorrelation around a smoother | curve recommendation withheld; log returns analysed instead |
| Validation | groups, ordering, sample size | grouped K-fold · rolling origin · leave-one-out (n < 20) · shuffled 5-fold |
| Finance context | x name: tenor, strike, yield, score, spot | reports where the specialised form (Nelson–Siegel, SVI, DCF…) ranked |
| Leakage | constant columns, row numbers, IDs, high-cardinality categories | removed before fitting |

## How a winner is chosen

1. **Score** — mean held-out negative log-likelihood per row. Gaussian, log-normal, binomial and Poisson models share one scale.
2. **Tie rule** — a model ties with the best when its gap is within **2 paired standard errors** *and* below **0.1 nats per row** (≈10% RMSE under Gaussian errors). The second condition stops small samples from calling everything equal.
3. **Recommendation** — the tied model with the fewest parameters.
4. **Guardrails** — warnings for peak shapes fitted to monotone data, tied models that disagree once extrapolated, heavy-tailed residuals, collinear features, and volatility clustering in return samples.

In multi-feature mode every numeric feature starts linear; a greedy search tries 11 shapes (quadratic, cubic, log, sqrt, reciprocal, power, exponential, sigmoid, peak, hinge) and accepts one only if AIC drops by at least 4, then tests product interactions (AIC drop ≥ 10). **The search is repeated inside every cross-validation fold**, so the score is not flattered by it, and fold-to-fold effect-curve correlation shows whether a shape is real.

## Performance

Version 0.2.0 rebuilt the numerical core without changing any result:

- **Variable projection.** 30 of the 32 forms are separable, mean = B(x, θ)·β. The linear coefficients β are solved exactly — weighted least squares, Gauss–Newton for log-normal errors, IRLS for binomial and Poisson targets — so the optimiser only searches the 0–2 nonlinear parameters θ. Purely linear forms become a single matrix solve. The two non-separable forms (Heston term structure, SVI) keep general least squares.
- **C kernel.** The tricube local-linear smoother (data profiling, the non-parametric reference, the random-walk test) walks a sorted window in C instead of partitioning the whole sample for every point.
- **KD-tree reference.** The multi-feature k-nearest-neighbour benchmark uses scipy's `cKDTree` with parallel queries.
- **Optional processes.** `analyze(..., n_jobs=-1)` spreads candidates (curve mode) or nested folds (multi-feature mode) across cores. Worker start-up costs about a second, so it pays off only for heavy workloads; on macOS and Windows call it under `if __name__ == "__main__":`.

Benchmark on an Apple M2, single process, same seed ([`benchmarks/bench.py`](benchmarks/bench.py), results in [`benchmarks/results.json`](benchmarks/results.json)):

| Workload | v0.1.0 | v0.2.0 | Speed-up |
|---|---:|---:|---:|
| additive model, 10,000 rows × 5 features (nested CV) | 56.8 s | 3.0 s | 19.1× |
| credit spreads, 1,500 × 5 (nested CV) | 6.3 s | 0.46 s | 13.8× |
| loan defaults, 4,000 × 3 (nested CV) | 10.2 s | 1.1 s | 9.6× |
| NAV path, n = 121 (rolling-origin CV) | 3.3 s | 0.45 s | 7.3× |
| vol smile, n = 25 | 0.97 s | 0.20 s | 4.8× |
| yield curve, n = 13 (leave-one-out) | 3.0 s | 0.65 s | 4.6× |
| market impact, n = 90 (two noise models) | 5.6 s | 1.2 s | 4.5× |
| curve, n = 3,000 (two noise models) | 13.4 s | 3.4 s | 4.0× |
| trade counts, n = 400 (Poisson) | 7.0 s | 2.2 s | 3.1× |
| default rate, n = 16 (binomial, leave-one-out) | 6.2 s | 5.4 s | 1.2× |
| return distribution, n = 2,520 | 0.45 s | 0.48 s | unchanged |
| **total** | **113.1 s** | **18.5 s** | **6.1×** |

All eleven workloads return the same recommendation as v0.1.0. The C smoother is 14× faster than the NumPy reference on 3,000 points, and the two implementations agree to 1e-14. The binomial leave-one-out case gains least: proportions fitted on the identity scale are clipped into [0, 1], which makes the likelihood flat, and the 16 folds dominate.

## Candidate library (32 forms)

| Family | Forms |
|---|---|
| Baseline, polynomial | constant, linear, quadratic, cubic |
| Growth / decay | exponential, exponential approach (mean reversion), double exponential |
| Power law | power, power + offset, tempered power |
| Diminishing returns | log, shifted log, square root, hyperbolic, rational |
| Saturation, sigmoid | Michaelis–Menten, Hill, logistic, Gompertz |
| Peak, piecewise, periodic | Gaussian peak, Lorentzian peak, hinge, soft hockey stick, sine + trend |
| Term structure | Nelson–Siegel, Svensson, Heston ATM-vol term structure |
| Volatility | SVI smile |
| Fixed income | DCF price (price–yield convexity) |
| Credit | constant-hazard and Weibull cumulative default curves |
| Options | Black–Scholes price-vs-underlying shape |

Every form carries bounds, domain rules (e.g. x > 0) and a search grid for its nonlinear parameters. For binary, proportion and count targets, logit- and log-link GLM versions are added automatically. Distribution mode compares 12 distributions (normal, Student-t, Laplace, logistic, skew-normal, Johnson SU, NIG, log-normal, gamma, Weibull, inverse Gaussian, exponential) by AIC and reports tail-quantile errors.

## Results on data with a known answer

All eleven simulated scenarios, same seed, default settings:

| Scenario | Ground truth | Identified | Verdict |
|---|---|---|---|
| Yield curve (n = 13) | Nelson–Siegel | exponential approach | ✗ truth ranked #7; the report warns the sample is too small |
| ATM vol term structure (n = 10) | Heston variance reversion | shifted log | ✗ truth ranked #3; the report warns the sample is too small |
| Volatility smile | SVI | SVI | ✓ |
| Bond price vs yield | semi-annual DCF | exponential approach | ≈ DCF tied, simpler model preferred |
| Market impact | 140·√q, multiplicative noise | square root, multiplicative | ✓ free exponent fits 0.509 |
| Default rate vs score | logistic | logistic (binomial) | ✓ |
| Intraday trade counts | Poisson, log-quadratic | log-quadratic (Poisson) | ✓ |
| Daily returns | Student-t (ν = 4) | Student-t | ✓ normal is 221 AIC worse |
| Portfolio NAV path | geometric Brownian motion | random walk → normal log returns | ✓ curve withheld |
| Credit spreads (5 features) | log-additive: convex leverage, saturating maturity, rating | same shapes; issuer ID removed | ✓ |
| Loan defaults (binary) | sigmoid threshold + linear utilization | same; row-number column removed | ✓ |

The random-walk test raised no false alarms on 200 smooth-curve, 200 linear-trend and 200 bond-curve series, and detected 95% of GBM paths and 80% of AR(1) φ = 0.95 series.

## Limitations

- Nonlinear parameters are still found by bounded least squares from a grid of starts, so a local optimum remains possible, especially for the non-separable SVI and Heston forms on a handful of points.
- With a dozen observations the ranking itself is unstable — the report says so, but more data is the real fix.
- Scores measure accuracy inside the observed range. Tied forms can diverge when extrapolated; the report estimates that disagreement, and the advice is not to extrapolate.
- Time series get rolling-origin validation but no autocorrelated-error model; distribution mode assumes independent draws and only warns about volatility clustering.
- Multi-feature mode is additive with a few product interactions; richer interactions need tree ensembles or similar models.

## Project layout

```
fincurve/
  core.py          analyze(): profiling, automatic decisions, warnings
  library.py       the 32 candidate forms (formula, basis, bounds, domain, search grid)
  fitting.py       variable projection, IRLS / Gauss–Newton inner solvers, general least squares
  families.py      Gaussian, log-normal, binomial and Poisson likelihoods
  kernels.py       compiled kernels with NumPy fallbacks
  _kernels.c       C implementation of the local-linear smoother
  parallel.py      optional process-based n_jobs
  profile.py       shape, noise, random-walk test, finance-context detection
  univariate.py    curve mode: fitting, cross-validation, extrapolation disagreement
  additive.py      multi-feature mode: shape search, interactions, nested CV
  distribution.py  distribution mode: 12 distributions, tail errors, clustering test
  scoring.py       CV splits, paired standard errors, tie rule
  report.py        report object and plain-language summary
  plotting.py      figures for the three modes
benchmarks/        wall-clock benchmark and recorded results
examples/          simulated scenarios with known ground truth, demo script and outputs
site/              exports real runs to the showcase page data
docs/              showcase page (GitHub Pages)
tests/run_tests.py 21 regression and numerical-equivalence tests
```

## Development

```bash
python setup.py build_ext --inplace
```

```bash
python tests/run_tests.py
```

```bash
python benchmarks/bench.py
```

```bash
python site/build_site_data.py
```

CI runs the tests on Linux and macOS twice — with the C extension and with the NumPy fallback — and checks that the package installs.

## License

MIT © 2026 Yuang Zhang
