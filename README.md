# ab-test

[![CI](https://github.com/ConorMcNamara/ab_test/actions/workflows/ci.yml/badge.svg)](https://github.com/ConorMcNamara/ab_test/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ConorMcNamara/ab_test/branch/main/graph/badge.svg)](https://codecov.io/gh/ConorMcNamara/ab_test)
[![Python](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with zuban](https://img.shields.io/badge/type%20checked-zuban-blue)](https://zubanls.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for designing, running, and analyzing A/B tests on binomial metrics (conversion rates, click-through rates, etc.).

## Features

- **Statistical tests** — Score (Rao), Likelihood Ratio, Z, Fisher's Exact, Barnard's Exact, Boschloo's Exact, and several power-divergence variants (Freeman-Tukey, Neyman, Cressie-Read, Modified Log-Likelihood)
- **Confidence intervals** — Binary search (score/LRT/z-test inversion), Wilson, Agresti-Coull, Jeffreys, Clopper-Pearson, Wald, and Delta method
- **Power & sample size** — Power calculations, minimum detectable lift, and required sample size via binary search
- **Lift types** — Relative, absolute, incremental, ROAS, and revenue lift
- **`ContingencyTable`** — A chainable class that ties everything together, with DataFrame export, serialization, and plotting
- **Bayesian inference** — Posterior sampling, P(B > A), expected loss, ROPE analysis, and lift probability thresholds

## Requirements

- Python >= 3.11
- [Poetry](https://python-poetry.org/) (for dependency management)

## Installation

```bash
git clone https://github.com/ConorMcNamara/ab_test.git
cd ab_test
poetry install --extras dev
```

## Quick Start

### Analyzing an experiment

```python
from ab_test.frequentist_binomial.contingency import ContingencyTable

ct = (
    ContingencyTable(name="Homepage Redesign", metric_name="purchases")
    .add("Control", successes=100, trials=1_000)
    .add("Treatment", successes=130, trials=1_000)
)

# Print the raw table
print(ct)

# Analyze relative lift with a score test and 95% CI
print(ct.analyze(lift="relative", test_method="score", alpha=0.05))
```

### Individual cell confidence intervals

```python
print(ct.analyze_individually(conf_int_method="wilson", alpha=0.05))
```

### Plotting results

```python
# Individual success rates with CIs
ct.analyze_individually()
ct.plot(is_individual=True, color="ibm")

# Comparative lift with CI
ct.analyze()
ct.plot(is_individual=False)
```

### Power and sample size

```python
from ab_test.frequentist_binomial.power_calculations import abtest_power, minimum_detectable_lift, required_sample_size

# Power for a given experiment design
power = abtest_power(group_sizes=[1_000, 1_000], baseline=0.10, alt_lift=0.20, lift="relative")
print(f"Power: {power:.1%}")  # → ~43%

# Minimum lift detectable at 80% power
mdl = minimum_detectable_lift(group_sizes=[1_000, 1_000], baseline=0.10, lift="relative")
print(f"MDL: {mdl:.1%}")  # → ~47%

# Sample size needed to detect a 20% relative lift at 80% power
n = required_sample_size(baseline=0.10, alt_lift=0.20, lift="relative")
print(f"Required n: {n:,}")
```

### Using a different statistical test

```python
from ab_test.frequentist_binomial.stats_tests import ab_test

p_value = ab_test(trials=[1_000, 1_000], successes=[100, 130], method="likelihood")
```

### Confidence interval methods

```python
from ab_test.frequentist_binomial.confidence_intervals import wilson_interval, confidence_interval

# Individual proportion CI
lb, ub = wilson_interval(s=100, n=1_000, alpha=0.05)

# Lift CI via binary search (most accurate)
lb, ub = confidence_interval(
    trials=[1_000, 1_000],
    successes=[100, 130],
    lift="relative",
    method="binary_search",
    alpha=0.05,
)
```

### Bayesian analysis

Use `BayesianContingencyTable` for a chainable high-level interface, `calculate_metrics` for an all-in-one result, or call the lower-level functions directly for more control.

```python
from ab_test.bayesian_binomial.contingency import BayesianContingencyTable

bct = (
    BayesianContingencyTable(name="Homepage Redesign", metric_name="purchases", spend=50_000, msrp=120.0)
    .add("Control",   successes=100, trials=1_000, alpha=1.0, beta=1.0)
    .add("Treatment", successes=130, trials=1_000, alpha=1.0, beta=1.0)
)

print(bct)                                  # grid table
print(bct.analyze(lift="relative"))         # relative lift + ROPE
print(bct.analyze(lift="revenue"))          # incremental revenue + ROPE
print(bct.analyze(lift="roas"))             # cost-per-acquisition + ROPE
bct.plot_individually()                     # posterior PDF chart
```

```python
import numpy as np
from ab_test.bayesian_binomial.stats_tests import calculate_metrics

# Relative / absolute lift
metrics = calculate_metrics(
    successes=np.array([100, 130]),
    trials=np.array([1_000, 1_000]),
    alphas=np.array([1.0, 1.0]),
    betas=np.array([1.0, 1.0]),
    n_samples=10_000,
    lift="relative",
    low_threshold=-0.01,
    high_threshold=0.01,
)

# Incremental / revenue / ROAS — pass spend and msrp as needed
metrics = calculate_metrics(
    successes=np.array([100, 130]),
    trials=np.array([1_000, 1_000]),
    alphas=np.array([1.0, 1.0]),
    betas=np.array([1.0, 1.0]),
    n_samples=10_000,
    lift="revenue",
    low_threshold=-5_000,
    high_threshold=5_000,
    msrp=120.0,
)
```

```python
from ab_test.bayesian_binomial.stats_tests import (
    probability_b_greater_than_a,
    expected_loss_b,
    calculate_rope,
    prob_lift_exceeds,
)
from ab_test.bayesian_binomial.utils import sample_beta

sample_a = sample_beta(s=100, n=1_000, alpha=1.0, beta=1.0, n_samples=10_000)
sample_b = sample_beta(s=130, n=1_000, alpha=1.0, beta=1.0, n_samples=10_000)

probability_b_greater_than_a(sample_a, sample_b)       # P(B > A)
expected_loss_b(sample_a, sample_b)                    # E[max(A - B, 0)]
prob_lift_exceeds(sample_a, sample_b, threshold=0.05)  # P(relative lift > 5%)
calculate_rope(sample_a, sample_b)                     # relative ROPE breakdown

# Incremental count ROPE (within ±500 conversions is negligible)
calculate_rope(sample_a, sample_b, lift="incremental", low=-500, high=500, trials=(1_000, 1_000))

# Revenue ROPE (within ±$5 000 is negligible)
calculate_rope(sample_a, sample_b, lift="revenue", low=-5_000, high=5_000, trials=(1_000, 1_000), msrp=120.0)

# ROAS / CPA ROPE (within ±$1 cost-per-acquisition is negligible)
calculate_rope(sample_a, sample_b, lift="roas", low=-1, high=1, trials=(1_000, 1_000), spend=50_000.0)
```

## API Reference

### `ContingencyTable`

| Method | Description |
|---|---|
| `.add(name, successes, trials)` | Add a cell; returns `self` for chaining |
| `.analyze(lift, test_method, conf_int_method, alpha, null_lift)` | Run significance test and compute lift + CI |
| `.analyze_individually(conf_int_method, alpha)` | CI for each cell independently |
| `.plot(is_individual, reverse_plot, color)` | Plotly dot-and-whisker chart |
| `.to_df(method, include_total)` | Export to pandas or polars DataFrame |
| `.to_list(include_total)` | Export to a plain list |
| `.to_numpy(include_total)` | Export to a NumPy array |
| `.serialize()` / `.deserialize(serial)` | JSON-compatible dict round-trip |

### `lift` options

| Value | Interpretation |
|---|---|
| `"relative"` | `(p_treatment - p_control) / p_control` |
| `"absolute"` | `p_treatment - p_control` |
| `"incremental"` | Incremental conversions normalized to equal group sizes |
| `"roas"` | Return on ad spend (`spend / incremental_conversions`) |
| `"revenue"` | Incremental revenue (`incremental_conversions × msrp`) |

### `test_method` options

`"score"`, `"likelihood"`, `"z"`, `"fisher"`, `"barnard"`, `"boschloo"`, `"modified_likelihood"`, `"freeman-tukey"`, `"neyman"`, `"cressie-read"`

### `conf_int_method` options

`"binary_search"`, `"wilson"`, `"jeffrey"`, `"agresti-coull"`, `"clopper-pearson"`, `"wald"`, `"delta"`

### Colorblind-friendly color palettes (`.plot`)

`"ibm"`, `"wong"`, `"ito"`, `"tol"`, `"tol_bright"`, `"tol_vibrant"`, `"tol_muted"`, `"tol_light"`

### `BayesianContingencyTable`

| Method | Description |
|---|---|
| `.add(cell_name, successes, trials, alpha, beta)` | Add a cell; returns `self` for chaining |
| `.analyze(lift, cred_int_method, confidence_level, is_sample, n_samples, low_threshold, high_threshold)` | Run Bayesian analysis and return a formatted summary with ROPE metrics |
| `.plot_individually(confidence_level, n_samples)` | Posterior PDF chart with HDI bars and P(B > A) title |
| `.to_df(method, include_total)` | Export to pandas, polars, PySpark, modin, ibis, or narwhals DataFrame |
| `.to_list(include_total)` | Export to a plain list |
| `.to_numpy(include_total)` | Export to a NumPy array |
| `.serialize()` / `.deserialize(serial)` | JSON-compatible dict round-trip |

Constructor: `BayesianContingencyTable(name, metric_name, spend=None, msrp=None)` — `spend` and `msrp` are required for `lift="roas"` and `lift="revenue"` respectively.

### Bayesian stats (`ab_test.bayesian_binomial`)

| Function | Description |
|---|---|
| `calculate_metrics(successes, trials, alphas, betas, n_samples, lift, low_threshold, high_threshold, spend, msrp)` | Draws posteriors and returns P(B > A), expected loss, and ROPE metrics in one call |
| `probability_b_greater_than_a(sample_a, sample_b)` | Proportion of posterior samples where B exceeds A |
| `expected_loss_b(sample_a, sample_b)` | E[max(A − B, 0)]: expected loss from choosing B |
| `calculate_rope(sample_a, sample_b, lift, low, high, trials, spend, msrp)` | Probability that lift falls within, above, or below the ROPE; supports all five lift types |
| `prob_lift_exceeds(sample_a, sample_b, lift, threshold)` | Probability that lift exceeds a given threshold |
| `sample_beta(s, n, alpha, beta, n_samples)` | Draw posterior samples from Beta(alpha + s, beta + n − s) |

### Bayesian `lift` options

| Value | Unit | Extra args required |
|---|---|---|
| `"relative"` | rate ratio `(B − A) / A` | — |
| `"absolute"` | rate difference `B − A` | — |
| `"incremental"` | count difference `(B − A) × max(trials)` | `trials` |
| `"revenue"` | incremental revenue `(B − A) × max(trials) × msrp` | `trials`, `msrp` |
| `"roas"` | CPA difference `spend/A_count − spend/B_count`; positive = B cheaper | `trials`, `spend` |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
