# ab-test

[![CI](https://github.com/ConorMcNamara/ab_test/actions/workflows/ci.yml/badge.svg)](https://github.com/ConorMcNamara/ab_test/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for designing, running, and analyzing A/B tests on binomial metrics (conversion rates, click-through rates, etc.).

## Features

- **Statistical tests** — Score (Rao), Likelihood Ratio, Z, Fisher's Exact, Barnard's Exact, Boschloo's Exact, and several power-divergence variants (Freeman-Tukey, Neyman, Cressie-Read, Modified Log-Likelihood)
- **Confidence intervals** — Binary search (score/LRT/z-test inversion), Wilson, Agresti-Coull, Jeffreys, Clopper-Pearson, Wald, and Delta method
- **Power & sample size** — Power calculations, minimum detectable lift, and required sample size via binary search
- **Lift types** — Relative, absolute, incremental, ROAS, and revenue lift
- **`ContingencyTable`** — A chainable class that ties everything together, with DataFrame export, serialization, and plotting

## Requirements

- Python >= 3.10
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
from ab_test.binomial.contingency import ContingencyTable

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
from ab_test.binomial.power_calculations import abtest_power, minimum_detectable_lift, required_sample_size

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
from ab_test.binomial.stats_tests import ab_test

p_value = ab_test(trials=[1_000, 1_000], successes=[100, 130], method="likelihood")
```

### Confidence interval methods

```python
from ab_test.binomial.confidence_intervals import wilson_interval, confidence_interval

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
