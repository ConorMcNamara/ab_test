# ab-test

[![CI](https://github.com/ConorMcNamara/ab_test/actions/workflows/ci.yml/badge.svg)](https://github.com/ConorMcNamara/ab_test/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ConorMcNamara/ab_test/branch/main/graph/badge.svg)](https://codecov.io/gh/ConorMcNamara/ab_test)
[![Python](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with zuban](https://img.shields.io/badge/type%20checked-zuban-blue)](https://zubanls.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for designing, running, and analyzing A/B tests on binomial metrics (conversion rates, click-through rates, etc.). It provides both frequentist and Bayesian approaches with full power analysis, sequential testing, and covariate-adjusted variance reduction.

## Features

| Category | Highlights | Docs |
|---|---|---|
| **Contingency tables** | Chainable builder, DataFrame export, serialization, plotting | [frequentist](docs/frequentist_binomial/contingency.rst) · [bayesian](docs/bayesian_binomial/contingency.rst) |
| **Statistical tests** | Score, LRT, Z, Fisher, Barnard, Boschloo, power-divergence variants | [docs](docs/frequentist_binomial/stats_tests.rst) |
| **Confidence / credible intervals** | Wilson, Agresti-Coull, Jeffreys, Clopper-Pearson, HDI, binary-search inversion | [frequentist](docs/frequentist_binomial/confidence_intervals.rst) · [bayesian](docs/bayesian_binomial/credible_intervals.rst) |
| **Power & sample size** | Power, MDL, required n — frequentist and Bayesian (P(B>A) or expected loss) | [frequentist](docs/frequentist_binomial/power_calculations.rst) · [bayesian](docs/bayesian_binomial/power_calculations.rst) |
| **Sequential testing** | mSPRT always-valid p-values and confidence sequences | [docs](docs/frequentist_binomial/msprt.rst) |
| **Variance reduction** | CUPAC (OLS) and MLRATE (any sklearn estimator, K-fold cross-fitting) | [docs](docs/frequentist_binomial/cupac.rst) |
| **Stratified analysis** | CMH test, Breslow-Day, MH odds ratio, Bayesian inverse-variance pooling | [frequentist](docs/frequentist_binomial/stratified.rst) · [bayesian](docs/bayesian_binomial/stratified.rst) |
| **Diff-in-diff** | Multi-period heterogeneity testing, pairwise comparisons, Cochran's Q | [frequentist](docs/frequentist_binomial/diff_in_diff.rst) · [bayesian](docs/bayesian_binomial/diff_in_diff.rst) |
| **Bayesian inference** | P(B > A), expected loss, ROPE analysis, lift probability thresholds | [docs](docs/bayesian_binomial/stats_tests.rst) |
| **Multiple testing** | Bonferroni, Sidak, Holm (FWER), Benjamini-Hochberg (FDR) | [docs](docs/corrections.rst) |
| **Diagnostics** | Sample ratio mismatch (SRM) detection | [docs](docs/diagnostics.rst) |
| **Lift types** | Relative, absolute, incremental, ROAS, and revenue — all methods | — |

## Installation

```bash
git clone https://github.com/ConorMcNamara/ab_test.git
cd ab_test
uv sync --extra dev
```

Optional extras:

```bash
uv sync --extra sklearn    # MLRATE variance reduction (scikit-learn)
uv sync --extra pyspark    # PySpark DataFrame export
uv sync --extra modin      # modin DataFrame export
uv sync --extra ibis       # ibis DataFrame export
uv sync --extra narwhals   # narwhals DataFrame export
```

Requires Python >= 3.13 and [uv](https://docs.astral.sh/uv/).

## Quick Start

```python
from ab_test.frequentist_binomial.contingency import ContingencyTable

ct = (
    ContingencyTable(name="Homepage Redesign", metric_name="purchases")
    .add("Control", successes=100, trials=1_000)
    .add("Treatment", successes=130, trials=1_000)
)
print(ct.analyze(lift="relative", test_method="score", alpha=0.05))
```

```python
from ab_test.bayesian_binomial.contingency import BayesianContingencyTable

bct = (
    BayesianContingencyTable(name="Homepage Redesign", metric_name="purchases")
    .add("Control",   successes=100, trials=1_000, alpha=1.0, beta=1.0)
    .add("Treatment", successes=130, trials=1_000, alpha=1.0, beta=1.0)
)
print(bct.analyze(lift="relative"))
```

See the [docs/](docs/) directory for detailed usage examples and API reference for each module.

## Reference Options

### `lift`

| Value | Interpretation |
|---|---|
| `"relative"` | `(p_treatment - p_control) / p_control` |
| `"absolute"` | `p_treatment - p_control` |
| `"incremental"` | Incremental conversions normalized to equal group sizes |
| `"roas"` | Return on ad spend (`incremental_conversions / spend`) |
| `"revenue"` | Incremental revenue (`incremental_conversions × msrp`) |

### `test_method`

`"score"`, `"likelihood"`, `"z"`, `"fisher"`, `"barnard"`, `"boschloo"`, `"modified_likelihood"`, `"freeman-tukey"`, `"neyman"`, `"cressie-read"`, `"msprt"`

### `conf_int_method` / `cred_int_method`

`"binary_search"`, `"wilson"`, `"jeffrey"`, `"agresti-coull"`, `"clopper-pearson"`, `"wald"`, `"delta"`, `"hdi"`, `"equal_tailed"`

### Color palettes (`.plot`)

`"ibm"`, `"wong"`, `"ito"`, `"tol"`, `"tol_bright"`, `"tol_vibrant"`, `"tol_muted"`, `"tol_light"`

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
