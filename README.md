# ab-test

[![CI](https://github.com/ConorMcNamara/ab_test/actions/workflows/ci.yml/badge.svg)](https://github.com/ConorMcNamara/ab_test/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ConorMcNamara/ab_test/branch/main/graph/badge.svg)](https://codecov.io/gh/ConorMcNamara/ab_test)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with zuban](https://img.shields.io/badge/type%20checked-zuban-blue)](https://zubanls.com)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A Python library for designing, running, and analyzing A/B tests on binomial metrics (conversion rates, click-through rates, etc.). It provides both frequentist and Bayesian approaches with full power analysis, sequential testing, and covariate-adjusted variance reduction.

## Features

| Category | Highlights | Docs |
|---|---|---|
| **Contingency tables** | Chainable builder, DataFrame export, serialization, plotting | [frequentist](docs/frequentist_binomial/contingency.rst) · [bayesian](docs/bayesian_binomial/contingency.rst) |
| **Statistical tests** | Score, LRT, Z, Fisher, Barnard, Boschloo, power-divergence variants | [docs](docs/frequentist_binomial/stats_tests.rst) |
| **Confidence / credible intervals** | Wilson, Agresti-Coull, Jeffreys, Clopper-Pearson, HDI, binary-search inversion | [frequentist](docs/frequentist_binomial/confidence_intervals.rst) · [bayesian](docs/bayesian_binomial/credible_intervals.rst) |
| **Power & sample size** | Power, MDL, required n — frequentist and Bayesian (P(B>A) or expected loss) | [frequentist](docs/frequentist_binomial/power_calculations.rst) · [bayesian](docs/bayesian_binomial/power_calculations.rst) |
| **Sequential testing** | mSPRT always-valid p-values and confidence sequences; group sequential designs with O'Brien-Fleming, Pocock and power-family alpha spending | [mSPRT](docs/frequentist_binomial/msprt.rst) · [group sequential](docs/frequentist_binomial/gst.rst) |
| **Variance reduction** | CUPAC (OLS), Lin (treatment-by-covariate interactions), and MLRATE (any sklearn estimator, K-fold cross-fitting) | [docs](docs/frequentist_binomial/cupac.rst) |
| **Stratified analysis** | CMH test, Breslow-Day, MH odds ratio, Bayesian inverse-variance pooling | [frequentist](docs/frequentist_binomial/stratified.rst) · [bayesian](docs/bayesian_binomial/stratified.rst) |
| **Diff-in-diff** | Multi-period heterogeneity testing, pairwise comparisons, Cochran's Q | [frequentist](docs/frequentist_binomial/diff_in_diff.rst) · [bayesian](docs/bayesian_binomial/diff_in_diff.rst) |
| **Bayesian inference** | P(B > A), expected loss, ROPE analysis, lift probability thresholds | [docs](docs/bayesian_binomial/stats_tests.rst) |
| **Equivalence testing** | TOST (two one-sided tests) and Bayesian ROPE equivalence | [frequentist](docs/frequentist_binomial/equivalence.rst) · [bayesian](docs/bayesian_binomial/equivalence.rst) |
| **Randomization inference** | Assumption-free permutation p-values, individual- and cluster-level, with parallel Monte Carlo | [docs](docs/frequentist_binomial/randomization_inference.rst) |
| **Multiple testing** | Bonferroni, Sidak, Holm (FWER), Benjamini-Hochberg (FDR) | [docs](docs/corrections.rst) |
| **Cluster-randomized trials** | Frequentist cluster-summary Welch test with ICC and design effect; Bayesian beta-binomial hierarchical model (posterior integrates over the ICC) and simulation-based assurance | [frequentist](docs/frequentist_binomial/cluster.rst) · [bayesian](docs/bayesian_binomial/cluster.rst) |
| **Diagnostics** | Sample ratio mismatch (SRM), time-trend (novelty/primacy) detection, and placebo tests | [docs](docs/diagnostics.rst) |
| **Lift types** | Relative, absolute, incremental, ROAS, CPA, and revenue (CPA is not available for ROPE, equivalence, or difference-in-differences) | — |

## Installation

Requires Python >= 3.11.

```bash
pip install git+https://github.com/ConorMcNamara/ab_test.git
```

Optional extras:

```bash
pip install "abtest-analysis[sklearn] @ git+https://github.com/ConorMcNamara/ab_test.git"   # MLRATE variance reduction (scikit-learn)
```

Other extras: `pyspark`, `modin`, `ibis`, and `narwhals` for DataFrame export.

### Development setup

```bash
git clone https://github.com/ConorMcNamara/ab_test.git
cd ab_test
uv sync --extra dev
```

Install extras with `uv sync --extra <name>` (requires [uv](https://docs.astral.sh/uv/)).

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

```text
+---------------------+-----------+
| Statistic           | Value     |
+=====================+===========+
| Metric              | relative  |
+---------------------+-----------+
| Metric Name         | purchases |
+---------------------+-----------+
| Control             | 10.0%     |
+---------------------+-----------+
| Treatment           | 13.0%     |
+---------------------+-----------+
| Lift                | 30.0%     |
+---------------------+-----------+
| Conf. Int. Lower ** | 1.79%     |
+---------------------+-----------+
| Conf. Int. Upper ** | 66.13%    |
+---------------------+-----------+
| p-value             | 0.0355*   |
+---------------------+-----------+
* next to the p-value means it's statistically significant at the 5% level
** 95% Confidence Interval
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
| `"cpa"` | Cost per acquisition (`spend / incremental_conversions`). Summarized on the incremental scale and transformed, since CPA has no posterior mean; not supported for ROPE, equivalence, or difference-in-differences, where `"roas"` is the per-dollar alternative |
| `"revenue"` | Incremental revenue (`incremental_conversions × msrp`) |

### `test_method`

`"score"`, `"likelihood"`, `"z"`, `"wald"`, `"fisher"`, `"barnard"`, `"boschloo"`, `"modified_likelihood"`, `"freeman-tukey"`, `"neyman"`, `"cressie-read"`, `"msprt"`

### `conf_int_method`

`"binary_search"`, `"wilson"`, `"jeffrey"`, `"agresti-coull"`, `"clopper-pearson"`, `"wald"`, `"delta"`

### `cred_int_method`

`"credible"` (equal-tailed), `"hdi"` (highest density interval)

### Color palettes (`.plot`)

`"ibm"`, `"wong"`, `"ito"`, `"tol"`, `"tol_bright"`, `"tol_vibrant"`, `"tol_muted"`, `"tol_light"`

## Acknowledgements

Special thanks to [abtesting-public](https://github.com/rwilson4/abtesting-public) by [@rwilson4](https://github.com/rwilson4) for being the inspiration for this package.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
