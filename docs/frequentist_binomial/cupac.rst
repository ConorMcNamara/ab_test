CUPAC / MLRATE (Variance Reduction)
====================================

This module provides two variance-reduction methods for A/B tests with
per-user covariate data:

* **CUPAC** (Controlled-experiment Using Pre-experiment data with Adjusted
  Covariates) fits OLS on control-group covariates and applies the CUPED
  adjustment.
* **MLRATE** (Machine Learning Regression-Adjusted Treatment Effect) replaces
  OLS with any scikit-learn-compatible estimator and uses K-fold cross-fitting
  to ensure valid inference regardless of model complexity.

Both methods adjust outcomes via the CUPED framework (``theta = Cov(y, y_hat) /
Var(y_hat)``) and estimate the treatment effect with HC2 robust standard
errors. Variance is reduced by a factor of ``(1 - R^2)``, which translates
directly to higher statistical power and smaller required sample sizes.

Unlike :class:`~ab_test.frequentist_binomial.contingency.ContingencyTable`,
which works with aggregate counts (successes and trials),
:class:`~ab_test.frequentist_binomial.cupac.CupacExperiment` accepts per-user
DataFrames with outcome, treatment indicator, and covariate columns. Both
pandas and polars DataFrames are accepted.

All covariate columns must be numeric. Nominal categorical features should be
one-hot encoded before being passed in, since OLS interprets numeric values as
continuous and will impose a spurious ordinal relationship on label-encoded
categories.

The module also provides power calculation wrappers that account for the
expected variance reduction from either method.

CUPAC Usage
-----------

.. code-block:: python

   from ab_test.frequentist_binomial.cupac import (
       CupacExperiment,
       cupac_adjusted_power,
   )
   from ab_test.frequentist_binomial.power_calculations import abtest_power

   # Analyze an experiment with OLS covariate adjustment
   exp = CupacExperiment(
       data=df,
       outcome_col="converted",
       treatment_col="group",
       covariate_cols=["pre_visits", "days_since_signup"],
       control_label="control",
       treatment_label="treatment",
   )
   print(exp.analyze())
   print(f"Variance reduction: {exp.variance_reduction:.1%}")

   # Power calculation with expected R-squared
   power = abtest_power(
       [1000, 1000], 0.10, 0.20,
       power=cupac_adjusted_power(r_squared=0.3),
   )

MLRATE Usage
------------

MLRATE trains on all groups (control + treatment) via cross-fitting, so each
unit's prediction is out-of-fold and the standard HC2 inference remains valid
even with flexible models like gradient boosting or random forests.

Requires ``scikit-learn``: install with ``pip install abtest-analysis[sklearn]``.

.. code-block:: python

   from sklearn.ensemble import GradientBoostingClassifier
   from ab_test.frequentist_binomial.cupac import CupacExperiment

   exp = CupacExperiment(
       data=df,
       outcome_col="converted",
       treatment_col="group",
       covariate_cols=["pre_visits", "pre_pageviews", "days_since_signup"],
       control_label="control",
       treatment_label="treatment",
       method="mlrate",
       estimator=GradientBoostingClassifier(n_estimators=100),
       n_folds=5,
   )
   print(exp.analyze())
   print(f"Variance reduction: {exp.variance_reduction:.1%}")

API Reference
-------------

.. automodule:: ab_test.frequentist_binomial.cupac
   :members:
   :exclude-members: _ols_fit, _hc2_standard_errors
