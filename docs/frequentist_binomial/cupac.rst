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

Any estimator with ``fit()`` and ``predict()`` methods works. If the estimator
also has ``predict_proba()`` (classifiers), the positive-class probability is
used automatically — this gives continuous predictions that correlate better
with the binary outcome than discrete 0/1 class labels.

.. code-block:: python

   from sklearn.ensemble import RandomForestClassifier
   from xgboost import XGBClassifier
   from lightgbm import LGBMClassifier

   # All of these work as MLRATE estimators
   CupacExperiment(..., method="mlrate", estimator=RandomForestClassifier())
   CupacExperiment(..., method="mlrate", estimator=XGBClassifier())
   CupacExperiment(..., method="mlrate", estimator=LGBMClassifier(verbose=-1))

When to Use CUPAC vs MLRATE
----------------------------

**Use CUPAC** (the default) when:

* You have a small number of numeric covariates with roughly linear
  relationships to the outcome.
* You want zero additional dependencies beyond numpy/scipy.
* OLS already captures most of the covariate signal.

**Use MLRATE** when:

* You have many covariates or expect nonlinear/interaction effects.
* You have a large sample (thousands of users per group) so the flexible
  model can learn reliably within each cross-fitting fold.
* You are willing to add scikit-learn (or XGBoost/LightGBM) as a dependency.

For binary outcomes, the Bernoulli noise floor limits the maximum variance
reduction from any model. The advantage of flexible models over OLS is most
pronounced when covariates have strong nonlinear relationships with the outcome
and sample sizes are large.

How Cross-Fitting Works
-----------------------

CUPAC fits its OLS model on control-group data only, then predicts for all
users. This is safe because OLS has low capacity — it cannot overfit. But a
flexible model (random forest, gradient boosting) trained on all data and
predicting on the same data would memorise outcomes, making the CUPED
adjustment over-correct and biasing the treatment effect toward zero.

MLRATE avoids this via K-fold cross-fitting (Guo et al., 2021):

1. Shuffle all users and split into K folds (default K = 5).
2. For each fold, train the estimator on the other K − 1 folds and predict
   outcomes for the held-out fold.
3. Assemble the full prediction vector — every user's prediction comes from a
   model that never saw their outcome.
4. Apply the standard CUPED adjustment: ``y_adj = y − θ (ŷ − mean(ŷ))``.
5. Estimate the ATE from adjusted outcomes with HC2 robust standard errors.

Because each prediction is out-of-fold, the adjustment cannot inflate
``θ`` beyond its true value, and the resulting ATE is unbiased regardless of
model complexity. The HC2 confidence intervals also maintain nominal coverage.

API Reference
-------------

.. automodule:: ab_test.frequentist_binomial.cupac
   :members:
   :exclude-members: _ols_fit, _hc2_standard_errors
