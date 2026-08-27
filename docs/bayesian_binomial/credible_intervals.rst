Credible Intervals
==================

Overview
--------

This module computes Bayesian credible intervals for the lift between two binomial
proportions, as well as individual credible intervals for single proportions.

Two interval methods are supported:

- **Equal-tailed credible intervals** (``method="credible"``): symmetric quantile-based
  intervals where equal probability mass falls in each tail.
- **Highest Density Intervals** (``method="hdi"``): the shortest interval containing the
  desired probability mass. Preferred when the posterior is skewed, as it always returns
  the narrowest interval for a given confidence level.

When ``is_sample=True`` (the default), intervals are computed from Monte Carlo draws
from the Beta posterior. When ``is_sample=False``, a normal approximation via the delta
method is used instead — faster but less accurate for small samples or extreme
proportions. Note that under the normal approximation, both methods return the same
symmetric interval.

The lift can be computed in either relative (``(B - A) / A``) or absolute (``B - A``)
terms.

Example
-------

.. code-block:: python

   from ab_test.bayesian_binomial.credible_intervals import (
       credible_interval,
       individual_credible_interval,
   )

   # Lift credible interval (HDI)
   lb, ub = credible_interval(
       successes=[100, 130], trials=[1000, 1000],
       prior_alphas=[1, 1], prior_betas=[1, 1],
       confidence_level=0.95, lift="relative", method="hdi",
   )
   print(f"95% HDI for relative lift: [{lb:.4f}, {ub:.4f}]")

   # Individual cell credible interval
   lb, ub = individual_credible_interval(
       s=130, n=1000, confidence_level=0.95,
       prior_alpha=1, prior_beta=1, method="credible",
   )
   print(f"95% credible interval for treatment: [{lb:.4f}, {ub:.4f}]")

API Reference
-------------

.. automodule:: ab_test.bayesian_binomial.credible_intervals
   :members:
