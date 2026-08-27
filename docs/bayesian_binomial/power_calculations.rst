Bayesian Power & Sample Size
============================

Overview
--------

This module provides simulation-based Bayesian power analysis for binomial A/B tests.
Unlike frequentist power calculations that rely on closed-form formulas, these functions
simulate many experiments under the alternative hypothesis and measure how often the
decision criterion is met.

Power can be defined by two criteria:

- **P(B > A) criterion** (:func:`~ab_test.bayesian_binomial.power_calculations.bayes_power_lift`):
  Power is the fraction of simulated experiments where P(B > A) meets or exceeds the
  ``confidence_level`` (e.g., 95%).
- **Expected loss criterion** (:func:`~ab_test.bayesian_binomial.power_calculations.bayes_power_loss`):
  Power is the fraction of simulated experiments where E[max(A - B, 0)] falls at or
  below a ``loss_threshold``, meaning the downside risk of picking B is acceptably small.

For each criterion, companion functions find the minimum sample size
(:func:`~ab_test.bayesian_binomial.power_calculations.bayes_minimum_sample_size`,
:func:`~ab_test.bayesian_binomial.power_calculations.bayes_minimum_sample_size_loss`)
and minimum detectable lift
(:func:`~ab_test.bayesian_binomial.power_calculations.bayes_minimum_detectable_lift`,
:func:`~ab_test.bayesian_binomial.power_calculations.bayes_minimum_detectable_lift_loss`)
via binary search.

Because power estimates are stochastic, results may vary slightly between calls.
Increase ``n_samples`` for more stable (but slower) results.

Example
-------

.. code-block:: python

   from ab_test.bayesian_binomial.power_calculations import (
       bayes_power_lift,
       bayes_minimum_sample_size,
       bayes_minimum_detectable_lift,
   )

   # Estimate power for a given sample size and effect
   power = bayes_power_lift(
       group_sizes=[5000, 5000],
       alphas=[1, 1], betas=[1, 1],
       baseline=0.10, alt_lift=0.20,
   )
   print(f"Power: {power:.2%}")

   # Find minimum sample size for 80% power
   n = bayes_minimum_sample_size(
       alphas=[1, 1], betas=[1, 1],
       baseline=0.10, alt_lift=0.20,
       target_power=0.80,
   )
   print(f"Min sample size per group: {n:,}")

   # Find minimum detectable lift at 80% power
   mdl = bayes_minimum_detectable_lift(
       group_size=5000,
       alphas=[1, 1], betas=[1, 1],
       baseline=0.10, target_power=0.80,
   )
   print(f"Min detectable lift: {mdl:.2%}")

API Reference
-------------

.. automodule:: ab_test.bayesian_binomial.power_calculations
   :members:
   :exclude-members: _resolve_alt_rate, _two_smallest_group_sizes, _simulate_posterior_draws, _search_min_sample_size, _search_min_lift
