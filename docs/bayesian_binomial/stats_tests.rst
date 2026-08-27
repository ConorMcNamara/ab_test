Bayesian Statistical Tests
==========================

Overview
--------

This module provides Bayesian metrics for comparing two binomial variants via
Monte Carlo posterior sampling. The core function, :func:`~ab_test.bayesian_binomial.stats_tests.calculate_metrics`,
draws posterior Beta samples for each variant and computes a suite of decision metrics
in a single call.

Available metrics include:

- **P(B > A)**: the posterior probability that variant B's conversion rate exceeds A's.
- **Expected loss**: the average downside of choosing B when A is actually better,
  computed as E[max(A - B, 0)]. A low expected loss indicates that choosing B carries
  little risk.
- **ROPE analysis**: the probability that the lift falls within a Region of Practical
  Equivalence, i.e., an interval where the difference is considered negligible.
- **Threshold probabilities**: the probability that the lift exceeds or falls below
  user-specified thresholds.

All metrics are computed from posterior Beta samples, so lift types beyond relative and
absolute (incremental, ROAS, revenue) are also supported.

Example
-------

.. code-block:: python

   from ab_test.bayesian_binomial.stats_tests import calculate_metrics

   results = calculate_metrics(
       successes=[100, 130], trials=[1000, 1000],
       alphas=[1, 1], betas=[1, 1],
       n_samples=100_000, lift="relative",
   )
   print(f"P(B > A): {results['Proportion of samples where B exceeds A']:.3f}")
   print(f"Expected loss: {results['Expected loss']:.5f}")
   print(f"Prob in ROPE: {results['Probability of ROPE']:.3f}")

API Reference
-------------

.. automodule:: ab_test.bayesian_binomial.stats_tests
   :members:
