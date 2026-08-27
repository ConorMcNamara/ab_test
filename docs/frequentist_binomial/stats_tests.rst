Statistical Tests
=================

This module provides multiple frequentist significance tests for 2x2
contingency tables. The :func:`~ab_test.frequentist_binomial.stats_tests.ab_test`
dispatcher selects among: Rao's score test (the recommended default, equivalent
to Pearson's chi-squared), likelihood ratio test, z-test, Fisher's exact test,
Barnard's exact test, Boschloo's exact test, and several power-divergence
variants (Freeman-Tukey, Neyman, Cressie-Read, modified log-likelihood). The
mSPRT test is also accessible through the dispatcher.

All tests return either a p-value or a boolean significance result when a
critical value is supplied. Passing a precomputed critical value avoids repeated
p-value calculations and makes Monte Carlo simulations roughly 5x faster.

All tests currently support exactly two groups. The score and likelihood ratio
tests support nonzero relative null lifts; the remaining tests require absolute
lift or a zero null.

Usage
-----

.. code-block:: python

   from ab_test.frequentist_binomial.stats_tests import ab_test, score_test

   # Use the dispatcher
   p_value = ab_test([1000, 1000], [100, 130], method="score")

   # Or call the test function directly
   p_value = score_test([1000, 1000], [100, 130])

   # With a precomputed critical value for faster simulations
   import scipy.stats as ss
   crit = ss.chi2.isf(0.05, df=1)
   is_significant = score_test([1000, 1000], [100, 130], crit=crit)

API Reference
-------------

.. automodule:: ab_test.frequentist_binomial.stats_tests
   :members:
   :exclude-members: _contingency_table, _test_result, _power_divergence_test, _validate_two_group
