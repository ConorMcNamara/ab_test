Exact Bernoulli E-Test
======================

An exact sequential test that two groups share the same success rate, based on
the e-process of Turner, Ly & Grünwald (2024). Like the mSPRT, its type-I
error stays at most ``alpha`` however often you look and whenever you stop.
Unlike the mSPRT, it uses no normal approximation and has no mixing scale to
tune, so it is exact at small samples and rare events.

The data are a sequence of looks, given as cumulative counts. Each look's new
observations form a block whose e-value compares each group's posterior-mean
rate from the earlier looks with the closest common rate. The e-process is the
running product of the block e-values, and the null is rejected once it
reaches ``1 / alpha``. The anytime-valid p-value is ``1 / max(e-process)``.

Things to know:

* The first look has no earlier data, so its e-value is 1. The test learns
  from the second look on, and more frequent looks give more power.
* The allocation and the look schedule must not depend on the outcomes.
* It tests only equal rates: there is no ``null_lift`` and no confidence
  interval. Use :mod:`~ab_test.frequentist_binomial.msprt` for those.

Compared with the mSPRT under continuous monitoring (40 looks of 250 per arm),
both keep the type-I error well below 5%. The mSPRT is more powerful when its
default ``tau`` suits the effect (about 0.955 vs 0.90 for 10% vs 12%), and the
e-test is more powerful when it does not (about 0.43 vs 0.26 for 2% vs 2.6%).

Usage
-----

.. code-block:: python

   from ab_test.frequentist_binomial.e_test import bernoulli_e_test

   # Cumulative trials and successes for control and treatment at each look
   trials = [[1000, 1000], [2000, 2000], [3000, 3000]]
   successes = [[100, 130], [205, 262], [300, 395]]

   result = bernoulli_e_test(trials, successes, alpha=0.05)
   result["rejected"], result["rejected_at"], result["p_value"]

   # From a list of ContingencyTables, one per checkpoint
   trials = [table.trials for table in tables]
   successes = [table.successes for table in tables]

API Reference
-------------

.. automodule:: ab_test.frequentist_binomial.e_test
   :members:
