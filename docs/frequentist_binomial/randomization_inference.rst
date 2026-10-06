Randomization Inference
=======================

Randomization inference computes p-values directly from the randomization
itself, with no distributional assumptions: shuffle the treatment labels many
times, recompute the test statistic each time, and ask how often a shuffled
result is at least as extreme as the observed one. It remains valid for small
samples and skewed outcomes where normal approximations are unreliable.

Two tests are provided:

* :func:`~ab_test.frequentist_binomial.randomization_inference.randomization_test`
  — permutes individual units in a 2x2 table. The permutation distribution
  is drawn from the hypergeometric distribution, so it is fully vectorized.
* :func:`~ab_test.frequentist_binomial.randomization_inference.cluster_randomization_test`
  — permutes whole clusters, for experiments randomized at the cluster level
  (stores, schools, regions). Permuting individuals there would understate
  the variance.

Usage
-----

.. code-block:: python

   from ab_test.frequentist_binomial.randomization_inference import (
       cluster_randomization_test,
       randomization_test,
   )

   # Individual-level randomization
   p = randomization_test(trials=[1000, 1000], successes=[100, 130], seed=0)

   # Cluster-level randomization: one entry per cluster
   p = cluster_randomization_test(
       successes_ctrl=[45, 52, 48],
       trials_ctrl=[500, 480, 510],
       successes_treat=[62, 58, 66],
       trials_treat=[490, 520, 505],
       seed=0,
   )

``n_permutations`` (default 10,000) controls the Monte Carlo precision.
``cluster_randomization_test`` can enumerate every assignment with
``exact=True`` when the number of clusters is small, and parallelizes Monte
Carlo permutations with ``n_jobs``.

API Reference
-------------

.. automodule:: ab_test.frequentist_binomial.randomization_inference
   :members:
