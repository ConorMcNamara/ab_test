Cluster-Randomized Trials
=========================

In a cluster-randomized trial (CRT), randomization occurs at the cluster
level -- stores, markets, time windows, or other groupings -- rather than at
the individual level. Because individuals within a cluster tend to be more
similar than individuals across clusters, outcomes are correlated within
clusters. This **intra-cluster correlation (ICC)** inflates variance and must
be accounted for in both sample-size planning and inference.

The key quantity is the **design effect** (DEFF), defined as
``1 + (m - 1) * ICC`` where ``m`` is the average cluster size. The design
effect indicates how many more individuals are needed relative to an
individually-randomized trial of the same size to achieve equivalent
statistical precision. An ICC of 0.02 with clusters of 50 individuals, for
example, roughly doubles the required sample size.

This module provides:

* **ICC estimation** from cluster-level binary outcomes via the ANOVA method
  (Ridout, Demetrio & Firth, 1999).
* **Power and sample-size calculations** adjusted for the design effect,
  integrated with the existing pluggable power framework.
* A **cluster-required-clusters** function that answers the key planning
  question: how many clusters per arm are needed?
* A :class:`~ab_test.frequentist_binomial.cluster.ClusterRandomizedTrial`
  class for post-experiment analysis using a cluster-summary Welch t-test
  (Donner & Klar, 2000).

Usage
-----

Analyze a completed cluster-randomized trial:

.. code-block:: python

   from ab_test.frequentist_binomial.cluster import ClusterRandomizedTrial

   crt = ClusterRandomizedTrial("Store Test", "conversion")
   crt.add("store_1", 45, 500, group="control")
   crt.add("store_2", 52, 480, group="control")
   crt.add("store_3", 38, 510, group="control")
   crt.add("store_4", 62, 490, group="treatment")
   crt.add("store_5", 58, 520, group="treatment")
   crt.add("store_6", 65, 500, group="treatment")

   print(crt.analyze(lift="relative"))

Compute the number of clusters per arm needed for a new experiment:

.. code-block:: python

   from ab_test.frequentist_binomial.cluster import cluster_required_clusters

   k = cluster_required_clusters(
       baseline=0.10,
       alt_lift=0.20,
       icc=0.02,
       cluster_size=50,
   )
   print(f"Clusters per arm: {k}")

Compare power curves with and without clustering:

.. code-block:: python

   from ab_test.frequentist_binomial.cluster import plot_cluster_power_curve

   fig = plot_cluster_power_curve(
       baseline=0.10,
       alt_lift=0.20,
       icc=0.02,
       avg_cluster_size=50,
   )
   fig.show()

When to Use This vs CupacExperiment
------------------------------------

Both this module and
:class:`~ab_test.frequentist_binomial.cupac.CupacExperiment` can handle
clustered data, but they serve different purposes:

* **This module** works with aggregate cluster-level counts (successes and
  trials per cluster) and does not require individual-level covariate data.
  Use it for standard cluster-randomized trial design and analysis.
* **CupacExperiment** with ``cluster_col`` performs covariate-adjusted
  analysis using individual-level data and CR2 cluster-robust standard
  errors. Use it when you have per-user covariates and want variance
  reduction on top of cluster adjustment.

API Reference
-------------

.. automodule:: ab_test.frequentist_binomial.cluster
   :members:
