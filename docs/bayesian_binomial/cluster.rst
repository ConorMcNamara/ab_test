Cluster-Randomized Trials (Bayesian)
=====================================

Bayesian analysis of cluster-randomized experiments using a beta-binomial
hierarchical model. Randomization occurs at the cluster level (stores,
markets, time windows) and the resulting intra-cluster correlation is
captured via method-of-moments estimation of per-arm Beta distributions.

Usage
-----

.. code-block:: python

    from ab_test.bayesian_binomial.cluster import BayesianClusterRandomizedTrial

    crt = BayesianClusterRandomizedTrial(name="Store Rollout", metric_name="conversion")
    for store, s, n in control_stores:
        crt.add(store, successes=s, trials=n, group="Control")
    for store, s, n in treatment_stores:
        crt.add(store, successes=s, trials=n, group="Treatment")

    print(crt.analyze(lift="relative"))
    print(f"ICC: {crt.pooled_icc:.4f}")

Power analysis uses simulation-based assurance:

.. code-block:: python

    from ab_test.bayesian_binomial.cluster import cluster_bayes_power_lift

    power = cluster_bayes_power_lift(
        n_clusters=20,
        cluster_size=500,
        icc=0.02,
        baseline=0.10,
        alt_lift=0.20,
    )

API Reference
-------------

.. automodule:: ab_test.bayesian_binomial.cluster
   :members:
   :undoc-members:
   :show-inheritance:
