Cluster-Randomized Trials (Bayesian)
=====================================

Bayesian analysis of cluster-randomized experiments using a beta-binomial
hierarchical model. Randomization occurs at the cluster level (stores,
markets, time windows). Within each arm, cluster rates follow a Beta
distribution whose mean and intra-cluster correlation (ICC) both get priors
(uniform on the mean, ``Beta(1/2, 1)`` on the ICC). The posterior of each
arm's rate integrates over the ICC instead of plugging in an estimate, so it
stays calibrated with few clusters, unequal cluster sizes, and large ICCs.
The power functions simulate this same posterior. Their expected-loss
threshold (``loss_threshold``) is a difference in rates, E[max(C - T, 0)],
whereas ``analyze()`` reports the expected loss in the units of the chosen
lift; for relative lift, convert a threshold taken from ``analyze()`` back to
a rate difference before planning.

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
