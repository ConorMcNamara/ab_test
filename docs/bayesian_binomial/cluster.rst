Cluster-Randomized Trials (Bayesian)
=====================================

Bayesian analysis of cluster-randomized experiments using a beta-binomial
hierarchical model. Randomization occurs at the cluster level (stores,
markets, time windows). Within each arm, cluster rates follow a Beta
distribution whose mean and intra-cluster correlation (ICC) both get priors
(uniform on the mean, ``Beta(1/2, 1)`` on the ICC). The posterior of each
arm's rate integrates over the ICC instead of plugging in an estimate, so it
stays calibrated with few clusters, unequal cluster sizes, and large ICCs.
The power functions simulate this same posterior.

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
        seed=42,
    )

Pass ``seed`` to any power, search or plot function for reproducible results. Each
cluster has its own random stream, so a search over the number of clusters compares
nested designs and returns the same answer every time.

Three or More Groups
--------------------

With three or more groups (the first group added is the control),
``analyze()`` reports each group's **probability of being best** and its
**expected loss**, E[best rate - its rate] (a difference in rates), from one
joint draw of the arm-level posteriors, then the **pairwise comparisons**
(``comparisons="control"`` or ``"all"``), each reported as a two-group
analysis would be: lift, credible interval, probability of being greater,
expected loss in the lift's units, and the ROPE probability with its default
scaled to that comparison's reference group. Posterior probabilities need no
multiple-comparison correction, but stopping as soon as one crosses a
threshold still inflates false wins. ``plot()`` and ``plot_pdf()`` show
every group. With two groups the output is unchanged.

API Reference
-------------

.. automodule:: ab_test.bayesian_binomial.cluster
   :members:
   :undoc-members:
   :show-inheritance:
