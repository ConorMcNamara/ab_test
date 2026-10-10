Bayesian Stratified Analysis
============================

When an experiment spans heterogeneous sub-populations (device type, region,
traffic source), pooling across strata controls for confounding and improves
precision.

The frequentist
:class:`~ab_test.frequentist_binomial.stratified.StratifiedContingencyTable`
uses the Cochran-Mantel-Haenszel framework with inverse-variance weighted
point estimates and Wald confidence intervals. The Bayesian counterpart
replaces those with inverse-variance weighted *posterior samples*, producing
credible intervals, P(T > C), expected loss, and ROPE probabilities.

Usage
-----

.. code-block:: python

   from ab_test.bayesian_binomial.stratified import BayesianStratifiedContingencyTable

   st = BayesianStratifiedContingencyTable("Campaign", "converted")
   st.add("Control",   successes=50,  trials=500,  alpha=1, beta=1, stratum="mobile")
   st.add("Treatment", successes=65,  trials=500,  alpha=1, beta=1, stratum="mobile")
   st.add("Control",   successes=100, trials=1000, alpha=1, beta=1, stratum="desktop")
   st.add("Treatment", successes=120, trials=1000, alpha=1, beta=1, stratum="desktop")

   # Pooled Bayesian analysis
   print(st.analyze(lift="absolute", confidence_level=0.95))

   # Per-stratum breakdown
   print(st.analyze_by_stratum(lift="absolute"))

   # Forest plot with pooled diamond
   st.plot(lift="absolute")

Three or More Variants
----------------------

Add any number of groups; the first one added is the control. With three or
more, ``analyze()`` reports:

* each group's **probability of being best** and its **expected loss**,
  E[best rate - its rate], from one joint posterior draw of every group's
  stratified rate. Each stratum's posterior rate is weighted by that stratum's
  share of all trials, so every group is standardized to the same stratum mix,
  and the loss is a difference in rates whatever ``lift`` is;
* **pairwise comparisons**, each reported as a two-group analysis of that pair
  would be: pooled lift, credible interval, probability of being greater,
  expected loss, ROPE probability and between-stratum tau.
  ``comparisons="control"`` (the default) compares each group with the
  control, and ``comparisons="all"`` compares every pair.

Scaled lifts and the default ROPE are expressed over the largest group's
total trials for every comparison, so identical rate differences give
identical values; the ROPE half-width is still 10% of each comparison's
reference group's rate. Posterior probabilities need no multiple-comparison
correction, but stopping as soon as one crosses a threshold still inflates
false wins. ``pooled_results`` then holds ``"prob_best"``,
``"expected_loss"``, ``"standardized_rate"`` and a ``"comparisons"`` dict
keyed by labels such as ``"B vs A"``, and ``heterogeneity_results`` is
``None`` (each comparison carries its own tau).

.. code-block:: python

   st = BayesianStratifiedContingencyTable("Landing Page", "Conversion Rate")
   for stratum, rates in {"mobile": (50, 62, 70), "desktop": (80, 90, 104)}.items():
       for cell, successes in zip(("A", "B", "C"), rates):
           st.add(cell, successes, 500, 1, 1, stratum=stratum)
   print(st.analyze(comparisons="all"))
   st.pooled_results["prob_best"]["C"]

Per-stratum output (:meth:`analyze_by_stratum` and :meth:`plot`) shows one
comparison across strata, so it stays two-group only and raises with three or
more groups.

API Reference
-------------

.. automodule:: ab_test.bayesian_binomial.stratified
   :members:
   :undoc-members:
   :show-inheritance:
