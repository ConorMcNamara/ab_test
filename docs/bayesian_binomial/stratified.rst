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

API Reference
-------------

.. automodule:: ab_test.bayesian_binomial.stratified
   :members:
   :undoc-members:
   :show-inheritance:
