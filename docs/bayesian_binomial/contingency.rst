Bayesian Contingency Table
==========================

Overview
--------

The :class:`~ab_test.bayesian_binomial.contingency.BayesianContingencyTable` class
provides a fully Bayesian analysis of A/B tests using Beta-Binomial conjugate priors.
Users specify Beta prior parameters (``alpha``, ``beta``) for each variant alongside
the observed successes and trials. The :meth:`analyze` method computes posterior means,
credible intervals (equal-tailed or HDI), the probability that variant B beats A,
expected loss from choosing B, and ROPE (Region of Practical Equivalence) analysis.
Supported lift types include relative, absolute, incremental, ROAS, and revenue.

The class also exposes :meth:`analyze_individually` for per-cell posterior summaries and
:meth:`plot_pdf` for visualizing overlapping posterior Beta distributions with HDI
annotations and a win-probability title. Like its frequentist counterpart, it supports
chainable ``add()`` calls, DataFrame export, and serialization.

A uniform prior ``Beta(1, 1)`` is a common non-informative choice. If historical data
is available, the prior can encode that knowledge — for example, ``Beta(10, 90)``
centers the prior at a 10% conversion rate with moderate confidence.

Example
-------

.. code-block:: python

   from ab_test.bayesian_binomial.contingency import BayesianContingencyTable

   bct = BayesianContingencyTable("My Experiment", "Conversion Rate")
   bct.add("Control", successes=100, trials=1000, alpha=1, beta=1)
   bct.add("Treatment", successes=130, trials=1000, alpha=1, beta=1)

   # Full analysis with relative lift
   print(bct.analyze(lift="relative"))

   # Individual cell posteriors
   print(bct.analyze_individually(cred_int_method="hdi"))

   # Plot overlapping posterior distributions
   fig = bct.plot_pdf(confidence_level=0.95)
   fig.show()

Three or More Variants
----------------------

With three or more cells, ``analyze()`` reports:

* each variant's **probability of being best** and its **expected loss**,
  E[best rate - its rate], both from one joint posterior draw across all
  variants (the loss is a difference in rates whatever ``lift`` is);
* **pairwise comparisons**, each reported as a two-variant analysis would be
  (lift, credible interval, probability of being greater, expected loss and
  ROPE probability, with the ROPE scaled to that comparison's reference
  variant). ``comparisons="control"`` (the default) compares each variant
  against the first cell added; ``comparisons="all"`` compares every pair.

Posterior probabilities need no multiple-comparison correction. They are
calibrated when the true rates behave like the prior, but stopping as soon as a
probability crosses a threshold still inflates false wins for a fixed truth.

.. code-block:: python

   table = BayesianContingencyTable("Checkout", "conversion")
   table.add("A", 100, 1000, 1, 1).add("B", 120, 1000, 1, 1).add("C", 140, 1000, 1, 1)
   print(table.analyze())

   table.incremental_results["prob_best"]["C"]
   table.incremental_results["comparisons"]["C vs A"]["ci_lower"]

Scaled lifts (``"incremental"``, ``"roas"``, ``"revenue"``, ``"cpa"``) are
expressed over one common number of units for every comparison, the table's
largest arm, so identical rate differences give identical lifts, and ROPE
thresholds and expected losses are in the same units for every comparison. The
output states the scale.

With two cells the output and ``incremental_results`` are unchanged, and
``comparisons`` is ignored.

API Reference
-------------

.. autoclass:: ab_test.bayesian_binomial.contingency.BayesianContingencyTable
   :members:
   :show-inheritance:
