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

API Reference
-------------

.. autoclass:: ab_test.bayesian_binomial.contingency.BayesianContingencyTable
   :members:
   :show-inheritance:
