mSPRT (Always-Valid Inference)
==============================

The Mixture Sequential Probability Ratio Test (mSPRT) provides always-valid
p-values and confidence sequences that control type-I error regardless of how
many times results are checked. This solves the "peeking problem" -- in
standard fixed-horizon tests, checking results repeatedly inflates the false
positive rate, but mSPRT p-values remain valid at any stopping time.

Based on Johari et al. (2017), the test mixes over a Gaussian prior
``N(0, tau^2)`` on the effect size. When ``tau`` is not specified, it defaults
to the larger of ``0.1`` times the pooled success rate (so the mixture covers
roughly +/-10% relative lifts) and the absolute effect implied by the null
lift, which keeps confidence sequences informative for rare events. ``tau``
must not shrink with the sample size (for
example, by tying it to the standard error), or the test loses its
always-valid guarantee. For best power, set ``tau`` (in absolute-rate units)
to the effect sizes typical of past experiments.

The reported p-value is ``min(1, 1 / Lambda_n)``, where ``Lambda_n`` is the
likelihood ratio at the current look. Johari et al.'s always-valid p-value is
the running minimum of this over all looks so far. The per-look value is never
below it, so it is valid but conservative, and it can rise between looks; take
the minimum over looks to get the always-valid p-value. Likewise,
:func:`~ab_test.frequentist_binomial.msprt.plot_msprt_over_time` draws each
look's interval without intersecting them across looks.

The module also includes
:func:`~ab_test.frequentist_binomial.msprt.plot_msprt_over_time` for
visualizing the point estimate and confidence sequence across time-ordered
checkpoints, and the mSPRT test can be used through the
:class:`~ab_test.frequentist_binomial.contingency.ContingencyTable` by passing
``test_method="msprt"`` to ``analyze()``.

Usage
-----

.. code-block:: python

   from ab_test.frequentist_binomial.msprt import msprt_test

   # Standalone p-value at the current look
   p_value = msprt_test([1000, 1000], [100, 130])

   # Through ContingencyTable
   from ab_test.frequentist_binomial.contingency import ContingencyTable

   ct = ContingencyTable("Experiment", "CVR")
   ct.add("Control", 100, 1000).add("Treatment", 130, 1000)
   print(ct.analyze(test_method="msprt"))

   # Plot confidence sequence over time
   from ab_test.frequentist_binomial.msprt import plot_msprt_over_time

   tables = [ct_week1, ct_week2, ct_week3]
   labels = ["Week 1", "Week 2", "Week 3"]
   fig = plot_msprt_over_time(tables, labels)
   fig.show()

API Reference
-------------

.. automodule:: ab_test.frequentist_binomial.msprt
   :members:
