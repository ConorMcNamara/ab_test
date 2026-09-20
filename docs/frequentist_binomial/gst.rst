Group Sequential Testing
========================

Group sequential designs allow experimenters to analyze A/B test results at
pre-planned interim analyses while controlling the overall type-I error rate.
At each interim look, the test statistic is compared against a pre-computed
boundary; if it exceeds the boundary, the experiment stops early for efficacy.

This module implements alpha spending functions following Lan & DeMets (1983)
and boundary computation via the recursive numerical integration of Jennison &
Turnbull (1999). Three spending function families are provided:

* **O'Brien-Fleming** -- very conservative early stopping, spends nearly all
  alpha at the final look. Best for experiments where early stopping should
  only happen for overwhelming effects.
* **Pocock** -- more uniform boundaries across looks. Better when you genuinely
  want equal opportunity to stop early at any look.
* **Power family** -- parameterized by ``rho``; ``rho=1`` is linear
  (Pocock-like), larger ``rho`` approaches O'Brien-Fleming behavior.

Usage
-----

Create a design and inspect the boundaries:

.. code-block:: python

   from ab_test.frequentist_binomial.gst import GroupSequentialDesign

   design = GroupSequentialDesign(n_analyses=4, alpha=0.05)
   print(design.summary())

Test at an interim analysis:

.. code-block:: python

   rejected = design.test(
       trials=[2000, 2000],
       successes=[180, 220],
       look=2,
   )

Compute the required sample size accounting for interim analyses:

.. code-block:: python

   from ab_test.frequentist_binomial.gst import gst_required_sample_size

   n = gst_required_sample_size(
       baseline=0.10,
       alt_lift=0.20,
       n_analyses=4,
   )

Compare power curves with a fixed-horizon test:

.. code-block:: python

   from ab_test.frequentist_binomial.gst import plot_gst_power_curve

   fig = plot_gst_power_curve(baseline=0.10, alt_lift=0.20, n_analyses=4)
   fig.show()

When to Use GST vs mSPRT
-------------------------

Both approaches control type-I error under repeated looks, but with
different trade-offs:

* **Group sequential testing** requires a pre-specified analysis schedule
  (number of looks and information fractions). In return, it provides exact
  boundaries, formal power calculations, and sample-size planning. Best when
  you have a fixed schedule of interim analyses.
* **mSPRT** (:mod:`~ab_test.frequentist_binomial.msprt`) allows peeking at
  any time without a pre-specified schedule. It is more flexible but cannot
  provide the same formal power guarantees.

API Reference
-------------

.. automodule:: ab_test.frequentist_binomial.gst
   :members:
