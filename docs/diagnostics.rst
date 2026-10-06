Pre-Analysis Diagnostics
========================

Run these checks before trusting experiment results. A significant sample
ratio mismatch, for example, can indicate a broken randomisation layer, a data
pipeline bug, or differential attrition — any of which can invalidate the
entire analysis regardless of the p-value it produces.

Sample Ratio Mismatch (SRM)
---------------------------

A chi-squared goodness-of-fit test that compares the observed traffic split
against the intended split. If you planned a 50/50 split but observed 48/52,
was that just sampling noise or a systematic problem?

Common causes of SRM:

- **Broken randomiser** — the hashing or bucketing logic is not uniform.
- **Differential bot filtering** — bots are removed from one group but not the
  other.
- **Triggered-analysis mismatch** — the event that triggers assignment differs
  from the event used to count users.
- **Lossy joins** — a pipeline join drops rows asymmetrically.

Usage
~~~~~

.. code-block:: python

   from ab_test.diagnostics import srm_test

   # Check a two-group 50/50 experiment
   stat, pvalue = srm_test([4800, 5200])
   print(f"SRM chi2={stat:.2f}, p={pvalue:.4f}")

   # Check a three-group experiment with a 50/25/25 split
   stat, pvalue = srm_test(
       [5000, 2600, 2400],
       expected_proportions=[0.50, 0.25, 0.25],
   )

Time Trend (Novelty and Primacy Effects)
----------------------------------------

Tests whether the treatment effect is stable over the course of the
experiment. Per-period absolute lifts are regressed on time with weighted
least squares; a significant slope means the effect is drifting.

- **Novelty effect** — the lift decays as users get used to the change, so an
  early readout overstates the long-run effect.
- **Primacy effect** — the lift grows as users learn the new experience, so an
  early readout understates it.

.. code-block:: python

   from ab_test.diagnostics import time_trend_test

   result = time_trend_test(
       successes_a=[100, 98, 102, 99, 101],
       trials_a=[1000] * 5,
       successes_b=[140, 128, 118, 110, 104],
       trials_b=[1000] * 5,
       labels=["Mon", "Tue", "Wed", "Thu", "Fri"],
   )
   result["diagnosis"]  # "novelty" -- the lift is shrinking day by day
   result["figure"].show()  # per-period lift, cumulative lift and trend line

Needs at least three periods. The returned dict also includes the slope, its
standard error, the p-value, and the per-period and cumulative lifts.

Placebo Test
------------

Compares the two groups on data the treatment cannot have affected — the
pre-experiment period for the same users, or a placebo outcome measured
before exposure. The true effect is zero by construction, so a significant
result points to pre-existing imbalance, a broken randomiser, or a pipeline
bug rather than a real effect.

.. code-block:: python

   from ab_test.diagnostics import placebo_test

   result = placebo_test(
       successes_a=480, trials_a=10_000,
       successes_b=495, trials_b=10_000,
   )
   result["failed"]  # False -- no difference where none should exist

The result also includes the placebo lift, a confidence interval from
inverting the score test, and the p-value. ``lift`` defaults to
``"absolute"``, which stays defined when the control group has no placebo
successes; ``test_method`` accepts any method supported by
:func:`ab_test.frequentist_binomial.stats_tests.ab_test`.

API Reference
-------------

.. automodule:: ab_test.diagnostics
   :members:
