Utilities
=========

Overview
--------

Helper functions for Bayesian binomial analysis. This module provides posterior Beta
sampling and posterior mean computation, which underpin the statistical tests, credible
intervals, and power calculations throughout the ``bayesian_binomial`` subpackage.

:func:`~ab_test.bayesian_binomial.utils.sample_beta` draws samples from the Beta
posterior ``Beta(alpha + s, beta + n - s)`` given observed data and a prior.
:func:`~ab_test.bayesian_binomial.utils.posterior_mean` returns the analytical mean
``(alpha + s) / (alpha + beta + n)`` without sampling.

Example
-------

.. code-block:: python

   from ab_test.bayesian_binomial.utils import sample_beta, posterior_mean

   # Draw 10,000 posterior samples
   samples = sample_beta(s=100, n=1000, alpha=1, beta=1, n_samples=10_000)
   print(f"Sample mean: {samples.mean():.4f}")

   # Analytical posterior mean
   mean = posterior_mean(s=100, n=1000, alpha=1, beta=1)
   print(f"Posterior mean: {mean:.4f}")

API Reference
-------------

.. automodule:: ab_test.bayesian_binomial.utils
   :members:
