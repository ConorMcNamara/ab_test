import numpy as np

def sample_beta(s: int, n: int, alpha: float, beta: float, n_samples: int) -> np.array:
    """Draw samples from the Beta posterior given observed binomial data.

    Combines the observed data with the Beta prior to form the posterior
    Beta(alpha + s, beta + n - s) and draws samples from it.

    Parameters
    ----------
    s : int
        Number of successes observed.
    n : int
        Total number of trials.
    alpha : float
        Alpha parameter of the Beta prior distribution.
    beta : float
        Beta parameter of the Beta prior distribution.
    n_samples : int
        Number of samples to draw from the posterior.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples,)`` containing draws from the posterior.
    """
    return np.random.beta(alpha + s, beta + n - s, n_samples)
