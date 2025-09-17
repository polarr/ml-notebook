import numpy as np
import statistics

def mean(y):
    """Computes the mean of the input array

    Parameters
    ----------
    y : an input numpy array

    Returns
    -------
    y_mean :
        Returns the mean of the input array
    """
    if len(y) == 0:
        return 0
    else:
        return np.mean(y)

def mode(y):
    """Computes the element with the maximum count

    Parameters
    ----------
    y : an input numpy array

    Returns
    -------
    y_mode :
        Returns the element with the maximum count
    """
    if len(y) == 0:
        return -1
    else:
        return statistics.mode(y.flatten())

def euclidean_dist_squared(X, Xtest):
    # FYI, sklearn.metrics.pairwise.euclidean_distances
    # does this but a little bit nicer; this code is just here so you can
    # easily see that it's not doing anything actually very complicated

    X_norms_sq = np.sum(X ** 2, axis=1)
    Xtest_norms_sq = np.sum(Xtest ** 2, axis=1)
    dots = X @ Xtest.T

    return X_norms_sq[:, np.newaxis] + Xtest_norms_sq[np.newaxis, :] - 2 * dots

def entropy(p):
    """
    A helper function that computes the entropy of the
    discrete distribution p (stored in a 1D numpy array).
    The elements of p should add up to 1.
    This function ensures lim p-->0 of p log(p) = 0
    which is mathematically true, but numerically results in NaN
    because log(0) returns -Inf.
    """
    plogp = 0 * p  # initialize full of zeros
    plogp[p > 0] = p[p > 0] * np.log(p[p > 0])  # only do the computation when p>0
    return -np.sum(plogp)

def sse(y):
    """
    A helper function that computes the sum of squared errors
    """
    if len(y) == 0:
        return 0
    y_mean = np.mean(y)
    return np.sum((y - y_mean) ** 2)