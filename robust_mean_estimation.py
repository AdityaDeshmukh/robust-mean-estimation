from bisect import bisect_left
from functools import partial
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

# -------------------------------------------------------------------------
# Robust Mean Estimation
# -------------------------------------------------------------------------

def _linear_operator_mv(v: np.ndarray, weights: np.ndarray, X: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Matrix-vector multiplication function for use in a LinearOperator.

    Parameters
    ----------
    v : np.ndarray
        Vector to multiply (dimension d).
    weights : np.ndarray
        Weights assigned to each sample (length n).
    X : np.ndarray
        Data matrix of shape (n, d).
    center : np.ndarray
        Center vector (dimension d).

    Returns
    -------
    np.ndarray
        Result of applying the weighted covariance operator to v.
    """
    centered_proj = weights * (X @ v - center @ v)
    return centered_proj @ X - center * np.sum(centered_proj)

def robust_mean(X: np.ndarray, sigma: float, c: float = 2.0, v: np.ndarray | None  = None, tol: float = 1e-3, max_iter: int = 100) -> np.ndarray:
    """
    Estimate the robust mean of a dataset using an iterative filtering procedure.

    References:
        1. "Topics in statistical inference with model uncertainty"
        https://www.ideals.illinois.edu/items/131409/bitstreams/436773/data.pdf (Algorithm 2).
            
        2. "Robust Mean Estimation in High Dimensions: An Outlier-Fraction Agnostic and EfficientAlgorithm"
        https://arxiv.org/pdf/1706.05047.pdf.
        
    Method:    
        This approach filters outliers by projecting onto directions of largest variance
        and trimming based on a variance threshold.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d) where n = number of samples, d = dimension, or
        a 1D array of shape (n,) for univariate data.
    sigma : float
        Spectral bound parameter that controls the trimming threshold.
    c : float
        Constant multiplier for the variance threshold (default: 2).
    v : np.ndarray | None
        Optional initial direction vector of shape (d,). If None, a random vector is used.
    tol : float
        Convergence tolerance (default: 1e-3).
    max_iter : int
        Maximum number of iterations (default: 100).

    Returns
    -------
    np.ndarray
        Estimated robust mean vector of shape (d,).
    """

    if X.ndim != 2:
        X = X.reshape(-1, 1)

    n, d = X.shape

    # Initializations
    if v is None:
        v = np.random.normal(0,1,d)
        v /= np.linalg.norm(v)
    weights = np.ones(n)/n
    threshold = c * sigma**2

    for _ in range(max_iter):
        prev_weights = weights.copy()

        # Weighted mean
        mu = np.average(X, axis=0, weights=weights)

        # Find direction of maximum variance
        if d > 1:
            # Compute top eigenvector of covariance operator
            lin_op = LinearOperator((d, d),matvec=partial(_linear_operator_mv, weights=weights, X=X, center = mu))
            _, eigvecs = eigsh(lin_op, k=1, which="LA", v0=v, ncv=15, tol=tol)
            v = eigvecs[:, 0]
        else:
            v = np.array([1.0])

        # Normalize direction
        v /= np.linalg.norm(v)

        # Project data onto v
        projected = X @ v
        median_proj = np.median(projected)

        # Compute squared deviations from median
        deviations = (projected - median_proj) ** 2
        sorted_indices = np.argsort(deviations)
        sorted_dev = deviations[sorted_indices]

        # Choose cutoff index based on variance constraint
        cutoff = bisect_left(np.cumsum(sorted_dev), n*threshold)
        
        # Binary weights: keep samples up to cutoff
        weights[sorted_indices[cutoff:]] = 0

        # Weighted mean estimate
        if np.sum(weights) == 0:
            raise RuntimeError("All weights are zero; consider increasing sigma or c, or sigma_min if using meta_robust_mean.")

        if np.array_equal(weights, prev_weights):
            # Converged
            return mu

    # If not converged, return last estimate
    return mu

def meta_robust_mean(X: np.ndarray, sigma_min: float = 1, theta: float = 1.1, c: float = 4) -> np.ndarray:
    """
    Reference:
        Meta-algorithm from Section 3 of Jain, Orlitsky, Ravindrakumar (2022).
        https://arxiv.org/pdf/2202.05453
    
    Method:
        Removes the need to know the sigma parameter.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d) where n = number of samples, d = dimension, or
        a 1D array of shape (n,) for univariate data.
    sigma_min : float
        Minimum candidate sigma value (default: 1).
    theta : float
        Geometric sequence factor (>1, default: 1.1).
    c : float
        Error bound constant (default: 4). 
        Typically, c=O(sqrt(eps)/(1-2*eps)) where eps is the true fraction of outliers.

    Returns
    -------
    x_hat : np.ndarray
        The robust meta-estimate of the mean.
    """

    if X.ndim != 2:
        X = X.reshape(-1, 1)
    n, d = X.shape
    
    # Find sigma_max
    if d > 1:
        lin_op = LinearOperator((d, d),matvec=partial(_linear_operator_mv, weights=np.ones(n)/n, X=X, center = np.mean(X, axis=0)))
        # Compute top eigenvalue of covariance operator
        eigvals, eigvecs = eigsh(lin_op, k=1, which="LA", ncv=15, tol=1e-3)
        v = eigvecs[:, 0]
        sigma_max = np.sqrt(eigvals[0])
    else:
        sigma_max = np.std(X, axis=0)[0]
    # Construct geometric sequence of candidate sigma values
    sigmas = []
    sigma = sigma_max
    while sigma > 0:
        if sigma < sigma_min:
            break
        sigmas.append(sigma)
        sigma /= theta
    sigmas = sorted(sigmas)

    # Run robust_mean for each sigma
    estimates = {sigma: robust_mean(X, sigma, v=v) for sigma in sigmas}

    # Find smallest sigma_hat satisfying Lemma 3 condition
    sigma_hat = None
    for sigma in sigmas:
        ok = True
        for s in sigmas:
            if s >= sigma:
                if np.linalg.norm(estimates[s]-estimates[sigma]) > c*(s + sigma):
                    ok = False
                    break
        if ok:
            sigma_hat = sigma
            break

    if sigma_hat is None:
        raise RuntimeError("No valid beta_hat found — check robust_mean outputs.")
    
    return estimates[sigma_hat]