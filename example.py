import numpy as np
from robust_mean_estimation import robust_mean

# Generate toy data: Gaussian cloud + outliers
mu, sigma = 0, 1
n, d = 500, 10
rng = np.random.default_rng(0)
X = rng.normal(mu, sigma, size=(n, d))             # inliers
outliers = rng.normal(10, 1, size=(int(0.2*n), d)) # outliers
X = np.vstack([X, outliers])

# Standard (naive) mean — sensitive to outliers
naive_mean = X.mean(axis=0)

# Coordinate-wise median — more robust but not optimal in high-dimensions
median = np.median(X, axis=0)

# Robust mean estimate
robust_mu = robust_mean(X, sigma)

print("Distance of naive mean from true mean:", round(np.linalg.norm(naive_mean - mu),5))
print("Distance of median from true mean:", round(np.linalg.norm(median - mu),5))
print("Distance of robust mean from true mean:", round(np.linalg.norm(robust_mu - mu),5))