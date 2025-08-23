# Robust Mean Estimation
Robust mean estimation is a statistical technique designed to compute a reliable mean in the presence of outliers and heavy-tailed inliers.
This repository provides fast, efficient and accurate methods (which do not require knowledge of fraction of outliers, in contrast with existing methods) for robustly estimating the mean of a dataset.
These methods are based on rigorous theoretical foundations (see references).

Two methods are provided in `robust_mean_estimation.py`:
1) `robust_mean` : outlier-fraction agnostic, but requires seqctral bound ($\Sigma\preceq\sigma^2 I$).
2) `meta_robust_mean` : outlier-fraction and spectral bound agnostic (only requires lower bound on $\sigma$).
   
Algorithms implemented from the following references:
1) Deshmukh, A. (2024). [Topics in statistical inference with model uncertainty](https://www.ideals.illinois.edu/items/131409/bitstreams/436773/data.pdf) (Doctoral dissertation, University of Illinois at Urbana-Champaign).
2) Deshmukh, A., Liu, J., & Veeravalli, V. V. (2023). [Robust mean estimation in high dimensions: An outlier-fraction agnostic and efficient algorithm](https://arxiv.org/abs/2102.08573). IEEE Transactions on Information Theory, 69(7), 4675-4690.
3) Jain, A., Orlitsky, A., & Ravindrakumar, V. (2022). [Robust estimation algorithms don't need to know the corruption level](https://arxiv.org/pdf/2202.05453). arXiv preprint arXiv:2202.05453.

## Dependencies
- numpy
- scipy

## Example
```python
import numpy as np
from robust_mean_estimation import robust_mean

# Generate toy data: Gaussian cloud + outliers
mu, sigma = 0, 1
n ,d = 500, 10
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
```
### Example output
```
Distance of naive mean from true mean: 5.26721
Distance of median from true mean: 0.8246
Distance of robust mean from true mean: 0.17066
```
