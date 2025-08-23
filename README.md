# Robust Mean Estimation
Robust mean estimation is a statistical technique designed to compute a reliable mean in the presence of outliers and heavy-tailed inliers.
This repository provides fast, efficient and accurate methods (which do not require knowledge of fraction of outliers, in contrast with existing methods) for robustly estimating the mean of a dataset.
These methods are based on rigorous theoretical foundations (see references).

Two methods are provided in `robust_mean_estimation.py`:
1) `robust_mean` : outlier-fraction agnostic, but requires seqctral bound ($\Sigma\preceq\sigma^2 I$).
2) `meta_robust_mean` : outlier-fraction and spectral bound agnostic (only requires lower bound on $\sigma$).
   
Implementation of algorithms from the following references:
1) Deshmukh, A. (2024). [Topics in statistical inference with model uncertainty](https://www.ideals.illinois.edu/items/131409/bitstreams/436773/data.pdf) (Doctoral dissertation, University of Illinois at Urbana-Champaign).
2) Deshmukh, A., Liu, J., & Veeravalli, V. V. (2023). [Robust mean estimation in high dimensions: An outlier-fraction agnostic and efficient algorithm](https://arxiv.org/abs/2102.08573). IEEE Transactions on Information Theory, 69(7), 4675-4690.
3) Jain, A., Orlitsky, A., & Ravindrakumar, V. (2022). [Robust estimation algorithms don't need to know the corruption level](https://arxiv.org/pdf/2202.05453). arXiv preprint arXiv:2202.05453.

## Dependencies
- numpy
- scipy
