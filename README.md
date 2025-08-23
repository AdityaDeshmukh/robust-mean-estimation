# Robust Mean Estimation
Robust mean estimation is a statistical technique designed to compute a reliable mean in the presence of outliers and heavy-tailed inliers.
Two functions are provided:
1) `robust_mean` : this algorithm does not require the knowledge of fraction of outliers
2) `meta_robust_mean` :  this algorithm does not require the knowledge of fraction of outliers, or the knowledge of spectral bound
   
Implementation of algorithms from the following references:
1) [Topics in statistical inference with model uncertainty](https://www.ideals.illinois.edu/items/131409/bitstreams/436773/data.pdf).
2) Deshmukh, A., Liu, J., & Veeravalli, V. V. (2023). [Robust mean estimation in high dimensions: An outlier-fraction agnostic and efficient algorithm](https://arxiv.org/abs/2102.08573). IEEE Transactions on Information Theory, 69(7), 4675-4690.
3) [Robust estimation algorithms don’t need to know the corruption level](https://arxiv.org/pdf/2202.05453).

This repository contains a NumPy + SciPy implementation of robust mean estimation algorithms.
