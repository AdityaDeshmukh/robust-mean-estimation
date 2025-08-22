# Robust Mean Estimation
Robust mean estimation is a statistical technique designed to compute a reliable mean in the presence of outliers.
Unlike the vanilla sample mean, which is highly sensitive to corrupted data, this implementation iteratively filters outliers by projecting data onto high-variance directions and trimming based on a variance budget.

This repository contains a NumPy + SciPy implementation of a robust mean estimation algorithm.
