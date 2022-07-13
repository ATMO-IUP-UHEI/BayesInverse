# BayesInverse

## Overview

Python module to perform the Bayesian Inversion of a forward model
$$y = Kx$$
with $y$ as the measurement vector, $K$ as the forward model matrix, and $x$ as the state vector.

## Installation
```bash
cd <path to repository>
pip install .
```
You can test the installation with
```bash
cd <path to repository>
pytest
```
## Example

```python
import numpy as np
from bayesinverse import Regression


m = 10  # Measurement dimension
n = 21  # State dimension

rng = np.random.default_rng(0)

x_prior = rng.normal(size=n)
K = rng.normal(size=(m, n))
y = K @ x_prior

# Create covariance vectors
x_covariance = rng.normal(size=n) ** 2
y_covariance = rng.normal(size=m) ** 2

# Simple Least Squares
reg = Regression(y, K)
x_reg = reg.fit()

# Bayesian diagonal covariance
reg = Regression(y, K, x_prior, x_covariance, y_covariance)
x_reg = reg.fit()
```
