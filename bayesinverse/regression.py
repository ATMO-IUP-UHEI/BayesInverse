import numpy as np
from pathlib import Path
import scipy.sparse as sparse
import scipy

import xarray as xr

# from .utils import .


class Regression:
    """
    Performs a regression based on the given parameters.

    Attributes
    ----------
    y : (m,) ndarray
        1-D measurement vector.
    K : (m, n) ndarray
        2-D forward model.
    x_prior : (n,) ndarray or None
        1-D vector of the prior estimate for the state.
    x_covariance : (n,) or (n, n) ndarray or None
        1-D or 2-D covariance of the prior estimate for the state.
    y_covariance : (m,) or (m, m) ndarray or None
        1-D variance of the measurement vector.
    x_covariance_inv_sqrt : (n, n) ndarray or None
        Square-root of the inverse covariance matrix `x_covariance`.
        
    Methods
    -------
    fit(cond=None)
        Performs a regression on the given input.
    """

    def __init__(
        self,
        y,
        K,
        x_prior=None,
        x_covariance=None,
        y_covariance=None,
    ):
        self.y = np.array(y)
        self.K = np.array(K)
        self.x_prior = np.array(x_prior)
        self.x_covariance = np.array(x_covariance)
        self.y_covariance = np.array(y_covariance)

        self.x_covariance_inv_sqrt = np.array(None)

    def fit(self, cond=None):
        if (
            len(self.x_prior.shape) == 0
            and len(self.x_covariance.shape) == 0
            and len(self.y_covariance.shape) == 0
        ):
            # Standard Least-Squares
            y_reg = self.y
            K_reg = self.K
        elif len(self.x_covariance.shape) == 1 and len(self.y_covariance.shape) == 1:
            # Bayesian Inversion with diagonal covariance matrix
            y_reg = np.concatenate(
                (
                    self.y / np.sqrt(self.y_covariance),
                    self.x_prior / np.sqrt(self.x_covariance),
                )
            )
            K_reg = np.concatenate(
                (
                    np.power(np.sqrt(self.y_covariance), -1).reshape(-1, 1) * self.K,
                    np.diag(np.power(np.sqrt(self.x_covariance), -1)),
                ),
            )
        elif len(self.x_covariance.shape) == 2 and len(self.y_covariance.shape) == 1:
            # Bayesian Inversion with off-diagonal covariance matrix
            # Inverse and square-root with Cholesky-Decomposition
            l = scipy.linalg.cholesky(self.x_covariance)
            self.x_covariance_inv_sqrt = np.linalg.inv(l)
            y_reg = np.concatenate(
                (
                    self.y / np.sqrt(self.y_covariance),
                    self.x_covariance_inv_sqrt @ self.x_prior,
                )
            )
            K_reg = np.concatenate(
                (
                    np.power(np.sqrt(self.y_covariance), -1).reshape(-1, 1) * self.K,
                    self.x_covariance_inv_sqrt,
                ),
            )

        self.x, self.res, self.rank, self.s = scipy.linalg.lstsq(
            K_reg, y_reg, cond=cond
        )
        return self.x
