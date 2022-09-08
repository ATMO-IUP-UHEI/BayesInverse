import numpy as np
from scipy.linalg import cholesky, lstsq


class LeastSquares:
    def __init__(
        self,
        y,
        K,
    ):
        self.y = np.array(y)
        self.K = np.array(K)

    def get_reg_params(self, alpha=0.0):
        reg_vector = np.sqrt(alpha) * np.ones(self.K.shape[1])
        self.y_reg = np.concatenate((self.y, reg_vector))
        self.K_reg = np.concatenate((self.K, np.diag(reg_vector)))
        return self.y_reg, self.K_reg

    def get_loss_terms(self, x, alpha):
        loss_regularization = alpha * np.sum(x**2)
        loss_least_squares = np.sum((self.y - self.K @ x) ** 2)
        return loss_regularization, loss_least_squares


class BayInv:
    def __init__(
        self,
        y,
        K,
        x_prior,
        x_covariance,
        y_covariance,
    ):
        self.y = np.array(y)
        self.K = np.array(K)
        self.x_prior = np.array(x_prior)
        self.x_covariance = np.array(x_covariance)
        self.y_covariance = np.array(y_covariance)

    def get_reg_params(self, alpha=1.0):
        self.y_reg = np.concatenate(
            (
                self.y / np.sqrt(self.y_covariance),
                np.sqrt(alpha) * self.x_prior / np.sqrt(self.x_covariance),
            )
        )
        self.K_reg = np.concatenate(
            (
                np.power(np.sqrt(self.y_covariance), -1).reshape(-1, 1) * self.K,
                np.sqrt(alpha) * np.diag(np.power(np.sqrt(self.x_covariance), -1)),
            ),
        )
        return self.y_reg, self.K_reg

    def get_loss_terms(self, x, alpha):
        loss_regularization = alpha * (
            (self.x_prior - x) @ np.diag(1 / self.x_covariance) @ (self.x_prior - x)
        )
        loss_least_squares = np.sum((self.y - self.K @ x) ** 2)
        return loss_regularization, loss_least_squares


class BayInvCov:
    def __init__(
        self,
        y,
        K,
        x_prior,
        x_covariance,
        y_covariance,
    ):
        self.y = np.array(y)
        self.K = np.array(K)
        self.x_prior = np.array(x_prior)
        self.x_covariance = np.array(x_covariance)
        self.y_covariance = np.array(y_covariance)

        # Inverse and square-root with Cholesky-Decomposition
        l = cholesky(self.x_covariance)
        self.x_covariance_inv_sqrt = np.linalg.inv(l)

    def get_reg_params(self, alpha=1.0):
        self.y_reg = np.concatenate(
            (
                self.y / np.sqrt(self.y_covariance),
                np.sqrt(alpha) * self.x_covariance_inv_sqrt @ self.x_prior,
            )
        )
        self.K_reg = np.concatenate(
            (
                np.power(np.sqrt(self.y_covariance), -1).reshape(-1, 1) * self.K,
                np.sqrt(alpha) * self.x_covariance_inv_sqrt,
            ),
        )
        return self.y_reg, self.K_reg

    def get_loss_terms(self, x, alpha):
        loss_regularization = alpha * (
            (self.x_prior - x)
            @ self.x_covariance_inv_sqrt
            @ self.x_covariance_inv_sqrt.T
            @ (self.x_prior - x)
        )
        loss_least_squares = np.sum((self.y - self.K @ x) ** 2)
        return loss_regularization, loss_least_squares


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
        """
        Depending on the given parameters, three types of fits are possible:
        1. Simple Least Squares
            (y, K)

        2. Bayesian inversion with diagonal covariance
            (y, K, x_prior, x_covariance, y_covariance)
            with `x_covariance` and `y_covariance` as 1-D array_like.

        3. Bayesian inversion with full covariance
            (y, K, x_prior, x_covariance, y_covariance)
            with `x_covariance` as 2-D array_like and `y_covariance` as 1-D array_like.

        Parameters
        ----------
        y : (m,) array_like
            1-D measurement vector.
        K : (m, n) array_like
            2-D forward model.
        x_prior : (n,) array_like or None
            1-D vector of the prior estimate for the state.
        x_covariance : (n,) or (n, n) array_like or None
            1-D or 2-D covariance of the prior estimate for the state.
        y_covariance : (m,) or (m, m) array_like or None
            1-D variance of the measurement vector.
        """
        self.y = np.array(y)
        self.K = np.array(K)
        self.x_prior = np.array(x_prior)
        self.x_covariance = np.array(x_covariance)
        self.y_covariance = np.array(y_covariance)

        # Choose model
        # Standard Least-Squares
        if (
            len(self.x_prior.shape) == 0
            and len(self.x_covariance.shape) == 0
            and len(self.y_covariance.shape) == 0
        ):
            self.model = LeastSquares(self.y, self.K)
        # Bayesian Inversion with diagonal covariance matrix
        elif len(self.x_covariance.shape) == 1 and len(self.y_covariance.shape) == 1:
            self.model = BayInv(
                self.y, self.K, self.x_prior, self.x_covariance, self.y_covariance
            )
        # Bayesian Inversion with off-diagonal covariance matrix
        elif len(self.x_covariance.shape) == 2 and len(self.y_covariance.shape) == 1:
            self.model = BayInvCov(
                self.y, self.K, self.x_prior, self.x_covariance, self.y_covariance
            )
        # Store inversion parameters
        self.y_reg, self.K_reg = self.model.get_reg_params()

    def fit(self, cond=None):
        """
        Fit the given forward model depending on the given parameters during the init.

        Parameters
        ----------
        cond : float, optional
            Cutoff for 'small' singular values; used to determine effective rank of a.
            Singular values smaller than cond * largest_singular_value are considered
            zero.

        Returns
        -------
        x_est : ndarray
            The estimated state vector.
        res : ndarray or float
            Residues of the loss function.
        rank : int
            Effective rank of the (adapted) forward matrix.
        s : ndarray or None
            Singular values of the (adapted) forward matrix.

        Note
        ----
        The residues, the rank, and the singular values are not from the forward matrix
        K in the case of the Bayesian Inversion! But from the adapted forward matrix.
        """
        self.x_est, self.res, self.rank, self.s = lstsq(
            self.K_reg, self.y_reg, cond=cond
        )
        return self.x_est, self.res, self.rank, self.s

    def compute_l_curve(self, alpha_list=[0.1, 1.0, 10.0], cond=None):
        """
        Compute the so-called l-curve.

        Parameters
        ----------
        alpha_list : list of float, optional
            List of the regularization parameters, by default None
        cond : float, optional
            Cutoff for 'small' singular values; used to determine effective rank of a.
            Singular values smaller than cond * largest_singular_value are considered
            zero.

        Returns
        -------
        inversion_params : dict
            The resulting output from the inversions. The dictionary contains lists with
            entries for each `alpha` of `alpha_list`:
             - "x_est" : The estimated state vector.
             - "res" : Residues of the loss function.
             - "rank" : Effective rank of the (adapted) forward matrix.
             - "s" : Singular values of the (adapted) forward matrix.
             - "loss_regularization" : The values of the regularization term of the inversion equation.
             - "loss_forward_model" : The values of the measurement term of the inversion equation.

        Examples
        --------
        To plot the l-curve:

        >>> inv_params = regression.compute_l_curve()
        >>> matplotlib.pyplot.plot(
        ...     inv_params["loss_regularization"],
        ...     inv_params["loss_forward_model"],
        ... )

        """
        inversion_params = dict(
            alpha=alpha_list,
            x_est=[],
            res=[],
            rank=[],
            s=[],
            loss_regularization=[],
            loss_forward_model=[],
        )
        for alpha in alpha_list:
            # Update inversion parameters
            self.y_reg, self.K_reg = self.model.get_reg_params(alpha)
            x_est, res, rank, s = self.fit(cond=cond)
            inversion_params["x_est"].append(x_est)
            inversion_params["res"].append(res)
            inversion_params["rank"].append(rank)
            inversion_params["s"].append(s)
            loss_regularization, loss_forward_model = self.model.get_loss_terms(
                x=self.x_est, alpha=alpha
            )
            inversion_params["loss_regularization"].append(loss_regularization)
            inversion_params["loss_forward_model"].append(loss_forward_model)
        return inversion_params
