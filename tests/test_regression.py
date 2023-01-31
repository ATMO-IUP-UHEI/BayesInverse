"""
Test the regression class from the module.
"""

import numpy as np

from context import Regression


def test_regression():
    """
    Test all three variants of the regression.

     - Simple Least Squares
     - Bayesian diagonal covariance
     - Bayesian with full covariance

    and the Cholesky decomposition.
    """
    tol = 1e-5
    m = 10  # Measurement dimension
    n = 21  # State dimension
    rng = np.random.default_rng(0)
    x_prior = rng.normal(size=n)
    K = rng.normal(size=(m, n))
    y = K @ x_prior
    # Create covariance matrix
    x_covariance = np.zeros((n, n))
    var = rng.normal(size=n) ** 2
    for col in range(n):
        for row in range(col + 1):
            if row == col:
                x_covariance[row, col] = var[row]
            else:
                cor = (rng.random(size=1) - 0.5) * 0.1
                x_covariance[row, col] = np.sqrt(var[row]) * cor * np.sqrt(var[col])
                x_covariance[col, row] = x_covariance[row, col]

    y_covariance = rng.normal(size=m) ** 2

    # Simple Least Squares
    reg = Regression(y, K)
    x_est, res, rank, s = reg.fit()
    inv_params = reg.compute_l_curve()

    # Bayesian diagonal covariance
    reg = Regression(y, K, x_prior, np.diag(x_covariance), y_covariance)
    x_est, res, rank, s = reg.fit()
    inv_params = reg.compute_l_curve()
    posterior_covariance = reg.get_posterior_covariance()
    assert posterior_covariance.shape == (n, n)
    gain = reg.get_gain()
    assert gain.shape == (n, m)
    averaging_kernel = reg.get_averaging_kernel()
    assert averaging_kernel.shape == (n, n)
    dof_signal = reg.get_dof_signal()
    dof_noise = reg.get_dof_noise()
    assert (dof_noise + dof_signal) - n <= tol
    information_content = reg.get_information_content()
    assert information_content.shape == tuple()

    # Bayesian with full covariance
    reg = Regression(y, K, x_prior, x_covariance, y_covariance)
    x_est, res, rank, s = reg.fit()
    inv_params = reg.compute_l_curve()
    posterior_covariance = reg.get_posterior_covariance()
    assert posterior_covariance.shape == (n, n)
    gain = reg.get_gain()
    assert gain.shape == (n, m)
    averaging_kernel = reg.get_averaging_kernel()
    assert averaging_kernel.shape == (n, n)
    dof_signal = reg.get_dof_signal()
    dof_noise = reg.get_dof_noise()
    assert (dof_noise + dof_signal) - n <= tol
    information_content = reg.get_information_content()
    assert information_content.shape == tuple()
    error_reduction = reg.get_error_reduction()
    assert error_reduction.shape == (n,)

    # Test Cholesky decomposition
    x_covariance_inv_sqrt = reg.model.x_covariance_inv_sqrt
    assert np.allclose(
        x_covariance_inv_sqrt @ x_covariance_inv_sqrt.T @ x_covariance,
        np.eye(n),
    )

    # With alpha
    alpha = 0.5

    # Simple Least Squares
    reg = Regression(y, K, alpha=alpha)
    x_est, res, rank, s = reg.fit()
    inv_params = reg.compute_l_curve()

    reg.set_y(2 * y)

    # Bayesian diagonal covariance
    reg = Regression(y, K, x_prior, np.diag(x_covariance), y_covariance, alpha=alpha)
    x_est, res, rank, s = reg.fit()
    inv_params = reg.compute_l_curve()

    reg.set_y(2 * y)
    reg.set_x_covariance(np.diag(x_covariance))
    reg.set_y_covariance(y_covariance)

    posterior_covariance = reg.get_posterior_covariance()
    assert posterior_covariance.shape == (n, n)
    assert np.allclose(posterior_covariance, posterior_covariance.T)
    correlation = reg.get_correlation()
    assert np.allclose(np.diag(correlation), 1.0)
    gain = reg.get_gain()
    assert gain.shape == (n, m)
    averaging_kernel = reg.get_averaging_kernel()
    assert averaging_kernel.shape == (n, n)
    error_reduction = reg.get_error_reduction()
    assert error_reduction.shape == (n,)

    # Bayesian with full covariance
    reg = Regression(y, K, x_prior, x_covariance, y_covariance, alpha=alpha)
    x_est, res, rank, s = reg.fit()
    inv_params = reg.compute_l_curve()
    
    reg.set_y(2 * y)
    reg.set_x_covariance(x_covariance)
    reg.set_y_covariance(y_covariance)

    posterior_covariance = reg.get_posterior_covariance()
    assert posterior_covariance.shape == (n, n)
    assert np.allclose(posterior_covariance, posterior_covariance.T)
    correlation = reg.get_correlation()
    assert np.allclose(np.diag(correlation), 1.0)
    gain = reg.get_gain()
    assert gain.shape == (n, m)
    averaging_kernel = reg.get_averaging_kernel()
    assert averaging_kernel.shape == (n, n)
    error_reduction = reg.get_error_reduction()
    assert error_reduction.shape == (n,)

    # Test Cholesky decomposition
    x_covariance_inv_sqrt = reg.model.x_covariance_inv_sqrt
    assert np.allclose(
        x_covariance_inv_sqrt @ x_covariance_inv_sqrt.T @ x_covariance,
        np.eye(n),
    )
