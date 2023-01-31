from context import Regression
import numpy as np

def test_optimal_alpha():
    y = np.random.normal(size=10)
    k = np.random.normal(size=(10,10))
    x = np.random.normal(size=10)
    xerr = np.abs(np.random.normal(size=10)/2)
    yerr = np.abs(np.random.normal(size=10)/2)
    reg = Regression(y, k, x, xerr, yerr)

    alpha_low = 1e-6
    alpha_high = 1

    alpha_opt = reg.optimal_alpha((alpha_low, alpha_high), 0.1)

    assert alpha_opt > alpha_low
    assert alpha_opt < alpha_high