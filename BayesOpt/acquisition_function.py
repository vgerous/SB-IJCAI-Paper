import warnings
import numpy as np
from scipy.stats import norm

def ei_acquisition(x, gp, y_max, xi):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean, std = gp.predict(x, return_std=True)

    z = (mean - y_max - xi)/(std + 1e-12)
    return [((mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z))[0], std[0]]