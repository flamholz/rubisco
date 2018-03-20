#!/usr/bin/python

import numpy as np

from scipy.stats import linregress
from scipy import odr
from sklearn import metrics
from sklearn.decomposition import PCA


def _linear(B, xs):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*xs + B[1]


def fit_power_law(log_xs, log_ys):
    """Returns exponent, prefactor, r2, p_val, exponent std_err.
    
    Fit a line to the log-scaled data using ordinary least squares.
    """
    mask = ~np.isnan(log_xs) & ~np.isnan(log_ys)
    xs_ = log_xs[mask]
    ys_ = log_ys[mask]
    
    slope, intercept, r_val, p_val, stderr = linregress(xs_, ys_)
    r2 = r_val**2
    
    prefactor = np.exp(intercept)
    exponent = slope
    return exponent, prefactor, r2, p_val, stderr


def fit_power_law_tls(log_xs, log_ys):
    """Returns exponent, prefactor, r2.
    
    Fit a line to the log-scaled data using total least squares.

    TODO: figure out how to calc p-val for this.
    What is null model? Do we even care? Could bootstrap.
    
    Implemented via PCA too.
    """
    mask = ~np.isnan(log_xs) & ~np.isnan(log_ys)
    mask &= np.isfinite(log_xs) & np.isfinite(log_ys)
    xs_ = log_xs[mask]
    ys_ = log_ys[mask]
    
    data = np.vstack([xs_, ys_]).T

    pca = PCA(n_components=1)
    fitted = pca.fit(data)
    comp0 = fitted.components_[0]

    slope = comp0[1] / comp0[0]
    intercept = fitted.mean_[1] - slope * fitted.mean_[0]
    pred_ys = slope*xs_ + intercept
    r2 = metrics.r2_score(ys_, pred_ys)
    
    prefactor = np.exp(intercept)
    exponent = slope
    return exponent, prefactor, r2
