#!/usr/bin/python

import numpy as np
import seaborn

from matplotlib import pyplot as plt
from scipy.stats import linregress, spearmanr, pearsonr
from scipy import odr
from sklearn import metrics
from sklearn.decomposition import PCA

from scipy.odr import Model, Data, ODR


def _lin_f(p, x):
    """Basic linear regression 'model' for use with ODR"""
    return (p[0] * x) + p[1]


def _slope_one(p, x):
    """Basic linear regression 'model' for use with ODR"""
    return x + p[0]


def log_spearmanr(log_xs, log_ys):
    mask = ~np.isnan(log_xs) & ~np.isnan(log_ys)
    xs_ = log_xs[mask]
    ys_ = log_ys[mask]
    return spearmanr(xs_, ys_)


def log_linregress(log_xs, log_ys):
    """Linregress for logscale data - filters out NaN."""
    mask = ~np.isnan(log_xs) & ~np.isnan(log_ys)
    xs_ = log_xs[mask]
    ys_ = log_ys[mask]
    return linregress(xs_, ys_)


def fit_power_law(log_xs, log_ys):
    """Returns exponent, prefactor, r2, p_val, exponent std_err.
    
    Fit a line to the log-scaled data using ordinary least squares.
    """
    mask = ~np.isnan(log_xs) & ~np.isnan(log_ys)
    xs_ = log_xs[mask]
    ys_ = log_ys[mask]
    
    slope, intercept, r_val, p_val, stderr = linregress(xs_, ys_)
    
    prefactor = np.exp(intercept)
    exponent = slope
    return exponent, prefactor, r_val, p_val, stderr


def fit_power_law_odr(log_xs, log_ys, unit_exp=False):
    """Perform an Orthogonal Distance Regression on the given data,
    using the same interface as the standard scipy.stats.linregress function.
    Arguments:
    x: x data
    y: y data
    Returns:
    [m, c, nan, nan, nan]
    Uses standard ordinary least squares to estimate the starting parameters
    then uses the scipy.odr interface to the ODRPACK Fortran code to do the
    orthogonal distance calculations.
    """
    mask = ~np.isnan(log_xs) & ~np.isnan(log_ys)
    mask &= np.isfinite(log_xs) & np.isfinite(log_ys)
    xs_ = log_xs[mask]
    ys_ = log_ys[mask]

    linreg = linregress(xs_, ys_)
    f = _lin_f
    beta0 = linreg[0:2]
    if unit_exp:
        f = _slope_one
        beta0 = np.array([linreg[1]])

    mod = Model(f)
    dat = Data(xs_, ys_)
    od = ODR(dat, mod, beta0=beta0)
    out = od.run()

    if unit_exp:
        slope = 1
        intercept = out.beta[0]
    else:
        slope, intercept = out.beta

    pred_ys = slope*xs_ + intercept
    r, P = pearsonr(ys_, pred_ys) 

    prefactor = np.exp(intercept)
    exponent = slope
    return exponent, prefactor, r


def bootstrap_power_law_odr(xs, ys, fraction=0.9, rounds=1000):
    """Generate a distribution of slopes and exponents.

    Uses subsampling and orthogonal distance regression to bootstrap a confidence interval.
    If std deviations are given, will also sample x and y from normal dist in each round. 

    Args:
        xs: x values in linear scale (not log)
        ys: y values in linear scale (not log)
        fraction: fraction of values to subsample in each round.
        rounds: number of rounds of bootstrapping to do. 
    """
    # get rid of NaNs
    mask = np.isfinite(xs) & np.isfinite(ys)
    masked_xs = np.array(xs)[mask]
    masked_ys = np.array(ys)[mask]

    tot = np.sum(mask)
    subset_size = int(fraction * tot)

    exponents = []
    prefactors = []
    rs = []
    for _ in range(rounds):
        idxs = np.random.choice(tot, subset_size)
        xs_sub = masked_xs[idxs]
        ys_sub = masked_ys[idxs]

        exp, pre, r = fit_power_law_odr(np.log(xs_sub), np.log(ys_sub))
        exponents.append(exp)
        prefactors.append(pre)
        rs.append(r)

    return np.array(exponents), np.array(prefactors), np.array(rs)


def plot_bootstrapped_ci(xs, exponents, prefactors, ci=0.95, figure=None, color=None, plot_range=True, lw=3):
    # Calculate a CI on the fit from the distribution of bootstrapped values.
    ci_end = (1.0 - ci)/2.0
    conf_range = [100-100*ci_end, 100*ci_end]
    pre_interval = np.percentile(prefactors, conf_range)
    exp_interval = np.percentile(exponents, conf_range)
    
    xs_double = np.vstack([xs, xs]).T
    ci_ys = np.power(xs_double, exp_interval)*pre_interval
    ci_ys2 = np.power(xs_double, exp_interval[::-1])*pre_interval
    ci_ys = np.hstack([ci_ys, ci_ys2])
    min_ys, max_ys = ci_ys.min(axis=1), ci_ys.max(axis=1)
    
    median_pre, median_exp = np.median(prefactors), np.median(exponents)
    median_ys = np.power(xs, median_exp)*median_pre
    
    # plot median
    figure = figure or plt.figure(figsize=(10,10))
    plt.plot(xs, median_ys, ls='--', lw=lw, figure=figure, color=color)
    
    # plot range
    if plot_range:
        plt.fill_between(xs, min_ys, max_ys, figure=figure, alpha=0.2, color=color)


def plot_bootstrapped_range(exponents, prefactors, figure=None):
    """Makes a plot of the parameter distribution and the middle 95% range."""
    pre_interval = np.percentile(prefactors, [2.5, 97.5])
    exp_interval = np.percentile(exponents, [2.5, 97.5])
    print(pre_interval)
    print(exp_interval)

    mean_exp = np.mean(exponents)
    median_exp = np.median(exponents)
    std_exp = np.std(exponents)

    mean_pre = np.mean(prefactors)
    median_pre = np.median(prefactors)
    std_pre = np.std(prefactors)

    colors = seaborn.color_palette('Set3', n_colors=8)

    figure = figure or plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.hist(exponents, bins=50, color=colors[0])
    plt.axvspan(exp_interval[0], exp_interval[1], color='grey', alpha=0.2)
    plt.axvline(mean_exp, ls='--', color=colors[4], alpha=0.6, lw=3)
    plt.axvline(median_exp, ls='--', color='k', alpha=0.6, lw=3)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(np.percentile(exponents, [1, 99]))

    plt.xlabel('Exponent', fontsize=18)

    plt.subplot(1,2,2)
    plt.hist(prefactors, bins=50, color=colors[2])
    plt.axvspan(pre_interval[0], pre_interval[1], color='grey', alpha=0.2, label='95% CI')
    plt.axvline(mean_pre, ls='--', color=colors[4], alpha=0.6, lw=3, label='Mean')
    plt.axvline(median_pre, ls='--', color='k', alpha=0.6, lw=3, label='Median')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(np.percentile(prefactors, [1, 99]))

    plt.xlabel('Exponential Prefactor', fontsize=18)
    plt.legend(loc=1, fontsize=14)