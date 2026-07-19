import numpy as np
from scipy import stats
from scipy.optimize import brentq


def ob_fleming_spending(t, alpha=0.05):
    """
    Lan-DeMets O'Brien-Fleming-type alpha spending function (two-sided).
    """
    z = stats.norm.ppf(1 - alpha / 2)
    return 2 - 2 * stats.norm.cdf(z / np.sqrt(t))

def group_sequential_boundaries(K, alpha=0.05, grid_half_width=10.0, h=0.01):
    """
    Compute two-sided O'Brien-Fleming group sequential stopping
    boundaries on the Z-statistic scale for K equally spaced interim
    looks.
    Returns (z_bounds, information_fractions, cumulative_alpha_spent).
    """
    t = np.array([(k + 1) / K for k in range(K)])
    cum_alpha = ob_fleming_spending(t, alpha)

    x = np.arange(-grid_half_width, grid_half_width + h, h)

    b = np.zeros(K)
    b[0] = np.sqrt(t[0]) * stats.norm.ppf(1 - cum_alpha[0] / 2)
    f = stats.norm.pdf(x, 0, np.sqrt(t[0]))
    f[np.abs(x) > b[0]] = 0.0

    for k in range(1, K):
        dt = t[k] - t[k - 1]
        inc_pdf = stats.norm.pdf(x, 0, np.sqrt(dt))
        g = np.convolve(f, inc_pdf, mode="same") * h

        cum = np.cumsum(g) * h
        m_prev = cum[-1]
        target_survival = 1 - cum_alpha[k]
        target_upper = (m_prev + target_survival) / 2

        idx = np.searchsorted(cum, target_upper)
        b[k] = x[idx]

        f = g.copy()
        f[np.abs(x) > b[k]] = 0.0

    z_bounds = b / np.sqrt(t)
    return z_bounds, t, cum_alpha

def sequential_power(z_bounds, t, theta, grid_half_width=10.0, h=0.01):
    """
    Probability of crossing the boundary at some look, given a
    standardized drift theta (each Z_k has mean theta*sqrt(t_k) under
    this alternative).
    """
    K = len(t)
    b = z_bounds * np.sqrt(t)
    x = np.arange(-grid_half_width, grid_half_width + h, h)

    f = stats.norm.pdf(x, theta * t[0], np.sqrt(t[0]))
    f[np.abs(x) > b[0]] = 0.0
    cross = 1 - np.sum(f) * h

    for k in range(1, K):
        dt = t[k] - t[k - 1]
        inc_pdf = stats.norm.pdf(x, theta * dt, np.sqrt(dt))
        g = np.convolve(f, inc_pdf, mode="same") * h
        mass_before = np.sum(g) * h
        inside = np.abs(x) <= b[k]
        cross += mass_before - np.sum(g[inside]) * h
        f = g.copy()
        f[~inside] = 0.0
    return cross


def inflation_factor(z_bounds, t, alpha, power, grid_half_width=10.0, h=0.01):
    """
    Ratio of the maximum sample size needed under a group sequential
    design to what a fixed-sample (single-analysis) design needs for
    the same alpha and power.
    """
    z_alpha2 = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    fixed_ncp = z_alpha2 + z_beta

    def f(theta):
        return sequential_power(z_bounds, t, theta, grid_half_width, h) - power

    theta_needed = brentq(f, 0.1, 10)
    return (theta_needed / fixed_ncp) ** 2