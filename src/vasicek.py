"""Vasicek building blocks: auxiliary discount factors and the exact Gaussian
law of integrated short rates (Section 3.2 and Appendix formulas used in
Section 6.1 of Ding, Liu and Rutkowski, "Choice of Collateral Currency in
Differential Swaps").

For an OU/Vasicek process dr_t = (theta - b*r_t) dt + sigma dW_t, the discount
factor B(t,S) = E[exp(-int_t^S r_v dv) | F_t] has the standard affine form

    B(t,S) = exp(A(t,S) - r_t * C(t,S)),   C(t,S) = (1 - exp(-b*(S-t))) / b,
    A(t,S) = (theta/b - sigma^2/(2 b^2)) * (S - t - C(t,S))
             - sigma^2/(4 b) * C(t,S)^2.

The mean and variance of the integral Y = int_t^S r_v dv follow from matching
the exponential-moment identity E[exp(-Y)] = exp(-E[Y] + 0.5*Var[Y]) with the
decomposition of r_v via its stochastic-integral representation (standard
Vasicek result, e.g. Brigo & Mercurio, "Interest Rate Models"):

    E[Y] = r_t * C(t,S) + theta/b * (S - t - C(t,S)),
    Var[Y] = sigma^2 / b^2 * [ (S-t) - 2*C(t,S) + (1 - exp(-2b(S-t))) / (2b) ].
"""
from __future__ import annotations

import numpy as np


def n_func(u: np.ndarray, S: float, b: float) -> np.ndarray:
    """Vasicek loading n(u, S) = (1 - exp(-b*(S-u))) / b used in sigma_S(u)."""
    tau = S - u
    return (1.0 - np.exp(-b * tau)) / b


def discount_factor(t: float, S: float, r_t, theta: float, b: float, sigma: float):
    """Conditional Vasicek discount factor B(t,S) = E[exp(-int_t^S r dv) | F_t].

    Y = int_t^S r_v dv is Gaussian (mean, var from ``integrated_rate_moments``),
    so B(t,S) = E[exp(-Y)] = exp(-E[Y] + 0.5*Var[Y]) by the Gaussian MGF.
    ``r_t`` may be a scalar or a numpy array (vectorised over Monte Carlo
    paths, used by the discrete hedge of Section 5); the return type follows
    numpy broadcasting rules and matches the input.
    """
    if S <= t:
        return 1.0 + 0.0 * np.asarray(r_t)
    mean, var = integrated_rate_moments(t, S, r_t, theta, b, sigma)
    return np.exp(-mean + 0.5 * var)


def integrated_rate_moments(t: float, S: float, r_t, theta: float, b: float, sigma: float):
    """Mean and variance of Y = int_t^S r_v dv given r_t, under drift theta."""
    if S <= t:
        return 0.0 * np.asarray(r_t), 0.0 * np.asarray(r_t)
    tau = S - t
    c = (1.0 - np.exp(-b * tau)) / b
    mean = r_t * c + theta / b * (tau - c)
    var = sigma ** 2 / b ** 2 * (tau - 2.0 * c + (1.0 - np.exp(-2.0 * b * tau)) / (2.0 * b))
    return mean, var


def conditional_exp_integral(t: float, U: float, T: float, r_t, theta: float,
                              b: float, sigma: float, realized=0.0):
    """E[exp(int_U^T r_v dv) | F_t] for an OU/Vasicek process with drift theta,
    for t <= U <= T (tower property + Gaussian affine structure; standard
    Vasicek result underlying the futures-rate formulas of Section 3.2).

    ``realized`` is the already-elapsed part int_U^t r_v dv, supplied when
    U <= t <= T (used by the discrete hedge of Section 5 on a simulation
    grid that has moved past the accrual start U). ``r_t`` and ``realized``
    may be scalars or numpy arrays (vectorised over Monte Carlo paths).
    """
    if T <= U:
        raise ValueError("conditional_exp_integral requires T > U")
    if t <= U:
        tau_tU = U - t
        if tau_tU > 0:
            mu = r_t * np.exp(-b * tau_tU) + theta / b * (1.0 - np.exp(-b * tau_tU))
            v = sigma ** 2 / (2.0 * b) * (1.0 - np.exp(-2.0 * b * tau_tU))
        else:
            mu, v = r_t, 0.0 * np.asarray(r_t)
        tau_UT = T - U
        c = (1.0 - np.exp(-b * tau_UT)) / b
        w = sigma ** 2 / b ** 2 * (tau_UT - 2.0 * c + (1.0 - np.exp(-2.0 * b * tau_UT)) / (2.0 * b))
        exponent = c * mu + theta / b * (tau_UT - c) + 0.5 * c ** 2 * v + 0.5 * w
        return np.exp(exponent)
    else:  # U <= t <= T
        mean, var = integrated_rate_moments(t, T, r_t, theta, b, sigma)
        return np.exp(np.asarray(realized) + mean + 0.5 * var)


def integral_covariance(t: float, U1: float, S1: float, U2: float, S2: float,
                         b1: float, sigma1: float, b2: float, sigma2: float, rho: float) -> float:
    """Covariance of X = int_{U1}^{S1} r^{(1)} dv and Y = int_{U2}^{S2} r^{(2)} dv,
    where the two OU processes share drivers with instantaneous correlation rho.

    Uses the stochastic-Fubini representation
        int_{U}^{S} r_v dv = deterministic + sigma * int_{U}^{S} n(s,S) dW_s
    restricted to [max(U1,U2), min(S1,S2)] ∩ [t, ∞), the only region where both
    Brownian increments are shared and unrealised as of t.
    """
    lo = max(U1, U2, t)
    hi = min(S1, S2)
    if hi <= lo:
        return 0.0
    s = np.linspace(lo, hi, 2001)
    n1 = n_func(s, S1, b1)
    n2 = n_func(s, S2, b2)
    integrand = n1 * n2
    cov = sigma1 * sigma2 * rho * np.trapz(integrand, s)
    return float(cov)
