"""Deterministic quanto correction Gamma_t^{qf}(U,T), equation (4.9) of
Ding, Liu and Rutkowski, "Choice of Collateral Currency in Differential Swaps"
(Section 4.1, page 16).

With sigma_S(u) := -sigma * n(u,S) and sigma_hat_S(u) := -sigma_hat * n_hat(u,S),

    Gamma_t^{qf}(U,T) =
        exp( int_t^U (rho_12*sigma_T(u) - rho_23*sigma_bar) * sigma_hat_U(u) du
             - int_t^T (rho_12*sigma_T(u) - rho_23*sigma_bar) * sigma_hat_T(u) du ),  t <= U,
        exp( -int_t^T (rho_12*sigma_T(u) - rho_23*sigma_bar) * sigma_hat_T(u) du ),   t >= U.

The correction depends only on volatility and correlation parameters, not on
the Vasicek drift constants: it is the covariance correction that converts the
"raw" domestic-measure exponential moment into the quoted €STR futures rate
(equation 4.10 and its proof, Section 4.1).
"""
from __future__ import annotations

import numpy as np
from scipy import integrate

from src.vasicek import n_func


def _sigma_S(u, S: float, sigma: float, b: float):
    return -sigma * n_func(u, S, b)


def _integrand(u, T: float, U: float, sigma: float, b: float, sigma_hat: float, b_hat: float,
               rho_12: float, rho_23: float, sigma_bar: float, hat_target: float):
    """Integrand (rho_12*sigma_T(u) - rho_23*sigma_bar) * sigma_hat_{hat_target}(u)."""
    sT = _sigma_S(u, T, sigma, b)
    s_hat = _sigma_S(u, hat_target, sigma_hat, b_hat)
    return (rho_12 * sT - rho_23 * sigma_bar) * s_hat


def quanto_correction(t: float, U: float, T: float, sigma: float, b: float,
                       sigma_hat: float, b_hat: float, rho_12: float, rho_23: float,
                       sigma_bar: float) -> float:
    """Gamma_t^{qf}(U,T) from equation (4.9)."""
    if T <= U:
        raise ValueError("quanto_correction requires T > U")
    exponent = 0.0
    if t <= U:
        if U > t:
            val_U, _ = integrate.quad(
                _integrand, t, U,
                args=(T, U, sigma, b, sigma_hat, b_hat, rho_12, rho_23, sigma_bar, U),
                limit=200,
            )
            exponent += val_U
        val_T, _ = integrate.quad(
            _integrand, t, T,
            args=(T, U, sigma, b, sigma_hat, b_hat, rho_12, rho_23, sigma_bar, T),
            limit=200,
        )
        exponent -= val_T
    else:
        val_T, _ = integrate.quad(
            _integrand, t, T,
            args=(T, U, sigma, b, sigma_hat, b_hat, rho_12, rho_23, sigma_bar, T),
            limit=200,
        )
        exponent = -val_T
    return float(np.exp(exponent))


def _integral_fg(t0, X, T: float, S: float, sigma: float, b: float, sigma_hat: float,
                  b_hat: float, rho_12: float, rho_23: float, sigma_bar: float):
    """Closed-form antiderivative of int_{t0}^{X} (rho_12*sigma_T(u) - rho_23*sigma_bar)
    * sigma_hat_S(u) du, obtained by expanding the two piecewise-exponential
    loadings sigma_T(u) = -sigma*n(u,T) and sigma_hat_S(u) = -sigma_hat*n_hat(u,S)
    into constant + exponential terms and integrating each term analytically
    (equation 4.9). Vectorised over numpy arrays t0, X (e.g. simulation grid
    points), scalar T, S.

    Algebraically equivalent to numerically integrating ``_integrand`` (used
    by :func:`quanto_correction`); kept separate and closed-form because
    Experiment 4's discrete hedge evaluates this quantity millions of times
    (every path, every rebalancing date, every outstanding coupon), for which
    per-call numerical quadrature would be computationally infeasible.
    """
    k1 = -rho_12 * sigma / b - rho_23 * sigma_bar
    k2 = -sigma_hat / b_hat
    k3 = sigma_hat / b_hat
    rsb = rho_12 * sigma / b

    term1 = k1 * k2 * (X - t0)
    term2 = k1 * k3 / b_hat * (np.exp(-b_hat * (S - X)) - np.exp(-b_hat * (S - t0)))
    term3 = rsb * k2 / b * (np.exp(-b * (T - X)) - np.exp(-b * (T - t0)))
    term4 = rsb * k3 / (b + b_hat) * (
        np.exp(-b * (T - X) - b_hat * (S - X)) - np.exp(-b * (T - t0) - b_hat * (S - t0))
    )
    return term1 + term2 + term3 + term4


def quanto_correction_closed_form(t, U: float, T: float, sigma: float, b: float,
                                   sigma_hat: float, b_hat: float, rho_12: float,
                                   rho_23: float, sigma_bar: float):
    """Closed-form, numpy-vectorised evaluation of Gamma_t^{qf}(U,T) (eq. 4.9),
    for scalar or array-valued ``t`` (e.g. a vector over Monte Carlo paths),
    with U <= T fixed scalars. Used by the discrete hedge of Section 5
    (Experiment 4); validated against the quadrature-based
    :func:`quanto_correction` in ``tests/test_quanto.py``.
    """
    if T <= U:
        raise ValueError("quanto_correction_closed_form requires T > U")
    t = np.asarray(t, dtype=float)
    le_U = t <= U
    val_U = np.where(
        le_U,
        _integral_fg(t, np.where(le_U, U, t), T, U, sigma, b, sigma_hat, b_hat, rho_12, rho_23, sigma_bar),
        0.0,
    )
    val_T = _integral_fg(t, T, T, T, sigma, b, sigma_hat, b_hat, rho_12, rho_23, sigma_bar)
    exponent = val_U - val_T
    return np.exp(exponent)
