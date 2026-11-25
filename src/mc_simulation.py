"""Exact-Gaussian Monte Carlo validation of the closed-form differential swap
prices (Section 6.1 of Ding, Liu and Rutkowski, "Choice of Collateral Currency
in Differential Swaps"). The joint transition law of the correlated Vasicek
short rates (r^d, r^f) and their accrual-period integrals is simulated exactly
period by period (Cholesky factor of the exact covariance matrix), with no
Euler discretisation, so that any gap to the closed form reflects pure
Monte Carlo sampling error only.

Under the domestic pricing measure Q the foreign short rate has drift c_hat,
not a_hat (Section 3.1; confirmed for the simulation grid by Section 6.4):

    dr_t^f = (c_hat - b_hat * r_t^f) dt + sigma_hat dZ_t^2.

The differential-swap payoff (eq. 4.4) is settled in the domestic currency
with no explicit FX conversion, so simulating (r^d, r^f) under Q is enough:
the FX process, sigma_bar and rho_13 never enter a Monte Carlo path (consistent
with Section 6.3's statement that Q_0 and rho_13 do not affect the price).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.pricing import ModelParams, collateral_adjustment_factor, payment_schedule


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo price estimate and its standard error (per unit notional)."""

    price: float
    std_error: float
    n_paths: int


def _weight_point(s: np.ndarray, delta: float, b: float) -> np.ndarray:
    """Stochastic loading of r_{t_prev+delta} on dW_{t_prev+s}, s in [0,delta]."""
    return np.exp(-b * (delta - s))


def _weight_integral(s: np.ndarray, delta: float, b: float) -> np.ndarray:
    """Stochastic loading of Y=int_{t_prev}^{t_prev+delta} r dv on dW_{t_prev+s}."""
    return (1.0 - np.exp(-b * (delta - s))) / b


def _period_covariance(delta: float, b_d: float, sigma_d: float, b_f: float,
                        sigma_f: float, rho_12: float) -> np.ndarray:
    """Exact 4x4 covariance matrix, order (r_d, r_f, Y_d, Y_f), of the one
    -period transition over an interval of length ``delta``. The transition
    is time-homogeneous (standard OU/Vasicek property), so the same matrix is
    reused for every period of equal length delta."""
    grid = np.linspace(0.0, delta, 4001)

    def cov(f1, b1, s1, f2, b2, s2, rho) -> float:
        integrand = f1(grid, delta, b1) * f2(grid, delta, b2)
        return float(s1 * s2 * rho * np.trapz(integrand, grid))

    var_rd = cov(_weight_point, b_d, sigma_d, _weight_point, b_d, sigma_d, 1.0)
    var_rf = cov(_weight_point, b_f, sigma_f, _weight_point, b_f, sigma_f, 1.0)
    var_yd = cov(_weight_integral, b_d, sigma_d, _weight_integral, b_d, sigma_d, 1.0)
    var_yf = cov(_weight_integral, b_f, sigma_f, _weight_integral, b_f, sigma_f, 1.0)
    cov_rd_rf = cov(_weight_point, b_d, sigma_d, _weight_point, b_f, sigma_f, rho_12)
    cov_rd_yd = cov(_weight_point, b_d, sigma_d, _weight_integral, b_d, sigma_d, 1.0)
    cov_rf_yf = cov(_weight_point, b_f, sigma_f, _weight_integral, b_f, sigma_f, 1.0)
    cov_rd_yf = cov(_weight_point, b_d, sigma_d, _weight_integral, b_f, sigma_f, rho_12)
    cov_rf_yd = cov(_weight_point, b_f, sigma_f, _weight_integral, b_d, sigma_d, rho_12)
    cov_yd_yf = cov(_weight_integral, b_d, sigma_d, _weight_integral, b_f, sigma_f, rho_12)

    sigma = np.array([
        [var_rd, cov_rd_rf, cov_rd_yd, cov_rd_yf],
        [cov_rd_rf, var_rf, cov_rf_yd, cov_rf_yf],
        [cov_rd_yd, cov_rf_yd, var_yd, cov_yd_yf],
        [cov_rd_yf, cov_rf_yf, cov_yd_yf, var_yf],
    ])
    # Symmetrise against residual quadrature asymmetry and check PSD.
    sigma = 0.5 * (sigma + sigma.T)
    eigvals = np.linalg.eigvalsh(sigma)
    assert eigvals.min() > -1e-10, f"period covariance not PSD: {eigvals}"
    return sigma


def simulate_mc_price(mp: ModelParams, T0: float, maturity: float, gamma: float,
                       kappa: float, beta: float, net_carry_bp: float, n_paths: int,
                       seed: int, delta: float = 0.5) -> MCResult:
    """Monte Carlo price of DS_0^{beta,gamma,kappa}(T0,maturity;i) (Propositions
    4.1-4.2), simulated under Q with the exact joint transition law of
    (r_d, r_f, Y_d, Y_f) at every semiannual step from 0 to ``maturity``.

    Periods with U_j < T0 (only present for single-period configurations with
    T0 > 0) roll (r_d, r_f) forward and accumulate the domestic discount
    factor, but do not contribute a coupon to the payoff.
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    dates = payment_schedule(0.0, maturity, delta)
    rng = np.random.default_rng(seed)
    sigma = _period_covariance(delta, mp.b_d, mp.sigma_d, mp.b_f, mp.sigma_f, mp.rho_12)
    chol = np.linalg.cholesky(sigma)

    r_d = np.full(n_paths, mp.r0_d)
    r_f = np.full(n_paths, mp.r0_f)
    cum_log_discount = np.zeros(n_paths)  # int_0^{T_j} r_d dv, accumulated
    pv = np.zeros(n_paths)

    e_bd, e_bf = math.exp(-mp.b_d * delta), math.exp(-mp.b_f * delta)
    c_d, c_f = (1.0 - e_bd) / mp.b_d, (1.0 - e_bf) / mp.b_f

    for j in range(1, len(dates)):
        Uj, Tj = dates[j - 1], dates[j]
        delta_j = Tj - Uj
        assert abs(delta_j - delta) < 1e-9, "payment schedule must be equally spaced"

        draws = rng.standard_normal((n_paths, 4)) @ chol.T
        mean_rd = r_d * e_bd + mp.a_d / mp.b_d * (1.0 - e_bd)
        mean_rf = r_f * e_bf + mp.c_hat / mp.b_f * (1.0 - e_bf)
        mean_yd = r_d * c_d + mp.a_d / mp.b_d * (delta - c_d)
        mean_yf = r_f * c_f + mp.c_hat / mp.b_f * (delta - c_f)

        r_d = mean_rd + draws[:, 0]
        r_f = mean_rf + draws[:, 1]
        Y_d = mean_yd + draws[:, 2]
        Y_f = mean_yf + draws[:, 3]

        cum_log_discount = cum_log_discount + Y_d

        if Uj >= T0 - 1e-9:
            A = collateral_adjustment_factor(Tj, beta, net_carry_bp)
            bracket = np.exp(Y_d) - gamma * np.exp(Y_f) - (1.0 - gamma + delta_j * kappa)
            pv += A * np.exp(-cum_log_discount) * bracket

    price = float(pv.mean())
    std_error = float(pv.std(ddof=1) / math.sqrt(n_paths))
    return MCResult(price=price, std_error=std_error, n_paths=n_paths)


def z_score(mc_price: float, closed_form_price: float, std_error: float) -> float:
    """z = (MC_price - ClosedForm_price) / MC_std_error (Section 6.1)."""
    return (mc_price - closed_form_price) / std_error
