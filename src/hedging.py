"""Discrete-time implementation of the futures-based replicating strategies of
Proposition 5.2 (Sections 5.1-5.2, Experiment of Section 6.4) of Ding, Liu and
Rutkowski, "Choice of Collateral Currency in Differential Swaps".

State variables (r^d, r^f, Q) are simulated under the domestic pricing measure
Q by Euler-Maruyama on a daily grid (Section 6.4):

    dr_t^d = (a_d - b*r_t^d) dt + sigma dZ_t^1
    dr_t^f = (c_hat - b_hat*r_t^f) dt + sigma_hat dZ_t^2
    dQ_t   = Q_t*(r_t^d - r_t^f + lambda^q) dt + Q_t*sigma_bar dZ_t^3,  lambda^q = alpha^d - alpha^f

with lambda^q = 0 under the Section 6.1 baseline (alpha^d = alpha^f = 0). At
each grid point the SOFR/€STR futures rates and the quanto correction are
evaluated in closed form (conditional on the current state), giving the
hedge ratios of equations (5.12) and (5.14); the reduced wealth equation
(5.15) is integrated on the same grid, with the realised SOFR/€STR averages
accrued by left-end Riemann sums and the foreign-futures gain translated to
domestic currency by the right-end FX rule Q_{t_{k+1}}*DeltaF^f_{t_{k+1}}
(Section 2.3, consistent with the implementation notes of Section 6.4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.pricing import ModelParams, collateral_adjustment_factor, multi_period_price, payment_schedule
from src.quanto import quanto_correction_closed_form
from src.vasicek import conditional_exp_integral, discount_factor, n_func


@dataclass(frozen=True)
class HedgeResult:
    """Terminal replication error statistics for one hedging strategy."""

    rmse_bp: float
    variance: float
    terminal_wealth: np.ndarray = field(repr=False)


def _sigma_loading(t: float, S: float, sigma: float, b: float) -> float:
    """sigma_S(t) = -sigma * n(t,S) (Vasicek volatility loading, eq. before 4.9)."""
    return -sigma * n_func(t, S, b)


def _sofr_futures(t: float, U: float, T: float, r_t_d, mp: ModelParams, realized):
    """1 + delta*F_t^d(U,T) = E_Q[exp(int_U^T r^d dv)|F_t], domestic Q-dynamics
    (drift a_d), Section 3.2."""
    return conditional_exp_integral(t, U, T, r_t_d, mp.a_d, mp.b_d, mp.sigma_d, realized)


def _estr_futures(t: float, U: float, T: float, r_t_f, mp: ModelParams, realized):
    """1 + delta*F_t^f(U,T) = E_Qhat[exp(int_U^T r^f dv)|F_t], foreign
    martingale measure Q-hat (drift a_hat), Definition 3.1."""
    return conditional_exp_integral(t, U, T, r_t_f, mp.a_hat, mp.b_f, mp.sigma_f, realized)


def _futures_and_components(t: float, Uj: float, Tj: float, r_d, r_f, mp: ModelParams,
                             beta: float, net_carry_bp: float, realized_d, realized_f):
    """One-stop evaluation, at time t <= Tj, of the quantities needed for both
    the coupon-j hedge ratios and its mark-to-market futures levels:
    I1(t) (only meaningful for t<=Uj), I2(t), I3(t), 1+delta*F_t^d(Uj,Tj) and
    1+delta*F_t^f(Uj,Tj) (Propositions 4.1/4.2, equations 5.12/5.14)."""
    before_start = t <= Uj + 1e-12
    A_t = collateral_adjustment_factor(Tj - t, beta, net_carry_bp)
    B_d_tT = discount_factor(t, Tj, r_d, mp.a_d, mp.b_d, mp.sigma_d)
    gamma_qf = quanto_correction_closed_form(
        t, Uj, Tj, mp.sigma_d, mp.b_d, mp.sigma_f, mp.b_f, mp.rho_12, mp.rho_23, mp.sigma_bar)
    one_plus_dF_d = _sofr_futures(t, Uj, Tj, r_d, mp, 0.0 if before_start else realized_d)
    one_plus_dF_f = _estr_futures(t, Uj, Tj, r_f, mp, 0.0 if before_start else realized_f)
    I3 = A_t * B_d_tT
    I2 = I3 * one_plus_dF_f * gamma_qf
    if before_start:
        B_d_tU = discount_factor(t, Uj, r_d, mp.a_d, mp.b_d, mp.sigma_d)
        I1 = A_t * B_d_tU
    else:
        I1 = None
    return I1, I2, I3, one_plus_dF_d, one_plus_dF_f, gamma_qf


def simulate_hedge(mp: ModelParams, maturity: float, gamma: float, kappa: float,
                    beta: float, net_carry_bp: float, n_paths: int, seed: int,
                    strategy: str, rebalance_every: int = 1, delta: float = 0.5,
                    dt: float = 1.0 / 252.0) -> HedgeResult:
    """Simulate the terminal replication error (Section 6.4) of one hedging
    strategy on a swap starting today (T0=0), of the given maturity.

    strategy: "unhedged", "sofr_only", or "full" (SOFR + €STR futures).
    rebalance_every: number of daily steps between hedge-ratio updates
    (1=daily, 5=weekly, 21=monthly); the state grid itself is always daily.
    Returns the terminal (post-last-coupon) replication error per path, in
    basis points of unit notional.
    """
    if strategy not in ("unhedged", "sofr_only", "full"):
        raise ValueError(f"unknown strategy {strategy!r}")

    dates = payment_schedule(0.0, maturity, delta)
    n_coupons = len(dates) - 1
    Us = list(dates[:-1])
    Ts = list(dates[1:])
    delta_js = [Ts[j] - Us[j] for j in range(n_coupons)]

    n_steps = int(round(maturity / dt))
    rng = np.random.default_rng(seed)
    corr = np.array([[1.0, mp.rho_12, mp.rho_13],
                      [mp.rho_12, 1.0, mp.rho_23],
                      [mp.rho_13, mp.rho_23, 1.0]])
    chol = np.linalg.cholesky(corr)
    sqrt_dt = math.sqrt(dt)

    r_d = np.full(n_paths, mp.r0_d)
    r_f = np.full(n_paths, mp.r0_f)
    logQ = np.full(n_paths, math.log(mp.q0))

    r_beta_spread = beta * net_carry_bp * 1e-4  # constant effective-rate spread over r_t^d

    price0 = multi_period_price(0.0, maturity, gamma, kappa, mp, beta, net_carry_bp)

    realized_d = [np.zeros(n_paths) for _ in range(n_coupons)]
    realized_f = [np.zeros(n_paths) for _ in range(n_coupons)]
    paid = [False] * n_coupons
    phi_d = [np.zeros(n_paths) for _ in range(n_coupons)]
    phi_f = [np.zeros(n_paths) for _ in range(n_coupons)]
    F_d_curr = [np.zeros(n_paths) for _ in range(n_coupons)]
    F_f_curr = [np.zeros(n_paths) for _ in range(n_coupons)]

    V = np.full(n_paths, price0)

    def _refresh(t: float, compute_ratios: bool):
        for j in range(n_coupons):
            if paid[j]:
                continue
            Uj, Tj, delta_j = Us[j], Ts[j], delta_js[j]
            I1, I2, I3, one_plus_dF_d, one_plus_dF_f, gamma_qf = _futures_and_components(
                t, Uj, Tj, r_d, r_f, mp, beta, net_carry_bp, realized_d[j], realized_f[j])
            # The futures contracts traded (and hence marked to market in the
            # wealth equation 5.15) are F_t^d, F_t^f themselves, not the
            # auxiliary quantities 1+delta*F returned by the Vasicek/Q-hat
            # conditional-expectation formulas; convert before storing.
            F_d_curr[j] = (one_plus_dF_d - 1.0) / delta_j
            F_f_curr[j] = (one_plus_dF_f - 1.0) / delta_j
            if not compute_ratios or strategy == "unhedged":
                continue
            if t <= Uj + 1e-12:
                sigma_U = _sigma_loading(t, Uj, mp.sigma_d, mp.b_d)
                sigma_T = _sigma_loading(t, Tj, mp.sigma_d, mp.b_d)
                nu = (1.0 / delta_j) * one_plus_dF_d * (sigma_U - sigma_T)
                numerator = sigma_U * I1 - sigma_T * (gamma * I2 + (1.0 - gamma + delta_j * kappa) * I3)
                phi_d[j] = numerator / nu
            else:
                phi_d[j] = delta_j * (gamma * I2 + (1.0 - gamma + delta_j * kappa) * I3) / one_plus_dF_d
            if strategy == "full":
                phi_f[j] = -gamma * delta_j / np.exp(logQ) * I3 * gamma_qf
            else:
                phi_f[j] = np.zeros(n_paths)

    t = 0.0
    _refresh(t, compute_ratios=True)

    for step in range(n_steps):
        t_next = t + dt
        is_rebalance = (step % rebalance_every == 0)

        for j in range(n_coupons):
            if paid[j] or Us[j] >= t_next - 1e-9:
                continue
            realized_d[j] = realized_d[j] + r_d * dt
            realized_f[j] = realized_f[j] + r_f * dt

        z = rng.standard_normal((n_paths, 3)) @ chol.T * sqrt_dt
        r_d_next = r_d + (mp.a_d - mp.b_d * r_d) * dt + mp.sigma_d * z[:, 0]
        r_f_next = r_f + (mp.c_hat - mp.b_f * r_f) * dt + mp.sigma_f * z[:, 1]
        logQ_next = logQ + (r_d - r_f - 0.5 * mp.sigma_bar ** 2) * dt + mp.sigma_bar * z[:, 2]

        r_beta_i = r_d + r_beta_spread
        cash_accrual = r_beta_i * V * dt

        r_d, r_f, logQ = r_d_next, r_f_next, logQ_next
        Q_next = np.exp(logQ)

        futures_pnl = np.zeros(n_paths)
        F_d_old, F_f_old = list(F_d_curr), list(F_f_curr)
        _refresh(t_next, compute_ratios=False)
        for j in range(n_coupons):
            if paid[j]:
                continue
            futures_pnl = futures_pnl + phi_d[j] * (F_d_curr[j] - F_d_old[j])
            futures_pnl = futures_pnl + phi_f[j] * Q_next * (F_f_curr[j] - F_f_old[j])

        V = V + cash_accrual + futures_pnl

        for j in range(n_coupons):
            if paid[j] or abs(Ts[j] - t_next) > 1e-9:
                continue
            coupon = np.exp(realized_d[j]) - gamma * np.exp(realized_f[j]) - (1.0 - gamma + delta_js[j] * kappa)
            V = V - coupon
            paid[j] = True
            phi_d[j] = np.zeros(n_paths)
            phi_f[j] = np.zeros(n_paths)

        if is_rebalance and not all(paid):
            _refresh(t_next, compute_ratios=True)

        t = t_next

    rmse_bp = float(np.sqrt(np.mean(V ** 2)) * 1e4)
    variance = float(np.var(V))
    return HedgeResult(rmse_bp=rmse_bp, variance=variance, terminal_wealth=V)
