"""Closed-form single- and multi-period differential swap pricing
(Propositions 4.1, 4.2 and Corollary 4.1 of Ding, Liu and Rutkowski,
"Choice of Collateral Currency in Differential Swaps", Sections 4.1-4.2).

Payoff (eq. 4.4): DS_T^{gamma,kappa}(U,T) = exp(int_U^T r^d dv)
    - gamma*exp(int_U^T r^f dv) - (1 - gamma + delta*kappa), with delta=T-U, P=1.

Price (eq. 4.5): DS_t = I^(1) - gamma*I^(2) - (1-gamma+delta*kappa)*I^(3), with
I^(1), I^(2), I^(3) given by (4.6)-(4.9). All quantities below are evaluated at
t=0, the only time used by the numerical studies of Section 6.
"""
from __future__ import annotations

from dataclasses import dataclass

import math

from src.quanto import quanto_correction
from src.vasicek import conditional_exp_integral, discount_factor


@dataclass(frozen=True)
class ModelParams:
    """Joint Vasicek-Garman-Kohlhagen parameters (Section 3.1, Assumption 3.1)."""

    r0_d: float
    b_d: float
    sigma_d: float
    a_d: float          # domestic Vasicek drift constant a (under Q)
    r0_f: float
    b_f: float
    sigma_f: float
    c_hat: float        # foreign Vasicek drift constant c_hat (under Q)
    a_hat: float         # foreign drift constant a_hat (under Q-hat)
    q0: float
    sigma_bar: float
    rho_12: float
    rho_13: float
    rho_23: float


def collateral_adjustment_factor(T_j: float, beta: float, net_carry_bp: float) -> float:
    """A_{0,T_j}^{beta,i} = exp(-int_0^{T_j} alpha_u^{beta,i} du) (Remark 4.1).

    For USD collateral alpha^{beta,d} = (1-beta)*alpha^d + beta*alpha^{c,d};
    with alpha^d = 0 (src.config.ALPHA_D) this reduces to beta*net_carry^{USD}.
    For EUR collateral alpha^{beta,f} = alpha^d + beta*(alpha^{c,f}-alpha^f).
    In both cases the constant deterministic spread integrates to
    net_carry * beta * T_j (net_carry_bp already passed in decimal form).
    """
    net_carry = net_carry_bp * 1e-4
    return float(math.exp(-beta * net_carry * T_j))


def futures_rate_at_t(t: float, U: float, T: float, r_t_f: float, mp: ModelParams,
                       realized: float = 0.0) -> float:
    """1 + delta*F_t^f(U,T) = E_{Qhat}[exp(int_U^T r^f dv) | F_t] (Definition
    3.1), computed under the foreign martingale measure Q-hat with drift
    a_hat, for t <= T (Section 3.2, eq. before 3.2)."""
    return conditional_exp_integral(t, U, T, r_t_f, mp.a_hat, mp.b_f, mp.sigma_f, realized)


def swap_components_t0(U: float, T: float, mp: ModelParams, beta: float, net_carry_bp: float):
    """Pricing components I^(1), I^(2), I^(3) at t=0 for the single accrual
    period [U,T] (Proposition 4.1, equations 4.6-4.9, specialised to t=0<=U)."""
    A = collateral_adjustment_factor(T, beta, net_carry_bp)
    B_d_0U = discount_factor(0.0, U, mp.r0_d, mp.a_d, mp.b_d, mp.sigma_d)
    B_d_0T = discount_factor(0.0, T, mp.r0_d, mp.a_d, mp.b_d, mp.sigma_d)
    one_plus_delta_F = futures_rate_at_t(0.0, U, T, mp.r0_f, mp)
    gamma_qf = quanto_correction(0.0, U, T, mp.sigma_d, mp.b_d, mp.sigma_f, mp.b_f,
                                  mp.rho_12, mp.rho_23, mp.sigma_bar)
    I1 = A * B_d_0U
    I2 = A * B_d_0T * one_plus_delta_F * gamma_qf
    I3 = A * B_d_0T
    return I1, I2, I3, one_plus_delta_F, gamma_qf


def single_period_price(U: float, T: float, gamma: float, kappa: float, mp: ModelParams,
                         beta: float, net_carry_bp: float) -> float:
    """Proposition 4.1: DS_0^{beta,gamma,kappa}(U,T;i), per unit notional (P=1)."""
    delta = T - U
    I1, I2, I3, _, _ = swap_components_t0(U, T, mp, beta, net_carry_bp)
    return I1 - gamma * I2 - (1.0 - gamma + delta * kappa) * I3


def payment_schedule(T0: float, maturity: float, delta: float = 0.5):
    """Payment dates T_1 < ... < T_n with T_0 as the first accrual start."""
    n = int(round((maturity - T0) / delta))
    return [T0 + j * delta for j in range(n + 1)]


def multi_period_components_t0(T0: float, maturity: float, mp: ModelParams, beta: float,
                                net_carry_bp: float, delta: float = 0.5):
    """List of (delta_j, I1_j, I2_j, I3_j) for each coupon j=1..n (Proposition 4.2)."""
    dates = payment_schedule(T0, maturity, delta)
    out = []
    for j in range(1, len(dates)):
        Uj, Tj = dates[j - 1], dates[j]
        delta_j = Tj - Uj
        I1, I2, I3, _, _ = swap_components_t0(Uj, Tj, mp, beta, net_carry_bp)
        out.append((delta_j, I1, I2, I3))
    return out


def multi_period_price(T0: float, maturity: float, gamma: float, kappa: float, mp: ModelParams,
                        beta: float, net_carry_bp: float, delta: float = 0.5) -> float:
    """Proposition 4.2: DS_0^{beta,gamma,kappa}(T_0,n;i), per unit notional."""
    comps = multi_period_components_t0(T0, maturity, mp, beta, net_carry_bp, delta)
    price = 0.0
    for delta_j, I1, I2, I3 in comps:
        price += I1 - gamma * I2 - (1.0 - gamma + delta_j * kappa) * I3
    return price


def par_fixed_margin(T0: float, maturity: float, gamma: float, mp: ModelParams, beta: float,
                      net_carry_bp: float, delta: float = 0.5) -> float:
    """Corollary 4.1: par fixed margin kappa_0^*(T_0,n;beta,gamma,i), eq. (4.15)."""
    comps = multi_period_components_t0(T0, maturity, mp, beta, net_carry_bp, delta)
    numerator = sum(I1 - gamma * I2 - (1.0 - gamma) * I3 for _, I1, I2, I3 in comps)
    denominator = sum(delta_j * I3 for delta_j, _, _, I3 in comps)
    return numerator / denominator


def par_fixed_margin_no_quanto(T0: float, maturity: float, gamma: float, mp: ModelParams, beta: float,
                                net_carry_bp: float, delta: float = 0.5) -> float:
    """Counterfactual par margin with Gamma^{qf} replaced by 1 (Section 6.3, eq. 6.2)."""
    dates = payment_schedule(T0, maturity, delta)
    numerator = 0.0
    denominator = 0.0
    for j in range(1, len(dates)):
        Uj, Tj = dates[j - 1], dates[j]
        delta_j = Tj - Uj
        A = collateral_adjustment_factor(Tj, beta, net_carry_bp)
        B_d_0U = discount_factor(0.0, Uj, mp.r0_d, mp.a_d, mp.b_d, mp.sigma_d)
        B_d_0T = discount_factor(0.0, Tj, mp.r0_d, mp.a_d, mp.b_d, mp.sigma_d)
        one_plus_delta_F = futures_rate_at_t(0.0, Uj, Tj, mp.r0_f, mp)
        I1 = A * B_d_0U
        I2_no_qf = A * B_d_0T * one_plus_delta_F  # Gamma^{qf} replaced by 1
        I3 = A * B_d_0T
        numerator += I1 - gamma * I2_no_qf - (1.0 - gamma) * I3
        denominator += delta_j * I3
    return numerator / denominator
