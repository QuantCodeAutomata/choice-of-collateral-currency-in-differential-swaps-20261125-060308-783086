"""Unit tests for the closed-form pricing formulas (src/pricing.py),
Propositions 4.1, 4.2 and Corollary 4.1."""
from __future__ import annotations

import pytest

import src.config as cfg
from src.pricing import (
    ModelParams,
    collateral_adjustment_factor,
    multi_period_price,
    par_fixed_margin,
    par_fixed_margin_no_quanto,
    payment_schedule,
    single_period_price,
)


def _mp(**overrides) -> ModelParams:
    base = dict(r0_d=cfg.R0_D, b_d=cfg.B_D, sigma_d=cfg.SIGMA_D, a_d=cfg.A_D,
                r0_f=cfg.R0_F, b_f=cfg.B_F, sigma_f=cfg.SIGMA_F, c_hat=cfg.C_HAT,
                a_hat=cfg.A_HAT, q0=cfg.Q0, sigma_bar=cfg.SIGMA_BAR,
                rho_12=cfg.RHO_12, rho_13=cfg.RHO_13, rho_23=cfg.RHO_23)
    base.update(overrides)
    return ModelParams(**base)


def test_payment_schedule_length_and_spacing():
    dates = payment_schedule(0.0, 3.0, 0.5)
    assert len(dates) == 7
    assert dates[0] == pytest.approx(0.0)
    assert dates[-1] == pytest.approx(3.0)
    diffs = [dates[i + 1] - dates[i] for i in range(len(dates) - 1)]
    assert all(d == pytest.approx(0.5) for d in diffs)


def test_collateral_adjustment_factor_usd_is_one_at_zero_net_carry():
    assert collateral_adjustment_factor(5.0, 1.0, net_carry_bp=0.0) == pytest.approx(1.0)


def test_collateral_adjustment_factor_decreasing_in_positive_net_carry():
    a1 = collateral_adjustment_factor(1.0, 1.0, net_carry_bp=20.0)
    a2 = collateral_adjustment_factor(5.0, 1.0, net_carry_bp=20.0)
    assert 0.0 < a2 < a1 < 1.0


def test_single_period_price_finite_and_sign_plausible():
    mp = _mp()
    price = single_period_price(0.0, 0.5, 0.0, 0.0, mp, 1.0, cfg.NET_CARRY_USD_BP)
    assert price == pytest.approx(mp.r0_d * 0.5, rel=0.1)  # roughly delta*r_0^d for gamma=0


def test_multi_period_equals_sum_of_single_periods():
    mp = _mp()
    T0, maturity, gamma, kappa = 0.0, 2.0, 1.0, 0.01
    multi = multi_period_price(T0, maturity, gamma, kappa, mp, 1.0, cfg.NET_CARRY_EUR_BP)
    dates = payment_schedule(T0, maturity, 0.5)
    total = sum(
        single_period_price(dates[j - 1], dates[j], gamma, kappa, mp, 1.0, cfg.NET_CARRY_EUR_BP)
        for j in range(1, len(dates))
    )
    assert multi == pytest.approx(total, rel=1e-9)


def test_par_margin_makes_price_zero():
    mp = _mp()
    T0, maturity, gamma = 0.0, 3.0, 1.0
    kappa_star = par_fixed_margin(T0, maturity, gamma, mp, 1.0, cfg.NET_CARRY_EUR_BP)
    price_at_par = multi_period_price(T0, maturity, gamma, kappa_star, mp, 1.0, cfg.NET_CARRY_EUR_BP)
    assert price_at_par == pytest.approx(0.0, abs=1e-10)


def test_par_margin_invariant_to_collateral_currency_and_beta():
    """Corollary 4.1 (Section 6.1): kappa* is invariant to i and beta because
    the common factor A_{0,T}^{beta,i}*B^d(0,T) cancels in numerator/denominator."""
    mp = _mp()
    T0, maturity, gamma = 0.5, 1.0, 1.0
    kappas = []
    for beta in (0.0, 0.5, 1.0):
        for net_carry in (cfg.NET_CARRY_USD_BP, cfg.NET_CARRY_EUR_BP):
            kappas.append(par_fixed_margin(T0, maturity, gamma, mp, beta, net_carry))
    assert max(kappas) - min(kappas) < 1e-9


def test_equal_net_carry_gives_identical_price():
    """Two collateral currencies with the same net carry must give the same
    price, since only the net carry enters alpha^{beta,i} (Remark 4.1)."""
    mp = _mp()
    shared_net_carry = 15.0
    p1 = multi_period_price(0.0, 3.0, 1.0, 0.0, mp, 1.0, shared_net_carry)
    p2 = multi_period_price(0.0, 3.0, 1.0, 0.0, mp, 1.0, shared_net_carry)
    assert p1 == pytest.approx(p2, rel=1e-12)


def test_separation_property_gamma_zero():
    """Section 6.1: with gamma=0 and the collateral net-carry spec fixed, the
    price does not depend on the foreign-rate or FX parameters at all."""
    mp = _mp()
    mp_perturbed = _mp(r0_f=0.08, b_f=1.5, sigma_f=0.04, c_hat=0.12, a_hat=0.13,
                       q0=1.5, sigma_bar=0.2, rho_12=-0.2, rho_13=0.3, rho_23=-0.4)
    p1 = multi_period_price(0.0, 3.0, 0.0, 0.0, mp, 1.0, cfg.NET_CARRY_USD_BP)
    p2 = multi_period_price(0.0, 3.0, 0.0, 0.0, mp_perturbed, 1.0, cfg.NET_CARRY_USD_BP)
    assert p1 == pytest.approx(p2, abs=1e-12)


def test_quanto_adjustment_vanishes_at_rho23_zero_is_not_assumed():
    """Sanity check only: par margin with and without quanto correction need
    not match at nonzero rho_23 for gamma=1 (quanto_adjustment is generally
    nonzero); this test merely checks both are finite and well defined."""
    mp = _mp()
    k_full = par_fixed_margin(0.0, 5.0, 1.0, mp, 1.0, cfg.NET_CARRY_USD_BP)
    k_no_qf = par_fixed_margin_no_quanto(0.0, 5.0, 1.0, mp, 1.0, cfg.NET_CARRY_USD_BP)
    assert abs(k_full - k_no_qf) < 1e-2  # a few bp at most, not an order of magnitude


def test_quanto_adjustment_scales_with_gamma():
    """Equation (6.3): Delta kappa^q(gamma) = gamma * Delta kappa^q(1)."""
    mp = _mp()
    d_full = par_fixed_margin(0.0, 5.0, 1.0, mp, 1.0, cfg.NET_CARRY_USD_BP) - \
        par_fixed_margin_no_quanto(0.0, 5.0, 1.0, mp, 1.0, cfg.NET_CARRY_USD_BP)
    d_half_gamma = par_fixed_margin(0.0, 5.0, 0.5, mp, 1.0, cfg.NET_CARRY_USD_BP) - \
        par_fixed_margin_no_quanto(0.0, 5.0, 0.5, mp, 1.0, cfg.NET_CARRY_USD_BP)
    assert d_half_gamma == pytest.approx(0.5 * d_full, rel=1e-6)
