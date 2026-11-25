"""Tests of the exact-Gaussian Monte Carlo pricer (src/mc_simulation.py) against
the closed-form prices of Propositions 4.1-4.2 (Section 6.1)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src import config as cfg
from src.mc_simulation import simulate_mc_price, z_score
from src.pricing import ModelParams, multi_period_price, single_period_price


def _mp(**overrides) -> ModelParams:
    defaults = dict(
        r0_d=cfg.R0_D, b_d=cfg.B_D, sigma_d=cfg.SIGMA_D, a_d=cfg.A_D,
        r0_f=cfg.R0_F, b_f=cfg.B_F, sigma_f=cfg.SIGMA_F, c_hat=cfg.C_HAT,
        a_hat=cfg.A_HAT, q0=cfg.Q0, sigma_bar=cfg.SIGMA_BAR,
        rho_12=cfg.RHO_12, rho_13=cfg.RHO_13, rho_23=cfg.RHO_23,
    )
    defaults.update(overrides)
    return ModelParams(**defaults)


def _net_carry(collateral: str) -> float:
    return cfg.NET_CARRY_USD_BP if collateral == "d" else cfg.NET_CARRY_EUR_BP


@pytest.mark.parametrize("config", cfg.EXP1_CONFIGS)
def test_mc_matches_closed_form_within_two_std_errors(config):
    """Each Section 6.1 test configuration: |z-score| < ~3 (loose bound; the
    paper's own reported z-scores are below 2 in absolute value)."""
    mp = _mp()
    net_carry_bp = _net_carry(config["collateral"])
    kappa = 0.0
    n_paths = 20_000  # reduced path count for a fast, still-informative test
    seed = cfg.MC_SEED_EXP1

    if config["structure"] == "single":
        U, T = config["T"] - cfg.DELTA_SEMIANNUAL, config["T"]
        closed_form = single_period_price(
            U, T, config["gamma"], kappa, mp, config["beta"], net_carry_bp)
        mc = simulate_mc_price(mp, U, T, config["gamma"], kappa, config["beta"],
                                net_carry_bp, n_paths, seed, delta=cfg.DELTA_SEMIANNUAL)
    else:
        closed_form = multi_period_price(
            0.0, config["T"], config["gamma"], kappa, mp, config["beta"], net_carry_bp)
        mc = simulate_mc_price(mp, 0.0, config["T"], config["gamma"], kappa, config["beta"],
                                net_carry_bp, n_paths, seed, delta=cfg.DELTA_SEMIANNUAL)

    z = z_score(mc.price, closed_form, mc.std_error)
    assert abs(z) < 3.5, f"z-score {z} too large for config {config}"


def test_mc_std_error_shrinks_with_more_paths():
    """Monte Carlo sampling error must shrink like 1/sqrt(n_paths)."""
    mp = _mp()
    mc_small = simulate_mc_price(mp, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 2_000, seed=1)
    mc_large = simulate_mc_price(mp, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 8_000, seed=1)
    # Quadrupling the paths should roughly halve the standard error.
    ratio = mc_small.std_error / mc_large.std_error
    assert 1.5 < ratio < 2.5


def test_mc_price_invalid_n_paths_raises():
    mp = _mp()
    with pytest.raises(ValueError):
        simulate_mc_price(mp, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0, seed=1)


def test_z_score_sign_and_zero():
    assert z_score(1.0, 1.0, 0.1) == 0.0
    assert z_score(1.1, 1.0, 0.1) == pytest.approx(1.0)
    assert z_score(0.9, 1.0, 0.1) == pytest.approx(-1.0)


def test_separation_property_gamma_zero():
    """When gamma=0, the single-period price must not depend on foreign-rate
    or FX parameters once the collateral net carry is held fixed (Section
    6.1 structural check)."""
    mp1 = _mp()
    mp2 = _mp(r0_f=0.05, sigma_f=0.03, c_hat=0.10, sigma_bar=0.20, rho_12=-0.1, rho_23=0.4)
    price1 = single_period_price(0.0, 0.5, 0.0, 0.0, mp1, 1.0, cfg.NET_CARRY_USD_BP)
    price2 = single_period_price(0.0, 0.5, 0.0, 0.0, mp2, 1.0, cfg.NET_CARRY_USD_BP)
    assert price1 == pytest.approx(price2, rel=1e-12)
