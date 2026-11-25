"""Tests of the discrete futures-hedging simulation (src/hedging.py, Section 5
and the Section 6.4 experiment)."""
from __future__ import annotations

import numpy as np
import pytest

from src import config as cfg
from src.hedging import simulate_hedge
from src.pricing import ModelParams, par_fixed_margin


def _mp(**overrides) -> ModelParams:
    defaults = dict(
        r0_d=cfg.R0_D, b_d=cfg.B_D, sigma_d=cfg.SIGMA_D, a_d=cfg.A_D,
        r0_f=cfg.R0_F, b_f=cfg.B_F, sigma_f=cfg.SIGMA_F, c_hat=cfg.C_HAT,
        a_hat=cfg.A_HAT, q0=cfg.Q0, sigma_bar=cfg.SIGMA_BAR,
        rho_12=cfg.RHO_12, rho_13=cfg.RHO_13, rho_23=cfg.RHO_23,
    )
    defaults.update(overrides)
    return ModelParams(**defaults)


MATURITY = 1.0  # short maturity so the discrete tests run quickly
N_PATHS = 500
SEED = 7


def _kappa_usd(mp, maturity, gamma):
    return par_fixed_margin(0.0, maturity, gamma, mp, cfg.BETA_FULL, cfg.NET_CARRY_USD_BP)


def test_invalid_strategy_raises():
    mp = _mp()
    kappa = _kappa_usd(mp, MATURITY, 1.0)
    with pytest.raises(ValueError):
        simulate_hedge(mp, MATURITY, 1.0, kappa, cfg.BETA_FULL, cfg.NET_CARRY_USD_BP,
                        N_PATHS, SEED, strategy="bogus")


def test_gamma_zero_sofr_only_equals_full_hedge():
    """When gamma=0 the €STR-futures position is identically zero (Prop. 5.2,
    eq. 5.14), so the SOFR-only and full hedges must coincide path by path."""
    mp = _mp()
    kappa = _kappa_usd(mp, MATURITY, 0.0)
    res_sofr = simulate_hedge(mp, MATURITY, 0.0, kappa, cfg.BETA_FULL, cfg.NET_CARRY_USD_BP,
                               N_PATHS, SEED, strategy="sofr_only")
    res_full = simulate_hedge(mp, MATURITY, 0.0, kappa, cfg.BETA_FULL, cfg.NET_CARRY_USD_BP,
                               N_PATHS, SEED, strategy="full")
    np.testing.assert_allclose(res_sofr.terminal_wealth, res_full.terminal_wealth, atol=1e-10)


def test_full_hedge_reduces_variance_versus_unhedged_and_sofr_only():
    """For gamma=1 the full SOFR+€STR hedge must materially reduce the
    replication-error variance relative to both the unhedged position and the
    SOFR-only (partial) hedge (Section 6.4, Table 4)."""
    mp = _mp()
    kappa = _kappa_usd(mp, MATURITY, 1.0)
    unhedged = simulate_hedge(mp, MATURITY, 1.0, kappa, cfg.BETA_FULL, cfg.NET_CARRY_USD_BP,
                               N_PATHS, SEED, strategy="unhedged")
    sofr_only = simulate_hedge(mp, MATURITY, 1.0, kappa, cfg.BETA_FULL, cfg.NET_CARRY_USD_BP,
                                N_PATHS, SEED, strategy="sofr_only")
    full = simulate_hedge(mp, MATURITY, 1.0, kappa, cfg.BETA_FULL, cfg.NET_CARRY_USD_BP,
                           N_PATHS, SEED, strategy="full")
    assert full.variance < sofr_only.variance < unhedged.variance
    assert full.rmse_bp < 10.0  # full hedge should be within a few bp even at low path count


def test_rmse_and_variance_are_non_negative():
    mp = _mp()
    kappa = _kappa_usd(mp, MATURITY, 1.0)
    res = simulate_hedge(mp, MATURITY, 1.0, kappa, cfg.BETA_FULL, cfg.NET_CARRY_USD_BP,
                          N_PATHS, SEED, strategy="full")
    assert res.rmse_bp >= 0.0
    assert res.variance >= 0.0
    assert res.terminal_wealth.shape == (N_PATHS,)


def test_monthly_rebalancing_increases_full_hedge_rmse_versus_daily():
    """Coarser rebalancing should not improve the hedge (Section 6.4, Figure 3:
    RMSE increases from daily to weekly to monthly)."""
    mp = _mp()
    kappa = _kappa_usd(mp, MATURITY, 1.0)
    daily = simulate_hedge(mp, MATURITY, 1.0, kappa, cfg.BETA_FULL, cfg.NET_CARRY_USD_BP,
                            N_PATHS, SEED, strategy="full", rebalance_every=1)
    monthly = simulate_hedge(mp, MATURITY, 1.0, kappa, cfg.BETA_FULL, cfg.NET_CARRY_USD_BP,
                              N_PATHS, SEED, strategy="full", rebalance_every=21)
    assert monthly.rmse_bp >= daily.rmse_bp
