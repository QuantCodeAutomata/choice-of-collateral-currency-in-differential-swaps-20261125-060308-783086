"""Unit tests for the Vasicek building blocks (src/vasicek.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.vasicek import (
    conditional_exp_integral,
    discount_factor,
    integral_covariance,
    integrated_rate_moments,
    n_func,
)


def test_n_func_limits():
    """n(u,S) -> 0 as S -> u, and n(u,S) -> 1/b as S-u -> infinity."""
    b = 3.0
    assert n_func(1.0, 1.0, b) == pytest.approx(0.0)
    assert n_func(0.0, 1000.0, b) == pytest.approx(1.0 / b, rel=1e-6)


def test_discount_factor_decreasing_in_maturity():
    """A positive short rate must make B(0,T) decreasing in T."""
    r0, theta, b, sigma = 0.03, 0.09, 3.0, 0.012
    b1 = discount_factor(0.0, 1.0, r0, theta, b, sigma)
    b2 = discount_factor(0.0, 5.0, r0, theta, b, sigma)
    assert 0.0 < b2 < b1 <= 1.0


def test_discount_factor_at_t_equals_T_is_one():
    assert discount_factor(2.0, 2.0, 0.03, 0.09, 3.0, 0.012) == pytest.approx(1.0)


def test_discount_factor_vectorised_over_paths():
    r0 = np.array([0.01, 0.03, 0.05])
    out = discount_factor(0.0, 2.0, r0, 0.09, 3.0, 0.012)
    assert out.shape == r0.shape
    assert np.all(np.diff(out) < 0)  # higher r0 -> lower discount factor


def test_integrated_rate_moments_positive_variance():
    mean, var = integrated_rate_moments(0.0, 3.0, 0.03, 0.09, 3.0, 0.012)
    assert var > 0.0
    assert mean > 0.0


def test_conditional_exp_integral_matches_discount_factor_identity():
    """E[exp(int_U^T r dv)|F_t] * B(t,T) should recover B(t,U) when seen as a
    product of the two legs' exponential martingale components, i.e. taking
    theta=0, sigma=0 (deterministic rate r_t=r) reduces to exp(r*(T-U))."""
    r, b, sigma = 0.03, 3.0, 0.0
    theta = b * r  # keeps the deterministic rate constant at r
    g = conditional_exp_integral(0.0, 1.0, 2.0, r, theta, b, sigma)
    assert g == pytest.approx(np.exp(r * 1.0), rel=1e-6)


def test_conditional_exp_integral_requires_T_gt_U():
    with pytest.raises(ValueError):
        conditional_exp_integral(0.0, 2.0, 1.0, 0.03, 0.09, 3.0, 0.012)


def test_conditional_exp_integral_straddling_t_consistent_with_realized():
    """For U <= t <= T, supplying realized = int_U^t r dv (here with sigma=0,
    deterministic) must reproduce the same value as evaluating from t=U."""
    r, b, sigma = 0.03, 3.0, 0.0
    theta = b * r
    direct = conditional_exp_integral(0.0, 1.0, 2.0, r, theta, b, sigma)
    realized = r * 0.4  # int_{1.0}^{1.4} r dv with deterministic r
    straddling = conditional_exp_integral(1.4, 1.0, 2.0, r, theta, b, sigma, realized=realized)
    assert straddling == pytest.approx(direct, rel=1e-6)


def test_integral_covariance_symmetric_and_zero_when_disjoint():
    cov_ab = integral_covariance(0.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.012, 3.0, 0.01, 0.25)
    cov_ba = integral_covariance(0.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.01, 3.0, 0.012, 0.25)
    assert cov_ab == pytest.approx(cov_ba, rel=1e-6)
    disjoint = integral_covariance(0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 0.012, 3.0, 0.01, 0.25)
    assert disjoint == pytest.approx(0.0, abs=1e-12)


def test_integral_covariance_zero_at_zero_correlation():
    cov = integral_covariance(0.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.012, 3.0, 0.01, 0.0)
    assert cov == pytest.approx(0.0, abs=1e-12)
