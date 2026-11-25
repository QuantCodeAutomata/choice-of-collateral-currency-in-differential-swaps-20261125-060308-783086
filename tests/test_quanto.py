"""Unit tests for the deterministic quanto correction (src/quanto.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.quanto import quanto_correction, quanto_correction_closed_form

SIGMA_D, B_D = 0.012, 3.0
SIGMA_F, B_F = 0.010, 3.0
RHO_12, SIGMA_BAR = 0.25, 0.075


def test_quanto_correction_vanishes_at_U_equal_T_limit():
    """As U -> T, the single-period interval collapses and Gamma -> 1."""
    gamma_qf = quanto_correction(0.0, 2.999999, 3.0, SIGMA_D, B_D, SIGMA_F, B_F,
                                  RHO_12, rho_23=0.10, sigma_bar=SIGMA_BAR)
    assert gamma_qf == pytest.approx(1.0, abs=1e-3)


def test_quanto_correction_requires_T_gt_U():
    with pytest.raises(ValueError):
        quanto_correction(0.0, 3.0, 2.0, SIGMA_D, B_D, SIGMA_F, B_F, RHO_12, 0.10, SIGMA_BAR)


@pytest.mark.parametrize("rho_23", [-0.4, -0.1, 0.1, 0.4])
def test_quanto_correction_positive(rho_23):
    gamma_qf = quanto_correction(0.0, 2.5, 3.0, SIGMA_D, B_D, SIGMA_F, B_F, RHO_12, rho_23, SIGMA_BAR)
    assert gamma_qf > 0.0


def test_closed_form_matches_quadrature():
    """src.quanto.quanto_correction_closed_form must reproduce the numerical
    quadrature of quanto_correction (same equation 4.9) to high precision."""
    for rho_23 in (-0.3, 0.0, 0.1, 0.4):
        quad = quanto_correction(0.0, 2.5, 3.0, SIGMA_D, B_D, SIGMA_F, B_F, RHO_12, rho_23, SIGMA_BAR)
        closed = float(quanto_correction_closed_form(
            0.0, 2.5, 3.0, SIGMA_D, B_D, SIGMA_F, B_F, RHO_12, rho_23, SIGMA_BAR))
        assert closed == pytest.approx(quad, rel=1e-6)


def test_closed_form_vectorised_over_t():
    t = np.array([0.0, 0.5, 1.0, 2.0])
    out = quanto_correction_closed_form(t, 2.5, 3.0, SIGMA_D, B_D, SIGMA_F, B_F, RHO_12, 0.10, SIGMA_BAR)
    assert out.shape == t.shape
    assert np.all(out > 0.0)


def test_quanto_correction_zero_correlations_gives_identity():
    """rho_12 = rho_23 = 0 removes all covariance terms, so Gamma = 1."""
    gamma_qf = quanto_correction(0.0, 2.5, 3.0, SIGMA_D, B_D, SIGMA_F, B_F,
                                  rho_12=0.0, rho_23=0.0, sigma_bar=SIGMA_BAR)
    assert gamma_qf == pytest.approx(1.0, rel=1e-9)
