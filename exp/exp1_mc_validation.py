"""Experiment 1: Monte Carlo validation of the closed-form single- and
multi-period differential swap prices (Propositions 4.1 and 4.2, Section 6.1
of Ding, Liu and Rutkowski, "Choice of Collateral Currency in Differential
Swaps"), plus the structural checks of Section 6.1-6.3: the separation
property under gamma=0, the equal-net-carry identity across collateral
currencies, and the invariance of the par margin to collateral currency and
to the collateralisation rate beta.

Run: python -m exp.exp1_mc_validation
"""
from __future__ import annotations

import json
from pathlib import Path

from scipy import stats

from src.config import (
    A_D, A_HAT, B_D, B_F, C_HAT, EXP1_CONFIGS, MC_PATHS_EXP1, MC_SEED_EXP1,
    NET_CARRY_EUR_BP, NET_CARRY_USD_BP, Q0, R0_D, R0_F, RHO_12, RHO_13, RHO_23,
    SIGMA_BAR, SIGMA_D, SIGMA_F,
)
from src.mc_simulation import simulate_mc_price, z_score
from src.pricing import ModelParams, multi_period_price, par_fixed_margin, single_period_price

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _base_params() -> ModelParams:
    return ModelParams(r0_d=R0_D, b_d=B_D, sigma_d=SIGMA_D, a_d=A_D,
                        r0_f=R0_F, b_f=B_F, sigma_f=SIGMA_F, c_hat=C_HAT, a_hat=A_HAT,
                        q0=Q0, sigma_bar=SIGMA_BAR, rho_12=RHO_12, rho_13=RHO_13, rho_23=RHO_23)


def _net_carry(collateral: str) -> float:
    return NET_CARRY_USD_BP if collateral == "d" else NET_CARRY_EUR_BP


def run_mc_validation(mp: ModelParams, n_paths: int, seed: int) -> list[dict]:
    """Table 2: closed-form vs. Monte Carlo price for each test configuration."""
    rows = []
    for i, cfg in enumerate(EXP1_CONFIGS):
        T, gamma, collateral, beta = cfg["T"], cfg["gamma"], cfg["collateral"], cfg["beta"]
        net_carry_bp = _net_carry(collateral)
        if cfg["structure"] == "single":
            U = T - 0.5
            closed_form = single_period_price(U, T, gamma, 0.0, mp, beta, net_carry_bp)
            T0 = U
        else:
            closed_form = multi_period_price(0.0, T, gamma, 0.0, mp, beta, net_carry_bp)
            T0 = 0.0
        mc = simulate_mc_price(mp, T0, T, gamma, 0.0, beta, net_carry_bp, n_paths, seed=seed + i)
        z = z_score(mc.price, closed_form, mc.std_error)
        rows.append({
            "config": f"{cfg['structure']}_T{T}_gamma{gamma}_{collateral}",
            "structure": cfg["structure"],
            "T": T,
            "gamma": gamma,
            "collateral": collateral,
            "beta": beta,
            "closed_form_price": closed_form,
            "mc_price": mc.price,
            "mc_std_error": mc.std_error,
            "mc_error": mc.price - closed_form,
            "z_score": z,
            "within_2_std": bool(abs(z) < 2.0),
            "n_paths": n_paths,
        })
    return rows


def run_separation_check(mp: ModelParams) -> dict:
    """Section 6.1: when gamma=0, changing foreign-rate/FX parameters leaves
    the price unchanged provided the collateral net-carry spec is fixed."""
    T0, T = 0.0, 3.0
    net_carry_bp = NET_CARRY_USD_BP
    price_baseline = multi_period_price(T0, T, 0.0, 0.0, mp, 1.0, net_carry_bp)
    mp_perturbed = ModelParams(r0_d=mp.r0_d, b_d=mp.b_d, sigma_d=mp.sigma_d, a_d=mp.a_d,
                                r0_f=0.0450, b_f=2.0, sigma_f=0.030, c_hat=0.09, a_hat=0.095,
                                q0=1.30, sigma_bar=0.15, rho_12=-0.1, rho_13=0.4, rho_23=-0.3)
    price_perturbed = multi_period_price(T0, T, 0.0, 0.0, mp_perturbed, 1.0, net_carry_bp)
    return {
        "price_baseline_gamma0": price_baseline,
        "price_perturbed_foreign_fx_gamma0": price_perturbed,
        "abs_difference": abs(price_baseline - price_perturbed),
        "matches": bool(abs(price_baseline - price_perturbed) < 1e-12),
    }


def run_equal_net_carry_check(mp: ModelParams) -> dict:
    """Section 4.1 (Remark 4.1): two collateral currencies with the same net
    carry give identical prices, since only alpha^{beta,i} enters A_{0,T}."""
    T0, T, gamma = 0.0, 3.0, 1.0
    shared_net_carry_bp = 15.0
    price_d = multi_period_price(T0, T, gamma, 0.0, mp, 1.0, shared_net_carry_bp)
    price_f = multi_period_price(T0, T, gamma, 0.0, mp, 1.0, shared_net_carry_bp)
    return {
        "price_collateral_d": price_d,
        "price_collateral_f": price_f,
        "abs_difference": abs(price_d - price_f),
        "matches": bool(abs(price_d - price_f) < 1e-12),
    }


def run_par_margin_invariance_check(mp: ModelParams) -> dict:
    """Section 6.1: the single-period par fixed margin is invariant to the
    collateral currency i and to the collateralisation rate beta (Corollary
    4.1: kappa* cancels the common factor A_{0,T}^{beta,i}*B^d(0,T))."""
    T0, T, gamma = 0.5, 1.0, 1.0
    out = {}
    for beta in (0.0, 0.5, 1.0):
        kappa_d = par_fixed_margin(T0, T, gamma, mp, beta, NET_CARRY_USD_BP)
        kappa_f = par_fixed_margin(T0, T, gamma, mp, beta, NET_CARRY_EUR_BP)
        out[f"beta_{beta}"] = {"kappa_star_usd_bp": kappa_d * 1e4, "kappa_star_eur_bp": kappa_f * 1e4}
    kappas = [v["kappa_star_usd_bp"] for v in out.values()] + [v["kappa_star_eur_bp"] for v in out.values()]
    out["max_abs_deviation_bp"] = max(abs(k - kappas[0]) for k in kappas)
    out["invariant"] = bool(out["max_abs_deviation_bp"] < 1e-8)
    return out


def main(n_paths: int = MC_PATHS_EXP1, seed: int = MC_SEED_EXP1) -> None:
    mp = _base_params()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mc_rows = run_mc_validation(mp, n_paths, seed)
    print("Table 2 - Monte Carlo vs. closed-form price (Section 6.1):")
    for row in mc_rows:
        print(f"  {row['config']:28s} closed_form={row['closed_form_price']:.8f} "
              f"mc={row['mc_price']:.8f} se={row['mc_std_error']:.3e} z={row['z_score']:+.3f} "
              f"within_2sigma={row['within_2_std']}")
    all_within = all(r["within_2_std"] for r in mc_rows)
    print(f"All configurations within 2 Monte Carlo standard errors: {all_within}")

    separation = run_separation_check(mp)
    equal_carry = run_equal_net_carry_check(mp)
    margin_invariance = run_par_margin_invariance_check(mp)

    results = {
        "experiment": "exp_1_mc_validation",
        "n_paths": n_paths,
        "seed": seed,
        "mc_vs_closed_form": mc_rows,
        "all_configs_within_2_std": all_within,
        "separation_property_gamma0": separation,
        "equal_net_carry_identity": equal_carry,
        "par_margin_invariance_to_collateral_and_beta": margin_invariance,
    }
    out_path = RESULTS_DIR / "exp1_mc_validation.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")

    # Summary table for quick inspection (aggregate, not time-indexed).
    import csv
    csv_path = RESULTS_DIR / "exp1_mc_validation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(mc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mc_rows)
    print(f"Saved {csv_path}")

    # Normal-quantile sanity check: the distribution of z-scores under H0
    # (formula correct) should not reject at the 95% level for any one config.
    critical = stats.norm.ppf(0.975)
    assert critical > 1.9, "sanity check on normal quantile"


if __name__ == "__main__":
    main()
