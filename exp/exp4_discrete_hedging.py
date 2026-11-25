"""Experiment 4: discrete-time implementation of the futures-based hedging
strategies of Proposition 5.2 (Sections 5.1-5.2 and 6.4 of Ding, Liu and
Rutkowski, "Choice of Collateral Currency in Differential Swaps").

Three strategies are compared on a 3-year semiannual differential swap,
simulated under Euler-Maruyama on a daily grid with common random paths
(Section 6.4):

* ``unhedged``  -- cash account only;
* ``sofr_only`` -- domestic (SOFR) futures only, €STR futures position set to zero;
* ``full``      -- both SOFR and €STR futures (Proposition 5.2).

Table 4 reports terminal RMSE (bp of notional) and variance ratio relative to
the unhedged strategy, for gamma in {0,1} and both collateral currencies,
under daily rebalancing. Figure 3 varies the rebalancing frequency (daily,
weekly, monthly) under EUR collateral.

Run: python -m exp.exp4_discrete_hedging [--full]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import (
    A_D, A_HAT, B_D, B_F, BETA_FULL, C_HAT, EXP4_DT, EXP4_GAMMAS,
    EXP4_MATURITY, EXP4_PATHS_PAPER, EXP4_PATHS_RUN, EXP4_REBALANCE_FREQS,
    EXP4_SEED, NET_CARRY_EUR_BP, NET_CARRY_USD_BP, Q0, R0_D, R0_F, RHO_12,
    RHO_13, RHO_23, SIGMA_BAR, SIGMA_D, SIGMA_F,
)
from src.hedging import simulate_hedge
from src.pricing import ModelParams, par_fixed_margin

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
_NET_CARRY = {"d": NET_CARRY_USD_BP, "f": NET_CARRY_EUR_BP}


def _base_params() -> ModelParams:
    return ModelParams(r0_d=R0_D, b_d=B_D, sigma_d=SIGMA_D, a_d=A_D,
                        r0_f=R0_F, b_f=B_F, sigma_f=SIGMA_F, c_hat=C_HAT, a_hat=A_HAT,
                        q0=Q0, sigma_bar=SIGMA_BAR, rho_12=RHO_12, rho_13=RHO_13, rho_23=RHO_23)


def run_table4(mp: ModelParams, n_paths: int, seed: int) -> list[dict]:
    """Table 4: terminal RMSE and variance ratio under daily rebalancing, for
    gamma in {0,1} and both collateral currencies, unhedged/SOFR-only/full
    strategies (Section 6.4)."""
    rows = []
    for gamma in EXP4_GAMMAS:
        for collateral in ("d", "f"):
            net_carry_bp = _NET_CARRY[collateral]
            kappa = par_fixed_margin(0.0, EXP4_MATURITY, gamma, mp, BETA_FULL, NET_CARRY_USD_BP)
            unhedged = simulate_hedge(mp, EXP4_MATURITY, gamma, kappa, BETA_FULL, net_carry_bp,
                                       n_paths, seed, "unhedged")
            sofr_only = simulate_hedge(mp, EXP4_MATURITY, gamma, kappa, BETA_FULL, net_carry_bp,
                                        n_paths, seed, "sofr_only")
            full = simulate_hedge(mp, EXP4_MATURITY, gamma, kappa, BETA_FULL, net_carry_bp,
                                   n_paths, seed, "full")
            rows.append({
                "gamma": gamma,
                "collateral": collateral,
                "unhedged_rmse_bp": unhedged.rmse_bp,
                "sofr_only_rmse_bp": sofr_only.rmse_bp,
                "full_rmse_bp": full.rmse_bp,
                "sofr_only_variance_ratio": sofr_only.variance / unhedged.variance,
                "full_variance_ratio": full.variance / unhedged.variance,
            })
    return rows


def run_figure3(mp: ModelParams, n_paths: int, seed: int) -> list[dict]:
    """Figure 3: terminal RMSE (bp) under EUR collateral as a function of the
    rebalancing frequency, for gamma=0 (SOFR-only) and gamma=1 (SOFR-only and
    full hedge) (Section 6.4)."""
    rows = []
    net_carry_bp = NET_CARRY_EUR_BP
    for freq_name, steps in EXP4_REBALANCE_FREQS.items():
        for gamma in EXP4_GAMMAS:
            kappa = par_fixed_margin(0.0, EXP4_MATURITY, gamma, mp, BETA_FULL, NET_CARRY_USD_BP)
            sofr_only = simulate_hedge(mp, EXP4_MATURITY, gamma, kappa, BETA_FULL, net_carry_bp,
                                        n_paths, seed, "sofr_only", rebalance_every=steps)
            row = {"frequency": freq_name, "gamma": gamma, "sofr_only_rmse_bp": sofr_only.rmse_bp}
            if gamma == 1.0:
                full = simulate_hedge(mp, EXP4_MATURITY, gamma, kappa, BETA_FULL, net_carry_bp,
                                       n_paths, seed, "full", rebalance_every=steps)
                row["full_rmse_bp"] = full.rmse_bp
            rows.append(row)
    return rows


def main(full: bool = False) -> None:
    mp = _base_params()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n_paths = EXP4_PATHS_PAPER if full else EXP4_PATHS_RUN
    seed = EXP4_SEED

    table4 = run_table4(mp, n_paths, seed)
    print(f"Table 4 - discrete hedge replication error, {n_paths} paths, daily rebalancing (Section 6.4):")
    for row in table4:
        print(f"  gamma={row['gamma']:.0f} collateral={row['collateral']} "
              f"unhedged={row['unhedged_rmse_bp']:.3f}bp sofr_only={row['sofr_only_rmse_bp']:.3f}bp "
              f"full={row['full_rmse_bp']:.3f}bp var_ratio(full)={row['full_variance_ratio']:.3e}")

    figure3 = run_figure3(mp, n_paths, seed)
    print("Figure 3 - EUR collateral terminal RMSE (bp) vs. rebalancing frequency:")
    for row in figure3:
        print(f"  {row}")

    results = {
        "experiment": "exp_4_discrete_futures_hedging",
        "n_paths": n_paths,
        "n_paths_paper": EXP4_PATHS_PAPER,
        "full_configuration": full,
        "seed": seed,
        "table4_hedge_comparison": table4,
        "figure3_rebalancing_frequency": figure3,
    }
    out_json = RESULTS_DIR / "exp4_discrete_hedging.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_json}")

    import csv
    out_csv = RESULTS_DIR / "exp4_table4.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table4[0].keys()))
        writer.writeheader()
        writer.writerows(table4)
    print(f"Saved {out_csv}")

    out_csv2 = RESULTS_DIR / "exp4_figure3.csv"
    with open(out_csv2, "w", newline="") as f:
        fieldnames = sorted({k for row in figure3 for k in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(figure3)
    print(f"Saved {out_csv2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 4: discrete futures hedging")
    parser.add_argument("--full", action="store_true",
                         help=f"use the paper's {EXP4_PATHS_PAPER} common paths instead of the reduced run")
    args = parser.parse_args()
    main(full=args.full)
