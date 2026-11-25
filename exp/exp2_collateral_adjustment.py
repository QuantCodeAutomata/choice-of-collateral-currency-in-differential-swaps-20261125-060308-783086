"""Experiment 2: collateral-currency value adjustment (Section 6.2 of Ding,
Liu and Rutkowski, "Choice of Collateral Currency in Differential Swaps"),
isolating the deterministic collateral-carry channel.

The collateral-currency value adjustment for a fixed margin kappa (eq. before
Table 3, Section 6.2) is

    Delta_V_0^{f-d}(kappa) = DS_0^{beta,gamma,kappa}(T_0,n;f) - DS_0^{beta,gamma,kappa}(T_0,n;d),

computed with the *same* contractual margin kappa priced under the EUR-
collateral (net carry 20bp) and USD-collateral (net carry 0bp) specifications.
Table 3 reports the change in the par margin (EUR minus USD) and the
CSA-switch value for an off-market contract with margin = USD-collateral par
+ 100bp; Figure 1 sweeps the off-market margin from -200bp to +200bp around
the USD-collateral par, for gamma=1, at four maturities.

Run: python -m exp.exp2_collateral_adjustment
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    A_D, A_HAT, B_D, B_F, C_HAT, EXP2_GAMMAS, EXP2_MATURITIES, EXP2_NOTIONAL,
    EXP2_OFFMARKET_BP, EXP2_TABLE3_OFFMARKET_BP, EXP2_TABLE_MATURITIES,
    NET_CARRY_EUR_BP, NET_CARRY_USD_BP, Q0, R0_D, R0_F, RHO_12, RHO_13, RHO_23,
    SIGMA_BAR, SIGMA_D, SIGMA_F, BETA_FULL,
)
from src.pricing import ModelParams, multi_period_price, par_fixed_margin

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _base_params() -> ModelParams:
    return ModelParams(r0_d=R0_D, b_d=B_D, sigma_d=SIGMA_D, a_d=A_D,
                        r0_f=R0_F, b_f=B_F, sigma_f=SIGMA_F, c_hat=C_HAT, a_hat=A_HAT,
                        q0=Q0, sigma_bar=SIGMA_BAR, rho_12=RHO_12, rho_13=RHO_13, rho_23=RHO_23)


def collateral_value_adjustment(maturity: float, gamma: float, kappa: float, mp: ModelParams,
                                 beta: float = BETA_FULL) -> float:
    """Delta_V_0^{f-d}(kappa), per unit notional (Section 6.2)."""
    price_f = multi_period_price(0.0, maturity, gamma, kappa, mp, beta, NET_CARRY_EUR_BP)
    price_d = multi_period_price(0.0, maturity, gamma, kappa, mp, beta, NET_CARRY_USD_BP)
    return price_f - price_d


def run_table3(mp: ModelParams) -> list[dict]:
    """Table 3: par-margin change (EUR minus USD) and CSA-switch value for a
    contract with fixed margin = USD-collateral par + 100bp, per gamma and
    maturity (Section 6.2)."""
    rows = []
    for gamma in EXP2_GAMMAS:
        for maturity in EXP2_TABLE_MATURITIES:
            kappa_star_usd = par_fixed_margin(0.0, maturity, gamma, mp, BETA_FULL, NET_CARRY_USD_BP)
            kappa_star_eur = par_fixed_margin(0.0, maturity, gamma, mp, BETA_FULL, NET_CARRY_EUR_BP)
            par_margin_change_bp = (kappa_star_eur - kappa_star_usd) * 1e4
            kappa_offmarket = kappa_star_usd + EXP2_TABLE3_OFFMARKET_BP * 1e-4
            adjustment = collateral_value_adjustment(maturity, gamma, kappa_offmarket, mp)
            csa_switch_value = adjustment * EXP2_NOTIONAL
            rows.append({
                "gamma": gamma,
                "maturity": maturity,
                "kappa_star_usd_bp": kappa_star_usd * 1e4,
                "kappa_star_eur_bp": kappa_star_eur * 1e4,
                "par_margin_change_bp": par_margin_change_bp,
                "offmarket_bp": EXP2_TABLE3_OFFMARKET_BP,
                "csa_switch_value_usd": csa_switch_value,
            })
    return rows


def run_figure1(mp: ModelParams) -> list[dict]:
    """Figure 1: Delta_V_0^{f-d}(kappa) per USD 100M notional, gamma=1, for
    off-market margins from -200bp to +200bp around the USD-collateral par,
    at four maturities (Section 6.2)."""
    gamma = 1.0
    rows = []
    for maturity in EXP2_MATURITIES:
        kappa_star_usd = par_fixed_margin(0.0, maturity, gamma, mp, BETA_FULL, NET_CARRY_USD_BP)
        for offmarket_bp in EXP2_OFFMARKET_BP:
            kappa = kappa_star_usd + offmarket_bp * 1e-4
            adjustment = collateral_value_adjustment(maturity, gamma, kappa, mp)
            rows.append({
                "maturity": maturity,
                "offmarket_bp": offmarket_bp,
                "adjustment_usd_100m_notional": adjustment * EXP2_NOTIONAL,
            })
    return rows


def _plot_figure1(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    maturities = sorted({r["maturity"] for r in rows})
    for maturity in maturities:
        sub = sorted((r for r in rows if r["maturity"] == maturity), key=lambda r: r["offmarket_bp"])
        x = [r["offmarket_bp"] for r in sub]
        y = [r["adjustment_usd_100m_notional"] for r in sub]
        ax.plot(x, y, marker="o", label=f"{int(maturity)}Y")
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.axvline(0.0, color="grey", linewidth=0.8)
    ax.set_xlabel("Off-market fixed margin relative to USD-collateral par (bp)")
    ax.set_ylabel("Delta V_0^{f-d}(kappa), USD, per USD 100M notional")
    ax.set_title("Figure 1: collateral-currency value adjustment (gamma=1)")
    ax.legend(title="Maturity")
    fig.tight_layout()
    out_path = RESULTS_DIR / "exp2_figure1_collateral_adjustment.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    mp = _base_params()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    table3 = run_table3(mp)
    print("Table 3 - EUR minus USD collateral, par-margin change and CSA-switch value:")
    for row in table3:
        print(f"  gamma={row['gamma']:.0f} T={row['maturity']:.0f}Y "
              f"par_margin_change={row['par_margin_change_bp']:.6f}bp "
              f"csa_switch_value=USD {row['csa_switch_value_usd']:,.0f}")

    figure1 = run_figure1(mp)
    plot_path = _plot_figure1(figure1)
    print(f"Saved {plot_path}")

    results = {
        "experiment": "exp_2_collateral_currency_value_adjustment",
        "notional_usd": EXP2_NOTIONAL,
        "table3_par_margin_change_and_csa_switch": table3,
        "figure1_offmarket_sweep": figure1,
    }
    out_json = RESULTS_DIR / "exp2_collateral_adjustment.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_json}")

    out_csv = RESULTS_DIR / "exp2_table3.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table3[0].keys()))
        writer.writeheader()
        writer.writerows(table3)
    print(f"Saved {out_csv}")

    out_csv2 = RESULTS_DIR / "exp2_figure1.csv"
    with open(out_csv2, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(figure1[0].keys()))
        writer.writeheader()
        writer.writerows(figure1)
    print(f"Saved {out_csv2}")


if __name__ == "__main__":
    main()
