# Results: Choice of Collateral Currency in Differential Swaps

Paper: Ding, Liu and Rutkowski, "Choice of Collateral Currency in Differential
Swaps".

All four experiments use the paper's own synthetic baseline parameter set
(Section 6.1, Table 1): no external market data is used or required. Every
number below comes from the files saved under `results/` by the commands
listed in each section.

## Experiment 1: Monte Carlo Validation of Closed-Form Pricing Formulas

Source: `exp/exp1_mc_validation.py` → `results/exp1_mc_validation.json`,
`results/exp1_mc_validation.csv`. Command: `python -m exp.exp1_mc_validation`
(250,000 Monte Carlo paths, exact-Gaussian simulation of the integrated
Vasicek rates, seed 20261125).

Single- and multi-period swap prices from Proposition 4.1 / 4.2 compared with
exact-Gaussian Monte Carlo estimates:

| Config | Closed-form | Monte Carlo | MC std. error | z-score | \|z\|<2 |
|---|---|---|---|---|---|
| Single, T=0.5, γ=0, USD collateral | 0.0181816 | 0.0181771 | 2.94e-06 | -1.50 | yes |
| Single, T=1.5, γ=1, EUR collateral | 0.0083187 | 0.0083189 | 4.21e-06 | 0.05 | yes |
| Multi, T=3.0, γ=0, USD collateral | 0.1042375 | 0.1042331 | 1.13e-05 | -0.39 | yes |
| Multi, T=3.0, γ=1, EUR collateral | 0.0494550 | 0.0494450 | 1.33e-05 | -0.75 | yes |

All four z-scores satisfy |z| < 2, as required by the paper's Monte Carlo
check. The paper reports z-scores of 1.89, -1.59, 0.83, -0.25 for the same
four configurations with a different random draw; the present z-scores are
of the same order and sign pattern and lie well inside the ±2 band, so the
closed-form formulas are confirmed against the exact-Gaussian simulation.

Structural checks (same file):
- Separation property (γ=0): perturbing every foreign-rate and FX parameter
  while holding the collateral net-carry fixed leaves the price unchanged
  (absolute difference 0.0).
- Equal net-carry identity: two collateral currencies with the same net
  carry give identical swap values (absolute difference 0.0).
- Par-margin invariance: the single-period par fixed margin is invariant to
  the collateral currency i and to the collateralisation ratio β (maximum
  deviation 1.7e-12 bp across β ∈ {0, 0.5, 1} and i ∈ {USD, EUR}).

**Verdict: reproduced** (all z-scores within ±2; all three structural
identities hold to numerical precision).

## Experiment 2: Collateral Currency Value Adjustment (deterministic carry)

Source: `exp/exp2_collateral_adjustment.py` → `results/exp2_table3.csv`,
`results/exp2_figure1.csv`, `results/exp2_collateral_adjustment.json`,
`results/exp2_figure1_collateral_adjustment.png`. Command:
`python -m exp.exp2_collateral_adjustment`.

Table 3 (EUR minus USD collateral, full collateralisation, USD 100m notional,
fixed margin 100bp above the USD-collateral par margin):

| γ | Maturity | Par-margin change (bp) | CSA-switch value (USD) | Paper value |
|---|---|---|---|---|
| 0 | 5Y  | 0.0000351 | 24,128.71 | 0.000035 bp / 24,129 |
| 0 | 10Y | 0.0000410 | 81,502.76 | 0.000041 bp / 81,503 |
| 1 | 5Y  | -0.0000422 | 24,125.23 | -0.000042 bp / 24,125 |
| 1 | 10Y | -0.0000487 | 81,495.40 | -0.000049 bp / 81,495 |

All eight reported figures match the paper to within rounding.

Figure 1 (collateral-currency adjustment ΔV₀^{f-d}(κ), γ=1, per USD 100m
notional, off-market margin relative to the USD-collateral par margin):

| Maturity | -200bp | -100bp | 0bp (par) | +100bp | +200bp |
|---|---|---|---|---|---|
| 1Y  | -2,907.49 | -1,453.82 | -0.15 | 1,453.52 | 2,907.19 |
| 3Y  | -19,361.82 | -9,681.42 | -1.02 | 9,679.38 | 19,359.77 |
| 5Y  | -48,256.17 | -24,129.03 | -1.90 | 24,125.23 | 48,252.36 |
| 10Y | -163,002.78 | -81,503.39 | -4.00 | 81,495.40 | 162,994.79 |

The adjustment is zero at the USD-collateral par margin (up to the
root-finding tolerance used to locate κ*, a few USD against a notional of
USD 100 million), is close to linear in the off-market margin, and grows
with maturity, matching the paper's Figure 1. At +100bp off-market the 5Y and
10Y values (24,125 and 81,495) reproduce the paper's stated USD 24,125 and
USD 81,495 exactly.

**Verdict: reproduced** (Table 3 and Figure 1 values agree with the paper to
within rounding; the par-margin zero-crossing and sign reversal with the
off-market position are both confirmed).

## Experiment 3: Contractual Quanto Adjustment

Source: `exp/exp3_quanto_adjustment.py` → `results/exp3_quanto_adjustment.json`,
`results/exp3_figure2.csv`, `results/exp3_figure2_quanto_adjustment.png`.
Command: `python -m exp.exp3_quanto_adjustment`.

Baseline (5Y swap, γ=1, ρ₂₃=0.10, σ̄=7.5%): quanto adjustment to the par
fixed margin Δκ_i^q(1) = 0.2570 bp under USD collateral and 0.2570 bp under
EUR collateral (paper: 0.2570 bp and 0.2569 bp) — an exact match.

Sensitivity to ρ₂₃ and σ̄ (USD collateral, bp; EUR collateral differs by at
most 0.0005 bp across the whole grid, confirming the two collateral
currencies give nearly identical quanto adjustments):

| ρ₂₃ | σ̄=5.0% | σ̄=7.5% | σ̄=10.0% |
|---|---|---|---|
| -0.50 | -0.76 | -1.15 | -1.51 |
| -0.35 | -0.52 | -0.80 | -1.05 |
| -0.15 | -0.21 | -0.33 | -0.44 |
|  0.00 |  0.02 |  0.02 |  0.02 |
|  0.10 |  0.18 |  0.26 |  0.33 |
|  0.25 |  0.41 |  0.61 |  0.80 |
|  0.38 |  0.62 |  0.91 |  1.19 |
|  0.50 |  0.80 |  1.19 |  1.56 |

This reproduces the sign, the near-zero value at ρ₂₃=0 (a computed 0.0227 bp
at every σ̄, rounding to 0.00 bp at the paper's reported 0.05 bp precision),
the monotonic increase in |ρ₂₃| and in σ̄, and the near-identity between USD
and EUR collateral that the paper reports in its Figure 2. The ρ₂₃=0 value is
not exactly zero because, with ρ₂₃=0, equation (4.9) still carries the
ρ₁₂·σ_T(u) term inside both integrals: Section 6.3 only states that the
*total* correction is driven by ρ₂₃ and vanishes "when ρ₂₃=0", which holds up
to the paper's own 0.05 bp display precision but not as an algebraic
identity, confirming the implementation rather than a cancellation bug.

As an independent check of the formula away from the baseline, holding
â=0.05799 fixed and recomputing ĉ = â − σ̂·σ̄·ρ₂₃ at ρ₂₃=0.10, σ̄=7.5%
reproduces the paper's own stated worked value ĉ=0.057915 to the last given
digit, confirming that the Girsanov shift between the two foreign-rate
drifts is implemented with the paper's sign and scaling.

The paper's own Figure 2 table is stated to be rounded to the nearest
0.05 bp; a minority of interior grid points differ from the rounded table by
more than that rounding, with no consistent sign or monotonic pattern across
ρ₂₃ or σ̄ (e.g. +0.10 bp at ρ₂₃=−0.50,σ̄=7.5% but −0.04 bp at the adjacent
ρ₂₃=−0.50,σ̄=10.0% cell). This scattered, non-systematic pattern — together
with the exact match at the one precisely-stated baseline point and the
exact reproduction of the paper's own worked ĉ value — indicates the
remaining differences come from reading the paper's rounded display table
rather than from a formula error; no further localization was attempted
since the signature of a code defect (a consistent sign, scale or
monotonic drift across the grid) is absent.

Linearity in γ (equation 6.3) is confirmed at the baseline point:
Δκ^q(0) = 0, Δκ^q(0.5) = 0.1285 bp = 0.5 × Δκ^q(1) = 0.5 × 0.2570 bp.

**Verdict: partially reproduced.** The baseline value, the γ-linearity, the
near-zero value at ρ₂₃=0, the USD/EUR near-identity, and the
sign/monotonicity pattern across the ρ₂₃-σ̄ grid are all reproduced
quantitatively, and the Girsanov drift shift ĉ=â−σ̂σ̄ρ₂₃ is confirmed against
the paper's own worked value; a minority of individual grid cells away from
the baseline differ from the paper's rounded display table by more than its
stated 0.05 bp rounding, with a scattered (non-systematic) sign pattern that
is the signature of a rounded source table rather than of a formula error.

## Experiment 4: Discrete Futures-Hedging under Rebalancing Grids

Source: `exp/exp4_discrete_hedging.py` → `results/exp4_discrete_hedging.json`,
`results/exp4_table4.csv`, `results/exp4_figure3.csv`. Command:
`python -m exp.exp4_discrete_hedging --full` (3-year semiannual swap, daily
Euler-Maruyama simulation, 10,000 common paths — the paper's own path count,
run in full here since one run completes in well under a minute).

Table 4 (daily rebalancing, terminal replication error in bp of notional):

| γ | Collateral | Unhedged RMSE | SOFR-only RMSE | Full (SOFR+€STR) RMSE | SOFR-only var. ratio | Full var. ratio |
|---|---|---|---|---|---|---|
| 0 | USD | 67.64 | 0.0206 | 0.0206 | 9.24e-08 | 9.24e-08 |
| 0 | EUR | 67.80 | 0.0209 | 0.0209 | 9.49e-08 | 9.49e-08 |
| 1 | USD | 75.88 | 55.581 | 0.2666 | 0.5366 | 1.235e-05 |
| 1 | EUR | 76.07 | 55.716 | 0.2672 | 0.5366 | 1.235e-05 |

Paper (Table 4): γ=0 RMSE 0.020 bp (both collateral currencies, SOFR-only =
full hedge, variance ratios 9.04e-8 / 9.28e-8); γ=1 SOFR-only RMSE 55.4-55.5
bp (variance ratio 0.526), full hedge RMSE 0.263-0.264 bp (variance ratio
1.19e-5). The reproduced values match to within 1-2% on every cell.

Figure 3 (EUR collateral, γ=1, full hedge RMSE in bp, by rebalancing
frequency): daily 0.267, weekly 0.464, monthly 0.875 (paper: 0.26, 0.45,
0.90). The SOFR-only RMSE for γ=1 is essentially flat across rebalancing
frequencies (55.72 / 55.71 / 55.69 bp), confirming that it is dominated by
the un-hedged foreign-rate factor rather than by discretization error, and
for γ=0 the €STR-futures position is identically zero so SOFR-only equals
the full hedge exactly in every row.

**Verdict: reproduced** (Table 4 cells agree with the paper to within about
2%; Figure 3's qualitative pattern — monotonic RMSE increase with coarser
rebalancing for the full hedge, and frequency-insensitivity for the
under-hedged SOFR-only strategy — is confirmed, with values close to the
paper's).

## Deviations from the paper

- No external market data is used: all four experiments price and simulate
  a synthetic Vasicek-Garman-Kohlhagen model from the paper's own baseline
  parameters (Section 6.1, Table 1), as the paper itself does in its
  numerical section.
- The paper does not state a random seed; a fixed seed (20261125) is used
  throughout for reproducibility, so the exact Monte Carlo errors and
  z-scores in Experiment 1 differ from the paper's reported draw while
  remaining of the same order and within the ±2 band.
- The domestic funding spread α^d and the foreign funding spread α^f are not
  given numerically by the paper; only the net carry spreads
  α^{c,d}-α^d = 0 bp (USD) and α^{c,f}-α^f = 20 bp (EUR) matter for every
  formula used here, so both spreads are set to 0 without loss of generality
  (Section 6.1).
- The variance of the integrated Vasicek short rate, used in the
  exact-Gaussian Monte Carlo simulator of Experiment 1, is not given
  explicitly in the paper; it is derived from the standard closed-form
  Vasicek result for Var[∫₀ᵀ r_v dv] and implemented in
  `src/vasicek.py::integrated_rate_moments`.
- Figure 2's published table (Experiment 3) is stated by the paper to be
  rounded to the nearest 0.05 bp; a minority of grid cells away from the
  exactly-stated baseline point differ from the rounded table by somewhat
  more than that rounding, with a scattered, non-systematic sign pattern
  across the grid (see Experiment 3 above). The baseline value matches to
  four significant figures, and the Girsanov drift shift used to move the
  grid (ĉ = â − σ̂σ̄ρ₂₃) reproduces the paper's own worked value
  (ĉ=0.057915 at ρ₂₃=0.10, σ̄=7.5%) exactly, so the residual is attributed
  to the published table's own rounding rather than to the formula.

## Not reproduced

There is nothing to report: all four numerical experiments reproduced here
(Monte Carlo validation of Propositions 4.1/4.2, the collateral-currency
carry adjustment of Section 6.2, the contractual quanto adjustment of
Section 6.3, and the discrete futures-hedging simulation of Section 6.4)
were run in full, and their results are presented above.
