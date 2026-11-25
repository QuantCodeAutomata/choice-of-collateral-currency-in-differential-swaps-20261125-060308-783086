"""Baseline parameters for the joint Vasicek-Garman-Kohlhagen model of
Ding, Liu and Rutkowski, "Choice of Collateral Currency in Differential Swaps".

All rate-dynamics parameters are reported under the domestic pricing measure Q
(Section 6.1, Table 1). No external market data is used: every numerical
value below is the paper's own synthetic benchmark parameter.
"""

# --- Domestic (USD / SOFR) Vasicek parameters under Q (Section 6.1, Table 1) ---
R0_D = 0.03670          # initial domestic overnight rate r_0^d (annualised)
B_D = 3.000              # domestic mean-reversion speed b
SIGMA_D = 0.01200        # domestic short-rate volatility sigma
A_D = 0.1101             # domestic Vasicek drift constant a under Q

# --- Foreign (EUR / €STR) Vasicek parameters under Q (Section 6.1, Table 1) ---
R0_F = 0.01933           # initial foreign overnight rate r_0^f (annualised)
B_F = 3.000               # foreign mean-reversion speed b_hat
SIGMA_F = 0.01000         # foreign short-rate volatility sigma_hat
C_HAT = 0.057915          # foreign Vasicek drift constant c_hat under Q (Table 1)
# Foreign drift constant a_hat under the foreign martingale measure Q-hat,
# a_hat = c_hat + sigma_hat * sigma_bar * rho_23 (Section 3.1, eq. before (3.1));
# baseline value 0.05799 is reported explicitly in Section 6.1.
A_HAT = 0.05799

# --- FX (USD per EUR) parameters (Section 6.1) ---
Q0 = 1.1782               # initial FX rate Q_0 (USD per EUR)
SIGMA_BAR = 0.075          # FX volatility sigma_bar

# --- Correlations (Section 6.1); correlation matrix of (Z^1, Z^2, Z^3) ---
RHO_12 = 0.25             # domestic rate / foreign rate correlation
RHO_13 = -0.25            # domestic rate / FX correlation
RHO_23 = 0.10             # foreign rate / FX correlation

# --- Collateral net-carry spreads (Section 6.1) ---
ALPHA_D = 0.0              # domestic funding spread alpha^d; only the net carry
                           # alpha^{c,i}-alpha^i matters for the pricing formulas
                           # (paper does not specify alpha^d numerically)
NET_CARRY_USD_BP = 0.0     # USD collateral net carry alpha^{c,d} - alpha^d, in bp
NET_CARRY_EUR_BP = 20.0    # EUR collateral net carry alpha^{c,f} - alpha^f, in bp

# --- Collateralisation and payment conventions (Section 6.1) ---
BETA_FULL = 1.0            # full collateralisation beta = 1 (used unless stated)
DELTA_SEMIANNUAL = 0.5     # semiannual accrual length delta_j, in years

# =====================================================================
# Experiment 1: Monte Carlo validation (Section 6.1, Table 2)
# =====================================================================
MC_PATHS_EXP1 = 250_000    # exact-Gaussian Monte Carlo paths (Section 6.1)
MC_SEED_EXP1 = 20261125    # fixed random seed (not specified in the paper)

# Test configurations (Section 6.1, Table 2): (structure, T, gamma, collateral, beta)
EXP1_CONFIGS = [
    {"structure": "single", "T": 0.5, "gamma": 0.0, "collateral": "d", "beta": 1.0},
    {"structure": "single", "T": 1.5, "gamma": 1.0, "collateral": "f", "beta": 1.0},
    {"structure": "multi", "T": 3.0, "gamma": 0.0, "collateral": "d", "beta": 1.0},
    {"structure": "multi", "T": 3.0, "gamma": 1.0, "collateral": "f", "beta": 1.0},
]

# =====================================================================
# Experiment 2: collateral-currency value adjustment (Section 6.2, Table 3, Fig 1)
# =====================================================================
EXP2_MATURITIES = [1.0, 3.0, 5.0, 10.0]   # swap maturities in years (Section 6.2)
EXP2_TABLE_MATURITIES = [5.0, 10.0]       # maturities reported in Table 3
EXP2_GAMMAS = [0.0, 1.0]                  # contractual parameter gamma
EXP2_NOTIONAL = 100_000_000.0             # USD 100 million notional (Section 6.2)
EXP2_OFFMARKET_BP = [-200.0, -100.0, 0.0, 100.0, 200.0]  # off-market margins, Fig. 1
EXP2_TABLE3_OFFMARKET_BP = 100.0          # Table 3 uses par + 100bp

# =====================================================================
# Experiment 3: contractual quanto adjustment (Section 6.3, Figure 2)
# =====================================================================
EXP3_MATURITY = 5.0        # five-year differential swap (Section 6.3)
EXP3_GAMMA = 1.0           # contractual leg weight used for Delta kappa^q(1)
# Grid of rho_23 values exactly as printed in Figure 2's data table (Section
# 6.3, page 24) and sigma_bar levels (Section 6.3); a_hat is held fixed at
# A_HAT and c_hat = a_hat - sigma_hat*sigma_bar*rho_23 is recomputed for each
# point.
EXP3_RHO23_GRID = [-0.50, -0.35, -0.15, 0.00, 0.10, 0.25, 0.38, 0.50]
EXP3_SIGMA_BAR_GRID = [0.05, 0.075, 0.10]

# =====================================================================
# Experiment 4: discrete futures hedging (Section 6.4, Table 4, Figure 3)
# =====================================================================
EXP4_MATURITY = 3.0        # three-year semiannual swap (Section 6.4)
EXP4_GAMMAS = [0.0, 1.0]
EXP4_COLLATERALS = ["d", "f"]
EXP4_NOTIONAL = 1.0        # results reported in basis points of USD 1 notional
EXP4_DT = 1.0 / 252.0       # daily Euler-Maruyama step (Section 6.4)
# Full paper configuration: 10,000 common paths (Section 6.4). A full run with
# the vectorised Euler-Maruyama grid completes in well under a minute, so the
# paper's path count is used directly; EXP4_PATHS_RUN == EXP4_PATHS_PAPER.
EXP4_PATHS_PAPER = 10_000
EXP4_PATHS_RUN = 10_000
EXP4_SEED = 20261125        # fixed random seed (not specified in the paper)
EXP4_REBALANCE_FREQS = {"daily": 1, "weekly": 5, "monthly": 21}  # steps per rebalance
