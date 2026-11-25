# Repository Guide for Coding Agents

This repository reproduces selected numerical experiments from Ding, Liu and
Rutkowski, "Choice of Collateral Currency in Differential Swaps": closed-form
pricing of collateralised SOFR/€STR differential swaps under a joint
Vasicek-Garman-Kohlhagen model, the deterministic collateral-carry and
stochastic quanto value adjustments between USD and EUR collateral, and
discrete futures-based hedging of the contractual foreign-rate leg.

## Layout

- `src/` — flat library modules, the single shared pipeline every experiment
  calls:
  - `config.py` — every paper-derived parameter as module-level UPPER_CASE
    variables, grouped by experiment, each with a source comment.
  - `vasicek.py` — Vasicek discount factors (unconditional and conditional)
    and the mean/variance of the integrated short rate.
  - `quanto.py` — the deterministic quanto correction Γ₀^{qf}(U,T)
    (equation 4.9), by numerical integration and by a closed-form reduction.
  - `pricing.py` — the collateral adjustment factor, the €STR futures rate,
    single- and multi-period swap pricing (Propositions 4.1, 4.2), and the
    par fixed margin (Corollary 4.1), with and without the quanto term.
  - `mc_simulation.py` — exact-Gaussian Monte Carlo pricer used to validate
    the closed-form formulas.
  - `hedging.py` — the discrete futures-hedging simulator (Proposition 5.2)
    with Euler-Maruyama state dynamics under Q.
- `exp/` — one script per experiment, each calling only the `src/` functions
  above and writing its outputs to `results/`.
- `tests/` — pytest unit tests for every `src/` module.
- `results/` — aggregate metrics (`.json`/`.csv`) and static figures
  (`.png`), plus `RESULTS.md` with the comparison to the paper.
- `data/` — empty package. No experiment here loads market data or uses a
  generated dataset: the model's own Monte Carlo and Euler-Maruyama paths
  are simulated directly inside `src/mc_simulation.py` and `src/hedging.py`
  as part of the pricing/hedging method itself, not as a prepared dataset.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -m "not market_data"   # no market_data tests exist; plain `python -m pytest` also works
python -m exp.exp1_mc_validation
python -m exp.exp2_collateral_adjustment
python -m exp.exp3_quanto_adjustment
python -m exp.exp4_discrete_hedging --full   # paper's full 10,000-path configuration
./reproduce.sh                                # tests + all four experiments
```

No API keys are needed anywhere in this repository.

## Conventions

- All experiment parameters live only in `src/config.py` as module-level
  variables with a unit and a source comment (a paper section/equation, or
  "not specified in the paper"); no parameter is hardcoded elsewhere.
- `exp/*.py` scripts call the `src/` functions with the signatures defined
  there; they do not duplicate pricing logic or invent their own inputs.
- Every run of an `exp/*.py` script overwrites the corresponding files under
  `results/`. After a rerun, re-check the numbers and verdicts in
  `results/RESULTS.md` against the new files and update them by hand; they
  are not regenerated automatically.
- Only aggregate metrics/tables and static `.png`/`.pdf` figures are saved
  under `results/`; no raw or simulated time series are ever committed, and
  there is no `.html` output anywhere in the repository.
