#!/usr/bin/env bash
# Runs the test suite and every experiment of
# "Choice of Collateral Currency in Differential Swaps", writing results to results/.
set -e

python -m pytest

python -m exp.exp1_mc_validation
python -m exp.exp2_collateral_adjustment
python -m exp.exp3_quanto_adjustment
python -m exp.exp4_discrete_hedging --full
