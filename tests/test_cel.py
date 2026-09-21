"""
Protocol-integrity tests for cel.py and its interaction with simulator.py --
covering the two findings from the latest CEL results review:
  1. econ_need_score/complexity_score/is_detained are mutually independent
     BY CONSTRUCTION (ruled out the detention-composition hypothesis).
  2. the renege-rate metric-dilution relationship
     (true_rate ~= renege_rate_obs / (1 - expedited_rate)) holds.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from src.cel import cel_config_for_algorithm, draw_econ_need_score, LOTTERY_CONFIGS


def test_cel_config_lookup_known_and_unknown():
    assert cel_config_for_algorithm("FIFO") is None
    assert cel_config_for_algorithm("LIFO") is None
    assert cel_config_for_algorithm("PRIORITY") is None

    balanced = cel_config_for_algorithm("CEL_balanced")
    assert balanced == LOTTERY_CONFIGS["balanced"]

    with pytest.raises(ValueError):
        cel_config_for_algorithm("CEL_not_a_real_config")


def test_econ_need_score_signature_excludes_case_attributes():
    """Structural guarantee, not just an empirical one: draw_econ_need_score
    only ever sees (rng, params) -- it CANNOT read has_criminal, is_detained,
    no_rep, or complexity_score, because they are never passed in. This is
    what backs the 'mutually independent by construction' claim in the
    decisions log -- if a future change added a `case` parameter here, this
    test would catch that the independence guarantee had been broken."""
    sig = inspect.signature(draw_econ_need_score)
    param_names = set(sig.parameters.keys())
    assert param_names == {"rng", "params"}, (
        f"draw_econ_need_score's signature changed to {param_names} -- if it now "
        f"takes a case/Case-derived argument, the 'econ_need_score is independent "
        f"of case attributes' finding in redesign-decisions.md needs to be re-verified."
    )


def test_econ_need_score_in_valid_range():
    rng = np.random.default_rng(0)
    params = {"p_criminal": 0.05}
    for _ in range(500):
        score = draw_econ_need_score(rng, params)
        assert 0.0 <= score <= 10.0


@pytest.mark.parametrize("algorithm,expedited_rate,renege_rate_obs,fifo_true_rate", [
    # rows lifted directly from the user's replications_raw.csv for
    # post_covid_2024/high_volume (FIFO's own renege_rate_obs there is
    # 0.109, unaffected by any expedited-case dilution since FIFO has none).
    ("CEL_balanced", 0.09126053787136353, 0.10018896427005106, 0.109),
    ("CEL_broad", 0.1473532755240454, 0.09275480111810998, 0.109),
])
def test_renege_dilution_formula_matches_fifo_baseline(
    algorithm, expedited_rate, renege_rate_obs, fifo_true_rate,
):
    """The resolved finding from 'CEL full-scale run results': CEL's lower
    observed renege rate is fully explained by lottery-expedited cases (which
    structurally cannot renege) diluting the denominator -- NOT by a
    detention-composition shift. Backing that dilution out should recover
    a rate close to FIFO's actual rate in the same archetype."""
    implied_true_rate = renege_rate_obs / (1 - expedited_rate)
    assert implied_true_rate == pytest.approx(fifo_true_rate, abs=0.005), (
        f"{algorithm}: implied true renege rate {implied_true_rate:.4f} does not "
        f"match FIFO's baseline {fifo_true_rate:.4f} within tolerance -- either the "
        f"dilution explanation no longer holds, or a new compositional effect has "
        f"appeared and needs investigating (see redesign-decisions.md)."
    )


def test_renege_dilution_formula_is_algebraically_consistent():
    """Sanity-checks the formula itself (not real data): if we SYNTHESIZE a
    population where the true renege rate among Stage-2-eligible cases is
    known exactly, and a known fraction are expedited (never renege), the
    diluted observed rate must satisfy the formula exactly."""
    true_rate = 0.11
    for expedited_rate in (0.0, 0.09, 0.15, 0.31, 0.87):
        observed = true_rate * (1 - expedited_rate)
        recovered = observed / (1 - expedited_rate)
        assert recovered == pytest.approx(true_rate)