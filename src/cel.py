"""
Conditional Economic Lottery (CEL) support: economic-need scoring and the
two locked-in lottery configurations ('balanced', 'broad') ported from the
original cel_algorithm.py onto the new worker-pool architecture.

Per redesign-decisions.md ("CEL lottery pool -- confirmed as-is, no
change"), the qualitative mechanism is kept identical to the original:
each case is scored on complexity_score (already computed by case.py) and
a new econ_need_score; cases clearing BOTH thresholds enter a monthly
lottery pool; each simulated month, a p_lottery fraction of the pool is
drawn at random and granted immediately (skipping Stage 2 entirely -- no
judge hearing, no further wait); the rest are dispatched to the normal
Stage 2 queue (FIFO baseline -- see simulator.py) and the pool resets with
no carry-over, matching the original's monthly-reset design exactly.

Per the 2026-09 decision (see redesign-decisions.md), only two of the
original six diagnostic configs are carried forward for this project's
"two CEL variants": 'balanced' (moderate welfare targeting, ~61% of
screened cases qualify for the pool) and 'broad' (permissive, ~87%
qualify -- tests whether expediting helps even when almost everyone is
eligible). The other four configs from cel_algorithm.py are kept below,
commented out, in case a reviewer wants the full six-way comparison later
-- nothing about the mechanism changes to add them back, only which
threshold/probability triple gets used.
"""

import numpy as np

ECON_WEIGHTS = {
    "skill_match": 4.0,
    "english_proficiency": 2.0,
    "work_history": 2.0,
    "age_prime": 2.0,
}
SKILL_TIER_PROBS = [0.20, 0.25, 0.35, 0.20]  # BLS JOLTS 2023 job-opening tiers
ENGLISH_PROBS = [0.30, 0.20, 0.50]           # TRAC language-diversity proxy

LOTTERY_CONFIGS = {
    "balanced": {
        "complexity_threshold": 3.0, "econ_threshold": 5.33, "p_lottery": 0.30,
        "label": "Balanced (comp<=3.0, econ>=5.33, p=30%) -- ~61% qualify, ~18% expedited",
    },
    "broad": {
        "complexity_threshold": 4.5, "econ_threshold": 4.0, "p_lottery": 0.40,
        "label": "Broad (comp<=4.5, econ>=4.0, p=40%) -- ~87% qualify, ~35% expedited",
    },
    # Kept for reference, not run by default (see module docstring):
    # "strict":       {"complexity_threshold": 0.0,  "econ_threshold": 8.0,  "p_lottery": 0.10},
    # "econ_focused": {"complexity_threshold": 1.5,  "econ_threshold": 6.67, "p_lottery": 0.20},
    # "comp_only":    {"complexity_threshold": 1.5,  "econ_threshold": 0.0,  "p_lottery": 0.15},
    # "econ_only":    {"complexity_threshold": 10.0, "econ_threshold": 6.67, "p_lottery": 0.15},
}

CEL_ALGORITHM_PREFIX = "CEL_"


def cel_config_for_algorithm(algorithm: str):
    """'CEL_balanced' -> LOTTERY_CONFIGS['balanced']; None for non-CEL algorithm
    strings (FIFO/LIFO/PRIORITY), so callers can branch on `is not None`."""
    if not algorithm.startswith(CEL_ALGORITHM_PREFIX):
        return None
    name = algorithm[len(CEL_ALGORITHM_PREFIX):]
    if name not in LOTTERY_CONFIGS:
        raise ValueError(f"Unknown CEL config '{name}'; available: {list(LOTTERY_CONFIGS)}")
    return LOTTERY_CONFIGS[name]


def draw_econ_need_score(rng: np.random.Generator, params: dict) -> float:
    """Same calibration as the original cel_algorithm.py (sourced there to
    BLS JOLTS 2023 / DHS OIS 2022 -- see that file's module docstring for
    the full sourcing notes); ported verbatim, minus the per-component
    breakdown dict the original returned (never consumed downstream here)."""
    ew = ECON_WEIGHTS
    skill_tier = int(rng.choice(4, p=SKILL_TIER_PROBS))
    english_score = int(rng.choice(3, p=ENGLISH_PROBS))
    p_crim = params.get("p_criminal", 0.018)
    work_score = 0 if (rng.random() < p_crim) else 2
    age_score = 2 if (rng.random() < 0.60) else 0

    econ_score = (
        ew["skill_match"] * (skill_tier / 3.0)
        + ew["english_proficiency"] * (english_score / 2.0)
        + ew["work_history"] * (work_score / 2.0)
        + ew["age_prime"] * (age_score / 2.0)
    )
    return round(float(econ_score), 2)