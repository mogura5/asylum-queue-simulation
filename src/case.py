"""
Shared Case model and case-generation logic for the asylum queue simulator.

COMPLEXITY MODEL (revised per redesign-decisions.md, section 3):
Complexity now represents how likely a case is to move through the system
via lower-due-process channels (in-absentia orders, expedited dockets,
fewer continuances) rather than "harder to adjudicate, therefore slower."

The calibration diagnostic checked the three complexity attributes that have
real TRAC fields behind them against actual processing_months, and found
the ORIGINAL direction backwards on all three:
    no_rep=True:            median  8.9mo   vs  19.4mo represented
    has_criminal_record=True: median 7.4mo   vs  16.0mo without
    multi_charge (>=2)=True:  median 12.1mo  vs  16.3mo with fewer

So complexity now scales Stage 2 service time DOWN as it increases, not up.
`is_family` and `filed_late` were dropped entirely: `case_type` (checked as
the intended source for is_family) has no 'F' value anywhere in the real
data, and filed_late never had a real field behind it at all -- both were
hardcoded assumptions (p_family=0.30, p_late=0.25) with zero grounding.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

# Weights re-normalized to sum to 10 across the three attributes that have
# real data behind them. Relative ordering kept from the original table
# (no_rep highest, then has_criminal and multi_charge) pending a fuller
# re-derivation of magnitudes from the diagnostic's numbers.
COMPLEXITY_WEIGHTS = {
    "no_rep": 5.0,
    "has_criminal": 3.0,
    "multi_charge": 2.0,
}

# Fraction by which Stage 2 service time is reduced at complexity_score == 10
# (fully "fast-tracked" profile: unrepresented, criminal record, multiple
# charges). At complexity_score == 0, no reduction is applied.
COMPLEXITY_SERVICE_REDUCTION = 0.50


@dataclass
class Case:
    case_id: int
    arrival_time: float          # months since sim start
    has_rep: bool
    is_detained: bool
    has_criminal: bool
    nbr_charges: int
    complexity_score: float      # 0-10, weighted composite (see above)
    base_priority: float
    service_time_s1: float
    service_time_s2: float       # base draw; scaled by complexity at service start
    is_appeal: bool = False
    econ_need_score: float = 0.0  # only meaningful for CEL runs; see cel.py

    # Filled during simulation (set by Stage; see stage.py)
    s1_queue_enter: float = 0.0
    s1_start: float = 0.0
    s1_end: float = 0.0
    s2_queue_enter: float = 0.0
    s2_start: float = 0.0
    s2_end: float = 0.0
    outcome: str = ""
    total_time: float = 0.0

    def effective_s2_time(self) -> float:
        """Stage 2 service time scaled DOWN by complexity (see module docstring)."""
        reduction = COMPLEXITY_SERVICE_REDUCTION * self.complexity_score / 10.0
        return self.service_time_s2 * (1.0 - reduction)

    def __lt__(self, other: "Case") -> bool:
        return self.base_priority > other.base_priority


def compute_complexity(has_rep: bool, has_criminal: bool, nbr_charges: int,
                        weights: dict = COMPLEXITY_WEIGHTS) -> float:
    return (
        weights["no_rep"] * (not has_rep) +
        weights["has_criminal"] * has_criminal +
        weights["multi_charge"] * (nbr_charges >= 2)
    )


def draw_service_time(rng: np.random.Generator, mean: float, std: float) -> float:
    """Log-normal service time draw. Falls back to exponential if std<=0."""
    if std <= 0 or mean <= 0:
        return max(0.01, rng.exponential(mean))
    sigma2 = np.log(1 + (std / mean) ** 2)
    mu_ln = np.log(mean) - sigma2 / 2
    return max(0.01, rng.lognormal(mu_ln, np.sqrt(sigma2)))


def make_case(rng: np.random.Generator, params: dict, arrival_time: float,
              case_id: int, is_appeal: bool = False,
              template: Optional[Case] = None) -> Case:
    """
    Generate a new case, OR -- for an appeal -- build the Stage-2 re-entry
    using the ORIGINAL case's real profile (`template`) rather than
    redrawing attributes from scratch. This replaces the original bug where
    an appeal simulated an entirely different random person's case, only
    keeping case_id and is_appeal=True (see redesign-decisions.md).

    Only Stage 2 hearing mechanics (service time, priority for the new
    hearing) are freshly generated for an appeal; representation, detention,
    criminal history, charge count, and complexity_score are carried over
    unchanged from `template`.
    """
    if template is not None:
        has_rep = template.has_rep
        detained = template.is_detained
        has_criminal = template.has_criminal
        nbr_charges = template.nbr_charges
        complexity = template.complexity_score
    else:
        has_rep = rng.random() < params["p_rep"]
        detained = rng.random() < params["p_detained"]
        has_criminal = rng.random() < params.get("p_criminal", 0.018)
        # charge_dist from sim_parameters_FINAL.json is rounded to 4dp by
        # compute_parameter.py's _charge_distribution(), which can leave it
        # summing to 1.0001 or 0.9999 instead of exactly 1.0 -- numpy's
        # rng.choice() requires an EXACT sum (tolerance ~1e-8), so it's
        # renormalized here defensively rather than assuming any calibrated
        # input is already exact.
        charge_p = np.asarray(params.get("charge_dist", [0.45, 0.35, 0.15, 0.05]), dtype=float)
        charge_p = charge_p / charge_p.sum()
        nbr_charges = int(rng.choice([0, 1, 2, 3], p=charge_p))
        complexity = compute_complexity(has_rep, has_criminal, nbr_charges)

    priority = 1.0
    if detained:
        priority += 3.0
    if has_rep:
        priority += 1.0
    if is_appeal:
        priority += 2.0
    priority -= (complexity / 10.0) * 1.0

    # Prefer the recalibrated effective_service_time_s1/s2 fields
    # (compute_parameter.py's fix: derived from REAL OBSERVED THROUGHPUT --
    # n_completed / cluster_months_observed -- rather than raw elapsed
    # pendency). service_time_mean/std describe total elapsed months from
    # arrival to decision, which already contains years of real queueing
    # delay; using them directly as server-occupied time double-counts the
    # wait and produces an artificially catastrophic overload (~1000x
    # utilization was observed before this fix -- see
    # redesign-decisions.md, "Service time recalibration"). Falls back to
    # the old 20/80 split of service_time_mean for params dicts that
    # predate the fix (e.g. smoke_test.py's synthetic TEST_PARAMS).
    if "effective_service_time_s1" in params and "effective_service_time_s2" in params:
        s1_mean = params["effective_service_time_s1"]
        s2_mean = params["effective_service_time_s2"]
        s1_std = params.get("effective_service_time_s1_std", s1_mean * 0.5)
        s2_std = params.get("effective_service_time_s2_std", s2_mean * 0.5)
    else:
        mu_s = params["service_time_mean"]
        std_s = params["service_time_std"] if params["service_time_std"] > 0 else mu_s * 0.5
        s1_mean, s1_std = mu_s * 0.20, std_s * 0.20
        s2_mean, s2_std = mu_s * 0.80, std_s * 0.80

    return Case(
        case_id=case_id,
        arrival_time=arrival_time,
        has_rep=has_rep,
        is_detained=detained,
        has_criminal=has_criminal,
        nbr_charges=nbr_charges,
        complexity_score=round(complexity, 2),
        base_priority=round(priority, 2),
        service_time_s1=draw_service_time(rng, s1_mean, s1_std),
        service_time_s2=draw_service_time(rng, s2_mean, s2_std),
        is_appeal=is_appeal,
    )