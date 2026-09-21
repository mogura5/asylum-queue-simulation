"""
Protocol-integrity tests for case.py: the two documented, deliberate design
decisions in the Case model (redesign-decisions.md, items "Appeal mechanism",
"Complexity weights", and "charge_dist rounding crash").
"""

import numpy as np
import pytest

from src.case import make_case, compute_complexity, Case


def test_appeal_reuses_template_profile_exactly(test_params):
    """Item: 'Appeal mechanism -- finalized design'. The original bug drew an
    entirely new random Case for an appeal, only keeping case_id/is_appeal.
    An appeal must instead carry over has_rep/is_detained/has_criminal/
    nbr_charges/complexity_score UNCHANGED from the original."""
    rng = np.random.default_rng(1)
    original = make_case(rng, test_params, arrival_time=0.0, case_id=1)

    appeal_rng = np.random.default_rng(999)  # different stream on purpose
    appeal = make_case(
        appeal_rng, test_params, arrival_time=5.0, case_id=1,
        is_appeal=True, template=original,
    )

    assert appeal.has_rep == original.has_rep
    assert appeal.is_detained == original.is_detained
    assert appeal.has_criminal == original.has_criminal
    assert appeal.nbr_charges == original.nbr_charges
    assert appeal.complexity_score == original.complexity_score
    assert appeal.is_appeal is True
    assert appeal.case_id == original.case_id
    # Only Stage 2 hearing mechanics are meant to be freshly drawn -- given a
    # different rng stream, the service-time draw should (almost certainly)
    # differ from the original's, proving it was NOT copied.
    assert appeal.service_time_s2 != original.service_time_s2


def test_appeal_priority_gets_appeal_bonus(test_params):
    """base_priority includes a +2.0 bump for is_appeal=True (case.py's
    make_case); confirms the appeal path runs through the same priority
    formula as a fresh case, just with is_appeal set."""
    rng = np.random.default_rng(2)
    original = make_case(rng, test_params, arrival_time=0.0, case_id=7)
    appeal = make_case(
        rng, test_params, arrival_time=1.0, case_id=7,
        is_appeal=True, template=original,
    )
    assert appeal.base_priority == pytest.approx(original.base_priority + 2.0)


@pytest.mark.parametrize("has_rep,has_criminal,nbr_charges", [
    (True, False, 0),
    (False, True, 3),
    (True, True, 2),
])
def test_complexity_sign_flip_direction(has_rep, has_criminal, nbr_charges):
    """Item: 'Complexity weights -- direction was backwards'. Complexity is
    now HIGHEST for unrepresented + criminal-record + multi-charge profiles
    (the fast-tracked, lower-due-process profile), not lowest."""
    fast_tracked = compute_complexity(has_rep=False, has_criminal=True, nbr_charges=3)
    slow_full_process = compute_complexity(has_rep=True, has_criminal=False, nbr_charges=0)
    assert fast_tracked > slow_full_process
    assert slow_full_process == 0.0


def test_effective_s2_time_decreases_with_complexity():
    """effective_s2_time() must scale DOWN as complexity rises (the sign-flip
    decision), and must equal the raw draw exactly at complexity_score=0."""
    base = Case(
        case_id=1, arrival_time=0.0, has_rep=True, is_detained=False,
        has_criminal=False, nbr_charges=0, complexity_score=0.0,
        base_priority=1.0, service_time_s1=1.0, service_time_s2=10.0,
    )
    assert base.effective_s2_time() == pytest.approx(10.0)

    fast_tracked = Case(
        case_id=2, arrival_time=0.0, has_rep=False, is_detained=False,
        has_criminal=True, nbr_charges=3, complexity_score=10.0,
        base_priority=1.0, service_time_s1=1.0, service_time_s2=10.0,
    )
    assert fast_tracked.effective_s2_time() < base.effective_s2_time()
    assert fast_tracked.effective_s2_time() == pytest.approx(5.0)  # 50% reduction at score=10


def test_charge_dist_renormalizes_when_not_exactly_one(test_params):
    """Item: 'Fixed in passing: charge_dist rounding crash'. A charge_dist
    that sums to 1.0001 (as compute_parameter.py's 4dp rounding can produce)
    must not raise, and drawn nbr_charges must stay within the valid range."""
    params = dict(test_params)
    params["charge_dist"] = [0.4501, 0.3501, 0.1501, 0.0501]  # sums to 1.0004
    rng = np.random.default_rng(3)
    for i in range(200):
        case = make_case(rng, params, arrival_time=0.0, case_id=i)
        assert case.nbr_charges in (0, 1, 2, 3)


def test_charge_dist_renormalizes_when_under_one(test_params):
    params = dict(test_params)
    params["charge_dist"] = [0.44, 0.34, 0.14, 0.04]  # sums to 0.96
    rng = np.random.default_rng(4)
    # Should not raise ValueError("probabilities do not sum to 1")
    for i in range(50):
        make_case(rng, params, arrival_time=0.0, case_id=i)