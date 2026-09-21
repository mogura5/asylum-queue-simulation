"""
End-to-end integration tests running the real AsylumSimulator (formalizes
smoke_test.py's checks as pytest assertions with fixed seeds, so a
regression fails CI instead of requiring someone to eyeball printed output).
"""

import pytest

from src.simulator import AsylumSimulator


@pytest.mark.parametrize("algorithm", ["FIFO", "LIFO", "PRIORITY"])
def test_runs_to_completion_without_error(test_params, algorithm):
    sim = AsylumSimulator(test_params, algorithm, n_officers=8, n_judges=6,
                           sim_months=36, seed=1)
    result = sim.run()
    assert result["n_completed"] > 0


def test_no_double_counting_under_harsh_understaffing(test_params):
    """Formalized version of smoke_test.py's check #3: dispatched-to-Stage-2
    cases must never be double-counted as both completed AND still queued.
    This is the exact invariant that would break if reneging ever left a
    ghost entry in the queue object."""
    dispatched_ids = []
    sim = AsylumSimulator(test_params, "FIFO", n_officers=2, n_judges=1,
                           sim_months=60, seed=3)
    orig_submit = sim.stage2.submit

    def tracking_submit(case):
        dispatched_ids.append(case.case_id)
        return orig_submit(case)

    sim.stage2.submit = tracking_submit
    result = sim.run()

    n_completed = result["n_completed"]
    backlog_s2 = result["backlog_s2"]
    n_dispatched = len(dispatched_ids)

    assert n_completed + backlog_s2 <= n_dispatched, (
        f"n_completed ({n_completed}) + backlog_s2 ({backlog_s2}) exceeds "
        f"n_dispatched ({n_dispatched}) -- a case was double-counted."
    )


def test_reneging_calibrates_near_configured_rate(test_params):
    """Checks the flat-probability-at-pull-time mechanism in isolation:
    observed renege_rate_obs should land close to
    p_renege * (1 - p_detained) (detained cases are exempt -- see
    test_stage.py), once two confounds are controlled for:

    1. appeal_rate is zeroed out here. With appeals left on, a denied case
       that appeals gets a SECOND stage-2 submission (and thus a second
       independent renege draw), which structurally inflates the observed
       rate above p_renege -- that's a real, separate, algebraically-derived
       effect (see redesign-decisions.md), not a mechanism bug, and
       conflating it with this test would make a genuine regression here
       harder to notice.
    2. capacity is set well above the arrival rate's requirement so the
       queue reaches steady state with ZERO backlog. Under sustained
       backlog, reneging (an instant, zero-duration outcome) always
       completes before the simulation's cutoff, while a case that
       proceeds to real service can be truncated mid-service at the exact
       moment the sim clock runs out (never calling on_complete, so it's
       silently dropped from n_completed). Because that truncation window
       only ever censors non-reneging outcomes, a congested run
       systematically inflates renege_rate_obs -- confirmed empirically:
       the same params at n_officers=6/n_judges=4/sim_months=48 (heavy,
       permanent backlog) gave renege_rate_obs=0.44 on a sample of just 9
       completed cases, while adequately staffed runs converge to ~0.28
       within 0.01-0.02 across seeds. This is a real methodological
       subtlety worth flagging for the paper (any welfare metric computed
       only over *completed* cases is vulnerable to this same
       end-of-window censoring bias when a scenario is chronically
       backlogged -- CEL's monthly lottery batches make it worth
       rechecking there too), not just a test-tuning detail.
    """
    params = dict(test_params)
    params["appeal_rate"] = 0.0
    sim = AsylumSimulator(params, "FIFO", n_officers=1000, n_judges=1100,
                           sim_months=200, seed=5)
    result = sim.run()
    assert result["backlog_s1"] == 0 and result["backlog_s2"] == 0, (
        "test capacity assumption broke -- queue is backlogged, so the "
        "tolerance below no longer accounts for end-of-window censoring."
    )
    expected = params["p_renege"] * (1 - params["p_detained"])
    observed = result["renege_rate_obs"]
    assert abs(observed - expected) < 0.03, (
        f"observed renege_rate_obs={observed:.3f} is far from the expected "
        f"~{expected:.3f} (= p_renege * (1 - p_detained)) -- reneging "
        f"mechanism may be broken."
    )


def test_cel_expedited_cases_never_appear_as_reneged(test_params):
    """Structural check on the CEL lottery pathway: a lottery-expedited case
    bypasses Stage 2 entirely, so it can never show up with outcome
    'reneged' -- if it did, that would mean expedited cases are somehow
    re-entering Stage 2's queue, contradicting the design."""
    params = dict(test_params)
    sim = AsylumSimulator(params, "CEL_balanced", n_officers=8, n_judges=6,
                           sim_months=36, seed=2)
    result = sim.run()
    expedited = [c for c in result["cases"] if c.outcome == "lottery_expedited"]
    assert len(expedited) > 0, "no cases were expedited -- check the smoke-scale thresholds/params"
    for c in expedited:
        assert c.s2_start == c.s2_end, "expedited case shows nonzero Stage 2 service time"
    reneged_case_ids = {c.case_id for c in result["cases"] if c.outcome == "reneged"}
    expedited_case_ids = {c.case_id for c in expedited}
    assert reneged_case_ids.isdisjoint(expedited_case_ids)