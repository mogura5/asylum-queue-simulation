"""
AsylumSimulator: wires the shared Case model (case.py), pluggable queue
strategies (queues.py), and the Stage worker-pool abstraction (stage.py)
into the two-stage tandem queue (Officer -> Judge) described in
redesign-decisions.md.

This replaces simulation_runner.py's AsylumSimulator class. FIFO, LIFO, and
Priority all run through this same class -- the only difference is which
QueueStrategy is plugged into each Stage. CEL's lottery pathway and the
Poisson-monthly-arrivals variant (fifo_lifo_experiment.py) are NOT ported
here yet -- this is the core two-stage mechanics only, to be reviewed before
building the remaining experiment scripts on top of it.

NOTE ON CAPACITY: n_officers/n_judges are passed in explicitly rather than
computed from a hardcoded national-total constant divided evenly across
archetypes (the original bug). The caller (whatever replaces
compute_parameter.py's archetype-building step) is responsible for deriving
real per-archetype officer/judge counts -- see redesign-decisions.md's
capacity calibration section (judges from real ij_code counts, officers
scaled proportionally to case volume).
"""

import simpy
import numpy as np
from case import Case, make_case
from queues import make_queue
from cel import cel_config_for_algorithm, draw_econ_need_score


class AsylumSimulator:
    def __init__(self, params: dict, algorithm: str,
                 n_officers: int, n_judges: int,
                 sim_months: float, seed: int):
        self.params = params
        self.algorithm = algorithm
        self.sim_months = sim_months
        self.rng = np.random.default_rng(seed)

        self.lam = params["arrival_rate_lambda"]
        self.grant_rate = params["grant_rate"]
        self.appeal_rate = params["appeal_rate"]
        self.p_renege = params["p_renege"]

        # CEL ("CEL_balanced" / "CEL_broad") is not itself a queue discipline
        # -- it's a lottery layer that sits between Stage 1 and Stage 2 (see
        # cel.py and _advance_to_stage2/_lottery_draw_process below). Cases
        # that don't win the lottery fall through to a plain FIFO Stage 2
        # queue, matching the original cel_algorithm.py's default baseline.
        self.cel_cfg = cel_config_for_algorithm(algorithm)
        self.lottery_pool: list[Case] = []

        self.env = simpy.Environment()
        self.completed_cases: list[Case] = []
        self._case_counter = 0

        from stage import Stage  # local import avoids a circular-import footgun

        queue_algorithm = "FIFO" if self.cel_cfg is not None else algorithm
        s1_queue = make_queue(queue_algorithm)
        s2_queue = make_queue(queue_algorithm)

        self.stage1 = Stage(
            self.env, capacity=n_officers, queue=s1_queue,
            service_time_fn=lambda c: c.service_time_s1,
            on_complete=self._advance_to_stage2,
            start_attr="s1_start", end_attr="s1_end", queue_enter_attr="s1_queue_enter",
        )
        self.stage2 = Stage(
            self.env, capacity=n_judges, queue=s2_queue,
            service_time_fn=lambda c: c.effective_s2_time(),
            on_complete=self._resolve_case,
            start_attr="s2_start", end_attr="s2_end", queue_enter_attr="s2_queue_enter",
            renege_prob=self.p_renege,
            renege_exempt=lambda c: c.is_detained,
            on_renege=self.completed_cases.append,
        )
        self.stage1.set_rng(self.rng)
        self.stage2.set_rng(self.rng)

    # ----------------------------------------------------------
    # Stage transitions
    # ----------------------------------------------------------

    def _advance_to_stage2(self, case: Case):
        """All screened cases proceed from Officer to Judge (matches
        original simulation_runner.py -- Stage 1 has no reneging path,
        confirmed in redesign-decisions.md) -- UNLESS this is a CEL run and
        the case clears both lottery thresholds, in which case it enters
        the monthly lottery pool instead (see _lottery_draw_process)."""
        if self.cel_cfg is not None:
            qualifies = (
                case.complexity_score <= self.cel_cfg["complexity_threshold"]
                and case.econ_need_score >= self.cel_cfg["econ_threshold"]
            )
            if qualifies:
                self.lottery_pool.append(case)
                return
        self.stage2.submit(case)

    def _lottery_draw_process(self):
        """Once per simulated month, draw p_lottery fraction of the current
        pool at random and grant them immediately (skip Stage 2 entirely --
        no judge hearing, no further wait). Everyone else in the pool is
        dispatched to the normal Stage 2 queue and the pool resets with no
        carry-over -- identical semantics to the original cel_algorithm.py
        (confirmed unchanged in redesign-decisions.md's CEL section)."""
        while True:
            yield self.env.timeout(1.0)
            if not self.lottery_pool:
                continue
            pool, self.lottery_pool = self.lottery_pool, []
            p_lottery = self.cel_cfg["p_lottery"]
            n_draw = int(len(pool) * p_lottery)
            if n_draw == 0 and self.rng.random() < p_lottery:
                n_draw = 1  # ensure at least one draw when the pool is tiny
            n_draw = min(n_draw, len(pool))
            selected = set()
            if n_draw > 0:
                idx = self.rng.choice(len(pool), size=n_draw, replace=False)
                selected = set(idx.tolist())
            for i, case in enumerate(pool):
                if i in selected:
                    case.s2_start = self.env.now
                    case.s2_end = self.env.now
                    case.total_time = self.env.now - case.arrival_time
                    case.outcome = "lottery_expedited"
                    self.completed_cases.append(case)
                else:
                    self.stage2.submit(case)

    def _resolve_case(self, case: Case):
        case.total_time = case.s2_end - case.arrival_time
        granted = self.rng.random() < self.grant_rate
        if granted:
            case.outcome = "appeal_granted" if case.is_appeal else "granted"
            self.completed_cases.append(case)
            return

        # Denied: may appeal (one appeal allowed, matching original scope)
        if not case.is_appeal and self.rng.random() < self.appeal_rate:
            case.outcome = "pending_appeal"
            # Appeal reuses the ORIGINAL case's real profile -- see
            # make_case()'s `template` handling in case.py. This replaces
            # the original bug where an appeal drew a brand-new random case.
            appeal_case = make_case(
                self.rng, self.params, self.env.now,
                case_id=case.case_id, is_appeal=True, template=case,
            )
            self.stage2.submit(appeal_case)
            # `case` itself is NOT added to completed_cases -- it's an
            # unresolved pending appeal, matching original semantics. Only
            # the appeal_case (same case_id) will eventually be appended.
        else:
            case.outcome = "denied"
            self.completed_cases.append(case)

    # ----------------------------------------------------------
    # Arrivals
    # ----------------------------------------------------------

    def _arrival_process(self):
        while True:
            yield self.env.timeout(self.rng.exponential(1.0 / self.lam))
            case = make_case(self.rng, self.params, self.env.now, self._case_counter)
            self._case_counter += 1
            if self.cel_cfg is not None:
                case.econ_need_score = draw_econ_need_score(self.rng, self.params)
            self.stage1.submit(case)

    # ----------------------------------------------------------
    # Run / results
    # ----------------------------------------------------------

    def run(self) -> dict:
        self.env.process(self._arrival_process())
        if self.cel_cfg is not None:
            self.env.process(self._lottery_draw_process())
        self.env.run(until=self.sim_months)
        return self._collect_results()

    def _collect_results(self) -> dict:
        cases = self.completed_cases
        if not cases:
            return {}

        wait_s1 = [c.s1_start - c.s1_queue_enter for c in cases if c.s1_start > 0]
        wait_s2 = [c.s2_start - c.s2_queue_enter for c in cases if c.s2_start > 0]
        total = [c.total_time for c in cases if c.total_time > 0]
        outcomes = [c.outcome for c in cases]
        n = len(cases)

        return {
            "algorithm": self.algorithm,
            "n_completed": n,
            "throughput": n / self.sim_months,
            "backlog_s1": len(self.stage1),
            "backlog_s2": len(self.stage2),
            "backlog_total": len(self.stage1) + len(self.stage2),
            # Cumulative worker-idle time (see stage.py) -- used by
            # diagnostics/verify_cel_stage2_idle.py to test whether CEL's
            # monthly lottery-batch withholding starves Stage 2 workers of
            # available cases relative to FIFO at the same scale.
            "stage1_idle_time": self.stage1.idle_time_total,
            "stage2_idle_time": self.stage2.idle_time_total,
            "mean_wait_s1": float(np.mean(wait_s1)) if wait_s1 else 0,
            "mean_wait_s2": float(np.mean(wait_s2)) if wait_s2 else 0,
            "mean_total_time": float(np.mean(total)) if total else 0,
            "p95_total_time": float(np.percentile(total, 95)) if total else 0,
            "grant_rate_obs": outcomes.count("granted") / n if n else 0,
            "denial_rate_obs": outcomes.count("denied") / n if n else 0,
            "renege_rate_obs": outcomes.count("reneged") / n if n else 0,
            "cases": cases,
        }