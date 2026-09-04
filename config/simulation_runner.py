"""
ASYLUM QUEUE SIMULATOR
======================
Models the U.S. asylum adjudication system as a two-stage tandem queue:
  Stage 1: Asylum Officer screening  (M/M/c1)
  Stage 2: Immigration Judge         (M/M/c2, with feedback loop for appeals)

"""

import simpy
import numpy as np
import pandas as pd
import json
import csv
import heapq
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

SCALE_FACTOR      = 0.30    # Simulate 30% of real volume; rates stay identical
SIM_MONTHS        = 100     # Simulation horizon in months
RANDOM_SEED       = 42
AGING_THRESHOLD   = 6.0     # Months before a waiting case gets a priority bump
AGING_BOOST       = 2.0     # Priority points added per aging interval

# Complexity / impatience model
# Each 1-point increase in complexity_score multiplies Stage 2 service time
# by (1 + COMPLEXITY_SERVICE_MULTIPLIER * score/10).
# At score=10: service time is multiplied by (1 + 0.6) = 1.6x baseline.
COMPLEXITY_SERVICE_MULTIPLIER = 0.60

# Patience model for reneging
# PATIENCE_BASE_MONTHS: mean patience for a fully patient (score=0) case
# PATIENCE_MIN_FRAC:    fraction of base patience for a fully impatient (score=1) case
# Example: base=18mo, min_frac=0.15 → impatient cases have mean patience of 2.7 months
PATIENCE_BASE_MONTHS = 18.0
PATIENCE_MIN_FRAC    = 0.15

# Complexity weight table  (attribute → weight, weights sum to 10)
# These map directly to TRAC data fields:
COMPLEXITY_WEIGHTS = {
    "no_rep":      3.0,   # unrepresented (attorney_flag != 1)
    "has_criminal":2.0,   # crim_ind == Y
    "is_family":   2.0,   # case_type == F
    "filed_late":  1.5,   # waited >1 year before filing (proxy: high nbr_of_charges)
    "multi_charge":1.5,   # nbr_of_charges >= 2
}


MAX_WAIT_MONTHS   = 36.0    # LIFO starvation cap — case gets forced to front

# Stage server counts (pre-scaling). Scale_factor applied at runtime.
# Calibrated from EOIR annual reports: ~800 asylum officers, ~700 IJs nationally.
# We divide by 4 archetypes and apply scale.
BASE_OFFICERS_PER_ARCHETYPE = 200   # Stage 1 servers
BASE_JUDGES_PER_ARCHETYPE   = 175   # Stage 2 servers

ALGORITHMS = ["FIFO", "LIFO", "PRIORITY"]

ERAS = ["pre_covid_2019", "post_covid_2024"]

ARCHETYPES = [
    "high_vol_low_grant",
    "high_vol_high_grant",
    "low_vol_low_grant",
    "low_vol_high_grant",
]


# ============================================================
# CASE DATA CLASS
# ============================================================

@dataclass
class Case:
    """
    Represents one asylum case moving through the two-stage queue.

    Complexity score (0-10): weighted composite of case attributes that
    increase adjudicative difficulty. Each unit adds
    COMPLEXITY_SERVICE_MULTIPLIER to Stage 2 service time. Score 0 = simple
    uncontested single-applicant case; score 10 = family case with criminal
    history, late filing, multiple charges, no representation.

    """
    case_id:           int
    arrival_time:      float   # months since sim start
    has_rep:           bool
    is_detained:       bool
    is_family:         bool    # family/group filing
    filed_late:        bool    # filed >1 year after entry
    has_criminal:      bool    # criminal indicator
    nbr_charges:       int     # number of charges (0, 1, 2+)
    complexity_score:  float   # 0-10 composite
    base_priority:     float
    service_time_s1:   float
    service_time_s2:   float   # base draw; scaled by complexity at service start
    is_appeal:         bool    = False

    # Filled during simulation
    s1_queue_enter:    float = 0.0
    s1_start:          float = 0.0
    s1_end:            float = 0.0
    s2_queue_enter:    float = 0.0
    s2_start:          float = 0.0
    s2_end:            float = 0.0
    outcome:           str   = ""
    total_time:        float = 0.0

    def effective_s2_time(self) -> float:
        """Stage 2 service time scaled by complexity score."""
        return self.service_time_s2 * (
            1.0 + COMPLEXITY_SERVICE_MULTIPLIER * self.complexity_score / 10.0
        )

    def __lt__(self, other):
        return self.base_priority > other.base_priority


# ============================================================
# QUEUE IMPLEMENTATIONS
# ============================================================

class FIFOQueue:
    """Standard first-in, first-out queue using deque."""
    def __init__(self):
        self._q = deque()

    def push(self, case: Case):
        self._q.append(case)

    def pop(self) -> Optional[Case]:
        return self._q.popleft() if self._q else None

    def __len__(self):
        return len(self._q)

    def apply_aging(self):
        pass  # No aging in FIFO


class LIFOQueue:
    """Last-in, first-out stack with starvation cap."""
    def __init__(self, max_wait: float = MAX_WAIT_MONTHS):
        self._stack = []
        self._entry_times = {}
        self._max_wait = max_wait

    def push(self, case: Case, current_time: float):
        self._stack.append(case)
        self._entry_times[case.case_id] = current_time

    def pop(self, current_time: float) -> Optional[Case]:
        if not self._stack:
            return None
        # Check if oldest case has exceeded starvation cap → force it out
        oldest_idx = None
        oldest_wait = -1
        for i, c in enumerate(self._stack):
            wait = current_time - self._entry_times.get(c.case_id, current_time)
            if wait > oldest_wait:
                oldest_wait = wait
                oldest_idx = i
        if oldest_wait >= self._max_wait:
            case = self._stack.pop(oldest_idx)
        else:
            case = self._stack.pop()  # normal LIFO
        self._entry_times.pop(case.case_id, None)
        return case

    def __len__(self):
        return len(self._stack)

    def apply_aging(self):
        pass


class PriorityQueue:
    """
    Non-preemptive max-priority queue using a min-heap (negated priority).
    Aging: cases waiting beyond AGING_THRESHOLD get a periodic priority boost,
    preventing indefinite starvation of low-priority cases.
    """
    def __init__(self, aging_threshold: float = AGING_THRESHOLD,
                 aging_boost: float = AGING_BOOST):
        self._heap = []          # (neg_priority, arrival_time, case)
        self._aging_threshold = aging_threshold
        self._aging_boost = aging_boost
        self._entry_times = {}

    def push(self, case: Case, current_time: float):
        heapq.heappush(self._heap, (-case.base_priority, case.arrival_time, case))
        self._entry_times[case.case_id] = current_time

    def pop(self, current_time: float) -> Optional[Case]:
        while self._heap:
            _, _, case = heapq.heappop(self._heap)
            self._entry_times.pop(case.case_id, None)
            return case
        return None

    def apply_aging(self, current_time: float):
        """Boost priority of any case that has waited too long."""
        new_heap = []
        for arr_time, case in self._heap:
            wait = current_time - self._entry_times.get(case.case_id, current_time)
            if wait >= self._aging_threshold:
                case.base_priority += self._aging_boost
                self._entry_times[case.case_id] = current_time  # reset aging clock
            heapq.heappush(new_heap, (-case.base_priority, arr_time, case))
        self._heap = new_heap

    def __len__(self):
        return len(self._heap)


def make_queue(algorithm: str):
    if algorithm == "FIFO":
        return FIFOQueue()
    elif algorithm == "LIFO":
        return LIFOQueue()
    elif algorithm == "PRIORITY":
        return PriorityQueue()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# ============================================================
# SIMULATOR
# ============================================================

class AsylumSimulator:
    """
    Discrete-event simulation of the two-stage asylum queue.

    Stage 1 (Officer): Cases arrive via Poisson process, join the officer
    queue, get screened. Most defensive cases pass through and enter Stage 2.

    Stage 2 (Judge): Cases wait for an IJ hearing. Outcome drawn from
    Bernoulli(grant_rate). Denied cases may appeal — they re-enter Stage 2
    with a priority bump. Cases that renege exit without a decision.
    """

    def __init__(self, params: dict, algorithm: str, scale_factor: float,
                 sim_months: float, seed: int,
                 capacity_multiplier: float = 1.0):
        self.params      = params
        self.algorithm   = algorithm
        self.scale_factor = scale_factor
        self.sim_months  = sim_months
        self.rng         = np.random.default_rng(seed)

        # Scaled arrival rate and server counts
        self.lam    = params["arrival_rate_lambda"] * scale_factor   # cases/month
        self.c1     = max(1, round(BASE_OFFICERS_PER_ARCHETYPE * scale_factor * capacity_multiplier))
        self.c2     = max(1, round(BASE_JUDGES_PER_ARCHETYPE   * scale_factor * capacity_multiplier))

        # Service time parameters (log-normal fit from mean/std)
        mu_s   = params["service_time_mean"]
        std_s  = params["service_time_std"] if params["service_time_std"] > 0 else mu_s * 0.5
        # Stage 1 ≈ 20% of total processing time (officer screening)
        # Stage 2 ≈ 80% of total processing time (IJ adjudication)
        self.s1_mean = mu_s * 0.20
        self.s2_mean = mu_s * 0.80
        self.s1_std  = std_s * 0.20
        self.s2_std  = std_s * 0.80

        self.params      = params   # store full dict for optional fields in _make_case
        self.grant_rate  = params["grant_rate"]
        self.appeal_rate = params["appeal_rate"]
        self.p_rep       = params["p_rep"]
        self.p_detained  = params["p_detained"]
        self.p_renege    = params["p_renege"]
        self.p_criminal  = params.get("p_criminal", 0.018)  # from TRAC summary stats

        # SimPy environment and shared resources
        self.env      = simpy.Environment()
        self.officers = simpy.Resource(self.env, capacity=self.c1)
        self.judges   = simpy.Resource(self.env, capacity=self.c2)

        # Separate queues for each stage
        self.s1_queue = make_queue(algorithm)
        self.s2_queue = make_queue(algorithm)

        # Telemetry
        self.completed_cases: list[Case] = []
        self.queue_log: list[tuple] = []   # (time, s1_len, s2_len)
        self._case_counter = 0

    # ----------------------------------------------------------
    # Case generation helpers
    # ----------------------------------------------------------

    def _draw_service_time(self, mean: float, std: float) -> float:
        """Log-normal service time draw. Falls back to exponential if std=0."""
        if std <= 0 or mean <= 0:
            return max(0.01, self.rng.exponential(mean))
        sigma2 = np.log(1 + (std / mean) ** 2)
        mu_ln  = np.log(mean) - sigma2 / 2
        return max(0.01, self.rng.lognormal(mu_ln, np.sqrt(sigma2)))

    def _make_case(self, arrival_time: float, is_appeal: bool = False) -> Case:
        """
        Generate one case by Bernoulli draws from calibrated proportions.

        Complexity score: weighted sum of binary attributes drawn from the
        COMPLEXITY_WEIGHTS table. Each attribute is an independent Bernoulli
        draw from its empirical proportion. The score is normalised to [0,10].
        """
        cid = self._case_counter
        self._case_counter += 1

        # ── Case attribute draws ──────────────────────────────────────────
        has_rep     = self.rng.random() < self.p_rep
        detained    = self.rng.random() < self.p_detained
        is_family   = self.rng.random() < self.params.get("p_family",   0.30)
        filed_late  = self.rng.random() < self.params.get("p_late",     0.25)
        has_criminal= self.rng.random() < self.params.get("p_criminal",
                                                          getattr(self,"p_criminal",0.02))
        nbr_charges = int(self.rng.choice([0,1,2,3],
                          p=self.params.get("charge_dist",[0.45,0.35,0.15,0.05])))

        # ── Complexity score (0-10) ───────────────────────────────────────
        cw = COMPLEXITY_WEIGHTS
        comp = (
            cw["no_rep"]       * (not has_rep)   +
            cw["has_criminal"] * has_criminal     +
            cw["is_family"]    * is_family        +
            cw["filed_late"]   * filed_late       +
            cw["multi_charge"] * (nbr_charges >= 2)
        )
        # comp is already in [0,10] because weights sum to 10

        # ── Priority score ────────────────────────────────────────────────
        priority = 1.0
        if detained:
            priority += 3.0
        if has_rep:
            priority += 1.0
        if is_appeal:
            priority += 2.0
        # Complexity penalty: very complex cases slightly deprioritised
        # so they do not crowd out simple cases while awaiting prep.
        # Operator can remove this line to change the welfare tradeoff.
        priority -= (comp / 10.0) * 1.0

        # ── Service times ─────────────────────────────────────────────────
        s1_time = self._draw_service_time(self.s1_mean, self.s1_std)
        s2_time = self._draw_service_time(self.s2_mean, self.s2_std)
        # Note: s2_time is the BASE draw. Actual service time at Stage 2
        # is case.effective_s2_time(), which applies the complexity multiplier.

        return Case(
            case_id=cid,
            arrival_time=arrival_time,
            has_rep=has_rep,
            is_detained=detained,
            is_family=is_family,
            filed_late=filed_late,
            has_criminal=has_criminal,
            nbr_charges=nbr_charges,
            complexity_score=round(comp, 2),
            base_priority=round(priority, 2),
            service_time_s1=s1_time,
            service_time_s2=s2_time,
            is_appeal=is_appeal,
        )

    # ----------------------------------------------------------
    # SimPy processes
    # ----------------------------------------------------------

    def _arrival_process(self):
        """Generate cases via Poisson process (exponential inter-arrivals)."""
        while True:
            inter_arrival = self.rng.exponential(1.0 / self.lam)
            yield self.env.timeout(inter_arrival)
            case = self._make_case(self.env.now)
            self.env.process(self._stage1_process(case))

    def _stage1_process(self, case: Case):
        """Stage 1: Officer screening."""
        case.s1_queue_enter = self.env.now
        self.s1_queue.push(case, self.env.now)

        with self.officers.request() as req:
            yield req
            next_case = self.s1_queue.pop(self.env.now) or case
            next_case.s1_start = self.env.now
            yield self.env.timeout(next_case.service_time_s1)
            next_case.s1_end = self.env.now

        # All screened cases proceed to Stage 2
        self.env.process(self._stage2_process(next_case))

    def _stage2_process(self, case: Case):
        """Stage 2: Immigration Judge adjudication."""
        case.s2_queue_enter = self.env.now

        if case.is_detained:
            patience = float("inf")
        else:
            patience = self.rng.exponential(PATIENCE_BASE_MONTHS)

        self.s2_queue.push(case, self.env.now)

        with self.judges.request() as req:
            # Yield for either service or reneging
            result = yield req | self.env.timeout(patience)

            if req in result:
                # Served
                served = self.s2_queue.pop(self.env.now) or case
                served.s2_start = self.env.now
                # Use complexity-scaled service time at Stage 2
                yield self.env.timeout(served.effective_s2_time())
                served.s2_end = self.env.now
                served.total_time = served.s2_end - served.arrival_time

                # Outcome
                granted = self.rng.random() < self.grant_rate
                if granted:
                    served.outcome = "appeal_granted" if served.is_appeal else "granted"
                else:
                    # Denied: may appeal
                    will_appeal = self.rng.random() < self.appeal_rate
                    if will_appeal and not served.is_appeal:  # one appeal allowed
                        served.outcome = "pending_appeal"
                        appeal_case = self._make_case(self.env.now, is_appeal=True)
                        appeal_case.case_id = served.case_id  # same case ID
                        self.env.process(self._stage2_process(appeal_case))
                    else:
                        served.outcome = "denied"

                if served.outcome != "pending_appeal":
                    self.completed_cases.append(served)
            else:
                # Reneged
                case.outcome = "reneged"
                case.total_time = self.env.now - case.arrival_time
                self.completed_cases.append(case)

    def _queue_monitor(self):
        """Log queue lengths at regular intervals."""
        while True:
            self.queue_log.append((
                self.env.now,
                len(self.s1_queue),
                len(self.s2_queue),
            ))
            yield self.env.timeout(1.0)   # log every 1 month

    def _aging_process(self):
        """Apply aging to priority queues periodically."""
        while True:
            yield self.env.timeout(1.0)
            self.s1_queue.apply_aging(self.env.now)
            self.s2_queue.apply_aging(self.env.now)

    # ----------------------------------------------------------
    # Run
    # ----------------------------------------------------------

    def run(self) -> dict:
        self.env.process(self._arrival_process())
        self.env.process(self._queue_monitor())
        self.env.process(self._aging_process())
        self.env.run(until=self.sim_months)
        return self._collect_results()

    def _collect_results(self) -> dict:
        cases = self.completed_cases
        if not cases:
            return {}

        wait_s1 = [c.s1_start - c.s1_queue_enter for c in cases if c.s1_start > 0]
        wait_s2 = [c.s2_start - c.s2_queue_enter for c in cases if c.s2_start > 0]
        total   = [c.total_time for c in cases if c.total_time > 0]

        outcomes = [c.outcome for c in cases]
        n = len(cases)

        backlog_s1 = len(self.s1_queue)
        backlog_s2 = len(self.s2_queue)

        # Queue length over time
        ql_df = pd.DataFrame(self.queue_log, columns=["time", "s1_len", "s2_len"])

        return {
            "algorithm":         self.algorithm,
            "n_completed":       n,
            "throughput":        n / self.sim_months,
            "backlog_s1":        backlog_s1,
            "backlog_s2":        backlog_s2,
            "backlog_total":     backlog_s1 + backlog_s2,
            "mean_wait_s1":      float(np.mean(wait_s1)) if wait_s1 else 0,
            "mean_wait_s2":      float(np.mean(wait_s2)) if wait_s2 else 0,
            "mean_total_time":   float(np.mean(total)) if total else 0,
            "p95_total_time":    float(np.percentile(total, 95)) if total else 0,
            "grant_rate_obs":    outcomes.count("granted") / n if n else 0,
            "denial_rate_obs":   outcomes.count("denied") / n if n else 0,
            "renege_rate_obs":   outcomes.count("reneged") / n if n else 0,
            "appeal_rate_obs":   (outcomes.count("appeal_granted") +
                                  [c.is_appeal for c in cases].count(True)) / n if n else 0,
            "avg_queue_s1":      ql_df["s1_len"].mean() if not ql_df.empty else 0,
            "avg_queue_s2":      ql_df["s2_len"].mean() if not ql_df.empty else 0,
            "queue_log":         ql_df,
            "cases":             cases,
        }


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_all_experiments(params_path: str, output_dir: str,
                        scale_factor: float = SCALE_FACTOR,
                        sim_months: float = SIM_MONTHS):
    """
    Run all algorithm × era × archetype combinations and write results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(params_path) as f:
        all_params = json.load(f)

    all_results = []
    summary_rows = []

    total_runs = len(ALGORITHMS) * len(ERAS) * len(ARCHETYPES)
    run_num = 0

    for era in ERAS:
        era_params = all_params.get(era, {})
        for archetype in ARCHETYPES:
            arch_params = era_params.get(archetype)
            if arch_params is None:
                print(f"  [SKIP] {era} / {archetype} — not in params file")
                continue

            # Skip low-confidence parameter sets
            if arch_params.get("_low_confidence", False):
                print(f"  [WARN] {era} / {archetype} — low confidence, running anyway")

            for algo in ALGORITHMS:
                run_num += 1
                label = f"{era}/{archetype}/{algo}"
                print(f"  [{run_num}/{total_runs}] Running {label}...")
                t0 = time.time()

                sim = AsylumSimulator(
                    params=arch_params,
                    algorithm=algo,
                    scale_factor=scale_factor,
                    sim_months=sim_months,
                    seed=RANDOM_SEED,
                )
                result = sim.run()

                elapsed = time.time() - t0
                print(f"    → {result.get('n_completed', 0):,} cases completed "
                      f"| backlog={result.get('backlog_total', 0):,} "
                      f"| mean_wait={result.get('mean_total_time', 0):.1f}mo "
                      f"| {elapsed:.1f}s")

                # Save queue log CSV
                ql = result.pop("queue_log", pd.DataFrame())
                cases = result.pop("cases", [])
                result.update({"era": era, "archetype": archetype})

                ql_path = output_dir / f"queue_log_{era}_{archetype}_{algo}.csv"
                ql.to_csv(ql_path, index=False)

                # Save per-case CSV
                if cases:
                    case_rows = [{
                        "case_id":       c.case_id,
                        "era":           era,
                        "archetype":     archetype,
                        "algorithm":     algo,
                        "arrival_time":  c.arrival_time,
                        "has_rep":       c.has_rep,
                        "is_detained":   c.is_detained,
                        "is_appeal":     c.is_appeal,
                        "complexity_score": c.complexity_score,
                        "is_family":        c.is_family,
                        "filed_late":       c.filed_late,
                        "has_criminal":     c.has_criminal,
                        "nbr_charges":      c.nbr_charges,
                        "s1_wait":          max(0, c.s1_start - c.s1_queue_enter),
                        "s2_wait":          max(0, c.s2_start - c.s2_queue_enter),
                        "total_time":       c.total_time,
                        "outcome":          c.outcome,
                    } for c in cases]
                    cases_path = output_dir / f"cases_{era}_{archetype}_{algo}.csv"
                    pd.DataFrame(case_rows).to_csv(cases_path, index=False)

                summary_rows.append(result)
                all_results.append(result)

    # ---- Summary table ----
    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if not isinstance(v, (pd.DataFrame, list))}
        for r in summary_rows
    ])
    summary_path = output_dir / "simulation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print(f"SIMULATION COMPLETE — {run_num} runs")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
    _print_comparison_table(summary_df)

    return summary_df


def _print_comparison_table(df: pd.DataFrame):
    """Print a readable algorithm comparison across eras and archetypes."""
    print("\n--- ALGORITHM COMPARISON: Mean Total Wait Time (months) ---")
    if df.empty:
        print("  No results.")
        return

    metrics = ["mean_total_time", "backlog_total", "throughput", "renege_rate_obs"]
    for era in df["era"].unique():
        print(f"\n  ERA: {era}")
        era_df = df[df["era"] == era]
        for archetype in era_df["archetype"].unique():
            print(f"    [{archetype}]")
            arch_df = era_df[era_df["archetype"] == archetype]
            for _, row in arch_df.iterrows():
                print(
                    f"      {row['algorithm']:8s} | "
                    f"wait={row['mean_total_time']:6.1f}mo | "
                    f"backlog={row['backlog_total']:6,.0f} | "
                    f"throughput={row['throughput']:6.1f}/mo | "
                    f"renege={row['renege_rate_obs']:.3f}"
                )


# ============================================================
# STATISTICAL COMPARISON
# ============================================================

def run_pairwise_ttests(output_dir: str):
    """
    Load per-case CSVs and run pairwise t-tests on total_time between algorithms.
    Reports significance of wait time differences (RQ2).
    """
    from scipy import stats

    output_dir = Path(output_dir)
    results = []

    for era in ERAS:
        for archetype in ARCHETYPES:
            data = {}
            for algo in ALGORITHMS:
                path = output_dir / f"cases_{era}_{archetype}_{algo}.csv"
                if path.exists():
                    df = pd.read_csv(path)
                    data[algo] = df["total_time"].dropna().values

            if len(data) < 2:
                continue

            algos = list(data.keys())
            for i in range(len(algos)):
                for j in range(i + 1, len(algos)):
                    a, b = algos[i], algos[j]
                    if len(data[a]) < 2 or len(data[b]) < 2:
                        continue
                    t_stat, p_val = stats.ttest_ind(data[a], data[b], equal_var=False)
                    mean_diff = np.mean(data[a]) - np.mean(data[b])
                    results.append({
                        "era": era,
                        "archetype": archetype,
                        "algo_A": a,
                        "algo_B": b,
                        "mean_A": round(np.mean(data[a]), 2),
                        "mean_B": round(np.mean(data[b]), 2),
                        "mean_diff_A_minus_B": round(mean_diff, 2),
                        "t_stat": round(t_stat, 3),
                        "p_value": round(p_val, 4),
                        "significant_p05": p_val < 0.05,
                    })

    ttest_df = pd.DataFrame(results)
    ttest_path = output_dir / "pairwise_ttests.csv"
    ttest_df.to_csv(ttest_path, index=False)

    print("\n--- PAIRWISE T-TEST RESULTS (RQ2) ---")
    print(ttest_df[["era", "archetype", "algo_A", "algo_B",
                     "mean_diff_A_minus_B", "p_value", "significant_p05"]].to_string(index=False))
    return ttest_df


# ============================================================
# ECONOMIC IMPACT ANALYSIS
# ============================================================

def compute_economic_impact(summary_df: pd.DataFrame, output_dir: str):
    """
    RQ4 & RQ5: Estimate lost worker compensation and federal revenue loss
    attributable to backlog wait time under each algorithm.

    Key assumptions (calibrated to DHS rule estimates):
      - Asylum seekers cannot work legally during the 180-day EAD waiting period
        and then during any additional backlog wait before a decision.
      - Avg annual compensation (if authorized): $35,000 (DHS baseline)
      - Federal income tax rate: 15% effective (DHS estimate)
      - Monthly lost wages = avg_annual_comp / 12 * (backlog_months - 6)
        where 6 months = EAD waiting period (already built into policy)
    """
    output_dir = Path(output_dir)

    # DHS rule baseline values
    AVG_ANNUAL_COMP  = 35_000       # USD, per DHS 2026 rule
    FED_TAX_RATE     = 0.15
    MONTHS_IN_YEAR   = 12

    monthly_comp = AVG_ANNUAL_COMP / MONTHS_IN_YEAR

    # Scale back up from simulation scale to real volume
    # We use archetype-level throughput × scale_factor⁻¹ to estimate real case count
    rows = []
    for _, r in summary_df.iterrows():
        # Backlog months = mean total wait - 6 month EAD grace period
        billable_wait = max(0, r["mean_total_time"] - 6.0)
        real_backlog  = r["backlog_total"] / SCALE_FACTOR 

        lost_wages_per_case = monthly_comp * billable_wait
        lost_tax_per_case   = lost_wages_per_case * FED_TAX_RATE

        total_lost_wages = lost_wages_per_case * real_backlog
        total_lost_tax   = lost_tax_per_case   * real_backlog

        rows.append({
            "era":                    r["era"],
            "archetype":              r["archetype"],
            "algorithm":              r["algorithm"],
            "mean_wait_months":       round(r["mean_total_time"], 2),
            "billable_wait_months":   round(billable_wait, 2),
            "real_backlog_cases":     round(real_backlog),
            "lost_wages_per_case_usd":round(lost_wages_per_case, 2),
            "total_lost_wages_usd":   round(total_lost_wages, 2),
            "total_lost_tax_usd":     round(total_lost_tax, 2),
        })

    econ_df = pd.DataFrame(rows)
    econ_path = output_dir / "economic_impact.csv"
    econ_df.to_csv(econ_path, index=False)

    print("\n--- ECONOMIC IMPACT SUMMARY ---")
    print(econ_df[["era", "archetype", "algorithm",
                   "mean_wait_months", "total_lost_wages_usd",
                   "total_lost_tax_usd"]].to_string(index=False))
    return econ_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Asylum Queue Simulator")
    parser.add_argument("--params",    default="../../data/sim_parameters_FINAL.json",
                        help="Path to sim_parameters_FINAL.json")
    parser.add_argument("--output",    default="../../data/sim_results/",
                        help="Output directory for results")
    parser.add_argument("--scale",     type=float, default=SCALE_FACTOR,
                        help="Scale factor (default 0.01 = 1%% of real volume)")
    parser.add_argument("--months",    type=float, default=SIM_MONTHS,
                        help="Simulation horizon in months (default 100)")
    parser.add_argument("--no-ttest",  action="store_true",
                        help="Skip pairwise t-tests")
    parser.add_argument("--no-econ",   action="store_true",
                        help="Skip economic impact analysis")
    args = parser.parse_args()

    print(f"\nASYLUM QUEUE SIMULATOR")
    print(f"Scale factor:  {args.scale} ({args.scale*100:.1f}% of real volume)")
    print(f"Sim horizon:   {args.months} months")
    print(f"Algorithms:    {ALGORITHMS}")
    print(f"Eras:          {ERAS}")
    print(f"Archetypes:    {ARCHETYPES}\n")

    summary = run_all_experiments(
        params_path=args.params,
        output_dir=args.output,
        scale_factor=args.scale,
        sim_months=args.months,
    )

    if not args.no_ttest:
        try:
            run_pairwise_ttests(args.output)
        except ImportError:
            print("scipy not installed — skipping t-tests. pip install scipy")

    if not args.no_econ:
        compute_economic_impact(summary, args.output)
