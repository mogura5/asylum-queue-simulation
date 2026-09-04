"""
CONDITIONAL ECONOMIC LOTTERY (CEL) ALGORITHM
=============================================
A fourth scheduling algorithm that models a welfare-conditioned, randomized
expedited pathway.

MECHANISM:
  Each case is scored on two dimensions:
    1. complexity_score (0-10, from main simulator) — lower = simpler case
    2. econ_need_score  (0-10, new)                — higher = better labor market match

ECONOMIC NEED SCORE CALIBRATION (sources: BLS JOLTS 2023, DHS OIS 2022):
  Component          Weight  Distribution
  skill_match        0-4     BLS job openings by skill tier
  english_proficiency 0-2    TRAC language data proxy
  work_history       0-2     Inverse of TRAC crim_ind rate
  age_prime          0-2     Prime working age 25-54 share

USAGE:
  python3 cel_algorithm.py
  python3 cel_algorithm.py --era pre_covid_2019 --archetype high_vol_low_grant
  python3 cel_algorithm.py --all-archetypes
"""

import simpy
import numpy as np
import pandas as pd
import json
import time
import argparse
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from simulation_runner import (
    Case, COMPLEXITY_WEIGHTS, COMPLEXITY_SERVICE_MULTIPLIER,
    PATIENCE_BASE_MONTHS, SCALE_FACTOR, SIM_MONTHS, RANDOM_SEED
)

# ── paths ─────────────────────────────────────────────────────────────
PARAMS_PATH = "../data/sim_parameters_FINAL.json"
OUTPUT_DIR  = "../data/cel_results"

# ── economic need score weights (sum to 10) ───────────────────────────
ECON_WEIGHTS = {
    "skill_match":         4.0,   # BLS labor shortage by occupation tier
    "english_proficiency": 2.0,   # language integration proxy
    "work_history":        2.0,   # prior employment (no criminal record proxy)
    "age_prime":           2.0,   # prime working age 25-54
}

# BLS JOLTS 2023: job openings by education/skill requirement
# Tier 0 (score=0):   agriculture/personal services — lowest unfilled openings ratio
# Tier 1 (score=1):   retail/food service — moderate shortage (~25% of openings)
# Tier 2 (score=2):   healthcare support, construction — high shortage (BLS top 10)
# Tier 3 (score=3):   professional/technical/STEM — highest wage, highest shortage
SKILL_TIER_PROBS = [0.20, 0.25, 0.35, 0.20]   # tiers 0-3; sum=1.0

# English proficiency proxy (from TRAC language diversity data)
ENGLISH_PROBS = [0.30, 0.20, 0.50]   # scores 0, 1, 2

# ── lottery configurations ─
#
# complexity_score: mean=1.73, p25=0.0, p50=1.5, p75=3.0, p90=4.0
# econ_need_score:  mean=6.43, p25=5.33, p50=6.67, p75=8.0, p90=8.67
#
# Six configs test different welfare philosophies:
#   strict:       near-perfect cases only (~9% qualify)
#   econ_focused: economically targeted + low complexity (~22% qualify)
#   balanced:     symmetric median cutoffs (~61% qualify)
#   broad:        permissive — tests whether volume expediting helps (~87% qualify)
#   comp_only:    complexity filter only, no econ requirement (~56% qualify)
#   econ_only:    economic filter only, no complexity requirement (~39% qualify)
# The last two are diagnostic: isolate each filter's individual contribution.
LOTTERY_CONFIGS = {
    "strict": {
        "complexity_threshold": 0.0,  "econ_threshold": 8.0,  "p_lottery": 0.10,
        "label": "Strict (comp=0, econ≥8.0, p=10%) — ~9% qualify, ~0.9% expedited"
    },
    "econ_focused": {
        "complexity_threshold": 1.5,  "econ_threshold": 6.67, "p_lottery": 0.20,
        "label": "Econ-focused (comp≤1.5, econ≥6.67, p=20%) — ~22% qualify, ~4.4% expedited"
    },
    "balanced": {
        "complexity_threshold": 3.0,  "econ_threshold": 5.33, "p_lottery": 0.30,
        "label": "Balanced (comp≤3.0, econ≥5.33, p=30%) — ~61% qualify, ~18% expedited"
    },
    "broad": {
        "complexity_threshold": 4.5,  "econ_threshold": 4.0,  "p_lottery": 0.40,
        "label": "Broad (comp≤4.5, econ≥4.0, p=40%) — ~87% qualify, ~35% expedited"
    },
    "comp_only": {
        "complexity_threshold": 1.5,  "econ_threshold": 0.0,  "p_lottery": 0.15,
        "label": "Comp-only (comp≤1.5, no econ filter, p=15%) — ~56% qualify, ~8% expedited"
    },
    "econ_only": {
        "complexity_threshold": 10.0, "econ_threshold": 6.67, "p_lottery": 0.15,
        "label": "Econ-only (no comp filter, econ≥6.67, p=15%) — ~39% qualify, ~6% expedited"
    },
}


# ── econ need score drawing ────────────────────────────────────────────

def draw_econ_need_score(rng: np.random.Generator, params: dict) -> tuple[float, dict]:
    """
    Draw an economic need score from calibrated distributions.

    Returns (econ_need_score, components_dict).

    """
    ew = ECON_WEIGHTS

    # Skill match: sample from BLS job opening tier distribution
    skill_tier = int(rng.choice(4, p=SKILL_TIER_PROBS))
    skill_score = skill_tier

    # English proficiency proxy
    english_score = int(rng.choice(3, p=ENGLISH_PROBS))  # 0, 1, or 2

    # Work history: inverse criminal record
    p_crim = params.get("p_criminal", 0.018)
    work_score = 0 if (rng.random() < p_crim) else 2

    # Prime working age: 25-54
    age_score = 2 if (rng.random() < 0.60) else 0

    econ_score = (
        ew["skill_match"]         * (skill_score / 3.0) +   # normalise 0-3 → 0-4
        ew["english_proficiency"] * (english_score / 2.0) +
        ew["work_history"]        * (work_score / 2.0) +
        ew["age_prime"]           * (age_score / 2.0)
    )

    components = {
        "skill_tier":    skill_tier,
        "skill_score":   round(ew["skill_match"] * skill_score / 3.0, 2),
        "english_score": round(ew["english_proficiency"] * english_score / 2.0, 2),
        "work_score":    round(ew["work_history"] * work_score / 2.0, 2),
        "age_score":     round(ew["age_prime"] * age_score / 2.0, 2),
    }
    return round(econ_score, 2), components


# ── CEL simulator ─────────────────────────────────────────────────────

class CELSimulator:
    """
    Two-stage asylum queue with Conditional Economic Lottery expedited path.

    Architecture:
      Stage 1 (Officer):  All cases pass through. During/after screening,
                          officer assigns complexity_score and econ_need_score.
                          Cases meeting thresholds are placed in lottery_pool.

      Lottery draw:       Each month, p_lottery fraction of the pool is
                          randomly selected and GRANTED immediately (expedited).
                          They skip Stage 2 entirely.

      Stage 2 (Judge):    Non-expedited cases join the normal queue under
                          the specified baseline algorithm (FIFO default).

    Metrics tracked beyond main simulator:
      - expedited_count, expedited_rate
      - mean_econ_score_expedited  (how "valuable" are the expedited cases?)
      - mean_econ_score_regular    (comparison group)
      - econ_welfare               (sum of econ scores for expedited cases)
      - gini_wait                  (inequality of wait times, 0=equal, 1=max unequal)
      - qualify_rate               (fraction meeting both thresholds)
    """

    def __init__(self, params: dict, lottery_config: dict,
                 baseline_algo: str = "FIFO",
                 scale_factor: float = SCALE_FACTOR,
                 sim_months: float = SIM_MONTHS,
                 seed: int = RANDOM_SEED, capacity_multiplier: float = 1.0):

        self.params           = params
        self.cfg              = lottery_config
        self.comp_thresh      = lottery_config["complexity_threshold"]
        self.econ_thresh      = lottery_config["econ_threshold"]
        self.p_lottery        = lottery_config["p_lottery"]
        self.baseline_algo    = baseline_algo
        self.scale_factor     = scale_factor
        self.sim_months       = sim_months
        self.rng              = np.random.default_rng(seed)

        self.lam   = params["arrival_rate_lambda"] * scale_factor
        self.c1 = max(1, round(200 * scale_factor * capacity_multiplier))
        self.c2 = max(1, round(175 * scale_factor * capacity_multiplier))
        mu         = params["service_time_mean"]
        std        = params["service_time_std"] if params["service_time_std"] > 0 else mu * 0.5
        self.s1_mean = mu * 0.20;  self.s1_std = std * 0.20
        self.s2_mean = mu * 0.80;  self.s2_std = std * 0.80

        self.grant_rate  = params["grant_rate"]
        self.appeal_rate = params["appeal_rate"]
        self.p_rep       = params["p_rep"]
        self.p_detained  = params["p_detained"]
        self.p_criminal  = params.get("p_criminal", 0.018)

        self.env      = simpy.Environment()
        self.officers = simpy.Resource(self.env, capacity=self.c1)
        self.judges   = simpy.Resource(self.env, capacity=self.c2)

        # Stage 2 queue uses the specified baseline algorithm
        from simulation_runner import FIFOQueue, LIFOQueue, PriorityQueue
        if baseline_algo == "LIFO":
            self.s2_queue = LIFOQueue()
        elif baseline_algo == "PRIORITY":
            from simulation_runner import AGING_THRESHOLD, AGING_BOOST
            self.s2_queue = PriorityQueue(AGING_THRESHOLD, AGING_BOOST)
        else:
            self.s2_queue = FIFOQueue()

        # Lottery pool: list of (case, econ_score) tuples awaiting monthly draw
        self.lottery_pool: list = []

        # Telemetry
        self.completed_regular: list   = []
        self.expedited_cases: list     = []
        self.queue_log: list           = []
        self._ctr = 0

    def _draw_svc(self, m, s):
        if s <= 0 or m <= 0:
            return max(0.01, self.rng.exponential(m))
        sv = np.log(1 + (s/m)**2)
        ml = np.log(m) - sv/2
        return max(0.01, self.rng.lognormal(ml, np.sqrt(sv)))

    def _make_case(self, arrival_time: float) -> tuple:
        cid = self._ctr; self._ctr += 1
        has_rep     = self.rng.random() < self.p_rep
        detained    = self.rng.random() < self.p_detained
        is_family   = self.rng.random() < self.params.get("p_family", 0.30)
        filed_late  = self.rng.random() < self.params.get("p_late", 0.25)
        has_criminal= self.rng.random() < self.p_criminal
        nbr_charges = int(self.rng.choice([0,1,2,3],
                          p=self.params.get("charge_dist",[0.45,0.35,0.15,0.05])))

        cw   = COMPLEXITY_WEIGHTS
        comp = (cw["no_rep"]*(not has_rep) + cw["has_criminal"]*has_criminal +
                cw["is_family"]*is_family + cw["filed_late"]*filed_late +
                cw["multi_charge"]*(nbr_charges >= 2))

        econ_score, econ_components = draw_econ_need_score(self.rng, self.params)

        priority = 1.0 + (3.0 if detained else 0) + (1.0 if has_rep else 0)

        s1t = self._draw_svc(self.s1_mean, self.s1_std)
        s2t = self._draw_svc(self.s2_mean, self.s2_std)
        eff_s2 = s2t * (1 + COMPLEXITY_SERVICE_MULTIPLIER * comp / 10.0)

        case = Case(
            case_id=cid, arrival_time=arrival_time,
            has_rep=has_rep, is_detained=detained,
            is_family=is_family, filed_late=filed_late,
            has_criminal=has_criminal, nbr_charges=nbr_charges,
            complexity_score=round(comp, 2),
            base_priority=round(priority, 2),
            service_time_s1=s1t, service_time_s2=eff_s2,
        )
        return case, econ_score, econ_components

    def _stage2_process(self, env, case):
        case.s2_queue_enter = env.now
        patience = (float("inf") if case.is_detained
                    else self.rng.exponential(PATIENCE_BASE_MONTHS))
        self.s2_queue.push(case, env.now)
        with self.judges.request() as req:
            result = yield req | env.timeout(patience)
            if req in result:
                served = self.s2_queue.pop(env.now) or case
                served.s2_start = env.now
                yield env.timeout(served.service_time_s2)
                served.s2_end = env.now
                served.total_time = served.s2_end - served.arrival_time
                if self.rng.random() < self.grant_rate:
                    served.outcome = "granted"
                else:
                    if self.rng.random() < self.appeal_rate:
                        served.outcome = "denied_appealed"
                    else:
                        served.outcome = "denied"
                self.completed_regular.append(served)
            else:
                case.outcome = "reneged"
                case.total_time = env.now - case.arrival_time
                self.completed_regular.append(case)

    def _stage1_process(self, env, case, econ_score):
        
        # every case passes through officer screening, gets scored on complexity and economic need
        
        case.s1_queue_enter = env.now
        with self.officers.request() as req:
            yield req
            case.s1_start = env.now
            yield env.timeout(case.service_time_s1)
            case.s1_end = env.now

        # Post-screening: check lottery eligibility
        qualifies = (case.complexity_score <= self.comp_thresh and
                     econ_score >= self.econ_thresh)
        if qualifies:
            self.lottery_pool.append((case, econ_score, env.now))
        else:
            env.process(self._stage2_process(env, case))

    def _arrival_process(self, env):
        while True:
            yield env.timeout(self.rng.exponential(1.0 / self.lam))
            case, econ_score = self._make_case(env.now)
            env.process(self._stage1_process(env, case, econ_score))

    def _lottery_draw_process(self, env):
        """Monthly lottery: draw p_lottery fraction of eligible pool."""
        while True:
            yield env.timeout(1.0)   # draw once per simulation month
            if not self.lottery_pool:
                continue

            n_draw = max(0, int(len(self.lottery_pool) * self.p_lottery))
            if n_draw == 0 and self.lottery_pool and self.rng.random() < self.p_lottery:
                n_draw = 1   # ensure at least one draw if pool is tiny

            if n_draw > 0:
                # Random selection without replacement from pool
                indices = self.rng.choice(len(self.lottery_pool),
                                          size=min(n_draw, len(self.lottery_pool)),
                                          replace=False)
                selected_indices = set(indices)
                for i, item in enumerate(self.lottery_pool):
                    if i in selected_indices:
                        case, econ_score, queue_enter_time = item
                        case.s2_queue_enter = queue_enter_time
                        case.s2_start       = env.now
                        case.s2_end         = env.now   # instant (officer grant)
                        case.total_time     = env.now - case.arrival_time
                        case.outcome        = "lottery_expedited"
                        self.expedited_cases.append((case, econ_score))
                    else:
                        # Not selected this round: send to Stage 2 queue
                        case, econ_score, _ = item
                        env.process(self._stage2_process(env, case))
                        # Remove from pool (they've been dispatched)
                self.lottery_pool = []   # pool resets each month after draw+dispatch

    def _monitor_process(self, env):
        while True:
            self.queue_log.append((env.now, len(self.lottery_pool), len(self.s2_queue)))
            yield env.timeout(1.0)

    def run(self) -> dict:
        self.env.process(self._arrival_process(self.env))
        self.env.process(self._lottery_draw_process(self.env))
        self.env.process(self._monitor_process(self.env))
        self.env.run(until=self.sim_months)
        return self._collect_results()

    def _collect_results(self) -> dict:
        n_exp  = len(self.expedited_cases)
        n_reg  = len(self.completed_regular)
        n_total = n_exp + n_reg

        # Wait times
        exp_waits  = [c.total_time for c, _ in self.expedited_cases if c.total_time > 0]
        reg_waits  = [c.total_time for c in self.completed_regular if c.total_time > 0]
        all_waits  = exp_waits + reg_waits

        # Econ scores
        exp_econ   = [e for _, e in self.expedited_cases]
        exp_comp   = [c.complexity_score for c, _ in self.expedited_cases]
        reg_comp   = [c.complexity_score for c in self.completed_regular]

        # Renege among regular-path cases
        reg_outcomes = [c.outcome for c in self.completed_regular]
        renege_count = reg_outcomes.count("reneged")

        # Gini coefficient of wait times (inequality measure)
        def gini(arr):
            if not arr or len(arr) < 2: return 0.0
            a = np.sort(np.array(arr))
            n = len(a)
            return float((2 * np.sum(np.arange(1, n+1) * a) - (n+1) * np.sum(a))
                         / (n * np.sum(a) + 1e-9))

        # Qualify rate
        total_arrivals = self._ctr
        # (approximate: expedited + pool_remaining + dispatched to s2)
        # We track this via expedited count as a fraction of arrivals
        qualify_rate_approx = n_exp / max(1, total_arrivals)

        # backlog_s2: cases in Stage 2 queue at simulation end
        # backlog_pool: cases qualified but not yet drawn from lottery pool
        # backlog_comparable: total unresolved = arrivals - completed
        backlog_s2         = len(self.s2_queue)
        backlog_pool       = len(self.lottery_pool)
        backlog_comparable = max(0, total_arrivals - n_total)

        return {
            "n_expedited":             n_exp,
            "n_regular":               n_reg,
            "n_total":                 n_total,
            "expedited_rate":          n_exp / max(1, n_total),
            "qualify_rate_approx":     qualify_rate_approx,
            "mean_wait_expedited":     float(np.mean(exp_waits)) if exp_waits else 0,
            "mean_wait_regular":       float(np.mean(reg_waits)) if reg_waits else 0,
            "mean_wait_all":           float(np.mean(all_waits)) if all_waits else 0,
            "renege_rate_regular":     renege_count / max(1, n_reg),
            "mean_econ_score_expedited":  float(np.mean(exp_econ)) if exp_econ else 0,
            "mean_comp_score_expedited":  float(np.mean(exp_comp)) if exp_comp else 0,
            "mean_comp_score_regular":    float(np.mean(reg_comp)) if reg_comp else 0,
            "econ_welfare_total":         float(np.sum(exp_econ)),
            "gini_wait":               gini(all_waits),
            "backlog_s2":              backlog_s2,
            "backlog_pool":            backlog_pool,
            "backlog_total":           backlog_comparable,
            "throughput_total":        n_total / self.sim_months,
            "throughput_expedited":    n_exp / self.sim_months,
        }


# ── experiment runner ──────────────────────────────────────────────────

def run_cel_experiment(era: str, archetype: str,
                       scale_factor: float = SCALE_FACTOR,
                       sim_months: float = SIM_MONTHS,
                       baseline_algo: str = "FIFO") -> pd.DataFrame:

    with open(PARAMS_PATH) as f:
        all_params = json.load(f)

    params = all_params.get(era, {}).get(archetype)
    if params is None:
        raise ValueError(f"Parameters not found for {era}/{archetype}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"\n{'='*65}")
    print(f"CEL EXPERIMENT: {era} / {archetype}")
    print(f"Scale: {scale_factor}  |  Months: {sim_months}  |  Baseline: {baseline_algo}")
    print(f"{'='*65}")

    # First: run baseline (no lottery) for comparison
    from simulation_runner import AsylumSimulator
    print(f"\n  [Baseline] {baseline_algo}...")
    t0 = time.time()
    sim = AsylumSimulator(params, baseline_algo, scale_factor, sim_months, RANDOM_SEED)
    br  = sim.run()
    print(f"    wait={br.get('mean_total_time',0):.1f}mo  "
          f"renege={br.get('renege_rate_obs',0):.3f}  "
          f"backlog={br.get('backlog_total',0):,}  ({time.time()-t0:.1f}s)")
    rows.append({
        "era": era, "archetype": archetype,
        "config": "BASELINE_" + baseline_algo,
        "complexity_threshold": "—", "econ_threshold": "—", "p_lottery": "—",
        "n_expedited": 0, "expedited_rate": 0,
        "mean_wait_expedited": "—",
        "mean_wait_regular": br.get("mean_total_time", 0),
        "mean_wait_all": br.get("mean_total_time", 0),
        "renege_rate_regular": br.get("renege_rate_obs", 0),
        "mean_econ_score_expedited": "—",
        "econ_welfare_total": 0,
        "gini_wait": "—",
        "backlog_total": br.get("backlog_total", 0),
        "throughput_total": br.get("throughput", 0),
    })

    # Then run each lottery configuration
    for cfg_name, cfg in LOTTERY_CONFIGS.items():
        print(f"\n  [{cfg_name}] {cfg['label']}...")
        t0 = time.time()
        cel = CELSimulator(params, cfg, baseline_algo, scale_factor, sim_months, RANDOM_SEED)
        r   = cel.run()
        elapsed = time.time() - t0
        print(f"    expedited={r['n_expedited']:,} ({r['expedited_rate']:.1%})  "
              f"wait_exp={r['mean_wait_expedited']:.1f}mo  "
              f"wait_reg={r['mean_wait_regular']:.1f}mo  "
              f"renege_reg={r['renege_rate_regular']:.3f}  "
              f"econ_welfare={r['econ_welfare_total']:.1f}  "
              f"gini={r['gini_wait']:.3f}  ({elapsed:.1f}s)")
        rows.append({
            "era": era, "archetype": archetype,
            "config": f"CEL_{cfg_name}",
            "complexity_threshold": cfg["complexity_threshold"],
            "econ_threshold": cfg["econ_threshold"],
            "p_lottery": cfg["p_lottery"],
            **r,
            "econ_score_mean": round(r.get("mean_econ_score_expedited", 0), 2),
            "comp_score_exp":  round(r.get("mean_comp_score_expedited", 0), 2),
            "comp_score_reg":  round(r.get("mean_comp_score_regular", 0), 2),
        })

    df = pd.DataFrame(rows)
    df.insert(0, "era", df.pop("era"))
    df.insert(1, "archetype", df.pop("archetype"))

    out_path = Path(OUTPUT_DIR) / f"cel_{era}_{archetype}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")

    # Summary
    print(f"\n{'='*65}")
    print("SUMMARY TABLE")
    print(f"{'='*65}")
    cols = ["config","mean_wait_all","mean_wait_expedited","renege_rate_regular",
            "econ_welfare_total","expedited_rate","gini_wait","backlog_total"]
    print(df[cols].to_string(index=False))

    return df


def run_all_archetypes(scale_factor: float = SCALE_FACTOR,
                       sim_months: float = SIM_MONTHS):
    eras      = ["pre_covid_2019", "post_covid_2024"]
    archetypes = ["high_vol_low_grant","high_vol_high_grant",
                  "low_vol_low_grant","low_vol_high_grant"]
    all_dfs = []
    for era in eras:
        for arch in archetypes:
            df = run_cel_experiment(era, arch, scale_factor, sim_months)
            all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = Path(OUTPUT_DIR) / "cel_all_results.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nAll results saved → {out_path}")
    return combined


# ── main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional Economic Lottery simulation")
    parser.add_argument("--era",       default="pre_covid_2019",
                        choices=["pre_covid_2019","post_covid_2024"])
    parser.add_argument("--archetype", default="high_vol_low_grant",
                        choices=["high_vol_low_grant","high_vol_high_grant",
                                 "low_vol_low_grant","low_vol_high_grant"])
    parser.add_argument("--scale",     type=float, default=SCALE_FACTOR)
    parser.add_argument("--months",    type=float, default=SIM_MONTHS)
    parser.add_argument("--baseline",  default="FIFO",
                        choices=["FIFO","LIFO","PRIORITY"],
                        help="Algorithm for Stage 2 non-expedited cases")
    parser.add_argument("--all-archetypes", action="store_true",
                        help="Run all eras × archetypes")
    args = parser.parse_args()

    if args.all_archetypes:
        run_all_archetypes(args.scale, args.months)
    else:
        run_cel_experiment(args.era, args.archetype, args.scale,
                           args.months, args.baseline)
