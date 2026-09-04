"""
FIFO vs LIFO COMPLEXITY WEIGHT EXPERIMENT
==========================================
Guideline points 1-3:
  1. Simulate and explore the difference between LIFO and FIFO
     using different complexity score weight configurations.
     New cases each month are drawn from Poisson(λ) so arrival
     counts vary stochastically — some months are busier than others.
  2. Produce results for the best FIFO and best LIFO configurations
     on two welfare objectives: minimize dropout (reneging) and
     minimize true_unresolved (backlog + reneged cases).
  3. Systematic reasoning for which weight configuration wins and why.

WHAT THIS VARIES:
  The weights assigned to the 5 complexity attributes. Each config
  assigns different importance to: criminal record, detention status,
  representation, family case, late filing, multiple charges.
  All weights sum to 10 so scores remain on the same 0-10 scale.

POISSON ARRIVALS:
  Instead of a fixed deterministic inter-arrival time, each simulation
  month draws its case count from Poisson(λ) where λ is the calibrated
  monthly arrival rate from sim_parameters_FINAL.json. This captures
  real-world variability — some months see twice the average filings.

SURGE TEST:
  For Group 1 (high_vol_low_grant), the simulation runs with a mid-run
  surge: for months 100-150, λ is multiplied by a surge_factor drawn
  from the ratio of post-COVID to pre-COVID arrival rates. This tests
  how quickly each algorithm degrades when a real surge hits.

OUTPUTS: ../../data/fifo_lifo_experiment/
  fifo_lifo_results.csv     — all runs (complexity configs × algos × archetypes × eras)
  fifo_lifo_summary.csv     — best config per welfare objective per archetype
  surge_results.csv         — surge test results
  queue_log_surge_*.csv     — queue length over time during surge

RUN:
  python3 fifo_lifo_experiment.py
  python3 fifo_lifo_experiment.py --era pre_covid_2019 --archetype high_vol_low_grant
  python3 fifo_lifo_experiment.py --surge-only
"""

import simpy
import numpy as np
import pandas as pd
import json
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from simulation_runner import (
    Case, FIFOQueue, LIFOQueue,
    COMPLEXITY_SERVICE_MULTIPLIER, PATIENCE_BASE_MONTHS,
    SCALE_FACTOR, SIM_MONTHS, RANDOM_SEED, MAX_WAIT_MONTHS
)

# ── paths ─────────────────────────────────────────────────────────────
PARAMS_PATH = "../data/sim_parameters_FINAL.json"
OUTPUT_DIR  = Path("../data/fifo_lifo_experiment")

ERAS       = ["pre_covid_2019", "post_covid_2024"]
ARCHETYPES = ["high_vol_low_grant","high_vol_high_grant",
              "low_vol_low_grant","low_vol_high_grant"]

# ── complexity weight configurations ──────────────────────────────────
# All weights sum to 10 so complexity_score range [0,10] is preserved.
# Keys map to: no_rep, has_criminal, is_family, filed_late, multi_charge
COMPLEXITY_CONFIGS = {
    "calibrated": {
        # Current baseline — from TRAC data proportions
        "no_rep": 3.0, "has_criminal": 2.0, "is_family": 2.0,
        "filed_late": 1.5, "multi_charge": 1.5,
        "description": "Baseline (TRAC-calibrated weights)"
    },
    "criminal_heavy": {
        # Criminal record and detention are the primary drivers
        "no_rep": 1.5, "has_criminal": 4.0, "is_family": 1.5,
        "filed_late": 1.5, "multi_charge": 1.5,
        "description": "Criminal/charges dominate complexity"
    },
    "rep_heavy": {
        # Lack of representation is the dominant complexity driver
        "no_rep": 5.0, "has_criminal": 1.5, "is_family": 1.5,
        "filed_late": 1.0, "multi_charge": 1.0,
        "description": "Representation gap dominates complexity"
    },
    "family_heavy": {
        # Family cases and late filings are most complex
        "no_rep": 1.5, "has_criminal": 1.5, "is_family": 4.0,
        "filed_late": 2.0, "multi_charge": 1.0,
        "description": "Family caseload dominates complexity"
    },
    "uniform": {
        # Equal weight to all attributes
        "no_rep": 2.0, "has_criminal": 2.0, "is_family": 2.0,
        "filed_late": 2.0, "multi_charge": 2.0,
        "description": "Uniform weights — complexity minimally differentiated"
    },
    "minimal": {
        # Complexity barely matters — near-uniform service times
        "no_rep": 1.0, "has_criminal": 1.0, "is_family": 4.0,
        "filed_late": 2.0, "multi_charge": 2.0,
        "description": "Minimal complexity variation (family only)"
    },
}


# ── Poisson arrival simulator ─────────────────────────────────────────

class FIFOLIFOSimulator:
    """
    Two-stage tandem queue simulator for FIFO/LIFO comparison.

    KEY DIFFERENCE from main simulator:
    Arrivals use Poisson sampling per month — at each 1-month tick,
    the number of new cases is drawn from Poisson(λ) rather than
    using individual Exponential(1/λ) inter-arrival times.
    This better captures real-world monthly filing variability.

    Complexity weights are configurable per run, enabling the
    sensitivity analysis across weight configs.
    """

    def __init__(self, params: dict, algorithm: str,
                 complexity_weights: dict,
                 scale_factor: float = SCALE_FACTOR,
                 sim_months: float = SIM_MONTHS,
                 seed: int = RANDOM_SEED,
                 surge_config: dict = None):

        self.params             = params
        self.algorithm          = algorithm
        self.complexity_weights = complexity_weights
        self.scale_factor       = scale_factor
        self.sim_months         = sim_months
        self.rng                = np.random.default_rng(seed)
        self.surge_config       = surge_config   # None or {start, end, factor}

        self.lam   = params["arrival_rate_lambda"] * scale_factor
        self.c1    = max(1, round(200 * scale_factor))
        self.c2    = max(1, round(175 * scale_factor))
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

        if algorithm == "LIFO":
            self.s1_queue = LIFOQueue(MAX_WAIT_MONTHS)
            self.s2_queue = LIFOQueue(MAX_WAIT_MONTHS)
        else:  # FIFO
            self.s1_queue = FIFOQueue()
            self.s2_queue = FIFOQueue()

        self.completed = []
        self.queue_log = []   # (time, s1_len, s2_len, arrivals_this_month)
        self._ctr = 0

    def _draw_svc(self, m, s):
        if s <= 0 or m <= 0:
            return max(0.01, self.rng.exponential(m))
        sv = np.log(1 + (s/m)**2)
        ml = np.log(m) - sv/2
        return max(0.01, self.rng.lognormal(ml, np.sqrt(sv)))

    def _compute_complexity(self, has_rep, has_criminal, is_family, filed_late, nbr_charges):
        cw = self.complexity_weights
        return (
            cw["no_rep"]       * (not has_rep)    +
            cw["has_criminal"] * has_criminal      +
            cw["is_family"]    * is_family         +
            cw["filed_late"]   * filed_late        +
            cw["multi_charge"] * (nbr_charges >= 2)
        )

    def _make_case(self, arrival_time: float) -> Case:
        cid = self._ctr; self._ctr += 1
        has_rep     = self.rng.random() < self.p_rep
        detained    = self.rng.random() < self.p_detained
        is_family   = self.rng.random() < self.params.get("p_family", 0.30)
        filed_late  = self.rng.random() < self.params.get("p_late", 0.25)
        has_criminal= self.rng.random() < self.p_criminal
        nbr_charges = int(self.rng.choice([0,1,2,3],
                          p=self.params.get("charge_dist",[0.45,0.35,0.15,0.05])))

        comp = self._compute_complexity(has_rep, has_criminal, is_family, filed_late, nbr_charges)
        priority = 1.0 + (3.0 if detained else 0) + (1.0 if has_rep else 0)

        s1t  = self._draw_svc(self.s1_mean, self.s1_std)
        s2t  = self._draw_svc(self.s2_mean, self.s2_std)
        eff_s2 = s2t * (1 + COMPLEXITY_SERVICE_MULTIPLIER * comp / 10.0)

        return Case(
            case_id=cid, arrival_time=arrival_time,
            has_rep=has_rep, is_detained=detained,
            is_family=is_family, filed_late=filed_late,
            has_criminal=has_criminal, nbr_charges=nbr_charges,
            complexity_score=round(comp, 2),
            base_priority=round(priority, 2),
            service_time_s1=s1t, service_time_s2=eff_s2,
        )

    def _stage2(self, env, case):
        case.s2_queue_enter = env.now
        patience = float("inf") if case.is_detained else self.rng.exponential(PATIENCE_BASE_MONTHS)
        self.s2_queue.push(case, env.now)
        with self.judges.request() as req:
            result = yield req | env.timeout(patience)
            if req in result:
                served = self.s2_queue.pop(env.now) or case
                served.s2_start = env.now
                yield env.timeout(served.service_time_s2)
                served.s2_end = env.now
                served.total_time = served.s2_end - served.arrival_time
                served.outcome = "granted" if self.rng.random() < self.grant_rate else "denied"
                self.completed.append(served)
            else:
                case.outcome = "reneged"
                case.total_time = env.now - case.arrival_time
                self.completed.append(case)

    def _stage1(self, env, case):
        case.s1_queue_enter = env.now
        self.s1_queue.push(case, env.now)
        with self.officers.request() as req:
            yield req
            nc = self.s1_queue.pop(env.now) or case
            nc.s1_start = env.now
            yield env.timeout(nc.service_time_s1)
            nc.s1_end = env.now
        env.process(self._stage2(env, nc))

    def _monthly_arrivals(self, env):
        """
        Poisson arrival process: each month, draw n ~ Poisson(λ_month)
        and spawn that many cases at uniformly random times within the month.
        λ_month can change during a surge period.
        """
        month = 0
        while True:
            # Effective λ this month (surge multiplier if active)
            lam_month = self.lam
            if self.surge_config:
                s, e, f = (self.surge_config["start"],
                           self.surge_config["end"],
                           self.surge_config["factor"])
                if s <= month < e:
                    lam_month = self.lam * f

            # Poisson draw for number of cases this month
            n_this_month = self.rng.poisson(lam_month)

            # Spread arrivals uniformly within the month
            arrival_offsets = sorted(self.rng.uniform(0, 1.0, n_this_month))
            for offset in arrival_offsets:
                t = env.now + offset
                case = self._make_case(t)
                env.process(self._stage1(env, case))

            # Snapshot: count completions and reneges since last tick
            n_completed_so_far = len(self.completed)
            n_reneged_so_far   = sum(1 for c in self.completed if c.outcome == "reneged")
            recent_waits       = [c.total_time for c in self.completed
                                  if c.total_time > 0]
            mean_wait_so_far   = float(np.mean(recent_waits)) if recent_waits else 0.0

            self.queue_log.append((env.now, len(self.s1_queue),
                                   len(self.s2_queue), n_this_month,
                                   n_completed_so_far, n_reneged_so_far,
                                   mean_wait_so_far))
            yield env.timeout(1.0)
            month += 1

    def run(self) -> dict:
        self.env.process(self._monthly_arrivals(self.env))
        self.env.run(until=self.sim_months)
        return self._collect()

    def _collect(self) -> dict:
        n = len(self.completed)
        if n == 0:
            return {}
        outcomes  = [c.outcome for c in self.completed]
        total_t   = [c.total_time for c in self.completed if c.total_time > 0]
        n_reneged = outcomes.count("reneged")
        n_decided = n - n_reneged
        backlog_s1 = len(self.s1_queue)
        backlog_s2 = len(self.s2_queue)
        # true_unresolved: in queue + reneged (no formal decision received)
        true_unresolved = backlog_s1 + backlog_s2 + n_reneged

        # Complexity-stratified wait times
        comp_quartiles = {}
        comp_scores = np.array([c.complexity_score for c in self.completed])
        wait_times  = np.array([c.total_time for c in self.completed])
        if len(comp_scores) > 10:
            q25, q50, q75 = np.percentile(comp_scores, [25, 50, 75])
            for label, mask in [
                ("q1_simple",  comp_scores <= q25),
                ("q2_low",    (comp_scores > q25) & (comp_scores <= q50)),
                ("q3_medium", (comp_scores > q50) & (comp_scores <= q75)),
                ("q4_complex", comp_scores > q75),
            ]:
                comp_quartiles[f"wait_{label}"] = (
                    float(np.mean(wait_times[mask])) if mask.sum() > 0 else 0
                )

        # Monthly arrival stats from queue_log
        arrivals_per_month = [r[3] for r in self.queue_log]

        return {
            "n_completed":      n,
            "n_decided":        n_decided,
            "n_reneged":        n_reneged,
            "mean_total_time":  float(np.mean(total_t)) if total_t else 0,
            "p95_total_time":   float(np.percentile(total_t, 95)) if total_t else 0,
            "renege_rate":      n_reneged / n,
            "decided_throughput": n_decided / self.sim_months,
            "backlog_s1":       backlog_s1,
            "backlog_s2":       backlog_s2,
            "backlog_total":    backlog_s1 + backlog_s2,
            "true_unresolved":  true_unresolved,
            "grant_rate":       outcomes.count("granted") / n,
            "mean_arrivals_per_month":  float(np.mean(arrivals_per_month)),
            "std_arrivals_per_month":   float(np.std(arrivals_per_month)),
            **comp_quartiles,
        }


# ── experiment runner ─────────────────────────────────────────────────

def run_fifo_lifo_experiment(
    era: str = None, archetype: str = None,
    scale_factor: float = SCALE_FACTOR,
    sim_months: float = SIM_MONTHS,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:

    with open(PARAMS_PATH) as f:
        all_params = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    eras_to_run       = [era] if era else ERAS
    archetypes_to_run = [archetype] if archetype else ARCHETYPES
    rows = []
    total = len(COMPLEXITY_CONFIGS) * 2 * len(eras_to_run) * len(archetypes_to_run)
    run_n = 0

    for era_key in eras_to_run:
        for arch in archetypes_to_run:
            params = all_params.get(era_key, {}).get(arch)
            if params is None:
                print(f"  [SKIP] {era_key}/{arch}")
                continue

            for cfg_name, cfg in COMPLEXITY_CONFIGS.items():
                cw = {k: cfg[k] for k in
                      ["no_rep","has_criminal","is_family","filed_late","multi_charge"]}

                for algo in ["FIFO", "LIFO"]:
                    run_n += 1
                    label = f"{era_key[:3]}/{arch[:8]}/{algo}/{cfg_name[:10]}"
                    print(f"  [{run_n}/{total}] {label}...", end=" ", flush=True)
                    t0 = time.time()

                    sim = FIFOLIFOSimulator(params, algo, cw, scale_factor, sim_months, RANDOM_SEED)
                    r   = sim.run()
                    elapsed = time.time() - t0

                    if r:
                        print(f"wait={r['mean_total_time']:.1f}mo "
                              f"renege={r['renege_rate']:.3f} "
                              f"unresolved={r['true_unresolved']:,} ({elapsed:.1f}s)")
                        rows.append({
                            "era": era_key, "archetype": arch,
                            "algorithm": algo, "complexity_config": cfg_name,
                            "config_description": cfg["description"],
                            **{k: v for k, v in r.items()},
                        })
                    else:
                        print("no results")

    df = pd.DataFrame(rows)
    out = output_dir / "fifo_lifo_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows → {out}")

    # Summary: best config per welfare objective per archetype-era-algo
    _print_summary(df, output_dir)
    return df


def _print_summary(df: pd.DataFrame, output_dir: Path):
    """Find the best complexity config for each welfare objective."""
    if df.empty:
        return

    print("\n" + "="*65)
    print("BEST COMPLEXITY CONFIG PER WELFARE OBJECTIVE")
    print("="*65)

    objectives = {
        "Minimize mean wait":       ("mean_total_time",  False),
        "Minimize dropout (renege)":("renege_rate",       False),
        "Minimize true_unresolved": ("true_unresolved",   False),
        "Maximize decided output":  ("decided_throughput",True),
    }

    summary_rows = []
    for era in df["era"].unique():
        for arch in df["archetype"].unique():
            for algo in ["FIFO","LIFO"]:
                sub = df[(df["era"]==era)&(df["archetype"]==arch)&(df["algorithm"]==algo)]
                if sub.empty:
                    continue
                for obj_name, (col, maximize) in objectives.items():
                    if col not in sub.columns:
                        continue
                    best = sub.loc[sub[col].idxmax() if maximize else sub[col].idxmin()]
                    summary_rows.append({
                        "era": era, "archetype": arch, "algorithm": algo,
                        "welfare_objective": obj_name,
                        "best_config": best["complexity_config"],
                        "mean_total_time": best["mean_total_time"],
                        "renege_rate": best["renege_rate"],
                        "true_unresolved": best["true_unresolved"],
                        "decided_throughput": best["decided_throughput"],
                    })

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(output_dir / "fifo_lifo_summary.csv", index=False)
    print(sdf[["era","archetype","algorithm","welfare_objective","best_config","mean_total_time","renege_rate"]].to_string(index=False))


def run_surge_test(
    scale_factor: float = SCALE_FACTOR,
    sim_months: float = SIM_MONTHS,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Surge test for Group 1 (HV-Low):
    Run pre-COVID λ for months 0-99, then surge to post-COVID λ
    for months 100-149, then back to pre-COVID for 150-200.
    This uses the actual λ values from sim_parameters_FINAL.json
    (not a synthetic multiplier) so the surge reflects real data.
    """
    with open(PARAMS_PATH) as f:
        all_params = json.load(f)

    pre_params  = all_params["pre_covid_2019"]["high_vol_low_grant"]
    post_params = all_params["post_covid_2024"]["high_vol_low_grant"]
    surge_factor = post_params["arrival_rate_lambda"] / pre_params["arrival_rate_lambda"]

    print(f"\n=== SURGE TEST ===")
    print(f"Base λ (pre-COVID):  {pre_params['arrival_rate_lambda']:.1f}/mo")
    print(f"Surge λ (post-COVID):{post_params['arrival_rate_lambda']:.1f}/mo")
    print(f"Surge factor:        {surge_factor:.2f}×")
    print(f"Surge period:        months 100-149\n")

    surge_cfg = {"start": 100, "end": 150, "factor": surge_factor}
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for algo in ["FIFO","LIFO"]:
        for cfg_name in ["calibrated","criminal_heavy","rep_heavy"]:
            cfg = COMPLEXITY_CONFIGS[cfg_name]
            cw  = {k: cfg[k] for k in ["no_rep","has_criminal","is_family","filed_late","multi_charge"]}
            print(f"  Surge {algo}/{cfg_name}...", end=" ", flush=True)
            t0 = time.time()
            sim = FIFOLIFOSimulator(pre_params, algo, cw, scale_factor,
                                    sim_months, RANDOM_SEED, surge_cfg)
            r = sim.run()
            elapsed = time.time() - t0

            # Save queue log for plotting
            ql_df = pd.DataFrame(sim.queue_log,
                                  columns=["month","s1_len","s2_len","arrivals",
                                           "n_completed","n_reneged","mean_wait_completed"])
            ql_df["algorithm"] = algo
            ql_df["complexity_config"] = cfg_name
            ql_path = output_dir / f"queue_log_surge_{algo}_{cfg_name}.csv"
            ql_df.to_csv(ql_path, index=False)

            print(f"wait={r['mean_total_time']:.1f}mo renege={r['renege_rate']:.3f} ({elapsed:.1f}s)")
            rows.append({
                "algorithm": algo,
                "complexity_config": cfg_name,
                "surge_factor": surge_factor,
                **r,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "surge_results.csv", index=False)
    print(f"\nSurge results saved → {output_dir}/surge_results.csv")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--era",        default=None, choices=ERAS)
    parser.add_argument("--archetype",  default=None, choices=ARCHETYPES)
    parser.add_argument("--scale",      type=float, default=SCALE_FACTOR)
    parser.add_argument("--months",     type=float, default=SIM_MONTHS)
    parser.add_argument("--surge-only", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.surge_only:
        run_surge_test(args.scale, args.months)
    else:
        run_fifo_lifo_experiment(args.era, args.archetype, args.scale, args.months)
        run_surge_test(args.scale, args.months)