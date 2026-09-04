"""
PRIORITY SCORING SENSITIVITY EXPERIMENT
========================================
Treats the priority queue's scoring weights as free parameters and runs
a structured grid of simulations to find which weight configuration best
achieves each welfare objective.

Four scoring dimensions are varied:
  detained_bonus     — extra priority for detained cases     [0, 1, 2, 3, 5]
  rep_bonus          — extra priority for represented cases  [0, 0.5, 1.0, 2.0]
  complexity_penalty — priority reduction per complexity pt  [0, 0.5, 1.0, 1.5]
  aging_threshold    — months before aging kicks in          [3, 6, 12, 24]

For each configuration we run one Priority simulation and record:
  mean_total_time    — lower = better if objective = minimize wait
  renege_rate        — lower = better if objective = minimize dropout
  backlog_total      — lower = better if objective = clear queue
  throughput         — higher = better if objective = maximize decisions

Results are saved to scoring_experiment_results.csv and a summary
table identifies the Pareto-optimal configuration for each objective.

Usage:
    python3 scoring_experiment.py
    python3 scoring_experiment.py --era pre_covid_2019 --archetype high_vol_low_grant
    python3 scoring_experiment.py --full-grid   # all eras × archetypes

Runtime: ~2-8 minutes depending on grid size and scale factor.
"""

import simpy
import numpy as np
import pandas as pd
import json
import time
import itertools
import argparse
from dataclasses import dataclass, field, replace
from typing import Optional
from collections import deque
from pathlib import Path

# ── import shared components from main simulator ──────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation_runner import (
    Case, PriorityQueue,
    COMPLEXITY_SERVICE_MULTIPLIER, PATIENCE_BASE_MONTHS,
    COMPLEXITY_WEIGHTS, SCALE_FACTOR, SIM_MONTHS, RANDOM_SEED
)

# ── paths ─────────────────────────────────────────────────────────────────
PARAMS_PATH = "../data/sim_parameters_FINAL.json"
OUTPUT_PATH = "../data/scoring_experiment_results.csv"

# ── scoring grid ──────────────────────────────────────────────────────────
# Each dimension represents a welfare hypothesis:
#   detained_bonus:     how much should custody urgency matter?
#   rep_bonus:          how much should representation matter?
#   complexity_penalty: should complex cases wait longer (penalised) or get
#                       faster resolution (bonus = negative penalty)?
#   aging_threshold:    how long before a case gets emergency priority?
SCORING_GRID = {
    "detained_bonus":     [0.0, 1.0, 3.0, 5.0],   # default = 3.0
    "rep_bonus":          [0.0, 0.5, 1.0, 2.0],    # default = 1.0
    "complexity_penalty": [0.0, 0.5, 1.0, 1.5],    # default = 1.0
    "aging_threshold":    [3.0, 6.0, 12.0, 24.0],  # default = 6.0 months
}

# For quick runs, use the reduced grid (vary one dimension at a time)
SCORING_GRID_REDUCED = {
    "detained_bonus":  [0.0, 1.0, 3.0, 5.0],
    "rep_bonus":       [0.0, 1.0],          # collapsed — barely matters
    "complexity_penalty": [0.0, 1.0],       # collapsed — need scale=0.30 to see it
    "aging_threshold": [3.0, 6.0, 12.0],    # drop 24 — it collapses throughput
}

# Default values (used when a dimension is held fixed in one-at-a-time runs)
DEFAULTS = {
    "detained_bonus": 3.0,
    "rep_bonus": 1.0,
    "complexity_penalty": 1.0,
    "aging_threshold": 6.0,
    "aging_boost": 2.0,
}


@dataclass
class ScoringConfig:
    """One priority scoring configuration to be tested."""
    detained_bonus:     float
    rep_bonus:          float
    complexity_penalty: float
    aging_threshold:    float
    aging_boost:        float = 2.0   # kept fixed — varies independently


def run_scored_simulation(params: dict, scoring: ScoringConfig,
                          scale_factor: float, sim_months: float,
                          seed: int,
                          capacity_multiplier: float = 1.0) -> dict:
    """
    Run one Priority-queue simulation with a given scoring configuration.
    Stripped-down version of AsylumSimulator that accepts external scoring.
    """
    lam    = params["arrival_rate_lambda"] * scale_factor
    c1     = max(1, round(200 * scale_factor * capacity_multiplier))
    c2     = max(1, round(175 * scale_factor * capacity_multiplier))
    mu     = params["service_time_mean"]
    std    = params["service_time_std"] if params["service_time_std"] > 0 else mu * 0.5
    s1_mean = mu * 0.20;  s1_std = std * 0.20
    s2_mean = mu * 0.80;  s2_std = std * 0.80
    grant_rate  = params["grant_rate"]
    appeal_rate = params["appeal_rate"]
    p_rep       = params["p_rep"]
    p_detained  = params["p_detained"]

    rng  = np.random.default_rng(seed)
    env  = simpy.Environment()
    officers = simpy.Resource(env, capacity=c1)
    judges   = simpy.Resource(env, capacity=c2)

    s1_queue = PriorityQueue(scoring.aging_threshold, scoring.aging_boost)
    s2_queue = PriorityQueue(scoring.aging_threshold, scoring.aging_boost)

    completed = []
    queue_log = []
    ctr = [0]

    def draw(m, s):
        if s <= 0 or m <= 0:
            return max(0.01, rng.exponential(m))
        s2v = np.log(1 + (s/m)**2)
        ml  = np.log(m) - s2v/2
        return max(0.01, rng.lognormal(ml, np.sqrt(s2v)))

    def make_case(arrival_time, is_appeal=False):
        cid = ctr[0]; ctr[0] += 1
        has_rep     = rng.random() < p_rep
        detained    = rng.random() < p_detained
        is_family   = rng.random() < params.get("p_family", 0.30)
        filed_late  = rng.random() < params.get("p_late", 0.25)
        has_criminal= rng.random() < params.get("p_criminal", 0.018)
        nbr_charges = int(rng.choice([0,1,2,3],
                          p=params.get("charge_dist", [0.45,0.35,0.15,0.05])))

        cw   = COMPLEXITY_WEIGHTS
        comp = (cw["no_rep"]       * (not has_rep)      +
                cw["has_criminal"] * has_criminal        +
                cw["is_family"]    * is_family           +
                cw["filed_late"]   * filed_late          +
                cw["multi_charge"] * (nbr_charges >= 2))

        # Priority score using THIS run's scoring configuration
        priority = 1.0
        if detained:
            priority += scoring.detained_bonus
        if has_rep:
            priority += scoring.rep_bonus
        if is_appeal:
            priority += 2.0
        priority -= (comp / 10.0) * scoring.complexity_penalty

        s1t = draw(s1_mean, s1_std)
        s2t = draw(s2_mean, s2_std)
        eff_s2 = s2t * (1.0 + COMPLEXITY_SERVICE_MULTIPLIER * comp / 10.0)

        return Case(
            case_id=cid, arrival_time=arrival_time,
            has_rep=has_rep, is_detained=detained,
            is_family=is_family, filed_late=filed_late,
            has_criminal=has_criminal, nbr_charges=nbr_charges,
            complexity_score=round(comp, 2),
            base_priority=round(priority, 2),
            service_time_s1=s1t, service_time_s2=eff_s2,
            is_appeal=is_appeal
        )

    def stage2(env, case):
        case.s2_queue_enter = env.now
        patience = float("inf") if case.is_detained else rng.exponential(PATIENCE_BASE_MONTHS)
        s2_queue.push(case, env.now)
        with judges.request() as req:
            result = yield req | env.timeout(patience)
            if req in result:
                served = s2_queue.pop(env.now) or case
                served.s2_start = env.now
                yield env.timeout(served.service_time_s2)
                served.s2_end = env.now
                served.total_time = served.s2_end - served.arrival_time
                if rng.random() < grant_rate:
                    served.outcome = "appeal_granted" if served.is_appeal else "granted"
                else:
                    if rng.random() < appeal_rate and not served.is_appeal:
                        served.outcome = "pending_appeal"
                        ac = make_case(env.now, is_appeal=True)
                        ac.case_id = served.case_id
                        env.process(stage2(env, ac))
                    else:
                        served.outcome = "denied"
                if served.outcome != "pending_appeal":
                    completed.append(served)
            else:
                case.outcome = "reneged"
                case.total_time = env.now - case.arrival_time
                completed.append(case)

    def stage1(env, case):
        case.s1_queue_enter = env.now
        s1_queue.push(case, env.now)
        with officers.request() as req:
            yield req
            nc = s1_queue.pop(env.now) or case
            nc.s1_start = env.now
            yield env.timeout(nc.service_time_s1)
            nc.s1_end = env.now
        env.process(stage2(env, nc))

    def arrivals(env):
        while True:
            yield env.timeout(rng.exponential(1.0/lam))
            env.process(stage1(env, make_case(env.now)))

    def monitor(env):
        while True:
            queue_log.append((env.now, len(s1_queue), len(s2_queue)))
            yield env.timeout(1.0)

    def aging_proc(env):
        while True:
            yield env.timeout(1.0)
            s1_queue.apply_aging(env.now)
            s2_queue.apply_aging(env.now)

    env.process(arrivals(env))
    env.process(monitor(env))
    env.process(aging_proc(env))
    env.run(until=sim_months)

    n = len(completed)
    if n == 0:
        return {}

    total_times = [c.total_time for c in completed if c.total_time > 0]
    outcomes    = [c.outcome for c in completed]

    return {
        "n_completed":     n,
        "mean_total_time": float(np.mean(total_times)) if total_times else 0,
        "p95_total_time":  float(np.percentile(total_times, 95)) if total_times else 0,
        "backlog_total":   len(s1_queue) + len(s2_queue),
        "throughput":      n / sim_months,
        "renege_rate":     outcomes.count("reneged") / n,
        "grant_rate_obs":  outcomes.count("granted") / n,
    }


def build_scoring_configs(grid: dict) -> list[ScoringConfig]:
    """Build all combinations from the grid."""
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    configs = []
    for combo in combos:
        d = dict(zip(keys, combo))
        configs.append(ScoringConfig(
            detained_bonus=d.get("detained_bonus", DEFAULTS["detained_bonus"]),
            rep_bonus=d.get("rep_bonus", DEFAULTS["rep_bonus"]),
            complexity_penalty=d.get("complexity_penalty", DEFAULTS["complexity_penalty"]),
            aging_threshold=d.get("aging_threshold", DEFAULTS["aging_threshold"]),
        ))
    return configs


def run_scoring_experiment(
    era: str = "pre_covid_2019",
    archetype: str = "high_vol_low_grant",
    scale_factor: float = SCALE_FACTOR,
    sim_months: float = SIM_MONTHS,
    grid: dict = None,
    output_csv: str = OUTPUT_PATH,
):
    """
    Run the full scoring sensitivity experiment for one era × archetype.

    For each scoring configuration, runs one Priority-queue simulation
    and records key welfare metrics. Also runs FIFO and LIFO as baselines
    so results can be expressed as improvement/degradation relative to them.
    """
    if grid is None:
        grid = SCORING_GRID_REDUCED

    with open(PARAMS_PATH) as f:
        all_params = json.load(f)

    params = all_params.get(era, {}).get(archetype)
    if params is None:
        raise ValueError(f"Parameters not found for {era}/{archetype}")

    configs = build_scoring_configs(grid)
    n_configs = len(configs)

    print(f"\n{'='*60}")
    print(f"SCORING SENSITIVITY EXPERIMENT")
    print(f"Era: {era}  |  Archetype: {archetype}")
    print(f"Configurations to test: {n_configs}")
    print(f"Scale factor: {scale_factor}  |  Sim months: {sim_months}")
    print(f"{'='*60}\n")

    # ── Baselines: FIFO and LIFO ──────────────────────────────────────────
    from simulation_runner import AsylumSimulator
    rows = []

    for baseline_algo in ["FIFO", "LIFO"]:
        print(f"  Running baseline: {baseline_algo}...")
        t0 = time.time()
        sim = AsylumSimulator(params, baseline_algo, scale_factor, sim_months, RANDOM_SEED)
        result = sim.run()
        elapsed = time.time() - t0
        row = {
            "algorithm": baseline_algo,
            "detained_bonus": "—",
            "rep_bonus": "—",
            "complexity_penalty": "—",
            "aging_threshold": "—",
            "mean_total_time":  round(result.get("mean_total_time", 0), 2),
            "p95_total_time":   round(result.get("p95_total_time", 0), 2),
            "backlog_total":    result.get("backlog_total", 0),
            "throughput":       round(result.get("throughput", 0), 2),
            "renege_rate":      round(result.get("renege_rate_obs", 0), 4),
            "grant_rate_obs":   round(result.get("grant_rate_obs", 0), 4),
            "runtime_s":        round(elapsed, 1),
        }
        rows.append(row)
        print(f"    → wait={row['mean_total_time']:.1f}mo "
              f"renege={row['renege_rate']:.3f} "
              f"backlog={row['backlog_total']:,} ({elapsed:.1f}s)")

    # ── Priority configurations ────────────────────────────────────────────
    print(f"\n  Running {n_configs} Priority configurations...\n")
    fifo_wait = rows[0]["mean_total_time"]   # for relative comparison

    for i, cfg in enumerate(configs):
        t0 = time.time()
        result = run_scored_simulation(params, cfg, scale_factor, sim_months, RANDOM_SEED)
        elapsed = time.time() - t0

        if not result:
            continue

        row = {
            "algorithm":        "PRIORITY",
            "detained_bonus":   cfg.detained_bonus,
            "rep_bonus":        cfg.rep_bonus,
            "complexity_penalty": cfg.complexity_penalty,
            "aging_threshold":  cfg.aging_threshold,
            "mean_total_time":  round(result["mean_total_time"], 2),
            "p95_total_time":   round(result["p95_total_time"], 2),
            "backlog_total":    result["backlog_total"],
            "throughput":       round(result["throughput"], 2),
            "renege_rate":      round(result["renege_rate"], 4),
            "grant_rate_obs":   round(result["grant_rate_obs"], 4),
            "runtime_s":        round(elapsed, 1),
        }
        rows.append(row)

        # Print every 10th config + any that beats FIFO on wait
        if (i % 10 == 0) or (row["mean_total_time"] < fifo_wait - 5):
            tag = " ← beats FIFO" if row["mean_total_time"] < fifo_wait - 5 else ""
            print(f"  [{i+1}/{n_configs}] det={cfg.detained_bonus} rep={cfg.rep_bonus} "
                  f"comp={cfg.complexity_penalty} age={cfg.aging_threshold} | "
                  f"wait={row['mean_total_time']:.1f} renege={row['renege_rate']:.3f}{tag}")

    # ── Save results ──────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.insert(0, "era", era)
    df.insert(1, "archetype", archetype)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} rows to: {output_csv}")

    # ── Summary: best config per welfare objective ─────────────────────────
    priority_df = df[df["algorithm"] == "PRIORITY"].copy()
    priority_df["detained_bonus"]     = priority_df["detained_bonus"].astype(float)
    priority_df["rep_bonus"]          = priority_df["rep_bonus"].astype(float)
    priority_df["complexity_penalty"] = priority_df["complexity_penalty"].astype(float)
    priority_df["aging_threshold"]    = priority_df["aging_threshold"].astype(float)

    print(f"\n{'='*60}")
    print("BEST PRIORITY CONFIGURATION PER WELFARE OBJECTIVE")
    print(f"{'='*60}")
    print(f"  FIFO baseline:  wait={rows[0]['mean_total_time']:.1f}mo  "
          f"renege={rows[0]['renege_rate']:.3f}")
    print(f"  LIFO baseline:  wait={rows[1]['mean_total_time']:.1f}mo  "
          f"renege={rows[1]['renege_rate']:.3f}\n")

    objectives = {
        "Minimize mean wait time":      ("mean_total_time",  False),
        "Minimize reneging (dropout)":  ("renege_rate",      False),
        "Minimize residual backlog":    ("backlog_total",     False),
        "Maximize throughput":          ("throughput",        True),
        "Minimize P95 wait time":       ("p95_total_time",    False),
    }

    summary_rows = []
    for obj_name, (col, ascending_is_best) in objectives.items():
        if ascending_is_best:
            best = priority_df.loc[priority_df[col].idxmax()]
        else:
            best = priority_df.loc[priority_df[col].idxmin()]
        print(f"  Objective: {obj_name}")
        print(f"    Best scoring: detained={best['detained_bonus']:.1f}  "
              f"rep={best['rep_bonus']:.1f}  "
              f"complexity_pen={best['complexity_penalty']:.1f}  "
              f"aging={best['aging_threshold']:.0f}mo")
        print(f"    Result:  wait={best['mean_total_time']:.1f}mo  "
              f"renege={best['renege_rate']:.3f}  "
              f"backlog={best['backlog_total']:,}\n")
        summary_rows.append({
            "welfare_objective":  obj_name,
            "detained_bonus":     best["detained_bonus"],
            "rep_bonus":          best["rep_bonus"],
            "complexity_penalty": best["complexity_penalty"],
            "aging_threshold":    best["aging_threshold"],
            "mean_total_time":    best["mean_total_time"],
            "renege_rate":        best["renege_rate"],
            "backlog_total":      best["backlog_total"],
            "throughput":         best["throughput"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_csv.replace(".csv", "_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    return df, summary_df


def run_full_grid_experiment(
    scale_factor: float = SCALE_FACTOR,
    sim_months: float = SIM_MONTHS,
):
    """Run experiment across all eras × archetypes."""
    eras       = ["pre_covid_2019", "post_covid_2024"]
    archetypes = ["high_vol_low_grant","high_vol_high_grant",
                  "low_vol_low_grant","low_vol_high_grant"]
    all_rows = []
    for era in eras:
        for arch in archetypes:
            df, _ = run_scoring_experiment(
                era=era, archetype=arch,
                scale_factor=scale_factor,
                sim_months=sim_months,
                output_csv=f"../data/scoring_{era}_{arch}.csv"
            )
            all_rows.append(df)
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv("../data/scoring_all_results.csv", index=False)
    print(f"\nFull grid complete. {len(combined)} total rows saved.")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Priority scoring sensitivity experiment")
    parser.add_argument("--era",       default="pre_covid_2019",
                        choices=["pre_covid_2019","post_covid_2024"])
    parser.add_argument("--archetype", default="high_vol_low_grant",
                        choices=["high_vol_low_grant","high_vol_high_grant",
                                 "low_vol_low_grant","low_vol_high_grant"])
    parser.add_argument("--scale",     type=float, default=SCALE_FACTOR)
    parser.add_argument("--months",    type=float, default=SIM_MONTHS)
    parser.add_argument("--full-grid", action="store_true",
                        help="Run all eras × archetypes (slow)")
    parser.add_argument("--output",    default=OUTPUT_PATH)
    args = parser.parse_args()

    if args.full_grid:
        run_full_grid_experiment(args.scale, args.months)
    else:
        run_scoring_experiment(
            era=args.era,
            archetype=args.archetype,
            scale_factor=args.scale,
            sim_months=args.months,
            output_csv=args.output,
        )