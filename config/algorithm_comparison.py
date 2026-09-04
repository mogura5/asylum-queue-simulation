"""
ALGORITHM COMPARISON EXPERIMENT
================================
Guideline points 8-9:
  8. Combine all algorithms against one another under the same simulation.
  9. Graph differences in: mean wait time, reneging rate, throughput,
     economic needs (econ_welfare), backlog, true_unresolved.

ALGORITHMS COMPARED (6 total):
  FIFO             — arrival order (procedural fairness baseline)
  LIFO             — newest first (current U.S. policy)
  PRIORITY_DEFAULT — detained+3, rep+1, aging=6mo (default scoring)
  PRIORITY_OPT     — best scoring from scoring experiment per archetype
  CEL_ECON_ONLY    — econ filter only (econ>=6.67, p=15%)
  CEL_BROAD        — liberal lottery (comp<=4.5, econ>=4, p=40%)

METRICS COLLECTED:
  mean_total_wait      — primary welfare metric
  p95_wait             — tail inequality
  renege_rate          — dropout
  decided_throughput   — actual decisions per month
  true_unresolved      — backlog + reneged (welfare backlog)
  econ_welfare         — sum of econ scores for expedited (CEL only)
  gini_wait            — wait time inequality
  wait_by_complexity   — mean wait broken out by complexity quartile

OUTPUTS: ../../data/algorithm_comparison/
  comparison_results.csv      — 48 rows (6 algos × 8 archetype-era)
  complexity_breakdown.csv    — wait by complexity quartile per algorithm

RUN:
  python3 algorithm_comparison.py
  python3 algorithm_comparison.py --era pre_covid_2019
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
    AsylumSimulator, SCALE_FACTOR, SIM_MONTHS, RANDOM_SEED
)
from cel_algorithm import CELSimulator, LOTTERY_CONFIGS

PARAMS_PATH    = "../data/sim_parameters_FINAL.json"
SCORING_SUMMARY= "../data/scoring_all_results.csv"
OUTPUT_DIR     = Path("../data/algorithm_comparison")
ERAS           = ["pre_covid_2019", "post_covid_2024"]
ARCHETYPES     = ["high_vol_low_grant","high_vol_high_grant",
                  "low_vol_low_grant","low_vol_high_grant"]

# Optimal Priority configs from scoring experiment (era, archetype) -> (det, rep, comp, age)
OPTIMAL_SCORING = {
    ("pre_covid_2019","high_vol_low_grant"):  (3.0, 0.0, 1.0, 12.0),
    ("pre_covid_2019","high_vol_high_grant"): (1.0, 1.0, 1.0,  3.0),
    ("pre_covid_2019","low_vol_low_grant"):   (0.0, 1.0, 1.0,  3.0),
    ("pre_covid_2019","low_vol_high_grant"):  (5.0, 0.0, 1.0,  6.0),
    ("post_covid_2024","high_vol_low_grant"): (5.0, 0.0, 1.0, 12.0),
    ("post_covid_2024","high_vol_high_grant"):(0.0, 0.0, 1.0,  3.0),
    ("post_covid_2024","low_vol_low_grant"):  (0.0, 0.0, 1.0,  3.0),
    ("post_covid_2024","low_vol_high_grant"): (0.0, 1.0, 1.0, 12.0),
}


def gini(arr):
    if not arr or len(arr) < 2:
        return 0.0
    a = np.sort(np.array(arr))
    n = len(a)
    return float((2*np.sum(np.arange(1,n+1)*a) - (n+1)*np.sum(a)) / (n*np.sum(a) + 1e-9))


def extract_metrics(result: dict, sim_cases=None) -> dict:
    """Standardize metrics across simulator types."""
    n          = result.get("n_completed", result.get("n_total", 0))
    n_reneged  = result.get("n_reneged", 0)
    n_decided  = result.get("n_decided", n - n_reneged)
    backlog    = result.get("backlog_total", 0)
    unresolved = result.get("true_unresolved", backlog + n_reneged)
    renege_r   = result.get("renege_rate_obs", result.get("renege_rate",
                 result.get("renege_rate_regular", n_reneged/max(1,n))))

    total_times = []
    complexity_scores = []
    if sim_cases:
        total_times      = [c.total_time for c in sim_cases if c.total_time > 0]
        complexity_scores= [c.complexity_score for c in sim_cases]

    # Complexity quartile wait times
    comp_wait = {}
    if len(complexity_scores) > 10:
        cs = np.array(complexity_scores)
        wt = np.array([c.total_time for c in sim_cases])
        q25, q50, q75 = np.percentile(cs, [25,50,75])
        for label, mask in [
            ("q1_simple",  cs <= q25),
            ("q2_low",    (cs > q25) & (cs <= q50)),
            ("q3_med",    (cs > q50) & (cs <= q75)),
            ("q4_complex", cs > q75),
        ]:
            comp_wait[f"wait_{label}"] = float(np.mean(wt[mask])) if mask.sum()>0 else 0

    return {
        "n_completed":       n,
        "n_decided":         n_decided,
        "n_reneged":         n_reneged,
        "mean_wait":         result.get("mean_total_time", result.get("mean_wait_all", 0)),
        "p95_wait":          result.get("p95_total_time",  result.get("p95_total_time", 0)),
        "renege_rate":       renege_r,
        "decided_throughput":result.get("decided_throughput", n_decided/200),
        "backlog_total":     backlog,
        "true_unresolved":   unresolved,
        "econ_welfare":      result.get("econ_welfare_total", 0),
        "gini_wait":         gini(total_times) if total_times else result.get("gini_wait",0),
        **comp_wait,
    }


def run_comparison(era: str = None, scale_factor: float = SCALE_FACTOR,
                   sim_months: float = SIM_MONTHS,
                   capacity_multiplier: float = 1.0) -> pd.DataFrame:

    with open(PARAMS_PATH) as f:
        all_params = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eras_to_run = [era] if era else ERAS
    rows = []

    for era_key in eras_to_run:
        for arch in ARCHETYPES:
            params = all_params.get(era_key, {}).get(arch)
            if params is None:
                continue

            print(f"\n{'─'*55}")
            print(f"{era_key} / {arch}")
            print(f"{'─'*55}")

            # ── 1. FIFO ─────────────────────────────────────────────
            for algo_label, algo_key in [("FIFO","FIFO"),("LIFO","LIFO")]:
                print(f"  [{algo_label}]...", end=" ", flush=True)
                t0 = time.time()
                sim = AsylumSimulator(params, algo_key, scale_factor, sim_months, RANDOM_SEED, capacity_multiplier=capacity_multiplier)
                r   = sim.run()
                cases = r.pop("cases", [])
                r.pop("queue_log", None)
                m = extract_metrics(r, cases)
                print(f"wait={m['mean_wait']:.1f}mo renege={m['renege_rate']:.3f} ({time.time()-t0:.1f}s)")
                rows.append({"era":era_key,"archetype":arch,"algorithm":algo_label,**m})

            # ── 2. Priority DEFAULT ──────────────────────────────────
            print(f"  [PRIORITY_DEFAULT]...", end=" ", flush=True)
            t0 = time.time()
            sim = AsylumSimulator(params, "PRIORITY", scale_factor, sim_months, RANDOM_SEED, capacity_multiplier=capacity_multiplier)
            r   = sim.run()
            cases = r.pop("cases", [])
            r.pop("queue_log", None)
            m = extract_metrics(r, cases)
            print(f"wait={m['mean_wait']:.1f}mo renege={m['renege_rate']:.3f} ({time.time()-t0:.1f}s)")
            rows.append({"era":era_key,"archetype":arch,"algorithm":"PRIORITY_DEFAULT",**m})

            # ── 3. Priority OPTIMAL ──────────────────────────────────
            key = (era_key, arch)
            if key in OPTIMAL_SCORING:
                det, rep, comp_pen, age = OPTIMAL_SCORING[key]
                print(f"  [PRIORITY_OPT det={det},age={age}]...", end=" ", flush=True)
                t0 = time.time()
                from scoring_experiment import run_scored_simulation, ScoringConfig
                cfg_opt = ScoringConfig(detained_bonus=det, rep_bonus=rep,
                                        complexity_penalty=comp_pen, aging_threshold=age)
                r_opt = run_scored_simulation(params, cfg_opt, scale_factor, sim_months, RANDOM_SEED, capacity_multiplier=capacity_multiplier)
                if r_opt:
                    n_tot  = r_opt.get("n_completed", 0)
                    rr     = r_opt.get("renege_rate", 0)
                    n_r    = r_opt.get("n_decided", None)   # may not exist on older build
                    if n_r is None:
                        n_r = int(rr * n_tot)
                    n_d    = n_tot - n_r
                    bklog  = r_opt.get("backlog_total", 0)
                    dtput  = r_opt.get("decided_throughput", n_d / max(1, sim_months))
                    m_opt  = {
                        "n_completed":       n_tot,
                        "n_decided":         n_d,
                        "n_reneged":         n_r,
                        "mean_wait":         r_opt.get("mean_total_time", 0),
                        "p95_wait":          r_opt.get("p95_total_time", 0),
                        "renege_rate":       rr,
                        "decided_throughput":dtput,
                        "backlog_total":     bklog,
                        "true_unresolved":   bklog + n_r,
                        "econ_welfare":      0,
                        "gini_wait":         0,
                    }
                    print(f"wait={m_opt['mean_wait']:.1f}mo renege={m_opt['renege_rate']:.3f} ({time.time()-t0:.1f}s)")
                    rows.append({"era":era_key,"archetype":arch,"algorithm":"PRIORITY_OPT",**m_opt})

            # ── 4. CEL econ_only ────────────────────────────────────
            print(f"  [CEL_ECON_ONLY]...", end=" ", flush=True)
            t0 = time.time()
            cel_eo = CELSimulator(params, LOTTERY_CONFIGS["econ_only"],
                                  "FIFO", scale_factor, sim_months, RANDOM_SEED, capacity_multiplier=capacity_multiplier)
            r_eo = cel_eo.run()
            m_eo = extract_metrics(r_eo)
            print(f"wait={m_eo['mean_wait']:.1f}mo renege={m_eo['renege_rate']:.3f} econ_welf={r_eo.get('econ_welfare_total',0):.0f} ({time.time()-t0:.1f}s)")
            rows.append({"era":era_key,"archetype":arch,"algorithm":"CEL_ECON_ONLY",**m_eo})

            # ── 5. CEL broad ────────────────────────────────────────
            print(f"  [CEL_BROAD]...", end=" ", flush=True)
            t0 = time.time()
            cel_br = CELSimulator(params, LOTTERY_CONFIGS["broad"],
                                  "FIFO", scale_factor, sim_months, RANDOM_SEED, capacity_multiplier=capacity_multiplier)
            r_br = cel_br.run()
            m_br = extract_metrics(r_br)
            print(f"wait={m_br['mean_wait']:.1f}mo renege={m_br['renege_rate']:.3f} econ_welf={r_br.get('econ_welfare_total',0):.0f} ({time.time()-t0:.1f}s)")
            rows.append({"era":era_key,"archetype":arch,"algorithm":"CEL_BROAD",**m_br})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "comparison_results.csv", index=False)
    print(f"\nSaved {len(df)} rows → {OUTPUT_DIR}/comparison_results.csv")
    _print_comparison_table(df)
    return df


def _print_comparison_table(df: pd.DataFrame):
    print("\n" + "="*75)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*75)
    for era in df["era"].unique():
        print(f"\n  ERA: {era}")
        for arch in df["archetype"].unique():
            print(f"    [{arch}]")
            sub = df[(df["era"]==era)&(df["archetype"]==arch)]
            for _, r in sub.iterrows():
                print(f"      {r['algorithm']:20s} | "
                      f"wait={r['mean_wait']:6.1f}mo | "
                      f"renege={r['renege_rate']:.3f} | "
                      f"unresolved={r['true_unresolved']:,.0f} | "
                      f"decided={r['decided_throughput']:.2f}/mo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--era",   default=None, choices=ERAS)
    parser.add_argument("--scale", type=float, default=SCALE_FACTOR)
    parser.add_argument("--months",type=float, default=SIM_MONTHS)
    parser.add_argument("--capacity",type=float, default=1.0)
    args = parser.parse_args()
    run_comparison(args.era, args.scale, args.months, args.capacity)