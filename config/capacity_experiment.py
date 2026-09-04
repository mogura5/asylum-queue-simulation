"""
CAPACITY SENSITIVITY EXPERIMENT (APPEND MODE)
==============================================
Tests how increasing the number of Asylum Officers and Immigration Judges 
impacts the performance of different scheduling algorithms.
Smartly resumes progress by reading the existing CSV and only running new multipliers.
"""

import pandas as pd
import time
import json
from pathlib import Path
import os

# Import your tweaked simulators
from simulation_runner import AsylumSimulator, SCALE_FACTOR, SIM_MONTHS, RANDOM_SEED
from cel_algorithm import CELSimulator, LOTTERY_CONFIGS

PARAMS_PATH = "../data/sim_parameters_FINAL.json"
OUTPUT_DIR  = Path("../data/capacity_experiment")

# Add your extreme bounds here (4.0, 6.0, 8.0, 10.0)
CAPACITY_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 10.0]

def run_capacity_test(era="post_covid_2024", archetype="high_vol_low_grant"):
    with open(PARAMS_PATH) as f:
        all_params = json.load(f)
    
    params = all_params[era][archetype]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"capacity_results_{era}.csv"
    
    # --- 1. SMART RESUME LOGIC ---
    processed_mults = set()
    existing_df = pd.DataFrame()
    
    if out_path.exists():
        print(f"Found existing data at {out_path}. Loading...")
        existing_df = pd.read_csv(out_path)
        if 'capacity_multiplier' in existing_df.columns:
            processed_mults = set(existing_df['capacity_multiplier'].unique())
            print(f"Already processed multipliers: {sorted(list(processed_mults))}")
    
    # Filter out the ones we've already done
    mults_to_run = [m for m in CAPACITY_MULTIPLIERS if m not in processed_mults]
    
    if not mults_to_run:
        print("\nAll multipliers in CAPACITY_MULTIPLIERS have already been processed! Exiting.")
        return

    print(f"\nRunning NEW Capacity Multipliers: {mults_to_run}")
    print("="*60)

    new_rows = []

    # --- 2. RUN ONLY THE NEW MULTIPLIERS ---
    for cap_mult in mults_to_run:
        print(f"\n--- Capacity Multiplier: {cap_mult}x ---")
        
        # 1. FIFO
        t0 = time.time()
        sim_fifo = AsylumSimulator(params, "FIFO", SCALE_FACTOR, SIM_MONTHS, RANDOM_SEED, capacity_multiplier=cap_mult)
        r_fifo = sim_fifo.run()
        print(f"  FIFO: wait={r_fifo.get('mean_total_time',0):.1f}mo, backlog={r_fifo.get('backlog_total',0)}")
        new_rows.append({"capacity_multiplier": cap_mult, "algorithm": "FIFO", **r_fifo})

        # 2. LIFO (Current Policy)
        sim_lifo = AsylumSimulator(params, "LIFO", SCALE_FACTOR, SIM_MONTHS, RANDOM_SEED, capacity_multiplier=cap_mult)
        r_lifo = sim_lifo.run()
        print(f"  LIFO: wait={r_lifo.get('mean_total_time',0):.1f}mo, backlog={r_lifo.get('backlog_total',0)}")
        new_rows.append({"capacity_multiplier": cap_mult, "algorithm": "LIFO", **r_lifo})

        # 3. Priority
        sim_pri = AsylumSimulator(params, "PRIORITY", SCALE_FACTOR, SIM_MONTHS, RANDOM_SEED, capacity_multiplier=cap_mult)
        r_pri = sim_pri.run()
        print(f"  PRIORITY: wait={r_pri.get('mean_total_time',0):.1f}mo, backlog={r_pri.get('backlog_total',0)}")
        new_rows.append({"capacity_multiplier": cap_mult, "algorithm": "PRIORITY_OPT", **r_pri})

        # 4. CEL (Broad)
        sim_cel = CELSimulator(params, LOTTERY_CONFIGS["broad"], "FIFO", SCALE_FACTOR, SIM_MONTHS, RANDOM_SEED, capacity_multiplier=cap_mult)
        r_cel = sim_cel.run()
        print(f"  CEL (Broad): wait={r_cel.get('mean_wait_all',0):.1f}mo, backlog={r_cel.get('backlog_total',0)}")
        
        # Normalize CEL metrics
        cel_normalized = {
            "mean_total_time": r_cel.get("mean_wait_all", 0),
            "backlog_total": r_cel.get("backlog_total", 0),
            "renege_rate_obs": r_cel.get("renege_rate_regular", 0),
            "throughput": r_cel.get("throughput_total", 0)
        }
        new_rows.append({"capacity_multiplier": cap_mult, "algorithm": "CEL_BROAD", **cel_normalized})

    # --- 3. COMBINE AND SAVE ---
    new_df = pd.DataFrame(new_rows)
    cols_to_keep = ['capacity_multiplier', 'algorithm', 'mean_total_time', 'backlog_total', 'renege_rate_obs', 'throughput']
    new_df = new_df[[c for c in cols_to_keep if c in new_df.columns]]
    
    # Combine old data with new data
    final_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Sort it so it looks clean in the CSV (by multiplier, then algorithm)
    final_df = final_df.sort_values(by=['capacity_multiplier', 'algorithm'])
    
    final_df.to_csv(out_path, index=False)
    print(f"\nSuccessfully appended {len(new_rows)} new rows. Total rows now: {len(final_df)}.")
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    run_capacity_test()