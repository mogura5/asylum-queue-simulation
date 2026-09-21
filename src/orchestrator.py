"""
Orchestration runner for the asylum queue simulator.

Replaces simulation_runner.py's run_all_experiments() / run_pairwise_ttests():
loads the real calibrated sim_parameters_FINAL.json and runs AsylumSimulator
(case.py/queues.py/stage.py/simulator.py) across every era x archetype x
algorithm combination, producing the results the paper's comparisons are
built on.

Design decisions locked in for this pass (see redesign-decisions.md,
"Next step: orchestration runner"):

1. SCALE FACTOR (0.30): real arrival rates now run into the tens of
   thousands per month for some archetypes (e.g. post-covid high_volume
   lambda ~= 37,069/month), so this scales BOTH arrival_rate_lambda AND
   n_officers/n_judges by the same factor. Scaling only one side would
   distort the utilization ratio (lambda / (c * mu)) and make backlog/wait
   results unrepresentative of the real system.

2. WARM-UP PERIOD (20 months of 100): the queue starts empty at t=0, so
   the earliest cases see artificially low congestion. Only cases that
   ARRIVE after WARMUP_MONTHS are included in the reported wait/throughput/
   outcome statistics -- this is the standard fix for initialization bias
   in steady-state discrete-event simulation. backlog_s1/backlog_s2 are an
   end-of-run snapshot and are reported as-is (not filtered), since they
   describe the state at simulation end regardless of history.

3. REPLICATIONS (15 independent seeded runs per era/archetype/algorithm):
   the original simulation_runner.py used a single fixed seed and treated
   within-run cases as the statistical sample for its t-tests -- a
   pseudoreplication risk (cases within one run aren't independent draws
   of "the algorithm's performance"). Here, each (era, archetype, algorithm)
   combination is run N_REPLICATIONS times with different seeds, and the
   summary + t-tests are computed over the REPLICATION-level means, not
   individual cases.

4. ALGORITHM SCOPE: FIFO / LIFO / PRIORITY were the first pass. CEL's
   lottery pathway is now ported (see cel.py + simulator.py's
   _advance_to_stage2/_lottery_draw_process) as two locked-in variants,
   "CEL_balanced" and "CEL_broad" -- pass them via --algorithms, they are
   NOT run by default (ALGORITHMS below is unchanged) since a full CEL
   grid is comparable in cost to the original FIFO/LIFO/PRIORITY run.

RESULTS MERGING: running with --algorithms limited to a subset (e.g. just
the two CEL variants) does NOT overwrite prior results for other
algorithms in the same --output directory. Existing replications_raw.csv
rows are loaded, combined with the new run's rows, deduplicated on
(era, archetype, algorithm, replication) keeping the newest, and
simulation_summary.csv / pairwise_ttests.csv are always regenerated from
that FULL merged set -- so the final tables cover every algorithm that
has ever been run into that output directory, not just this invocation's.
Pass --fresh to disable this and overwrite from scratch instead.

Outputs (written to --output):
  replications_raw.csv   -- one row per (era, archetype, algorithm, replication)
  simulation_summary.csv -- mean/std aggregated across replications
  pairwise_ttests.csv    -- t-tests on replication-level mean_total_time

LOGGING: this module uses the standard `logging` module (logger name
"orchestrator") instead of print(). Verbosity is controlled by --log-level
(default INFO); pass --log-level DEBUG for per-task detail or WARNING to
quiet routine progress output. Every entry point that can fail on bad
input (a malformed params file, a broken replication, a write to a
read-only --output path) catches the specific error, logs it clearly, and
either continues (a single failed replication is skipped, not fatal) or
exits with a non-zero status and a one-line explanation (a missing params
file, an empty --algorithms list) -- never a raw traceback.
"""

import sys
import os
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from simulator import AsylumSimulator

log = logging.getLogger("orchestrator")

# ============================================================
# CONFIGURATION
# ============================================================

SCALE_FACTOR    = 0.30   # applied to arrival_rate_lambda AND n_officers/n_judges
SIM_MONTHS      = 100    # total simulated horizon
WARMUP_MONTHS   = 20     # cases arriving before this are excluded from stats
N_REPLICATIONS  = 15     # independent seeded runs per (era, archetype, algorithm)
BASE_SEED       = 42     # seeds used: BASE_SEED, BASE_SEED+1, ..., BASE_SEED+N-1

ALGORITHMS = ["FIFO", "LIFO", "PRIORITY"]  # pass --algorithms CEL_balanced CEL_broad explicitly

# NOTE: n_completed_post_warmup is included here (it wasn't previously),
# so simulation_summary.csv carries n_completed_post_warmup_mean/_std --
# needed by figures/generate_figures.py's completion-rate/backlog-fraction
# panel, which otherwise skips itself with a "missing column" warning.
_AGG_COLS = [
    "n_completed_post_warmup",
    "throughput", "backlog_s1", "backlog_s2", "backlog_total",
    "mean_wait_s1", "mean_wait_s2", "mean_total_time", "p95_total_time",
    "grant_rate_obs", "denial_rate_obs", "renege_rate_obs",
]

# Only populated for CEL_* replications (see cel.py / simulator.py); absent
# (NaN) for FIFO/LIFO/PRIORITY rows. Aggregated automatically when present.
_CEL_AGG_COLS = [
    "n_expedited_post_warmup", "expedited_rate",
    "mean_wait_expedited", "mean_wait_regular",
    "econ_welfare_total", "gini_wait",
]

_REQUIRED_PARAM_KEYS = {
    "arrival_rate_lambda", "n_officers", "n_judges", "grant_rate",
    "appeal_rate", "p_renege", "p_rep", "p_detained",
}


class OrchestratorError(Exception):
    """Raised for orchestrator-level failures (bad params file, empty
    algorithm list, etc.) that should stop the run with a clear message
    rather than an unhandled traceback."""


# ============================================================
# SCALING
# ============================================================

def _scaled_params(raw_params: dict, scale_factor: float) -> dict:
    """Copy of raw_params with arrival_rate_lambda, n_officers, n_judges
    scaled by scale_factor (server counts floored to >=1 via round()+max()).
    Everything else (rates, probabilities, charge_dist, service-time stats)
    is left untouched -- those are per-case parameters, not volume/capacity
    counts, so they don't need scaling.

    Raises OrchestratorError if raw_params is missing a required key or a
    required numeric field isn't actually numeric -- surfaced here, at
    scaling time, rather than as a confusing TypeError deep inside case.py
    once the simulation is already running.
    """
    missing = _REQUIRED_PARAM_KEYS - raw_params.keys()
    if missing:
        raise OrchestratorError(
            f"params entry is missing required key(s): {sorted(missing)} "
            f"-- check sim_parameters_FINAL.json for this era/archetype."
        )
    for key in ("arrival_rate_lambda", "n_officers", "n_judges"):
        if not isinstance(raw_params[key], (int, float)):
            raise OrchestratorError(
                f"params['{key}'] = {raw_params[key]!r} is not numeric "
                f"(got {type(raw_params[key]).__name__}) -- check "
                f"sim_parameters_FINAL.json for this era/archetype."
            )

    scaled = dict(raw_params)
    scaled["arrival_rate_lambda"] = raw_params["arrival_rate_lambda"] * scale_factor
    scaled["n_officers"] = max(1, round(raw_params["n_officers"] * scale_factor))
    scaled["n_judges"] = max(1, round(raw_params["n_judges"] * scale_factor))
    return scaled


# ============================================================
# ONE REPLICATION
# ============================================================

def _post_warmup_cases(cases, warmup_months: float):
    """Steady-state cohort: cases whose entire journey (queueing + service)
    happened after the system had time to fill up from empty-at-t=0, so
    early artificially-uncongested cases don't bias the averages."""
    return [c for c in cases if c.arrival_time >= warmup_months]


def _summarize(cases, sim_months: float, warmup_months: float,
               backlog_s1: int, backlog_s2: int, algorithm: str) -> Optional[dict]:
    n = len(cases)
    if n == 0:
        return None

    wait_s1 = [c.s1_start - c.s1_queue_enter for c in cases if c.s1_start > 0]
    wait_s2 = [c.s2_start - c.s2_queue_enter for c in cases if c.s2_start > 0]
    total = [c.total_time for c in cases if c.total_time > 0]
    outcomes = [c.outcome for c in cases]
    steady_state_months = max(1e-9, sim_months - warmup_months)

    summary = {
        "algorithm": algorithm,
        "n_completed_post_warmup": n,
        "throughput": n / steady_state_months,
        "backlog_s1": backlog_s1,
        "backlog_s2": backlog_s2,
        "backlog_total": backlog_s1 + backlog_s2,
        "mean_wait_s1": float(np.mean(wait_s1)) if wait_s1 else 0.0,
        "mean_wait_s2": float(np.mean(wait_s2)) if wait_s2 else 0.0,
        "mean_total_time": float(np.mean(total)) if total else 0.0,
        "p95_total_time": float(np.percentile(total, 95)) if total else 0.0,
        "grant_rate_obs": outcomes.count("granted") / n,
        "denial_rate_obs": outcomes.count("denied") / n,
        "renege_rate_obs": outcomes.count("reneged") / n,
    }

    # CEL-specific welfare/equity metrics (see cel.py). Computed from the
    # same post-warmup cohort as everything else above, so these are
    # directly comparable to the FIFO/LIFO/PRIORITY steady-state numbers --
    # NOT recomputed over the full run, which would mix in warm-up-period
    # cases and break that comparability.
    if algorithm.startswith("CEL_"):
        expedited = [c for c in cases if c.outcome == "lottery_expedited"]
        regular = [c for c in cases if c.outcome != "lottery_expedited"]
        exp_waits = [c.total_time for c in expedited if c.total_time > 0]
        reg_waits = [c.total_time for c in regular if c.total_time > 0]
        all_waits = exp_waits + reg_waits

        def _gini(arr):
            if not arr or len(arr) < 2:
                return 0.0
            a = np.sort(np.array(arr))
            n2 = len(a)
            return float(
                (2 * np.sum(np.arange(1, n2 + 1) * a) - (n2 + 1) * np.sum(a))
                / (n2 * np.sum(a) + 1e-9)
            )

        summary.update({
            "n_expedited_post_warmup": len(expedited),
            "expedited_rate": len(expedited) / n,
            "mean_wait_expedited": float(np.mean(exp_waits)) if exp_waits else 0.0,
            "mean_wait_regular": float(np.mean(reg_waits)) if reg_waits else 0.0,
            "econ_welfare_total": float(sum(c.econ_need_score for c in expedited)),
            "gini_wait": _gini(all_waits),
        })

    return summary


def run_one_replication(scaled_params: dict, algorithm: str,
                         sim_months: float, warmup_months: float, seed: int) -> Optional[dict]:
    sim = AsylumSimulator(
        params=scaled_params,
        algorithm=algorithm,
        n_officers=scaled_params["n_officers"],
        n_judges=scaled_params["n_judges"],
        sim_months=sim_months,
        seed=seed,
    )
    result = sim.run()
    cases = result.get("cases", []) if result else []
    post_warmup = _post_warmup_cases(cases, warmup_months)
    return _summarize(
        post_warmup, sim_months, warmup_months,
        result.get("backlog_s1", 0) if result else 0,
        result.get("backlog_s2", 0) if result else 0,
        algorithm,
    )


# ============================================================
# FULL GRID
# ============================================================

def _report_task_result(i: int, total: int, out: dict, elapsed):
    label = out["label"]
    summary = out["summary"]
    error = out.get("error")
    timing = f"{elapsed:.1f}s" if elapsed is not None else "done"

    if error is not None:
        log.error("[%d/%d] %s -- FAILED: %s", i, total, label, error)
        return
    if summary is None:
        log.warning("[%d/%d] %s -- no post-warmup completions, skipped", i, total, label)
        return
    log.info(
        "[%d/%d] %s (%s) -> wait=%.1fmo backlog=%s renege=%.3f",
        i, total, label, timing,
        summary["mean_total_time"], f"{summary['backlog_total']:,}",
        summary["renege_rate_obs"],
    )


def _run_task(task: dict) -> dict:
    """Top-level (picklable) worker function for ProcessPoolExecutor: runs
    one replication and returns the summary dict merged with its
    era/archetype/replication/seed labels, or a dict with summary=None if
    the replication had no post-warmup completions.

    Any exception raised while building/running the simulator for this one
    task is caught here and returned as an error string rather than
    propagated -- a single bad replication (e.g. a pathological RNG draw
    triggering a division by zero somewhere downstream) should not abort
    the entire grid, sequential or parallel. The failure is still visible:
    it's logged by the caller via _report_task_result and never silently
    dropped."""
    try:
        summary = run_one_replication(
            task["scaled_params"], task["algorithm"],
            task["sim_months"], task["warmup_months"], task["seed"],
        )
    except Exception as exc:  # noqa: BLE001 -- intentionally broad: isolate one task's failure
        return {"label": task["label"], "summary": None, "error": f"{type(exc).__name__}: {exc}"}

    if summary is not None:
        summary.update({
            "era": task["era"], "archetype": task["archetype"],
            "replication": task["replication"], "seed": task["seed"],
        })
    return {"label": task["label"], "summary": summary, "error": None}


def run_all_experiments(params_path: str, output_dir: str,
                         scale_factor: float = SCALE_FACTOR,
                         sim_months: float = SIM_MONTHS,
                         warmup_months: float = WARMUP_MONTHS,
                         n_replications: int = N_REPLICATIONS,
                         base_seed: int = BASE_SEED,
                         algorithms: list = None,
                         n_workers: int = 1,
                         merge: bool = True):
    """
    merge: when True (default) and replications_raw.csv already exists in
    output_dir, this run's rows are combined with the existing ones
    (deduplicated on era/archetype/algorithm/replication, newest wins)
    before summary/ttests are regenerated -- so running with a narrower
    --algorithms list (e.g. just the CEL variants) never destroys results
    from a prior run of the others. Pass merge=False (CLI: --fresh) to
    overwrite output_dir from scratch instead.

    n_workers: 1 runs sequentially (default, safest for debugging). >1 runs
    replications in parallel across processes via ProcessPoolExecutor --
    each replication is fully independent (its own AsylumSimulator/env/RNG),
    so this parallelizes cleanly with no shared state. Given how long a
    single high-volume replication can take at real scale (tens of seconds
    to low minutes -- see redesign-decisions.md's timing note), using
    n_workers = number of available cores is strongly recommended for the
    full grid.

    Raises OrchestratorError for problems with the inputs themselves
    (missing/unreadable params file, malformed JSON, empty algorithms
    list, an output directory that can't be created/written to) -- these
    stop the run immediately with a clear message rather than partway
    through a multi-hour grid.
    """
    algorithms = algorithms or ALGORITHMS
    if not algorithms:
        raise OrchestratorError("--algorithms resolved to an empty list -- nothing to run.")

    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OrchestratorError(f"cannot create/write output directory {output_dir}: {exc}") from exc

    params_path = Path(params_path)
    if not params_path.exists():
        raise OrchestratorError(
            f"params file not found: {params_path} -- pass --params, or run "
            f"compute_parameter.py first to generate it."
        )
    try:
        with open(params_path) as f:
            all_params = json.load(f)
    except json.JSONDecodeError as exc:
        raise OrchestratorError(f"{params_path} is not valid JSON: {exc}") from exc

    tasks = []
    for era, era_params in all_params.items():
        for archetype, raw_params in era_params.items():
            try:
                scaled = _scaled_params(raw_params, scale_factor)
            except OrchestratorError as exc:
                raise OrchestratorError(f"{era}/{archetype}: {exc}") from exc
            if raw_params.get("_censored_sample"):
                log.warning(
                    "%s/%s is flagged _censored_sample=True -- service_time_*/"
                    "effective_service_time_*/grant_rate are calibrated from a "
                    "right-censored sample (see redesign-decisions.md item 1); "
                    "results below inherit that caveat.",
                    era, archetype,
                )
            for algo in algorithms:
                for rep in range(n_replications):
                    tasks.append({
                        "scaled_params": scaled, "algorithm": algo,
                        "sim_months": sim_months, "warmup_months": warmup_months,
                        "seed": base_seed + rep, "era": era, "archetype": archetype,
                        "replication": rep,
                        "label": f"{era}/{archetype}/{algo} rep{rep} (seed={base_seed + rep})",
                    })

    total_runs = len(tasks)
    replication_rows = []
    failed_labels = []
    t_start = time.time()

    if n_workers <= 1:
        for i, task in enumerate(tasks, 1):
            t0 = time.time()
            out = _run_task(task)
            elapsed = time.time() - t0
            _report_task_result(i, total_runs, out, elapsed)
            if out.get("error"):
                failed_labels.append(out["label"])
            elif out["summary"] is not None:
                replication_rows.append(out["summary"])
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        log.info("Running %d replications across %d worker processes...", total_runs, n_workers)
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_run_task, task): task for task in tasks}
                for i, future in enumerate(as_completed(futures), 1):
                    task = futures[future]
                    try:
                        out = future.result()
                    except Exception as exc:  # noqa: BLE001 -- a worker process itself crashed
                        out = {"label": task["label"], "summary": None,
                               "error": f"worker process crashed: {type(exc).__name__}: {exc}"}
                    _report_task_result(i, total_runs, out, None)
                    if out.get("error"):
                        failed_labels.append(out["label"])
                    elif out["summary"] is not None:
                        replication_rows.append(out["summary"])
        except KeyboardInterrupt:
            log.error("Interrupted -- shutting down worker pool and exiting without writing results.")
            raise

    log.info("Total wall-clock time: %.1fs", time.time() - t_start)
    if failed_labels:
        log.warning(
            "%d of %d replications failed and were excluded from the results "
            "(see the FAILED lines above for each error): %s",
            len(failed_labels), total_runs,
            ", ".join(failed_labels[:5]) + (", ..." if len(failed_labels) > 5 else ""),
        )

    new_df = pd.DataFrame(replication_rows)
    replications_path = output_dir / "replications_raw.csv"

    dedup_keys = ["era", "archetype", "algorithm", "replication"]
    if merge and replications_path.exists():
        try:
            old_df = pd.read_csv(replications_path)
        except pd.errors.EmptyDataError:
            # A previous run wrote zero completed replications, leaving a
            # header-less empty file -- nothing to merge, treat as absent.
            old_df = pd.DataFrame()
        except (OSError, pd.errors.ParserError) as exc:
            raise OrchestratorError(f"cannot read existing {replications_path}: {exc}") from exc
        replications_df = pd.concat([old_df, new_df], ignore_index=True)
        replications_df = replications_df.drop_duplicates(subset=dedup_keys, keep="last")
        log.info(
            "Merged with existing %s (%d prior rows + %d new -> %d total, deduplicated on %s)",
            replications_path, len(old_df), len(new_df), len(replications_df), dedup_keys,
        )
    else:
        replications_df = new_df

    try:
        replications_df.to_csv(replications_path, index=False)
    except OSError as exc:
        raise OrchestratorError(f"cannot write {replications_path}: {exc}") from exc
    log.info("Saved %d replication rows to %s", len(replications_df), replications_path)

    if replications_df.empty:
        log.warning("No completed replications -- skipping summary aggregation.")
        return replications_df, pd.DataFrame()

    agg_cols = [c for c in (_AGG_COLS + _CEL_AGG_COLS) if c in replications_df.columns]
    summary_df = (
        replications_df
        .groupby(["era", "archetype", "algorithm"])[agg_cols]
        .agg(["mean", "std"])
    )
    summary_df.columns = ["_".join(c) for c in summary_df.columns]
    summary_df = summary_df.reset_index()
    summary_path = output_dir / "simulation_summary.csv"
    try:
        summary_df.to_csv(summary_path, index=False)
    except OSError as exc:
        raise OrchestratorError(f"cannot write {summary_path}: {exc}") from exc
    log.info("Saved aggregated summary (%d reps/combo) to %s", n_replications, summary_path)

    _log_comparison_table(summary_df)
    return replications_df, summary_df


def _log_comparison_table(summary_df: pd.DataFrame):
    log.info("--- ALGORITHM COMPARISON: mean_total_time (months), mean +/- std across replications ---")
    if summary_df.empty:
        log.info("No results.")
        return
    for era in summary_df["era"].unique():
        log.info("ERA: %s", era)
        era_df = summary_df[summary_df["era"] == era]
        for archetype in era_df["archetype"].unique():
            log.info("  [%s]", archetype)
            arch_df = era_df[era_df["archetype"] == archetype]
            for _, row in arch_df.iterrows():
                log.info(
                    "    %-8s | wait=%6.1f+/-%.1fmo | backlog=%8.0f | throughput=%7.1f/mo | renege=%.3f",
                    row["algorithm"], row["mean_total_time_mean"], row["mean_total_time_std"],
                    row["backlog_total_mean"], row["throughput_mean"], row["renege_rate_obs_mean"],
                )


# ============================================================
# STATISTICAL COMPARISON (replication-level, not case-level)
# ============================================================

def run_pairwise_ttests(replications_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Pairwise t-tests on mean_total_time across REPLICATIONS (independent
    seeded runs), not across individual cases within one run -- see the
    module docstring's point 3 for why this replaces the original's
    within-run case-level comparison."""
    from scipy import stats

    output_dir = Path(output_dir)
    results = []

    for (era, archetype), grp in replications_df.groupby(["era", "archetype"]):
        algos = sorted(grp["algorithm"].unique())
        for i in range(len(algos)):
            for j in range(i + 1, len(algos)):
                a, b = algos[i], algos[j]
                a_vals = grp.loc[grp["algorithm"] == a, "mean_total_time"].values
                b_vals = grp.loc[grp["algorithm"] == b, "mean_total_time"].values
                if len(a_vals) < 2 or len(b_vals) < 2:
                    continue
                t_stat, p_val = stats.ttest_ind(a_vals, b_vals, equal_var=False)
                results.append({
                    "era": era,
                    "archetype": archetype,
                    "algo_A": a,
                    "algo_B": b,
                    "mean_A": round(float(np.mean(a_vals)), 2),
                    "mean_B": round(float(np.mean(b_vals)), 2),
                    "mean_diff_A_minus_B": round(float(np.mean(a_vals) - np.mean(b_vals)), 2),
                    "t_stat": round(float(t_stat), 3),
                    "p_value": round(float(p_val), 4),
                    "significant_p05": bool(p_val < 0.05),
                    "n_replications_A": len(a_vals),
                    "n_replications_B": len(b_vals),
                })

    ttest_df = pd.DataFrame(results)
    ttest_path = output_dir / "pairwise_ttests.csv"
    try:
        ttest_df.to_csv(ttest_path, index=False)
    except OSError as exc:
        raise OrchestratorError(f"cannot write {ttest_path}: {exc}") from exc

    log.info("--- PAIRWISE T-TESTS (replication-level means) ---")
    if not ttest_df.empty:
        log.info(
            "\n%s",
            ttest_df[["era", "archetype", "algo_A", "algo_B",
                      "mean_diff_A_minus_B", "p_value", "significant_p05"]].to_string(index=False),
        )
    log.info("Saved to %s", ttest_path)
    return ttest_df


# ============================================================
# MAIN
# ============================================================

def _configure_logging(level_name: str):
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Asylum Queue Simulator -- orchestration runner")
    parser.add_argument("--params", default="../data/sim_parameters_FINAL.json",
                        help="Path to sim_parameters_FINAL.json")
    parser.add_argument("--output", default="../data/sim_results/",
                        help="Output directory for results")
    parser.add_argument("--scale", type=float, default=SCALE_FACTOR,
                        help="Applied to arrival_rate_lambda AND n_officers/n_judges (default 0.30)")
    parser.add_argument("--months", type=float, default=SIM_MONTHS,
                        help="Simulation horizon in months (default 100)")
    parser.add_argument("--warmup", type=float, default=WARMUP_MONTHS,
                        help="Warm-up months excluded from stats (default 20)")
    parser.add_argument("--replications", type=int, default=N_REPLICATIONS,
                        help="Independent seeded runs per combo (default 15)")
    parser.add_argument("--seed", type=int, default=BASE_SEED,
                        help="Base seed; replications use seed, seed+1, ... (default 42)")
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS,
                        help="Algorithms to run (default: FIFO LIFO PRIORITY). "
                             "Also accepts CEL_balanced / CEL_broad -- e.g. "
                             "--algorithms CEL_balanced CEL_broad to run only "
                             "the two CEL variants without redoing the others.")
    parser.add_argument("--no-ttest", action="store_true",
                        help="Skip pairwise t-tests")
    parser.add_argument("--fresh", action="store_true",
                        help="Overwrite --output from scratch instead of merging "
                             "with any existing replications_raw.csv there "
                             "(default: merge, so a narrower --algorithms run "
                             "never destroys prior results for other algorithms)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes for replications (default 1 = "
                             "sequential). Each replication is fully independent, so "
                             "set this to your CPU count for the full grid -- a single "
                             "high-volume replication at real scale can take tens of "
                             "seconds to low minutes.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default INFO).")
    args = parser.parse_args()

    _configure_logging(args.log_level)

    log.info("ASYLUM QUEUE SIMULATOR -- ORCHESTRATION RUNNER")
    log.info("Scale factor:  %s (applied to arrivals AND server counts)", args.scale)
    log.info("Sim horizon:   %s months (warm-up: first %s excluded from stats)", args.months, args.warmup)
    log.info("Replications:  %s per (era, archetype, algorithm), base seed %s", args.replications, args.seed)
    log.info("Algorithms:    %s", args.algorithms)
    log.info("Workers:       %s", args.workers)

    try:
        replications_df, summary_df = run_all_experiments(
            params_path=args.params,
            output_dir=args.output,
            scale_factor=args.scale,
            sim_months=args.months,
            warmup_months=args.warmup,
            n_replications=args.replications,
            base_seed=args.seed,
            algorithms=args.algorithms,
            n_workers=args.workers,
            merge=not args.fresh,
        )
    except OrchestratorError as exc:
        log.error("Run aborted: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        log.error("Interrupted by user -- no results were written for the in-progress grid.")
        sys.exit(130)

    if not args.no_ttest and not replications_df.empty:
        try:
            run_pairwise_ttests(replications_df, args.output)
        except ImportError:
            log.warning("scipy not installed -- skipping t-tests. pip install scipy")
        except OrchestratorError as exc:
            log.error("t-tests skipped: %s", exc)