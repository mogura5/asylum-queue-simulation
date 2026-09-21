"""
verify_cel_stage2_idle.py -- diagnostic for the "CEL Stage 2 throughput is
measurably suppressed" puzzle logged in redesign-decisions.md ("A real,
unresolved puzzle: CEL's non-expedited Stage 2 throughput is measurably
suppressed, not just diluted"). This is diagnostic tooling, not part of the
production pipeline (orchestrator.py) -- it does not write into
data/sim_results/ and is not required before reporting the existing
throughput/backlog numbers.

Implements the two lowest-effort verification steps recommended in that
log entry:

  (a) IDLE-TIME INSTRUMENTATION ("option 2" in the follow-up discussion).
      stage.py now tracks Stage.idle_time_total -- cumulative simulated
      time every worker in a stage spent blocked waiting for a case,
      summed across all of that stage's workers. Runs a CEL replication
      and a FIFO replication at IDENTICAL scaled params/capacity/seed and
      reports Stage 2's idle fraction for each. If the lottery's monthly
      batch-and-release pattern is starving Stage 2 of available work
      during the holding window, CEL's Stage 2 idle fraction should be
      measurably higher than FIFO's even though both archetypes are
      nominally overloaded (rho >> 1, so a work-conserving queue should
      almost never see its workers idle).

  (b) PROFILING PASS ("option 4"). Runs cProfile on the same CEL
      replication and prints the top functions by cumulative time, to
      confirm the "lottery winners generate no Stage 2 event-stream
      overhead" hypothesis offered for why the CEL grid ran ~46x faster
      than the linear estimate from the FIFO/LIFO/PRIORITY grid (see
      "CEL full-scale run results" -> "Runtime came in far under the
      estimate").

USAGE:
    python3 diagnostics/verify_cel_stage2_idle.py \\
        --params ../data/sim_parameters_FINAL.json \\
        --era post_covid_2024 --archetype high_volume \\
        --scale 0.10 --months 100 --warmup 20 --seed 42 \\
        --cel-config balanced

Defaults to a small synthetic params dict (no --params needed) so this can
be run standalone to sanity-check the instrumentation itself before
pointing it at the real calibrated file.
"""

import argparse
import cProfile
import io
import json
import logging
import pstats
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "src"),
)

from simulator import AsylumSimulator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("verify_cel_stage2_idle")

_SYNTHETIC_PARAMS = {
    "arrival_rate_lambda": 500.0,
    "service_time_mean": 20.0,
    "service_time_std": 10.0,
    "effective_service_time_s1": 0.05,
    "effective_service_time_s2": 0.045,
    "grant_rate": 0.10,
    "appeal_rate": 0.30,
    "p_rep": 0.6,
    "p_detained": 0.1,
    "p_renege": 0.30,
    "p_criminal": 0.05,
    "n_officers": 40,
    "n_judges": 35,
}


def _load_scaled_params(args) -> dict:
    if args.params is None:
        log.info("No --params given -- using a small synthetic overloaded scenario.")
        return dict(_SYNTHETIC_PARAMS)

    with open(args.params) as f:
        all_params = json.load(f)
    try:
        raw = all_params[args.era][args.archetype]
    except KeyError as exc:
        raise SystemExit(
            f"era/archetype not found in {args.params}: {exc}. "
            f"Available eras: {list(all_params)}"
        ) from exc

    scaled = dict(raw)
    scaled["arrival_rate_lambda"] = raw["arrival_rate_lambda"] * args.scale
    scaled["n_officers"] = max(1, round(raw["n_officers"] * args.scale))
    scaled["n_judges"] = max(1, round(raw["n_judges"] * args.scale))
    return scaled


def _run_and_report(algorithm: str, scaled_params: dict, months: float,
                     warmup: float, seed: int) -> dict:
    sim = AsylumSimulator(
        params=scaled_params, algorithm=algorithm,
        n_officers=scaled_params["n_officers"], n_judges=scaled_params["n_judges"],
        sim_months=months, seed=seed,
    )
    result = sim.run()

    n_judges = scaled_params["n_judges"]
    stage2_idle_frac = result["stage2_idle_time"] / (n_judges * months) if n_judges else 0.0
    stage1_idle_frac = result["stage1_idle_time"] / (scaled_params["n_officers"] * months)

    log.info(
        "%-14s n_completed=%6d  stage1_idle=%.4f (frac of officer-months)  "
        "stage2_idle=%.4f (frac of judge-months)",
        algorithm, result.get("n_completed", 0), stage1_idle_frac, stage2_idle_frac,
    )
    return {"result": result, "stage2_idle_frac": stage2_idle_frac, "stage1_idle_frac": stage1_idle_frac}


def compare_idle_time(scaled_params: dict, cel_algorithm: str, months: float,
                       warmup: float, seed: int) -> None:
    log.info("--- (a) Stage idle-time comparison: FIFO vs. %s, same params/seed ---", cel_algorithm)
    fifo = _run_and_report("FIFO", scaled_params, months, warmup, seed)
    cel = _run_and_report(cel_algorithm, scaled_params, months, warmup, seed)

    delta = cel["stage2_idle_frac"] - fifo["stage2_idle_frac"]
    log.info(
        "Stage 2 idle-fraction delta (%s - FIFO): %+.4f -- %s",
        cel_algorithm, delta,
        "supports the batching-starvation hypothesis (CEL's Stage 2 idles more)"
        if delta > 0.001 else
        "does NOT show meaningfully more Stage 2 idling under CEL -- "
        "hypothesis not supported by this run; consider a shorter lottery "
        "cycle test or re-examining _advance_to_stage2/_lottery_draw_process.",
    )


def profile_cel_replication(scaled_params: dict, cel_algorithm: str,
                             months: float, seed: int, top_n: int = 20) -> None:
    log.info("--- (b) cProfile pass on one %s replication ---", cel_algorithm)

    def _run():
        sim = AsylumSimulator(
            params=scaled_params, algorithm=cel_algorithm,
            n_officers=scaled_params["n_officers"], n_judges=scaled_params["n_judges"],
            sim_months=months, seed=seed,
        )
        sim.run()

    profiler = cProfile.Profile()
    profiler.enable()
    _run()
    profiler.disable()

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(top_n)
    log.info("\n%s", buf.getvalue())
    log.info(
        "Check the above for how much cumulative time is spent inside "
        "Stage._worker / Stage.submit for stage2 vs. stage1, and inside "
        "_lottery_draw_process -- the runtime-estimate finding in "
        "redesign-decisions.md predicts a disproportionately small share "
        "of total time attributable to Stage 2 event processing under CEL, "
        "since lottery winners never call Stage2.submit()."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", default=None,
                         help="Path to sim_parameters_FINAL.json. Omit to use a small built-in synthetic overloaded scenario.")
    parser.add_argument("--era", default=None, help="Era key (required if --params is given).")
    parser.add_argument("--archetype", default=None, help="Archetype key (required if --params is given).")
    parser.add_argument("--scale", type=float, default=0.10, help="Scale factor applied to arrivals/capacity (default 0.10).")
    parser.add_argument("--months", type=float, default=100.0, help="Simulation horizon (default 100).")
    parser.add_argument("--warmup", type=float, default=20.0, help="Warm-up months (default 20; informational only here).")
    parser.add_argument("--seed", type=int, default=42, help="Seed shared by both replications (default 42).")
    parser.add_argument("--cel-config", default="balanced", choices=["balanced", "broad"],
                         help="Which CEL variant to compare against FIFO (default balanced).")
    parser.add_argument("--skip-profile", action="store_true", help="Run only the idle-time comparison, skip cProfile.")
    args = parser.parse_args()

    if args.params and (not args.era or not args.archetype):
        parser.error("--era and --archetype are required when --params is given.")

    scaled_params = _load_scaled_params(args)
    cel_algorithm = f"CEL_{args.cel_config}"

    compare_idle_time(scaled_params, cel_algorithm, args.months, args.warmup, args.seed)
    if not args.skip_profile:
        profile_cel_replication(scaled_params, cel_algorithm, args.months, args.seed)


if __name__ == "__main__":
    main()