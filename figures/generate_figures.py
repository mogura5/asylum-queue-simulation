"""
generate_figures.py -- publication-quality figures from the orchestrator's
real output files (simulation_summary.csv / replications_raw.csv), rather
than a hand-pasted CSV literal.

WHY THIS EXISTS (vs. capacity_figure.py's approach):
capacity_figure.py embeds a fixed printed table as a string literal and
plots exactly that. That's fine for a one-off exploratory plot, but it
means every time the underlying results change (a rerun, a bug fix like
the mean_wait_s2 correction, a new algorithm), someone has to hand-copy a
new table into the script. This script instead reads the orchestrator's
actual CSV outputs directly, so regenerating every figure after a rerun is
one command with no copy-pasting -- the kind of small reproducibility
investment that matters once a pipeline has been rerun as many times as
this one has (see redesign-decisions.md's run history).

This file lives in figures/ (run it from there, or from the repo root as
`python3 figures/generate_figures.py ...` -- it has no import dependency
on the rest of the package, only pandas/matplotlib/seaborn, so its
location doesn't affect anything else).

USAGE (from the figures/ directory):
    python3 generate_figures.py --input ../data/sim_results --output .

Expects, in --input:
    simulation_summary.csv   (one row per era/archetype/algorithm, with
                               *_mean and *_std columns -- produced by
                               orchestrator.py::run_all_experiments())
    replications_raw.csv     (one row per replication -- used only for the
                               completion-rate/backlog-fraction panel, which
                               needs n_completed_post_warmup per replication)

Both are read defensively: a column that doesn't exist yet (e.g. this was
run before extending _summarize() with backlog-age tracking) causes that
one figure to be skipped with a warning, not a crash -- so this script
keeps working across every stage of the pipeline's evolution rather than
needing to be re-edited each time a new metric is added upstream.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")

ALGO_ORDER = ["FIFO", "LIFO", "PRIORITY", "CEL_balanced", "CEL_broad"]
ALGO_COLORS = {
    "FIFO": "#4C72B0",
    "LIFO": "#DD8452",
    "PRIORITY": "#55A868",
    "CEL_balanced": "#8172B2",
    "CEL_broad": "#C44E52",
}
ERA_ORDER = ["pre_covid_2019", "post_covid_2024"]
ARCHETYPE_ORDER = ["high_volume", "low_volume"]


def load_results(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load simulation_summary.csv (required) and replications_raw.csv
    (optional -- only needed for the completion-rate panel)."""
    summary_path = input_dir / "simulation_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} not found -- run orchestrator.py first, or point "
            f"--input at the directory passed as its --output."
        )
    summary = pd.read_csv(summary_path)

    raw_path = input_dir / "replications_raw.csv"
    raw = pd.read_csv(raw_path) if raw_path.exists() else None
    if raw is None:
        log.warning(
            "%s not found -- skipping the per-replication completion-rate panel.",
            raw_path,
        )
    return summary, raw


def _archetype_label(era: str, archetype: str) -> str:
    era_label = "Pre-COVID '19" if era == "pre_covid_2019" else "Post-COVID '24"
    vol_label = "High-volume" if archetype == "high_volume" else "Low-volume"
    return f"{era_label}\n{vol_label}"


def _ordered_present(values: list[str], order: list[str]) -> list[str]:
    """Filter `order` down to values actually present in the data, preserving
    the canonical order -- so a run missing CEL still plots cleanly."""
    present = set(values)
    return [v for v in order if v in present]


def plot_headline_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    """4-panel figure: mean wait, end backlog, throughput, renege rate --
    the same headline numbers tabulated by hand in redesign-decisions.md,
    now generated directly from the data with 95% CI error bars instead of
    point estimates alone."""
    df = summary.copy()
    df["combo"] = [
        _archetype_label(e, a) for e, a in zip(df["era"], df["archetype"])
    ]
    combo_order = [
        _archetype_label(e, a)
        for e in ERA_ORDER
        for a in ARCHETYPE_ORDER
        if not df[(df.era == e) & (df.archetype == a)].empty
    ]
    algos = _ordered_present(df["algorithm"].unique().tolist(), ALGO_ORDER)

    panels = [
        ("mean_total_time_mean", "mean_total_time_std", "Mean total wait (months)"),
        ("backlog_total_mean", None, "End-of-run backlog (cases)"),
        ("throughput_mean", "throughput_std", "Throughput (cases/month)"),
        ("renege_rate_obs_mean", "renege_rate_obs_std", "Observed renege rate"),
    ]
    panels = [p for p in panels if p[0] in df.columns]
    if not panels:
        log.warning("None of the expected headline columns found -- skipping headline figure.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    n_algos = len(algos)
    bar_width = 0.8 / max(n_algos, 1)

    for ax, (mean_col, std_col, ylabel) in zip(axes, panels):
        for i, algo in enumerate(algos):
            sub = df[df["algorithm"] == algo].set_index("combo").reindex(combo_order)
            x = [j + i * bar_width for j in range(len(combo_order))]
            yerr = sub[std_col] if std_col and std_col in sub.columns else None
            ax.bar(
                x, sub[mean_col], width=bar_width, label=algo,
                color=ALGO_COLORS.get(algo), yerr=yerr, capsize=3,
            )
        ax.set_xticks([j + bar_width * (n_algos - 1) / 2 for j in range(len(combo_order))])
        ax.set_xticklabels(combo_order, fontsize=9)
        ax.set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_algos, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        "Headline comparison across scheduling algorithms\n"
        "(error bars: ±1 std across replications, where available)",
        fontsize=14, fontweight="bold",
    )
    plt.subplots_adjust(bottom=0.14, hspace=0.35, wspace=0.3)
    out = output_dir / "fig_headline_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out)


def plot_lifo_wait_paradox(summary: pd.DataFrame, raw: pd.DataFrame | None, output_dir: Path) -> None:
    """Side-by-side mean-wait vs. backlog-fraction panel -- the figure that
    makes the LIFO wait-time paradox legible at a glance: LIFO's mean wait
    looks best, but its backlog share is statistically the same as
    FIFO/PRIORITY's, because 'mean wait' is conditioned on completion (see
    redesign-decisions.md, 'Key finding: the LIFO wait-time paradox')."""
    needed = {"mean_total_time_mean", "backlog_total_mean", "n_completed_post_warmup_mean"}
    if not needed.issubset(summary.columns):
        log.warning("Missing columns for LIFO-paradox figure (%s) -- skipping.",
                    needed - set(summary.columns))
        return

    df = summary.copy()
    df["backlog_fraction"] = df["backlog_total_mean"] / (
        df["backlog_total_mean"] + df["n_completed_post_warmup_mean"]
    )
    df["combo"] = [_archetype_label(e, a) for e, a in zip(df["era"], df["archetype"])]
    combo_order = [
        _archetype_label(e, a)
        for e in ERA_ORDER for a in ARCHETYPE_ORDER
        if not df[(df.era == e) & (df.archetype == a)].empty
    ]
    algos = _ordered_present(df["algorithm"].unique().tolist(), ALGO_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (col, title) in zip(
        axes, [("mean_total_time_mean", "Mean wait among completers (months)"),
               ("backlog_fraction", "Fraction of arrivals still backlogged")]
    ):
        for algo in algos:
            sub = df[df["algorithm"] == algo].set_index("combo").reindex(combo_order)
            ax.plot(combo_order, sub[col], marker="o", label=algo, color=ALGO_COLORS.get(algo))
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", labelrotation=0, labelsize=8)

    axes[0].set_ylabel("Months")
    axes[1].set_ylabel("Fraction")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(algos), bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "The LIFO paradox: lowest mean wait, same backlog\n"
        "(low mean wait reflects who completes, not overall system relief)",
        fontsize=13, fontweight="bold",
    )
    plt.subplots_adjust(bottom=0.28, wspace=0.25)
    out = output_dir / "fig_lifo_paradox.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out)


def plot_cel_welfare_targeting(summary: pd.DataFrame, output_dir: Path) -> None:
    """CEL-specific welfare metrics: does the lottery actually deliver fast
    relief to the expedited subset without making the regular queue
    meaningfully worse? Skips cleanly if no CEL rows are present."""
    cel = summary[summary["algorithm"].astype(str).str.startswith("CEL_")].copy()
    needed = {"mean_wait_expedited_mean", "mean_wait_regular_mean", "expedited_rate_mean"}
    if cel.empty or not needed.issubset(cel.columns):
        log.warning("No CEL rows or missing welfare columns -- skipping CEL welfare figure.")
        return

    cel["combo"] = [_archetype_label(e, a) for e, a in zip(cel["era"], cel["archetype"])]
    combo_order = [
        _archetype_label(e, a)
        for e in ERA_ORDER for a in ARCHETYPE_ORDER
        if not cel[(cel.era == e) & (cel.archetype == a)].empty
    ]
    configs = _ordered_present(cel["algorithm"].unique().tolist(), ["CEL_balanced", "CEL_broad"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.35
    for i, config in enumerate(configs):
        sub = cel[cel["algorithm"] == config].set_index("combo").reindex(combo_order)
        x = [j + i * width for j in range(len(combo_order))]
        axes[0].bar([j - width / 2 for j in x], sub["mean_wait_expedited_mean"],
                    width=width / 2, label=f"{config} (expedited)",
                    color=ALGO_COLORS.get(config), alpha=0.9)
        axes[0].bar([j + width / 2 for j in x], sub["mean_wait_regular_mean"],
                    width=width / 2, label=f"{config} (regular)",
                    color=ALGO_COLORS.get(config), alpha=0.45)
        axes[1].plot(combo_order, sub["expedited_rate_mean"] * 100, marker="o", label=config,
                     color=ALGO_COLORS.get(config))

    axes[0].set_xticks(range(len(combo_order)))
    axes[0].set_xticklabels(combo_order, fontsize=8)
    axes[0].set_ylabel("Mean wait (months)")
    axes[0].set_title("Expedited vs. regular-track wait, by CEL config", fontsize=11)
    axes[0].legend(fontsize=7)

    axes[1].set_ylabel("% of screened cases expedited")
    axes[1].set_title("Lottery expedited rate", fontsize=11)
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", labelsize=8)

    fig.suptitle("CEL welfare-targeting: is the lottery delivering fast relief?",
                 fontsize=13, fontweight="bold")
    plt.subplots_adjust(wspace=0.3, top=0.85)
    out = output_dir / "fig_cel_welfare_targeting.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True,
                         help="Directory containing simulation_summary.csv (and, optionally, replications_raw.csv) -- the same path passed as orchestrator.py's --output.")
    parser.add_argument("--output", type=Path, default=Path("figures"),
                         help="Directory to write PNG figures into (created if missing). Default: ./figures")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    summary, raw = load_results(args.input)

    plot_headline_comparison(summary, args.output)
    plot_lifo_wait_paradox(summary, raw, args.output)
    plot_cel_welfare_targeting(summary, args.output)

    log.info("Done. Figures written to %s", args.output.resolve())


if __name__ == "__main__":
    main()