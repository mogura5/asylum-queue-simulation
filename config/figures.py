"""
PAPER FIGURES — ASYLUM QUEUE SIMULATION
========================================
Generates all figures and tables for the paper, organized in sections:

  Section 1: Dataset descriptives (from mock_data_FINAL.csv)
  Section 2: Simulation results — wait time & backlog
  Section 3: Statistical tests (RQ2)
  Section 4: Surge comparison (RQ3)
  Section 5: Economic impact (RQ4/RQ5)

Run from the src/calibration directory:
    python3 figures.py

Outputs saved to: ../../figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
DATA_CSV    = "../data/mock_data_FINAL.csv"
RESULTS_DIR = "../data/sim_results"
PARAMS_JSON = "../data/sim_parameters_FINAL.json"
FIG_DIR     = Path("../figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALGO_COLORS = {"FIFO": "#2166ac", "LIFO": "#d6604d", "PRIORITY": "#4dac26"}
ALGO_ORDER  = ["FIFO", "LIFO", "PRIORITY"]
ERA_LABELS  = {"pre_covid_2019": "Pre-COVID (2019)", "post_covid_2024": "Post-COVID (2024)"}
ARCH_SHORT  = {
    "high_vol_low_grant":  "Group 1 (HV-Low)",
    "high_vol_high_grant": "Group 2 (HV-High)",
    "low_vol_low_grant":   "Group 3 (LV-Low)",
    "low_vol_high_grant":  "Group 4 (LV-High)",
}
ARCH_SHORT_FLAT = ARCH_SHORT   # flat labels, no newlines

# Optimal Priority configs from scoring experiment (era, archetype) -> (label, wait, renege)
SCORING_BEST = {
    ("pre_covid_2019","high_vol_low_grant"):  ("Opt-Priority det=3,age=12",  103.8, 0.059),
    ("pre_covid_2019","high_vol_high_grant"): ("Opt-Priority det=1,age=3",   115.4, 0.677),
    ("pre_covid_2019","low_vol_low_grant"):   ("Opt-Priority det=0,age=3",   114.3, 0.669),
    ("pre_covid_2019","low_vol_high_grant"):  ("Opt-Priority det=5,age=6",    95.2, 0.083),
    ("post_covid_2024","high_vol_low_grant"): ("Opt-Priority det=5,age=12",   97.2, 0.115),
    ("post_covid_2024","high_vol_high_grant"):("Opt-Priority det=0,age=3",   110.8, 0.708),
    ("post_covid_2024","low_vol_low_grant"):  ("Opt-Priority det=0,age=3",   110.6, 0.718),
    ("post_covid_2024","low_vol_high_grant"): ("Opt-Priority det=0,age=12",  111.0, 0.690),
}

# CEL paths and colors
CEL_DIR     = Path("../data/cel_results")
SCORING_DIR = Path("../data")

CEL_COLORS = {
    "BASELINE_FIFO":    "#2166ac",
    "CEL_strict":       "#fee090",
    "CEL_econ_focused": "#fdae61",
    "CEL_balanced":     "#f46d43",
    "CEL_broad":        "#d73027",
    "CEL_comp_only":    "#abdda4",
    "CEL_econ_only":    "#2ca25f",
}
CEL_ORDER = ["BASELINE_FIFO","CEL_strict","CEL_econ_focused","CEL_balanced",
             "CEL_broad","CEL_comp_only","CEL_econ_only"]
CEL_LABELS = {
    "BASELINE_FIFO":    "FIFO baseline",
    "CEL_strict":       "Strict (comp=0, econ>=8, p=10%)",
    "CEL_econ_focused": "Econ-focused (comp<=1.5, econ>=6.67, p=20%)",
    "CEL_balanced":     "Balanced (comp<=3, econ>=5.33, p=30%)",
    "CEL_broad":        "Broad (comp<=4.5, econ>=4, p=40%)",
    "CEL_comp_only":    "Comp-only (comp<=1.5, p=15%)",
    "CEL_econ_only":    "Econ-only (econ>=6.67, p=15%)",
}

def save(name):
    path = FIG_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATASET DESCRIPTIVES
# ═══════════════════════════════════════════════════════════════════════════════

def load_case_data():
    print("Loading case data...")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    df["arrival_date"]    = pd.to_datetime(df["arrival_date"],    errors="coerce")
    df["completion_date"] = pd.to_datetime(df["completion_date"], errors="coerce")
    df["arrival_year"]    = df["arrival_date"].dt.year
    return df

def fig1_annual_filings(df):
    """Annual asylum case filings 2005–2024."""
    print("Fig 1: Annual filings...")
    counts = (df[df["arrival_year"].between(2005, 2024)]
              .groupby("arrival_year").size().reset_index(name="n"))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(counts["arrival_year"], counts["n"] / 1000,
           color="#4393c3", edgecolor="white", linewidth=0.4)
    ax.axvline(2020, color="black", linestyle="--", linewidth=1, label="COVID-19 (2020)")
    ax.set_xlabel("Year of Filing")
    ax.set_ylabel("Cases Filed (thousands)")
    ax.set_title("Annual Asylum Case Filings, 2005–2024")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
    save("fig1_annual_filings")

def fig2_processing_time_dist(df):
    """Processing time distribution by court cluster."""
    print("Fig 2: Processing time distribution...")
    completed = df[~df["is_pending"] & df["processing_months"].notna()
                   & df["court_cluster"].notna() & (df["court_cluster"] != -1)]
    cluster_names = {
        c: ARCH_SHORT.get(n, str(c))
        for c, n in completed.groupby("court_cluster")["court_cluster"].first().items()
    }
    # Map cluster id to archetype name via grant rate ordering
    # (replicating the archetype_map logic)
    summary = (completed.groupby("court_cluster")
               .agg(n=("idncase","count"), gr=("asylum_granted","mean"))
               .sort_values(["gr","n"], ascending=[True,False]))
    labels = ["HV-Low","HV-High","LV-Low","LV-High"]
    id_to_label = dict(zip(summary.index, labels))

    fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=False)
    for ax, (cid, grp) in zip(axes, completed.groupby("court_cluster")):
        label = id_to_label.get(cid, str(cid))
        vals = grp["processing_months"].clip(0, 150)
        ax.hist(vals, bins=40, color="#4393c3", edgecolor="white", linewidth=0.3)
        ax.axvline(vals.mean(),   color="#d6604d", linestyle="-",  linewidth=1.5,
                   label=f"Mean {vals.mean():.0f}mo")
        ax.axvline(vals.median(), color="#333333", linestyle="--", linewidth=1.5,
                   label=f"Median {vals.median():.0f}mo")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Months")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Number of Cases")
    fig.suptitle("Processing Time Distribution by Court Archetype (Completed Cases)",
                 fontsize=11, y=1.02)
    save("fig2_processing_time_dist")

def fig3_grant_rate_by_cluster(df):
    """Grant rate and representation rate by court cluster."""
    print("Fig 3: Grant rates by cluster...")
    valid = df[df["court_cluster"] != -1]
    summary = (valid.groupby("court_cluster")
               .agg(grant_rate=("asylum_granted","mean"),
                    rep_rate=("has_representation","mean"),
                    detained_rate=("is_detained","mean"),
                    n=("idncase","count"))
               .sort_values(["grant_rate","n"], ascending=[True,False]))
    labels = ["HV-Low","HV-High","LV-Low","LV-High"][:len(summary)]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.25
    ax.bar(x - width, summary["grant_rate"],    width, label="Grant Rate",       color="#4393c3")
    ax.bar(x,          summary["rep_rate"],      width, label="Representation",   color="#92c5de")
    ax.bar(x + width,  summary["detained_rate"], width, label="Detained",         color="#d6604d")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Proportion")
    ax.set_title("Grant Rate, Representation, and Detention by Court Archetype")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    save("fig3_grant_rate_by_cluster")

def fig4_pending_vs_completed(df):
    """Stacked bar: pending vs completed cases by year."""
    print("Fig 4: Pending vs completed...")
    yr = (df[df["arrival_year"].between(2000, 2024)]
          .groupby(["arrival_year","is_pending"]).size()
          .unstack(fill_value=0).rename(columns={False:"Completed", True:"Pending"}))
    fig, ax = plt.subplots(figsize=(10, 4))
    yr[["Completed","Pending"]].div(1000).plot(
        kind="bar", stacked=True, ax=ax,
        color=["#4393c3","#d6604d"], edgecolor="white", linewidth=0.3, width=0.8)
    ax.set_xlabel("Filing Year")
    ax.set_ylabel("Cases (thousands)")
    ax.set_title("Asylum Cases by Filing Year: Completed vs. Still Pending")
    ax.set_xticklabels(yr.index, rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
    ax.legend()
    save("fig4_pending_vs_completed")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SIMULATION RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def load_summary():
    return pd.read_csv(f"{RESULTS_DIR}/simulation_summary.csv")

def load_cases_all():
    dfs = []
    for f in Path(RESULTS_DIR).glob("cases_*.csv"):
        parts = f.stem.split("_", 1)[1].rsplit("_", 1)
        if len(parts) == 2:
            rest, algo = parts
            era_arch = rest.rsplit("_", 1)
            df = pd.read_csv(f)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_queue_logs():
    dfs = {}
    for f in Path(RESULTS_DIR).glob("queue_log_*.csv"):
        # queue_log_{era}_{archetype}_{algo}.csv
        name = f.stem[len("queue_log_"):]
        df = pd.read_csv(f)
        dfs[name] = df
    return dfs

def fig5_mean_wait_grouped(summary):
    """Grouped bar: mean total wait time by algorithm, archetype, era."""
    print("Fig 5: Mean wait time grouped bar...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    eras = ["pre_covid_2019", "post_covid_2024"]
    archetypes = ["high_vol_low_grant","high_vol_high_grant","low_vol_low_grant","low_vol_high_grant"]
    x = np.arange(len(archetypes))
    width = 0.25

    for ax, era in zip(axes, eras):
        era_df = summary[summary["era"] == era]
        for i, algo in enumerate(ALGO_ORDER):
            vals = [era_df[(era_df["archetype"]==a) & (era_df["algorithm"]==algo)
                           ]["mean_total_time"].values[0]
                    if len(era_df[(era_df["archetype"]==a) & (era_df["algorithm"]==algo)]) > 0
                    else 0
                    for a in archetypes]
            ax.bar(x + (i - 1) * width, vals, width, label=algo,
                   color=ALGO_COLORS[algo], edgecolor="white", linewidth=0.4)
        ax.set_title(ERA_LABELS[era], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_SHORT[a] for a in archetypes], fontsize=9)
        ax.set_xlabel("Court Archetype")
        ax.set_ylabel("Mean Total Wait Time (months)")

    axes[0].legend(title="Algorithm")
    fig.suptitle("Mean Total Wait Time by Algorithm and Court Archetype", fontsize=12)
    save("fig5_mean_wait_grouped")

def fig6_backlog_comparison(summary):
    """Backlog size at end of simulation — algorithm × archetype × era."""
    print("Fig 6: Backlog comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    eras = ["pre_covid_2019", "post_covid_2024"]
    archetypes = ["high_vol_low_grant","high_vol_high_grant","low_vol_low_grant","low_vol_high_grant"]
    x = np.arange(len(archetypes))
    width = 0.25

    for ax, era in zip(axes, eras):
        era_df = summary[summary["era"] == era]
        for i, algo in enumerate(ALGO_ORDER):
            vals = [era_df[(era_df["archetype"]==a) & (era_df["algorithm"]==algo)
                           ]["backlog_total"].values[0] / 1000
                    if len(era_df[(era_df["archetype"]==a) & (era_df["algorithm"]==algo)]) > 0
                    else 0
                    for a in archetypes]
            ax.bar(x + (i - 1) * width, vals, width, label=algo,
                   color=ALGO_COLORS[algo], edgecolor="white", linewidth=0.4)
        ax.set_title(ERA_LABELS[era], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_SHORT[a] for a in archetypes], fontsize=9)
        ax.set_xlabel("Court Archetype")
        ax.set_ylabel("Residual Backlog (thousands of cases)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}k"))

    axes[0].legend(title="Algorithm")
    fig.suptitle("Residual Backlog at End of Simulation (200 months)", fontsize=12)
    save("fig6_backlog_comparison")

def fig7_queue_length_over_time(qlogs):
    """Queue length over time for one representative archetype, pre-COVID."""
    print("Fig 7: Queue length over time...")
    target_arch = "high_vol_low_grant"
    target_era  = "pre_covid_2019"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, algo in zip(axes, ALGO_ORDER):
        key = f"{target_era}_{target_arch}_{algo}"
        if key not in qlogs:
            ax.set_title(f"{algo}\n(no data)")
            continue
        df = qlogs[key]
        total = df["s1_len"] + df["s2_len"]
        ax.plot(df["time"], total / 1000, color=ALGO_COLORS[algo], linewidth=1)
        ax.fill_between(df["time"], total / 1000, alpha=0.15, color=ALGO_COLORS[algo])
        ax.set_title(algo, fontsize=11)
        ax.set_xlabel("Simulation Month")

    axes[0].set_ylabel("Queue Length (thousands)")
    fig.suptitle(f"Queue Length Over Time — High-Volume, Low-Grant Courts (Pre-COVID)",
                 fontsize=11)
    save("fig7_queue_length_over_time")

def fig8_renege_rate(summary):
    """Reneging (missed hearing) rate by algorithm — Priority's key advantage."""
    print("Fig 8: Renege rate...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    archetypes = ["high_vol_low_grant","high_vol_high_grant","low_vol_low_grant","low_vol_high_grant"]
    x = np.arange(len(archetypes))
    width = 0.25

    for ax, era in zip(axes, ["pre_covid_2019","post_covid_2024"]):
        era_df = summary[summary["era"] == era]
        for i, algo in enumerate(ALGO_ORDER):
            vals = [era_df[(era_df["archetype"]==a) & (era_df["algorithm"]==algo)
                           ]["renege_rate_obs"].values[0]
                    if len(era_df[(era_df["archetype"]==a) & (era_df["algorithm"]==algo)]) > 0
                    else 0
                    for a in archetypes]
            ax.bar(x + (i - 1) * width, vals, width, label=algo,
                   color=ALGO_COLORS[algo], edgecolor="white", linewidth=0.4)
        ax.set_title(ERA_LABELS[era])
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_SHORT[a] for a in archetypes], fontsize=9)
        ax.set_ylabel("Reneging Rate")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    axes[0].legend(title="Algorithm")
    fig.suptitle("Reneging Rate (Missed Hearings) by Algorithm and Court Archetype", fontsize=12)
    save("fig8_renege_rate")

def fig9_wait_time_dist_boxplot(cases_df):
    """Box plot of total wait time distribution — FIFO vs LIFO vs PRIORITY."""
    print("Fig 9: Wait time distribution boxplot...")
    if cases_df.empty:
        print("  no cases data, skipping")
        return

    archetypes = ["high_vol_low_grant","high_vol_high_grant","low_vol_low_grant","low_vol_high_grant"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharey=False)

    for col, arch in enumerate(archetypes):
        for row, era in enumerate(["pre_covid_2019","post_covid_2024"]):
            ax = axes[row][col]
            data = cases_df[(cases_df["archetype"] == arch) & (cases_df["era"] == era)]
            groups = [data[data["algorithm"] == a]["total_time"].clip(0, 200).dropna().values
                      for a in ALGO_ORDER]
            bp = ax.boxplot(groups, patch_artist=True, notch=False,
                            medianprops=dict(color="black", linewidth=1.5),
                            whiskerprops=dict(linewidth=0.8),
                            flierprops=dict(marker=".", markersize=1, alpha=0.3))
            for patch, algo in zip(bp["boxes"], ALGO_ORDER):
                patch.set_facecolor(ALGO_COLORS[algo])
                patch.set_alpha(0.7)
            ax.set_xticklabels(ALGO_ORDER, fontsize=7)
            ax.set_ylabel("Total Wait (months)" if col == 0 else "")
            if row == 0:
                ax.set_title(ARCH_SHORT[arch], fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{ERA_LABELS[era]}\n\nTotal Wait (months)", fontsize=8)

    fig.suptitle("Distribution of Total Case Wait Time by Algorithm, Archetype, and Era",
                 fontsize=11, y=1.01)
    save("fig9_wait_time_boxplot")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def fig10_ttest_heatmap():
    """Heatmap of pairwise t-test mean differences and significance."""
    print("Fig 10: T-test heatmap...")
    ttest_path = f"{RESULTS_DIR}/pairwise_ttests.csv"
    if not os.path.exists(ttest_path):
        print("  no ttest file, skipping")
        return
    df = pd.read_csv(ttest_path)

    # Pivot mean_diff for FIFO vs LIFO (the most policy-relevant comparison)
    subset = df[df["algo_A"] == "FIFO"][df["algo_B"] == "LIFO"] if "FIFO" in df["algo_A"].values else df
    archetypes = ["high_vol_low_grant","high_vol_high_grant","low_vol_low_grant","low_vol_high_grant"]
    eras = ["pre_covid_2019","post_covid_2024"]

    matrix = np.zeros((len(eras), len(archetypes)))
    sig_matrix = np.zeros((len(eras), len(archetypes)), dtype=bool)

    for i, era in enumerate(eras):
        for j, arch in enumerate(archetypes):
            row = df[(df["era"] == era) & (df["archetype"] == arch) &
                     (df["algo_A"] == "FIFO") & (df["algo_B"] == "LIFO")]
            if not row.empty:
                matrix[i, j]     = row["mean_diff_A_minus_B"].values[0]
                sig_matrix[i, j] = row["significant_p05"].values[0]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-20, vmax=20, aspect="auto")
    ax.set_xticks(range(len(archetypes)))
    ax.set_xticklabels([ARCH_SHORT[a] for a in archetypes])
    ax.set_yticks(range(len(eras)))
    ax.set_yticklabels([ERA_LABELS[e] for e in eras])
    plt.colorbar(im, ax=ax, label="Mean Diff: FIFO − LIFO (months)")

    for i in range(len(eras)):
        for j in range(len(archetypes)):
            txt = f"{matrix[i,j]:.1f}"
            if sig_matrix[i, j]:
                txt += "*"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="white" if abs(matrix[i,j]) > 10 else "black")

    ax.set_title("FIFO − LIFO Mean Wait Difference (months) — * p < 0.05", fontsize=11)
    save("fig10_ttest_heatmap")

def fig11_algo_ranking_consistency():
    """Slope graph showing algorithm ranking is stable across eras (RQ3)."""
    print("Fig 11: Algorithm ranking stability...")
    summary = load_summary()
    archetypes = ["high_vol_low_grant","high_vol_high_grant","low_vol_low_grant","low_vol_high_grant"]

    fig, axes = plt.subplots(1, 4, figsize=(13, 5))
    for ax, arch in zip(axes, archetypes):
        for algo in ALGO_ORDER:
            vals = []
            for era in ["pre_covid_2019","post_covid_2024"]:
                row = summary[(summary["era"]==era) & (summary["archetype"]==arch) &
                              (summary["algorithm"]==algo)]
                vals.append(row["mean_total_time"].values[0] if not row.empty else np.nan)
            ax.plot([0, 1], vals, "o-", color=ALGO_COLORS[algo],
                    linewidth=2, markersize=6, label=algo)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre-COVID\n2019", "Post-COVID\n2024"], fontsize=8)
        ax.set_title(ARCH_SHORT[arch], fontsize=9)
        ax.set_ylabel("Mean Total Wait (months)" if arch == archetypes[0] else "")

    axes[-1].legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.suptitle("Algorithm Wait-Time Ranking Across Eras (RQ3: Stability Under Surge)",
                 fontsize=11)
    save("fig11_ranking_stability")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SURGE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def fig12_pre_post_wait_change(summary):
    """Percent change in wait time from pre- to post-COVID by algorithm."""
    print("Fig 12: Pre vs post surge wait change...")
    archetypes = ["high_vol_low_grant","high_vol_high_grant","low_vol_low_grant","low_vol_high_grant"]
    x = np.arange(len(archetypes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, algo in enumerate(ALGO_ORDER):
        pct_changes = []
        for arch in archetypes:
            pre  = summary[(summary["era"]=="pre_covid_2019")  & (summary["archetype"]==arch) &
                           (summary["algorithm"]==algo)]["mean_total_time"]
            post = summary[(summary["era"]=="post_covid_2024") & (summary["archetype"]==arch) &
                           (summary["algorithm"]==algo)]["mean_total_time"]
            if pre.empty or post.empty or pre.values[0] == 0:
                pct_changes.append(0)
            else:
                pct_changes.append((post.values[0] - pre.values[0]) / pre.values[0] * 100)
        ax.bar(x + (i-1)*width, pct_changes, width, label=algo,
               color=ALGO_COLORS[algo], edgecolor="white", linewidth=0.4)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_SHORT[a] for a in archetypes])
    ax.set_ylabel("% Change in Mean Wait Time\n(Post-COVID vs Pre-COVID)")
    ax.set_title("Performance Degradation Under Surge Conditions (RQ3)")
    ax.legend(title="Algorithm")
    save("fig12_surge_degradation")

def fig13_backlog_surge(summary):
    """Backlog growth from pre to post COVID — shows system stress."""
    print("Fig 13: Backlog surge...")
    archetypes = ["high_vol_low_grant","high_vol_high_grant","low_vol_low_grant","low_vol_high_grant"]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(archetypes))
    width = 0.25

    for i, algo in enumerate(ALGO_ORDER):
        pre_backlogs  = [summary[(summary["era"]=="pre_covid_2019")  & (summary["archetype"]==a) &
                                 (summary["algorithm"]==algo)]["backlog_total"].values[0] / 1000
                         if len(summary[(summary["era"]=="pre_covid_2019") &
                                        (summary["archetype"]==a) &
                                        (summary["algorithm"]==algo)]) > 0 else 0
                         for a in archetypes]
        post_backlogs = [summary[(summary["era"]=="post_covid_2024") & (summary["archetype"]==a) &
                                 (summary["algorithm"]==algo)]["backlog_total"].values[0] / 1000
                         if len(summary[(summary["era"]=="post_covid_2024") &
                                        (summary["archetype"]==a) &
                                        (summary["algorithm"]==algo)]) > 0 else 0
                         for a in archetypes]
        ratios = [p/r if r > 0 else 0 for p, r in zip(post_backlogs, pre_backlogs)]
        ax.bar(x + (i-1)*width, ratios, width, label=algo,
               color=ALGO_COLORS[algo], edgecolor="white", linewidth=0.4)

    ax.axhline(1, color="black", linewidth=0.8, linestyle="--", label="No change")
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_SHORT[a] for a in archetypes])
    ax.set_ylabel("Backlog Ratio\n(Post-COVID / Pre-COVID)")
    ax.set_title("Backlog Accumulation Ratio Under Surge (Post ÷ Pre COVID)")
    ax.legend(title="Algorithm")
    save("fig13_backlog_surge")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ECONOMIC IMPACT
# ═══════════════════════════════════════════════════════════════════════════════

def load_econ():
    df = pd.read_csv(f"{RESULTS_DIR}/economic_impact.csv")
    # Normalise column names — strip whitespace in case of encoding artefacts
    df.columns = df.columns.str.strip()
    print(f"  economic_impact columns: {list(df.columns)}")
    return df

def fig14_lost_wages_by_algo(econ):
    """Total lost wages by algorithm for each era — the headline economic figure."""
    print("Fig 14: Lost wages by algorithm...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    archetypes = ["high_vol_low_grant","high_vol_high_grant","low_vol_low_grant","low_vol_high_grant"]
    x = np.arange(len(archetypes))
    width = 0.25

    for ax, era in zip(axes, ["pre_covid_2019","post_covid_2024"]):
        era_df = econ[econ["era"] == era]
        for i, algo in enumerate(ALGO_ORDER):
            vals = [era_df[(era_df["archetype"]==a) & (era_df["algorithm"]==algo)
                           ]["total_lost_wages_usd"].values[0] / 1e9
                    if len(era_df[(era_df["archetype"]==a) & (era_df["algorithm"]==algo)]) > 0
                    else 0
                    for a in archetypes]
            ax.bar(x + (i-1)*width, vals, width, label=algo,
                   color=ALGO_COLORS[algo], edgecolor="white", linewidth=0.4)
        ax.set_title(ERA_LABELS[era])
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_SHORT[a] for a in archetypes], fontsize=9)
        ax.set_ylabel("Estimated Lost Wages ($B)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:.0f}B"))

    axes[0].legend(title="Algorithm")
    fig.suptitle("Estimated Total Lost Worker Compensation by Algorithm (RQ4)", fontsize=12)
    save("fig14_lost_wages")

def fig15_tax_revenue_loss(econ):
    """Federal tax revenue loss by algorithm — RQ5."""
    print("Fig 15: Tax revenue loss...")
    fig, ax = plt.subplots(figsize=(10, 5))
    # Sum across archetypes per algo+era for a cleaner headline view
    agg = (econ.groupby(["era","algorithm"])["total_lost_tax_usd"]
           .sum().reset_index())
    eras = ["pre_covid_2019","post_covid_2024"]
    x = np.arange(len(eras))
    width = 0.25

    for i, algo in enumerate(ALGO_ORDER):
        vals = [agg[(agg["era"]==e) & (agg["algorithm"]==algo)
                    ]["total_lost_tax_usd"].values[0] / 1e9
                if len(agg[(agg["era"]==e) & (agg["algorithm"]==algo)]) > 0
                else 0
                for e in eras]
        ax.bar(x + (i-1)*width, vals, width, label=algo,
               color=ALGO_COLORS[algo], edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([ERA_LABELS[e] for e in eras])
    ax.set_ylabel("Estimated Federal Tax Revenue Loss ($B)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:.0f}B"))
    ax.set_title("Estimated Federal Tax Revenue Loss Attributable to Backlog (RQ5)")
    ax.legend(title="Algorithm")
    save("fig15_tax_revenue_loss")

def fig16_savings_vs_lifo(econ):
    """Savings vs LIFO baseline — how much does FIFO/PRIORITY recover?"""
    print("Fig 16: Savings vs LIFO baseline...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    archetypes = ["high_vol_low_grant","high_vol_high_grant","low_vol_low_grant","low_vol_high_grant"]
    x = np.arange(len(archetypes))
    width = 0.3

    for ax, era in zip(axes, ["pre_covid_2019","post_covid_2024"]):
        era_df = econ[econ["era"] == era]
        for i, algo in enumerate(["FIFO","PRIORITY"]):
            savings = []
            for arch in archetypes:
                lifo_row = era_df[(era_df["archetype"]==arch) & (era_df["algorithm"]=="LIFO")]
                algo_row = era_df[(era_df["archetype"]==arch) & (era_df["algorithm"]==algo)]
                if not lifo_row.empty and not algo_row.empty:
                    diff = (lifo_row["total_lost_wages_usd"].values[0] -
                            algo_row["total_lost_wages_usd"].values[0]) / 1e9
                    savings.append(diff)
                else:
                    savings.append(0)
            color = ALGO_COLORS[algo]
            ax.bar(x + (i - 0.5)*width, savings, width, label=algo,
                   color=color, edgecolor="white", linewidth=0.4)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(ERA_LABELS[era])
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_SHORT[a] for a in archetypes], fontsize=9)
        ax.set_ylabel("Lost Wage Savings vs LIFO ($B)\n(positive = better than LIFO)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:.0f}B"))

    axes[0].legend(title="Algorithm")
    fig.suptitle("Lost Wage Savings Relative to LIFO Baseline (Current Policy)", fontsize=11)
    save("fig16_savings_vs_lifo")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def table1_sim_parameters():
    """Print Table 1: Simulation parameters by archetype (pre-COVID)."""
    import json
    if not os.path.exists(PARAMS_JSON):
        print("  params json not found")
        return
    with open(PARAMS_JSON) as f:
        params = json.load(f)

    era_data = params.get("pre_covid_2019", {})
    rows = []
    for arch, p in era_data.items():
        if arch.startswith("_"):
            continue
        rows.append({
            "Archetype":       arch,
            "λ (cases/mo)":    p["arrival_rate_lambda"],
            "Svc Mean (mo)":   p["service_time_mean"],
            "Svc Median (mo)": p["service_time_median"],
            "Svc Std":         p["service_time_std"],
            "Grant Rate":      f"{p['grant_rate']:.3f}",
            "Appeal Rate":     f"{p['appeal_rate']:.3f}",
            "P(Rep)":          f"{p['p_rep']:.3f}",
            "P(Detained)":     f"{p['p_detained']:.3f}",
            "P(Renege)":       f"{p['p_renege']:.3f}",
        })
    df = pd.DataFrame(rows)
    print("\nTABLE 1: Simulation Parameters — Pre-COVID 2019")
    print(df.to_string(index=False))
    df.to_csv(FIG_DIR / "table1_sim_parameters.csv", index=False)
    print(f"  saved → {FIG_DIR}/table1_sim_parameters.csv")

def table2_main_results(summary):
    """Print Table 2: Key metrics per algorithm × archetype × era."""
    cols = ["era","archetype","algorithm","mean_total_time","backlog_total",
            "throughput","renege_rate_obs","grant_rate_obs"]
    df = summary[cols].copy()
    df["mean_total_time"] = df["mean_total_time"].round(1)
    df["backlog_total"]   = df["backlog_total"].astype(int)
    df["throughput"]      = df["throughput"].round(1)
    df["renege_rate_obs"] = df["renege_rate_obs"].round(3)
    df["grant_rate_obs"]  = df["grant_rate_obs"].round(3)
    print("\nTABLE 2: Main Simulation Results")
    print(df.to_string(index=False))
    df.to_csv(FIG_DIR / "table2_main_results.csv", index=False)
    print(f"  saved → {FIG_DIR}/table2_main_results.csv")

def table3_economic(econ):
    """Print Table 3: Economic impact summary."""
    # Use explicit column sums to avoid pandas version issues with lambda in named agg
    grp = econ.groupby(["era","algorithm"])
    agg = pd.DataFrame({
        "era":                grp["era"].first().values,
        "algorithm":          grp["algorithm"].first().values,
        "total_lost_wages_B": grp["total_lost_wages_usd"].sum().values / 1e9,
        "total_lost_tax_B":   grp["total_lost_tax_usd"].sum().values / 1e9,
        "mean_wait_months":   grp["mean_wait_months"].mean().values,
    }, index=grp.groups.keys()).reset_index(drop=True).round(2)
    print("\nTABLE 3: Economic Impact Summary (aggregated across archetypes)")
    print(agg.to_string(index=False))
    agg.to_csv(FIG_DIR / "table3_economic.csv", index=False)
    print(f"  saved → {FIG_DIR}/table3_economic.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SCORING EXPERIMENT FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def load_scoring_all():
    path = SCORING_DIR / "scoring_all_results.csv"
    if not path.exists():
        print(f"  scoring_all_results.csv not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["detained_bonus","aging_threshold","mean_total_time","renege_rate","decided_throughput"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fig17_scoring_wait_heatmap(scoring_df):
    """
    Heatmap of mean wait across detained_bonus x aging_threshold for Group 1 pre-COVID.
    Key insight: aging threshold matters far more than detained bonus.
    """
    print("Fig 17: Scoring heatmap...")
    if scoring_df.empty:
        return
    sub = scoring_df[(scoring_df["era"]=="pre_covid_2019") &
                     (scoring_df["archetype"]=="high_vol_low_grant") &
                     (scoring_df["algorithm"]=="PRIORITY")].copy()
    if sub.empty:
        print("  no Priority rows for Group 1 pre-COVID scoring")
        return
    pivot = (sub.groupby(["aging_threshold","detained_bonus"])["mean_total_time"]
               .mean().unstack("detained_bonus"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios":[2.2, 1]})
    ax = axes[0]
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=95, vmax=145)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"det={c:.0f}" for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"age={r:.0f}mo" for r in pivot.index], fontsize=9)
    ax.set_title("Mean Wait Time (months) — Priority Scoring Grid\nGroup 1 (HV-Low), Pre-COVID 2019", fontsize=10)
    plt.colorbar(im, ax=ax, label="Mean wait (months)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=9, color="white" if (val > 130 or val < 105) else "black",
                        fontweight="bold")

    axes[1].axis("off")
    txt = ("Reference baselines\n"
           "(Group 1, pre-COVID)\n\n"
           "FIFO:       120.9 mo\n"
           "LIFO:       116.4 mo\n\n"
           "Best Priority configs:\n"
           "det=3, age=12:  103.8 mo *\n"
           "det=5, age=12:  103.8 mo *\n\n"
           "Worst Priority config:\n"
           "det=3, age=3:   135.7 mo x\n\n"
           "Key finding:\n"
           "Aging threshold (rows)\n"
           "dominates detained\n"
           "bonus (columns).\n"
           "age=12 transforms\n"
           "Priority from worst\n"
           "to best algorithm.")
    axes[1].text(0.05, 0.95, txt, transform=axes[1].transAxes,
                 fontsize=9, va="top", family="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8", edgecolor="#aaaaaa"))
    save("fig17_scoring_heatmap")


def fig18_scoring_pareto(scoring_df, summary):
    """
    Pareto frontier: mean wait vs renege across all Priority scoring configs.
    Groups 1 and 4 overlaid, pre- and post-COVID panels.
    """
    print("Fig 18: Scoring Pareto frontier...")
    if scoring_df.empty:
        return
    eras = ["pre_covid_2019","post_covid_2024"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    arch_styles = {
        "high_vol_low_grant": ("o", "#2166ac", "Group 1 (HV-Low)"),
        "low_vol_high_grant": ("s", "#4dac26", "Group 4 (LV-High)"),
    }
    for ax, era in zip(axes, eras):
        for arch, (marker, color, label) in arch_styles.items():
            sub = scoring_df[(scoring_df["era"]==era) & (scoring_df["archetype"]==arch) &
                             (scoring_df["algorithm"]=="PRIORITY")].dropna(subset=["renege_rate","mean_total_time"])
            if sub.empty:
                continue
            ax.scatter(sub["renege_rate"], sub["mean_total_time"],
                       marker=marker, color=color, alpha=0.30, s=20, zorder=2)
            # Pareto frontier
            sub_s = sub.sort_values("renege_rate")
            pareto, best_w = [], float("inf")
            for _, row in sub_s.iterrows():
                if row["mean_total_time"] < best_w:
                    best_w = row["mean_total_time"]; pareto.append(row)
            if pareto:
                pf = pd.DataFrame(pareto)
                ax.plot(pf["renege_rate"], pf["mean_total_time"], "-",
                        color=color, linewidth=2, alpha=0.9, zorder=3, label=label)
            # Baselines
            for algo, ls, bc in [("FIFO","--","#333333"),("LIFO",":","#d6604d")]:
                row = summary[(summary["era"]==era)&(summary["archetype"]==arch)&
                              (summary["algorithm"]==algo)]
                if not row.empty:
                    ax.scatter(row["renege_rate_obs"].values[0],
                               row["mean_total_time"].values[0],
                               marker="D", s=60, color=bc, zorder=5,
                               label=f"{algo} ({label.split()[0]})" if arch=="high_vol_low_grant" else "")
        ax.set_xlabel("Renege Rate", fontsize=10)
        ax.set_ylabel("Mean Total Wait (months)", fontsize=10)
        ax.set_title(ERA_LABELS[era], fontsize=11)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5,-0.04), fontsize=8)
    fig.suptitle("Priority Scoring Space: Wait vs Renege Tradeoff\nPareto frontier shown — lower-left corner is better on both objectives", fontsize=11)
    save("fig18_scoring_pareto")


def fig19_optimal_vs_baselines(scoring_df, summary):
    """
    Grouped bar: FIFO / LIFO / Priority-default / Priority-optimal
    on mean wait (left) and renege (right), pre-COVID, all four groups.
    The headline scoring-experiment figure.
    """
    print("Fig 19: Optimal Priority vs baselines...")
    archetypes = ["high_vol_low_grant","high_vol_high_grant","low_vol_low_grant","low_vol_high_grant"]
    era = "pre_covid_2019"
    algo_styles = [
        ("FIFO",         "#2166ac", "FIFO",              ""),
        ("LIFO",         "#d6604d", "LIFO",              ""),
        ("PRIORITY",     "#4dac26", "Priority (default)",""),
        ("PRIORITY_OPT", "#1a7b3a", "Priority (optimal)","///"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(archetypes))
    width = 0.19

    for ax_i, (col, ylabel, is_pct) in enumerate([
        ("mean_total_time", "Mean Total Wait (months)", False),
        ("renege",          "Renege Rate",               True),
    ]):
        ax = axes[ax_i]
        for i, (akey, color, label, hatch) in enumerate(algo_styles):
            vals = []
            for arch in archetypes:
                if akey == "PRIORITY_OPT":
                    key = (era, arch)
                    v = SCORING_BEST[key][1] if col == "mean_total_time" else SCORING_BEST[key][2]
                else:
                    sum_col = "mean_total_time" if col == "mean_total_time" else "renege_rate_obs"
                    row = summary[(summary["era"]==era)&(summary["archetype"]==arch)&
                                  (summary["algorithm"]==akey)]
                    v = row[sum_col].values[0] if not row.empty else np.nan
                vals.append(v)
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, vals, width, label=label,
                          color=color, hatch=hatch, edgecolor="white", linewidth=0.4, alpha=0.88)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    fmt = f"{v:.0f}" if not is_pct else f"{v:.0%}"
                    ax.text(bar.get_x()+bar.get_width()/2,
                            bar.get_height()+(0.4 if not is_pct else 0.004),
                            fmt, ha="center", va="bottom", fontsize=6.5, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_SHORT_FLAT.get(a,a) for a in archetypes], fontsize=8)
        ax.set_ylabel(ylabel)
        if is_pct:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("FIFO vs LIFO vs Priority — Default and Optimal Scoring\nPre-COVID 2019, All Four Court Groups", fontsize=11)
    save("fig19_optimal_vs_baselines")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CEL ALGORITHM FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def load_cel_all():
    path = CEL_DIR / "cel_all_results.csv"
    if path.exists():
        return pd.read_csv(path)
    parts = list(CEL_DIR.glob("cel_*.csv"))
    if not parts:
        print(f"  no CEL results in {CEL_DIR}")
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)


def fig20_cel_tradeoff_curve(cel_df):
    """
    CEL tradeoff curve: expedited fraction vs mean wait, renege, econ welfare.
    Groups 1 and 4, pre-COVID. Dashed baseline = FIFO.
    """
    print("Fig 20: CEL tradeoff curve...")
    if cel_df.empty:
        return
    era = "pre_covid_2019"
    groups = {
        "high_vol_low_grant": ("Group 1 (HV-Low)", "o", "#2166ac"),
        "low_vol_high_grant": ("Group 4 (LV-High)", "s", "#4dac26"),
    }
    metrics = [
        ("mean_wait_all",       "Mean Wait (months)"),
        ("renege_rate_regular", "Renege Rate — regular path"),
        ("econ_welfare_total",  "Aggregate Econ Welfare Score"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (metric, ylabel) in zip(axes, metrics):
        for arch, (label, marker, color) in groups.items():
            sub = cel_df[(cel_df["era"]==era)&(cel_df["archetype"]==arch)].copy()
            if sub.empty or metric not in sub.columns:
                continue
            curve = sub[sub["config"]!="BASELINE_FIFO"].copy()
            curve["expedited_rate"] = pd.to_numeric(curve["expedited_rate"], errors="coerce")
            curve[metric]           = pd.to_numeric(curve[metric],           errors="coerce")
            curve = curve.sort_values("expedited_rate").dropna(subset=["expedited_rate",metric])
            bl = sub[sub["config"]=="BASELINE_FIFO"]
            ax.plot(curve["expedited_rate"], curve[metric],
                    f"{marker}-", color=color, linewidth=2.2, markersize=8, label=label)
            if not bl.empty:
                bval = pd.to_numeric(bl[metric].values[0], errors="coerce")
                if not np.isnan(bval):
                    ax.axhline(bval, color=color, linestyle="--", linewidth=1.2, alpha=0.55)
        ax.set_xlabel("Fraction Expedited")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        if "renege" in metric:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_title(ylabel, fontsize=9)
        ax.text(0.03, 0.97, "--- FIFO baseline", transform=ax.transAxes,
                fontsize=7, va="top", color="gray", style="italic")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5,-0.04), fontsize=9)
    fig.suptitle("CEL Tradeoff: More Expediting Reduces Wait and Renege but Increases Inequality", fontsize=11)
    save("fig20_cel_tradeoff")


def fig21_cel_econ_vs_comp_filter(cel_df):
    """
    Diagnostic: comp_only vs econ_only at equal expedited volume.
    2x2 grid of metrics for Groups 1 and 4.
    """
    print("Fig 21: CEL filter comparison...")
    if cel_df.empty:
        return
    era   = "pre_covid_2019"
    archs = ["high_vol_low_grant","low_vol_high_grant"]
    alabs = ["Group 1 (HV-Low)","Group 4 (LV-High)"]
    cfgs  = ["CEL_comp_only","CEL_econ_only"]
    ccols = {"CEL_comp_only":"#abdda4","CEL_econ_only":"#2ca25f"}
    clabs = {"CEL_comp_only":"Comp-only filter","CEL_econ_only":"Econ-only filter"}
    metrics = [
        ("mean_wait_regular",        "Wait — regular path (mo)"),
        ("renege_rate_regular",      "Renege rate — regular path"),
        ("econ_welfare_total",       "Aggregate econ welfare"),
        ("mean_econ_score_expedited","Mean econ score of expedited cases"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax_i, (metric, title) in enumerate(metrics):
        ax = axes[ax_i // 2][ax_i % 2]
        x = np.arange(len(archs))
        width = 0.3
        for i, cfg in enumerate(cfgs):
            vals = []
            for arch in archs:
                row = cel_df[(cel_df["era"]==era)&(cel_df["archetype"]==arch)&(cel_df["config"]==cfg)]
                v = pd.to_numeric(row[metric].values[0], errors="coerce") if not row.empty and metric in row.columns else np.nan
                vals.append(v)
            bars = ax.bar(x+(i-0.5)*width, vals, width, label=clabs[cfg],
                          color=ccols[cfg], edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.015,
                            f"{v:.2f}" if v < 20 else f"{v:.0f}",
                            ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(alabs, fontsize=9)
        ax.set_title(title, fontsize=9)
        if "renege" in metric:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        if ax_i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Comp-Only vs Econ-Only Filter — Controlled Comparison at Equal Expedited Volume\nEcon filter wins on welfare targeting; comp filter has marginal wait advantage in Group 4", fontsize=10)
    plt.tight_layout()
    save("fig21_cel_filter_comparison")


def fig22_cel_gini_equity(cel_df):
    """
    Gini coefficient by CEL config — equity cost of lottery-based expediting.
    Monotonically rising Gini = more inequality as expedited fraction grows.
    """
    print("Fig 22: CEL Gini equity tradeoff...")
    if cel_df.empty:
        return
    era   = "pre_covid_2019"
    archs = ["high_vol_low_grant","low_vol_high_grant"]
    alabs = ["Group 1 (HV-Low, det=4.6%)","Group 4 (LV-High, det=21.9%)"]
    plot_cfgs   = ["CEL_strict","CEL_econ_focused","CEL_balanced","CEL_broad","CEL_comp_only","CEL_econ_only"]
    short_labs  = ["Strict","Econ-focused","Balanced","Broad","Comp only","Econ only"]
    colors_seq  = ["#fee090","#fdae61","#f46d43","#d73027","#abdda4","#2ca25f"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, arch, label in zip(axes, archs, alabs):
        sub = cel_df[(cel_df["era"]==era)&(cel_df["archetype"]==arch)]
        ginis, exps = [], []
        for cfg in plot_cfgs:
            row = sub[sub["config"]==cfg]
            if row.empty or "gini_wait" not in row.columns:
                ginis.append(np.nan); exps.append(np.nan)
            else:
                ginis.append(pd.to_numeric(row["gini_wait"].values[0], errors="coerce"))
                exps.append(pd.to_numeric(row["expedited_rate"].values[0], errors="coerce"))
        bars = ax.bar(range(len(plot_cfgs)), ginis, color=colors_seq, edgecolor="white", linewidth=0.5)
        for bar, g, e in zip(bars, ginis, exps):
            if not np.isnan(g):
                ax.text(bar.get_x()+bar.get_width()/2, g+0.001,
                        f"Gini={g:.3f}\n({e:.0%} exp.)",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(plot_cfgs)))
        ax.set_xticklabels(short_labs, fontsize=8.5)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Gini Coefficient of Wait Times\n(0=equal, 1=max unequal)" if ax==axes[0] else "")
        ax.set_ylim(0, 0.32)
    fig.suptitle("CEL Equity Cost: Gini Rises Monotonically with Expedited Fraction\nBroader configs improve mean outcomes but concentrate lottery advantages", fontsize=10)
    save("fig22_cel_gini")


def fig23_master_comparison(summary, cel_df, scoring_df):
    """
    Master comparison: FIFO / LIFO / Priority-default / Priority-optimal /
    CEL-econ_only / CEL-broad on mean wait (left) and renege (right).
    Group 1, pre-COVID. The definitive 'which algorithm wins what' figure.
    """
    print("Fig 23: Master comparison (all algorithms)...")
    era  = "pre_covid_2019"
    arch = "high_vol_low_grant"

    entries = []
    for algo, color, label in [("FIFO","#2166ac","FIFO"),("LIFO","#d6604d","LIFO"),
                                ("PRIORITY","#4dac26","Priority (default)")]:
        row = summary[(summary["era"]==era)&(summary["archetype"]==arch)&(summary["algorithm"]==algo)]
        if not row.empty:
            entries.append({"label":label,"wait":row["mean_total_time"].values[0],
                            "renege":row["renege_rate_obs"].values[0],"color":color,"hatch":""})

    if (era,arch) in SCORING_BEST:
        entries.append({"label":"Priority (optimal)","wait":SCORING_BEST[(era,arch)][1],
                        "renege":SCORING_BEST[(era,arch)][2],"color":"#1a7b3a","hatch":"///"})

    if not cel_df.empty:
        for cfg, label, color in [("CEL_econ_only","CEL (econ-only)","#2ca25f"),
                                   ("CEL_broad",     "CEL (broad)",    "#d73027")]:
            row = cel_df[(cel_df["era"]==era)&(cel_df["archetype"]==arch)&(cel_df["config"]==cfg)]
            if not row.empty:
                w = pd.to_numeric(row["mean_wait_all"].values[0], errors="coerce")
                r = pd.to_numeric(row["renege_rate_regular"].values[0], errors="coerce")
                entries.append({"label":label,"wait":w,"renege":r,"color":color,"hatch":"xx"})

    if not entries:
        print("  no data for master comparison"); return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(entries))
    for ax, vals_key, ylabel, is_pct, ylim in [
        (axes[0], "wait",   "Mean Total Wait (months)", False, (75, 150)),
        (axes[1], "renege", "Renege Rate",               True,  (0,  0.85)),
    ]:
        vals = [e[vals_key] for e in entries]
        for xi, (v, e) in enumerate(zip(vals, entries)):
            if not np.isnan(v):
                ax.bar(xi, v, color=e["color"], hatch=e["hatch"],
                       edgecolor="#444", linewidth=0.5, alpha=0.88)
                fmt = f"{v:.1%}" if is_pct else f"{v:.1f}mo"
                ax.text(xi, v + (ylim[1]*0.012), fmt,
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([e["label"] for e in entries], fontsize=8.5, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        if is_pct:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        # Mark LIFO as current policy
        lifo_idx = next((i for i,e in enumerate(entries) if "LIFO" in e["label"]), None)
        if lifo_idx is not None:
            ax.annotate("Current\npolicy", xy=(lifo_idx, vals[lifo_idx]),
                        xytext=(lifo_idx+0.5, vals[lifo_idx]+(ylim[1]*0.08)),
                        fontsize=7.5, color="#d6604d",
                        arrowprops=dict(arrowstyle="->", color="#d6604d", lw=1))

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#aaa",   label="Queue-discipline (FIFO/LIFO/Priority)"),
        Patch(facecolor="#1a7b3a",hatch="///", label="Optimized Priority (scoring expt)"),
        Patch(facecolor="#2ca25f",hatch="xx",  label="CEL (expedited pathway)"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5,-0.06), fontsize=8.5)
    fig.suptitle("All Algorithms on Two Welfare Objectives — Group 1 (HV-Low), Pre-COVID 2019\nLeft: minimize mean wait   |   Right: minimize dropout (reneging)", fontsize=11)
    plt.tight_layout()
    save("fig23_master_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FIFO vs LIFO COMPLEXITY WEIGHT EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

FL_DIR  = Path("../data/fifo_lifo_experiment")
CMP_DIR = Path("../data/algorithm_comparison")

CONFIG_COLORS = {
    "calibrated":     "#2166ac",
    "criminal_heavy": "#d6604d",
    "rep_heavy":      "#4dac26",
    "family_heavy":   "#984ea3",
    "uniform":        "#ff7f00",
    "minimal":        "#999999",
}
CONFIG_LABELS = {
    "calibrated":     "Calibrated (baseline)",
    "criminal_heavy": "Criminal-heavy",
    "rep_heavy":      "Rep-gap heavy",
    "family_heavy":   "Family-heavy",
    "uniform":        "Uniform weights",
    "minimal":        "Minimal complexity",
}


def load_fl_results():
    p = FL_DIR / "fifo_lifo_results.csv"
    if not p.exists():
        print(f"  fifo_lifo_results.csv not found at {p}")
        return pd.DataFrame()
    return pd.read_csv(p)


def load_fl_summary():
    p = FL_DIR / "fifo_lifo_summary.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_surge():
    p = FL_DIR / "surge_results.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_surge_logs():
    logs = {}
    for f in FL_DIR.glob("queue_log_surge_*.csv"):
        key = f.stem[len("queue_log_surge_"):]
        logs[key] = pd.read_csv(f)
    return logs


def fig24_fl_complexity_wait(fl_df):
    """
    Fig 24: Mean wait by complexity weight config — FIFO vs LIFO side by side.
    Two panels: pre-COVID (left) and post-COVID (right).
    Shows that LIFO's mean-wait advantage is robust across all complexity configs.
    """
    print("Fig 24: FIFO vs LIFO mean wait by complexity config...")
    if fl_df.empty:
        return
    archetypes = ["high_vol_low_grant","high_vol_high_grant","low_vol_low_grant","low_vol_high_grant"]
    configs    = list(CONFIG_COLORS.keys())
    eras       = ["pre_covid_2019","post_covid_2024"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    x = np.arange(len(archetypes))
    width = 0.13

    for ax, era in zip(axes, eras):
        era_df = fl_df[fl_df["era"] == era]
        for ci, cfg in enumerate(configs):
            for ai, algo in enumerate(["FIFO","LIFO"]):
                vals = []
                for arch in archetypes:
                    row = era_df[(era_df["archetype"]==arch) &
                                 (era_df["algorithm"]==algo) &
                                 (era_df["complexity_config"]==cfg)]
                    vals.append(row["mean_total_time"].values[0] if not row.empty else np.nan)
                hatch = "" if algo == "FIFO" else "///"
                offset = (ci * 2 + ai - len(configs) + 0.5) * width * 0.48
                ax.bar(x + offset, vals, width*0.45,
                       color=CONFIG_COLORS[cfg], hatch=hatch,
                       edgecolor="white", linewidth=0.3, alpha=0.85,
                       label=f"{CONFIG_LABELS[cfg]} ({'FIFO' if algo=='FIFO' else 'LIFO'})"
                             if arch == archetypes[0] else "")

        ax.set_title(ERA_LABELS[era], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_SHORT_FLAT.get(a,a) for a in archetypes], fontsize=9)
        ax.set_ylabel("Mean Total Wait Time (months)")
        ax.axhline(0, color="black", linewidth=0.5)

    # Compact legend outside
    from matplotlib.patches import Patch
    legend_els = ([Patch(facecolor=CONFIG_COLORS[c], label=CONFIG_LABELS[c])
                   for c in configs] +
                  [Patch(facecolor="#aaa", label="FIFO (solid)"),
                   Patch(facecolor="#aaa", hatch="///", label="LIFO (hatched)")])
    fig.legend(handles=legend_els, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.12), fontsize=7.5)
    fig.suptitle("Mean Wait Time by Complexity Weight Configuration — FIFO vs LIFO\n"
                 "LIFO advantage is robust across all complexity configurations",
                 fontsize=11)
    save("fig24_fl_complexity_wait")


def fig25_fl_best_head_to_head(fl_summary):
    """
    Fig 25: Best FIFO config vs Best LIFO config — head to head on four metrics.
    One panel per welfare objective, both eras, Group 1 and Group 4.
    The key 'which complexity config wins for each objective' figure.
    """
    print("Fig 25: Best FIFO vs best LIFO head-to-head...")
    if fl_summary.empty:
        return

    objectives = ["Minimize mean wait","Minimize dropout (renege)",
                  "Minimize true_unresolved","Maximize decided output"]
    archs      = ["high_vol_low_grant","low_vol_high_grant"]
    arch_labs  = ["Group 1 (HV-Low)","Group 4 (LV-High)"]
    eras       = ["pre_covid_2019","post_covid_2024"]
    era_labs   = ["Pre-COVID 2019","Post-COVID 2024"]

    metric_map = {
        "Minimize mean wait":       ("mean_total_time",   "Mean Wait (months)",   False),
        "Minimize dropout (renege)":("renege_rate",       "Renege Rate",           True),
        "Minimize true_unresolved": ("true_unresolved",   "True Unresolved Cases", False),
        "Maximize decided output":  ("decided_throughput","Decided Throughput/mo", False),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax_i, obj in enumerate(objectives):
        ax = axes[ax_i // 2][ax_i % 2]
        col, ylabel, is_pct = metric_map[obj]

        x     = np.arange(len(archs) * len(eras))
        width = 0.35
        xtick_labels = [f"{al}\n{el}" for el in era_labs for al in arch_labs]

        for ai, algo in enumerate(["FIFO","LIFO"]):
            vals, cfgs = [], []
            for era in eras:
                for arch in archs:
                    row = fl_summary[(fl_summary["era"]==era) &
                                     (fl_summary["archetype"]==arch) &
                                     (fl_summary["algorithm"]==algo) &
                                     (fl_summary["welfare_objective"]==obj)]
                    if not row.empty:
                        vals.append(row[col].values[0])
                        cfgs.append(row["best_config"].values[0])
                    else:
                        vals.append(np.nan); cfgs.append("")

            color = ALGO_COLORS["FIFO"] if algo == "FIFO" else ALGO_COLORS["LIFO"]
            bars  = ax.bar(x + (ai-0.5)*width, vals, width, label=algo,
                           color=color, edgecolor="white", linewidth=0.4, alpha=0.88)
            for bar, v, cfg in zip(bars, vals, cfgs):
                if not np.isnan(v):
                    # Config label inside bar
                    ax.text(bar.get_x()+bar.get_width()/2,
                            bar.get_height()*0.5,
                            CONFIG_LABELS.get(cfg, cfg)[:8],
                            ha="center", va="center", fontsize=5.5,
                            color="white", fontweight="bold", rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, fontsize=7.5)
        ax.set_title(f"Objective: {obj}", fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8)
        if is_pct:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        if ax_i == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Best Complexity Config per Welfare Objective — FIFO vs LIFO\n"
                 "Config label shown inside bar; LIFO and FIFO often disagree on optimal config",
                 fontsize=11)
    plt.tight_layout()
    save("fig25_fl_best_head_to_head")


def fig26_surge_queue_length(surge_logs):
    """
    Fig 26: FIFO vs LIFO during arrival surge — 4 panels showing where
    the algorithms actually differ. Queue LENGTH is identical (same capacity),
    but cumulative reneges and rolling mean wait diverge meaningfully.

    Panel 1: Total queue length (shows they're the same — capacity-bound)
    Panel 2: Cumulative reneges over time (LIFO spikes post-surge)
    Panel 3: Rolling 10-month mean wait of completed cases (LIFO lower pre-surge,
             then converges as backlog overwhelms both)
    Panel 4: Monthly new arrivals (shows when the surge hits)
    """
    print("Fig 26: Surge test — 4-panel FIFO vs LIFO divergence...")
    if not surge_logs:
        print("  no surge log files found")
        return

    cfg  = "calibrated"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    titles = [
        "Total Queue Length (thousands)\n[Identical — capacity-bound]",
        "Cumulative Reneged Cases\n[LIFO reneges more post-surge]",
        "Rolling 10-mo Mean Wait — Completed Cases (months)\n[LIFO serves recent cases faster pre-surge]",
        "Monthly New Arrivals\n[Surge: months 100-149]",
    ]

    for algo, color in [("FIFO", ALGO_COLORS["FIFO"]), ("LIFO", ALGO_COLORS["LIFO"])]:
        key = f"{algo}_{cfg}"
        if key not in surge_logs:
            print(f"  missing key: {key}")
            continue
        df = surge_logs[key]

        # Panel 0: queue length
        total_q = df["s1_len"] + df["s2_len"]
        axes[0].plot(df["month"], total_q/1000, color=color, linewidth=1.5, label=algo)

        # Panel 1: cumulative reneges
        if "n_reneged" in df.columns:
            axes[1].plot(df["month"], df["n_reneged"], color=color, linewidth=1.5, label=algo)
        else:
            axes[1].text(0.5, 0.5, "Re-run fifo_lifo_experiment.py\nto generate richer logs",
                         transform=axes[1].transAxes, ha="center", va="center", fontsize=9)

        # Panel 2: rolling mean wait
        if "mean_wait_completed" in df.columns:
            roll = df["mean_wait_completed"].rolling(10, min_periods=1).mean()
            axes[2].plot(df["month"], roll, color=color, linewidth=1.5, label=algo)
        else:
            axes[2].text(0.5, 0.5, "Re-run fifo_lifo_experiment.py\nto generate richer logs",
                         transform=axes[2].transAxes, ha="center", va="center", fontsize=9)

        # Panel 3: monthly arrivals
        axes[3].plot(df["month"], df["arrivals"], color=color, linewidth=1, alpha=0.7, label=algo)

    # Surge shading on all panels
    for ax, title in zip(axes, titles):
        ax.axvspan(100, 150, alpha=0.10, color="#d6604d", label="Surge period" if ax==axes[0] else "")
        ax.axvline(100, color="#d6604d", linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(150, color="#d6604d", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlabel("Simulation Month", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Queue Length (thousands)")
    axes[1].set_ylabel("Cumulative Reneges")
    axes[2].set_ylabel("Rolling Mean Wait (months)")
    axes[3].set_ylabel("Cases Arriving")

    fig.suptitle(
        "FIFO vs LIFO During Arrival Surge — Group 1 (HV-Low), Calibrated Complexity\n"
        "Queue length is identical (capacity-bound); reneging and wait time diverge",
        fontsize=11)
    plt.tight_layout()
    save("fig26_surge_queue_length")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — FULL ALGORITHM COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

ALL_ALGO_COLORS = {
    "FIFO":            "#2166ac",
    "LIFO":            "#d6604d",
    "PRIORITY_DEFAULT":"#4dac26",
    "PRIORITY_OPT":    "#1a7b3a",
    "CEL_ECON_ONLY":   "#2ca25f",
    "CEL_BROAD":       "#b2182b",
}
ALL_ALGO_LABELS = {
    "FIFO":            "FIFO",
    "LIFO":            "LIFO (current policy)",
    "PRIORITY_DEFAULT":"Priority (default)",
    "PRIORITY_OPT":    "Priority (optimal)",
    "CEL_ECON_ONLY":   "CEL econ-only",
    "CEL_BROAD":       "CEL broad",
}
ALL_ALGO_ORDER = ["FIFO","LIFO","PRIORITY_DEFAULT","PRIORITY_OPT","CEL_ECON_ONLY","CEL_BROAD"]


def load_comparison():
    p = CMP_DIR / "comparison_results.csv"
    if not p.exists():
        print(f"  comparison_results.csv not found at {p}")
        return pd.DataFrame()
    return pd.read_csv(p)


def fig27_comparison_five_metrics(comp_df):
    """
    Fig 27: Five-metric comparison of all 6 algorithms.
    5-panel grid: mean wait, renege rate, decided throughput,
    true_unresolved, econ welfare.
    Two eras side by side within each panel.
    Groups 1 and 4 shown (most contrasting archetypes).
    """
    print("Fig 27: Full algorithm comparison — 5 metrics...")
    if comp_df.empty:
        return

    archs     = ["high_vol_low_grant","low_vol_high_grant"]
    arch_labs = ["Grp 1", "Grp 4"]
    eras      = ["pre_covid_2019","post_covid_2024"]
    era_labs  = ["Pre-COVID", "Post-COVID"]

    metrics = [
        ("mean_wait",          "Mean Wait (months)",         False),
        ("renege_rate",        "Renege Rate",                 True),
        ("decided_throughput", "Decided Throughput (cases/mo)",False),
        ("true_unresolved",    "True Unresolved (k cases)",   False),
        ("econ_welfare",       "Econ Welfare Score",          False),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    n_combos  = len(archs) * len(eras)
    x         = np.arange(n_combos)
    width     = 0.12
    xtick_labels = [f"{al}\n{el}" for el in era_labs for al in arch_labs]

    for ax, (col, ylabel, is_pct) in zip(axes, metrics):
        for i, algo in enumerate(ALL_ALGO_ORDER):
            vals = []
            for era in eras:
                for arch in archs:
                    row = comp_df[(comp_df["era"]==era) &
                                  (comp_df["archetype"]==arch) &
                                  (comp_df["algorithm"]==algo)]
                    v = row[col].values[0] if not row.empty and col in row.columns else np.nan
                    # Scale true_unresolved to thousands
                    if col == "true_unresolved" and not np.isnan(v):
                        v = v / 1000
                    vals.append(v)

            offset = (i - len(ALL_ALGO_ORDER)/2 + 0.5) * width
            ax.bar(x + offset, vals, width,
                   label=ALL_ALGO_LABELS[algo],
                   color=ALL_ALGO_COLORS[algo],
                   edgecolor="white", linewidth=0.3, alpha=0.88)

        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, fontsize=7)
        ax.set_title(ylabel, fontsize=8, pad=4)
        ax.set_ylabel(ylabel, fontsize=7)
        if is_pct:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        elif col == "true_unresolved":
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}k"))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.12), fontsize=8)
    fig.suptitle("All Six Algorithms Across Five Welfare Metrics — Groups 1 & 4",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save("fig27_comparison_five_metrics")


def fig28_complexity_quartile_wait(comp_df):
    """
    Fig 28: Wait time by complexity quartile per algorithm.
    Shows which cases each algorithm helps — simple vs complex.
    Group 1 pre-COVID only (where the pathology is most visible).
    Key finding: PRIORITY_DEFAULT helps Q4 (complex) and Q1 (simple)
    but abandons Q2/Q3 entirely.
    """
    print("Fig 28: Wait time by complexity quartile...")
    if comp_df.empty:
        return

    era  = "pre_covid_2019"
    arch = "high_vol_low_grant"
    sub  = comp_df[(comp_df["era"]==era) & (comp_df["archetype"]==arch)].copy()

    quartiles = ["wait_q1_simple","wait_q2_low","wait_q3_med","wait_q4_complex"]
    q_labels  = ["Q1 Simple\n(comp≤p25)","Q2 Low\n(p25-p50)","Q3 Medium\n(p50-p75)","Q4 Complex\n(comp>p75)"]

    algos_with_data = [a for a in ALL_ALGO_ORDER
                       if a in sub["algorithm"].values and
                       not sub[sub["algorithm"]==a][quartiles[0]].isna().all()]

    if not algos_with_data:
        print("  no quartile data available"); return

    x     = np.arange(len(quartiles))
    width = 0.8 / len(algos_with_data)
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, algo in enumerate(algos_with_data):
        row = sub[sub["algorithm"]==algo]
        if row.empty: continue
        vals = [pd.to_numeric(row[q].values[0], errors="coerce") for q in quartiles]
        # Replace 0 with NaN (0 means no cases in that quartile)
        vals = [v if v > 1 else np.nan for v in vals]
        offset = (i - len(algos_with_data)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=ALL_ALGO_LABELS.get(algo, algo),
                      color=ALL_ALGO_COLORS.get(algo, "#888"),
                      edgecolor="white", linewidth=0.4, alpha=0.88)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.5, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(q_labels, fontsize=9)
    ax.set_ylabel("Mean Total Wait Time (months)")
    ax.legend(fontsize=8, ncol=2)
    ax.annotate("Missing bars = zero completed cases\nin that quartile (Priority default pathology)",
                xy=(1, 0), xytext=(1.5, ax.get_ylim()[1]*0.85),
                fontsize=8, color="#d6604d",
                arrowprops=dict(arrowstyle="->", color="#d6604d", lw=0.8))
    ax.set_title(
        "Mean Wait by Case Complexity Quartile — Group 1 (HV-Low), Pre-COVID 2019\n"
        "Priority (default) serves Q1 and Q4 while abandoning Q2/Q3 entirely",
        fontsize=10)
    save("fig28_complexity_quartile_wait")


def fig29_true_unresolved_comparison(comp_df):
    """
    Fig 29: True unresolved (backlog + reneged) vs backlog_total.
    Shows the welfare-relevant gap: reneged cases are invisible in backlog_total
    but represent real people who received no decision.
    One panel each for Group 1 pre-COVID and post-COVID.
    """
    print("Fig 29: True unresolved vs backlog comparison...")
    if comp_df.empty:
        return

    arch = "high_vol_low_grant"
    eras = ["pre_covid_2019","post_covid_2024"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, era in zip(axes, eras):
        sub = comp_df[(comp_df["era"]==era) & (comp_df["archetype"]==arch)].copy()
        algos = [a for a in ALL_ALGO_ORDER if a in sub["algorithm"].values]
        x = np.arange(len(algos))
        width = 0.35

        backlog_vals      = []
        reneged_extra_vals= []
        for algo in algos:
            row = sub[sub["algorithm"]==algo]
            if row.empty:
                backlog_vals.append(0); reneged_extra_vals.append(0)
            else:
                bt = row["backlog_total"].values[0]  / 1000
                tu = row["true_unresolved"].values[0]/ 1000
                backlog_vals.append(bt)
                reneged_extra_vals.append(max(0, tu - bt))

        bars1 = ax.bar(x, backlog_vals, width, label="In-queue backlog",
                       color="#4393c3", edgecolor="white", linewidth=0.4)
        bars2 = ax.bar(x, reneged_extra_vals, width, bottom=backlog_vals,
                       label="+ Reneged (no decision)", color="#d6604d",
                       edgecolor="white", linewidth=0.4, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([ALL_ALGO_LABELS.get(a,a) for a in algos],
                           fontsize=7.5, rotation=20, ha="right")
        ax.set_ylabel("Cases (thousands)")
        ax.set_title(ERA_LABELS[era], fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}k"))

    axes[0].legend(fontsize=8)
    fig.suptitle(
        "True Unresolved = In-Queue Backlog + Reneged Cases (No Decision Received)\n"
        "Group 1 (HV-Low) — Reneged cases visible only in welfare-adjusted backlog",
        fontsize=11)
    save("fig29_true_unresolved")


def fig30_decided_throughput(comp_df):
    """
    Fig 30: Decided throughput (actual grants+denials per month) by algorithm.
    The honest productivity metric — excludes reneges.
    Highlights that Priority_default has very low decided throughput
    despite appearing efficient on mean wait.
    All 4 groups, both eras.
    """
    print("Fig 30: Decided throughput...")
    if comp_df.empty:
        return

    archetypes = ["high_vol_low_grant","high_vol_high_grant",
                  "low_vol_low_grant","low_vol_high_grant"]
    eras = ["pre_covid_2019","post_covid_2024"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x     = np.arange(len(archetypes))
    width = 0.12

    for ax, era in zip(axes, eras):
        era_df = comp_df[comp_df["era"]==era]
        for i, algo in enumerate(ALL_ALGO_ORDER):
            vals = []
            for arch in archetypes:
                row = era_df[(era_df["archetype"]==arch) & (era_df["algorithm"]==algo)]
                vals.append(row["decided_throughput"].values[0] if not row.empty else np.nan)
            offset = (i - len(ALL_ALGO_ORDER)/2 + 0.5) * width
            ax.bar(x + offset, vals, width,
                   label=ALL_ALGO_LABELS.get(algo, algo),
                   color=ALL_ALGO_COLORS.get(algo, "#888"),
                   edgecolor="white", linewidth=0.3, alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_SHORT_FLAT.get(a,a) for a in archetypes], fontsize=8)
        ax.set_title(ERA_LABELS[era], fontsize=11)
        ax.set_ylabel("Decided Cases per Month\n(grants + denials, excl. reneges)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08), fontsize=8)
    fig.suptitle(
        "Decided Throughput by Algorithm — Actual Adjudicative Output\n"
        "Priority (optimal) has low throughput because it processes detained cases slowly",
        fontsize=11)
    save("fig30_decided_throughput")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\nGenerating all figures → {FIG_DIR}\n{'='*55}")

    # Section 1: Dataset
    try:
        df = load_case_data()
        print("\n[Section 1: Dataset descriptives]")
        fig1_annual_filings(df)
        fig2_processing_time_dist(df)
        fig3_grant_rate_by_cluster(df)
        fig4_pending_vs_completed(df)
    except FileNotFoundError:
        print(f"  mock_data_FINAL.csv not found at {DATA_CSV}, skipping section 1")

    # Simulation results
    summary = pd.DataFrame()
    try:
        summary = load_summary()
        cases   = load_cases_all()
        qlogs   = load_queue_logs()

        print("\n[Section 2: Simulation results]")
        fig5_mean_wait_grouped(summary)
        fig6_backlog_comparison(summary)
        fig7_queue_length_over_time(qlogs)
        fig8_renege_rate(summary)
        if not cases.empty:
            fig9_wait_time_dist_boxplot(cases)
        else:
            print("  Fig 9 skipped — no cases CSVs found")

        print("\n[Section 3: Statistical tests]")
        fig10_ttest_heatmap()
        fig11_algo_ranking_consistency()

        print("\n[Section 4: Surge comparison]")
        fig12_pre_post_wait_change(summary)
        fig13_backlog_surge(summary)

    except FileNotFoundError as e:
        print(f"  Simulation results not found ({e}), skipping sections 2-4")

    # Economic impact
    econ = pd.DataFrame()
    try:
        econ = load_econ()
        print("\n[Section 5: Economic impact]")
        fig14_lost_wages_by_algo(econ)
        fig15_tax_revenue_loss(econ)
        fig16_savings_vs_lifo(econ)
    except FileNotFoundError:
        print("  economic_impact.csv not found, skipping section 5")

    # Section 6: Scoring experiment
    scoring_df = pd.DataFrame()
    try:
        scoring_df = load_scoring_all()
        if not scoring_df.empty:
            print("\n[Section 6: Scoring experiment]")
            fig17_scoring_wait_heatmap(scoring_df)
            if not summary.empty:
                fig18_scoring_pareto(scoring_df, summary)
                fig19_optimal_vs_baselines(scoring_df, summary)
        else:
            print("  scoring data not found, skipping section 6")
    except Exception as e:
        print(f"  Section 6 error: {e}")

    # Section 7: CEL algorithm
    cel_df = pd.DataFrame()
    try:
        cel_df = load_cel_all()
        if not cel_df.empty:
            print("\n[Section 7: CEL algorithm]")
            fig20_cel_tradeoff_curve(cel_df)
            fig21_cel_econ_vs_comp_filter(cel_df)
            fig22_cel_gini_equity(cel_df)
            if not summary.empty:
                fig23_master_comparison(summary, cel_df, scoring_df)
        else:
            print("  CEL data not found, skipping section 7")
    except Exception as e:
        print(f"  Section 7 error: {e}")

    # Section 8: FIFO vs LIFO complexity experiment
    try:
        fl_df      = load_fl_results()
        fl_summary = load_fl_summary()
        surge_logs = load_surge_logs()
        if not fl_df.empty:
            print("\n[Section 8: FIFO vs LIFO complexity experiment]")
            fig24_fl_complexity_wait(fl_df)
            if not fl_summary.empty:
                fig25_fl_best_head_to_head(fl_summary)
            if surge_logs:
                fig26_surge_queue_length(surge_logs)
        else:
            print("  fifo_lifo_results not found, skipping section 8")
    except Exception as e:
        print(f"  Section 8 error: {e}")

    # Section 9: Full algorithm comparison
    try:
        comp_df = load_comparison()
        if not comp_df.empty:
            print("\n[Section 9: Algorithm comparison]")
            fig27_comparison_five_metrics(comp_df)
            fig28_complexity_quartile_wait(comp_df)
            fig29_true_unresolved_comparison(comp_df)
            fig30_decided_throughput(comp_df)
        else:
            print("  comparison_results not found, skipping section 9")
    except Exception as e:
        print(f"  Section 9 error: {e}")

    # Tables
    print("\n[Summary tables]")
    table1_sim_parameters()
    if not summary.empty:
        table2_main_results(summary)
    if not econ.empty:
        table3_economic(econ)

    print(f"\nDone. All outputs in {FIG_DIR}")

if __name__ == "__main__":
    main()