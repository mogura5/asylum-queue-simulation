import pandas as pd
import json
import numpy as np

# Minimum completed cases for a cluster/era to be considered reliably calibrated.
MIN_COMPLETED_CASES = 500

# Below this completed-fraction, service_time/grant_rate stats for that
# era/archetype are computed from a right-censored sample and are flagged.
# See redesign-decisions.md item 1: post-covid archetypes complete at only
# 26-33% vs. 70-98% pre-covid, which biases these stats toward
# fast-resolving cases (quick denials/withdrawals) while slow cases
# (plausibly most eventual grants) are still pending and excluded. This is
# a lightweight transparency fix (flag + warn), NOT a Kaplan-Meier-style
# censoring correction -- that fuller fix is deferred (see decisions log).
CENSORING_WARNING_THRESHOLD = 0.50

# National totals used to anchor officer/judge capacity, split
# proportionally to each archetype's share of case volume within its era.
#
# Judges: the ij_code.nunique()-per-archetype approach was tried first (see
# redesign-decisions.md capacity section) but was rejected after the
# real-data run -- it counts every distinct judge who ever rotated onto a
# case from a given arrival cohort over that cohort's ENTIRE multi-year
# resolution lifetime (avg 30-44mo, some 6+ years), not concurrent
# headcount, and produced 786-926 "judges" per archetype vs. the real
# national IJ corps of ~700. Switched to the same volume-proportional
# scaling already used for officers, anchored to the real posted national
# IJ headcount (~700 across all states combined) instead.
#
# Officers: Stage 1 has no direct headcount source in this data at all (no
# per-officer identifiers), so volume-proportional scaling was always the
# agreed stand-in here.
NATIONAL_JUDGES_TOTAL = 700
NATIONAL_OFFICERS_TOTAL = 800

# Charge-count buckets used by case.py's make_case() (rng.choice over
# these 4 buckets with a calibrated probability vector).
CHARGE_BUCKETS = [0, 1, 2, 3]

# ============================================
# PARAMETER CALCULATION
# ============================================


def _assign_archetype_labels(valid_courts_df: pd.DataFrame) -> dict:
    """
    Map raw K-Means cluster IDs to stable semantic archetype labels.

    For the k=2 case (current scope -- see redesign-decisions.md item 4),
    there's no way to know ahead of time whether volume or grant_rate is
    the axis that actually separates the two clusters, so this picks it
    dynamically: whichever dimension shows the larger relative gap between
    the two cluster means becomes the label axis. This replaces the old
    fixed 4-label table (which assumed both axes mattered at once) now
    that there are only two clusters to name.
    """
    summary = (
        valid_courts_df.groupby("court_cluster")
        .agg(
            total_cases=("idncase", "count"),
            grant_rate=("asylum_granted", "mean"),
        )
        .reset_index()
    )
    n_clusters = len(summary)

    if n_clusters == 2:
        vol = summary["total_cases"].to_numpy(dtype=float)
        grant = summary["grant_rate"].to_numpy(dtype=float)

        vol_rel_gap = abs(vol[0] - vol[1]) / max(vol.mean(), 1e-9)
        grant_rel_gap = abs(grant[0] - grant[1]) / max(grant.mean(), 1e-9)

        if vol_rel_gap >= grant_rel_gap:
            summary = summary.sort_values("total_cases", ascending=False).reset_index(drop=True)
            labels = ["high_volume", "low_volume"]
            axis = "volume"
        else:
            summary = summary.sort_values("grant_rate", ascending=False).reset_index(drop=True)
            labels = ["high_grant", "low_grant"]
            axis = "grant_rate"
        print(f"  Archetype axis chosen: {axis} "
              f"(volume rel. gap={vol_rel_gap:.2f}, grant_rate rel. gap={grant_rel_gap:.2f})")
        return dict(zip(summary["court_cluster"], labels))

    # Fallback for any other cluster count (kept generic/deterministic;
    # the k=2 branch above is the one actually used at current scope).
    summary = summary.sort_values(
        ["grant_rate", "total_cases"], ascending=[True, False]
    ).reset_index(drop=True)
    labels = [f"cluster_{i}_by_grant_rate" for i in range(n_clusters)]
    return dict(zip(summary["court_cluster"], labels))


def _charge_distribution(cluster_group: pd.DataFrame) -> list:
    """Empirical P(nbr_of_charges = 0/1/2/3+) for this cluster, replacing
    the old hardcoded [0.45, 0.35, 0.15, 0.05] default."""
    charges = cluster_group["nbr_of_charges"].fillna(0).clip(upper=CHARGE_BUCKETS[-1])
    dist = charges.value_counts(normalize=True).reindex(CHARGE_BUCKETS, fill_value=0.0)
    total = dist.sum()
    if total <= 0:
        # No usable data -- fall back to a flat split rather than the old
        # hand-picked default, so a data gap is visible as "flat" not
        # silently identical to a real calibrated shape.
        return [0.25, 0.25, 0.25, 0.25]
    return (dist / total).round(4).tolist()


def generate_simulation_parameters(input_csv: str, output_json: str):
    print("=" * 60)
    print("CALCULATING ERA-SPECIFIC SIMULATION PARAMETERS")
    print("=" * 60)

    print(f"\nLoading {input_csv}...")
    df = pd.read_csv(input_csv)
    df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")
    df["arrival_year"] = df["arrival_date"].dt.year

    eras = {
        "pre_covid_2019":  df[df["arrival_year"] == 2019].copy(),
        "post_covid_2024": df[df["arrival_year"] == 2024].copy(),
    }

    sim_parameters = {}

    all_valid = df[df["court_cluster"] != -1]
    archetype_map = _assign_archetype_labels(all_valid)
    print(f"\nCluster → archetype mapping (full dataset):")
    for k, v in archetype_map.items():
        print(f"  K-Means cluster {k}  →  {v}")

    for era_name, era_df in eras.items():
        print(f"\n{'='*60}")
        print(f"ERA: {era_name}  (n={len(era_df):,} cases)")
        print(f"{'='*60}")

        month_dist = era_df["arrival_date"].dt.month.value_counts().sort_index()
        print(f"Month coverage: {sorted(era_df['arrival_date'].dt.month.unique())}")
        print(f"Cases per month:\n{month_dist.to_string()}")

        valid_courts = era_df[era_df["court_cluster"] != -1]
        sim_parameters[era_name] = {}

        # Era-wide total used to scale both officer and judge capacity
        # proportionally to each archetype's share of case volume within
        # this era (anchored to NATIONAL_OFFICERS_TOTAL / NATIONAL_JUDGES_TOTAL
        # respectively). See redesign-decisions.md capacity section.
        era_total_cases = len(valid_courts)

        for cluster_id, cluster_group in valid_courts.groupby("court_cluster"):
            archetype = archetype_map.get(cluster_id, f"cluster_{cluster_id}")

            # Arrival rate uses cluster-specific month coverage (not era-wide)
            cluster_months = max(1, cluster_group["arrival_date"].dt.month.nunique())
            cases_per_month = len(cluster_group) / float(cluster_months)

            completed = cluster_group[~cluster_group["is_pending"]]
            if not completed.empty:
                avg_service   = completed["processing_months"].mean()
                med_service   = completed["processing_months"].median()
                std_service   = completed["processing_months"].std()
                pct90_service = completed["processing_months"].quantile(0.90)
            else:
                avg_service = med_service = std_service = pct90_service = 0.0

            # Grant rate: P(protection granted | case completed)
            grant_rate = completed["asylum_granted"].mean() if not completed.empty else 0.0

            # Appeal rate: P(appeal | IJ denied) — routes denied to appeal path
            denied = completed[completed["decision_category"] == "denied"]
            appeal_rate = denied["has_appeal"].mean() if not denied.empty else 0.0

            # Reneging restricted to completed cases (pending = hasn't had chance yet)
            p_renege   = completed["missed_hearing"].mean() if not completed.empty else 0.0
            p_rep      = cluster_group["has_representation"].mean()
            p_detained = cluster_group["is_detained"].mean()

            # Criminal-record rate -- now calibrated from real data instead
            # of the hardcoded 0.018/0.05 defaults scattered through the
            # old simulator files. Uses the whole cluster (not just
            # completed), matching p_rep/p_detained's treatment above.
            p_criminal = cluster_group["has_criminal_record"].mean()

            # Charge-count distribution -- real empirical shape, replacing
            # the old hardcoded [0.45, 0.35, 0.15, 0.05].
            charge_dist = _charge_distribution(cluster_group)

            # Judges AND officers both now scale proportionally to this
            # archetype's share of era-wide case volume, anchored to their
            # respective national totals -- NOT a uniform national-total/2
            # split (the original confirmed capacity bug, which divided by
            # archetype count with no regard for volume). See the
            # NATIONAL_JUDGES_TOTAL comment above for why judges moved off
            # the ij_code.nunique() approach.
            volume_share = len(cluster_group) / era_total_cases if era_total_cases else 0.0
            n_judges = max(1, round(NATIONAL_JUDGES_TOTAL * volume_share))
            n_officers = max(1, round(NATIONAL_OFFICERS_TOTAL * volume_share))

            # Effective per-server service time -- NOT the raw
            # service_time_mean/service_time_std above. Those describe the
            # average total ELAPSED months from arrival to decision, which
            # already contains years of real queueing delay (the backlog
            # itself). Feeding that number into the simulator as if it were
            # server-occupied time (the standard M/M/c "service time")
            # double-counts the wait as work and produces an impossibly
            # overloaded queue -- verified when the orchestration runner's
            # smoke test showed utilization of ~1000x at real scale (see
            # redesign-decisions.md, "Service time recalibration" case
            # study). Instead, back out the effective occupied-time-per-case
            # from the REAL OBSERVED THROUGHPUT of this archetype: how many
            # cases this many judges/officers actually completed per month,
            # historically. This makes simulated capacity match reality by
            # construction, so the simulator's arrival-vs-capacity ratio
            # reflects the real backlog dynamic instead of an artifact of
            # conflating pendency with service time.
            observed_throughput = (
                len(completed) / cluster_months if cluster_months else 0.0
            )
            if observed_throughput > 0:
                effective_service_time_s1 = n_officers / observed_throughput
                effective_service_time_s2 = n_judges / observed_throughput
            else:
                # No completions to derive real throughput from -- fall back
                # to the old 20/80 split of raw elapsed time so the
                # simulator still has a usable (if imperfect) number rather
                # than a divide-by-zero.
                effective_service_time_s1 = avg_service * 0.20
                effective_service_time_s2 = avg_service * 0.80

            # Preserve the coefficient of variation from the real elapsed-time
            # distribution when scaling to the effective time, since there's
            # no real distributional data for "occupied time" itself.
            cv = (std_service / avg_service) if avg_service > 0 else 0.5
            effective_service_time_s1_std = effective_service_time_s1 * cv
            effective_service_time_s2_std = effective_service_time_s2 * cv

            n_cases     = len(cluster_group)
            n_completed = len(completed)
            n_denied    = len(denied)
            low_confidence = n_completed < MIN_COMPLETED_CASES
            completion_rate = n_completed / n_cases if n_cases else 0.0
            censored_sample = completion_rate < CENSORING_WARNING_THRESHOLD

            if low_confidence:
                print(
                    f"  ⚠️  WARNING [{archetype}]: only {n_completed} completed cases "
                    f"(threshold={MIN_COMPLETED_CASES}). Parameters are low-confidence."
                )
            if censored_sample:
                print(
                    f"  ⚠️  CENSORING WARNING [{archetype}]: only {completion_rate:.1%} of "
                    f"cases completed (threshold={CENSORING_WARNING_THRESHOLD:.0%}). "
                    f"service_time_*, grant_rate, AND effective_service_time_* here are "
                    f"computed from a right-censored, fast-resolving subsample -- "
                    f"observed_throughput in particular reflects only how fast the "
                    f"currently-completed cases resolved, not the true long-run rate "
                    f"(most of this cohort hasn't had a chance to complete yet). Treat "
                    f"with caution (see redesign-decisions.md, data checklist item 1)."
                )

            sim_parameters[era_name][archetype] = {
                "_n_cases":                 n_cases,
                "_n_completed":             n_completed,
                "_n_denied":                n_denied,
                "_completion_rate":         round(completion_rate, 3),
                "_censored_sample":         censored_sample,
                "_low_confidence":          low_confidence,
                "_cluster_months_observed": cluster_months,
                "arrival_rate_lambda":      round(cases_per_month, 2),
                "service_time_mean":        round(avg_service, 2),
                "service_time_median":      round(med_service, 2),
                "service_time_std":         round(std_service, 2),
                "service_time_p90":         round(pct90_service, 2),
                "grant_rate":               round(grant_rate, 3),
                "appeal_rate":              round(appeal_rate, 3),
                "p_rep":                    round(p_rep, 3),
                "p_detained":               round(p_detained, 3),
                "p_renege":                 round(p_renege, 3),
                "p_criminal":               round(p_criminal, 3),
                "charge_dist":              charge_dist,
                "n_judges":                 n_judges,
                "n_officers":               n_officers,
                "_observed_throughput":         round(observed_throughput, 2),
                "effective_service_time_s1":     round(effective_service_time_s1, 3),
                "effective_service_time_s2":     round(effective_service_time_s2, 3),
                "effective_service_time_s1_std": round(effective_service_time_s1_std, 3),
                "effective_service_time_s2_std": round(effective_service_time_s2_std, 3),
            }

    with open(output_json, "w") as f:
        json.dump(sim_parameters, f, indent=4)
    print(f"\nSaved parameters to: {output_json}")

    print("\n--- PREVIEW: pre_covid_2019 ---")
    for archetype, params in sim_parameters["pre_covid_2019"].items():
        print(f"\n  [{archetype}]")
        for k, v in params.items():
            print(f"    {k}: {v}")

    return sim_parameters

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    generate_simulation_parameters(
        input_csv="../data/mock_data_FINAL.csv",
        output_json="../data/sim_parameters_FINAL.json",
    )