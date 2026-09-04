import pandas as pd
import json
import numpy as np

# Minimum completed cases for a cluster/era to be considered reliably calibrated.
MIN_COMPLETED_CASES = 500

# Stable semantic archetype labels.
# Assigned by sorting clusters on (grant_rate ASC, volume DESC) so the mapping
# is deterministic across K-Means runs even if cluster IDs shuffle.
ARCHETYPE_LABELS = [
    "high_vol_low_grant",
    "high_vol_high_grant",
    "low_vol_low_grant",
    "low_vol_high_grant",
]

# ============================================
# PARAMETER CALCULATION
# ============================================


def _assign_archetype_labels(valid_courts_df: pd.DataFrame) -> dict:
    """
    Map raw K-Means cluster IDs to stable semantic archetype labels.
    Sorting: grant_rate ASC (low-grant first), then volume DESC as tiebreaker.
    grant_rate is derived from asylum_granted (relief table) — the correct signal.
    """
    summary = (
        valid_courts_df.groupby("court_cluster")
        .agg(
            total_cases=("idncase", "count"),
            grant_rate=("asylum_granted", "mean"),
        )
        .reset_index()
        .sort_values(["grant_rate", "total_cases"], ascending=[True, False])
        .reset_index(drop=True)
    )
    n_clusters = len(summary)
    labels = ARCHETYPE_LABELS[:n_clusters]
    return dict(zip(summary["court_cluster"], labels))


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

            n_cases     = len(cluster_group)
            n_completed = len(completed)
            n_denied    = len(denied)
            low_confidence = n_completed < MIN_COMPLETED_CASES

            if low_confidence:
                print(
                    f"  ⚠️  WARNING [{archetype}]: only {n_completed} completed cases "
                    f"(threshold={MIN_COMPLETED_CASES}). Parameters are low-confidence."
                )

            sim_parameters[era_name][archetype] = {
                "_n_cases":                 n_cases,
                "_n_completed":             n_completed,
                "_n_denied":                n_denied,
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