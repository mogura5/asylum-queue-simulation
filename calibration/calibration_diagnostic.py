"""
CALIBRATION DIAGNOSTIC SCRIPT (standalone — not part of the simulator)
=======================================================================
Run this against your local raw/processed files to answer five open
calibration questions before we rewrite the simulator. It only PRINTS
summary tables and writes a couple of small CSVs — no case-level data
needs to leave your machine.

Usage:
    python3 calibration_diagnostic.py --data-dir ../data

Expects, in --data-dir:
    master.csv
    reliefApplications.csv
    mock_data_FINAL.csv   (the already-processed case-level output)

Outputs (written to --data-dir/diagnostic_out/):
    cosc_vs_cinput_gap_summary.csv
    code_mapping_coverage.csv
    complexity_attr_vs_processing_time.csv
    court_cluster_table.csv
    silhouette_scores.csv

Everything also prints to the console so you can paste results back
without needing to send any files.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 1. cosc_date vs cinput_date gap
# ============================================================

def check_cosc_vs_cinput(data_dir: Path, out_dir: Path):
    section("1. cosc_date (NTA issued) vs cinput_date (NTA received by court)")

    usecols = ["idncase", "idnproceeding", "cosc_date", "cinput_date"]
    master = pd.read_csv(data_dir / "master.csv", usecols=usecols, dtype=str, low_memory=False, encoding="latin1")
    master["cosc_date"] = pd.to_datetime(master["cosc_date"], errors="coerce")
    master["cinput_date"] = pd.to_datetime(master["cinput_date"], errors="coerce")

    both = master.dropna(subset=["cosc_date", "cinput_date"]).copy()
    both["gap_days"] = (both["cinput_date"] - both["cosc_date"]).dt.days
    both["gap_months"] = both["gap_days"] / 30.44
    both["cinput_year"] = both["cinput_date"].dt.year

    print(f"Proceedings with both dates present: {len(both):,} / {len(master):,}")
    print(f"Negative gap (cinput before cosc — check for swapped/odd records): "
          f"{(both['gap_days'] < 0).sum():,} ({(both['gap_days'] < 0).mean():.2%})")

    pos = both[both["gap_days"] >= 0]
    print("\nGap distribution (cinput_date - cosc_date), all proceedings with gap >= 0:")
    print(pos["gap_months"].describe(percentiles=[.25, .5, .75, .9, .95]).to_string())

    thresholds = [1, 3, 6, 12, 24]
    print("\nShare of proceedings with gap exceeding threshold:")
    rows = []
    for t in thresholds:
        share = (pos["gap_months"] > t).mean()
        print(f"  > {t:>2} months: {share:.2%}")
        rows.append({"threshold_months": t, "share_exceeding": share})

    print("\nGap distribution by cinput_date year (era check):")
    by_year = pos[pos["cinput_year"].between(2015, 2026)].groupby("cinput_year")["gap_months"].agg(
        ["count", "mean", "median"]
    )
    print(by_year.to_string())

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "cosc_vs_cinput_gap_summary.csv", index=False)
    by_year.to_csv(out_dir / "cosc_vs_cinput_gap_by_year.csv")
    print(f"\nSaved: cosc_vs_cinput_gap_summary.csv, cosc_vs_cinput_gap_by_year.csv")


# ============================================================
# 2. Code mapping coverage: case_type, dec_code, appl_dec
# ============================================================

def check_code_mappings(data_dir: Path, out_dir: Path):
    section("2. Code mapping coverage — case_type / dec_code / appl_dec")

    # Import the actual mapping dicts from data_processor.py so we're checking
    # against the real current assumptions, not a re-typed copy.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        sys.path.insert(0, str(data_dir.parent))  # in case data_processor.py lives one level up
        from data_processor import DEC_CODE_MAP, RELIEF_DECISION_MAP  # type: ignore
    except ImportError:
        print("Could not import data_processor.py directly — place this script next to it, "
              "or adjust sys.path. Falling back to inline copies for the check.")
        DEC_CODE_MAP = {
            'A': 'granted', 'G': 'granted', 'W': 'granted',
            'R': 'denied', 'D': 'denied',
            'X': 'other_exit', 'T': 'other_exit', 'V': 'other_exit',
            'E': 'other_exit', 'O': 'other_exit', 'Z': 'other_exit',
            'U': 'other_exit', 'L': 'other_exit', 'J': 'other_exit',
            'H': 'other_exit', 'S': 'other_exit', 'C': 'other_exit',
        }
        RELIEF_DECISION_MAP = {
            'G': 'granted', 'C': 'granted', 'F': 'granted', 'I': 'granted',
            'D': 'denied',
            'O': 'other_exit', 'W': 'other_exit', 'A': 'other_exit',
        }

    master = pd.read_csv(data_dir / "master.csv", usecols=["case_type", "dec_code"], dtype=str, low_memory=False, encoding="latin1")
    relief = pd.read_csv(data_dir / "reliefApplications.csv", usecols=["appl_code", "appl_dec"], dtype=str, low_memory=False, encoding="latin1")
    for col in ["case_type", "dec_code"]:
        master[col] = master[col].str.strip().str.upper()
    relief["appl_code"] = relief["appl_code"].str.strip().str.upper()
    relief["appl_dec"] = relief["appl_dec"].str.strip().str.upper()

    print("\n-- case_type value counts (checking for an 'F' family code) --")
    ct_counts = master["case_type"].value_counts(dropna=False)
    print(ct_counts.to_string())
    if "F" in ct_counts.index:
        print("\n'F' IS present in case_type — is_family may be derivable after all.")
    else:
        print("\n'F' is NOT present in case_type — the simulator's is_family=case_type=='F' "
              "assumption does not match this data; case_type does not appear to encode family status.")

    print("\n-- dec_code coverage vs DEC_CODE_MAP --")
    dec_counts = master["dec_code"].value_counts(dropna=False)
    print(dec_counts.to_string())
    unmapped_dec = [c for c in dec_counts.index if pd.notna(c) and c not in DEC_CODE_MAP]
    if unmapped_dec:
        print(f"\nUNMAPPED dec_code values found in data: {unmapped_dec}")
        print("These rows currently map to NaN decision_category and get silently dropped from "
              "grant/deny/other_exit accounting.")
    else:
        print("\nAll observed dec_code values are covered by DEC_CODE_MAP.")

    print("\n-- appl_dec coverage vs RELIEF_DECISION_MAP (asylum relief applications only) --")
    relief_asy = relief[relief["appl_code"] == "ASYL"]
    appl_counts = relief_asy["appl_dec"].value_counts(dropna=False)
    print(appl_counts.to_string())
    unmapped_appl = [c for c in appl_counts.index if pd.notna(c) and c not in RELIEF_DECISION_MAP]
    if unmapped_appl:
        print(f"\nUNMAPPED appl_dec values found in data: {unmapped_appl}")
    else:
        print("\nAll observed appl_dec values are covered by RELIEF_DECISION_MAP.")

    coverage_rows = []
    total_dec = dec_counts.sum()
    covered_dec = sum(c for v, c in dec_counts.items() if pd.notna(v) and v in DEC_CODE_MAP)
    coverage_rows.append({"field": "dec_code", "total_rows": total_dec,
                           "covered_rows": covered_dec, "coverage_pct": covered_dec / total_dec})
    total_appl = appl_counts.sum()
    covered_appl = sum(c for v, c in appl_counts.items() if pd.notna(v) and v in RELIEF_DECISION_MAP)
    coverage_rows.append({"field": "appl_dec", "total_rows": total_appl,
                           "covered_rows": covered_appl, "coverage_pct": covered_appl / total_appl})
    pd.DataFrame(coverage_rows).to_csv(out_dir / "code_mapping_coverage.csv", index=False)
    print(f"\nSaved: code_mapping_coverage.csv")


# ============================================================
# 3. Complexity attributes vs processing_months
# ============================================================

def check_complexity_correlations(data_dir: Path, out_dir: Path):
    section("3. Complexity attributes vs. processing_months (completed cases only)")

    df = pd.read_csv(
        data_dir / "mock_data_FINAL.csv",
        usecols=["is_pending", "processing_months", "has_representation",
                 "has_criminal_record", "nbr_of_charges"],
    )
    completed = df[df["is_pending"] == False].copy()
    print(f"Completed cases available: {len(completed):,}")

    completed["no_rep"] = ~completed["has_representation"].astype(bool)
    completed["multi_charge"] = completed["nbr_of_charges"].fillna(0) >= 2

    rows = []
    for label, mask in [
        ("no_rep = True", completed["no_rep"] == True),
        ("no_rep = False", completed["no_rep"] == False),
        ("has_criminal_record = True", completed["has_criminal_record"] == True),
        ("has_criminal_record = False", completed["has_criminal_record"] == False),
        ("multi_charge (>=2) = True", completed["multi_charge"] == True),
        ("multi_charge (>=2) = False", completed["multi_charge"] == False),
    ]:
        sub = completed.loc[mask, "processing_months"].dropna()
        rows.append({
            "attribute_value": label,
            "n": len(sub),
            "mean_processing_months": sub.mean(),
            "median_processing_months": sub.median(),
        })
        print(f"  {label:32s} n={len(sub):>8,}  mean={sub.mean():6.2f}mo  median={sub.median():6.2f}mo")

    print("\nis_family and filed_late are excluded from this check — neither has a real field "
          "backing it in the current pipeline (see section 2 for the case_type finding on is_family; "
          "filed_late has no proxy field loaded at all).")

    pd.DataFrame(rows).to_csv(out_dir / "complexity_attr_vs_processing_time.csv", index=False)
    print(f"\nSaved: complexity_attr_vs_processing_time.csv")


# ============================================================
# 4. Court clustering validation
# ============================================================

def check_clustering(data_dir: Path, out_dir: Path):
    section("4. Court archetype clustering — silhouette check for k=4 vs alternatives")

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    df = pd.read_csv(data_dir / "mock_data_FINAL.csv", usecols=["base_city_code", "asylum_granted"])

    court_stats = df.groupby("base_city_code").agg(
        total_cases=("base_city_code", "count"),
        grant_rate=("asylum_granted", "mean"),
    ).reset_index()
    valid_courts = court_stats[court_stats["total_cases"] > 100].copy()
    print(f"Courts with >100 cases (same filter as data_processor.py): {len(valid_courts):,}")

    valid_courts.to_csv(out_dir / "court_cluster_table.csv", index=False)
    print("Saved: court_cluster_table.csv (small — safe to attach if you want me to double check)")

    scaler = StandardScaler()
    features = scaler.fit_transform(valid_courts[["total_cases", "grant_rate"]])

    rows = []
    print("\nSilhouette score by k (higher is better separated; current code uses k=4 unconditionally):")
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(features)
        score = silhouette_score(features, labels)
        print(f"  k={k}: silhouette={score:.4f}")
        rows.append({"k": k, "silhouette_score": score})

    pd.DataFrame(rows).to_csv(out_dir / "silhouette_scores.csv", index=False)
    print(f"\nSaved: silhouette_scores.csv")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Calibration diagnostic for the asylum queue simulator")
    parser.add_argument("--data-dir", default="../data", help="Directory containing master.csv, "
                         "reliefApplications.csv, and mock_data_FINAL.csv")
    parser.add_argument("--skip", nargs="*", default=[], choices=["gap", "codes", "complexity", "cluster"],
                         help="Sections to skip, e.g. --skip cluster")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / "diagnostic_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "gap" not in args.skip:
        try:
            check_cosc_vs_cinput(data_dir, out_dir)
        except FileNotFoundError as e:
            print(f"\n[SKIPPED section 1 — file not found] {e}")

    if "codes" not in args.skip:
        try:
            check_code_mappings(data_dir, out_dir)
        except FileNotFoundError as e:
            print(f"\n[SKIPPED section 2 — file not found] {e}")

    if "complexity" not in args.skip:
        try:
            check_complexity_correlations(data_dir, out_dir)
        except FileNotFoundError as e:
            print(f"\n[SKIPPED section 3 — file not found] {e}")

    if "cluster" not in args.skip:
        try:
            check_clustering(data_dir, out_dir)
        except FileNotFoundError as e:
            print(f"\n[SKIPPED section 4 — file not found] {e}")

    section("DONE")
    print(f"All summary CSVs written to: {out_dir}")
    print("You can paste the console output back directly, or attach the small CSVs from "
          "diagnostic_out/ — none of them contain case-level/individual data.")


if __name__ == "__main__":
    main()