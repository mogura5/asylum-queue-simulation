"""
Protocol-integrity tests for orchestrator.py's result-merging behavior
(redesign-decisions.md, 'Result merging (so CEL doesn't require redoing
FIFO/LIFO/PRIORITY)'). This is pure pandas logic lifted out of
run_all_experiments() so it can be tested without running a real simulation.
"""

import pandas as pd
import pytest

DEDUP_KEYS = ["era", "archetype", "algorithm", "replication"]


def merge_replications(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Exact logic from orchestrator.py::run_all_experiments -- kept as a
    standalone helper here so it's testable in isolation. If this drifts
    from the real implementation, resync it (or better, have orchestrator.py
    import this function instead of duplicating the logic)."""
    merged = pd.concat([old_df, new_df], ignore_index=True)
    return merged.drop_duplicates(subset=DEDUP_KEYS, keep="last")


def _row(era, archetype, algorithm, replication, throughput):
    return {
        "era": era, "archetype": archetype, "algorithm": algorithm,
        "replication": replication, "throughput": throughput,
    }


def test_merge_adds_new_algorithm_without_touching_existing_rows():
    """The core promise: running CEL after FIFO/LIFO/PRIORITY must not
    overwrite or drop any of the prior 180 rows."""
    old_df = pd.DataFrame([
        _row("pre_covid_2019", "high_volume", "FIFO", 0, 3781.8),
        _row("pre_covid_2019", "high_volume", "LIFO", 0, 3781.2),
    ])
    new_df = pd.DataFrame([
        _row("pre_covid_2019", "high_volume", "CEL_balanced", 0, 3751.3),
    ])
    merged = merge_replications(old_df, new_df)

    assert len(merged) == 3
    fifo_row = merged[merged["algorithm"] == "FIFO"].iloc[0]
    assert fifo_row["throughput"] == pytest.approx(3781.8)


def test_merge_deduplicates_on_key_keeping_newest():
    """Re-running the SAME (era, archetype, algorithm, replication) must
    replace the old row with the new one, not duplicate it."""
    old_df = pd.DataFrame([
        _row("post_covid_2024", "low_volume", "CEL_balanced", 3, 100.0),  # stale/buggy run
    ])
    new_df = pd.DataFrame([
        _row("post_covid_2024", "low_volume", "CEL_balanced", 3, 275.7),  # corrected re-run
    ])
    merged = merge_replications(old_df, new_df)

    assert len(merged) == 1
    assert merged.iloc[0]["throughput"] == pytest.approx(275.7), (
        "merge kept the OLD row instead of the newest one -- keep='last' "
        "must win on a dedup collision."
    )


def test_merge_is_idempotent_when_rerun_with_identical_data():
    df = pd.DataFrame([
        _row("pre_covid_2019", "high_volume", "FIFO", r, 3781.8) for r in range(15)
    ])
    once = merge_replications(df, df)
    twice = merge_replications(once, df)
    assert len(once) == 15
    assert len(twice) == 15


def test_merge_matches_users_actual_observed_counts():
    """Regression pin for the actual merge the user ran: 180 prior rows +
    120 new CEL rows -> 300 total, with no overlap (CEL wasn't run before)."""
    old_df = pd.DataFrame([
        _row("e", "a", algo, r, 1.0)
        for algo in ("FIFO", "LIFO", "PRIORITY")
        for r in range(60)  # 3 algos x 60 "rows" standing in for the 180 total
    ])
    new_df = pd.DataFrame([
        _row("e", "a", algo, r, 2.0)
        for algo in ("CEL_balanced", "CEL_broad")
        for r in range(60)  # 2 algos x 60 standing in for the 120 total
    ])
    merged = merge_replications(old_df, new_df)
    assert len(old_df) == 180
    assert len(new_df) == 120
    assert len(merged) == 300