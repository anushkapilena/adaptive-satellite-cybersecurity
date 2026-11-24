
from pathlib import Path
import json

import numpy as np
import pandas as pd

# =====================================================
# PHASE 9A.2 - DATASET INTEGRITY AUDIT
# =====================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ESA-Mission1"
SOURCE = ROOT / "results" / "phase9a1"
OUT = ROOT / "results" / "phase9a2"

OUT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]

print("\n========== PHASE 9A.2 START ==========")

# =====================================================
# 1. LOAD CONFIGURATION AND OFFICIAL LABELS
# =====================================================

with open(SOURCE / "config.json", encoding="utf-8") as f:
    config = json.load(f)

labels = pd.read_csv(DATA / "labels.csv")

required = {"ID", "Channel", "StartTime", "EndTime"}
assert required.issubset(labels.columns)

labels["StartTime"] = pd.to_datetime(
    labels["StartTime"], utc=True
)

labels["EndTime"] = pd.to_datetime(
    labels["EndTime"], utc=True
)

assert (labels["StartTime"] <= labels["EndTime"]).all()

event_buffer = pd.Timedelta(
    hours=config["global_event_buffer_hours"]
)

# Exclude every official labelled interval,
# irrespective of its telemetry channel.
intervals = sorted(
    (
        (row.StartTime - event_buffer).value,
        (row.EndTime + event_buffer).value
    )
    for row in labels.itertuples(index=False)
)

# Merge overlapping event intervals.
merged = []

for start, end in intervals:
    if not merged or start > merged[-1][1]:
        merged.append([start, end])
    else:
        merged[-1][1] = max(merged[-1][1], end)

event_starts = np.array(
    [a for a, b in merged], dtype=np.int64
)
event_ends = np.array(
    [b for a, b in merged], dtype=np.int64
)

print("Official unique events:", labels["ID"].nunique())
print("Official label rows:", len(labels))
print("Merged exclusion intervals:", len(merged))

# =====================================================
# 2. CHECK EVENT INTERSECTION
# =====================================================

def event_intersections(starts, ends):
    """
    Check whether a saved sequence interval intersects
    any buffered official ESA anomaly interval.
    """
    positions = np.searchsorted(
        event_starts,
        ends,
        side="right"
    ) - 1

    intersects = np.zeros(len(starts), dtype=bool)

    possible = positions >= 0

    intersects[possible] = (
        event_ends[positions[possible]]
        >= starts[possible]
    )

    return intersects

# =====================================================
# 3. AUDIT SAVED SEQUENCES
# =====================================================

reports = []
all_data = {}

for split in SPLITS:
    print(f"\nAuditing {split}...")

    with np.load(SOURCE / f"{split}_normal.npz") as data:
        X = data["X"]
        starts = data["start_ns"]
        ends = data["end_ns"]

    all_data[split] = {
        "starts": starts,
        "ends": ends
    }

    n = len(X)

    finite = bool(np.isfinite(X).all())
    shape_ok = bool(
        X.ndim == 3
        and X.shape[1] == config["sequence_length"]
        and X.shape[2] == len(config["channels"])
    )

    ordered = bool(np.all(starts[1:] >= starts[:-1]))
    valid_ranges = bool(np.all(starts <= ends))

    within_overlaps = int(np.sum(
        starts[1:] <= ends[:-1]
    ))

    event_hits = event_intersections(starts, ends)
    event_hit_count = int(event_hits.sum())

    expected_duration_ns = (
        (config["sequence_length"] - 1)
        * config["grid_seconds"]
        * 1_000_000_000
    )

    duration_errors = int(np.sum(
        (ends - starts) != expected_duration_ns
    ))

    # Every saved sequence should belong to its
    # declared chronological split.
    split_start, split_end = [
        pd.Timestamp(x).value
        for x in config["split_boundaries"][split]
    ]

    outside_split = int(np.sum(
        (starts < split_start)
        | (ends > split_end)
    ))

    # Audit only the normalized values saved in NPZ.
    # This does not reconstruct original source samples.
    report = {
        "split": split,
        "sequences": int(n),
        "finite_values": finite,
        "expected_shape": shape_ok,
        "chronological_order": ordered,
        "valid_start_end": valid_ranges,
        "within_split_overlaps": within_overlaps,
        "buffered_event_intersections": event_hit_count,
        "unexpected_sequence_durations": duration_errors,
        "outside_split_bounds": outside_split,
    }

    reports.append(report)

    for key, value in report.items():
        print(f"  {key}: {value}")

# =====================================================
# 4. CROSS-SPLIT CHECK
# =====================================================

cross_split = {}

for left, right in [
    ("train", "validation"),
    ("validation", "test")
]:
    left_end = all_data[left]["ends"].max()
    right_start = all_data[right]["starts"].min()

    gap_seconds = (
        right_start - left_end
    ) / 1_000_000_000

    passed = bool(left_end < right_start)

    cross_split[f"{left}_to_{right}"] = {
        "passed": passed,
        "gap_seconds": float(gap_seconds)
    }

    print(
        f"\n{left} -> {right}: "
        f"{'PASS' if passed else 'FAIL'} "
        f"(gap {gap_seconds:.1f} seconds)"
    )

# =====================================================
# 5. SAVE AUDIT REPORT
# =====================================================

df = pd.DataFrame(reports)

df.to_csv(
    OUT / "integrity_audit.csv",
    index=False
)

required_pass = all(
    r["finite_values"]
    and r["expected_shape"]
    and r["chronological_order"]
    and r["valid_start_end"]
    and r["within_split_overlaps"] == 0
    and r["buffered_event_intersections"] == 0
    and r["unexpected_sequence_durations"] == 0
    and r["outside_split_bounds"] == 0
    for r in reports
)

required_pass = required_pass and all(
    result["passed"]
    for result in cross_split.values()
)

summary = {
    "phase": "9A.2",
    "status": (
        "SAVED_DATASET_CHECKS_PASSED"
        if required_pass
        else "AUDIT_ISSUES_FOUND"
    ),
    "source": str(SOURCE),
    "official_unique_events": int(labels["ID"].nunique()),
    "merged_event_intervals": len(merged),
    "reports": reports,
    "cross_split": cross_split,
    "limitations": [
        "Audit checks saved windows, not original sample provenance.",
        "Original observation timestamps are not preserved.",
        "Resampling correctness cannot be proven from NPZ alone.",
        "Normal candidates are not verified clean.",
        "Operational anomalies and cyberattacks are not integrated.",
        "No benchmark-comparable model evaluation performed."
    ]
}

with open(
    OUT / "integrity_audit.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\n========== PHASE 9A.2 COMPLETE ==========")

print("Status:", summary["status"])
print("Audit saved to:", OUT)

if required_pass:
    print("Saved-window integrity checks passed.")
else:
    print("Issues found. Review before proceeding.")
