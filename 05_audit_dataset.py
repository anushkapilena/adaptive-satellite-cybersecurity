
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "phase4"
AUDIT = ROOT / "results" / "phase5"
AUDIT.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(
    OUT / "event_channel_summary.csv"
)

splits = pd.read_csv(
    OUT / "event_split_plan.csv"
)

print("\n========== DATASET QUALITY AUDIT ==========")
print("Event-channel files:", len(summary))
print("Unique events:", summary["event_id"].nunique())
print("Total rows:", int(summary["rows"].sum()))
print("Target-positive rows:", int(summary["target_rows"].sum()))
print(
    "Other-event overlapping rows:",
    int(summary["other_event_rows"].sum())
)

# Event-level coverage
coverage = (
    summary.groupby("event_id")
    .agg(
        channels=("channel", "nunique"),
        rows=("rows", "sum"),
        positive_rows=("target_rows", "sum"),
        other_event_rows=("other_event_rows", "sum"),
        start=("start", "min"),
        end=("end", "max")
    )
    .reset_index()
)

coverage["has_positive_samples"] = (
    coverage["positive_rows"] > 0
)

print("\n========== EVENT COVERAGE ==========")
print(
    "Events with positive samples:",
    int(coverage["has_positive_samples"].sum())
)
print(
    "Events with ZERO positive samples:",
    int((~coverage["has_positive_samples"]).sum())
)

print("\n========== PROVISIONAL SPLITS ==========")
splits["start"] = pd.to_datetime(
    splits["start"], utc=True
)

merged = coverage.merge(
    splits[["event_id", "split"]],
    on="event_id",
    how="left",
    validate="one_to_one"
)

print(
    merged.groupby("split")
    .agg(
        events=("event_id", "nunique"),
        channels=("channels", "sum"),
        positive_rows=("positive_rows", "sum")
    )
)

# Detect temporal overlap between events in different splits.
merged["start"] = pd.to_datetime(
    merged["start"], utc=True
)
merged["end"] = pd.to_datetime(
    merged["end"], utc=True
)

merged = merged.sort_values("start")
overlaps = []

for i in range(len(merged)):
    a = merged.iloc[i]
    for j in range(i + 1, len(merged)):
        b = merged.iloc[j]

        if b["start"] > a["end"] + pd.Timedelta(hours=6):
            break

        if a["split"] == b["split"]:
            continue

        # Windows include 3 hours before and after each event.
        if (
            a["end"] + pd.Timedelta(hours=3)
            >= b["start"] - pd.Timedelta(hours=3)
        ):
            overlaps.append({
                "event_a": a["event_id"],
                "split_a": a["split"],
                "event_b": b["event_id"],
                "split_b": b["split"]
            })

overlap_df = pd.DataFrame(
    overlaps,
    columns=["event_a", "split_a", "event_b", "split_b"]
)

print("\n========== SPLIT LEAKAGE CHECK ==========")
print(
    "Cross-split overlapping event windows:",
    len(overlap_df)
)

coverage.to_csv(
    AUDIT / "event_coverage.csv",
    index=False
)

merged.to_csv(
    AUDIT / "event_coverage_with_splits.csv",
    index=False
)

overlap_df.to_csv(
    AUDIT / "cross_split_overlaps.csv",
    index=False
)

print("\n========== AUDIT COMPLETE ==========")
print("Saved to:", AUDIT)
