
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "phase9a"
OUT = ROOT / "results" / "phase9a1"
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]

print("\n========== PHASE 9A.1 START ==========")

summary = {}

for split in SPLITS:
    print(f"\nChecking {split}...")

    with np.load(SOURCE / f"{split}_normal.npz") as data:
        X = data["X"]
        starts = data["start_ns"]
        ends = data["end_ns"]

    assert len(X) == len(starts) == len(ends)
    assert np.all(starts <= ends)

    # Sort all sequences chronologically.
    order = np.argsort(starts, kind="stable")

    X = X[order]
    starts = starts[order]
    ends = ends[order]

    # Greedy selection of non-overlapping intervals.
    keep = []
    last_end = None

    for i in range(len(starts)):
        if last_end is None or starts[i] > last_end:
            keep.append(i)
            last_end = ends[i]

    keep = np.asarray(keep, dtype=np.int64)

    X_clean = X[keep]
    starts_clean = starts[keep]
    ends_clean = ends[keep]

    # Verify that no retained windows overlap.
    remaining_overlaps = int(np.sum(
        starts_clean[1:] <= ends_clean[:-1]
    ))

    assert remaining_overlaps == 0

    # Save corrected sequences.
    np.savez_compressed(
        OUT / f"{split}_normal.npz",
        X=X_clean,
        start_ns=starts_clean,
        end_ns=ends_clean
    )

    # Save corrected manifest.
    manifest = pd.DataFrame({
        "sequence_id": np.arange(len(X_clean)),
        "start_time": pd.to_datetime(
            starts_clean, utc=True
        ),
        "end_time": pd.to_datetime(
            ends_clean, utc=True
        ),
        "ground_truth_label": 0,
        "label_description": "Normal Candidate"
    })

    manifest.to_csv(
        OUT / f"{split}_manifest.csv",
        index=False
    )

    removed = len(X) - len(X_clean)

    summary[split] = {
        "before": int(len(X)),
        "after": int(len(X_clean)),
        "removed": int(removed),
        "remaining_overlaps": remaining_overlaps
    }

    print("Before:", len(X))
    print("After:", len(X_clean))
    print("Removed:", removed)
    print("Remaining overlaps:", remaining_overlaps)

# Cross-split chronological separation.
for left, right in [
    ("train", "validation"),
    ("validation", "test")
]:
    with np.load(OUT / f"{left}_normal.npz") as a:
        last_end = a["end_ns"].max()

    with np.load(OUT / f"{right}_normal.npz") as b:
        first_start = b["start_ns"].min()

    assert last_end < first_start

    print(f"{left} -> {right}: PASS")

# Preserve original scaler metadata.
with open(SOURCE / "config.json") as f:
    config = json.load(f)

config["phase"] = "9A.1"
config["parent_phase"] = "9A"
config["overlap_correction"] = summary
config["limitations"] = config.get("limitations", []) + [
    "This correction removes overlapping saved windows.",
    "Global resampling-grid consistency is not yet audited.",
    "Source observation timestamps are not preserved.",
    "Normal-candidate labels are not verified clean.",
    "Operational anomaly integration is pending."
]

with open(OUT / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n========== PHASE 9A.1 COMPLETE ==========")
print("Saved:", OUT)
print("Original Phase 9A files unchanged.")
