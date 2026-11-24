
from pathlib import Path
from zipfile import ZipFile
import json

import numpy as np
import pandas as pd

# ======================================================
# PHASE 9A.3 - SOURCE TIMESTAMP & RESAMPLING AUDIT
# ======================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ESA-Mission1"
SOURCE = ROOT / "results" / "phase9a1"
OUT = ROOT / "results" / "phase9a3"
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]
SAMPLES_PER_SPLIT = 100
SEED = 42

rng = np.random.default_rng(SEED)

print("\n========== PHASE 9A.3 START ==========")

with open(SOURCE / "config.json", encoding="utf-8") as f:
    config = json.load(f)

CHANNELS = config["channels"]
MAX_AGE_SECONDS = config["max_observation_age_seconds"]
GRID_SECONDS = config["grid_seconds"]
SEQ_LEN = config["sequence_length"]

scaler_mean = np.asarray(
    config["scaler_mean"], dtype=np.float64
)
scaler_scale = np.asarray(
    config["scaler_scale"], dtype=np.float64
)

# ======================================================
# 1. LOAD ORIGINAL TELEMETRY
# ======================================================

original = {}

for channel in CHANNELS:
    print("Loading original:", channel)

    zip_path = DATA / "channels" / f"{channel}.zip"

    # Only unpickle trusted official ESA data.
    with ZipFile(zip_path) as z:
        with z.open(channel) as f:
            df = pd.read_pickle(f)

    idx = pd.to_datetime(df.index)

    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    series = pd.Series(
        pd.to_numeric(
            df[channel], errors="coerce"
        ).to_numpy(dtype=np.float32),
        index=idx
    ).sort_index()

    if series.index.has_duplicates:
        series = series.groupby(level=0).last()

    original[channel] = series

# ======================================================
# 2. AUDIT RANDOM SAVED SEQUENCES
# ======================================================

records = []

for split in SPLITS:

    print(f"\nAuditing split: {split}")

    with np.load(SOURCE / f"{split}_normal.npz") as data:
        X = data["X"]
        starts = data["start_ns"]
        ends = data["end_ns"]

    count = min(SAMPLES_PER_SPLIT, len(X))

    # Randomly select saved windows for source verification.
    chosen = rng.choice(
        len(X), size=count, replace=False
    )

    for sequence_idx in chosen:

        start = pd.Timestamp(
            int(starts[sequence_idx]), tz="UTC"
        )

        grid = pd.date_range(
            start=start,
            periods=SEQ_LEN,
            freq=f"{GRID_SECONDS}s"
        )

        for channel_idx, channel in enumerate(CHANNELS):

            series = original[channel]

            # Locate latest original observation
            # at or before each grid timestamp.
            positions = series.index.searchsorted(
                grid, side="right"
            ) - 1

            for timestep, position in enumerate(positions):

                grid_time = grid[timestep]

                record = {
                    "split": split,
                    "sequence_idx": int(sequence_idx),
                    "channel": channel,
                    "timestep": int(timestep),
                    "grid_time": str(grid_time),
                    "source_time": None,
                    "observation_age_seconds": None,
                    "future_source_used": False,
                    "stale_source_used": False,
                    "value_matches": False
                }

                if position < 0:
                    records.append(record)
                    continue

                source_time = series.index[position]

                age = (
                    grid_time - source_time
                ).total_seconds()

                record["source_time"] = str(source_time)
                record["observation_age_seconds"] = float(age)

                record["future_source_used"] = bool(age < 0)
                record["stale_source_used"] = bool(
                    age > MAX_AGE_SECONDS
                )

                source_value = float(series.iloc[position])

                # Reproduce the normalization applied
                # in Phase 9A.
                expected_scaled = (
                    source_value - scaler_mean[channel_idx]
                ) / scaler_scale[channel_idx]

                actual_scaled = float(
                    X[sequence_idx, timestep, channel_idx]
                )

                record["value_matches"] = bool(
                    np.isclose(
                        actual_scaled,
                        expected_scaled,
                        rtol=1e-4,
                        atol=1e-4
                    )
                )

                records.append(record)

    print("Sequences sampled:", count)

# ======================================================
# 3. SAVE DETAILED AUDIT
# ======================================================

df = pd.DataFrame(records)

df.to_csv(
    OUT / "source_timestamp_audit.csv",
    index=False
)

future_count = int(df["future_source_used"].sum())
stale_count = int(df["stale_source_used"].sum())
mismatch_count = int((~df["value_matches"]).sum())

missing_source_count = int(
    df["source_time"].isna().sum()
)

summary = {
    "phase": "9A.3",
    "sampled_sequences_per_split": SAMPLES_PER_SPLIT,
    "audited_values": int(len(df)),
    "future_source_count": future_count,
    "stale_source_count": stale_count,
    "value_mismatch_count": mismatch_count,
    "missing_source_count": missing_source_count,
    "sampling_type": "random source-level audit",
    "limitations": [
        "This is a sampled audit, not a full-data proof.",
        "Source timestamps are reconstructed for sampled windows.",
        "Full source provenance is not yet stored in NPZ.",
        "Normal-candidate purity is not proven."
    ]
}

passed = (
    future_count == 0
    and stale_count == 0
    and mismatch_count == 0
    and missing_source_count == 0
)

summary["status"] = (
    "SAMPLED_SOURCE_AUDIT_PASSED"
    if passed
    else "SOURCE_AUDIT_ISSUES_FOUND"
)

with open(
    OUT / "source_audit_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\n========== PHASE 9A.3 COMPLETE ==========")

print("Audited values:", len(df))
print("Future source used:", future_count)
print("Stale source used:", stale_count)
print("Value mismatches:", mismatch_count)
print("Missing source:", missing_source_count)
print("Status:", summary["status"])
print("Saved:", OUT)
