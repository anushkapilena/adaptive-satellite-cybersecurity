
from pathlib import Path
from zipfile import ZipFile
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ESA-Mission1"
OUT = ROOT / "results" / "phase7"
OUT.mkdir(parents=True, exist_ok=True)

CHANNELS = [
    "channel_14",
    "channel_21",
    "channel_29"
]

GRID_SECONDS = 90
MAX_AGE_SECONDS = 180
SAFETY_HOURS = 3
SEQ_LEN = 20

MAX_SAMPLES_PER_SPLIT = 20000
CHUNK_DAYS = 14

# ==================================================
# LOAD OFFICIAL EVENT LABELS
# ==================================================

labels = pd.read_csv(DATA / "labels.csv")
types = pd.read_csv(DATA / "anomaly_types.csv")

labels["StartTime"] = pd.to_datetime(
    labels["StartTime"],
    utc=True
)

labels["EndTime"] = pd.to_datetime(
    labels["EndTime"],
    utc=True
)

events = labels.merge(
    types[["ID", "Category"]],
    on="ID",
    how="left",
    validate="many_to_one"
)

# ==================================================
# LOAD SELECTED TELEMETRY CHANNELS
# ==================================================

series_dict = {}

for channel in CHANNELS:

    print("Loading:", channel)

    path = DATA / "channels" / f"{channel}.zip"

    # Load only trusted official ESA dataset files.
    with ZipFile(path) as z:
        with z.open(channel) as f:
            df = pd.read_pickle(f)

    df.index = pd.to_datetime(df.index)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    s = df[channel].sort_index()

    if s.index.has_duplicates:
        s = s.groupby(level=0).last()

    series_dict[channel] = s

# ==================================================
# GLOBAL TIME RANGE AND CHRONOLOGICAL SPLITS
# ==================================================

start = max(
    s.index.min()
    for s in series_dict.values()
)

end = min(
    s.index.max()
    for s in series_dict.values()
)

print("\nCommon time range:", start, "to", end)

train_end = start + (end - start) * 0.60
val_end = start + (end - start) * 0.80

print("Train boundary:", train_end)
print("Validation boundary:", val_end)

# ==================================================
# EXCLUDE OFFICIAL LABELLED EVENTS
# ==================================================

# All labelled categories are excluded:
# Anomaly, Rare Event, Communication Gap.

relevant = events[
    events["Channel"].isin(CHANNELS)
].copy()

intervals = []

for row in relevant.itertuples(index=False):

    interval_start = (
        row.StartTime
        - pd.Timedelta(hours=SAFETY_HOURS)
    )

    interval_end = (
        row.EndTime
        + pd.Timedelta(hours=SAFETY_HOURS)
    )

    intervals.append(
        (interval_start, interval_end)
    )

intervals.sort()

# Merge overlapping intervals.
merged_intervals = []

for a, b in intervals:

    if (
        not merged_intervals
        or a > merged_intervals[-1][1]
    ):
        merged_intervals.append([a, b])

    else:
        merged_intervals[-1][1] = max(
            merged_intervals[-1][1],
            b
        )

print(
    "Merged excluded intervals:",
    len(merged_intervals)
)

# ==================================================
# BUILD MULTIVARIATE NORMAL-CANDIDATE SEQUENCES
# ==================================================

def make_sequences(split_start, split_end, split_name):

    print("\nProcessing:", split_name)

    samples = []
    sample_times = []

    current = split_start

    while current < split_end:

        chunk_end = min(
            current + pd.Timedelta(days=CHUNK_DAYS),
            split_end
        )

        grid = pd.date_range(
            start=current,
            end=chunk_end,
            freq=f"{GRID_SECONDS}s",
            tz="UTC"
        )

        aligned = pd.DataFrame(index=grid)

        # Align channels using past observations only.
        for channel, s in series_dict.items():

            lookback_start = (
                current
                - pd.Timedelta(
                    seconds=MAX_AGE_SECONDS
                )
            )

            local = s.loc[
                lookback_start:chunk_end
            ]

            aligned[channel] = local.reindex(
                grid,
                method="ffill",
                tolerance=pd.Timedelta(
                    seconds=MAX_AGE_SECONDS
                )
            )

        # FIX 1: Writable NumPy array.
        valid = (
            aligned.notna()
            .all(axis=1)
            .to_numpy(copy=True)
        )

        timestamps = aligned.index

        # Remove timestamps near labelled events.
        for a, b in merged_intervals:

            if b < current:
                continue

            if a > chunk_end:
                break

            valid &= ~(
                (timestamps >= a)
                & (timestamps <= b)
            )

        values = aligned[CHANNELS].to_numpy(
            dtype=np.float32
        )

        # Generate sequences without crossing
        # missing or excluded timestamps.
        for i in range(
            SEQ_LEN - 1,
            len(values)
        ):

            window_start = (
                i - SEQ_LEN + 1
            )

            if not valid[
                window_start:i + 1
            ].all():
                continue

            samples.append(
                values[
                    window_start:i + 1
                ].copy()
            )

            sample_times.append(
                timestamps[i]
            )

            if (
                len(samples)
                >= MAX_SAMPLES_PER_SPLIT
            ):
                break

        if (
            len(samples)
            >= MAX_SAMPLES_PER_SPLIT
        ):
            break

        # FIX 2: No unnecessary gap.
        current = chunk_end

    if not samples:
        raise RuntimeError(
            f"No valid sequences for {split_name}"
        )

    return (
        np.stack(samples),
        pd.DatetimeIndex(sample_times)
    )

# ==================================================
# GENERATE CHRONOLOGICAL DATASETS
# ==================================================

X_train, t_train = make_sequences(
    start,
    train_end,
    "train"
)

X_val, t_val = make_sequences(
    train_end,
    val_end,
    "validation"
)

X_test, t_test = make_sequences(
    val_end,
    end,
    "test"
)

# ==================================================
# TRAINING-ONLY STANDARDIZATION
# ==================================================

scaler = StandardScaler()

scaler.fit(
    X_train.reshape(
        -1,
        len(CHANNELS)
    )
)

def scale_sequences(X):

    original_shape = X.shape

    scaled = scaler.transform(
        X.reshape(
            -1,
            len(CHANNELS)
        )
    )

    return scaled.reshape(
        original_shape
    ).astype(np.float32)

X_train = scale_sequences(X_train)
X_val = scale_sequences(X_val)
X_test = scale_sequences(X_test)

# ==================================================
# SAVE ML DATASETS
# ==================================================

np.savez_compressed(
    OUT / "ml_sequences.npz",
    X_train=X_train,
    X_val=X_val,
    X_test=X_test
)

config = {
    "channels": CHANNELS,
    "sequence_length": SEQ_LEN,
    "grid_seconds": GRID_SECONDS,
    "max_age_seconds": MAX_AGE_SECONDS,
    "safety_hours": SAFETY_HOURS,
    "chunk_days": CHUNK_DAYS,
    "train_boundary": str(train_end),
    "validation_boundary": str(val_end),
    "train_sequences": len(X_train),
    "validation_sequences": len(X_val),
    "test_sequences": len(X_test),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "note": (
        "Normal-candidate multivariate sequence "
        "prototype. Final anomaly-event evaluation "
        "and temporal coverage checks pending."
    )
}

with open(
    OUT / "config.json",
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=2
    )

# ==================================================
# FINAL OUTPUT
# ==================================================

print("\n========== PHASE 7 COMPLETE ==========")

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\nSaved files:")
print("-", OUT / "ml_sequences.npz")
print("-", OUT / "config.json")

print("\nNOTE:")
print("These are normal-candidate sequences.")
print("Final event-level testing is still pending.")
