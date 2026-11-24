
from pathlib import Path
from zipfile import ZipFile
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =====================================================
# PHASE 9A - IMPROVED NORMAL-CANDIDATE DATASET
# =====================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ESA-Mission1"
OUT = ROOT / "results" / "phase9a"
OUT.mkdir(parents=True, exist_ok=True)

CHANNELS = ["channel_14", "channel_21", "channel_29"]

GRID_SECONDS = 90
MAX_AGE_SECONDS = 180
SEQ_LEN = 20

EVENT_BUFFER_HOURS = 3
SPLIT_BUFFER_HOURS = 3

TARGET_PER_SPLIT = 20000
CHUNKS_PER_SPLIT = 200
CHUNK_DAYS = 7

SEED = 42
rng = np.random.default_rng(SEED)

print("\n========== PHASE 9A START ==========")

# =====================================================
# 1. LOAD OFFICIAL LABELS
# =====================================================

labels = pd.read_csv(DATA / "labels.csv")

labels["StartTime"] = pd.to_datetime(
    labels["StartTime"], utc=True
)

labels["EndTime"] = pd.to_datetime(
    labels["EndTime"], utc=True
)

# GLOBAL exclusion across ALL official channels.
intervals = []

buffer = pd.Timedelta(hours=EVENT_BUFFER_HOURS)

for row in labels.itertuples(index=False):
    intervals.append((
        row.StartTime - buffer,
        row.EndTime + buffer
    ))

intervals.sort()

merged = []

for a, b in intervals:
    if not merged or a > merged[-1][1]:
        merged.append([a, b])
    else:
        merged[-1][1] = max(merged[-1][1], b)

excluded_starts = np.array(
    [a.value for a, b in merged],
    dtype=np.int64
)

excluded_ends = np.array(
    [b.value for a, b in merged],
    dtype=np.int64
)

print("Official events:", labels["ID"].nunique())
print("Global excluded intervals:", len(merged))

# =====================================================
# 2. LOAD THREE TELEMETRY CHANNELS
# =====================================================

series_dict = {}

for channel in CHANNELS:
    print("Loading:", channel)

    zip_path = DATA / "channels" / f"{channel}.zip"

    # Pickle is used only for trusted official ESA files.
    with ZipFile(zip_path) as z:
        with z.open(channel) as f:
            df = pd.read_pickle(f)

    idx = pd.to_datetime(df.index)

    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    s = pd.Series(
        pd.to_numeric(
            df[channel], errors="coerce"
        ).to_numpy(dtype=np.float32),
        index=idx,
        name=channel
    ).sort_index()

    if s.index.has_duplicates:
        s = s.groupby(level=0).last()

    series_dict[channel] = s

start = max(s.index.min() for s in series_dict.values())
end = min(s.index.max() for s in series_dict.values())

print("Common start:", start)
print("Common end:", end)

# =====================================================
# 3. CHRONOLOGICAL SPLITS WITH PURGE BUFFERS
# =====================================================

boundary1 = start + (end - start) * 0.60
boundary2 = start + (end - start) * 0.80

purge = pd.Timedelta(hours=SPLIT_BUFFER_HOURS)

splits = {
    "train": (start, boundary1 - purge),
    "validation": (boundary1 + purge, boundary2 - purge),
    "test": (boundary2 + purge, end)
}

for name, (a, b) in splits.items():
    print(name, ":", a, "to", b)

# =====================================================
# 4. BUILD TEMPORALLY DISTRIBUTED SEQUENCES
# =====================================================

def globally_valid(times):
    """
    Return False for timestamps inside any globally
    excluded event interval.
    """
    ns = times.asi8

    positions = np.searchsorted(
        excluded_starts,
        ns,
        side="right"
    ) - 1

    valid = np.ones(len(ns), dtype=bool)

    inside_range = positions >= 0

    valid[inside_range] = (
        ns[inside_range]
        > excluded_ends[positions[inside_range]]
    )

    return valid


def build_split(name, split_start, split_end):

    print("\nProcessing:", name)

    # Spread sampling anchors across the full split.
    available_seconds = (
        split_end - split_start
    ).total_seconds()

    chunk_seconds = CHUNK_DAYS * 86400

    if available_seconds <= chunk_seconds:
        raise ValueError("Split shorter than one chunk")

    offsets = np.linspace(
        0,
        available_seconds - chunk_seconds,
        CHUNKS_PER_SPLIT
    )

    target_per_chunk = int(
        np.ceil(TARGET_PER_SPLIT / CHUNKS_PER_SPLIT)
    )

    all_X = []
    all_start = []
    all_end = []

    for chunk_number, offset in enumerate(offsets):

        chunk_start = split_start + pd.Timedelta(
            seconds=float(offset)
        )

        chunk_end = min(
            chunk_start + pd.Timedelta(days=CHUNK_DAYS),
            split_end
        )

        # A fixed 90-second grid within this chunk.
        grid = pd.date_range(
            start=chunk_start,
            end=chunk_end,
            freq=f"{GRID_SECONDS}s"
        )

        aligned = pd.DataFrame(index=grid)

        for channel, s in series_dict.items():

            local = s.loc[
                chunk_start - pd.Timedelta(
                    seconds=MAX_AGE_SECONDS
                ):chunk_end
            ]

            aligned[channel] = local.reindex(
                grid,
                method="ffill",
                tolerance=pd.Timedelta(
                    seconds=MAX_AGE_SECONDS
                )
            )

        values = aligned[CHANNELS].to_numpy(
            dtype=np.float32
        )

        valid = (
            np.isfinite(values).all(axis=1)
            & globally_valid(grid)
        )

        # Non-overlapping candidate windows
        # within each sampled chunk.
        candidates = []

        for window_start in range(
            0,
            len(values) - SEQ_LEN + 1,
            SEQ_LEN
        ):
            window_end = window_start + SEQ_LEN

            if valid[window_start:window_end].all():
                candidates.append(window_start)

        if not candidates:
            continue

        # Distribute selected windows within each chunk.
        take = min(
            target_per_chunk,
            len(candidates)
        )

        chosen_positions = np.linspace(
            0,
            len(candidates) - 1,
            take,
            dtype=int
        )

        for position in chosen_positions:

            i = candidates[position]
            j = i + SEQ_LEN

            all_X.append(values[i:j].copy())

            all_start.append(grid[i].value)
            all_end.append(grid[j - 1].value)

        if (chunk_number + 1) % 25 == 0:
            print(
                f"  Chunks: {chunk_number + 1}/"
                f"{CHUNKS_PER_SPLIT}, "
                f"sequences: {len(all_X)}"
            )

    if not all_X:
        raise RuntimeError(
            f"No valid sequences generated for {name}"
        )

    X = np.stack(all_X).astype(np.float32)

    starts = np.asarray(
        all_start, dtype=np.int64
    )

    ends = np.asarray(
        all_end, dtype=np.int64
    )

    # Deduplicate by exact sequence start/end.
    pairs = np.column_stack((starts, ends))

    _, unique_idx = np.unique(
        pairs,
        axis=0,
        return_index=True
    )

    unique_idx.sort()

    X = X[unique_idx]
    starts = starts[unique_idx]
    ends = ends[unique_idx]

    # Preserve chronological order.
    order = np.argsort(starts)

    X = X[order]
    starts = starts[order]
    ends = ends[order]

    if len(X) > TARGET_PER_SPLIT:
        selected = np.linspace(
            0,
            len(X) - 1,
            TARGET_PER_SPLIT,
            dtype=int
        )

        X = X[selected]
        starts = starts[selected]
        ends = ends[selected]

    print("Final sequences:", X.shape)

    return X, starts, ends


raw = {}

for name, (a, b) in splits.items():
    raw[name] = build_split(name, a, b)

# =====================================================
# 5. CROSS-SPLIT LEAKAGE AUDIT
# =====================================================

def check_separation(left, right):
    left_end = raw[left][2].max()
    right_start = raw[right][1].min()

    if left_end >= right_start:
        raise RuntimeError(
            f"Temporal overlap: {left} and {right}"
        )

    print(
        f"{left} -> {right}: "
        "chronological separation passed"
    )

check_separation("train", "validation")
check_separation("validation", "test")

# Check sequence windows do not overlap
# within a split.
for name, (X, starts, ends) in raw.items():

    order = np.argsort(starts)

    sorted_starts = starts[order]
    sorted_ends = ends[order]

    overlap_count = int(np.sum(
        sorted_starts[1:] <= sorted_ends[:-1]
    ))

    print(
        f"{name} within-split overlapping "
        f"windows: {overlap_count}"
    )

# =====================================================
# 6. TRAINING-ONLY SCALER
# =====================================================

scaler = StandardScaler()

scaler.fit(
    raw["train"][0].reshape(
        -1, len(CHANNELS)
    )
)

def transform(X):
    shape = X.shape

    return scaler.transform(
        X.reshape(-1, len(CHANNELS))
    ).reshape(shape).astype(np.float32)

# =====================================================
# 7. SAVE IMPROVED DATASETS
# =====================================================

for name, (X, starts, ends) in raw.items():

    scaled = transform(X)

    np.savez_compressed(
        OUT / f"{name}_normal.npz",
        X=scaled,
        start_ns=starts,
        end_ns=ends
    )

    manifest = pd.DataFrame({
        "sequence_id": np.arange(len(X)),
        "start_time": pd.to_datetime(
            starts, utc=True
        ),
        "end_time": pd.to_datetime(
            ends, utc=True
        ),
        "ground_truth_label": 0,
        "label_description": "Normal Candidate"
    })

    manifest.to_csv(
        OUT / f"{name}_manifest.csv",
        index=False
    )

config = {
    "phase": "9A",
    "source": "ESA-AD v2 Mission 1",
    "channels": CHANNELS,
    "sequence_length": SEQ_LEN,
    "grid_seconds": GRID_SECONDS,
    "max_observation_age_seconds": MAX_AGE_SECONDS,
    "global_event_buffer_hours": EVENT_BUFFER_HOURS,
    "split_purge_hours": SPLIT_BUFFER_HOURS,
    "sampling_chunks_per_split": CHUNKS_PER_SPLIT,
    "sampling_chunk_days": CHUNK_DAYS,
    "random_seed": SEED,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "split_boundaries": {
        name: [str(a), str(b)]
        for name, (a, b) in splits.items()
    },
    "limitations": [
        "Normal candidates are not verified clean.",
        "Selected channels are only a prototype subset.",
        "Source observation timestamps are not retained.",
        "Attack and operational event integration pending.",
        "This is not a final ESA benchmark reproduction."
    ]
}

with open(OUT / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n========== PHASE 9A COMPLETE ==========")

for name, (X, starts, ends) in raw.items():
    print(
        f"{name}: {X.shape}, "
        f"coverage: "
        f"{pd.Timestamp(starts.min(), tz='UTC')} "
        f"to {pd.Timestamp(ends.max(), tz='UTC')}"
    )

print("\nSaved:", OUT)
print("Original Phase 7 and 8 files unchanged.")
