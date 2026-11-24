
from pathlib import Path
import json

import numpy as np
import pandas as pd

# ============================================================
# PHASE 9B - CONTROLLED CYBERATTACK GENERATION
# ============================================================

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "phase9a1"
OUT = ROOT / "results" / "phase9b"
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]
SEED = 42
SAMPLES_PER_CLASS = 300

# Attack begins at timestep 12 of a 20-step sequence.
ATTACK_START_STEP = 12

# Replay source must end at least one day before target.
MIN_REPLAY_HISTORY_HOURS = 24

# Manipulation size, measured in training-scaled units.
MANIPULATION_MIN = 2.5
MANIPULATION_MAX = 4.0

rng = np.random.default_rng(SEED)

print("\n========== PHASE 9B START ==========")

with open(SOURCE / "config.json", encoding="utf-8") as f:
    source_config = json.load(f)

CHANNELS = source_config["channels"]
SEQ_LEN = source_config["sequence_length"]

assert SEQ_LEN > ATTACK_START_STEP
assert len(CHANNELS) > 0

MIN_HISTORY_NS = (
    MIN_REPLAY_HISTORY_HOURS * 3600 * 1_000_000_000
)

all_summaries = {}

# ============================================================
# 1. ATTACK FUNCTIONS
# ============================================================

def inject_manipulation(original):
    """
    Gradually alter one channel over the final timesteps.
    The original array is never modified.
    """
    attacked = original.copy()

    channel_idx = int(rng.integers(0, len(CHANNELS)))

    magnitude = float(
        rng.uniform(MANIPULATION_MIN, MANIPULATION_MAX)
    )

    direction = int(rng.choice([-1, 1]))

    attack_steps = SEQ_LEN - ATTACK_START_STEP

    offset = np.linspace(
        magnitude * 0.25,
        magnitude,
        attack_steps,
        dtype=np.float32
    )

    attacked[ATTACK_START_STEP:, channel_idx] += (
        direction * offset
    )

    return attacked, {
        "modified_channel": CHANNELS[channel_idx],
        "magnitude_scaled_units": magnitude,
        "direction": direction
    }


def inject_replay(target, historical):
    """
    Replace the final part of a target window with readings
    from a genuinely earlier saved telemetry window.
    """
    attacked = target.copy()

    replay_length = SEQ_LEN - ATTACK_START_STEP

    # Use an earlier contiguous segment from historical data.
    attacked[ATTACK_START_STEP:] = historical[
        :replay_length
    ]

    return attacked


# ============================================================
# 2. PROCESS EACH CHRONOLOGICAL SPLIT
# ============================================================

for split in SPLITS:

    print(f"\nProcessing: {split}")

    with np.load(SOURCE / f"{split}_normal.npz") as data:
        X = data["X"].astype(np.float32)
        starts = data["start_ns"].astype(np.int64)
        ends = data["end_ns"].astype(np.int64)

    assert len(X) == len(starts) == len(ends)
    assert X.shape[1:] == (SEQ_LEN, len(CHANNELS))
    assert np.isfinite(X).all()
    assert np.all(starts[1:] >= starts[:-1])

    # A replay target must have an earlier historical source
    # within the same split.
    eligible = []

    for i in range(len(X)):
        latest_allowed_end = starts[i] - MIN_HISTORY_NS

        # Because ends are chronological, this gives the
        # number of windows ending before the cutoff.
        source_count = int(
            np.searchsorted(
                ends,
                latest_allowed_end,
                side="right"
            )
        )

        if source_count > 0:
            eligible.append(i)

    eligible = np.asarray(eligible, dtype=np.int64)

    if len(eligible) < SAMPLES_PER_CLASS:
        raise RuntimeError(
            f"{split}: only {len(eligible)} eligible "
            f"replay targets; need {SAMPLES_PER_CLASS}"
        )

    # Spread targets across the eligible chronological range.
    target_positions = np.linspace(
        0,
        len(eligible) - 1,
        SAMPLES_PER_CLASS,
        dtype=np.int64
    )

    target_indices = eligible[target_positions]

    if len(np.unique(target_indices)) != SAMPLES_PER_CLASS:
        raise RuntimeError("Duplicate target indices selected")

    normal_samples = []
    manipulation_samples = []
    replay_samples = []

    manifest = []

    for sample_number, target_idx in enumerate(target_indices):

        target_idx = int(target_idx)
        original = X[target_idx]

        start_ns = int(starts[target_idx])
        end_ns = int(ends[target_idx])

        step_ns = (
            source_config["grid_seconds"]
            * 1_000_000_000
        )

        attack_start_ns = (
            start_ns + ATTACK_START_STEP * step_ns
        )

        attack_end_ns = end_ns

        # --------------------------------------------
        # Class 0: normal candidate
        # --------------------------------------------
        normal_samples.append(original.copy())

        manifest.append({
            "sample_id": f"{split}_{sample_number:04d}_normal",
            "label": 0,
            "class_name": "normal_candidate",
            "target_sequence_idx": target_idx,
            "target_start_ns": start_ns,
            "target_end_ns": end_ns,
            "attack_start_ns": None,
            "attack_end_ns": None,
            "modified_channels": "",
            "replay_source_idx": None,
            "replay_source_start_ns": None,
            "replay_source_end_ns": None,
            "manipulation_magnitude": None,
            "manipulation_direction": None
        })

        # --------------------------------------------
        # Class 2: telemetry manipulation
        # --------------------------------------------
        manipulated, info = inject_manipulation(original)

        if np.array_equal(manipulated, original):
            raise RuntimeError("Manipulation had no effect")

        manipulation_samples.append(manipulated)

        manifest.append({
            "sample_id": f"{split}_{sample_number:04d}_manip",
            "label": 2,
            "class_name": "telemetry_manipulation",
            "target_sequence_idx": target_idx,
            "target_start_ns": start_ns,
            "target_end_ns": end_ns,
            "attack_start_ns": attack_start_ns,
            "attack_end_ns": attack_end_ns,
            "modified_channels": info["modified_channel"],
            "replay_source_idx": None,
            "replay_source_start_ns": None,
            "replay_source_end_ns": None,
            "manipulation_magnitude":
                info["magnitude_scaled_units"],
            "manipulation_direction": info["direction"]
        })

        # --------------------------------------------
        # Class 3: historical telemetry replay
        # --------------------------------------------
        latest_allowed_end = start_ns - MIN_HISTORY_NS

        source_count = int(
            np.searchsorted(
                ends,
                latest_allowed_end,
                side="right"
            )
        )

        if source_count <= 0:
            raise RuntimeError("No valid historical source")

        source_idx = int(rng.integers(0, source_count))

        historical = X[source_idx]

        replayed = inject_replay(original, historical)

        if np.array_equal(replayed, original):
            raise RuntimeError("Replay had no effect")

        if int(ends[source_idx]) > latest_allowed_end:
            raise RuntimeError("Replay history check failed")

        replay_samples.append(replayed)

        # Source segment is the first 8 samples of
        # the historical sequence.
        replay_length = SEQ_LEN - ATTACK_START_STEP

        source_segment_end_ns = int(
            starts[source_idx]
            + (replay_length - 1) * step_ns
        )

        manifest.append({
            "sample_id": f"{split}_{sample_number:04d}_replay",
            "label": 3,
            "class_name": "historical_replay",
            "target_sequence_idx": target_idx,
            "target_start_ns": start_ns,
            "target_end_ns": end_ns,
            "attack_start_ns": attack_start_ns,
            "attack_end_ns": attack_end_ns,
            "modified_channels": ";".join(CHANNELS),
            "replay_source_idx": source_idx,
            "replay_source_start_ns":
                int(starts[source_idx]),
            "replay_source_end_ns":
                source_segment_end_ns,
            "manipulation_magnitude": None,
            "manipulation_direction": None
        })

    # ====================================================
    # 3. COMBINE AND SHUFFLE LABELLED SAMPLES
    # ====================================================

    X_all = np.concatenate([
        np.stack(normal_samples),
        np.stack(manipulation_samples),
        np.stack(replay_samples)
    ]).astype(np.float32)

    y_all = np.concatenate([
        np.zeros(SAMPLES_PER_CLASS, dtype=np.int64),
        np.full(SAMPLES_PER_CLASS, 2, dtype=np.int64),
        np.full(SAMPLES_PER_CLASS, 3, dtype=np.int64)
    ])

    # Manifest was constructed in triplets; reorder it
    # into the same class-block order as X_all.
    rows = pd.DataFrame(manifest)

    ordered_rows = pd.concat([
        rows[rows["label"] == 0],
        rows[rows["label"] == 2],
        rows[rows["label"] == 3]
    ], ignore_index=True)

    assert len(ordered_rows) == len(y_all)
    assert np.array_equal(
        ordered_rows["label"].to_numpy(dtype=np.int64),
        y_all
    )

    permutation = rng.permutation(len(y_all))

    X_all = X_all[permutation]
    y_all = y_all[permutation]
    ordered_rows = ordered_rows.iloc[
        permutation
    ].reset_index(drop=True)

    assert np.isfinite(X_all).all()
    assert X_all.shape == (
        3 * SAMPLES_PER_CLASS,
        SEQ_LEN,
        len(CHANNELS)
    )

    assert np.array_equal(
        ordered_rows["label"].to_numpy(dtype=np.int64),
        y_all
    )

    # ====================================================
    # 4. SAVE DATA AND MANIFEST
    # ====================================================

    np.savez_compressed(
        OUT / f"{split}_attacks.npz",
        X=X_all,
        y=y_all,
        target_start_ns=ordered_rows[
            "target_start_ns"
        ].to_numpy(dtype=np.int64),
        target_end_ns=ordered_rows[
            "target_end_ns"
        ].to_numpy(dtype=np.int64)
    )

    # Human-readable timestamp columns.
    for col in [
        "target_start_ns",
        "target_end_ns",
        "attack_start_ns",
        "attack_end_ns",
        "replay_source_start_ns",
        "replay_source_end_ns"
    ]:
        ordered_rows[col.replace("_ns", "_utc")] = (
            pd.to_datetime(
                ordered_rows[col],
                unit="ns",
                utc=True,
                errors="coerce"
            )
        )

    ordered_rows.to_csv(
        OUT / f"{split}_attack_manifest.csv",
        index=False
    )

    counts = {
        int(label): int((y_all == label).sum())
        for label in np.unique(y_all)
    }

    all_summaries[split] = {
        "total": int(len(y_all)),
        "shape": list(X_all.shape),
        "class_counts": counts
    }

    print("Output shape:", X_all.shape)
    print("Class counts:", counts)
    print("Historical replay checks: PASS")

# ============================================================
# 5. SAVE CONFIGURATION
# ============================================================

attack_config = {
    "phase": "9B",
    "parent_dataset": "phase9a1",
    "seed": SEED,
    "channels": CHANNELS,
    "sequence_length": SEQ_LEN,
    "samples_per_class_per_split": SAMPLES_PER_CLASS,
    "attack_start_step": ATTACK_START_STEP,
    "minimum_replay_history_hours":
        MIN_REPLAY_HISTORY_HOURS,
    "manipulation_scaled_units": [
        MANIPULATION_MIN,
        MANIPULATION_MAX
    ],
    "class_mapping": {
        "0": "normal_candidate",
        "1": "official_operational_anomaly_reserved",
        "2": "telemetry_manipulation",
        "3": "historical_replay"
    },
    "splits": all_summaries,
    "limitations": [
        "Attacks are controlled synthetic experiments.",
        "Replay uses earlier saved normal-candidate windows.",
        "Replay substitutes the same three channels.",
        "Values are in training-standardized units.",
        "Original source timestamps are reconstructed "
        "from saved sequence metadata.",
        "No actual telecommand injection is performed.",
        "Official operational anomalies are not yet included.",
        "Attack classes may contain detectable synthetic "
        "artifacts; robustness testing is required.",
        "No model performance is established in this phase."
    ]
}

with open(
    OUT / "attack_config.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(attack_config, f, indent=2)

print("\n========== PHASE 9B COMPLETE ==========")
print("Output directory:", OUT)
print("Original datasets remain unchanged.")
