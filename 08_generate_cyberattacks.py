
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# PHASE 8: CONTROLLED CYBERATTACK DATASET
# ==================================================

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results" / "phase7" / "ml_sequences.npz"
CONFIG = ROOT / "results" / "phase7" / "config.json"

OUT = ROOT / "results" / "phase8"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)

# Label 1 is reserved for official operational anomalies.
LABELS = {
    0: "Normal Candidate",
    1: "Operational Anomaly",
    2: "Telemetry Manipulation",
    3: "Replay Attack"
}

# Use non-overlapping source windows where possible.
# Each Phase 7 sequence contains 20 readings.
SOURCE_STRIDE = 20

# Number of examples per category, per split.
SAMPLES_PER_CLASS = 300

# Attack affects the final 8 readings of a sequence.
ATTACK_LENGTH = 8

# Minimum observable difference for replay injection.
MIN_REPLAY_DIFFERENCE = 0.05

# ==================================================
# LOAD PHASE 7 DATA
# ==================================================

if not INPUT.exists():
    raise FileNotFoundError(
        "Phase 7 dataset not found. Run Phase 7 first."
    )

with np.load(INPUT) as data:
    datasets = {
        "train": data["X_train"].copy(),
        "validation": data["X_val"].copy(),
        "test": data["X_test"].copy()
    }

with open(CONFIG, "r") as f:
    phase7_config = json.load(f)

CHANNELS = phase7_config["channels"]

print("\n========== PHASE 8 START ==========")
print("Channels:", CHANNELS)
print("Random seed:", SEED)

# ==================================================
# ATTACK 1: TELEMETRY MANIPULATION
# ==================================================

def inject_manipulation(clean):

    attacked = clean.copy()

    n_steps, n_channels = attacked.shape

    start = n_steps - ATTACK_LENGTH

    channel = int(
        RNG.integers(0, n_channels)
    )

    # Values are already standardized in Phase 7.
    # Inject a controlled gradual offset.
    direction = RNG.choice([-1.0, 1.0])

    strength = RNG.uniform(2.5, 4.0)

    offset = np.linspace(
        0.5,
        strength,
        ATTACK_LENGTH,
        dtype=np.float32
    )

    attacked[start:, channel] += (
        direction * offset
    )

    mask = np.zeros(
        (n_steps, n_channels),
        dtype=np.uint8
    )

    mask[start:, channel] = 1

    details = {
        "affected_channel": CHANNELS[channel],
        "attack_start_step": int(start),
        "attack_end_step": int(n_steps - 1),
        "injection_strength": float(strength),
        "replay_source_start": None
    }

    return attacked, mask, details

# ==================================================
# ATTACK 2: REPLAY ATTACK
# ==================================================

def replay_difference(clean):

    source = clean[:ATTACK_LENGTH, :]

    destination = clean[-ATTACK_LENGTH:, :]

    return float(
        np.max(np.abs(source - destination))
    )


def inject_replay(clean):

    attacked = clean.copy()

    n_steps, n_channels = attacked.shape

    start = n_steps - ATTACK_LENGTH

    # Replay an earlier recorded segment into
    # the later part of the same sequence.
    recorded = clean[:ATTACK_LENGTH, :].copy()

    attacked[start:, :] = recorded

    mask = np.zeros(
        (n_steps, n_channels),
        dtype=np.uint8
    )

    mask[start:, :] = 1

    details = {
        "affected_channel": "All selected channels",
        "attack_start_step": int(start),
        "attack_end_step": int(n_steps - 1),
        "injection_strength": None,
        "replay_source_start": 0
    }

    return attacked, mask, details

# ==================================================
# CREATE BALANCED DATASET FOR EACH SPLIT
# ==================================================

def build_split(split_name, original):

    print(f"\nProcessing {split_name}...")

    print("Original shape:", original.shape)

    if original.ndim != 3:
        raise ValueError(
            "Expected (samples, timesteps, channels)"
        )

    if original.shape[1] < 2 * ATTACK_LENGTH:
        raise ValueError(
            "Sequence is too short for replay injection."
        )

    # Phase 7 windows overlap heavily.
    # Taking every 20th window reduces reuse
    # of the same underlying observations.
    candidate_indices = np.arange(
        0,
        len(original),
        SOURCE_STRIDE
    )

    RNG.shuffle(candidate_indices)

    required = 3 * SAMPLES_PER_CLASS

    if len(candidate_indices) < required:
        raise ValueError(
            f"{split_name}: Need {required} distinct "
            f"source windows, found {len(candidate_indices)}."
        )

    # Choose replay candidates with an observable
    # difference between source and destination.
    replay_candidates = [
        int(i)
        for i in candidate_indices
        if replay_difference(original[i])
        >= MIN_REPLAY_DIFFERENCE
    ]

    if len(replay_candidates) < SAMPLES_PER_CLASS:
        raise ValueError(
            f"{split_name}: Not enough observable "
            "replay candidates. Review the threshold."
        )

    replay_ids = replay_candidates[
        :SAMPLES_PER_CLASS
    ]

    replay_set = set(replay_ids)

    remaining = [
        int(i)
        for i in candidate_indices
        if int(i) not in replay_set
    ]

    normal_ids = remaining[
        :SAMPLES_PER_CLASS
    ]

    manipulation_ids = remaining[
        SAMPLES_PER_CLASS:
        2 * SAMPLES_PER_CLASS
    ]

    X = []
    X_clean = []
    y = []
    masks = []
    source_ids = []
    manifest = []

    groups = [
        (0, normal_ids),
        (2, manipulation_ids),
        (3, replay_ids)
    ]

    for label, indices in groups:

        for source_index in indices:

            clean = original[
                source_index
            ].astype(np.float32).copy()

            if label == 0:

                attacked = clean.copy()

                mask = np.zeros_like(
                    clean,
                    dtype=np.uint8
                )

                details = {
                    "affected_channel": "None",
                    "attack_start_step": None,
                    "attack_end_step": None,
                    "injection_strength": None,
                    "replay_source_start": None
                }

            elif label == 2:

                attacked, mask, details = (
                    inject_manipulation(clean)
                )

            elif label == 3:

                attacked, mask, details = (
                    inject_replay(clean)
                )

            X.append(attacked)
            X_clean.append(clean)
            y.append(label)
            masks.append(mask)
            source_ids.append(source_index)

            record = {
                "split": split_name,
                "source_sequence_index": source_index,
                "label": label,
                "category": LABELS[label],
                "max_absolute_change": float(
                    np.max(
                        np.abs(attacked - clean)
                    )
                ),
                **details
            }

            manifest.append(record)

    X = np.stack(X).astype(np.float32)
    X_clean = np.stack(X_clean).astype(np.float32)

    y = np.asarray(
        y,
        dtype=np.int64
    )

    masks = np.stack(masks)

    source_ids = np.asarray(
        source_ids,
        dtype=np.int64
    )

    # Shuffle examples without losing alignment.
    order = RNG.permutation(len(y))

    X = X[order]
    X_clean = X_clean[order]
    y = y[order]
    masks = masks[order]
    source_ids = source_ids[order]

    manifest = pd.DataFrame(
        manifest
    ).iloc[order].reset_index(drop=True)

    manifest.insert(
        0,
        "sample_id",
        np.arange(len(manifest))
    )

    # Save labelled dataset.
    np.savez_compressed(
        OUT / f"{split_name}_attacks.npz",
        X=X,
        X_clean=X_clean,
        y=y,
        injection_mask=masks,
        source_sequence_index=source_ids
    )

    manifest.to_csv(
        OUT / f"{split_name}_manifest.csv",
        index=False
    )

    print("Generated shape:", X.shape)

    for label in [0, 2, 3]:

        count = int(
            np.sum(y == label)
        )

        print(
            f"  {LABELS[label]}: {count}"
        )

    return X, X_clean, y, manifest

# ==================================================
# GENERATE ALL THREE SPLITS
# ==================================================

outputs = {}

for split_name, original in datasets.items():

    outputs[split_name] = build_split(
        split_name,
        original
    )

# ==================================================
# SAVE EXAMPLE ATTACK PLOTS
# ==================================================

def plot_example(label, filename):

    X, X_clean, y, manifest = outputs["test"]

    matches = np.flatnonzero(
        y == label
    )

    if len(matches) == 0:
        return

    idx = int(matches[0])

    clean = X_clean[idx]
    attacked = X[idx]

    fig, axes = plt.subplots(
        len(CHANNELS),
        1,
        figsize=(11, 7),
        sharex=True
    )

    for c, channel in enumerate(CHANNELS):

        axes[c].plot(
            clean[:, c],
            label="Original telemetry",
            linewidth=2
        )

        axes[c].plot(
            attacked[:, c],
            label="Attack-injected telemetry",
            linestyle="--",
            linewidth=2
        )

        axes[c].axvspan(
            clean.shape[0] - ATTACK_LENGTH,
            clean.shape[0] - 1,
            alpha=0.15,
            color="red",
            label="Injection interval"
        )

        axes[c].set_ylabel(channel)
        axes[c].grid(alpha=0.3)

    axes[0].legend(
        loc="best",
        fontsize=8
    )

    axes[-1].set_xlabel(
        "Sequence step (90-second grid)"
    )

    fig.suptitle(
        f"Controlled {LABELS[label]} Example"
    )

    fig.tight_layout()

    fig.savefig(
        OUT / filename,
        dpi=160
    )

    plt.close(fig)


plot_example(
    2,
    "telemetry_manipulation_example.png"
)

plot_example(
    3,
    "replay_attack_example.png"
)

# ==================================================
# SAVE CONFIGURATION
# ==================================================

metadata = {
    "phase": 8,
    "dataset": "ESA-AD v2 Mission 1",
    "channels": CHANNELS,
    "random_seed": SEED,
    "source": "Phase 7 normal-candidate sequences",
    "source_stride": SOURCE_STRIDE,
    "samples_per_class_per_split": SAMPLES_PER_CLASS,
    "attack_length_steps": ATTACK_LENGTH,
    "grid_seconds": phase7_config["grid_seconds"],
    "labels": {
        str(k): v
        for k, v in LABELS.items()
    },
    "generated_labels": [0, 2, 3],
    "reserved_label": 1,
    "limitations": [
        "This is a controlled synthetic attack prototype.",
        "Operational anomaly examples are not yet integrated.",
        "Normal candidates are not verified clean telemetry.",
        "Replay copies an earlier segment from the same window.",
        "Synthetic attacks do not establish real cyber attribution.",
        "Source timestamps are not included in Phase 7 NPZ.",
        "Final evaluation requires event-aware temporal sampling."
    ]
}

with open(
    OUT / "attack_config.json",
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )

# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n========== PHASE 8 COMPLETE ==========")

total = 0

for split_name, result in outputs.items():

    X, X_clean, y, manifest = result

    print(
        f"{split_name}: {X.shape}"
    )

    total += len(y)

print("\nTotal labelled examples:", total)

print("\nSaved folder:", OUT)

print("\nGenerated files:")
for file in sorted(OUT.iterdir()):
    print("-", file.name)

print("\nIMPORTANT:")
print("Label 1 (Operational Anomaly) is reserved.")
print("It has NOT been generated in Phase 8.")
print("Final event-aware evaluation is pending.")
