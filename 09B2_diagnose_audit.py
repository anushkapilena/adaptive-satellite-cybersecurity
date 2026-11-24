
from pathlib import Path
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
NORMAL = ROOT / "results" / "phase9a1"
ATTACK = ROOT / "results" / "phase9b"

SPLITS = ["train", "validation", "test"]

print("\n========== PHASE 9B.2 DIAGNOSIS ==========")


def parse_ns(value):
    """Parse integer or scientific-notation timestamps."""
    if pd.isna(value) or str(value).strip() == "":
        return None

    try:
        return int(Decimal(str(value)))
    except (InvalidOperation, ValueError):
        return None


for split in SPLITS:

    print(f"\n========== {split.upper()} ==========")

    with np.load(NORMAL / f"{split}_normal.npz") as d:
        normal_X = d["X"]
        normal_starts = d["start_ns"]
        normal_ends = d["end_ns"]

    with np.load(ATTACK / f"{split}_attacks.npz") as d:
        attack_X = d["X"]
        attack_y = d["y"]

    manifest = pd.read_csv(
        ATTACK / f"{split}_attack_manifest.csv",
        dtype=str
    )

    counts = {
        "normal_value_errors": 0,
        "manipulation_value_errors": 0,
        "replay_value_errors": 0,
        "target_timestamp_errors": 0,
        "attack_timestamp_errors": 0,
        "replay_source_timestamp_errors": 0,
        "label_errors": 0
    }

    for i, row in manifest.iterrows():

        label = int(row["label"])
        target_idx = int(row["target_sequence_idx"])

        original = normal_X[target_idx]
        attacked = attack_X[i]

        if int(attack_y[i]) != label:
            counts["label_errors"] += 1

        if (
            parse_ns(row["target_start_ns"])
            != int(normal_starts[target_idx])
            or parse_ns(row["target_end_ns"])
            != int(normal_ends[target_idx])
        ):
            counts["target_timestamp_errors"] += 1

        if label == 0:

            if not np.array_equal(attacked, original):
                counts["normal_value_errors"] += 1

        elif label == 2:

            channel = row["modified_channels"]
            channels = ["channel_14", "channel_21", "channel_29"]

            if channel not in channels:
                counts["manipulation_value_errors"] += 1
                continue

            channel_idx = channels.index(channel)

            before_ok = np.array_equal(
                attacked[:12], original[:12]
            )

            other_channels = [
                j for j in range(3)
                if j != channel_idx
            ]

            others_ok = np.array_equal(
                attacked[:, other_channels],
                original[:, other_channels]
            )

            changed = not np.array_equal(
                attacked[12:, channel_idx],
                original[12:, channel_idx]
            )

            if not (before_ok and others_ok and changed):
                counts["manipulation_value_errors"] += 1

        elif label == 3:

            source_idx = int(Decimal(row["replay_source_idx"]))
            historical = normal_X[source_idx]

            before_ok = np.array_equal(
                attacked[:12], original[:12]
            )

            replay_ok = np.array_equal(
                attacked[12:], historical[:8]
            )

            if not (before_ok and replay_ok):
                counts["replay_value_errors"] += 1

            # Compare source timestamps with authoritative
            # original NPZ metadata.
            expected_source_start = int(
                normal_starts[source_idx]
            )

            if (
                parse_ns(row["replay_source_start_ns"])
                != expected_source_start
            ):
                counts["replay_source_timestamp_errors"] += 1

        if label in (2, 3):

            expected_attack_start = (
                int(normal_starts[target_idx])
                + 12 * 90 * 1_000_000_000
            )

            expected_attack_end = int(
                normal_ends[target_idx]
            )

            if (
                parse_ns(row["attack_start_ns"])
                != expected_attack_start
                or parse_ns(row["attack_end_ns"])
                != expected_attack_end
            ):
                counts["attack_timestamp_errors"] += 1

    for key, value in counts.items():
        print(f"{key}: {value}")

print("\n========== DIAGNOSIS COMPLETE ==========")
