
from pathlib import Path
from decimal import Decimal
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

NORMAL_DIR = ROOT / "results" / "phase9a1"
ATTACK_DIR = ROOT / "results" / "phase9b"
BACKUP_DIR = ATTACK_DIR / "original_manifests_backup"

BACKUP_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]

ATTACK_START_STEP = 12
GRID_SECONDS = 90
STEP_NS = GRID_SECONDS * 1_000_000_000
REPLAY_LENGTH = 8

print("\n========== PHASE 9B.3 START ==========")


def parse_index(value):
    if pd.isna(value) or str(value).strip() == "":
        return None

    number = Decimal(str(value))

    if number != number.to_integral_value():
        raise ValueError(f"Non-integer index: {value}")

    return int(number)


def to_utc(ns):
    if ns is None:
        return ""

    return pd.Timestamp(
        int(ns), unit="ns", tz="UTC"
    ).isoformat()


for split in SPLITS:

    print(f"\nRepairing: {split}")

    with np.load(
        NORMAL_DIR / f"{split}_normal.npz"
    ) as d:
        starts = d["start_ns"]
        ends = d["end_ns"]

    with np.load(
        ATTACK_DIR / f"{split}_attacks.npz"
    ) as d:
        attack_y = d["y"]
        attack_starts = d["target_start_ns"]
        attack_ends = d["target_end_ns"]

    manifest_path = (
        ATTACK_DIR / f"{split}_attack_manifest.csv"
    )

    # Read as text to avoid further numeric conversions.
    manifest = pd.read_csv(
        manifest_path,
        dtype=str,
        keep_default_na=False
    )

    if len(manifest) != len(attack_y):
        raise RuntimeError(
            f"{split}: manifest length mismatch"
        )

    # Save original manifest before changing anything.
    backup_path = BACKUP_DIR / manifest_path.name

    if not backup_path.exists():
        shutil.copy2(manifest_path, backup_path)

    repaired = []

    for i, row in manifest.iterrows():

        label = parse_index(row["label"])
        target_idx = parse_index(
            row["target_sequence_idx"]
        )

        if label not in (0, 2, 3):
            raise RuntimeError(
                f"{split} row {i}: invalid label {label}"
            )

        if target_idx is None or not (
            0 <= target_idx < len(starts)
        ):
            raise RuntimeError(
                f"{split} row {i}: invalid target index"
            )

        target_start = int(starts[target_idx])
        target_end = int(ends[target_idx])

        if (
            int(attack_y[i]) != label
            or int(attack_starts[i]) != target_start
            or int(attack_ends[i]) != target_end
        ):
            raise RuntimeError(
                f"{split} row {i}: NPZ provenance mismatch"
            )

        attack_start = None
        attack_end = None
        source_start = None
        source_end = None

        if label in (2, 3):
            attack_start = (
                target_start
                + ATTACK_START_STEP * STEP_NS
            )
            attack_end = target_end

        if label == 3:

            source_idx = parse_index(
                row["replay_source_idx"]
            )

            if source_idx is None or not (
                0 <= source_idx < len(starts)
            ):
                raise RuntimeError(
                    f"{split} row {i}: invalid replay index"
                )

            source_start = int(starts[source_idx])
            source_end = (
                source_start
                + (REPLAY_LENGTH - 1) * STEP_NS
            )

        else:
            source_idx = None

        fixed = dict(row)

        fixed["target_start_ns"] = str(target_start)
        fixed["target_end_ns"] = str(target_end)

        fixed["attack_start_ns"] = (
            str(attack_start)
            if attack_start is not None else ""
        )

        fixed["attack_end_ns"] = (
            str(attack_end)
            if attack_end is not None else ""
        )

        fixed["replay_source_idx"] = (
            str(source_idx)
            if source_idx is not None else ""
        )

        fixed["replay_source_start_ns"] = (
            str(source_start)
            if source_start is not None else ""
        )

        fixed["replay_source_end_ns"] = (
            str(source_end)
            if source_end is not None else ""
        )

        # Update readable timestamps only if these
        # columns exist in the original manifest.
        readable = {
            "target_start_utc": to_utc(target_start),
            "target_end_utc": to_utc(target_end),
            "attack_start_utc": to_utc(attack_start),
            "attack_end_utc": to_utc(attack_end),
            "replay_source_start_utc": to_utc(source_start),
            "replay_source_end_utc": to_utc(source_end)
        }

        for column, value in readable.items():
            if column in fixed:
                fixed[column] = value

        repaired.append(fixed)

    # Write integer timestamps as exact decimal strings.
    repaired_df = pd.DataFrame(repaired)

    repaired_df.to_csv(
        manifest_path,
        index=False
    )

    # Re-read as strings and verify exact values.
    check = pd.read_csv(
        manifest_path,
        dtype=str,
        keep_default_na=False
    )

    if len(check) != len(repaired_df):
        raise RuntimeError(
            f"{split}: saved row count mismatch"
        )

    for column in [
        "target_start_ns",
        "target_end_ns",
        "attack_start_ns",
        "attack_end_ns",
        "replay_source_idx",
        "replay_source_start_ns",
        "replay_source_end_ns"
    ]:
        if check[column].tolist() != (
            repaired_df[column].tolist()
        ):
            raise RuntimeError(
                f"{split}: failed verification for {column}"
            )

    print("Rows repaired:", len(check))
    print("Exact timestamp verification: PASS")
    print("Original backup:", backup_path)

print("\n========== PHASE 9B.3 COMPLETE ==========")
print("NPZ attack datasets were not modified.")
