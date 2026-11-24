
from pathlib import Path
import json
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd

# ============================================================
# PHASE 9B.1 - COMPLETE CYBERATTACK INTEGRITY AUDIT
# ============================================================

ROOT = Path(__file__).resolve().parent
NORMAL_DIR = ROOT / "results" / "phase9a1"
ATTACK_DIR = ROOT / "results" / "phase9b"
OUT = ROOT / "results" / "phase9b1"
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]

print("\n========== PHASE 9B.1 START ==========")

with open(ATTACK_DIR / "attack_config.json", encoding="utf-8") as f:
    attack_config = json.load(f)

with open(NORMAL_DIR / "config.json", encoding="utf-8") as f:
    normal_config = json.load(f)

CHANNELS = attack_config["channels"]
ATTACK_START = attack_config["attack_start_step"]
SEQ_LEN = attack_config["sequence_length"]

STEP_NS = int(normal_config["grid_seconds"] * 1_000_000_000)
MIN_HISTORY_NS = int(
    attack_config["minimum_replay_history_hours"]
    * 3600 * 1_000_000_000
)

def exact_int(value):
    """Read integer-valued CSV fields without float precision loss."""
    try:
        number = Decimal(str(value).strip())
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError(f"Expected integer-valued field, got {value!r}")
        return int(number)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid integer-valued field: {value!r}") from exc


results = []
examples = []

# ============================================================
# 1. PROCESS ALL THREE SPLITS
# ============================================================

for split in SPLITS:

    print(f"\nAuditing: {split}")

    with np.load(NORMAL_DIR / f"{split}_normal.npz") as data:
        normal_X = data["X"]
        normal_starts = data["start_ns"]
        normal_ends = data["end_ns"]

    with np.load(ATTACK_DIR / f"{split}_attacks.npz") as data:
        attack_X = data["X"]
        attack_y = data["y"]
        attack_starts = data["target_start_ns"]
        attack_ends = data["target_end_ns"]

    manifest = pd.read_csv(
        ATTACK_DIR / f"{split}_attack_manifest.csv",
        dtype=str,
        keep_default_na=False
    )

    assert len(attack_X) == len(attack_y) == len(manifest)

    counters = {
        "samples_checked": 0,
        "label_mismatches": 0,
        "target_timestamp_mismatches": 0,
        "normal_mismatches": 0,
        "manipulation_errors": 0,
        "replay_errors": 0,
        "history_errors": 0,
        "invalid_values": 0
    }

    for i, row in manifest.iterrows():

        counters["samples_checked"] += 1

        label = exact_int(row["label"])
        target_idx = exact_int(row["target_sequence_idx"])

        sample = attack_X[i]
        original = normal_X[target_idx]

        if not np.isfinite(sample).all():
            counters["invalid_values"] += 1

        # --------------------------------------------
        # 2. VERIFY LABEL AND TARGET PROVENANCE
        # --------------------------------------------

        if int(attack_y[i]) != label:
            counters["label_mismatches"] += 1

        if (
            int(attack_starts[i]) != int(normal_starts[target_idx])
            or int(attack_ends[i]) != int(normal_ends[target_idx])
            or exact_int(row["target_start_ns"]) != int(normal_starts[target_idx])
            or exact_int(row["target_end_ns"]) != int(normal_ends[target_idx])
        ):
            counters["target_timestamp_mismatches"] += 1

        # --------------------------------------------
        # 3. CLASS 0 - NORMAL
        # --------------------------------------------

        if label == 0:

            if not np.array_equal(sample, original):
                counters["normal_mismatches"] += 1

        # --------------------------------------------
        # 4. CLASS 2 - MANIPULATION
        # --------------------------------------------

        elif label == 2:

            channel_name = row["modified_channels"]

            if channel_name not in CHANNELS:
                counters["manipulation_errors"] += 1
                continue

            channel_idx = CHANNELS.index(channel_name)

            # Before attack, all channels unchanged.
            before_ok = np.array_equal(
                sample[:ATTACK_START],
                original[:ATTACK_START]
            )

            # All other channels must remain unchanged.
            other_channels = [
                j for j in range(len(CHANNELS))
                if j != channel_idx
            ]

            others_ok = np.array_equal(
                sample[:, other_channels],
                original[:, other_channels]
            )

            difference = (
                sample[ATTACK_START:, channel_idx]
                - original[ATTACK_START:, channel_idx]
            )

            magnitude = float(row["manipulation_magnitude"])
            direction = exact_int(row["manipulation_direction"])

            expected_offset = direction * np.linspace(
                magnitude * 0.25,
                magnitude,
                SEQ_LEN - ATTACK_START,
                dtype=np.float32
            )

            offset_ok = np.allclose(
                difference,
                expected_offset,
                rtol=1e-5,
                atol=1e-5
            )

            # Verify recorded attack interval.
            expected_start_ns = (
                int(normal_starts[target_idx])
                + ATTACK_START * STEP_NS
            )

            timestamps_ok = (
                exact_int(row["attack_start_ns"]) == expected_start_ns
                and exact_int(row["attack_end_ns"])
                == int(normal_ends[target_idx])
            )

            if not (
                before_ok
                and others_ok
                and offset_ok
                and timestamps_ok
            ):
                counters["manipulation_errors"] += 1

        # --------------------------------------------
        # 5. CLASS 3 - HISTORICAL REPLAY
        # --------------------------------------------

        elif label == 3:

            if not row["replay_source_idx"].strip():
                counters["replay_errors"] += 1
                continue

            source_idx = exact_int(row["replay_source_idx"])

            if not (0 <= source_idx < len(normal_X)):
                counters["replay_errors"] += 1
                continue

            historical = normal_X[source_idx]

            replay_len = SEQ_LEN - ATTACK_START

            before_ok = np.array_equal(
                sample[:ATTACK_START],
                original[:ATTACK_START]
            )

            replay_ok = np.array_equal(
                sample[ATTACK_START:],
                historical[:replay_len]
            )

            expected_source_end = (
                int(normal_starts[source_idx])
                + (replay_len - 1) * STEP_NS
            )

            source_metadata_ok = (
                exact_int(row["replay_source_start_ns"])
                == int(normal_starts[source_idx])
                and exact_int(row["replay_source_end_ns"])
                == expected_source_end
            )

            attack_metadata_ok = (
                exact_int(row["attack_start_ns"])
                == int(normal_starts[target_idx])
                + ATTACK_START * STEP_NS
                and exact_int(row["attack_end_ns"])
                == int(normal_ends[target_idx])
            )

            # History condition must hold using the
            # full original source window end.
            history_ok = (
                int(normal_ends[source_idx])
                <= int(normal_starts[target_idx])
                - MIN_HISTORY_NS
            )

            if not history_ok:
                counters["history_errors"] += 1

            if not (
                before_ok
                and replay_ok
                and source_metadata_ok
                and attack_metadata_ok
            ):
                counters["replay_errors"] += 1

        else:
            counters["label_mismatches"] += 1

        # Store a few examples for visualization.
        if (
            split == "test"
            and label in (2, 3)
            and len(examples) < 6
        ):
            examples.append({
                "label": label,
                "original": original.copy(),
                "attacked": sample.copy()
            })

    counters["split"] = split

    counters["total_errors"] = sum(
        value
        for key, value in counters.items()
        if key not in ("samples_checked", "split")
    )

    results.append(counters)

    print("Samples checked:", counters["samples_checked"])
    print("Label mismatches:", counters["label_mismatches"])
    print("Timestamp mismatches:",
          counters["target_timestamp_mismatches"])
    print("Normal mismatches:", counters["normal_mismatches"])
    print("Manipulation errors:", counters["manipulation_errors"])
    print("Replay errors:", counters["replay_errors"])
    print("History errors:", counters["history_errors"])
    print("Invalid values:", counters["invalid_values"])
    print("Total errors:", counters["total_errors"])

# ============================================================
# 6. SAVE AUDIT REPORT
# ============================================================

report = pd.DataFrame(results)

report.to_csv(
    OUT / "attack_integrity_audit.csv",
    index=False
)

total_errors = int(report["total_errors"].sum())
total_samples = int(report["samples_checked"].sum())

status = (
    "ATTACK_INTEGRITY_PASSED"
    if total_errors == 0
    else "ATTACK_INTEGRITY_ISSUES_FOUND"
)

summary = {
    "phase": "9B.1",
    "status": status,
    "total_samples_checked": total_samples,
    "total_errors": total_errors,
    "split_reports": results,
    "limitations": [
        "Checks consistency against the defined injection rules.",
        "Does not establish realism of cyberattack scenarios.",
        "Does not establish model detection performance.",
        "Does not prove normal candidates are anomaly-free.",
        "Replay uses historical saved telemetry sequences.",
        "No actual spacecraft network packets are replayed."
    ]
}

with open(
    OUT / "attack_integrity_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

# ============================================================
# 7. OPTIONAL ATTACK VISUALIZATION
# ============================================================

if examples:
    import matplotlib.pyplot as plt

    for number, example in enumerate(examples[:4], start=1):

        original = example["original"]
        attacked = example["attacked"]

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            original[:, 0],
            label="Original channel 14",
            linewidth=2
        )

        ax.plot(
            attacked[:, 0],
            label="Attacked channel 14",
            linestyle="--",
            linewidth=2
        )

        ax.axvline(
            ATTACK_START,
            linestyle=":",
            label="Attack start"
        )

        ax.set_title(
            f"Phase 9B - Attack class {example['label']}"
        )
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Standardized telemetry value")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()

        fig.savefig(
            OUT / f"attack_example_{number}.png",
            dpi=150
        )

        plt.close(fig)

print("\n========== PHASE 9B.1 COMPLETE ==========")
print("Total samples audited:", total_samples)
print("Total errors:", total_errors)
print("Status:", status)
print("Saved:", OUT)
