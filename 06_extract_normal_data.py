
from pathlib import Path
from zipfile import ZipFile
import gc

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ESA-Mission1"
OUT = ROOT / "results" / "phase6"
NORMAL_DIR = OUT / "normal_channels"

NORMAL_DIR.mkdir(parents=True, exist_ok=True)

# Exclude readings near any official event.
SAFETY_BUFFER_HOURS = 3

labels = pd.read_csv(DATA / "labels.csv")

labels["StartTime"] = pd.to_datetime(
    labels["StartTime"], utc=True
)
labels["EndTime"] = pd.to_datetime(
    labels["EndTime"], utc=True
)

# Every labelled category is excluded:
# anomaly, rare event and communication gap.
labels_by_channel = {
    channel: group
    for channel, group in labels.groupby("Channel")
}

channel_paths = sorted(
    (DATA / "channels").glob("*.zip")
)

summary = []
failures = []

print("\n========== NORMAL TELEMETRY EXTRACTION ==========")
print("Channels found:", len(channel_paths))

for number, path in enumerate(channel_paths, start=1):
    channel = path.stem

    try:
        # Load trusted official ESA dataset only.
        with ZipFile(path) as z:
            with z.open(channel) as f:
                df = pd.read_pickle(f)

        df.index = pd.to_datetime(df.index)

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = df.sort_index()

        total_rows = len(df)

        # Start with all readings as candidates.
        keep = np.ones(total_rows, dtype=bool)

        if channel in labels_by_channel:
            channel_labels = labels_by_channel[channel]

            for row in channel_labels.itertuples(index=False):
                start = row.StartTime - pd.Timedelta(
                    hours=SAFETY_BUFFER_HOURS
                )
                end = row.EndTime + pd.Timedelta(
                    hours=SAFETY_BUFFER_HOURS
                )

                # Efficient timestamp-based interval lookup.
                left = df.index.searchsorted(start, side="left")
                right = df.index.searchsorted(end, side="right")

                keep[left:right] = False

        normal = df.iloc[np.flatnonzero(keep)].copy()

        normal.index.name = "datetime"

        output = NORMAL_DIR / f"{channel}.parquet"
        normal.to_parquet(output)

        summary.append({
            "channel": channel,
            "total_rows": total_rows,
            "normal_candidate_rows": len(normal),
            "excluded_rows": total_rows - len(normal),
            "normal_candidate_percentage": round(
                100 * len(normal) / total_rows, 2
            ) if total_rows else 0
        })

        print(
            f"[{number}/{len(channel_paths)}] {channel}: "
            f"{len(normal):,}/{total_rows:,} candidates"
        )

        del df, normal
        gc.collect()

    except Exception as exc:
        failures.append({
            "channel": channel,
            "error": str(exc)
        })
        print(f"FAILED: {channel}: {exc}")

summary_df = pd.DataFrame(summary)
summary_df.to_csv(
    OUT / "normal_extraction_summary.csv",
    index=False
)

pd.DataFrame(
    failures,
    columns=["channel", "error"]
).to_csv(
    OUT / "failures.csv",
    index=False
)

print("\n========== FINAL SUMMARY ==========")
print("Channels processed:", len(summary))
print("Failures:", len(failures))

if len(summary_df):
    print(
        "Total normal candidate rows:",
        f"{summary_df['normal_candidate_rows'].sum():,}"
    )
    print(
        "Total excluded rows:",
        f"{summary_df['excluded_rows'].sum():,}"
    )

print("Output folder:", OUT)
print("\nNOTE: These are unlabelled normal candidates,")
print("not independently verified clean telemetry.")
