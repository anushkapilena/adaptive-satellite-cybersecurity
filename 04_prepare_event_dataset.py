
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
import json
import gc

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ESA-Mission1"
OUT = ROOT / "results" / "phase4"
OUT.mkdir(parents=True, exist_ok=True)

# First run 10 events. Change to None for all events.
MAX_EVENTS = None
CONTEXT_HOURS = 3

labels = pd.read_csv(DATA / "labels.csv")
types = pd.read_csv(DATA / "anomaly_types.csv")

labels["StartTime"] = pd.to_datetime(
    labels["StartTime"], utc=True
)
labels["EndTime"] = pd.to_datetime(
    labels["EndTime"], utc=True
)

events = labels.merge(
    types[["ID", "Category", "Class", "Subclass"]],
    on="ID",
    how="left",
    validate="many_to_one"
)

anomalies = events[
    events["Category"].astype(str).str.strip().eq("Anomaly")
].copy()

event_order = (
    anomalies.groupby("ID")["StartTime"]
    .min()
    .sort_values()
)

event_ids = event_order.index.tolist()

if MAX_EVENTS is not None:
    event_ids = event_ids[:MAX_EVENTS]

print("Selected anomaly events:", len(event_ids))

# Group intervals by channel, including non-anomaly
# event categories, to flag context contamination.
all_intervals = defaultdict(list)

for row in events.itertuples(index=False):
    all_intervals[row.Channel].append(
        (row.StartTime, row.EndTime, row.ID, row.Category)
    )

# Cache a channel only for the current event.
def load_channel(channel):
    path = DATA / "channels" / f"{channel}.zip"

    if not path.exists():
        raise FileNotFoundError(path)

    with ZipFile(path) as z:
        with z.open(channel) as f:
            df = pd.read_pickle(f)

    df.index = pd.to_datetime(df.index)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.sort_index()
    return df[[channel]]


records = []
failures = []

for number, event_id in enumerate(event_ids, start=1):
    subset = anomalies[anomalies["ID"] == event_id]

    event_start = subset["StartTime"].min()
    event_end = subset["EndTime"].max()

    window_start = event_start - pd.Timedelta(
        hours=CONTEXT_HOURS
    )
    window_end = event_end + pd.Timedelta(
        hours=CONTEXT_HOURS
    )

    affected_channels = sorted(subset["Channel"].unique())

    print(
        f"\n[{number}/{len(event_ids)}] "
        f"{event_id}: {len(affected_channels)} channels"
    )

    event_dir = OUT / "events" / event_id
    event_dir.mkdir(parents=True, exist_ok=True)

    for channel in affected_channels:
        try:
            df = load_channel(channel)
            df = df.loc[window_start:window_end].copy()

            if df.empty:
                raise ValueError("No telemetry in event window")

            # Native timestamps, no artificial resampling.
            df.index.name = "datetime"

            # Label only this event's official interval.
            # Point events may have no exact sampled timestamp.
            df["target_event"] = (
                (df.index >= event_start) &
                (df.index <= event_end)
            ).astype("int8")

            # Mark other annotated intervals overlapping
            # the same channel's context window.
            df["other_labelled_event"] = 0

            for s, e, other_id, category in all_intervals[channel]:
                if other_id == event_id:
                    continue
                if e < window_start or s > window_end:
                    continue

                mask = (df.index >= s) & (df.index <= e)
                df.loc[mask, "other_labelled_event"] = 1

            output = event_dir / f"{channel}.parquet"
            df.to_parquet(output)

            records.append({
                "event_id": event_id,
                "channel": channel,
                "start": event_start,
                "end": event_end,
                "rows": len(df),
                "target_rows": int(df["target_event"].sum()),
                "other_event_rows": int(
                    df["other_labelled_event"].sum()
                ),
                "file": str(output.relative_to(ROOT))
            })

            del df
            gc.collect()

        except Exception as exc:
            failures.append({
                "event_id": event_id,
                "channel": channel,
                "error": str(exc)
            })
            print("  FAILED:", channel, exc)

    print("  Completed")

summary = pd.DataFrame(records)
summary.to_csv(OUT / "event_channel_summary.csv", index=False)

failure_df = pd.DataFrame(
    failures,
    columns=["event_id", "channel", "error"]
)
failure_df.to_csv(OUT / "failures.csv", index=False)

# Split by chronological event order, never random rows.
n = len(event_ids)
train_end = int(n * 0.60)
val_end = int(n * 0.80)

splits = []

for i, event_id in enumerate(event_ids):
    if i < train_end:
        split = "train_candidate"
    elif i < val_end:
        split = "validation_candidate"
    else:
        split = "test_candidate"

    splits.append({
        "event_id": event_id,
        "start": event_order[event_id],
        "split": split
    })

pd.DataFrame(splits).to_csv(
    OUT / "event_split_plan.csv",
    index=False
)

print("\n========== FINAL SUMMARY ==========")
print("Events selected:", len(event_ids))
print("Event-channel datasets saved:", len(records))
print("Failures:", len(failures))
print("Total telemetry rows saved:",
      int(summary["rows"].sum()) if len(summary) else 0)
print("Output:", OUT)

print("\nNOTE:")
print("These are event windows, not clean training splits.")
print("Event split plan is provisional.")
print("Normal-only training and leakage checks come next.")
print("Point events with zero target rows need special handling.")
