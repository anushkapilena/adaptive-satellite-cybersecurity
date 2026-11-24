
from pathlib import Path
from zipfile import ZipFile
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ESA-Mission1"
OUT = ROOT / "results" / "phase9c0"
OUT.mkdir(parents=True, exist_ok=True)

config = json.loads(
    (ROOT / "results/phase9a1/config.json").read_text()
)

channels = config["channels"]
grid_seconds = config["grid_seconds"]
max_age = pd.Timedelta(
    seconds=config["max_observation_age_seconds"]
)
seq_len = config["sequence_length"]

summary = pd.read_csv(
    ROOT / "results/phase4/event_channel_summary.csv"
)
summary = summary[summary["channel"].isin(channels)].copy()

summary["start"] = pd.to_datetime(summary["start"], utc=True)
summary["end"] = pd.to_datetime(summary["end"], utc=True)

events = summary.groupby("event_id").agg(
    start=("start", "min"),
    end=("end", "max"),
    labelled_channels=("channel", "nunique")
).sort_values("start")

series = {}

for channel in channels:
    print("Loading", channel, flush=True)

    with ZipFile(DATA / "channels" / f"{channel}.zip") as z:
        with z.open(channel) as f:
            df = pd.read_pickle(f)  # Trusted official ESA archive only

    idx = pd.to_datetime(df.index, utc=True)
    s = pd.Series(
        pd.to_numeric(df[channel], errors="coerce")
        .to_numpy(dtype=np.float32),
        index=idx
    ).sort_index()

    if s.index.has_duplicates:
        s = s.groupby(level=0).last()

    series[channel] = s
    del df

rows = []

for event_id, event in events.iterrows():
    # Audit only the official event interval.
    # The builder will separately handle pre-event context.
    grid = pd.date_range(
        start=event.start,
        end=event.end,
        freq=f"{grid_seconds}s"
    )

    aligned = pd.DataFrame(index=grid)

    for channel in channels:
        local = series[channel].loc[
            event.start - max_age:event.end
        ]

        aligned[channel] = local.reindex(
            grid,
            method="ffill",
            tolerance=max_age
        )

    valid = np.isfinite(
        aligned[channels].to_numpy(dtype=np.float32)
    ).all(axis=1)

    # Number of complete non-overlapping windows.
    complete = sum(
        valid[i:i + seq_len].all()
        for i in range(0, len(valid) - seq_len + 1, seq_len)
    )

    rows.append({
        "event_id": event_id,
        "labelled_channels": int(event.labelled_channels),
        "grid_points": len(grid),
        "valid_points": int(valid.sum()),
        "coverage_percent": round(
            100 * valid.mean(), 2
        ) if len(valid) else 0,
        "complete_windows": int(complete)
    })

result = pd.DataFrame(rows)
result.to_csv(OUT / "coverage_audit.csv", index=False)

print("\n========== PHASE 9C0 COVERAGE ==========")
print("Events:", len(result))
print(
    "Events with >=1 complete window:",
    int((result.complete_windows > 0).sum())
)
print(
    "Events with zero complete windows:",
    int((result.complete_windows == 0).sum())
)
print(
    "Median coverage (%):",
    result.coverage_percent.median()
)
print(
    "Minimum coverage (%):",
    result.coverage_percent.min()
)
print("\nLowest coverage events:")
print(
    result.sort_values("coverage_percent")
    .head(10).to_string(index=False)
)
print("\nSaved:", OUT / "coverage_audit.csv")
