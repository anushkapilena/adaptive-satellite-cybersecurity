
from pathlib import Path
from zipfile import ZipFile
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ESA-Mission1"
OUT = ROOT / "results" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)

EVENT_ID = "id_2"
CHANNELS = ["channel_14", "channel_21", "channel_29"]
WINDOW_HOURS = 6
GRID_SECONDS = 90

# ---------------- DATASET INVENTORY ----------------

channels = pd.read_csv(DATA / "channels.csv")
labels = pd.read_csv(DATA / "labels.csv")
types = pd.read_csv(DATA / "anomaly_types.csv")

labels["ID"] = labels["ID"].astype(str)
types["ID"] = types["ID"].astype(str)

events = labels.merge(
    types,
    on="ID",
    how="left",
    validate="many_to_one"
)

print("\n========== ESA MISSION1 INVENTORY ==========")
print("Metadata channels:", len(channels))
print("Channel ZIP files:", len(list((DATA / "channels").glob("*.zip"))))
print("Unique labelled events:", labels["ID"].nunique())
print("Total label rows:", len(labels))

print("\nEvent categories:")
print(
    events.drop_duplicates("ID")["Category"]
    .fillna("Unknown")
    .value_counts()
)

print("\nEvent dimensionality:")
print(
    events.drop_duplicates("ID")["Dimensionality"]
    .fillna("Unknown")
    .value_counts()
)

events.to_csv(OUT / "event_inventory.csv", index=False)

channel_summary = (
    events.groupby("Channel")
    .agg(
        annotation_rows=("ID", "size"),
        unique_events=("ID", "nunique")
    )
    .reset_index()
    .sort_values("unique_events", ascending=False)
)

channel_summary.to_csv(
    OUT / "channel_event_summary.csv", index=False
)

# ---------------- OFFICIAL EVENT ----------------

selected = events[
    (events["ID"] == EVENT_ID) &
    (events["Channel"].isin(CHANNELS))
].copy()

if len(selected) != len(CHANNELS):
    raise ValueError("Selected event/channel labels missing.")

start = pd.to_datetime(selected["StartTime"], utc=True).min()
end = pd.to_datetime(selected["EndTime"], utc=True).max()

window_start = start - pd.Timedelta(hours=WINDOW_HOURS)
window_end = end + pd.Timedelta(hours=WINDOW_HOURS)

print("\n========== SELECTED EVENT ==========")
print("Event:", EVENT_ID)
print("Start:", start)
print("End:", end)
print("Channels:", CHANNELS)

# ---------------- LOAD TELEMETRY ----------------

loaded = {}

for name in CHANNELS:
    path = DATA / "channels" / f"{name}.zip"

    # Official trusted ESA pickle files only
    with ZipFile(path) as z:
        with z.open(name) as f:
            frame = pd.read_pickle(f)

    frame.index = pd.to_datetime(frame.index)

    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")

    frame = frame.sort_index()
    frame = frame.loc[window_start:window_end, [name]]

    loaded[name] = frame

    print(
        f"{name}: {len(frame)} readings, "
        f"missing={int(frame[name].isna().sum())}"
    )

# ---------------- TIME ALIGNMENT ----------------

# Use a regular grid and only past observations.
# No future interpolation: avoids look-ahead leakage.
grid = pd.date_range(
    start=window_start,
    end=window_end,
    freq=f"{GRID_SECONDS}s",
    tz="UTC"
)

aligned = pd.DataFrame(index=grid)
aligned.index.name = "datetime"

for name, frame in loaded.items():
    series = frame[name]

    if series.index.has_duplicates:
        series = series.groupby(level=0).last()

    # At each grid timestamp use most recent available value.
    aligned[name] = series.reindex(
        grid,
        method="ffill"
    )

# Drop timestamps without complete channel coverage
before_drop = len(aligned)
aligned = aligned.dropna(subset=CHANNELS)

print("\n========== ALIGNED DATA ==========")
print("Grid rows:", before_drop)
print("Complete rows:", len(aligned))
print("Missing rows removed:", before_drop - len(aligned))

if aligned.empty:
    raise ValueError("No overlapping multivariate telemetry.")

# ---------------- OFFICIAL LABELS ----------------

aligned["operational_anomaly"] = (
    (aligned.index >= start) &
    (aligned.index <= end)
).astype(int)

aligned["event_id"] = np.where(
    aligned["operational_anomaly"] == 1,
    EVENT_ID,
    ""
)

aligned.to_parquet(OUT / "id2_multivariate.parquet")
aligned.to_csv(OUT / "id2_multivariate.csv")

print("Labelled anomaly rows:",
      int(aligned["operational_anomaly"].sum()))

# ---------------- PLOT ALL 3 CHANNELS ----------------

fig, axes = plt.subplots(
    len(CHANNELS), 1,
    figsize=(14, 9),
    sharex=True
)

for ax, name in zip(axes, CHANNELS):
    ax.plot(
        aligned.index,
        aligned[name],
        linewidth=1,
        label=name
    )

    ax.axvspan(
        start, end,
        color="red",
        alpha=0.15,
        label="Official id_2 interval"
    )

    ax.set_ylabel(name)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

axes[-1].set_xlabel("Timestamp (UTC)")
fig.suptitle(
    "ESA Mission1 - Multivariate Operational Anomaly id_2"
)

plt.tight_layout()
plt.savefig(
    OUT / "id2_multivariate_plot.png",
    dpi=180
)
plt.close()

# ---------------- SUMMARY ----------------

summary = {
    "mission": "ESA-Mission1",
    "event": EVENT_ID,
    "channels": CHANNELS,
    "event_start_utc": str(start),
    "event_end_utc": str(end),
    "grid_seconds": GRID_SECONDS,
    "aligned_rows": len(aligned),
    "anomaly_rows": int(aligned["operational_anomaly"].sum()),
    "normal_window_rows": int(
        (aligned["operational_anomaly"] == 0).sum()
    ),
    "note": (
        "Local proof-of-concept window. "
        "Rows outside id_2 are not independently "
        "verified as globally anomaly-free."
    )
}

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n========== COMPLETED ==========")
print("Outputs saved to:", OUT)
print("Files:")
for path in sorted(OUT.iterdir()):
    print(" -", path.name)
