
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import matplotlib.pyplot as plt

# Project paths
project = Path(__file__).resolve().parent
dataset = project / "ESA-Mission1"
results = project / "results"
results.mkdir(exist_ok=True)

# Load official ESA labels
labels = pd.read_csv(dataset / "labels.csv")

event = labels[
    (labels["ID"] == "id_2") &
    (labels["Channel"] == "channel_14")
].iloc[0]

start = pd.to_datetime(event["StartTime"], utc=True)
end = pd.to_datetime(event["EndTime"], utc=True)

# Load real satellite telemetry
zip_path = dataset / "channels" / "channel_14.zip"

with ZipFile(zip_path, "r") as z:
    with z.open("channel_14") as f:
        df = pd.read_pickle(f)

# Dataset index has no timezone information.
# Interpret timestamps as UTC for matching ESA labels.
df.index = pd.to_datetime(df.index).tz_localize("UTC")

# Include 3 hours before and after the event
window_start = start - pd.Timedelta(hours=3)
window_end = end + pd.Timedelta(hours=3)

sample = df.loc[window_start:window_end]

print("Readings in selected window:", len(sample))
print("Anomaly start:", start)
print("Anomaly end:", end)

# Plot telemetry
plt.figure(figsize=(13, 5))

plt.plot(
    sample.index,
    sample["channel_14"],
    color="steelblue",
    linewidth=1,
    label="Satellite telemetry"
)

# Highlight official anomaly interval
plt.axvspan(
    start,
    end,
    color="red",
    alpha=0.2,
    label="ESA labelled anomaly (id_2)"
)

plt.title("ESA Mission1 - Channel 14 - Official Anomaly id_2")
plt.xlabel("Timestamp (UTC)")
plt.ylabel("Telemetry Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output = results / "channel_14_id2_anomaly.png"
plt.savefig(output, dpi=180)
plt.show()

print("Graph saved:", output)


import numpy as np

# Split the selected window using official labels
normal_before = sample.loc[sample.index < start, "channel_14"]
during_anomaly = sample.loc[
    (sample.index >= start) & (sample.index <= end),
    "channel_14"
]
normal_after = sample.loc[sample.index > end, "channel_14"]

print("\n--- TELEMETRY ANALYSIS ---")

for name, values in [
    ("Before anomaly", normal_before),
    ("During anomaly", during_anomaly),
    ("After anomaly", normal_after),
]:
    print(f"\n{name}")
    print("Readings:", len(values))

    if not values.empty:
        print("Mean:", round(float(values.mean()), 6))
        print("Std deviation:", round(float(values.std()), 6))
        print("Minimum:", round(float(values.min()), 6))
        print("Maximum:", round(float(values.max()), 6))
