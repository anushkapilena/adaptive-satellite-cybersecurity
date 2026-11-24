
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project = Path(__file__).resolve().parent
dataset = project / "ESA-Mission1"
results = project / "results"
results.mkdir(exist_ok=True)

# Load official anomaly interval
labels = pd.read_csv(dataset / "labels.csv")
event = labels[
    (labels["ID"] == "id_2") &
    (labels["Channel"] == "channel_14")
].iloc[0]

start = pd.to_datetime(event["StartTime"], utc=True)
end = pd.to_datetime(event["EndTime"], utc=True)

# Load official ESA telemetry (trusted pickle only)
with ZipFile(dataset / "channels" / "channel_14.zip") as z:
    with z.open("channel_14") as f:
        df = pd.read_pickle(f)

df.index = pd.to_datetime(df.index).tz_localize("UTC")

# Select local demonstration window
sample = df.loc[
    start - pd.Timedelta(hours=3):
    end + pd.Timedelta(hours=3)
].copy()

# Use pre-event readings as a provisional baseline
train = sample.loc[sample.index < start, "channel_14"]

mean = train.mean()
std = train.std()

if std == 0 or np.isnan(std):
    raise ValueError("Cannot calculate Z-score with zero/invalid std.")

# Calculate anomaly scores
sample["z_score"] = (
    (sample["channel_14"] - mean).abs() / std
)

threshold = 3.0
sample["predicted_anomaly"] = sample["z_score"] > threshold

# Ground truth from ESA labels
sample["actual_anomaly"] = (
    (sample.index >= start) &
    (sample.index <= end)
)

tp = int((sample["predicted_anomaly"] & sample["actual_anomaly"]).sum())
fp = int((sample["predicted_anomaly"] & ~sample["actual_anomaly"]).sum())
fn = int((~sample["predicted_anomaly"] & sample["actual_anomaly"]).sum())
tn = int((~sample["predicted_anomaly"] & ~sample["actual_anomaly"]).sum())

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
fpr = fp / (fp + tn) if fp + tn else 0

print("\n--- BASELINE DETECTION RESULTS ---")
print("Threshold:", threshold)
print("Normal mean:", mean)
print("Normal std:", std)
print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("False Positive Rate:", round(fpr, 4))

# First detected alert inside labelled interval
detected = sample.loc[
    sample["predicted_anomaly"] & sample["actual_anomaly"]
]

if not detected.empty:
    first_alert = detected.index[0]
    latency = (first_alert - start).total_seconds()
    print("First alert:", first_alert)
    print("Detection delay (seconds):", latency)
else:
    print("No alert detected inside labelled interval.")

# Plot
plt.figure(figsize=(13, 5))
plt.plot(sample.index, sample["channel_14"],
         label="Telemetry", linewidth=1)

alerts = sample[sample["predicted_anomaly"]]
plt.scatter(alerts.index, alerts["channel_14"],
            color="red", s=18, label="Predicted alerts")

plt.axvspan(start, end, color="orange",
            alpha=0.2, label="ESA id_2 interval")

plt.title("ESA Mission1 - Channel 14 - Z-score Baseline")
plt.xlabel("Timestamp (UTC)")
plt.ylabel("Telemetry Value")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(results / "baseline_detection_id2.png", dpi=180)

# Save results for later comparison
sample.to_csv(results / "baseline_predictions_id2.csv")

plt.show()
print("\nBaseline outputs saved successfully!")
