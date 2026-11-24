
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "phase1" / "id2_multivariate.parquet"
OUT = ROOT / "results" / "phase2"
OUT.mkdir(parents=True, exist_ok=True)

CHANNELS = ["channel_14", "channel_21", "channel_29"]

# Load prepared telemetry
df = pd.read_parquet(SOURCE).sort_index()

print("\n========== DATA VALIDATION ==========")
print("Shape:", df.shape)
print("Duplicate timestamps:", df.index.duplicated().sum())
print("Missing values:")
print(df[CHANNELS].isna().sum())
print("Sampling intervals:")
print(df.index.to_series().diff().value_counts().head())

# Chronological split: 60% train, 20% validation, 20% test
n = len(df)
train_end = int(n * 0.60)
val_end = int(n * 0.80)

train = df.iloc[:train_end].copy()
val = df.iloc[train_end:val_end].copy()
test = df.iloc[val_end:].copy()

print("\n========== CHRONOLOGICAL SPLIT ==========")

for name, part in [
    ("Train", train),
    ("Validation", val),
    ("Test", test)
]:
    print(
        name,
        "Rows:", len(part),
        "Anomaly rows:",
        int(part["operational_anomaly"].sum())
    )

# Fit scaler on training period only
scaler = StandardScaler()
scaler.fit(train[CHANNELS])

for part in (train, val, test):
    part[CHANNELS] = scaler.transform(part[CHANNELS])

train.to_parquet(OUT / "train.parquet")
val.to_parquet(OUT / "validation.parquet")
test.to_parquet(OUT / "test.parquet")

# Save preprocessing parameters
config = {
    "channels": CHANNELS,
    "train_rows": len(train),
    "validation_rows": len(val),
    "test_rows": len(test),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "warning": (
        "Demonstration split only. The selected window "
        "contains one labelled event and does not "
        "constitute a valid final benchmark split."
    )
}

with open(OUT / "preprocessing_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n========== COMPLETED ==========")
print("Saved files:")
for path in sorted(OUT.iterdir()):
    print("-", path.name)
