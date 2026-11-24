
from pathlib import Path
from zipfile import ZipFile
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ESA-Mission1"

print("\n========== PHASE 9A DATA CHECK ==========")

labels = pd.read_csv(DATA / "labels.csv")

print("\nLabel columns:")
print(labels.columns.tolist())

print("\nLabel rows:", len(labels))
print("Unique event IDs:", labels["ID"].nunique())

print("\nFirst 5 labels:")
print(labels.head().to_string(index=False))

for channel in [
    "channel_14",
    "channel_21",
    "channel_29"
]:
    path = DATA / "channels" / f"{channel}.zip"

    with ZipFile(path) as z:
        with z.open(channel) as f:
            df = pd.read_pickle(f)

    print(f"\n{channel}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Index type:", type(df.index).__name__)
    print("First timestamp:", df.index.min())
    print("Last timestamp:", df.index.max())

    times = pd.to_datetime(df.index)

    diffs = (
        pd.Series(times)
        .sort_values()
        .diff()
        .dropna()
        .dt.total_seconds()
    )

    print("Median interval (seconds):", diffs.median())
    print("Most common intervals:")
    print(diffs.value_counts().head(3).to_string())

print("\n========== CHECK COMPLETE ==========")
