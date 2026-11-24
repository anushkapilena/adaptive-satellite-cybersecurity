
from pathlib import Path
from zipfile import ZipFile
import pandas as pd

project = Path(__file__).resolve().parent
zip_path = project / "ESA-Mission1" / "channels" / "channel_14.zip"

with ZipFile(zip_path, "r") as z:
    with z.open("channel_14") as f:
        df = pd.read_pickle(f)

print("\nDATASET LOADED SUCCESSFULLY!")

print("\nShape (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


import matplotlib.pyplot as plt

# First 1000 telemetry readings
sample = df.iloc[:1000]

plt.figure(figsize=(12, 5))

plt.plot(
    sample.index,
    sample["channel_14"],
    color="blue",
    linewidth=1
)

plt.title("ESA Mission1 - Satellite Telemetry (Channel 14)")
plt.xlabel("Timestamp")
plt.ylabel("Telemetry Value")
plt.grid(True)
plt.tight_layout()

# Save graph
output_dir = project / "results"
output_dir.mkdir(exist_ok=True)

plt.savefig(output_dir / "channel_14_telemetry.png", dpi=150)

plt.show()

print("\nGraph saved successfully!")
