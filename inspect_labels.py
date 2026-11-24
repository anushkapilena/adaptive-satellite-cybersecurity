
from pathlib import Path
import pandas as pd

project = Path(__file__).resolve().parent
dataset = project / "ESA-Mission1"

labels = pd.read_csv(dataset / "labels.csv")
types = pd.read_csv(dataset / "anomaly_types.csv")

print("\nLABEL COLUMNS:")
print(labels.columns.tolist())

print("\nANOMALY TYPE COLUMNS:")
print(types.columns.tolist())

print("\nID_2 LABELS:")
print(labels[labels["ID"].astype(str).str.lower() == "id_2"].to_string(index=False))

print("\nID_2 TYPE:")
print(types[types["ID"].astype(str).str.lower() == "id_2"].to_string(index=False))
