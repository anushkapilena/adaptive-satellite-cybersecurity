
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ESA-Mission1"

labels = pd.read_csv(DATA / "labels.csv")
types = pd.read_csv(DATA / "anomaly_types.csv")

events = labels.merge(
    types[["ID", "Category"]],
    on="ID",
    how="left",
    validate="many_to_one"
)

events["StartTime"] = pd.to_datetime(
    events["StartTime"], utc=True
)
events["EndTime"] = pd.to_datetime(
    events["EndTime"], utc=True
)

unique = (
    events.sort_values("StartTime")
    .drop_duplicates("ID")
    .sort_values("StartTime")
)

print("\n========== MISSION TIMELINE ==========")
print("First labelled event:", unique["StartTime"].min())
print("Last labelled event:", unique["EndTime"].max())

print("\n========== EVENTS BY YEAR ==========")
print(pd.crosstab(
    unique["StartTime"].dt.year,
    unique["Category"]
))

print("\n========== EVENTS BY MONTH ==========")
print(pd.crosstab(
    unique["StartTime"].dt.to_period("M"),
    unique["Category"]
).tail(24))

print("\n========== FIRST 10 EVENTS ==========")
print(
    unique[["ID", "StartTime", "EndTime", "Category"]]
    .head(10)
    .to_string(index=False)
)

print("\n========== LAST 10 EVENTS ==========")
print(
    unique[["ID", "StartTime", "EndTime", "Category"]]
    .tail(10)
    .to_string(index=False)
)
