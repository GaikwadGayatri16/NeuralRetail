import pandas as pd
from datetime import timedelta

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

file_path = BASE_DIR / "data" / "processed" / "cleaned_retail.csv"

df = pd.read_csv(file_path)

print("Loaded:", file_path)
# Convert date
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    format="%d-%m-%Y %H:%M"
)

# --------------------------
# RFM FEATURES
# --------------------------

snapshot_date = df["InvoiceDate"].max() + timedelta(days=1)

rfm = (
    df.groupby("Customer ID")
    .agg(
        Recency=(
            "InvoiceDate",
            lambda x: (
                snapshot_date - x.max()
            ).days
        ),

        Frequency=(
            "Invoice",
            "nunique"
        ),

        Monetary=(
            "Revenue",
            "sum"
        )
    )
    .reset_index()
)

# --------------------------
# EXTRA FEATURES
# --------------------------

customer_features = (
    df.groupby("Customer ID")
    .agg(
        TotalOrders=(
            "Invoice",
            "nunique"
        ),

        TotalProducts=(
            "Quantity",
            "sum"
        ),

        AvgOrderValue=(
            "Revenue",
            "mean"
        ),

        LifetimeRevenue=(
            "Revenue",
            "sum"
        ),

        FirstPurchase=(
            "InvoiceDate",
            "min"
        ),

        LastPurchase=(
            "InvoiceDate",
            "max"
        )
    )
    .reset_index()
)

# Customer Age

customer_features["CustomerAgeDays"] = (
    customer_features["LastPurchase"]
    - customer_features["FirstPurchase"]
).dt.days

# Days Since Last Purchase

customer_features["DaysSinceLastPurchase"] = (
    snapshot_date
    - customer_features["LastPurchase"]
).dt.days

# --------------------------
# MERGE FEATURES
# --------------------------

feature_store = pd.merge(
    rfm,
    customer_features,
    on="Customer ID"
)

# --------------------------
# SAVE
# --------------------------

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

output_path = BASE_DIR / "data" / "processed" / "feature_store.csv"

feature_store.to_csv(
    output_path,
    index=False
)

print("Saved:", output_path)
print("Feature Store Created")

print(feature_store.head())

print(
    "Shape:",
    feature_store.shape
)