import pandas as pd


# Loading both sheets


file_path = "data/raw/online_retail_II.xlsx"

df_2009 = pd.read_excel(
    file_path,
    sheet_name="Year 2009-2010"
)

df_2010 = pd.read_excel(
    file_path,
    sheet_name="Year 2010-2011"
)


# Merge datasets


df = pd.concat(
    [df_2009, df_2010],
    ignore_index=True
)

print("Original Shape:", df.shape)


# Remove missing Customer IDs


df = df.dropna(
    subset=["Customer ID"]
)


# Remove cancelled invoices


df = df[
    ~df["Invoice"].astype(str).str.startswith("C")
]


# Remove invalid quantity


df = df[
    df["Quantity"] > 0
]


# Remove invalid prices


df = df[
    df["Price"] > 0
]


# Remove duplicates


df = df.drop_duplicates()


# Convert date


df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"]
)

# Create Revenue Column


df["Revenue"] = (
    df["Quantity"] *
    df["Price"]
)


# Save cleaned dataset


output_path = (
    "data/processed/cleaned_retail.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("Cleaned Shape:", df.shape)
print("Saved Successfully!")