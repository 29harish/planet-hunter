
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Set up paths
# -------------------------------
DATA_DIR = Path("data")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "kepler_data.csv"
OUTPUT_FILE = DATA_DIR / "kepler_cleaned.csv"

# -------------------------------
# 2. Load the NASA dataset
# -------------------------------
df = pd.read_csv(INPUT_FILE)

print("\n===== ORIGINAL DATASET =====")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst five rows:")
print(df.head().to_string())

print("\nOriginal target distribution:")
print(df["koi_disposition"].value_counts(dropna=False))

# -------------------------------
# 3. Select astronomical features
# -------------------------------
features = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_model_snr",
    "koi_impact",
    "koi_time0bk",
    "koi_num_transits"
]

required = features + ["koi_disposition", "kepid"]

missing_columns = [
    column for column in required
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Keep Kepler star ID for grouped train/test splitting.
# This prevents observations of the same star from
# appearing in both training and testing data.
cleaned = df[required].copy()

# -------------------------------
# 4. Create classification labels
# -------------------------------
label_mapping = {
    "CONFIRMED": 1,
    "CANDIDATE": 1,
    "FALSE POSITIVE": 0
}

cleaned["target"] = cleaned[
    "koi_disposition"
].map(label_mapping)

# Remove rows with unknown or missing labels
cleaned = cleaned.dropna(
    subset=["target", "kepid"]
).copy()

cleaned["target"] = cleaned["target"].astype(int)

# -------------------------------
# 5. Convert features to numbers
# -------------------------------
for feature in features:
    cleaned[feature] = pd.to_numeric(
        cleaned[feature],
        errors="coerce"
    )

# Replace infinite values with NaN
cleaned[features] = cleaned[features].replace(
    [np.inf, -np.inf],
    np.nan
)

# Don't impute missing values here.
# The ML pipeline in Step 4 will fit an imputer
# using the training data ONLY.

print("\n===== MISSING VALUES =====")
print(
    cleaned[features]
    .isnull()
    .sum()
    .sort_values(ascending=False)
)

# -------------------------------
# 6. Remove rows with no measurements
# -------------------------------
cleaned = cleaned.dropna(
    subset=features,
    how="all"
).copy()

# -------------------------------
# 7. Save prepared data
# -------------------------------
cleaned.to_csv(OUTPUT_FILE, index=False)

print("\n===== CLEANED DATASET =====")
print("Rows:", cleaned.shape[0])
print("Columns:", cleaned.shape[1])

print("\nBinary class distribution:")
print(
    cleaned["target"]
    .value_counts()
    .rename(index={
        0: "False Positive",
        1: "Candidate / Confirmed"
    })
)

print("\nSaved to:", OUTPUT_FILE)

# -------------------------------
# 8. Plot class distribution
# -------------------------------
sns.set_theme(style="whitegrid")

plt.figure(figsize=(7, 5))

counts = (
    cleaned["target"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

ax = sns.barplot(
    x=["False Positive", "Candidate / Confirmed"],
    y=counts.values
)

ax.set(
    title="Kepler Object Classification Distribution",
    xlabel="Object Category",
    ylabel="Number of Objects"
)

plt.tight_layout()
plt.savefig(
    REPORT_DIR / "class_distribution.png",
    dpi=200
)
plt.close()

# -------------------------------
# 9. Plot missing values
# -------------------------------
missing_pct = (
    cleaned[features]
    .isnull()
    .mean()
    .sort_values(ascending=False)
    * 100
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=missing_pct.values,
    y=missing_pct.index
)

plt.title("Missing Values by Astronomical Feature")
plt.xlabel("Missing Values (%)")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig(
    REPORT_DIR / "missing_values.png",
    dpi=200
)
plt.close()

print("\nGraphs saved inside reports/ folder.")
print("\nSTEP 3 COMPLETED SUCCESSFULLY!")
