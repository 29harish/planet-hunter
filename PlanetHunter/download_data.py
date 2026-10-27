
from pathlib import Path
from io import StringIO
import requests
import pandas as pd

# Create data folder
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# NASA Exoplanet Archive TAP endpoint
url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Download the Kepler cumulative catalog
query = "SELECT * FROM cumulative"

params = {
    "query": query,
    "format": "csv"
}

print("Downloading NASA Kepler dataset...")

try:
    response = requests.get(
        url,
        params=params,
        timeout=120
    )
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))

    if "koi_disposition" not in df.columns:
        raise ValueError("Expected target column was not found.")

    output = data_dir / "kepler_data.csv"
    df.to_csv(output, index=False)

    print("\nDownload successful!")
    print("Dataset saved to:", output)
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])

    print("\nFirst 5 records:")
    print(df.head().to_string())

    print("\nClassification labels:")
    print(df["koi_disposition"].value_counts(dropna=False))

except Exception as error:
    print("Download failed:", error)
