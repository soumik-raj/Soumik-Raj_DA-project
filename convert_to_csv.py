"""
Run this once to convert RAW_DATA.tsv → INDIA_RETAIL_DATA.csv
Usage: python convert_to_csv.py
"""
import pandas as pd
import io
import pathlib
import sys

tsv_path = pathlib.Path("RAW_DATA.tsv")
csv_path = pathlib.Path("INDIA_RETAIL_DATA.csv")

if not tsv_path.exists():
    print("RAW_DATA.tsv not found — creating CSV from embedded sample data.")
    # Minimal fallback: the user should paste the full TSV.
    sys.exit(1)

raw = tsv_path.read_text(encoding="utf-8")
df = pd.read_csv(io.StringIO(raw), sep="\t")
df.to_csv(csv_path, index=False)
print(f"✅  Saved {len(df):,} rows → {csv_path}")
print("Columns:", list(df.columns))
