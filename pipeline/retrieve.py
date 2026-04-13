"""
pipeline/retrieve.py

Fetches all records stored in MiniVaultDB and reconstructs them as a
pandas DataFrame ready for analysis.

Usage:
    python -m pipeline.retrieve
    python -m pipeline.retrieve --db vault_data --out data/retrieved.csv
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minivaultdb.adapter import MiniVaultDBAdapter


# ──────────────────────────────────────────────────────────────────────────────
# Core retrieval function
# ──────────────────────────────────────────────────────────────────────────────

def retrieve_to_dataframe(
    db_path: str = "vault_data",
    include_key: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scan all records from MiniVaultDB and return a pandas DataFrame.

    Parameters
    ----------
    db_path     : MiniVaultDB storage directory
    include_key : add a '__key' column with the MiniVaultDB key
    verbose     : print summary info

    Returns
    -------
    pd.DataFrame with one row per stored record.
    """
    adapter = MiniVaultDBAdapter(db_path)

    start = time.perf_counter()
    pairs = adapter.scan_all()
    elapsed = time.perf_counter() - start

    if not pairs:
        print("⚠ No records found in the database.")
        return pd.DataFrame()

    rows = []
    for key, record in pairs:
        if include_key:
            record = {"__key": key, **record}
        rows.append(record)

    df = pd.DataFrame(rows)

    if verbose:
        print(f"✓ Retrieved {len(df)} records in {elapsed:.3f}s")
        print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retrieve records from MiniVaultDB")
    parser.add_argument("--db", default="vault_data", help="MiniVaultDB storage path")
    parser.add_argument("--out", default=None,
                        help="Optional: save DataFrame to CSV (e.g. data/retrieved.csv)")
    parser.add_argument("--no-key", action="store_true",
                        help="Exclude the __key column from output")
    args = parser.parse_args()

    df = retrieve_to_dataframe(
        db_path=args.db,
        include_key=not args.no_key,
    )

    if df.empty:
        print("No data to display. Run pipeline/ingest.py first.")
        return

    print("\nFirst 5 rows:")
    print(df.head())

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\n✓ Saved to {out_path}")


if __name__ == "__main__":
    main()
