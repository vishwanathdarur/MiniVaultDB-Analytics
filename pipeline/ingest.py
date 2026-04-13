"""
pipeline/ingest.py

Reads a structured CSV dataset and ingests every row into MiniVaultDB
as a key–value pair:
    key   → "<prefix>_<row_index>"  (or a column you designate as the ID)
    value → JSON-encoded row dict

Usage:
    python -m pipeline.ingest --csv data/dataset.csv --prefix record
    python -m pipeline.ingest --csv data/dataset.csv --id-col PassengerId
"""

import argparse
import csv
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minivaultdb.adapter import MiniVaultDBAdapter


# ──────────────────────────────────────────────────────────────────────────────
# Core ingestion function
# ──────────────────────────────────────────────────────────────────────────────

def ingest_csv(
    csv_path: str,
    db_path: str = "vault_data",
    key_prefix: str = "record",
    id_column: str | None = None,
    delimiter: str = ",",
    verbose: bool = True,
) -> int:
    """
    Ingest a CSV file into MiniVaultDB.

    Parameters
    ----------
    csv_path   : path to the CSV file
    db_path    : MiniVaultDB storage directory
    key_prefix : prefix used when auto-generating keys (ignored if id_column set)
    id_column  : if given, use this column's value as the key
    delimiter  : CSV delimiter (default ',')
    verbose    : print progress

    Returns
    -------
    Number of records successfully written.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    adapter = MiniVaultDBAdapter(db_path)

    written = 0
    skipped = 0
    start = time.perf_counter()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        if id_column and id_column not in (reader.fieldnames or []):
            raise ValueError(
                f"id_column '{id_column}' not found in CSV columns: {reader.fieldnames}"
            )

        batch: list[tuple[str, dict]] = []

        for idx, row in enumerate(reader):
            # Build key
            if id_column:
                key_val = row.get(id_column, "").strip()
                if not key_val:
                    skipped += 1
                    continue
                key = f"{key_prefix}_{key_val}"
            else:
                key = f"{key_prefix}_{idx}"

            # Clean row: strip whitespace, keep None for empty strings
            clean_row: dict = {}
            for col, val in row.items():
                if val is None or val.strip() == "":
                    clean_row[col] = None
                else:
                    # Try numeric coercion
                    stripped = val.strip()
                    try:
                        clean_row[col] = int(stripped)
                    except ValueError:
                        try:
                            clean_row[col] = float(stripped)
                        except ValueError:
                            clean_row[col] = stripped

            batch.append((key, clean_row))

            # Flush every 500 records
            if len(batch) >= 500:
                adapter.put_many(batch)
                written += len(batch)
                batch.clear()
                if verbose:
                    print(f"  Written {written} records...", end="\r")

        # Flush remainder
        if batch:
            adapter.put_many(batch)
            written += len(batch)

    elapsed = time.perf_counter() - start

    if verbose:
        print(f"\n✓ Ingestion complete")
        print(f"  Records written : {written}")
        print(f"  Records skipped : {skipped}")
        print(f"  Time taken      : {elapsed:.2f}s")
        print(f"  DB path         : {db_path}")

    return written


# ──────────────────────────────────────────────────────────────────────────────
# Download a sample dataset if none is provided
# ──────────────────────────────────────────────────────────────────────────────

def download_sample_dataset(output_path: str = "data/titanic.csv") -> str:
    """Download the Titanic dataset as a sample structured dataset."""
    import urllib.request

    url = (
        "https://raw.githubusercontent.com/datasciencedojo/datasets"
        "/master/titanic.csv"
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading sample dataset → {out}")
    urllib.request.urlretrieve(url, out)
    print(f"✓ Saved to {out}")
    return str(out)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest a CSV into MiniVaultDB")
    parser.add_argument("--csv", default=None, help="Path to input CSV file")
    parser.add_argument("--db", default="vault_data", help="MiniVaultDB storage path")
    parser.add_argument("--prefix", default="record", help="Key prefix")
    parser.add_argument("--id-col", default=None, help="Column to use as key ID")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter")
    parser.add_argument("--download-sample", action="store_true",
                        help="Download Titanic sample dataset and ingest it")
    args = parser.parse_args()

    csv_path = args.csv

    if args.download_sample or csv_path is None:
        csv_path = download_sample_dataset("data/titanic.csv")

    ingest_csv(
        csv_path=csv_path,
        db_path=args.db,
        key_prefix=args.prefix,
        id_column=args.id_col,
        delimiter=args.delimiter,
    )


if __name__ == "__main__":
    main()
