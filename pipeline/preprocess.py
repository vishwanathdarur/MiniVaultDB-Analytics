"""
pipeline/preprocess.py

Cleans and transforms the retrieved DataFrame into a form ready for
machine learning: handles missing values, encodes categoricals, scales
numerics, and splits into features (X) and target (y).

Usage (as module):
    from pipeline.preprocess import preprocess
    X_train, X_test, y_train, y_test, preprocessor = preprocess(df, target_col="Survived")

Usage (CLI):
    python -m pipeline.preprocess --db vault_data --target Survived
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.retrieve import retrieve_to_dataframe


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _detect_column_types(
    df: pd.DataFrame,
    exclude_cols: list[str],
    max_categorical_cardinality: int = 20,
) -> tuple[list[str], list[str]]:
    """
    Heuristically split columns into numeric and categorical.
    Columns in exclude_cols (e.g. target, key) are always skipped.
    """
    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= max_categorical_cardinality:
                categorical_cols.append(col)
            # columns with very high cardinality (IDs, free text) are dropped

    return numeric_cols, categorical_cols


# ──────────────────────────────────────────────────────────────────────────────
# Main preprocessing function
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_cols: list[str] | None = None,
    verbose: bool = True,
) -> tuple[Any, Any, Any, Any, ColumnTransformer]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df           : raw DataFrame (as returned by retrieve_to_dataframe)
    target_col   : name of the column to predict
    test_size    : fraction reserved for testing
    random_state : reproducibility seed
    drop_cols    : additional columns to drop before modelling
    verbose      : print a summary

    Returns
    -------
    X_train, X_test, y_train, y_test, fitted_preprocessor
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. "
                         f"Available: {list(df.columns)}")

    df = df.copy()

    # ── 1. Drop unwanted columns ────────────────────────────────────────────
    always_drop = ["__key"]
    if drop_cols:
        always_drop.extend(drop_cols)

    df.drop(columns=[c for c in always_drop if c in df.columns], inplace=True)

    # ── 2. Encode target ────────────────────────────────────────────────────
    y_raw = df.pop(target_col)

    if not pd.api.types.is_numeric_dtype(y_raw):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), name=target_col)
        if verbose:
            print(f"  Target encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    else:
        y = y_raw.reset_index(drop=True)

    # Drop rows where target is null
    valid_mask = y.notna()
    df = df[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    # ── 3. Detect column types ──────────────────────────────────────────────
    numeric_cols, categorical_cols = _detect_column_types(df, exclude_cols=[])

    if verbose:
        print(f"\n  Numeric columns    ({len(numeric_cols)}): {numeric_cols}")
        print(f"  Categorical columns ({len(categorical_cols)}): {categorical_cols}")
        dropped = [c for c in df.columns
                   if c not in numeric_cols and c not in categorical_cols]
        if dropped:
            print(f"  Dropped (high-cardinality): {dropped}")

    # Drop high-cardinality text columns
    df = df[numeric_cols + categorical_cols]

    # ── 4. Build sklearn ColumnTransformer ──────────────────────────────────
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # ── 5. Train / test split ───────────────────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test  = preprocessor.transform(X_test_raw)

    if verbose:
        print(f"\n✓ Preprocessing complete")
        print(f"  Train shape : {X_train.shape}")
        print(f"  Test shape  : {X_test.shape}")
        print(f"  Class distribution (train): {dict(pd.Series(y_train).value_counts().sort_index())}")

    return X_train, X_test, y_train, y_test, preprocessor


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess data from MiniVaultDB")
    parser.add_argument("--db", default="vault_data")
    parser.add_argument("--target", default="Survived", help="Target column name")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = retrieve_to_dataframe(db_path=args.db)
    if df.empty:
        print("No data. Run pipeline/ingest.py first.")
        return

    X_train, X_test, y_train, y_test, _ = preprocess(
        df, target_col=args.target, test_size=args.test_size
    )
    print(f"\nReady for training — X_train: {X_train.shape}, X_test: {X_test.shape}")


if __name__ == "__main__":
    main()
