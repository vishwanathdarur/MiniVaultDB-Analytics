"""
pipeline/predict.py

End-to-end prediction on new incoming data:
    1. Ingest new records into MiniVaultDB
    2. Retrieve them back
    3. Apply saved preprocessor
    4. Run saved model
    5. Output predictions

Usage:
    python -m pipeline.predict --csv data/new_records.csv --target Survived
    python -m pipeline.predict --demo   # runs with synthetic sample data
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minivaultdb.adapter import MiniVaultDBAdapter
from pipeline.ingest import ingest_csv
from pipeline.retrieve import retrieve_to_dataframe

MODELS_DIR = Path("models")


# ──────────────────────────────────────────────────────────────────────────────
# Prediction function
# ──────────────────────────────────────────────────────────────────────────────

def predict(
    new_data: pd.DataFrame | str,
    model_path: str = "models/model_rf.pkl",
    preprocessor_path: str = "models/preprocessor.pkl",
    target_col: str | None = None,
    db_path: str = "vault_predict",
    key_prefix: str = "predict",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run predictions on new data through the full MiniVaultDB → ML pipeline.

    Parameters
    ----------
    new_data          : pandas DataFrame OR path to a CSV file
    model_path        : path to saved joblib model
    preprocessor_path : path to saved joblib ColumnTransformer
    target_col        : if present in new_data, it will be dropped before prediction
    db_path           : temp MiniVaultDB path for storing incoming records
    key_prefix        : key prefix for the temp store
    verbose           : print results

    Returns
    -------
    DataFrame with original columns + '__prediction' column.
    """
    # ── Load model & preprocessor ───────────────────────────────────────────
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Run pipeline/train.py first."
        )
    model        = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # ── Accept CSV path or DataFrame ────────────────────────────────────────
    if isinstance(new_data, str):
        df_input = pd.read_csv(new_data)
    else:
        df_input = new_data.copy()

    original_df = df_input.copy()

    # Drop target if accidentally included
    if target_col and target_col in df_input.columns:
        df_input = df_input.drop(columns=[target_col])

    # ── Ingest into MiniVaultDB ─────────────────────────────────────────────
    if verbose:
        print(f"\n── Ingesting {len(df_input)} new records into MiniVaultDB ──")

    adapter = MiniVaultDBAdapter(db_path)
    for i, row in df_input.iterrows():
        key = f"{key_prefix}_{i}"
        adapter.put_record(key, row.to_dict())

    # ── Retrieve back ───────────────────────────────────────────────────────
    if verbose:
        print("── Retrieving from MiniVaultDB ──")
    df_retrieved = retrieve_to_dataframe(db_path=db_path, include_key=False, verbose=False)

    # ── Preprocess ──────────────────────────────────────────────────────────
    # Drop high-cardinality and internal columns (mirror what train did)
    internal_cols = ["__key"] + ([target_col] if target_col else [])
    df_clean = df_retrieved.drop(
        columns=[c for c in internal_cols if c in df_retrieved.columns]
    )

    X = preprocessor.transform(df_clean)

    # ── Predict ─────────────────────────────────────────────────────────────
    predictions = model.predict(X)

    result_df = original_df.copy()
    result_df["__prediction"] = predictions

    # Add probability if the model supports it
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            result_df["__probability"] = proba[:, 1].round(4)

    if verbose:
        print(f"\n✓ Predictions complete for {len(result_df)} records")
        print(result_df[["__prediction"] +
                         (["__probability"] if "__probability" in result_df.columns else [])
                         ].value_counts().to_string())

    return result_df


# ──────────────────────────────────────────────────────────────────────────────
# Demo: create synthetic Titanic-like records and predict
# ──────────────────────────────────────────────────────────────────────────────

def run_demo():
    demo_records = pd.DataFrame({
        "PassengerId": [1000, 1001, 1002],
        "Pclass":      [1,    3,    2],
        "Name":        ["Demo, Mr. A", "Demo, Ms. B", "Demo, Dr. C"],
        "Sex":         ["male", "female", "female"],
        "Age":         [35,     22,       45],
        "SibSp":       [0,      1,        0],
        "Parch":       [0,      0,        1],
        "Ticket":      ["D001", "D002",   "D003"],
        "Fare":        [71.28,  7.25,     26.55],
        "Cabin":       [None,   None,     "B22"],
        "Embarked":    ["C",    "S",      "Q"],
    })

    print("Demo input:")
    print(demo_records[["Pclass", "Sex", "Age", "Fare"]].to_string())

    result = predict(
        new_data=demo_records,
        target_col="Survived",
        db_path="vault_demo",
        key_prefix="demo",
    )

    print("\nPredictions:")
    print(result[["Pclass", "Sex", "Age", "Fare", "__prediction",
                  "__probability"] if "__probability" in result.columns else
                 ["Pclass", "Sex", "Age", "Fare", "__prediction"]].to_string())


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run end-to-end predictions")
    parser.add_argument("--csv", default=None, help="Path to new data CSV")
    parser.add_argument("--target", default="Survived", help="Target column to drop")
    parser.add_argument("--model", default="models/model_rf.pkl")
    parser.add_argument("--preprocessor", default="models/preprocessor.pkl")
    parser.add_argument("--db", default="vault_predict")
    parser.add_argument("--out", default=None, help="Save predictions CSV")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic demo data")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if not args.csv:
        print("Provide --csv <path> or --demo")
        return

    result = predict(
        new_data=args.csv,
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        target_col=args.target,
        db_path=args.db,
    )

    if args.out:
        result.to_csv(args.out, index=False)
        print(f"✓ Predictions saved → {args.out}")
    else:
        print(result.to_string())


if __name__ == "__main__":
    main()
