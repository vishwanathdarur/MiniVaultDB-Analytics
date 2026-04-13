"""
pipeline/train.py

Trains a RandomForest classifier (or any sklearn estimator) on the
preprocessed data and saves the model + preprocessor to disk.

Usage:
    python -m pipeline.train --db vault_data --target Survived
    python -m pipeline.train --db vault_data --target Survived --model lr
"""

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.retrieve import retrieve_to_dataframe
from pipeline.preprocess import preprocess

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────

def get_model(name: str):
    registry = {
        "rf":  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "lr":  LogisticRegression(max_iter=500, random_state=42),
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(registry)}")
    return registry[name]


# ──────────────────────────────────────────────────────────────────────────────
# Training function
# ──────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    db_path: str = "vault_data",
    target_col: str = "Survived",
    model_name: str = "rf",
    test_size: float = 0.2,
    drop_cols: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Full train + evaluate loop.

    Returns
    -------
    dict with keys: model, preprocessor, metrics
    """
    # 1. Retrieve
    print("\n── Step 1: Retrieve from MiniVaultDB ──")
    df = retrieve_to_dataframe(db_path=db_path, verbose=verbose)
    if df.empty:
        raise RuntimeError("No records found. Run pipeline/ingest.py first.")

    # 2. Preprocess
    print("\n── Step 2: Preprocess ──")
    X_train, X_test, y_train, y_test, preprocessor = preprocess(
        df,
        target_col=target_col,
        test_size=test_size,
        drop_cols=drop_cols,
        verbose=verbose,
    )

    # 3. Train
    print(f"\n── Step 3: Train ({model_name.upper()}) ──")
    model = get_model(model_name)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"  Training time: {train_time:.2f}s")

    # 4. Evaluate
    print("\n── Step 4: Evaluate ──")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    metrics = {"accuracy": acc, "model_name": model_name}

    # AUC (only for binary classification)
    n_classes = len(np.unique(y_train))
    if n_classes == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    if verbose:
        print(f"  Accuracy : {acc:.4f}")
        if "roc_auc" in metrics:
            print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Feature importances (RandomForest only)
    if hasattr(model, "feature_importances_") and verbose:
        try:
            # Get feature names from preprocessor
            num_names = preprocessor.transformers_[0][2] if preprocessor.transformers_ else []
            ohe = preprocessor.named_transformers_.get("cat")
            cat_names = (
                list(ohe.named_steps["encoder"].get_feature_names_out())
                if ohe is not None else []
            )
            feat_names = list(num_names) + cat_names
            importances = model.feature_importances_
            top_n = min(10, len(importances))
            top_idx = np.argsort(importances)[::-1][:top_n]
            print(f"\n  Top {top_n} feature importances:")
            for i in top_idx:
                name = feat_names[i] if i < len(feat_names) else f"feature_{i}"
                print(f"    {name:<35} {importances[i]:.4f}")
        except Exception:
            pass  # skip if feature names can't be resolved

    # 5. Save
    model_path = MODELS_DIR / f"model_{model_name}.pkl"
    prep_path  = MODELS_DIR / "preprocessor.pkl"
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, prep_path)
    print(f"\n✓ Model saved      → {model_path}")
    print(f"✓ Preprocessor saved → {prep_path}")

    return {"model": model, "preprocessor": preprocessor, "metrics": metrics}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train a model on MiniVaultDB data")
    parser.add_argument("--db", default="vault_data")
    parser.add_argument("--target", default="Survived")
    parser.add_argument("--model", default="rf", choices=["rf", "lr"])
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    train_and_evaluate(
        db_path=args.db,
        target_col=args.target,
        model_name=args.model,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
