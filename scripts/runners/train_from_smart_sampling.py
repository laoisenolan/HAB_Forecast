#!/usr/bin/env python3
"""
Train HAB models from the smart sampling dataset and save models for the dashboard.

Behavior:
- If the dataset includes 'hab_species' and 'hab_occurrence', trains species-specific XGBoost
  models using SpeciesSpecificXGBoostTrainer and saves them under
  cache/species_models_improved/<Species>/{model.pkl,scaler.pkl,selector.pkl,metadata.json}.
- Otherwise, trains a global binary XGBoost classifier on 'has_hab' (or 'hab_occurrence') and
  saves it under cache/species_models_improved/Global/.

Usage:
  python scripts/train_models_from_smart_sampling.py \
    --data cache/smart_sampling/spatial_temporal_training_dataset.parquet

Optional:
  --csv cache/smart_sampling/spatial_temporal_training_dataset.csv  # fallback
  --species "Karenia mikimotoi" "Heterosigma akashiwo" "Chattonella antiqua"
  --no-optuna  # disable hyperparameter optimization for speed
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model.species_specific_trainer import SpeciesSpecificXGBoostTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_dataset(parquet_path: Optional[Path], csv_path: Optional[Path]) -> pd.DataFrame:
    """Load the smart sampling dataset.

    Tries parquet first for speed/memory, then CSV.
    """
    if parquet_path and parquet_path.exists():
        try:
            logger.info(f"Loading dataset (parquet): {parquet_path}")
            return pd.read_parquet(parquet_path)
        except Exception as e:
            logger.warning(f"Failed to read parquet ({e}), falling back to CSV")
    if csv_path and csv_path.exists():
        logger.info(f"Loading dataset (csv): {csv_path}")
        # Use dtype hints where safe to reduce memory
        return pd.read_csv(csv_path)
    raise FileNotFoundError("No dataset found at provided parquet/csv paths")


def train_species_models(data_path_for_trainer: str, target_species: Optional[List[str]], optimize: bool) -> dict:
    """Train species models using the built-in trainer and save under species_models_improved.
    Returns the result dictionary from trainer.
    """
    trainer = SpeciesSpecificXGBoostTrainer(
        cache_dir="cache/species_models_improved",
        n_jobs=-1,
        random_state=42,
    )
    results = trainer.train_all_species(
        data_path=data_path_for_trainer,
        target_species=target_species,
        optimize_hyperparameters=optimize,
        save_models=True,
    )
    return results


def save_training_summary(results: dict, out_dir: Path) -> None:
    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_models": len(results),
        "species": [],
    }
    for species, res in results.items():
        metrics = res.get("metrics", {})
        summary["species"].append({
            "name": species,
            "f1": float(metrics.get("f1_score", 0.0)),
            "auc": float(metrics.get("auc", 0.0)),
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "training_samples": int(res.get("training_samples", 0)),
            "test_samples": int(res.get("test_samples", 0)),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved training summary: {out_dir / 'training_summary.json'}")


def train_global_model(df: pd.DataFrame, out_dir: Path) -> None:
    """Train a global binary classifier on has_hab/hab_occurrence."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
    import xgboost as xgb
    import pickle

    out_dir.mkdir(parents=True, exist_ok=True)

    # Identify target
    target_col = "has_hab" if "has_hab" in df.columns else ("hab_occurrence" if "hab_occurrence" in df.columns else None)
    if target_col is None:
        raise ValueError("Dataset missing target column ('has_hab' or 'hab_occurrence')")

    # Build feature set (drop non-predictive/location identifiers)
    exclude = {target_col, "timestamp", "date", "latitude", "longitude", "grid_cell_id", "bay_region", "hab_species"}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].fillna(0.0).values
    y = df[target_col].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selector = SelectKBest(f_classif, k=min(50, X_train_scaled.shape[1]))
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.1,
        n_estimators=200,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_sel, y_train)

    y_pred = model.predict(X_test_sel)
    y_proba = model.predict_proba(X_test_sel)[:, 1] if len(np.unique(y)) > 1 else np.zeros_like(y_test, dtype=float)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else 0.0
    }

    # Save artifacts to align with dashboard expectations
    model_dir = out_dir / "Global"
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "model.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open(model_dir / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open(model_dir / "selector.pkl", 'wb') as f:
        pickle.dump(selector, f)

    # Derive feature importance on original column names for readability
    selected_idx = selector.get_support(indices=True)
    selected_features = [feature_cols[i] for i in selected_idx]
    feature_importance = dict(zip(selected_features, model.feature_importances_.tolist()))

    metadata = {
        'species': 'Global',
        'metrics': metrics,
        'feature_importance': feature_importance,
        'selected_features': selected_features,
        'training_samples': int(len(y_train)),
        'test_samples': int(len(y_test)),
        'positive_rate': float(y_train.mean() if len(y_train) else 0.0),
        'timestamp': datetime.now().isoformat()
    }
    with open(model_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved global model under: {model_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train HAB models from smart sampling dataset")
    parser.add_argument("--data", type=str, default="cache/smart_sampling/spatial_temporal_training_dataset.parquet")
    parser.add_argument("--csv", type=str, default="cache/smart_sampling/spatial_temporal_training_dataset.csv")
    parser.add_argument("--species", nargs="*", default=[
        "Karenia mikimotoi", "Heterosigma akashiwo", "Chattonella antiqua",
        "Cochlodinium polykrikoides", "Noctiluca scintillans", "Prorocentrum minimum",
        "Skeletonema spp."
    ])
    parser.add_argument("--no-optuna", action="store_true", help="Disable hyperparameter optimization for speed")

    args = parser.parse_args()

    parquet_path = Path(args.data) if args.data else None
    csv_path = Path(args.csv) if args.csv else None

    df = read_dataset(parquet_path, csv_path)
    logger.info(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    models_out_dir = Path("cache/species_models_improved")
    models_out_dir.mkdir(parents=True, exist_ok=True)

    # If species labels are present, use the species trainer directly on CSV for compatibility
    if ("hab_species" in df.columns) and ("hab_occurrence" in df.columns):
        logger.info("Species columns detected; training species-specific models")

        # Ensure a CSV file exists for the trainer
        data_path_for_trainer = str(csv_path) if csv_path and csv_path.exists() else str(models_out_dir / "temp_training_data.csv")
        if not Path(data_path_for_trainer).exists():
            df.to_csv(data_path_for_trainer, index=False)

        results = train_species_models(
            data_path_for_trainer=data_path_for_trainer,
            target_species=args.species,
            optimize=not args.no_optuna,
        )
        save_training_summary(results, models_out_dir)
        logger.info("Species model training complete")

    else:
        logger.warning("Species columns not found; training a global model instead")
        train_global_model(df, models_out_dir)

    logger.info("All training tasks completed")


if __name__ == "__main__":
    main()

