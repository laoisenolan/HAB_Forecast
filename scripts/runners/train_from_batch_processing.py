#!/usr/bin/env python3
"""
Production-Scale HAB Model Training on Batch Processing Dataset

This script trains species-specific XGBoost models on the large-scale batch processing dataset
(739K+ records) with optimized memory usage and performance.

Features:
- Memory-efficient processing for large datasets
- Species-specific model training with class balancing
- Robust cross-validation and hyperparameter optimization
- Comprehensive evaluation and model persistence
- Progress tracking and detailed logging

Usage:
    python scripts/train_production_models_batch_processing.py [--options]

Options:
    --data: Path to training dataset (default: cache/batch_processing/final_training_dataset.parquet)
    --output: Output directory for models (default: cache/production_models)
    --species: Specific species to train (default: all available)
    --cv_folds: Number of CV folds (default: 5)
    --test_size: Test split size (default: 0.2)
    --chunk_size: Processing chunk size for memory management (default: 50000)
    --optimize: Enable hyperparameter optimization (default: True)
"""

import argparse
import logging
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict
import warnings
import gc
import time

# ML libraries
import xgboost as xgb
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, 
    cross_val_score, validation_curve
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
    roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionHABTrainer:
    """
    Production-scale HAB model trainer optimized for large datasets
    """
    
    def __init__(self, output_dir: str = "cache/production_models", random_state: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # Model storage
        self.species_models = {}
        self.species_scalers = {}
        self.species_selectors = {}
        self.species_performance = {}
        
        # Training configuration
        self.xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Hyperparameter search space
        self.param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        
        logger.info("Production HAB Trainer Initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_and_analyze_dataset(self, data_path: str, chunk_size: int = 50000) -> pd.DataFrame:
        """Load and analyze the large dataset with memory optimization"""
        logger.info(f"Loading dataset: {data_path}")
        start_time = time.time()
        
        try:
            # Load dataset
            if data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            else:
                df = pd.read_csv(data_path)
            
            load_time = time.time() - start_time
            logger.info(f"Dataset loaded in {load_time:.2f} seconds")
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Analyze target distribution
            if 'has_hab' in df.columns:
                target_col = 'has_hab'
            elif 'hab_occurrence' in df.columns:
                target_col = 'hab_occurrence'
            else:
                raise ValueError("No valid target column found (has_hab or hab_occurrence)")
            
            target_dist = df[target_col].value_counts()
            positive_rate = target_dist.get(1, 0) / len(df)
            logger.info(f"Target distribution:")
            logger.info(f"   Positive: {target_dist.get(1, 0):,} ({positive_rate:.4%})")
            logger.info(f"   Negative: {target_dist.get(0, 0):,} ({1-positive_rate:.4%})")
            
            # Analyze species distribution if available
            if 'hab_species' in df.columns:
                species_dist = df[df[target_col] == 1]['hab_species'].value_counts()
                logger.info(f"Species distribution (top 10):")
                for i, (species, count) in enumerate(species_dist.head(10).items()):
                    logger.info(f"   {i+1}. {species}: {count:,} samples")
            
            # Check for missing values
            missing_cols = df.isnull().sum()
            missing_cols = missing_cols[missing_cols > 0]
            if len(missing_cols) > 0:
                logger.info(f"Missing values found in {len(missing_cols)} columns")
                logger.info(f"   Top missing: {missing_cols.head().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for training with memory optimization"""
        logger.info("Preparing features...")
        
        # Define columns to exclude from features
        exclude_cols = [
            'date', 'grid_cell_id', 'latitude', 'longitude',
            'has_hab', 'hab_occurrence', 'hab_species', 
            'hab_severity', 'dataset_id', 'collection_method',
            'quality_flag', 'spatial_resolution', 'temporal_resolution'
        ]
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        logger.info(f"Feature columns: {len(feature_cols)}")
        
        # Handle missing values
        df_features = df[feature_cols].copy()
        
        # Handle categorical columns - convert to numeric
        categorical_cols = df_features.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            logger.info(f"Converting {len(categorical_cols)} categorical columns to numeric")
            for col in categorical_cols:
                # Use label encoding for categorical variables
                unique_vals = df_features[col].fillna('missing').unique()
                value_map = {val: i for i, val in enumerate(unique_vals)}
                df_features[col] = df_features[col].fillna('missing').map(value_map)
        
        # Fill missing values with median for numeric columns
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_features[col].isnull().sum() > 0:
                median_val = df_features[col].median()
                df_features[col].fillna(median_val, inplace=True)
        
        # Handle infinite values
        df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_features.fillna(0, inplace=True)
        
        # Ensure all columns are numeric
        for col in df_features.columns:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
        df_features.fillna(0, inplace=True)
        
        logger.info(f"Features prepared: {df_features.shape}")
        return df_features, feature_cols
    
    def create_species_dataset(self, df: pd.DataFrame, species: str, 
                             negative_ratio: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """Create balanced dataset for specific species"""
        logger.info(f"Creating dataset for {species}")
        
        # Get target column
        target_col = 'has_hab' if 'has_hab' in df.columns else 'hab_occurrence'
        
        # Filter positive samples for this species
        if 'hab_species' in df.columns:
            positive_mask = (df[target_col] == 1) & (df['hab_species'] == species)
        else:
            positive_mask = df[target_col] == 1
        
        positive_samples = df[positive_mask]
        logger.info(f"   Positive samples: {len(positive_samples):,}")
        
        if len(positive_samples) == 0:
            logger.warning(f"No positive samples found for {species}")
            return None, None
        
        # Sample negative examples
        negative_samples = df[df[target_col] == 0]
        n_negative = min(int(len(positive_samples) * negative_ratio), len(negative_samples))
        negative_samples = negative_samples.sample(n=n_negative, random_state=self.random_state)
        logger.info(f"   Negative samples: {len(negative_samples):,}")
        
        # Combine datasets
        species_df = pd.concat([positive_samples, negative_samples], ignore_index=True)
        
        # Prepare features and target
        features_df, feature_cols = self.prepare_features(species_df)
        target = species_df[target_col].values
        
        logger.info(f"   Final dataset: {len(species_df):,} samples")
        logger.info(f"   Class balance: {np.mean(target):.4%} positive")
        
        return features_df.values, target
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                cv_folds: int = 3) -> Dict[str, Any]:
        """Optimize hyperparameters using GridSearchCV"""
        logger.info("Optimizing hyperparameters...")
        
        # Use smaller parameter grid for large datasets
        small_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        # Create XGBoost classifier
        xgb_clf = xgb.XGBClassifier(**self.xgb_params)
        
        # Setup GridSearchCV
        grid_search = GridSearchCV(
            xgb_clf, 
            small_param_grid,
            cv=cv_folds,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        # Fit with subset for large datasets
        if len(X) > 100000:
            sample_size = 50000
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample, y_sample = X[indices], y[indices]
        else:
            X_sample, y_sample = X, y
        
        grid_search.fit(X_sample, y_sample)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def train_species_model(self, species: str, df: pd.DataFrame, 
                          cv_folds: int = 5, test_size: float = 0.2,
                          optimize: bool = True) -> Dict[str, Any]:
        """Train model for specific species"""
        logger.info(f"Training model for {species}")
        logger.info("=" * 60)
        
        # Create species dataset
        X, y = self.create_species_dataset(df, species)
        if X is None:
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, 
            stratify=y, random_state=self.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection
        feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
        selector = SelectKBest(f_classif, k=min(50, X.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        selected_features = np.array(feature_cols)[selector.get_support()]
        logger.info(f"Selected features: {len(selected_features)}")
        
        # Hyperparameter optimization
        if optimize:
            best_params = self.optimize_hyperparameters(X_train_selected, y_train, cv_folds)
        else:
            best_params = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}
        
        # Train final model
        model_params = {**self.xgb_params, **best_params}
        model = xgb.XGBClassifier(**model_params)
        
        logger.info("Training final model...")
        model.fit(X_train_selected, y_train)
        
        # Evaluate model
        train_pred = model.predict(X_train_selected)
        train_prob = model.predict_proba(X_train_selected)[:, 1]
        test_pred = model.predict(X_test_selected)
        test_prob = model.predict_proba(X_test_selected)[:, 1]
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, train_pred, train_prob)
        test_metrics = self.calculate_metrics(y_test, test_pred, test_prob)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=cv_folds, scoring='f1')
        
        logger.info(f"Training Results for {species}:")
        logger.info(f"   Train F1: {train_metrics['f1']:.4f}")
        logger.info(f"   Test F1: {test_metrics['f1']:.4f}")
        logger.info(f"   Train AUC: {train_metrics['auc']:.4f}")
        logger.info(f"   Test AUC: {test_metrics['auc']:.4f}")
        logger.info(f"   CV F1: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        
        # Store model components
        self.species_models[species] = model
        self.species_scalers[species] = scaler
        self.species_selectors[species] = selector
        
        # Save model
        self.save_species_model(species, model, scaler, selector, selected_features, 
                              train_metrics, test_metrics, cv_scores, best_params)
        
        # Performance summary
        performance = {
            'species': species,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': best_params,
            'n_features': len(selected_features),
            'train_size': len(y_train),
            'test_size': len(y_test)
        }
        
        self.species_performance[species] = performance
        
        # Cleanup memory
        del X, y, X_train, X_test, X_train_scaled, X_test_scaled
        del X_train_selected, X_test_selected
        gc.collect()
        
        logger.info("Model training completed")
        return performance
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
            'ap': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        }
    
    def save_species_model(self, species: str, model, scaler, selector, 
                          selected_features, train_metrics, test_metrics, 
                          cv_scores, best_params):
        """Save trained model and metadata"""
        species_dir = self.output_dir / species.replace(' ', '_').replace('/', '_')
        species_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        with open(species_dir / 'model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open(species_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(species_dir / 'selector.pkl', 'wb') as f:
            pickle.dump(selector, f)
        
        # Save metadata
        metadata = {
            'species': species,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'XGBoost',
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': best_params,
            'selected_features': selected_features.tolist(),
            'n_features': len(selected_features)
        }
        
        with open(species_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved: {species_dir}")
    
    def train_all_species(self, data_path: str, target_species: Optional[List[str]] = None,
                         cv_folds: int = 5, test_size: float = 0.2, 
                         chunk_size: int = 50000, optimize: bool = True) -> Dict[str, Any]:
        """Train models for all or specified species"""
        logger.info("Starting production model training")
        logger.info("=" * 80)
        
        # Load dataset
        df = self.load_and_analyze_dataset(data_path, chunk_size)
        
        # Determine species to train
        if target_species is None:
            if 'hab_species' in df.columns:
                target_col = 'has_hab' if 'has_hab' in df.columns else 'hab_occurrence'
                all_species = df[df[target_col] == 1]['hab_species'].value_counts()
                # Train on species with at least 10 samples
                target_species = all_species[all_species >= 10].index.tolist()
            else:
                target_species = ['Global_HAB']  # Train global model
        
        logger.info(f"Training models for {len(target_species)} species")
        logger.info(f"   Species: {target_species}")
        
        # Train models
        results = {}
        training_start = time.time()
        
        for i, species in enumerate(target_species):
            logger.info(f"\n{'='*80}")
            logger.info(f"SPECIES {i+1}/{len(target_species)}: {species}")
            logger.info(f"{'='*80}")
            
            try:
                performance = self.train_species_model(
                    species, df, cv_folds, test_size, optimize
                )
                if performance:
                    results[species] = performance
                    logger.info(f"{species} model trained successfully")
                else:
                    logger.warning(f"Skipped {species} (insufficient data)")
                    
            except Exception as e:
                logger.error(f"Error training {species}: {e}")
                continue
        
        total_time = time.time() - training_start
        
        # Save training summary
        self.save_training_summary(results, total_time, data_path)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING COMPLETED")
        logger.info(f"   Models trained: {len(results)}")
        logger.info(f"   Total time: {total_time/60:.2f} minutes")
        logger.info(f"   Output directory: {self.output_dir}")
        logger.info(f"{'='*80}")
        
        return results
    
    def save_training_summary(self, results: Dict[str, Any], total_time: float, data_path: str):
        """Save comprehensive training summary"""
        summary = {
            'training_timestamp': datetime.now().isoformat(),
            'data_path': data_path,
            'total_time_minutes': total_time / 60,
            'models_trained': len(results),
            'species_performance': results
        }
        
        # Calculate averages
        if results:
            avg_metrics = {}
            for metric in ['f1', 'auc', 'precision', 'recall']:
                test_scores = [r['test_metrics'][metric] for r in results.values() if r['test_metrics'][metric] > 0]
                if test_scores:
                    avg_metrics[f'avg_test_{metric}'] = np.mean(test_scores)
            
            summary['average_metrics'] = avg_metrics
        
        # Save summary
        summary_path = self.output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved: {summary_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Production HAB Model Training')
    parser.add_argument('--data', default='cache/batch_processing/final_training_dataset.parquet',
                       help='Path to training dataset')
    parser.add_argument('--output', default='cache/production_models',
                       help='Output directory for models')
    parser.add_argument('--species', nargs='+', default=None,
                       help='Specific species to train')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test split size')
    parser.add_argument('--chunk_size', type=int, default=50000,
                       help='Processing chunk size')
    parser.add_argument('--no_optimize', action='store_true',
                       help='Disable hyperparameter optimization')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ProductionHABTrainer(output_dir=args.output)
    
    # Check if data file exists
    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        return
    
    logger.info(f"Using dataset: {args.data}")
    
    # Train models
    results = trainer.train_all_species(
        data_path=args.data,
        target_species=args.species,
        cv_folds=args.cv_folds,
        test_size=args.test_size,
        chunk_size=args.chunk_size,
        optimize=not args.no_optimize
    )
    
    if results:
        # Print final summary
        logger.info("\nFINAL RESULTS SUMMARY:")
        for species, perf in results.items():
            logger.info(f"   {species}: F1={perf['test_metrics']['f1']:.3f}, "
                       f"AUC={perf['test_metrics']['auc']:.3f}")
        
        logger.info(f"\nSuccessfully trained {len(results)} production models!")
    else:
        logger.warning("No models were successfully trained")


if __name__ == "__main__":
    main()