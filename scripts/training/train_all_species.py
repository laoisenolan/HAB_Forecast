#!/usr/bin/env python3
"""
Train Species-Specific Models on Full Batch Processing Dataset

This script combines all batch CSV files or uses the parquet file to train 
individual species-specific models on the complete 739K+ record dataset.

Usage:
    python scripts/train_all_species_models.py [--options]
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
import warnings
import gc
import time
import glob

# ML libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('species_training_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SpeciesSpecificTrainer:
    """
    Train individual species-specific models on the full dataset
    """
    
    def __init__(self, output_dir: str = "cache/species_models_2025", random_state: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # XGBoost parameters
        self.xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0,
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.9,
            'colsample_bytree': 0.9
        }
        
        logger.info("Species-Specific Trainer Initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_combined_dataset(self, use_csv: bool = False) -> pd.DataFrame:
        """Load the complete dataset either from parquet or combined CSV files"""
        logger.info("Loading complete dataset...")
        
        if not use_csv:
            # Try to load from parquet first
            parquet_path = "cache/batch_processing/final_training_dataset.parquet"
            if Path(parquet_path).exists():
                logger.info(f"Loading from parquet: {parquet_path}")
                df = pd.read_parquet(parquet_path)
                logger.info(f"Dataset loaded: {df.shape}")
                return df
        
        # Load and combine all CSV files from batches
        logger.info("Loading and combining CSV files from batches...")
        csv_files = glob.glob("cache/batch_processing/batches/*/spatial_temporal_training_dataset.csv")
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in batch directories")
        
        logger.info(f"Found {len(csv_files)} CSV files to combine")
        
        dfs = []
        for i, csv_file in enumerate(csv_files):
            try:
                batch_df = pd.read_csv(csv_file)
                dfs.append(batch_df)
                if (i + 1) % 12 == 0:  # Progress every 12 files (1 year)
                    logger.info(f"Loaded {i+1}/{len(csv_files)} files...")
            except Exception as e:
                logger.warning(f"Error loading {csv_file}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No valid CSV files could be loaded")
        
        # Combine all dataframes
        logger.info("Combining all datasets...")
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined dataset shape: {df.shape}")
        
        return df
    
    def load_hab_species_data(self) -> pd.DataFrame:
        """Load HAB species data to get species names"""
        hab_file = "test_data/HAB_2017_2023.csv"
        if Path(hab_file).exists():
            hab_df = pd.read_csv(hab_file)
            logger.info(f"HAB species data loaded: {hab_df.shape}")
            return hab_df
        else:
            logger.warning("HAB species file not found, will train global model only")
            return None
    
    def get_target_species(self, df: pd.DataFrame, hab_df: pd.DataFrame, 
                          min_samples: int = 20) -> List[str]:
        """Get list of species to train models for"""
        logger.info("Determining target species for training...")
        
        if hab_df is None:
            logger.info("No species data available, training global model")
            return ['Global_HAB']
        
        # Get species with sufficient samples
        species_counts = hab_df['plankton_species'].value_counts()
        target_species = species_counts[species_counts >= min_samples].index.tolist()
        
        logger.info(f"Found {len(target_species)} species with >= {min_samples} samples:")
        for species, count in species_counts.head(10).items():
            logger.info(f"  {species}: {count} samples")
        
        return target_species
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for training"""
        logger.info("Preparing features...")
        
        # Define columns to exclude
        exclude_cols = [
            'date', 'grid_cell_id', 'latitude', 'longitude',
            'has_hab', 'hab_occurrence', 'hab_species', 
            'hab_severity', 'dataset_id', 'collection_method',
            'quality_flag', 'spatial_resolution', 'temporal_resolution'
        ]
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        logger.info(f"Feature columns: {len(feature_cols)}")
        
        df_features = df[feature_cols].copy()
        
        # Handle categorical columns
        categorical_cols = df_features.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            logger.info(f"Converting {len(categorical_cols)} categorical columns")
            for col in categorical_cols:
                unique_vals = df_features[col].fillna('missing').unique()
                value_map = {val: i for i, val in enumerate(unique_vals)}
                df_features[col] = df_features[col].fillna('missing').map(value_map)
        
        # Handle missing values
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_features[col].isnull().sum() > 0:
                median_val = df_features[col].median()
                df_features[col].fillna(median_val, inplace=True)
        
        # Handle infinite values
        df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_features.fillna(0, inplace=True)
        
        # Ensure all numeric
        for col in df_features.columns:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
        df_features.fillna(0, inplace=True)
        
        logger.info(f"Features prepared: {df_features.shape}")
        return df_features, feature_cols
    
    def create_species_dataset(self, df: pd.DataFrame, hab_df: pd.DataFrame, 
                             species: str, negative_ratio: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """Create balanced dataset for specific species"""
        logger.info(f"Creating dataset for {species}")
        
        if species == 'Global_HAB':
            # Global model - use all HAB positive vs negative
            positive_mask = df['has_hab'] == 1
            positive_samples = df[positive_mask]
        else:
            # Species-specific model using date-based matching
            if hab_df is None:
                logger.error("HAB species data required for species-specific models")
                return None, None
            
            # Get HAB events for this species
            species_events = hab_df[hab_df['plankton_species'] == species]
            logger.info(f"  Found {len(species_events)} HAB events for {species}")
            
            # Create positive samples by matching dates (simplified approach)
            positive_samples = []
            for _, event in species_events.iterrows():
                try:
                    # Find records from the same time period
                    event_date = pd.to_datetime(event['start_date']).date()
                    df_dates = pd.to_datetime(df['date']).dt.date
                    
                    # Match records from the same date
                    date_matches = df[df_dates == event_date]
                    
                    if len(date_matches) > 0:
                        # Take a sample of records from that date
                        sample_size = min(10, len(date_matches))  # Max 10 records per event
                        sampled = date_matches.sample(n=sample_size, random_state=self.random_state)
                        positive_samples.append(sampled)
                
                except Exception as e:
                    logger.debug(f"Error processing event: {e}")
                    continue
            
            if not positive_samples:
                logger.warning(f"No matching data found for {species}")
                return None, None
            
            positive_samples = pd.concat(positive_samples, ignore_index=True)
            positive_samples = positive_samples.drop_duplicates()
        
        logger.info(f"  Positive samples: {len(positive_samples):,}")
        
        if len(positive_samples) == 0:
            return None, None
        
        # Sample negative examples
        negative_samples = df[df['has_hab'] == 0]
        n_negative = min(int(len(positive_samples) * negative_ratio), len(negative_samples))
        negative_samples = negative_samples.sample(n=n_negative, random_state=self.random_state)
        logger.info(f"  Negative samples: {len(negative_samples):,}")
        
        # Combine datasets
        species_df = pd.concat([positive_samples, negative_samples], ignore_index=True)
        
        # Prepare features and target
        features_df, feature_cols = self.prepare_features(species_df)
        
        # Create target (1 for positive, 0 for negative)
        target = np.zeros(len(species_df))
        target[:len(positive_samples)] = 1
        
        logger.info(f"  Final dataset: {len(species_df):,} samples")
        logger.info(f"  Class balance: {np.mean(target):.4%} positive")
        
        return features_df.values, target
    
    def train_species_model(self, species: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train model for specific species"""
        logger.info(f"Training model for {species}")
        logger.info("=" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection
        selector = SelectKBest(f_classif, k=min(50, X.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        logger.info(f"Selected features: {X_train_selected.shape[1]}")
        
        # Train model
        model = xgb.XGBClassifier(**self.xgb_params)
        model.fit(X_train_selected, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train_selected)
        train_prob = model.predict_proba(X_train_selected)[:, 1]
        test_pred = model.predict(X_test_selected)
        test_prob = model.predict_proba(X_test_selected)[:, 1]
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, train_pred, train_prob)
        test_metrics = self.calculate_metrics(y_test, test_pred, test_prob)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='f1')
        
        logger.info(f"Results for {species}:")
        logger.info(f"  Train F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        logger.info(f"  Test F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")
        logger.info(f"  CV F1: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        
        # Save model
        self.save_model(species, model, scaler, selector, train_metrics, test_metrics, cv_scores)
        
        return {
            'species': species,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_features': X_train_selected.shape[1],
            'train_size': len(y_train),
            'test_size': len(y_test)
        }
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
            'ap': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        }
    
    def save_model(self, species: str, model, scaler, selector, 
                   train_metrics, test_metrics, cv_scores):
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
            'model_type': 'XGBoost_SpeciesSpecific',
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_features': selector.k
        }
        
        with open(species_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved: {species_dir}")
    
    def train_all_species(self, target_species: Optional[List[str]] = None, 
                         use_csv: bool = False, min_samples: int = 20) -> Dict[str, Any]:
        """Train models for all species"""
        logger.info("Starting species-specific model training")
        logger.info("=" * 80)
        
        # Load datasets
        df = self.load_combined_dataset(use_csv)
        hab_df = self.load_hab_species_data()
        
        # Determine species to train
        if target_species is None:
            target_species = self.get_target_species(df, hab_df, min_samples)
        
        logger.info(f"Training models for {len(target_species)} species")
        
        # Train models
        results = {}
        training_start = time.time()
        
        for i, species in enumerate(target_species):
            logger.info(f"\n{'='*80}")
            logger.info(f"SPECIES {i+1}/{len(target_species)}: {species}")
            logger.info(f"{'='*80}")
            
            try:
                # Create species dataset
                X, y = self.create_species_dataset(df, hab_df, species)
                if X is None:
                    logger.warning(f"Skipped {species} (insufficient data)")
                    continue
                
                # Train model
                performance = self.train_species_model(species, X, y)
                results[species] = performance
                
                # Cleanup memory
                del X, y
                gc.collect()
                
                logger.info(f"{species} model trained successfully")
                
            except Exception as e:
                logger.error(f"Error training {species}: {e}")
                continue
        
        total_time = time.time() - training_start
        
        # Save summary
        self.save_training_summary(results, total_time)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING COMPLETED")
        logger.info(f"  Models trained: {len(results)}")
        logger.info(f"  Total time: {total_time/60:.2f} minutes")
        logger.info(f"{'='*80}")
        
        return results
    
    def save_training_summary(self, results: Dict[str, Any], total_time: float):
        """Save training summary"""
        summary = {
            'training_timestamp': datetime.now().isoformat(),
            'total_time_minutes': total_time / 60,
            'models_trained': len(results),
            'species_performance': results
        }
        
        if results:
            # Calculate averages
            avg_metrics = {}
            for metric in ['f1', 'auc', 'precision', 'recall']:
                test_scores = [r['test_metrics'][metric] for r in results.values()]
                avg_metrics[f'avg_test_{metric}'] = np.mean(test_scores)
            summary['average_metrics'] = avg_metrics
        
        summary_path = self.output_dir / 'species_training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved: {summary_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Species-Specific HAB Model Training')
    parser.add_argument('--output', default='cache/species_models_2025',
                       help='Output directory for models')
    parser.add_argument('--species', nargs='+', default=None,
                       help='Specific species to train')
    parser.add_argument('--use_csv', action='store_true',
                       help='Use CSV files instead of parquet')
    parser.add_argument('--min_samples', type=int, default=20,
                       help='Minimum samples required per species')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SpeciesSpecificTrainer(output_dir=args.output)
    
    # Train models
    results = trainer.train_all_species(
        target_species=args.species,
        use_csv=args.use_csv,
        min_samples=args.min_samples
    )
    
    if results:
        logger.info("\nFINAL RESULTS SUMMARY:")
        for species, perf in results.items():
            logger.info(f"  {species}: F1={perf['test_metrics']['f1']:.3f}, "
                       f"AUC={perf['test_metrics']['auc']:.3f}")
        
        logger.info(f"\nSuccessfully trained {len(results)} species-specific models!")
    else:
        logger.warning("No models were successfully trained")


if __name__ == "__main__":
    main()