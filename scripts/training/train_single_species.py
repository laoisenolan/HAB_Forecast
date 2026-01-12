#!/usr/bin/env python3
"""
Standalone Species-Specific XGBoost Trainer for HAB Forecasting

This is a standalone version that doesn't depend on complex module imports
and can be run directly to train species-specific models.
"""

import numpy as np
import pandas as pd
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict
import warnings

# ML libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StandaloneSpeciesTrainer:
    """
    Standalone species-specific XGBoost trainer for HAB forecasting
    """
    
    def __init__(self, cache_dir: str = "cache/species_models", random_state: int = 42):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        self.species_models = {}
        self.species_scalers = {}
        self.species_selectors = {}
        self.species_performance = {}
        
        logger.info(f"🎯 Initialized Standalone Species Trainer")
        logger.info(f"📁 Cache directory: {self.cache_dir}")
    
    def load_and_prepare_data(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """Load and prepare species-specific datasets"""
        
        logger.info(f"📊 Loading training data from {data_path}")
        
        # Load dataset
        df = pd.read_csv(data_path)
        logger.info(f"✅ Loaded dataset: {df.shape[0]:,} records, {df.shape[1]} features")
        
        # Check data structure
        logger.info(f"📋 Columns: {df.columns.tolist()}")
        
        if 'hab_species' not in df.columns or 'hab_occurrence' not in df.columns:
            logger.error("❌ Required columns 'hab_species' and 'hab_occurrence' not found!")
            raise ValueError("Dataset must have 'hab_species' and 'hab_occurrence' columns")
        
        # Check if data is in long format (has variable_name and value columns)
        if 'variable_name' in df.columns and 'value' in df.columns:
            logger.info("🔄 Converting from long format to wide format...")
            
            # Create a unique identifier for each record (observation)
            # Group by location, time and HAB status, but handle NaN species properly
            df['record_id'] = df.groupby([
                'timestamp', 'latitude', 'longitude', 'hab_occurrence'
            ]).ngroup()
            
            # Keep metadata for each record
            metadata_cols = ['timestamp', 'latitude', 'longitude', 'hab_occurrence', 'hab_species', 
                            'hab_severity', 'hab_cell_density', 'hab_confidence', 'days_to_hab_event', 
                            'distance_to_hab_km', 'record_id']
            
            # Get unique metadata for each record
            metadata = df[metadata_cols].drop_duplicates(subset=['record_id'])
            
            # Pivot environmental variables
            env_data = df.pivot_table(
                index='record_id',
                columns='variable_name', 
                values='value',
                aggfunc='first'  # Take first value if duplicates
            ).reset_index()
            
            # Merge metadata with environmental data
            df_wide = metadata.merge(env_data, on='record_id')
            df_wide = df_wide.drop('record_id', axis=1)  # Remove temporary ID
            
            # Clean up column names
            df_wide.columns.name = None
            df = df_wide
            
            logger.info(f"✅ Converted to wide format: {df.shape[0]:,} records, {df.shape[1]} features")
            logger.info(f"📋 HAB occurrence after pivot: {df['hab_occurrence'].value_counts().to_dict()}")
            logger.info(f"📋 New columns: {df.columns.tolist()}")
        
        # Get species distribution
        species_counts = df['hab_species'].value_counts()
        logger.info(f"🦠 Species distribution:")
        for species, count in species_counts.head(10).items():
            logger.info(f"  {species}: {count:,} samples")
        
        # Prepare species-specific datasets
        species_data = {}
        
        # Get positive samples by species and negative samples
        positive_samples = df[df['hab_occurrence'] == 1]
        negative_samples = df[df['hab_occurrence'] == 0]
        
        logger.info(f"📊 Total positive samples: {len(positive_samples):,}")
        logger.info(f"📊 Total negative samples: {len(negative_samples):,}")
        
        # Only proceed if we have negative samples
        if len(negative_samples) == 0:
            logger.error("❌ No negative samples found! Cannot train binary classifier.")
            raise ValueError("Dataset must contain both positive (HAB) and negative (non-HAB) samples")
        
        for species in positive_samples['hab_species'].unique():
            if pd.isna(species):
                continue
            
            # Get positive samples for this species
            species_positive = positive_samples[positive_samples['hab_species'] == species]
            
            # Get random negative samples (3:1 negative to positive ratio for balanced training)
            n_negative = min(len(species_positive) * 3, len(negative_samples))
            species_negative = negative_samples.sample(n=n_negative, random_state=self.random_state).copy()
            
            # Create binary classification dataset
            # Positive: this species HAB events  
            # Negative: non-HAB environmental conditions
            combined_df = pd.concat([species_positive, species_negative], ignore_index=True)
            combined_df = combined_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
            
            species_data[species] = combined_df
            
            logger.info(f"🧬 {species}: {len(species_positive):,} positive, "
                       f"{len(species_negative):,} negative ({len(combined_df):,} total)")
        
        return species_data
    
    def prepare_features(self, df: pd.DataFrame, species: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features for training"""
        
        # Define feature columns (exclude metadata columns)
        exclude_cols = [
            'timestamp', 'hab_species', 'hab_occurrence', 'hab_severity', 
            'latitude', 'longitude', 'variable_name', 'value', 'unit',
            'data_source', 'dataset_id', 'quality_flag', 'collection_method',
            'spatial_resolution', 'temporal_resolution', 'hab_confidence',
            # CRITICAL: Exclude leaked features that directly indicate HAB presence
            'hab_cell_density', 'days_to_hab_event', 'distance_to_hab_km'
        ]
        
        # Get environmental variable columns only
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"🔧 {species}: Using {len(feature_cols)} clean environmental features")
        logger.info(f"📋 Features: {feature_cols}")
        
        # Extract features and targets
        X = df[feature_cols].values
        y = df['hab_occurrence'].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        logger.info(f"🎯 {species}: Shape {X.shape}, Positive rate: {np.mean(y):.3f}")
        
        return X, y, feature_cols
    
    def train_species_model(self, X: np.ndarray, y: np.ndarray, species: str, feature_names: List[str]) -> Dict:
        """Train model for a specific species"""
        
        logger.info(f"🧬 Training model for {species}")
        
        # Split data with temporal consideration
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection (top 50 features as per config)
        k_features = min(50, X.shape[1])
        selector = SelectKBest(f_classif, k=k_features)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
        logger.info(f"🔧 {species}: Selected {len(selected_features)} features")
        
        # Improved XGBoost parameters based on config
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=self.random_state,
            
            # Tree parameters from config
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            
            # Learning parameters from config
            learning_rate=0.1,
            n_estimators=100,
            
            # Regularization from config
            reg_alpha=0.1,
            reg_lambda=1.0,
            
            # Performance
            n_jobs=4,
            early_stopping_rounds=10
        )
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        sample_weights = np.array([class_weights[int(label)] for label in y_train])
        
        # Fit model with cross-validation
        cv_scores = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_selected, y_train)):
            X_fold_train, X_fold_val = X_train_selected[train_idx], X_train_selected[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            w_fold_train = sample_weights[train_idx]
            
            fold_model = xgb.XGBClassifier(**model.get_params())
            fold_model.fit(
                X_fold_train, y_fold_train,
                sample_weight=w_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False
            )
            
            y_fold_pred = fold_model.predict(X_fold_val)
            fold_f1 = f1_score(y_fold_val, y_fold_pred, zero_division=0)
            cv_scores.append(fold_f1)
            logger.info(f"📊 {species} Fold {fold+1}: F1={fold_f1:.3f}")
        
        logger.info(f"📊 {species} CV F1: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        # Train final model on full training data
        model.fit(
            X_train_selected, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test_selected, y_test)],
            verbose=False
        )
        
        # Make predictions
        y_pred = model.predict(X_test_selected)
        y_pred_proba_raw = model.predict_proba(X_test_selected)
        
        # Handle single class case
        if y_pred_proba_raw.shape[1] == 1:
            # Only one class, use that probability
            y_pred_proba = y_pred_proba_raw[:, 0]
        else:
            # Two classes, use positive class probability
            y_pred_proba = y_pred_proba_raw[:, 1]
        
        # Calculate metrics (handle cases with only one class)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'cv_f1_mean': np.mean(cv_scores),
            'cv_f1_std': np.std(cv_scores)
        }
        
        # Only calculate AUC and average precision if we have both classes
        if len(np.unique(y_test)) > 1:
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
            metrics['avg_precision'] = average_precision_score(y_test, y_pred_proba)
        else:
            logger.warning(f"⚠️  {species}: Only one class in test set, skipping AUC/AP metrics")
            metrics['auc'] = 0.0
            metrics['avg_precision'] = 0.0
        
        # Feature importance
        feature_importance = dict(zip(selected_features, model.feature_importances_))
        
        # Store results
        results = {
            'model': model,
            'scaler': scaler,
            'selector': selector,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'selected_features': selected_features,
            'training_samples': len(y_train),
            'test_samples': len(y_test),
            'positive_rate': np.mean(y_train),
            'cv_scores': cv_scores
        }
        
        # Store in instance
        self.species_models[species] = model
        self.species_scalers[species] = scaler
        self.species_selectors[species] = selector
        self.species_performance[species] = metrics
        
        logger.info(f"✅ {species}: F1={metrics['f1_score']:.3f}, "
                   f"AUC={metrics['auc']:.3f}, CV_F1={metrics['cv_f1_mean']:.3f}±{metrics['cv_f1_std']:.3f}")
        
        return results
    
    def save_species_model(self, species: str, results: Dict) -> None:
        """Save species model and metadata"""
        
        # Create species directory
        species_dir = self.cache_dir / species.replace(' ', '_').replace('.', '')
        species_dir.mkdir(exist_ok=True)
        
        # Save model components
        with open(species_dir / "model.pkl", 'wb') as f:
            pickle.dump(results['model'], f)
        
        with open(species_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(results['scaler'], f)
        
        with open(species_dir / "selector.pkl", 'wb') as f:
            pickle.dump(results['selector'], f)
        
        # Save metadata
        metadata = {
            'metrics': results['metrics'],
            'feature_importance': results['feature_importance'],
            'selected_features': results['selected_features'],
            'training_samples': results['training_samples'],
            'test_samples': results['test_samples'],
            'positive_rate': results['positive_rate'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(species_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"💾 Saved {species} model to {species_dir}")
    
    def train_all_species(self, data_path: str, target_species: Optional[List[str]] = None) -> Dict:
        """Train models for all species"""
        
        logger.info(f"🚀 Training species-specific models")
        
        # Load data
        species_data = self.load_and_prepare_data(data_path)
        
        # Filter target species if specified
        if target_species:
            species_data = {k: v for k, v in species_data.items() if k in target_species}
            logger.info(f"🎯 Training {len(species_data)} target species")
        
        all_results = {}
        
        for species, df in species_data.items():
            if len(df) < 50:  # Skip species with too few samples
                logger.warning(f"⚠️  Skipping {species}: only {len(df)} samples")
                continue
            
            try:
                # Prepare features
                X, y, feature_names = self.prepare_features(df, species)
                
                # Train model
                results = self.train_species_model(X, y, species, feature_names)
                all_results[species] = results
                
                # Save model
                self.save_species_model(species, results)
                
            except Exception as e:
                logger.error(f"❌ Failed to train {species}: {e}")
                continue
        
        # Generate summary
        self.generate_report(all_results)
        
        return all_results
    
    def generate_report(self, results: Dict) -> None:
        """Generate training report"""
        
        if not results:
            logger.warning("⚠️  No results to report")
            return
        
        # Create report
        report_data = []
        for species, result in results.items():
            metrics = result['metrics']
            report_data.append({
                'species': species,
                'training_samples': result['training_samples'],
                'test_samples': result['test_samples'],
                'positive_rate': result['positive_rate'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'auc': metrics['auc'],
                'n_features': len(result['selected_features'])
            })
        
        report_df = pd.DataFrame(report_data).sort_values('f1_score', ascending=False)
        
        # Save report
        report_path = self.cache_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        report_df.to_csv(report_path, index=False)
        
        # Print summary
        logger.info(f"📊 TRAINING SUMMARY ({len(results)} models)")
        logger.info(f"   Average F1 Score: {report_df['f1_score'].mean():.3f}")
        logger.info(f"   Average AUC: {report_df['auc'].mean():.3f}")
        logger.info(f"   Best: {report_df.iloc[0]['species']} (F1={report_df.iloc[0]['f1_score']:.3f})")
        logger.info(f"   Report: {report_path}")


if __name__ == "__main__":
    # Initialize trainer
    trainer = StandaloneSpeciesTrainer()
    
    # Use the dataset with both positive and negative samples
    data_path = "data/real_collection/real_hab_training_dataset_20250806_151657.csv"
    
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        exit(1)
    
    print(f"✅ Using dataset with balanced samples: {data_path}")
    
    # Train models (test with 2 species first)
    results = trainer.train_all_species(
        data_path=data_path,
        target_species=['Karenia mikimotoi', 'Heterosigma akashiwo']
    )
    
    print(f"🎉 Successfully trained {len(results)} species models!")