#!/usr/bin/env python3
"""
Comprehensive Species-Specific HAB Model Training
Uses the massive 650MB dataset to train individual models for each species
Based on your existing src/model/species_specific_trainer.py infrastructure
"""

import logging
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings

import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score,
    precision_score, recall_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_species_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveSpeciesTrainer:
    """
    Comprehensive species-specific trainer using the massive dataset
    Builds on your existing species_specific_trainer.py infrastructure
    """
    
    def __init__(self, output_dir: str = "cache/species_models_improved_massive"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Species training requirements
        self.min_samples_per_species = 10  # Minimum samples to train a model
        self.target_species = []  # Will be populated from data
        
        # XGBoost parameters for species-specific models
        self.base_xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            'n_estimators': 150,  # More trees for species-specific
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
        
        logger.info("🧬 Comprehensive Species-Specific Trainer Initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_and_analyze_dataset(self) -> pd.DataFrame:
        """Load and analyze the massive dataset for species"""
        logger.info("📊 Loading massive dataset for species analysis...")
        
        df = pd.read_parquet('cache/batch_processing/final_training_dataset.parquet')
        logger.info(f"✅ Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
        logger.info(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Analyze species distribution
        logger.info("🐟 Analyzing species distribution...")
        
        if 'hab_species' not in df.columns:
            logger.error("❌ No 'hab_species' column found!")
            return df
        
        species_counts = df['hab_species'].value_counts()
        logger.info(f"Total unique species: {len(species_counts)}")
        
        # Filter species with sufficient samples
        viable_species = species_counts[species_counts >= self.min_samples_per_species]
        logger.info(f"Species with ≥{self.min_samples_per_species} samples: {len(viable_species)}")
        
        logger.info("📈 Top species by sample count:")
        for species, count in viable_species.head(10).items():
            if species != 'none':  # Skip negative samples
                logger.info(f"  • {species}: {count:,} samples")
        
        self.target_species = [species for species in viable_species.index if species != 'none']
        logger.info(f"✅ Target species for training: {len(self.target_species)}")
        
        return df
    
    def prepare_base_features(self, df: pd.DataFrame):
        """Prepare base feature set for all species models"""
        logger.info("⚡ Preparing base feature set...")
        
        # Core environmental features
        base_features = [
            'sst', 'sst_mean', 'ocean_temperature_hycom', 'sea_surface_elevation',
            'ocean_current_u', 'ocean_current_v', 'wind_u', 'wind_v',
            'chlor_a', 'chlor_a_mean', 'turbidity_index', 'ocean_salinity',
            'precipitation', 'distance_to_coast'
        ]
        
        # Add spatial features (limited to avoid explosion)
        spatial_features = [col for col in df.columns if '_spatial' in col and '_neighbor' not in col][:15]
        base_features.extend(spatial_features)
        
        # Add temporal features (limited)
        temporal_features = [col for col in df.columns if '_lag' in col or '_ma' in col or '_diff' in col][:15]
        base_features.extend(temporal_features)
        
        # Keep only existing columns
        available_features = [col for col in base_features if col in df.columns]
        
        logger.info(f"✅ Base features available: {len(available_features)}")
        logger.info(f"  • Core environmental: {len([f for f in available_features if f in base_features[:14]])}")
        logger.info(f"  • Spatial features: {len([f for f in available_features if '_spatial' in f])}")
        logger.info(f"  • Temporal features: {len([f for f in available_features if '_lag' in f or '_ma' in f or '_diff' in f])}")
        
        return available_features
    
    def create_species_dataset(self, df: pd.DataFrame, species: str, features: list):
        """Create binary classification dataset for a specific species"""
        logger.info(f"🎯 Creating dataset for {species}...")
        
        # Create binary labels: 1 if this species, 0 otherwise
        species_labels = (df['hab_species'] == species).astype(int)
        
        positive_samples = species_labels.sum()
        negative_samples = len(species_labels) - positive_samples
        
        logger.info(f"  Positive samples: {positive_samples:,}")
        logger.info(f"  Negative samples: {negative_samples:,}")
        logger.info(f"  Positive rate: {positive_samples/len(species_labels)*100:.3f}%")
        
        # Prepare features
        X = df[features].copy()
        y = species_labels
        
        # Handle missing values
        for col in X.columns:
            if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(0)
        
        return X, y, positive_samples
    
    def train_species_model(self, X, y, species: str, positive_samples: int):
        """Train XGBoost model for a specific species"""
        logger.info(f"🚀 Training {species} model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"  Train: {len(X_train):,} samples, {y_train.sum():,} positive")
        logger.info(f"  Test:  {len(X_test):,} samples, {y_test.sum():,} positive")
        
        # Feature selection for this species
        if len(X.columns) > 20:  # Only if we have many features
            selector = SelectKBest(f_classif, k=min(20, len(X.columns)))
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            selected_features = X.columns[selector.get_support()].tolist()
            logger.info(f"  Selected {len(selected_features)} features for {species}")
        else:
            X_train_selected = X_train.values
            X_test_selected = X_test.values
            selected_features = X.columns.tolist()
            selector = None
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Calculate class weight
        if y_train.sum() > 0:
            scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        else:
            scale_pos_weight = 1.0
        
        # Species-specific XGBoost parameters
        species_params = self.base_xgb_params.copy()
        species_params['scale_pos_weight'] = scale_pos_weight
        
        # Adjust parameters based on sample size
        if positive_samples < 50:
            species_params['n_estimators'] = 100  # Fewer trees for small datasets
            species_params['max_depth'] = 4       # Shallower trees
            species_params['learning_rate'] = 0.05 # Slower learning
        
        # Train model
        model = xgb.XGBClassifier(**species_params)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        logger.info(f"  ✅ {species} Results:")
        logger.info(f"     AUC: {metrics['auc']:.3f}")
        logger.info(f"     F1:  {metrics['f1']:.3f}")
        logger.info(f"     Precision: {metrics['precision']:.3f}")
        logger.info(f"     Recall: {metrics['recall']:.3f}")
        
        # Save species model
        species_dir = self.output_dir / species.replace(' ', '_').replace('/', '_')
        species_dir.mkdir(exist_ok=True)
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'selector': selector,
            'features': selected_features,
            'metrics': metrics,
            'training_info': {
                'species': species,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'positive_samples': positive_samples,
                'positive_rate': positive_samples / len(X),
                'model_params': species_params
            }
        }
        
        # Save individual components (matching your existing structure)
        with open(species_dir / 'model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open(species_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        if selector:
            with open(species_dir / 'selector.pkl', 'wb') as f:
                pickle.dump(selector, f)
        
        # Save metadata
        metadata = {
            'species': species,
            'features': selected_features,
            'metrics': metrics,
            'training_info': model_data['training_info'],
            'created_at': datetime.now().isoformat()
        }
        
        with open(species_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"  ✅ {species} model saved to {species_dir}")
        
        return model_data
    
    def run_comprehensive_training(self):
        """Run comprehensive species-specific training"""
        logger.info("🧬 STARTING COMPREHENSIVE SPECIES-SPECIFIC TRAINING")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        
        try:
            # Load and analyze dataset
            df = self.load_and_analyze_dataset()
            
            # Prepare base features
            features = self.prepare_base_features(df)
            
            # Train models for each species
            results = {}
            successful_models = 0
            
            logger.info(f"\n🚀 Training models for {len(self.target_species)} species...")
            
            for i, species in enumerate(self.target_species, 1):
                logger.info(f"\n📊 [{i}/{len(self.target_species)}] Processing {species}")
                
                try:
                    # Create species dataset
                    X, y, positive_samples = self.create_species_dataset(df, species, features)
                    
                    # Train species model
                    model_data = self.train_species_model(X, y, species, positive_samples)
                    
                    results[species] = model_data['metrics']
                    successful_models += 1
                    
                except Exception as e:
                    logger.error(f"  ❌ Failed to train {species}: {e}")
                    results[species] = {'error': str(e)}
            
            duration = datetime.now() - start_time
            
            # Save training summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration.total_seconds(),
                'dataset_info': {
                    'total_samples': len(df),
                    'date_range': f"{df['date'].min()} to {df['date'].max()}",
                    'features_used': len(features)
                },
                'species_training': {
                    'target_species_count': len(self.target_species),
                    'successful_models': successful_models,
                    'failed_models': len(self.target_species) - successful_models
                },
                'species_results': results,
                'features': features
            }
            
            summary_path = self.output_dir / 'comprehensive_training_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info("🏆 COMPREHENSIVE SPECIES TRAINING COMPLETE!")
            logger.info(f"Duration: {duration.total_seconds():.0f} seconds")
            logger.info(f"Successful models: {successful_models}/{len(self.target_species)}")
            logger.info(f"Summary saved: {summary_path}")
            
            # Show top performing models
            successful_results = {k: v for k, v in results.items() if 'error' not in v}
            if successful_results:
                logger.info("\n🏅 Top performing models by AUC:")
                sorted_by_auc = sorted(successful_results.items(), key=lambda x: x[1]['auc'], reverse=True)
                for species, metrics in sorted_by_auc[:5]:
                    logger.info(f"  • {species}: AUC {metrics['auc']:.3f}, F1 {metrics['f1']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Comprehensive training failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    trainer = ComprehensiveSpeciesTrainer()
    trainer.run_comprehensive_training()

if __name__ == "__main__":
    main()