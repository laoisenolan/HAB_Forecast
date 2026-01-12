#!/usr/bin/env python3
"""
ULTIMATE Species-Specific HAB Model Training
Uses the species-integrated dataset: cache/species_integrated/environmental_with_species.parquet
Trains individual XGBoost models for each HAB species with proper evaluation
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
    precision_score, recall_score, accuracy_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_species_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UltimateSpeciesTrainer:
    """
    Ultimate species-specific trainer using the integrated species-environment dataset
    """
    
    def __init__(self, output_dir: str = "cache/species_models_ultimate"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training requirements
        self.min_samples_per_species = 5  # Lower threshold since we have limited species data
        self.target_species = []
        
        # XGBoost parameters optimized for species prediction
        self.base_xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            'n_estimators': 200,  # More trees for better performance
            'max_depth': 8,       # Deeper trees for complex patterns
            'learning_rate': 0.05, # Slower learning for better generalization
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
        
        logger.info("🧬 ULTIMATE Species-Specific Trainer Initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_species_integrated_dataset(self) -> pd.DataFrame:
        """Load the species-integrated dataset"""
        logger.info("📊 Loading species-integrated dataset...")
        
        df = pd.read_parquet('cache/species_integrated/environmental_with_species.parquet')
        logger.info(f"✅ Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
        logger.info(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def analyze_species_distribution(self, df: pd.DataFrame):
        """Analyze species distribution in the integrated dataset"""
        logger.info("🐟 Analyzing species distribution...")
        
        # Overall HAB distribution
        hab_counts = df['has_hab'].value_counts()
        logger.info(f"Total samples: {len(df):,}")
        logger.info(f"HAB positive: {hab_counts.get(1, 0):,} ({hab_counts.get(1, 0)/len(df)*100:.3f}%)")
        logger.info(f"HAB negative: {hab_counts.get(0, 0):,} ({hab_counts.get(0, 0)/len(df)*100:.3f}%)")
        
        # Species distribution (only positive samples)
        hab_positive = df[df['has_hab'] == 1]
        species_counts = hab_positive['hab_species'].value_counts()
        
        logger.info(f"\n🧬 Species distribution in positive samples:")
        logger.info(f"Total positive samples: {len(hab_positive):,}")
        logger.info(f"Unique species: {len(species_counts[species_counts.index != 'none'])}")
        
        for species, count in species_counts.items():
            if species != 'none':
                logger.info(f"  • {species}: {count:,} samples ({count/len(hab_positive)*100:.1f}%)")
        
        # Determine viable species for training
        viable_species = species_counts[
            (species_counts >= self.min_samples_per_species) & 
            (species_counts.index != 'none')
        ]
        
        self.target_species = viable_species.index.tolist()
        logger.info(f"\n✅ Species viable for training (≥{self.min_samples_per_species} samples): {len(self.target_species)}")
        
        return hab_positive, species_counts
    
    def prepare_features_for_species_training(self, df: pd.DataFrame):
        """Prepare optimized feature set for species training"""
        logger.info("⚡ Preparing features for species training...")
        
        # Core environmental features
        core_features = [
            'sst', 'sst_mean', 'ocean_temperature_hycom', 'sea_surface_elevation',
            'ocean_current_u', 'ocean_current_v', 'wind_u', 'wind_v',
            'chlor_a', 'chlor_a_mean', 'turbidity_index', 'ocean_salinity',
            'precipitation', 'distance_to_coast'
        ]
        
        # Enhanced spatial features
        spatial_features = [col for col in df.columns if '_spatial' in col and '_neighbor' not in col][:20]
        
        # Enhanced temporal features
        temporal_features = [col for col in df.columns if any(x in col for x in ['_lag', '_ma', '_diff', '_seasonal'])][:20]
        
        # Biogeochemical features
        bio_features = [col for col in df.columns if any(x in col for x in ['nutrient', 'phyto', 'zoo', 'carbon'])][:10]
        
        # Combine all features
        all_features = core_features + spatial_features + temporal_features + bio_features
        
        # Keep only existing columns
        available_features = [col for col in all_features if col in df.columns]
        
        logger.info(f"✅ Features selected: {len(available_features)}")
        logger.info(f"  • Core environmental: {len([f for f in available_features if f in core_features])}")
        logger.info(f"  • Spatial: {len([f for f in available_features if '_spatial' in f])}")
        logger.info(f"  • Temporal: {len([f for f in available_features if any(x in f for x in ['_lag', '_ma', '_diff'])])}")
        logger.info(f"  • Biogeochemical: {len([f for f in available_features if any(x in f for x in ['nutrient', 'phyto', 'zoo'])])}")
        
        return available_features
    
    def create_species_binary_dataset(self, df: pd.DataFrame, target_species: str, features: list):
        """Create binary classification dataset for a specific species"""
        logger.info(f"🎯 Creating binary dataset for: {target_species}")
        
        # Create binary labels: 1 if this species, 0 if negative OR different species
        # This approach treats each species vs. "everything else"
        species_mask = df['hab_species'] == target_species
        negative_mask = df['has_hab'] == 0
        
        # Include all negative samples + positive samples of this species
        dataset_mask = species_mask | negative_mask
        subset_df = df[dataset_mask].copy()
        
        # Create binary target: 1 for target species, 0 for everything else
        y = (subset_df['hab_species'] == target_species).astype(int)
        
        positive_samples = y.sum()
        negative_samples = len(y) - positive_samples
        
        logger.info(f"  Species samples: {positive_samples:,}")
        logger.info(f"  Non-species samples: {negative_samples:,}")
        logger.info(f"  Positive rate: {positive_samples/len(y)*100:.3f}%")
        
        # Prepare features
        X = subset_df[features].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(0)
        
        return X, y, positive_samples
    
    def train_species_model(self, X, y, species: str, positive_samples: int):
        """Train optimized XGBoost model for a specific species"""
        logger.info(f"🚀 Training model for: {species}")
        logger.info(f"   Total samples: {len(X):,}, Positive: {positive_samples:,}")
        
        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"   Train: {len(X_train):,} samples, {y_train.sum():,} positive")
        logger.info(f"   Test:  {len(X_test):,} samples, {y_test.sum():,} positive")
        
        # Feature selection (if we have many features)
        if len(X.columns) > 30:
            k_features = min(30, len(X.columns))
            selector = SelectKBest(f_classif, k=k_features)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            selected_features = X.columns[selector.get_support()].tolist()
            logger.info(f"   Selected {len(selected_features)} most informative features")
        else:
            X_train_selected = X_train.values
            X_test_selected = X_test.values
            selected_features = X.columns.tolist()
            selector = None
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Calculate class weight for imbalanced data
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1.0
        
        # Adjust parameters based on sample size
        species_params = self.base_xgb_params.copy()
        species_params['scale_pos_weight'] = scale_pos_weight
        
        if positive_samples < 20:
            # Conservative parameters for very small datasets
            species_params.update({
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.03,
                'subsample': 0.9,
                'colsample_bytree': 0.9
            })
            logger.info("   Using conservative parameters for small dataset")
        elif positive_samples < 50:
            # Moderate parameters
            species_params.update({
                'n_estimators': 150,
                'max_depth': 7,
                'learning_rate': 0.04
            })
        
        # Train XGBoost model
        logger.info("   Training XGBoost...")
        model = xgb.XGBClassifier(**species_params)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate comprehensive metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        logger.info(f"   ✅ {species} Results:")
        logger.info(f"      AUC: {metrics['auc']:.3f}")
        logger.info(f"      F1:  {metrics['f1']:.3f}")
        logger.info(f"      Precision: {metrics['precision']:.3f}")
        logger.info(f"      Recall: {metrics['recall']:.3f}")
        logger.info(f"      Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        
        # Feature importance
        feature_importance = model.feature_importances_
        top_features = []
        if len(selected_features) == len(feature_importance):
            importance_pairs = list(zip(selected_features, feature_importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            top_features = [(feat, imp) for feat, imp in importance_pairs[:5]]
            
            logger.info(f"   🔥 Top 5 features for {species}:")
            for feat, imp in top_features:
                logger.info(f"      • {feat}: {imp:.3f}")
        
        # Save species model
        species_safe_name = species.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        species_dir = self.output_dir / species_safe_name
        species_dir.mkdir(exist_ok=True)
        
        # Complete model package
        model_package = {
            'model': model,
            'scaler': scaler,
            'selector': selector,
            'features': selected_features,
            'metrics': metrics,
            'feature_importance': top_features,
            'confusion_matrix': {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)},
            'training_info': {
                'species': species,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'positive_samples': positive_samples,
                'positive_rate': positive_samples / len(X),
                'model_params': species_params,
                'created_at': datetime.now().isoformat()
            }
        }
        
        # Save individual components (compatible with your existing infrastructure)
        with open(species_dir / 'model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open(species_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        if selector:
            with open(species_dir / 'selector.pkl', 'wb') as f:
                pickle.dump(selector, f)
        
        # Enhanced metadata
        metadata = {
            'species': species,
            'features': selected_features,
            'feature_count': len(selected_features),
            'metrics': metrics,
            'feature_importance': top_features,
            'confusion_matrix': model_package['confusion_matrix'],
            'training_info': model_package['training_info']
        }
        
        with open(species_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"   ✅ Model saved: {species_dir}")
        
        return model_package
    
    def run_ultimate_species_training(self):
        """Run the complete ultimate species training pipeline"""
        logger.info("🧬 STARTING ULTIMATE SPECIES-SPECIFIC TRAINING")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Load species-integrated dataset
            df = self.load_species_integrated_dataset()
            
            # Analyze species distribution
            hab_positive, species_counts = self.analyze_species_distribution(df)
            
            if len(self.target_species) == 0:
                logger.error("❌ No viable species found for training!")
                return
            
            # Prepare features
            features = self.prepare_features_for_species_training(df)
            
            # Train models for each viable species
            logger.info(f"\n🚀 Training models for {len(self.target_species)} species...")
            
            results = {}
            successful_models = 0
            model_performances = []
            
            for i, species in enumerate(self.target_species, 1):
                logger.info(f"\n📊 [{i}/{len(self.target_species)}] Processing: {species}")
                
                try:
                    # Create species-specific dataset
                    X, y, positive_samples = self.create_species_binary_dataset(df, species, features)
                    
                    # Train model
                    model_package = self.train_species_model(X, y, species, positive_samples)
                    
                    # Store results
                    results[species] = model_package['metrics']
                    model_performances.append((species, model_package['metrics']['auc'], model_package['metrics']['f1']))
                    successful_models += 1
                    
                except Exception as e:
                    logger.error(f"   ❌ Failed to train {species}: {e}")
                    results[species] = {'error': str(e)}
            
            duration = datetime.now() - start_time
            
            # Save comprehensive training summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration.total_seconds(),
                'dataset_info': {
                    'total_samples': len(df),
                    'hab_positive_samples': len(hab_positive),
                    'date_range': f"{df['date'].min()} to {df['date'].max()}",
                    'features_used': len(features),
                    'feature_list': features
                },
                'species_analysis': {
                    'total_unique_species': len(species_counts[species_counts.index != 'none']),
                    'viable_species_count': len(self.target_species),
                    'species_distribution': species_counts.to_dict()
                },
                'training_results': {
                    'successful_models': successful_models,
                    'failed_models': len(self.target_species) - successful_models,
                    'species_results': results
                },
                'model_performances': model_performances
            }
            
            summary_path = self.output_dir / 'ultimate_species_training_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Final results
            logger.info("🏆 ULTIMATE SPECIES TRAINING COMPLETE!")
            logger.info(f"Duration: {duration.total_seconds():.0f} seconds")
            logger.info(f"Successful models: {successful_models}/{len(self.target_species)}")
            logger.info(f"Summary saved: {summary_path}")
            
            # Show top performing models
            if model_performances:
                model_performances.sort(key=lambda x: x[1], reverse=True)  # Sort by AUC
                logger.info("\n🏅 TOP PERFORMING SPECIES MODELS:")
                for species, auc, f1 in model_performances[:10]:
                    logger.info(f"   • {species[:40]:40} | AUC: {auc:.3f} | F1: {f1:.3f}")
            
            logger.info(f"\n🎯 MODELS READY FOR DEPLOYMENT!")
            logger.info(f"   Model directory: {self.output_dir}")
            logger.info(f"   Compatible with existing infrastructure")
            
        except Exception as e:
            logger.error(f"❌ Ultimate training failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    trainer = UltimateSpeciesTrainer()
    trainer.run_ultimate_species_training()

if __name__ == "__main__":
    main()