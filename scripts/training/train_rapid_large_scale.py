#!/usr/bin/env python3
"""
24-Hour Rapid HAB Training using the MASSIVE 650MB dataset
Uses cache/batch_processing/final_training_dataset.parquet (2017-2024)
"""

import logging
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rapid_massive_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RapidMassiveTrainer:
    """24-hour rapid trainer using the MASSIVE 650MB dataset"""
    
    def __init__(self, output_dir: str = "cache/rapid_massive_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Our coverage area
        self.coverage_bounds = {
            'lat_min': 33.575, 'lat_max': 34.425,
            'lon_min': 132.575, 'lon_max': 133.425
        }
        
        logger.info("🚨 24-Hour Rapid MASSIVE Dataset Trainer")
        logger.info(f"Coverage area: {self.coverage_bounds}")
    
    def load_massive_dataset(self) -> pd.DataFrame:
        """Load the massive 650MB dataset"""
        logger.info("📊 Loading MASSIVE 650MB dataset...")
        
        df = pd.read_parquet('cache/batch_processing/final_training_dataset.parquet')
        logger.info(f"✅ Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
        logger.info(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"🌍 Spatial: lat {df['lat'].min():.3f}-{df['lat'].max():.3f}, lon {df['lon'].min():.3f}-{df['lon'].max():.3f}")
        
        return df
    
    def filter_to_our_area(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to our coverage area"""
        logger.info("🎯 Filtering to our coverage area...")
        
        our_data = df[
            (df['lat'] >= self.coverage_bounds['lat_min']) & 
            (df['lat'] <= self.coverage_bounds['lat_max']) &
            (df['lon'] >= self.coverage_bounds['lon_min']) & 
            (df['lon'] <= self.coverage_bounds['lon_max'])
        ].copy()
        
        logger.info(f"✅ Our area samples: {len(our_data):,} ({len(our_data)/len(df)*100:.2f}% of total)")
        
        if len(our_data) > 0:
            logger.info(f"📅 Date range in our area: {our_data['date'].min()} to {our_data['date'].max()}")
        
        return our_data
    
    def analyze_hab_labels(self, df: pd.DataFrame) -> None:
        """Analyze HAB labels in the dataset"""
        logger.info("🌊 Analyzing HAB labels...")
        
        if 'has_hab' in df.columns:
            hab_counts = df['has_hab'].value_counts()
            total = len(df)
            positive = hab_counts.get(1, 0)
            negative = hab_counts.get(0, 0)
            
            logger.info(f"✅ Total samples: {total:,}")
            logger.info(f"✅ HAB positive: {positive:,} ({positive/total*100:.3f}%)")
            logger.info(f"✅ HAB negative: {negative:,} ({negative/total*100:.3f}%)")
            
            if positive == 0:
                logger.error("❌ NO POSITIVE SAMPLES!")
                return False
        else:
            logger.error("❌ No 'has_hab' column found!")
            return False
        
        if 'hab_species' in df.columns:
            species_counts = df['hab_species'].value_counts()
            logger.info(f"🐟 Unique species: {len(species_counts)}")
            logger.info("Top species:")
            for species, count in species_counts.head().items():
                logger.info(f"  • {species}: {count:,}")
        
        return True
    
    def prepare_features(self, df: pd.DataFrame):
        """Prepare features for training"""
        logger.info("⚡ Preparing features...")
        
        # Core environmental features
        feature_columns = [
            'sst', 'chlor_a', 'turbidity_index', 'ocean_temperature_hycom',
            'sea_surface_elevation', 'ocean_current_u', 'ocean_current_v',
            'wind_u', 'wind_v', 'ocean_salinity', 'precipitation'
        ]
        
        # Add available spatial features (limited)
        spatial_features = [col for col in df.columns if '_spatial' in col and '_neighbor' not in col][:10]
        feature_columns.extend(spatial_features)
        
        # Add available temporal features (limited)
        temporal_features = [col for col in df.columns if '_lag' in col or '_ma' in col][:10]
        feature_columns.extend(temporal_features)
        
        # Keep only existing columns
        available_features = [col for col in feature_columns if col in df.columns]
        logger.info(f"✅ Available features: {len(available_features)}")
        
        # Prepare feature matrix
        X = df[available_features].copy()
        y = df['has_hab'].copy()
        
        # Handle missing values
        logger.info("🔧 Handling missing values...")
        for col in X.columns:
            if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(0)
        
        logger.info(f"✅ Feature matrix: {X.shape}")
        logger.info(f"✅ Target distribution: HAB+ {y.sum():,}, HAB- {(~y.astype(bool)).sum():,}")
        
        return X, y, available_features
    
    def train_rapid_model(self, X, y, features):
        """Train rapid XGBoost model"""
        logger.info("🚀 Training rapid XGBoost model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Train: {len(X_train):,} samples, {y_train.sum():,} positive")
        logger.info(f"Test:  {len(X_test):,} samples, {y_test.sum():,} positive")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        # Train XGBoost
        logger.info("Training XGBoost...")
        
        # Calculate class weight
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        logger.info(f"✅ F1 Score: {f1:.3f}")
        logger.info(f"✅ AUC: {auc:.3f}")
        
        # Save model
        model_data = {
            'model': model,
            'scaler': scaler,
            'features': features,
            'metrics': {'f1': f1, 'auc': auc},
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'positive_samples': int(y_train.sum())
        }
        
        model_path = self.output_dir / 'rapid_massive_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✅ Model saved: {model_path}")
        
        return model_data
    
    def run_rapid_sprint(self):
        """Run the complete rapid sprint"""
        logger.info("🚨 STARTING RAPID MASSIVE DATASET SPRINT")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Load massive dataset
            df = self.load_massive_dataset()
            
            # Filter to our area
            our_data = self.filter_to_our_area(df)
            
            if len(our_data) == 0:
                logger.error("❌ No data in our coverage area!")
                return
            
            # Analyze HAB labels
            if not self.analyze_hab_labels(our_data):
                logger.error("❌ Cannot proceed without proper HAB labels!")
                return
            
            # Prepare features
            X, y, features = self.prepare_features(our_data)
            
            # Train model
            model_data = self.train_rapid_model(X, y, features)
            
            duration = datetime.now() - start_time
            
            # Save summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration.total_seconds(),
                'total_dataset_size': len(df),
                'our_area_size': len(our_data),
                'coverage_bounds': self.coverage_bounds,
                'model_metrics': model_data['metrics'],
                'features_used': len(features)
            }
            
            summary_path = self.output_dir / 'rapid_massive_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info("🏆 RAPID MASSIVE SPRINT COMPLETE!")
            logger.info(f"Duration: {duration.total_seconds():.0f} seconds")
            logger.info(f"F1: {model_data['metrics']['f1']:.3f}")
            logger.info(f"AUC: {model_data['metrics']['auc']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Sprint failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    trainer = RapidMassiveTrainer()
    trainer.run_rapid_sprint()

if __name__ == "__main__":
    main()