#!/usr/bin/env python3
"""
Fast Demo Model Training

Train production-ready HAB models for demo using the properly rebuilt dataset.
This script focuses on speed and demo-readiness.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastDemoTrainer:
    """Fast training for demo models"""
    
    def __init__(self, output_dir='cache/demo_models'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load the rebuilt dataset"""
        logger.info("Loading rebuilt HAB dataset...")
        
        df = pd.read_parquet('cache/rebuilt_dataset/hab_training_dataset_rebuilt.parquet')
        
        logger.info(f"Dataset loaded: {len(df)} records")
        logger.info(f"Positive samples: {df['has_hab'].sum()} ({df['has_hab'].mean()*100:.2f}%)")
        
        # Show species distribution
        if df['has_hab'].sum() > 0:
            species_dist = df[df['has_hab']]['hab_species'].value_counts()
            logger.info("Top HAB species:")
            for species, count in species_dist.head(5).items():
                logger.info(f"  {species}: {count}")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        logger.info("Preparing features...")
        
        # Select feature columns (exclude metadata)
        feature_cols = [col for col in df.columns if col not in [
            'date', 'grid_cell_id', 'has_hab', 'hab_species', 'hab_event_id'
        ]]
        
        X = df[feature_cols].copy()
        y = df['has_hab'].astype(int)
        
        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            logger.info(f"Converting {len(categorical_cols)} categorical columns")
            for col in categorical_cols:
                X[col] = pd.Categorical(X[col]).codes
        
        # Fill NaN values
        X = X.fillna(X.median())
        
        # Ensure all values are numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        logger.info(f"Features prepared: {X.shape}")
        logger.info(f"Feature columns: {len(feature_cols)}")
        
        return X, y, feature_cols
    
    def train_global_model(self, X, y, feature_cols):
        """Train a global HAB detection model"""
        logger.info("Training global HAB detection model...")
        logger.info("=" * 60)
        
        # Use stratified random split instead of temporal to ensure positive samples in test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        logger.info(f"Train set: {len(X_train)} samples ({y_train.sum()} positive)")
        logger.info(f"Test set: {len(X_test)} samples ({y_test.sum()} positive)")
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection
        selector = SelectKBest(f_classif, k=50)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        logger.info(f"Selected {len(selected_features)} top features")
        
        # Train XGBoost model
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train_selected, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_selected)
        y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info("Global Model Results:")
        logger.info(f"  AUC: {auc:.4f}")
        logger.info(f"  F1 Score: {report['1']['f1-score']:.4f}")
        logger.info(f"  Precision: {report['1']['precision']:.4f}")
        logger.info(f"  Recall: {report['1']['recall']:.4f}")
        
        # Save model
        model_path = self.output_dir / 'global_hab_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': scaler,
                'selector': selector,
                'features': selected_features,
                'feature_cols': feature_cols
            }, f)
        
        logger.info(f"Global model saved to {model_path}")
        
        return {
            'model_type': 'Global_HAB',
            'auc': float(auc) if not np.isnan(auc) else 0.0,
            'f1_score': float(report['1']['f1-score']),
            'precision': float(report['1']['precision']),
            'recall': float(report['1']['recall']),
            'n_features': int(len(selected_features)),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test))
        }
    
    def train_species_models(self, df, X, y, feature_cols):
        """Train species-specific models for top species"""
        logger.info("Training species-specific models...")
        logger.info("=" * 60)
        
        results = {}
        
        # Get top species with enough samples
        species_counts = df[df['has_hab']]['hab_species'].value_counts()
        top_species = species_counts[species_counts >= 100].index.tolist()[:3]  # Top 3 species
        
        logger.info(f"Training models for {len(top_species)} species")
        
        for species in top_species:
            logger.info(f"\nTraining model for: {species}")
            
            # Create binary target for this species
            species_mask = (df['hab_species'] == species) | (df['hab_species'].isna())
            df_species = df[species_mask].copy()
            X_species = X.loc[species_mask]
            y_species = (df_species['hab_species'] == species).astype(int)
            
            if y_species.sum() < 50:
                logger.warning(f"Skipping {species} - insufficient samples ({y_species.sum()})")
                continue
            
            # Balance dataset (sample negatives)
            pos_idx = y_species[y_species == 1].index
            neg_idx = y_species[y_species == 0].index
            
            # Sample 3:1 negative to positive ratio
            n_neg_sample = min(len(neg_idx), len(pos_idx) * 3)
            neg_sample = np.random.choice(neg_idx, n_neg_sample, replace=False)
            
            balanced_idx = np.concatenate([pos_idx, neg_sample])
            X_balanced = X_species.loc[balanced_idx]
            y_balanced = y_species.loc[balanced_idx]
            
            logger.info(f"  Balanced dataset: {len(X_balanced)} samples "
                       f"({y_balanced.sum()} positive, {len(y_balanced) - y_balanced.sum()} negative)")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
            )
            
            # Feature scaling and selection
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            selector = SelectKBest(f_classif, k=min(30, X_train.shape[1]))
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            # Train model
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            
            model.fit(X_train_selected, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_selected)
            y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
            
            auc = roc_auc_score(y_test, y_pred_proba)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            logger.info(f"  Results - AUC: {auc:.4f}, F1: {report['1']['f1-score']:.4f}")
            
            # Save model
            safe_species = species.replace(' ', '_').replace('/', '_')
            model_path = self.output_dir / f'{safe_species}_model.pkl'
            
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': scaler,
                    'selector': selector,
                    'species': species,
                    'features': [feature_cols[i] for i in selector.get_support(indices=True)]
                }, f)
            
            results[species] = {
                'model_type': 'Species_Specific',
                'species': species,
                'auc': float(auc),
                'f1_score': float(report['1']['f1-score']),
                'precision': float(report['1']['precision']),
                'recall': float(report['1']['recall']),
                'n_features': int(selector.get_support().sum()),
                'train_samples': int(len(X_train)),
                'test_samples': int(len(X_test))
            }
        
        return results
    
    def save_summary(self, global_result, species_results):
        """Save training summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'global_model': global_result,
            'species_models': species_results,
            'total_models': int(1 + len(species_results)),
            'demo_ready': True
        }
        
        summary_path = self.output_dir / 'demo_training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to {summary_path}")
        
        # Print demo-ready summary
        logger.info("\n" + "=" * 80)
        logger.info("DEMO MODELS READY!")
        logger.info("=" * 80)
        logger.info(f"Global HAB Model: AUC={global_result['auc']:.3f}, F1={global_result['f1_score']:.3f}")
        
        for species, result in species_results.items():
            logger.info(f"{species}: AUC={result['auc']:.3f}, F1={result['f1_score']:.3f}")
        
        logger.info(f"\nTotal models trained: {1 + len(species_results)}")
        logger.info(f"Models saved to: {self.output_dir}")
        logger.info("Ready for demo! 🚀")
    
    def run_training(self):
        """Run complete training pipeline"""
        logger.info("Starting fast demo model training...")
        
        # Load data
        df = self.load_data()
        
        # Prepare features
        X, y, feature_cols = self.prepare_features(df)
        
        # Train global model
        global_result = self.train_global_model(X, y, feature_cols)
        
        # Train species models
        species_results = self.train_species_models(df, X, y, feature_cols)
        
        # Save summary
        self.save_summary(global_result, species_results)
        
        return global_result, species_results

def main():
    trainer = FastDemoTrainer()
    trainer.run_training()

if __name__ == "__main__":
    main()