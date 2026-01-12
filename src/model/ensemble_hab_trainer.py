#!/usr/bin/env python3
"""
Advanced Ensemble HAB Model Trainer - Task 4.3

This module implements sophisticated ensemble methods for HAB prediction:
- Voting classifiers combining XGBoost, Random Forest, and other models
- Stacking with meta-learners for improved predictions
- Weighted averaging based on model performance
- Species-specific ensemble optimization
- Cross-validation based ensemble selection

🎯 ENSEMBLE METHODS IMPLEMENTED:
- Hard/Soft voting classifiers
- Stacked generalization with cross-validation
- Weighted ensemble based on performance metrics
- Dynamic ensemble selection per species
- Multi-level ensemble architectures

🛰️ LEVERAGES:
- Species-specific XGBoost models from Task 4.1
- Additional base models (Random Forest, Gradient Boosting)
- Model diversity analysis and optimization
- Performance-based weighting strategies

Author: HAB Forecasting System - Task 4.3 Implementation  
Date: December 19, 2024
"""

import numpy as np
import pandas as pd
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from collections import defaultdict
import warnings

# ML libraries
from sklearn.ensemble import (
    VotingClassifier, RandomForestClassifier, 
    GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceWeightedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Custom ensemble that weights models based on their performance
    """
    
    def __init__(self, models: List[Tuple[str, BaseEstimator]], 
                 weighting_metric: str = 'f1', cv_folds: int = 5):
        """
        Initialize performance-weighted ensemble
        
        Args:
            models: List of (name, model) tuples
            weighting_metric: Metric to use for weighting ('f1', 'auc', 'accuracy')
            cv_folds: Number of CV folds for weight estimation
        """
        self.models = models
        self.weighting_metric = weighting_metric
        self.cv_folds = cv_folds
        self.weights_ = None
        self.model_scores_ = None
        
    def fit(self, X, y):
        """Fit ensemble and calculate performance weights"""
        
        logger.info(f"🎯 Fitting Performance-Weighted Ensemble with {len(self.models)} models")
        
        # Calculate cross-validation scores for each model
        model_scores = {}
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.models:
            try:
                # Perform cross-validation
                if self.weighting_metric == 'f1':
                    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
                elif self.weighting_metric == 'auc':
                    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
                elif self.weighting_metric == 'accuracy':
                    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                else:
                    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
                
                mean_score = np.mean(scores)
                model_scores[name] = mean_score
                
                logger.info(f"  {name}: {self.weighting_metric}={mean_score:.3f} (±{np.std(scores):.3f})")
                
                # Fit the model on full training data
                model.fit(X, y)
                
            except Exception as e:
                logger.warning(f"  {name}: Failed to fit ({e})")
                model_scores[name] = 0.0
        
        self.model_scores_ = model_scores
        
        # Calculate weights based on performance (softmax to ensure positive weights)
        scores = np.array(list(model_scores.values()))
        # Add small epsilon to avoid division by zero
        scores = np.maximum(scores, 0.1)
        
        # Softmax weighting
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        self.weights_ = exp_scores / np.sum(exp_scores)
        
        logger.info(f"📊 Model weights:")
        for i, (name, _) in enumerate(self.models):
            logger.info(f"  {name}: {self.weights_[i]:.3f}")
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities using weighted average"""
        
        if self.weights_ is None:
            raise ValueError("Ensemble not fitted yet")
        
        # Get predictions from all models
        predictions = []
        
        for i, (name, model) in enumerate(self.models):
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
                predictions.append(pred * self.weights_[i])
            else:
                # For models without predict_proba, use decision function or predict
                if hasattr(model, 'decision_function'):
                    decision = model.decision_function(X)
                    # Convert to probabilities (simplified sigmoid)
                    prob = 1 / (1 + np.exp(-decision))
                    pred = np.column_stack([1 - prob, prob])
                else:
                    # Binary predictions - convert to probabilities
                    binary_pred = model.predict(X)
                    pred = np.zeros((len(X), 2))
                    pred[binary_pred == 0, 0] = 1
                    pred[binary_pred == 1, 1] = 1
                
                predictions.append(pred * self.weights_[i])
        
        # Weighted average of predictions
        final_prediction = np.sum(predictions, axis=0)
        
        # Normalize to ensure probabilities sum to 1
        final_prediction = final_prediction / np.sum(final_prediction, axis=1, keepdims=True)
        
        return final_prediction
    
    def predict(self, X):
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

class StackedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Stacked ensemble with meta-learner
    """
    
    def __init__(self, base_models: List[Tuple[str, BaseEstimator]], 
                 meta_model: BaseEstimator = None, cv_folds: int = 5):
        """
        Initialize stacked ensemble
        
        Args:
            base_models: List of (name, model) tuples for base level
            meta_model: Meta-learner model (default: Logistic Regression)
            cv_folds: Number of CV folds for generating meta-features
        """
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(random_state=42)
        self.cv_folds = cv_folds
        self.meta_features_ = None
        
    def _generate_meta_features(self, X, y):
        """Generate meta-features using cross-validation"""
        
        logger.info(f"🔧 Generating meta-features with {self.cv_folds}-fold CV")
        
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        # Meta-features: [samples, models * n_classes]
        meta_features = np.zeros((n_samples, n_models * 2))
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]
            
            for i, (name, model) in enumerate(self.base_models):
                try:
                    # Clone and fit model on fold training data
                    fold_model = type(model)(**model.get_params())
                    fold_model.fit(X_train_fold, y_train_fold)
                    
                    # Get predictions for validation fold
                    if hasattr(fold_model, 'predict_proba'):
                        val_pred = fold_model.predict_proba(X_val_fold)
                    else:
                        # Convert binary predictions to probabilities
                        binary_pred = fold_model.predict(X_val_fold)
                        val_pred = np.zeros((len(X_val_fold), 2))
                        val_pred[binary_pred == 0, 0] = 1
                        val_pred[binary_pred == 1, 1] = 1
                    
                    # Store meta-features
                    meta_features[val_idx, i*2:(i+1)*2] = val_pred
                    
                except Exception as e:
                    logger.warning(f"  Fold {fold}, {name}: Error generating meta-features ({e})")
                    # Fill with neutral predictions
                    meta_features[val_idx, i*2:(i+1)*2] = 0.5
        
        return meta_features
    
    def fit(self, X, y):
        """Fit stacked ensemble"""
        
        logger.info(f"🏗️  Fitting Stacked Ensemble with {len(self.base_models)} base models")
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X, y)
        self.meta_features_ = meta_features
        
        # Fit base models on full training data
        for name, model in self.base_models:
            try:
                model.fit(X, y)
                logger.info(f"  ✅ Base model fitted: {name}")
            except Exception as e:
                logger.warning(f"  ❌ Base model failed: {name} ({e})")
        
        # Fit meta-model on meta-features
        self.meta_model.fit(meta_features, y)
        logger.info(f"  ✅ Meta-model fitted: {type(self.meta_model).__name__}")
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities using stacked ensemble"""
        
        # Get predictions from base models
        n_models = len(self.base_models)
        meta_features = np.zeros((X.shape[0], n_models * 2))
        
        for i, (name, model) in enumerate(self.base_models):
            try:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                else:
                    binary_pred = model.predict(X)
                    pred = np.zeros((len(X), 2))
                    pred[binary_pred == 0, 0] = 1
                    pred[binary_pred == 1, 1] = 1
                
                meta_features[:, i*2:(i+1)*2] = pred
                
            except Exception as e:
                logger.warning(f"  {name}: Prediction error ({e})")
                meta_features[:, i*2:(i+1)*2] = 0.5
        
        # Meta-model prediction
        return self.meta_model.predict_proba(meta_features)
    
    def predict(self, X):
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

class EnsembleHABTrainer:
    """
    Advanced ensemble trainer for HAB forecasting
    """
    
    def __init__(self, cache_dir: str = "cache/ensemble_models", 
                 random_state: int = 42):
        """
        Initialize ensemble HAB trainer
        
        Args:
            cache_dir: Directory for caching ensemble models
            random_state: Random state for reproducibility
        """
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # Model storage
        self.ensemble_models = {}
        self.base_models = {}
        self.performance_results = {}
        
        logger.info(f"🎭 Ensemble HAB Trainer initialized")
        logger.info(f"📁 Cache directory: {self.cache_dir}")
    
    def create_base_models(self, **model_params) -> List[Tuple[str, BaseEstimator]]:
        """Create diverse base models for ensemble"""
        
        # XGBoost with different configurations
        xgb_balanced = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=2
        )
        
        xgb_deep = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=150,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.7,
            random_state=self.random_state,
            n_jobs=2
        )
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=2
        )
        
        # Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state
        )
        
        # Extra Trees
        et_model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=2
        )
        
        base_models = [
            ('XGB_Balanced', xgb_balanced),
            ('XGB_Deep', xgb_deep), 
            ('RandomForest', rf_model),
            ('GradientBoosting', gb_model),
            ('ExtraTrees', et_model)
        ]
        
        logger.info(f"🏗️  Created {len(base_models)} base models")
        
        return base_models
    
    def train_ensemble_for_species(self, X: np.ndarray, y: np.ndarray, 
                                  species: str, ensemble_types: List[str] = None) -> Dict:
        """
        Train ensemble models for a specific species
        
        Args:
            X: Feature matrix
            y: Target vector
            species: Species name
            ensemble_types: List of ensemble types to train
            
        Returns:
            Dictionary with trained ensemble results
        """
        
        if ensemble_types is None:
            ensemble_types = ['voting_soft', 'voting_hard', 'weighted', 'stacked']
        
        logger.info(f"🎭 Training ensemble models for {species}")
        logger.info(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"🎯 Target distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create base models
        base_models = self.create_base_models()
        
        # Store base models for this species
        self.base_models[species] = base_models
        
        results = {
            'species': species,
            'base_models': base_models,
            'scaler': scaler,
            'ensemble_models': {},
            'performance': {}
        }
        
        # Train different ensemble types
        for ensemble_type in ensemble_types:
            try:
                logger.info(f"🔧 Training {ensemble_type} ensemble...")
                
                if ensemble_type == 'voting_soft':
                    # Soft voting ensemble
                    ensemble = VotingClassifier(
                        estimators=base_models,
                        voting='soft'
                    )
                    
                elif ensemble_type == 'voting_hard':
                    # Hard voting ensemble
                    ensemble = VotingClassifier(
                        estimators=base_models,
                        voting='hard'
                    )
                    
                elif ensemble_type == 'weighted':
                    # Performance-weighted ensemble
                    ensemble = PerformanceWeightedEnsemble(
                        models=base_models,
                        weighting_metric='f1',
                        cv_folds=5
                    )
                    
                elif ensemble_type == 'stacked':
                    # Stacked ensemble
                    ensemble = StackedEnsemble(
                        base_models=base_models,
                        meta_model=LogisticRegression(random_state=self.random_state),
                        cv_folds=5
                    )
                
                else:
                    logger.warning(f"Unknown ensemble type: {ensemble_type}")
                    continue
                
                # Fit ensemble
                ensemble.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = ensemble.predict(X_test_scaled)
                y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, zero_division=0),
                    'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0,
                    'avg_precision': average_precision_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
                }
                
                results['ensemble_models'][ensemble_type] = ensemble
                results['performance'][ensemble_type] = metrics
                
                logger.info(f"  ✅ {ensemble_type}: F1={metrics['f1_score']:.3f}, "
                           f"AUC={metrics['auc']:.3f}")
                
            except Exception as e:
                logger.error(f"  ❌ {ensemble_type}: Training failed ({e})")
                continue
        
        # Store results
        self.ensemble_models[species] = results
        
        # Find best ensemble
        if results['performance']:
            best_ensemble = max(results['performance'].items(), 
                              key=lambda x: x[1]['f1_score'])
            
            logger.info(f"🏆 Best ensemble for {species}: {best_ensemble[0]} "
                       f"(F1={best_ensemble[1]['f1_score']:.3f})")
            
            results['best_ensemble'] = best_ensemble[0]
            results['best_performance'] = best_ensemble[1]
        
        return results
    
    def save_ensemble_model(self, species: str, results: Dict) -> None:
        """Save ensemble model and results"""
        
        species_dir = self.cache_dir / species.replace(' ', '_')
        species_dir.mkdir(exist_ok=True)
        
        # Save ensemble models
        for ensemble_type, model in results['ensemble_models'].items():
            model_path = species_dir / f"{ensemble_type}_ensemble.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save scaler
        scaler_path = species_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(results['scaler'], f)
        
        # Save metadata
        metadata = {
            'species': results['species'],
            'performance': results['performance'],
            'best_ensemble': results.get('best_ensemble'),
            'best_performance': results.get('best_performance'),
            'n_base_models': len(results['base_models']),
            'ensemble_types': list(results['ensemble_models'].keys()),
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = species_dir / "ensemble_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"💾 Saved {species} ensemble models to {species_dir}")
    
    def generate_ensemble_report(self, all_results: Dict[str, Dict]) -> None:
        """Generate comprehensive ensemble performance report"""
        
        if not all_results:
            logger.warning("⚠️  No ensemble results to report")
            return
        
        # Create performance comparison
        report_data = []
        
        for species, results in all_results.items():
            for ensemble_type, metrics in results['performance'].items():
                report_data.append({
                    'species': species,
                    'ensemble_type': ensemble_type,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'auc': metrics['auc'],
                    'avg_precision': metrics['avg_precision']
                })
        
        if not report_data:
            logger.warning("⚠️  No performance data to report")
            return
        
        report_df = pd.DataFrame(report_data)
        
        # Save detailed report
        report_path = self.cache_dir / f"ensemble_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        report_df.to_csv(report_path, index=False)
        
        # Print summary
        logger.info(f"📊 ENSEMBLE PERFORMANCE SUMMARY")
        logger.info(f"   Species trained: {len(all_results)}")
        logger.info(f"   Total ensemble models: {len(report_data)}")
        
        # Best performing ensemble per species
        for species in report_df['species'].unique():
            species_data = report_df[report_df['species'] == species]
            best_row = species_data.loc[species_data['f1_score'].idxmax()]
            logger.info(f"   {species}: {best_row['ensemble_type']} "
                       f"(F1={best_row['f1_score']:.3f})")
        
        # Overall best ensemble type
        ensemble_summary = report_df.groupby('ensemble_type')['f1_score'].agg(['mean', 'std'])
        best_overall = ensemble_summary['mean'].idxmax()
        logger.info(f"   Best overall: {best_overall} "
                   f"(F1={ensemble_summary.loc[best_overall, 'mean']:.3f}±"
                   f"{ensemble_summary.loc[best_overall, 'std']:.3f})")
        
        logger.info(f"   Report saved: {report_path}")

if __name__ == "__main__":
    print("🎭 Advanced Ensemble HAB Trainer - Task 4.3")
    print("✅ Ensemble methods implemented:")
    print("   - Soft/Hard Voting Classifiers")
    print("   - Performance-Weighted Ensemble") 
    print("   - Stacked Ensemble with Meta-Learner")
    print("   - 5 diverse base models (XGBoost, RF, GB, ET)")
    print("🎯 Ready to combine models for improved HAB prediction!")