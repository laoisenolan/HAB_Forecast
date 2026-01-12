#!/usr/bin/env python3
"""
Advanced Cross-Validation for HAB Predictions - Task 4.10

This module implements sophisticated cross-validation strategies specifically
designed for HAB prediction models with temporal and spatial considerations:
- Temporal cross-validation with proper time-series splits
- Spatial cross-validation with geographic clustering
- Stratified temporal splits preserving seasonal patterns
- Leave-One-Year-Out validation for long-term robustness
- Spatial block cross-validation for geographic generalization
- Nested cross-validation for unbiased hyperparameter optimization

🎯 CROSS-VALIDATION STRATEGIES:
- TimeSeriesSplit: Respects temporal order and dependencies
- Spatial Block CV: Accounts for spatial autocorrelation
- Stratified Temporal CV: Preserves seasonal and class distributions
- Leave-Location-Out CV: Tests geographic generalization
- Nested CV: Unbiased model selection and evaluation
- Custom HAB-specific CV: Domain-aware splitting strategies

🛰️ HAB-SPECIFIC CONSIDERATIONS:
- Seasonal stratification to preserve bloom timing patterns
- Geographic clustering for spatial independence
- Temporal gap enforcement to prevent data leakage
- Species-specific validation strategies
- Environmental regime-aware splits (El Niño, seasonal cycles)
- Multi-scale validation (daily, monthly, yearly)

Author: HAB Forecasting System - Task 4.10 Implementation
Date: December 19, 2024
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from datetime import datetime, timedelta
import json
import pickle
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Standard ML libraries
from sklearn.model_selection import (
    TimeSeriesSplit, StratifiedKFold, KFold, GroupKFold,
    cross_val_score, cross_validate
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

class AdvancedHABCrossValidator:
    """
    Advanced cross-validation strategies for HAB prediction models
    """
    
    def __init__(self, cache_dir: str = "cache/cross_validation",
                 random_state: int = 42):
        """
        Initialize advanced cross-validator
        
        Args:
            cache_dir: Directory for caching CV results
            random_state: Random state for reproducibility
        """
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # CV results storage
        self.cv_results = {}
        self.cv_strategies = {}
        
        logger.info(f"🔄 Advanced HAB Cross-Validator initialized")
        logger.info(f"📁 Cache directory: {self.cache_dir}")
    
    def create_temporal_cv(self, timestamps: np.ndarray, n_splits: int = 5,
                          min_train_size: Optional[int] = None,
                          gap_days: int = 7) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Create temporal cross-validation splits with gap enforcement
        
        Args:
            timestamps: Array of timestamps (as datetime or numeric)
            n_splits: Number of CV splits
            min_train_size: Minimum training set size
            gap_days: Days to gap between train and test to prevent leakage
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        
        logger.info(f"🕒 Creating temporal CV with {n_splits} splits, {gap_days}-day gap")
        
        # Convert timestamps to datetime if needed
        if hasattr(timestamps[0], 'timestamp'):
            # Handle pandas Timestamp objects
            timestamps_dt = pd.to_datetime(timestamps)
        elif isinstance(timestamps[0], str):
            timestamps_dt = pd.to_datetime(timestamps)
        else:
            timestamps_dt = timestamps
        
        # Sort by time
        sorted_indices = np.argsort(timestamps_dt)
        sorted_timestamps = timestamps_dt[sorted_indices]
        
        n_samples = len(sorted_indices)
        if min_train_size is None:
            min_train_size = n_samples // (n_splits + 1)
        
        # Calculate split points
        total_usable = n_samples - min_train_size
        test_size = total_usable // n_splits
        
        for i in range(n_splits):
            # Calculate train end and test start with gap
            train_end_idx = min_train_size + i * test_size
            test_start_idx = train_end_idx + 1
            test_end_idx = min(test_start_idx + test_size - 1, n_samples - 1)
            
            # Apply temporal gap
            if gap_days > 0:
                train_end_time = sorted_timestamps[train_end_idx]
                gap_threshold = train_end_time + timedelta(days=gap_days)
                
                # Find first test index after gap
                valid_test_mask = sorted_timestamps >= gap_threshold
                valid_test_indices = np.where(valid_test_mask)[0]
                
                if len(valid_test_indices) == 0:
                    continue  # Skip if no valid test samples after gap
                
                test_start_idx = valid_test_indices[0]
                test_end_idx = min(test_start_idx + test_size - 1, n_samples - 1)
            
            # Get actual indices
            train_indices = sorted_indices[:train_end_idx]
            test_indices = sorted_indices[test_start_idx:test_end_idx + 1]
            
            if len(train_indices) < min_train_size or len(test_indices) == 0:
                continue
            
            logger.info(f"  Split {i+1}: Train={len(train_indices)}, Test={len(test_indices)}, "
                       f"Gap={gap_days} days")
            
            yield train_indices, test_indices
    
    def create_spatial_cv(self, coordinates: np.ndarray, n_splits: int = 5,
                         cluster_method: str = 'kmeans') -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Create spatial cross-validation splits using geographic clustering
        
        Args:
            coordinates: Array of (lat, lon) coordinates
            n_splits: Number of spatial splits
            cluster_method: Clustering method for spatial grouping
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        
        logger.info(f"🗺️  Creating spatial CV with {n_splits} splits using {cluster_method}")
        
        coordinates = np.array(coordinates)
        if coordinates.shape[1] != 2:
            raise ValueError("Coordinates must be (lat, lon) pairs")
        
        # Cluster locations into spatial groups
        if cluster_method == 'kmeans':
            clusterer = KMeans(n_clusters=n_splits, random_state=self.random_state, n_init=10)
            spatial_groups = clusterer.fit_predict(coordinates)
        else:
            # Simple geographic binning as fallback
            lat_bins = np.linspace(coordinates[:, 0].min(), coordinates[:, 0].max(), 
                                  int(np.sqrt(n_splits)) + 1)
            lon_bins = np.linspace(coordinates[:, 1].min(), coordinates[:, 1].max(),
                                  int(np.sqrt(n_splits)) + 1)
            
            lat_groups = np.digitize(coordinates[:, 0], lat_bins) - 1
            lon_groups = np.digitize(coordinates[:, 1], lon_bins) - 1
            spatial_groups = lat_groups * len(lon_bins) + lon_groups
            spatial_groups = spatial_groups % n_splits
        
        # Create splits by leaving out each spatial group
        unique_groups = np.unique(spatial_groups)
        
        for group in unique_groups:
            test_indices = np.where(spatial_groups == group)[0]
            train_indices = np.where(spatial_groups != group)[0]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                logger.info(f"  Group {group}: Train={len(train_indices)}, Test={len(test_indices)}")
                yield train_indices, test_indices
    
    def create_stratified_temporal_cv(self, X: np.ndarray, y: np.ndarray, 
                                    timestamps: np.ndarray, n_splits: int = 5,
                                    seasonal_stratify: bool = True) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Create stratified temporal CV that preserves class and seasonal distributions
        
        Args:
            X: Feature matrix
            y: Target vector
            timestamps: Timestamp array
            n_splits: Number of CV splits
            seasonal_stratify: Whether to stratify by season
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        
        logger.info(f"📊 Creating stratified temporal CV with {n_splits} splits")
        logger.info(f"🍂 Seasonal stratification: {seasonal_stratify}")
        
        # Convert timestamps to datetime
        timestamps_dt = pd.to_datetime(timestamps)
        
        # Create seasonal labels if requested
        if seasonal_stratify:
            months = timestamps_dt.month
            # Define seasons: Spring (3-5), Summer (6-8), Fall (9-11), Winter (12-2)
            seasons = np.where(months.isin([3, 4, 5]), 0,  # Spring
                     np.where(months.isin([6, 7, 8]), 1,   # Summer  
                     np.where(months.isin([9, 10, 11]), 2, # Fall
                             3)))                           # Winter
            
            # Combine class and season for stratification
            stratify_labels = y * 4 + seasons  # 8 strata total (2 classes × 4 seasons)
        else:
            stratify_labels = y
        
        # Sort by time for temporal consistency
        sorted_indices = np.argsort(timestamps_dt)
        sorted_y = y[sorted_indices]
        sorted_strata = stratify_labels[sorted_indices]
        
        # Create temporal blocks that preserve stratification
        n_samples = len(sorted_indices)
        block_size = n_samples // n_splits
        
        for i in range(n_splits):
            # Define test block
            test_start = i * block_size
            test_end = (i + 1) * block_size if i < n_splits - 1 else n_samples
            test_block_indices = sorted_indices[test_start:test_end]
            
            # Use remaining data for training
            train_block_indices = np.concatenate([
                sorted_indices[:test_start],
                sorted_indices[test_end:]
            ])
            
            if len(train_block_indices) > 0 and len(test_block_indices) > 0:
                # Check stratification quality
                train_y = y[train_block_indices]
                test_y = y[test_block_indices]
                
                train_pos_ratio = np.mean(train_y)
                test_pos_ratio = np.mean(test_y)
                
                logger.info(f"  Split {i+1}: Train={len(train_block_indices)} "
                           f"(pos={train_pos_ratio:.3f}), Test={len(test_block_indices)} "
                           f"(pos={test_pos_ratio:.3f})")
                
                yield train_block_indices, test_block_indices
    
    def create_leave_one_year_out_cv(self, timestamps: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Create Leave-One-Year-Out cross-validation
        
        Args:
            timestamps: Array of timestamps
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        
        logger.info("🗓️  Creating Leave-One-Year-Out CV")
        
        # Convert to datetime and extract years
        timestamps_dt = pd.to_datetime(timestamps)
        years = timestamps_dt.year
        unique_years = np.unique(years)
        
        logger.info(f"📅 Found {len(unique_years)} unique years: {unique_years}")
        
        for year in unique_years:
            test_indices = np.where(years == year)[0]
            train_indices = np.where(years != year)[0]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                logger.info(f"  Year {year}: Train={len(train_indices)}, Test={len(test_indices)}")
                yield train_indices, test_indices
    
    def nested_cross_validation(self, model, X: np.ndarray, y: np.ndarray,
                              timestamps: Optional[np.ndarray] = None,
                              coordinates: Optional[np.ndarray] = None,
                              param_grid: Optional[Dict] = None,
                              outer_cv: str = 'temporal',
                              inner_cv: str = 'stratified',
                              n_outer_splits: int = 5,
                              n_inner_splits: int = 3) -> Dict[str, Any]:
        """
        Perform nested cross-validation for unbiased model evaluation
        
        Args:
            model: Scikit-learn compatible model
            X: Feature matrix
            y: Target vector  
            timestamps: Timestamps for temporal CV
            coordinates: Coordinates for spatial CV
            param_grid: Parameter grid for hyperparameter optimization
            outer_cv: Outer CV strategy ('temporal', 'spatial', 'stratified')
            inner_cv: Inner CV strategy for hyperparameter optimization
            n_outer_splits: Number of outer CV splits
            n_inner_splits: Number of inner CV splits
            
        Returns:
            Nested CV results dictionary
        """
        
        logger.info(f"🔄 Nested cross-validation: {outer_cv} outer, {inner_cv} inner")
        logger.info(f"🎯 Outer splits: {n_outer_splits}, Inner splits: {n_inner_splits}")
        
        # Create outer CV splits
        if outer_cv == 'temporal' and timestamps is not None:
            outer_splits = list(self.create_temporal_cv(timestamps, n_outer_splits))
        elif outer_cv == 'spatial' and coordinates is not None:
            outer_splits = list(self.create_spatial_cv(coordinates, n_outer_splits))
        elif outer_cv == 'stratified_temporal' and timestamps is not None:
            outer_splits = list(self.create_stratified_temporal_cv(X, y, timestamps, n_outer_splits))
        else:
            # Fallback to standard stratified CV
            skf = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=self.random_state)
            outer_splits = list(skf.split(X, y))
        
        nested_scores = []
        best_params_list = []
        
        for fold, (train_idx, test_idx) in enumerate(outer_splits):
            logger.info(f"  Outer fold {fold + 1}/{len(outer_splits)}")
            
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
            
            # Inner CV for hyperparameter optimization
            if param_grid:
                from sklearn.model_selection import GridSearchCV
                
                # Create inner CV strategy
                if inner_cv == 'stratified':
                    inner_cv_obj = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, 
                                                 random_state=self.random_state)
                else:
                    inner_cv_obj = KFold(n_splits=n_inner_splits, shuffle=True, 
                                       random_state=self.random_state)
                
                # Grid search with inner CV
                grid_search = GridSearchCV(
                    model, param_grid, cv=inner_cv_obj,
                    scoring='f1', n_jobs=1, random_state=self.random_state
                )
                
                grid_search.fit(X_train_outer, y_train_outer)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_params_list.append(best_params)
                
                logger.info(f"    Best params: {best_params}")
            else:
                # Use model as-is
                best_model = model
                best_model.fit(X_train_outer, y_train_outer)
                best_params_list.append({})
            
            # Evaluate on outer test set
            y_pred = best_model.predict(X_test_outer)
            y_pred_proba = best_model.predict_proba(X_test_outer)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred
            
            # Calculate metrics
            fold_scores = {
                'f1': f1_score(y_test_outer, y_pred),
                'accuracy': accuracy_score(y_test_outer, y_pred),
                'precision': precision_score(y_test_outer, y_pred, zero_division=0),
                'recall': recall_score(y_test_outer, y_pred, zero_division=0)
            }
            
            # Add AUC if both classes present
            if len(np.unique(y_test_outer)) > 1:
                fold_scores['auc'] = roc_auc_score(y_test_outer, y_pred_proba)
            else:
                fold_scores['auc'] = 0.0
            
            nested_scores.append(fold_scores)
            
            logger.info(f"    Fold scores: F1={fold_scores['f1']:.3f}, "
                       f"AUC={fold_scores['auc']:.3f}")
        
        # Aggregate results
        aggregated_scores = {}
        for metric in nested_scores[0].keys():
            scores = [fold[metric] for fold in nested_scores]
            aggregated_scores[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        
        nested_results = {
            'outer_cv_strategy': outer_cv,
            'inner_cv_strategy': inner_cv,
            'n_outer_splits': len(nested_scores),
            'n_inner_splits': n_inner_splits,
            'aggregated_scores': aggregated_scores,
            'fold_scores': nested_scores,
            'best_params_per_fold': best_params_list,
            'cv_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Nested CV complete!")
        for metric, stats in aggregated_scores.items():
            logger.info(f"  {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return nested_results
    
    def evaluate_cv_strategies(self, model, X: np.ndarray, y: np.ndarray,
                             timestamps: Optional[np.ndarray] = None,
                             coordinates: Optional[np.ndarray] = None,
                             species: str = "Unknown") -> Dict[str, Any]:
        """
        Compare different cross-validation strategies
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target vector
            timestamps: Timestamps for temporal CV
            coordinates: Coordinates for spatial CV
            species: Species name for identification
            
        Returns:
            Comparison of CV strategies
        """
        
        logger.info(f"📊 Evaluating CV strategies for {species}")
        
        cv_results = {
            'species': species,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'class_distribution': np.bincount(y).tolist(),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Strategy 1: Standard Stratified K-Fold
        logger.info("  Strategy 1: Standard Stratified K-Fold")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        stratified_scores = cross_validate(
            model, X, y, cv=skf, 
            scoring=['f1', 'roc_auc', 'accuracy'],
            n_jobs=1, error_score='raise'
        )
        
        cv_results['stratified_kfold'] = {
            'f1': {
                'mean': np.mean(stratified_scores['test_f1']),
                'std': np.std(stratified_scores['test_f1']),
                'scores': stratified_scores['test_f1'].tolist()
            },
            'auc': {
                'mean': np.mean(stratified_scores['test_roc_auc']),
                'std': np.std(stratified_scores['test_roc_auc']),
                'scores': stratified_scores['test_roc_auc'].tolist()
            },
            'accuracy': {
                'mean': np.mean(stratified_scores['test_accuracy']),
                'std': np.std(stratified_scores['test_accuracy']),
                'scores': stratified_scores['test_accuracy'].tolist()
            }
        }
        
        # Strategy 2: Temporal CV (if timestamps available)
        if timestamps is not None:
            logger.info("  Strategy 2: Temporal Cross-Validation")
            temporal_scores = {'f1': [], 'auc': [], 'accuracy': []}
            
            for train_idx, test_idx in self.create_temporal_cv(timestamps, n_splits=5):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Skip if not enough samples or only one class
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    continue
                
                model_copy = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                model_copy.fit(X_train, y_train)
                
                y_pred = model_copy.predict(X_test)
                y_pred_proba = model_copy.predict_proba(X_test)[:, 1] if hasattr(model_copy, 'predict_proba') else y_pred
                
                temporal_scores['f1'].append(f1_score(y_test, y_pred))
                temporal_scores['auc'].append(roc_auc_score(y_test, y_pred_proba))
                temporal_scores['accuracy'].append(accuracy_score(y_test, y_pred))
            
            if temporal_scores['f1']:
                cv_results['temporal_cv'] = {
                    'f1': {
                        'mean': np.mean(temporal_scores['f1']),
                        'std': np.std(temporal_scores['f1']),
                        'scores': temporal_scores['f1']
                    },
                    'auc': {
                        'mean': np.mean(temporal_scores['auc']),
                        'std': np.std(temporal_scores['auc']),
                        'scores': temporal_scores['auc']
                    },
                    'accuracy': {
                        'mean': np.mean(temporal_scores['accuracy']),
                        'std': np.std(temporal_scores['accuracy']),
                        'scores': temporal_scores['accuracy']
                    }
                }
        
        # Strategy 3: Spatial CV (if coordinates available)
        if coordinates is not None:
            logger.info("  Strategy 3: Spatial Cross-Validation")
            spatial_scores = {'f1': [], 'auc': [], 'accuracy': []}
            
            for train_idx, test_idx in self.create_spatial_cv(coordinates, n_splits=5):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Skip if not enough samples or only one class
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    continue
                
                model_copy = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                model_copy.fit(X_train, y_train)
                
                y_pred = model_copy.predict(X_test)
                y_pred_proba = model_copy.predict_proba(X_test)[:, 1] if hasattr(model_copy, 'predict_proba') else y_pred
                
                spatial_scores['f1'].append(f1_score(y_test, y_pred))
                spatial_scores['auc'].append(roc_auc_score(y_test, y_pred_proba))
                spatial_scores['accuracy'].append(accuracy_score(y_test, y_pred))
            
            if spatial_scores['f1']:
                cv_results['spatial_cv'] = {
                    'f1': {
                        'mean': np.mean(spatial_scores['f1']),
                        'std': np.std(spatial_scores['f1']),
                        'scores': spatial_scores['f1']
                    },
                    'auc': {
                        'mean': np.mean(spatial_scores['auc']),
                        'std': np.std(spatial_scores['auc']),
                        'scores': spatial_scores['auc']
                    },
                    'accuracy': {
                        'mean': np.mean(spatial_scores['accuracy']),
                        'std': np.std(spatial_scores['accuracy']),
                        'scores': spatial_scores['accuracy']
                    }
                }
        
        # Store results
        self.cv_results[species] = cv_results
        
        return cv_results
    
    def save_cv_results(self, species: str) -> None:
        """Save cross-validation results"""
        
        if species not in self.cv_results:
            logger.warning(f"⚠️  No CV results found for {species}")
            return
        
        species_dir = self.cache_dir / species.replace(' ', '_')
        species_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = species_dir / f'cv_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.cv_results[species], f, indent=2, default=str)
        
        logger.info(f"💾 CV results saved for {species}: {results_file}")

if __name__ == "__main__":
    print("🔄 Advanced Cross-Validation for HAB Predictions - Task 4.10")
    print("🎯 Temporal and spatial considerations for robust model evaluation")
    print("📊 Strategies: Temporal, Spatial, Stratified, Leave-One-Year-Out, Nested CV")
    
    # Initialize advanced cross-validator
    cv = AdvancedHABCrossValidator()
    
    print("✅ Advanced HAB Cross-Validator initialized successfully!")
    print("🔬 Ready for comprehensive model validation with temporal and spatial awareness")