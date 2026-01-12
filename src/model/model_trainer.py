"""
XGBoost Model Trainer for HAB Forecasting System

This module handles XGBoost model training, hyperparameter optimization,
and model evaluation for the HAB forecasting system.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import warnings
import pickle
import json

# ML libraries
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Using grid search for hyperparameter optimization.")

from src.utils.config_loader import ConfigLoader, get_data_dir

logger = logging.getLogger(__name__)

class XGBoostTrainer:
    """XGBoost model trainer for HAB forecasting"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the XGBoost trainer"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_model_params_config()
        self.data_dir = get_data_dir()
        
        # Get configuration sections
        self.xgboost_params = self.config.get("xgboost_params", {})
        self.training_config = self.config.get("training", {})
        self.feature_selection_config = self.config.get("feature_selection", {})
        self.optimization_config = self.config.get("hyperparameter_optimization", {})
        self.evaluation_config = self.config.get("evaluation", {})
        
        # Initialize components
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.selected_features = None
        
        # Training results
        self.training_results = {}
        self.evaluation_results = {}
        self.optimization_results = {}
        
        # Output directories
        self.model_dir = self.data_dir / "models"
        self.results_dir = self.data_dir / "results"
        self.plots_dir = self.data_dir / "plots"
        
        for directory in [self.model_dir, self.results_dir, self.plots_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
    def prepare_dataset(self, dataset: xr.Dataset, target_variable: str = "hab_labels") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare dataset for training
        
        Args:
            dataset: Input dataset with features and labels
            target_variable: Name of the target variable
            
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info("Preparing dataset for training")
        
        try:
            # Separate features and target
            feature_vars = [var for var in dataset.data_vars if var != target_variable]
            target_data = dataset[target_variable]
            
            # Convert to DataFrame
            features_df = dataset[feature_vars].to_dataframe().reset_index()
            target_series = target_data.to_dataframe().reset_index()[target_variable]
            
            # Handle missing values
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            target_series = target_series.fillna(0)
            
            # Remove time-based columns if present
            time_columns = ['time', 'latitude', 'longitude']
            features_df = features_df.drop(columns=[col for col in time_columns if col in features_df.columns])
            
            # Store feature names
            self.feature_names = features_df.columns.tolist()
            
            logger.info(f"Prepared dataset: {features_df.shape[0]} samples, {features_df.shape[1]} features")
            return features_df, target_series
            
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            raise
            
    def create_temporal_split(self, features_df: pd.DataFrame, target_series: pd.Series) -> Tuple:
        """
        Create temporal train/validation/test split
        
        Args:
            features_df: Feature DataFrame
            target_series: Target series
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Creating temporal train/validation/test split")
        
        try:
            # Get split ratios
            train_ratio = self.training_config.get("train_ratio", 0.7)
            val_ratio = self.training_config.get("validation_ratio", 0.15)
            
            # Calculate split indices
            n_samples = len(features_df)
            train_end = int(n_samples * train_ratio)
            val_end = int(n_samples * (train_ratio + val_ratio))
            
            # Split data
            X_train = features_df.iloc[:train_end]
            X_val = features_df.iloc[train_end:val_end]
            X_test = features_df.iloc[val_end:]
            
            y_train = target_series.iloc[:train_end]
            y_val = target_series.iloc[train_end:val_end]
            y_test = target_series.iloc[val_end:]
            
            logger.info(f"Temporal split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Temporal split failed: {e}")
            raise
            
    def handle_risk_percentage(self, y_train: pd.Series) -> Dict[str, float]:
        """
        Handle risk percentage data characteristics
        
        Args:
            y_train: Training target series (risk percentages)
            
        Returns:
            Risk statistics dictionary
        """
        logger.info("Analyzing risk percentage characteristics")
        
        try:
            # Calculate risk statistics
            risk_stats = {
                "mean_risk": float(y_train.mean()),
                "std_risk": float(y_train.std()),
                "min_risk": float(y_train.min()),
                "max_risk": float(y_train.max()),
                "median_risk": float(y_train.median())
            }
            
            # Check risk distribution
            low_risk = (y_train <= 25).sum()
            moderate_risk = ((y_train > 25) & (y_train <= 50)).sum()
            high_risk = ((y_train > 50) & (y_train <= 75)).sum()
            extreme_risk = (y_train > 75).sum()
            
            risk_stats.update({
                "low_risk_count": int(low_risk),
                "moderate_risk_count": int(moderate_risk),
                "high_risk_count": int(high_risk),
                "extreme_risk_count": int(extreme_risk)
            })
            
            logger.info(f"Risk distribution: Low={low_risk}, Moderate={moderate_risk}, High={high_risk}, Extreme={extreme_risk}")
            logger.info(f"Mean risk: {risk_stats['mean_risk']:.2f}%, Std: {risk_stats['std_risk']:.2f}%")
            
            return risk_stats
            
        except Exception as e:
            logger.error(f"Risk percentage analysis failed: {e}")
            return {"mean_risk": 50.0, "std_risk": 25.0}
            
    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame = None, 
                      X_test: pd.DataFrame = None) -> Tuple:
        """
        Scale features using specified method
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            
        Returns:
            Tuple of scaled features
        """
        logger.info("Scaling features")
        
        try:
            # Get scaling configuration
            scale_features = self.training_config.get("scale_features", True)
            scaling_method = self.training_config.get("scaling_method", "standard_scaler")
            exclude_vars = self.training_config.get("exclude_from_scaling", [])
            
            if not scale_features:
                return X_train, X_val, X_test
                
            # Select features to scale
            scale_columns = [col for col in X_train.columns if col not in exclude_vars]
            
            # Initialize scaler
            if scaling_method == "standard_scaler":
                self.scaler = StandardScaler()
            elif scaling_method == "minmax_scaler":
                self.scaler = MinMaxScaler()
            elif scaling_method == "robust_scaler":
                self.scaler = RobustScaler()
            else:
                logger.warning(f"Unknown scaling method: {scaling_method}, using standard scaler")
                self.scaler = StandardScaler()
                
            # Fit scaler on training data
            X_train_scaled = X_train.copy()
            X_train_scaled[scale_columns] = self.scaler.fit_transform(X_train[scale_columns])
            
            # Transform validation and test data
            X_val_scaled = None
            X_test_scaled = None
            
            if X_val is not None:
                X_val_scaled = X_val.copy()
                X_val_scaled[scale_columns] = self.scaler.transform(X_val[scale_columns])
                
            if X_test is not None:
                X_test_scaled = X_test.copy()
                X_test_scaled[scale_columns] = self.scaler.transform(X_test[scale_columns])
                
            logger.info(f"Scaled {len(scale_columns)} features using {scaling_method}")
            return X_train_scaled, X_val_scaled, X_test_scaled
            
        except Exception as e:
            logger.error(f"Feature scaling failed: {e}")
            return X_train, X_val, X_test
            
    def select_features(self, X_train: pd.DataFrame, y_train: pd.Series, 
                       X_val: pd.DataFrame = None, X_test: pd.DataFrame = None) -> Tuple:
        """
        Select features using specified methods
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            X_test: Test features
            
        Returns:
            Tuple of selected features
        """
        logger.info("Selecting features")
        
        try:
            methods = self.feature_selection_config.get("methods", ["correlation"])
            top_n = self.feature_selection_config.get("top_n_features", 50)
            
            selected_features = set(X_train.columns)
            
            # Apply feature selection methods
            for method in methods:
                if method == "correlation":
                    selected_features = self._correlation_selection(X_train, selected_features)
                elif method == "variance":
                    selected_features = self._variance_selection(X_train, selected_features)
                elif method == "importance":
                    selected_features = self._importance_selection(X_train, y_train, selected_features)
                    
            # Limit to top N features
            if len(selected_features) > top_n:
                # Use importance-based selection to get top N
                importance_scores = self._compute_feature_importance(X_train[list(selected_features)], y_train)
                top_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
                selected_features = [feature for feature, _ in top_features]
            else:
                selected_features = list(selected_features)
                
            self.selected_features = selected_features
            
            # Apply feature selection
            X_train_selected = X_train[selected_features]
            X_val_selected = X_val[selected_features] if X_val is not None else None
            X_test_selected = X_test[selected_features] if X_test is not None else None
            
            logger.info(f"Selected {len(selected_features)} features")
            return X_train_selected, X_val_selected, X_test_selected
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return X_train, X_val, X_test
            
    def _correlation_selection(self, X_train: pd.DataFrame, current_features: set) -> set:
        """Select features based on correlation"""
        try:
            threshold = self.feature_selection_config.get("correlation_threshold", 0.95)
            corr_matrix = X_train[list(current_features)].corr().abs()
            
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
            
            return current_features - set(to_drop)
        except Exception as e:
            logger.warning(f"Correlation selection failed: {e}")
            return current_features
            
    def _variance_selection(self, X_train: pd.DataFrame, current_features: set) -> set:
        """Select features based on variance"""
        try:
            threshold = self.feature_selection_config.get("variance_threshold", 0.01)
            variances = X_train[list(current_features)].var()
            high_variance = variances[variances > threshold].index
            return current_features & set(high_variance)
        except Exception as e:
            logger.warning(f"Variance selection failed: {e}")
            return current_features
            
    def _importance_selection(self, X_train: pd.DataFrame, y_train: pd.Series, current_features: set) -> set:
        """Select features based on importance"""
        try:
            threshold = self.feature_selection_config.get("importance_threshold", 0.01)
            importance_scores = self._compute_feature_importance(X_train[list(current_features)], y_train)
            important_features = [feature for feature, score in importance_scores.items() if score > threshold]
            return current_features & set(important_features)
        except Exception as e:
            logger.warning(f"Importance selection failed: {e}")
            return current_features
            
    def _compute_feature_importance(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Compute feature importance using correlation"""
        try:
            importance_scores = {}
            for feature in X_train.columns:
                correlation = abs(X_train[feature].corr(y_train))
                importance_scores[feature] = correlation
            return importance_scores
        except Exception as e:
            logger.warning(f"Feature importance computation failed: {e}")
            return {feature: 1.0 for feature in X_train.columns}
            
    def train_baseline_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame = None, y_val: pd.Series = None) -> xgb.XGBRegressor:
        """
        Train baseline XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Trained XGBoost model
        """
        logger.info("Training baseline XGBoost model")
        
        try:
            # Get XGBoost parameters
            params = self.xgboost_params.copy()
            
            # Handle risk percentage characteristics
            if self.training_config.get("risk_calibration", True):
                risk_stats = self.handle_risk_percentage(y_train)
                # Set base score to mean risk for better initialization
                params["base_score"] = risk_stats["mean_risk"] / 100.0
                
            # Initialize model (use regressor for risk percentage)
            self.model = xgb.XGBRegressor(**params)
            
            # Prepare evaluation set
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
                
            # Train model
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            logger.info("Baseline model training complete")
            return self.model
            
        except Exception as e:
            logger.error(f"Baseline model training failed: {e}")
            raise
            
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Best hyperparameters
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, skipping hyperparameter optimization")
            return self.xgboost_params
            
        logger.info("Optimizing hyperparameters using Optuna")
        
        try:
            # Get optimization configuration
            n_trials = self.optimization_config.get("n_trials", 100)
            cv_folds = self.optimization_config.get("cv_folds", 5)
            scoring = self.optimization_config.get("scoring", "f1_score")
            
            def objective(trial):
                # Define hyperparameter search space
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'random_state': 42
                }
                
                # Handle risk percentage characteristics
                if self.training_config.get("risk_calibration", True):
                    risk_stats = self.handle_risk_percentage(y_train)
                    # Set base score to mean risk for better initialization
                    params["base_score"] = risk_stats["mean_risk"] / 100.0
                    
                # Cross-validation
                cv = TimeSeriesSplit(n_splits=cv_folds)
                scores = []
                
                for train_idx, val_idx in cv.split(X_train):
                    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    model = xgb.XGBRegressor(**params)
                    model.fit(X_cv_train, y_cv_train, verbose=False)
                    
                    y_pred = model.predict(X_cv_val)
                    if scoring == "rmse":
                        score = -np.sqrt(mean_squared_error(y_cv_val, y_pred))  # Negative for maximization
                    elif scoring == "r2_score":
                        score = r2_score(y_cv_val, y_pred)
                    else:
                        score = -np.sqrt(mean_squared_error(y_cv_val, y_pred))
                        
                    scores.append(score)
                    
                return np.mean(scores)
                
            # Run optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            best_params = study.best_params
            best_params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': 42
            })
            
            # Handle class imbalance
            if self.training_config.get("handle_imbalance", True):
                scale_pos_weight = self.handle_class_imbalance(y_train)
                best_params["scale_pos_weight"] = scale_pos_weight
                
            self.optimization_results = {
                "best_params": best_params,
                "best_score": study.best_value,
                "n_trials": n_trials
            }
            
            logger.info(f"Hyperparameter optimization complete. Best score: {study.best_value:.4f}")
            return best_params
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            return self.xgboost_params
            
    def train_optimized_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame = None, y_val: pd.Series = None) -> xgb.XGBRegressor:
        """
        Train model with optimized hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Trained XGBoost model
        """
        logger.info("Training optimized XGBoost model")
        
        try:
            # Get optimized parameters
            if self.optimization_results:
                params = self.optimization_results["best_params"]
            else:
                params = self.xgboost_params.copy()
                
            # Initialize model (use regressor for risk percentage)
            self.model = xgb.XGBRegressor(**params)
            
            # Prepare evaluation set
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
                
            # Train model
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            logger.info("Optimized model training complete")
            return self.model
            
        except Exception as e:
            logger.error(f"Optimized model training failed: {e}")
            raise
            
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance for risk percentage prediction
        
        Args:
            X_test: Test features
            y_test: Test target (risk percentages)
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating risk percentage model performance")
        
        try:
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Ensure predictions are within valid range (0-100%)
            y_pred = np.clip(y_pred, 0, 100)
            
            # Calculate regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
            
            metrics = {
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
                "mape": np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-8))) * 100,
                "r2_score": r2_score(y_test, y_pred),
                "explained_variance": explained_variance_score(y_test, y_pred)
            }
            
            # Risk threshold analysis
            risk_thresholds = self.evaluation_config.get("risk_thresholds", {
                "low_risk": 25.0,
                "moderate_risk": 50.0,
                "high_risk": 75.0,
                "extreme_risk": 90.0
            })
            
            threshold_metrics = {}
            for threshold_name, threshold_value in risk_thresholds.items():
                # Calculate accuracy at each risk threshold
                y_test_binary = y_test > threshold_value
                y_pred_binary = y_pred > threshold_value
                
                threshold_metrics[f"{threshold_name}_accuracy"] = accuracy_score(y_test_binary, y_pred_binary)
                threshold_metrics[f"{threshold_name}_precision"] = precision_score(y_test_binary, y_pred_binary, zero_division=0)
                threshold_metrics[f"{threshold_name}_recall"] = recall_score(y_test_binary, y_pred_binary, zero_division=0)
            
            metrics.update(threshold_metrics)
            
            # Risk distribution analysis
            risk_stats = {
                "predicted_mean_risk": float(y_pred.mean()),
                "actual_mean_risk": float(y_test.mean()),
                "predicted_std_risk": float(y_pred.std()),
                "actual_std_risk": float(y_test.std())
            }
            
            metrics.update(risk_stats)
            
            self.evaluation_results = metrics
            
            logger.info(f"Risk model evaluation complete. RMSE: {metrics['rmse']:.2f}%, R²: {metrics['r2_score']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Risk model evaluation failed: {e}")
            raise
            
    def save_model(self, filename: str = None) -> Path:
        """Save trained model and metadata"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hab_forecast_model_{timestamp}.pkl"
            
        model_path = self.model_dir / filename
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        # Save metadata
        metadata = {
            "model_type": "XGBoost",
            "training_date": datetime.now().isoformat(),
            "feature_names": self.feature_names,
            "selected_features": self.selected_features,
            "xgboost_params": self.xgboost_params,
            "training_results": self.training_results,
            "evaluation_results": self.evaluation_results,
            "optimization_results": self.optimization_results
        }
        
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Model saved: {model_path}")
        return model_path
        
    def load_model(self, model_path: Path) -> xgb.XGBRegressor:
        """Load trained model and metadata"""
        try:
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
                
            # Load metadata
            metadata_path = model_path.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get("feature_names")
                    self.selected_features = metadata.get("selected_features")
                    
            logger.info(f"Model loaded: {model_path}")
            return self.model
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise 