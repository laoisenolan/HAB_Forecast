"""
Ensemble Model Trainer: Advanced Model for Improved Accuracy

This module implements ensemble methods combining XGBoost and Random Forest
for improved HAB risk prediction accuracy.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import json
import pickle

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.model_trainer import XGBoostTrainer
from src.utils.config_loader import ConfigLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnsembleTrainer:
    """
    Ensemble model trainer combining multiple algorithms for improved accuracy
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the ensemble trainer"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_model_params_config()
        
        # Initialize base trainers
        self.xgb_trainer = XGBoostTrainer(config)
        
        # Ensemble configuration
        self.ensemble_config = self.config.get("ensemble", {
            "models": ["xgboost", "random_forest"],
            "weights": [0.6, 0.4],
            "voting_method": "weighted_average",
            "optimize_weights": True
        })
        
        # Model storage
        self.models = {}
        self.weights = []
        self.feature_importance = {}
        
        logger.info("Ensemble trainer initialized")
    
    def train_ensemble_model(self, 
                           X_train: pd.DataFrame, 
                           y_train: pd.Series,
                           X_val: pd.DataFrame = None,
                           y_val: pd.Series = None,
                           optimize_weights: bool = True) -> Dict[str, Any]:
        """
        Train ensemble model with multiple algorithms
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            optimize_weights: Whether to optimize ensemble weights
            
        Returns:
            Training results dictionary
        """
        logger.info("Training ensemble model")
        
        try:
            results = {}
            
            # Train XGBoost model
            logger.info("Training XGBoost model...")
            xgb_model = self.xgb_trainer.train_baseline_model(X_train, y_train, X_val, y_val)
            self.models["xgboost"] = xgb_model
            
            # Get XGBoost predictions
            xgb_train_pred = xgb_model.predict(X_train)
            xgb_val_pred = xgb_model.predict(X_val) if X_val is not None else None
            
            # Train Random Forest model
            logger.info("Training Random Forest model...")
            rf_model = self._train_random_forest(X_train, y_train, X_val, y_val)
            self.models["random_forest"] = rf_model
            
            # Get Random Forest predictions
            rf_train_pred = rf_model.predict(X_train)
            rf_val_pred = rf_model.predict(X_val) if X_val is not None else None
            
            # Optimize ensemble weights
            if optimize_weights and X_val is not None:
                logger.info("Optimizing ensemble weights...")
                optimal_weights = self._optimize_weights(
                    [xgb_val_pred, rf_val_pred], y_val
                )
                self.weights = optimal_weights
            else:
                # Use default weights
                self.weights = self.ensemble_config["weights"]
            
            # Create ensemble predictions
            ensemble_train_pred = self._ensemble_predict([xgb_train_pred, rf_train_pred])
            ensemble_val_pred = self._ensemble_predict([xgb_val_pred, rf_val_pred]) if X_val is not None else None
            
            # Calculate ensemble metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            train_metrics = {
                "rmse": np.sqrt(mean_squared_error(y_train, ensemble_train_pred)),
                "mae": mean_absolute_error(y_train, ensemble_train_pred),
                "r2_score": r2_score(y_train, ensemble_train_pred)
            }
            
            if X_val is not None:
                val_metrics = {
                    "rmse": np.sqrt(mean_squared_error(y_val, ensemble_val_pred)),
                    "mae": mean_absolute_error(y_val, ensemble_val_pred),
                    "r2_score": r2_score(y_val, ensemble_val_pred)
                }
            else:
                val_metrics = None
            
            # Calculate feature importance
            self._calculate_ensemble_feature_importance(X_train.columns)
            
            results = {
                "models_trained": list(self.models.keys()),
                "ensemble_weights": self.weights,
                "train_metrics": train_metrics,
                "validation_metrics": val_metrics,
                "feature_importance": self.feature_importance
            }
            
            logger.info(f"Ensemble training complete. Train R²: {train_metrics['r2_score']:.4f}")
            if val_metrics:
                logger.info(f"Validation R²: {val_metrics['r2_score']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            raise
    
    def _train_random_forest(self, 
                           X_train: pd.DataFrame, 
                           y_train: pd.Series,
                           X_val: pd.DataFrame = None,
                           y_val: pd.Series = None):
        """Train Random Forest model"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # Random Forest parameters
            rf_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1
            }
            
            # Train model
            rf_model = RandomForestRegressor(**rf_params)
            rf_model.fit(X_train, y_train)
            
            logger.info("Random Forest model trained successfully")
            return rf_model
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            raise
    
    def _optimize_weights(self, predictions_list: List[np.ndarray], y_true: pd.Series) -> List[float]:
        """Optimize ensemble weights using validation data"""
        try:
            from scipy.optimize import minimize
            from sklearn.metrics import mean_squared_error
            
            def objective(weights):
                # Normalize weights
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                
                # Calculate ensemble prediction
                ensemble_pred = np.zeros_like(predictions_list[0])
                for i, pred in enumerate(predictions_list):
                    ensemble_pred += weights[i] * pred
                
                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
                return rmse
            
            # Initial weights
            initial_weights = [0.5, 0.5]
            
            # Optimize weights
            result = minimize(
                objective,
                initial_weights,
                method='L-BFGS-B',
                bounds=[(0, 1)] * len(predictions_list)
            )
            
            # Normalize optimal weights
            optimal_weights = result.x / np.sum(result.x)
            
            logger.info(f"Optimal weights: {optimal_weights}")
            return optimal_weights.tolist()
            
        except Exception as e:
            logger.error(f"Weight optimization failed: {e}")
            return [0.6, 0.4]  # Default weights
    
    def _ensemble_predict(self, predictions_list: List[np.ndarray]) -> np.ndarray:
        """Generate ensemble predictions"""
        try:
            ensemble_pred = np.zeros_like(predictions_list[0])
            
            for i, pred in enumerate(predictions_list):
                ensemble_pred += self.weights[i] * pred
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            raise
    
    def _calculate_ensemble_feature_importance(self, feature_names: List[str]):
        """Calculate ensemble feature importance"""
        try:
            # Get feature importance from each model
            xgb_importance = self.models["xgboost"].feature_importances_
            rf_importance = self.models["random_forest"].feature_importances_
            
            # Weighted ensemble importance
            ensemble_importance = (
                self.weights[0] * xgb_importance + 
                self.weights[1] * rf_importance
            )
            
            # Create feature importance dictionary
            self.feature_importance = {
                feature_names[i]: float(ensemble_importance[i])
                for i in range(len(feature_names))
            }
            
            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), 
                      key=lambda x: x[1], reverse=True)
            )
            
            logger.info("Ensemble feature importance calculated")
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions"""
        try:
            predictions_list = []
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                pred = model.predict(X)
                predictions_list.append(pred)
            
            # Generate ensemble prediction
            ensemble_pred = self._ensemble_predict(predictions_list)
            
            # Clip to valid range
            ensemble_pred = np.clip(ensemble_pred, 0, 100)
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            raise
    
    def evaluate_ensemble(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate ensemble model performance"""
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
            
            # Generate predictions
            y_pred = self.predict(X_test)
            
            # Calculate metrics
            metrics = {
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2_score": r2_score(y_test, y_pred),
                "explained_variance": explained_variance_score(y_test, y_pred),
                "mape": np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            }
            
            # Individual model comparison
            individual_metrics = {}
            for model_name, model in self.models.items():
                individual_pred = model.predict(X_test)
                individual_pred = np.clip(individual_pred, 0, 100)
                
                individual_metrics[model_name] = {
                    "rmse": np.sqrt(mean_squared_error(y_test, individual_pred)),
                    "r2_score": r2_score(y_test, individual_pred)
                }
            
            results = {
                "ensemble_metrics": metrics,
                "individual_metrics": individual_metrics,
                "ensemble_weights": self.weights,
                "feature_importance": self.feature_importance
            }
            
            logger.info("Ensemble evaluation complete")
            return results
            
        except Exception as e:
            logger.error(f"Ensemble evaluation failed: {e}")
            raise
    
    def save_ensemble_model(self, filepath: str):
        """Save ensemble model to file"""
        try:
            model_data = {
                "models": self.models,
                "weights": self.weights,
                "feature_importance": self.feature_importance,
                "ensemble_config": self.ensemble_config,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Ensemble model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
            raise
    
    def load_ensemble_model(self, filepath: str):
        """Load ensemble model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data["models"]
            self.weights = model_data["weights"]
            self.feature_importance = model_data["feature_importance"]
            
            logger.info(f"Ensemble model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise 