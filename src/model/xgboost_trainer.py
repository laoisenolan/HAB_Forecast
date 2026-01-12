"""
XGBoost Trainer for HAB Forecasting System

This module handles XGBoost model training and optimization.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Optional, Tuple
import logging
import joblib

logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """
    Trains XGBoost models for HAB forecasting.
    
    This class handles the training, hyperparameter optimization,
    and evaluation of XGBoost models for HAB prediction.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the XGBoostTrainer.
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        self.default_params = config.get('default_params', {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': 42,
            'n_jobs': -1
        })
        self.hyperparameter_grid = config.get('hyperparameter_grid', {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        })
        
    def prepare_data(self, data: pd.DataFrame,
                    target_column: str = 'hab_occurrence',
                    feature_columns: Optional[List[str]] = None,
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            data: DataFrame with features and target
            target_column: Name of the target column
            feature_columns: List of feature column names
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            if feature_columns is None:
                feature_columns = [col for col in data.columns if col != target_column]
            
            X = data[feature_columns]
            y = data[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            logger.info(f"Prepared data: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            raise
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   optimize_hyperparameters: bool = True) -> xgb.XGBRegressor:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            optimize_hyperparameters: Whether to perform hyperparameter optimization
            
        Returns:
            Trained XGBoost model
        """
        try:
            if optimize_hyperparameters:
                logger.info("Performing hyperparameter optimization...")
                
                # Create base model
                base_model = xgb.XGBRegressor(**self.default_params)
                
                # Perform grid search
                grid_search = GridSearchCV(
                    base_model,
                    self.hyperparameter_grid,
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                
                logger.info(f"Best parameters: {grid_search.best_params_}")
                logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")
                
                model = grid_search.best_estimator_
                
            else:
                logger.info("Training model with default parameters...")
                model = xgb.XGBRegressor(**self.default_params)
                model.fit(X_train, y_train)
            
            logger.info("Model training completed")
            return model
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise
    
    def evaluate_model(self, model: xgb.XGBRegressor,
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate trained model.
        
        Args:
            model: Trained XGBoost model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate additional metrics
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            evaluation_metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'feature_importance': dict(zip(model.feature_names_in_, model.feature_importances_))
            }
            
            logger.info(f"Model evaluation completed:")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  MAPE: {mape:.2f}%")
            
            return evaluation_metrics
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            raise
    
    def save_model(self, model: xgb.XGBRegressor, filepath: str):
        """
        Save trained model to file.
        
        Args:
            model: Trained XGBoost model
            filepath: Path to save the model
        """
        try:
            joblib.dump(model, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str) -> xgb.XGBRegressor:
        """
        Load trained model from file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded XGBoost model
        """
        try:
            model = joblib.load(filepath)
            logger.info(f"Model loaded from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_feature_importance(self, model: xgb.XGBRegressor) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            model: Trained XGBoost model
            
        Returns:
            DataFrame with feature importance
        """
        try:
            importance_df = pd.DataFrame({
                'feature': model.feature_names_in_,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info(f"Feature importance calculated for {len(importance_df)} features")
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            raise 