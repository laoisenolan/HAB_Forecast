"""
HAB Predictor for HAB Forecasting System

This module handles prediction pipeline for HAB forecasting.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HABPredictor:
    """
    Makes predictions using trained HAB forecasting models.
    
    This class handles the prediction pipeline for HAB forecasting,
    including data preprocessing and post-processing of predictions.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the HABPredictor.
        
        Args:
            config: Configuration dictionary with prediction parameters
        """
        self.config = config
        self.model_path = config.get('model_path', 'models/hab_model.pkl')
        self.scaler_path = config.get('scaler_path', 'models/scaler.pkl')
        self.threshold = config.get('prediction_threshold', 0.5)
        
    def load_model(self, model_path: Optional[str] = None) -> object:
        """
        Load trained model from file.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        try:
            path = model_path or self.model_path
            model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_scaler(self, scaler_path: Optional[str] = None) -> object:
        """
        Load data scaler from file.
        
        Args:
            scaler_path: Path to the saved scaler
            
        Returns:
            Loaded scaler
        """
        try:
            path = scaler_path or self.scaler_path
            scaler = joblib.load(path)
            logger.info(f"Scaler loaded from {path}")
            return scaler
            
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame,
                       feature_columns: Optional[List[str]] = None,
                       scaler: Optional[object] = None) -> np.ndarray:
        """
        Preprocess data for prediction.
        
        Args:
            data: DataFrame with features
            feature_columns: List of feature column names
            scaler: Data scaler (optional)
            
        Returns:
            Preprocessed feature array
        """
        try:
            if feature_columns is None:
                feature_columns = [col for col in data.columns if col not in ['date', 'region']]
            
            X = data[feature_columns]
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Scale features if scaler is provided
            if scaler is not None:
                X = scaler.transform(X)
            
            logger.info(f"Preprocessed data: {X.shape}")
            return X
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def predict(self, data: pd.DataFrame,
                model: Optional[object] = None,
                scaler: Optional[object] = None,
                feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            data: DataFrame with features
            model: Trained model (optional, will load if not provided)
            scaler: Data scaler (optional)
            feature_columns: List of feature column names
            
        Returns:
            Array of predictions
        """
        try:
            # Load model if not provided
            if model is None:
                model = self.load_model()
            
            # Load scaler if not provided
            if scaler is None:
                try:
                    scaler = self.load_scaler()
                except:
                    logger.warning("No scaler found, using unscaled data")
                    scaler = None
            
            # Preprocess data
            X = self.preprocess_data(data, feature_columns, scaler)
            
            # Make predictions
            predictions = model.predict(X)
            
            logger.info(f"Made predictions for {len(predictions)} samples")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_proba(self, data: pd.DataFrame,
                     model: Optional[object] = None,
                     scaler: Optional[object] = None,
                     feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Make probability predictions using the trained model.
        
        Args:
            data: DataFrame with features
            model: Trained model (optional, will load if not provided)
            scaler: Data scaler (optional)
            feature_columns: List of feature column names
            
        Returns:
            Array of probability predictions
        """
        try:
            # Load model if not provided
            if model is None:
                model = self.load_model()
            
            # Load scaler if not provided
            if scaler is None:
                try:
                    scaler = self.load_scaler()
                except:
                    logger.warning("No scaler found, using unscaled data")
                    scaler = None
            
            # Preprocess data
            X = self.preprocess_data(data, feature_columns, scaler)
            
            # Make probability predictions
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
            else:
                # For regression models, convert to probabilities
                raw_predictions = model.predict(X)
                probabilities = 1 / (1 + np.exp(-raw_predictions))  # Sigmoid transformation
                probabilities = np.column_stack([1 - probabilities, probabilities])
            
            logger.info(f"Made probability predictions for {len(probabilities)} samples")
            return probabilities
            
        except Exception as e:
            logger.error(f"Error making probability predictions: {e}")
            raise
    
    def classify_predictions(self, predictions: np.ndarray,
                           threshold: Optional[float] = None) -> np.ndarray:
        """
        Classify predictions based on threshold.
        
        Args:
            predictions: Array of predictions
            threshold: Classification threshold
            
        Returns:
            Array of binary classifications
        """
        try:
            threshold = threshold or self.threshold
            classifications = (predictions > threshold).astype(int)
            
            logger.info(f"Classified predictions with threshold {threshold}")
            return classifications
            
        except Exception as e:
            logger.error(f"Error classifying predictions: {e}")
            raise
    
    def create_prediction_dataframe(self, data: pd.DataFrame,
                                  predictions: np.ndarray,
                                  probabilities: Optional[np.ndarray] = None,
                                  classifications: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Create DataFrame with predictions and original data.
        
        Args:
            data: Original feature DataFrame
            predictions: Array of predictions
            probabilities: Array of probability predictions (optional)
            classifications: Array of classifications (optional)
            
        Returns:
            DataFrame with predictions
        """
        try:
            result = data.copy()
            result['prediction'] = predictions
            
            if probabilities is not None:
                result['probability'] = probabilities[:, 1] if probabilities.ndim > 1 else probabilities
            
            if classifications is not None:
                result['classification'] = classifications
            
            logger.info(f"Created prediction DataFrame with {len(result)} rows")
            return result
            
        except Exception as e:
            logger.error(f"Error creating prediction DataFrame: {e}")
            raise
    
    def get_prediction_summary(self, predictions: np.ndarray,
                             classifications: Optional[np.ndarray] = None) -> Dict:
        """
        Get summary statistics for predictions.
        
        Args:
            predictions: Array of predictions
            classifications: Array of classifications (optional)
            
        Returns:
            Dictionary with prediction summary
        """
        try:
            summary = {
                'mean_prediction': np.mean(predictions),
                'std_prediction': np.std(predictions),
                'min_prediction': np.min(predictions),
                'max_prediction': np.max(predictions),
                'median_prediction': np.median(predictions)
            }
            
            if classifications is not None:
                summary['positive_classifications'] = np.sum(classifications)
                summary['negative_classifications'] = len(classifications) - np.sum(classifications)
                summary['positive_rate'] = np.mean(classifications)
            
            logger.info(f"Generated prediction summary")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating prediction summary: {e}")
            raise 