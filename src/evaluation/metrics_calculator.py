"""
Metrics Calculator for HAB Forecasting System

This module handles performance metrics computation for model evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates performance metrics for HAB forecasting models.
    
    This class handles the computation of various performance metrics
    for binary classification and regression tasks in HAB prediction.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the MetricsCalculator.
        
        Args:
            config: Configuration dictionary with metrics parameters
        """
        self.config = config
        self.threshold = config.get('classification_threshold', 0.5)
        self.metrics_to_compute = config.get('metrics_to_compute', [
            'accuracy', 'precision', 'recall', 'f1', 'auc',
            'mse', 'mae', 'r2'
        ])
        
    def calculate_classification_metrics(self, y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      y_prob: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary with classification metrics
        """
        try:
            metrics = {}
            
            # Basic classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
            
            # AUC if probabilities are provided
            if y_prob is not None:
                if y_prob.ndim > 1:
                    y_prob = y_prob[:, 1]  # Use positive class probability
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = {
                'tn': int(cm[0, 0]),
                'fp': int(cm[0, 1]),
                'fn': int(cm[1, 0]),
                'tp': int(cm[1, 1])
            }
            
            # Additional metrics
            metrics['specificity'] = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            metrics['sensitivity'] = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
            
            logger.info(f"Calculated classification metrics for {len(y_true)} samples")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}")
            raise
    
    def calculate_regression_metrics(self, y_true: np.ndarray,
                                   y_pred: np.ndarray) -> Dict:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with regression metrics
        """
        try:
            metrics = {}
            
            # Basic regression metrics
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Additional metrics
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['smape'] = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
            
            # Correlation
            metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
            
            logger.info(f"Calculated regression metrics for {len(y_true)} samples")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating regression metrics: {e}")
            raise
    
    def calculate_spatial_metrics(self, y_true: np.ndarray,
                                y_pred: np.ndarray,
                                coordinates: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate spatial accuracy metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            coordinates: Spatial coordinates (optional)
            
        Returns:
            Dictionary with spatial metrics
        """
        try:
            metrics = {}
            
            # Spatial accuracy (if coordinates provided)
            if coordinates is not None:
                # Calculate spatial correlation
                spatial_corr = np.corrcoef(y_true, y_pred)[0, 1]
                metrics['spatial_correlation'] = spatial_corr
                
                # Calculate spatial error patterns
                spatial_errors = y_true - y_pred
                metrics['spatial_error_mean'] = np.mean(spatial_errors)
                metrics['spatial_error_std'] = np.std(spatial_errors)
            
            # Contingency table for binary classification
            if len(np.unique(y_true)) == 2 and len(np.unique(y_pred)) == 2:
                cm = confusion_matrix(y_true, y_pred)
                metrics['spatial_accuracy'] = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
            
            logger.info(f"Calculated spatial metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating spatial metrics: {e}")
            raise
    
    def calculate_temporal_metrics(self, y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 timestamps: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate temporal accuracy metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            timestamps: Temporal indices (optional)
            
        Returns:
            Dictionary with temporal metrics
        """
        try:
            metrics = {}
            
            # Temporal correlation
            temporal_corr = np.corrcoef(y_true, y_pred)[0, 1]
            metrics['temporal_correlation'] = temporal_corr
            
            # Temporal error patterns
            temporal_errors = y_true - y_pred
            metrics['temporal_error_mean'] = np.mean(temporal_errors)
            metrics['temporal_error_std'] = np.std(temporal_errors)
            
            # Trend analysis
            if timestamps is not None:
                # Calculate trend correlation
                trend_true = np.polyfit(timestamps, y_true, 1)[0]
                trend_pred = np.polyfit(timestamps, y_pred, 1)[0]
                metrics['trend_correlation'] = np.corrcoef([trend_true], [trend_pred])[0, 1]
            
            logger.info(f"Calculated temporal metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating temporal metrics: {e}")
            raise
    
    def generate_classification_report(self, y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    target_names: Optional[List[str]] = None) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names for target classes
            
        Returns:
            Classification report string
        """
        try:
            if target_names is None:
                target_names = ['No HAB', 'HAB']
            
            report = classification_report(y_true, y_pred, target_names=target_names)
            logger.info("Generated classification report")
            return report
            
        except Exception as e:
            logger.error(f"Error generating classification report: {e}")
            raise
    
    def calculate_all_metrics(self, y_true: np.ndarray,
                            y_pred: np.ndarray,
                            y_prob: Optional[np.ndarray] = None,
                            coordinates: Optional[np.ndarray] = None,
                            timestamps: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate all available metrics.
        
        Args:
            y_true: True values/labels
            y_pred: Predicted values/labels
            y_prob: Predicted probabilities (optional)
            coordinates: Spatial coordinates (optional)
            timestamps: Temporal indices (optional)
            
        Returns:
            Dictionary with all metrics
        """
        try:
            all_metrics = {}
            
            # Determine if this is classification or regression
            if len(np.unique(y_true)) == 2 and len(np.unique(y_pred)) == 2:
                # Classification metrics
                all_metrics.update(self.calculate_classification_metrics(y_true, y_pred, y_prob))
                
                # Generate classification report
                all_metrics['classification_report'] = self.generate_classification_report(y_true, y_pred)
            else:
                # Regression metrics
                all_metrics.update(self.calculate_regression_metrics(y_true, y_pred))
            
            # Spatial metrics
            if coordinates is not None:
                all_metrics.update(self.calculate_spatial_metrics(y_true, y_pred, coordinates))
            
            # Temporal metrics
            if timestamps is not None:
                all_metrics.update(self.calculate_temporal_metrics(y_true, y_pred, timestamps))
            
            logger.info(f"Calculated comprehensive metrics")
            return all_metrics
            
        except Exception as e:
            logger.error(f"Error calculating all metrics: {e}")
            raise 