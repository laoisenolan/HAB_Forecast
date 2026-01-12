"""
Temporal Validator for HAB Forecasting System

This module handles temporal validation for model evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TemporalValidator:
    """
    Validates temporal accuracy of HAB forecasting models.
    
    This class handles temporal validation, including
    time series analysis and temporal prediction accuracy.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the TemporalValidator.
        
        Args:
            config: Configuration dictionary with temporal validation parameters
        """
        self.config = config
        self.validation_window = config.get('validation_window', 30)  # days
        self.forecast_horizons = config.get('forecast_horizons', [1, 3, 7])  # days
        
    def validate_temporal_accuracy(self, y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 timestamps: np.ndarray) -> Dict:
        """
        Validate temporal accuracy of predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            timestamps: Temporal indices
            
        Returns:
            Dictionary with temporal validation metrics
        """
        try:
            # Sort by time
            time_order = np.argsort(timestamps)
            y_true_sorted = y_true[time_order]
            y_pred_sorted = y_pred[time_order]
            timestamps_sorted = timestamps[time_order]
            
            # Calculate temporal correlation
            temporal_corr = np.corrcoef(y_true_sorted, y_pred_sorted)[0, 1]
            
            # Calculate temporal error patterns
            temporal_errors = y_true_sorted - y_pred_sorted
            error_mean = np.mean(temporal_errors)
            error_std = np.std(temporal_errors)
            
            # Calculate trend correlation
            trend_true = np.polyfit(timestamps_sorted, y_true_sorted, 1)[0]
            trend_pred = np.polyfit(timestamps_sorted, y_pred_sorted, 1)[0]
            trend_corr = np.corrcoef([trend_true], [trend_pred])[0, 1]
            
            metrics = {
                'temporal_correlation': temporal_corr,
                'temporal_error_mean': error_mean,
                'temporal_error_std': error_std,
                'trend_correlation': trend_corr,
                'temporal_rmse': np.sqrt(np.mean(temporal_errors**2)),
                'temporal_mae': np.mean(np.abs(temporal_errors))
            }
            
            logger.info(f"Validated temporal accuracy: correlation = {temporal_corr:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error validating temporal accuracy: {e}")
            raise
    
    def validate_forecast_horizons(self, y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 timestamps: np.ndarray,
                                 horizons: Optional[List[int]] = None) -> Dict:
        """
        Validate predictions at different forecast horizons.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            timestamps: Temporal indices
            horizons: List of forecast horizons in days
            
        Returns:
            Dictionary with horizon-specific metrics
        """
        try:
            horizons = horizons or self.forecast_horizons
            horizon_metrics = {}
            
            for horizon in horizons:
                # Find predictions at this horizon
                horizon_mask = np.diff(timestamps) == horizon
                if np.any(horizon_mask):
                    horizon_true = y_true[1:][horizon_mask]
                    horizon_pred = y_pred[:-1][horizon_mask]
                    
                    if len(horizon_true) > 0:
                        # Calculate metrics for this horizon
                        horizon_corr = np.corrcoef(horizon_true, horizon_pred)[0, 1]
                        horizon_rmse = np.sqrt(np.mean((horizon_true - horizon_pred)**2))
                        horizon_mae = np.mean(np.abs(horizon_true - horizon_pred))
                        
                        horizon_metrics[f'{horizon}d'] = {
                            'correlation': horizon_corr,
                            'rmse': horizon_rmse,
                            'mae': horizon_mae,
                            'sample_count': len(horizon_true)
                        }
            
            logger.info(f"Validated forecast horizons: {list(horizon_metrics.keys())}")
            return horizon_metrics
            
        except Exception as e:
            logger.error(f"Error validating forecast horizons: {e}")
            raise
    
    def calculate_temporal_trends(self, y_true: np.ndarray,
                                y_pred: np.ndarray,
                                timestamps: np.ndarray) -> Dict:
        """
        Calculate temporal trends in predictions vs observations.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            timestamps: Temporal indices
            
        Returns:
            Dictionary with trend analysis
        """
        try:
            # Sort by time
            time_order = np.argsort(timestamps)
            y_true_sorted = y_true[time_order]
            y_pred_sorted = y_pred[time_order]
            timestamps_sorted = timestamps[time_order]
            
            # Calculate trends
            trend_true = np.polyfit(timestamps_sorted, y_true_sorted, 1)
            trend_pred = np.polyfit(timestamps_sorted, y_pred_sorted, 1)
            
            # Calculate seasonal patterns (if enough data)
            seasonal_patterns = {}
            if len(timestamps_sorted) > 365:  # At least one year
                # Simple seasonal decomposition
                seasonal_patterns = self._calculate_seasonal_patterns(
                    y_true_sorted, y_pred_sorted, timestamps_sorted
                )
            
            trends = {
                'true_trend_slope': trend_true[0],
                'pred_trend_slope': trend_pred[0],
                'true_trend_intercept': trend_true[1],
                'pred_trend_intercept': trend_pred[1],
                'trend_difference': trend_true[0] - trend_pred[0],
                'seasonal_patterns': seasonal_patterns
            }
            
            logger.info(f"Calculated temporal trends")
            return trends
            
        except Exception as e:
            logger.error(f"Error calculating temporal trends: {e}")
            raise
    
    def _calculate_seasonal_patterns(self, y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   timestamps: np.ndarray) -> Dict:
        """
        Calculate seasonal patterns in the data.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            timestamps: Temporal indices
            
        Returns:
            Dictionary with seasonal patterns
        """
        try:
            # Simple seasonal analysis
            seasonal_stats = {}
            
            # Group by month (assuming timestamps are in days)
            months = (timestamps % 365) // 30
            unique_months = np.unique(months)
            
            for month in unique_months:
                month_mask = months == month
                if np.sum(month_mask) > 0:
                    month_true = y_true[month_mask]
                    month_pred = y_pred[month_mask]
                    
                    seasonal_stats[f'month_{month}'] = {
                        'true_mean': np.mean(month_true),
                        'pred_mean': np.mean(month_pred),
                        'true_std': np.std(month_true),
                        'pred_std': np.std(month_pred),
                        'correlation': np.corrcoef(month_true, month_pred)[0, 1]
                    }
            
            return seasonal_stats
            
        except Exception as e:
            logger.error(f"Error calculating seasonal patterns: {e}")
            return {}
    
    def validate_persistence(self, y_true: np.ndarray,
                           y_pred: np.ndarray,
                           timestamps: np.ndarray) -> Dict:
        """
        Validate model performance against persistence forecast.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            timestamps: Temporal indices
            
        Returns:
            Dictionary with persistence validation metrics
        """
        try:
            # Sort by time
            time_order = np.argsort(timestamps)
            y_true_sorted = y_true[time_order]
            y_pred_sorted = y_pred[time_order]
            
            # Calculate persistence forecast (previous value)
            persistence_pred = np.roll(y_true_sorted, 1)
            persistence_pred[0] = y_true_sorted[0]  # First value
            
            # Calculate metrics for model vs persistence
            model_rmse = np.sqrt(np.mean((y_true_sorted - y_pred_sorted)**2))
            persistence_rmse = np.sqrt(np.mean((y_true_sorted - persistence_pred)**2))
            
            model_mae = np.mean(np.abs(y_true_sorted - y_pred_sorted))
            persistence_mae = np.mean(np.abs(y_true_sorted - persistence_pred))
            
            # Calculate skill score
            skill_score = 1 - (model_rmse / persistence_rmse)
            
            persistence_metrics = {
                'model_rmse': model_rmse,
                'persistence_rmse': persistence_rmse,
                'model_mae': model_mae,
                'persistence_mae': persistence_mae,
                'skill_score': skill_score,
                'improvement_over_persistence': (persistence_rmse - model_rmse) / persistence_rmse
            }
            
            logger.info(f"Validated persistence: skill score = {skill_score:.4f}")
            return persistence_metrics
            
        except Exception as e:
            logger.error(f"Error validating persistence: {e}")
            raise
    
    def calculate_temporal_consistency(self, y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     timestamps: np.ndarray) -> Dict:
        """
        Calculate temporal consistency of predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            timestamps: Temporal indices
            
        Returns:
            Dictionary with temporal consistency metrics
        """
        try:
            # Sort by time
            time_order = np.argsort(timestamps)
            y_true_sorted = y_true[time_order]
            y_pred_sorted = y_pred[time_order]
            
            # Calculate temporal consistency
            true_changes = np.diff(y_true_sorted)
            pred_changes = np.diff(y_pred_sorted)
            
            # Direction consistency
            direction_consistency = np.mean(
                np.sign(true_changes) == np.sign(pred_changes)
            )
            
            # Magnitude consistency
            magnitude_correlation = np.corrcoef(np.abs(true_changes), np.abs(pred_changes))[0, 1]
            
            # Temporal smoothness
            true_smoothness = np.std(np.diff(true_changes))
            pred_smoothness = np.std(np.diff(pred_changes))
            smoothness_ratio = pred_smoothness / true_smoothness if true_smoothness > 0 else 0
            
            consistency_metrics = {
                'direction_consistency': direction_consistency,
                'magnitude_correlation': magnitude_correlation,
                'true_smoothness': true_smoothness,
                'pred_smoothness': pred_smoothness,
                'smoothness_ratio': smoothness_ratio
            }
            
            logger.info(f"Calculated temporal consistency: direction = {direction_consistency:.4f}")
            return consistency_metrics
            
        except Exception as e:
            logger.error(f"Error calculating temporal consistency: {e}")
            raise 