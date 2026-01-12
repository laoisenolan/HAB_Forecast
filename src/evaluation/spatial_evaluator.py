"""
Spatial Evaluator for HAB Forecasting System

This module handles spatial accuracy assessment for model evaluation.
"""

import pandas as pd
import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SpatialEvaluator:
    """
    Evaluates spatial accuracy of HAB forecasting models.
    
    This class handles spatial accuracy assessment, including
    spatial correlation analysis and regional performance evaluation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the SpatialEvaluator.
        
        Args:
            config: Configuration dictionary with spatial evaluation parameters
        """
        self.config = config
        self.regions = config.get('regions', {})
        self.spatial_threshold = config.get('spatial_threshold', 0.1)
        
    def calculate_spatial_correlation(self, y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   coordinates: np.ndarray) -> Dict:
        """
        Calculate spatial correlation between true and predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            coordinates: Spatial coordinates (lat, lon)
            
        Returns:
            Dictionary with spatial correlation metrics
        """
        try:
            # Calculate spatial correlation
            spatial_corr = np.corrcoef(y_true, y_pred)[0, 1]
            
            # Calculate spatial error patterns
            spatial_errors = y_true - y_pred
            error_mean = np.mean(spatial_errors)
            error_std = np.std(spatial_errors)
            
            # Calculate spatial autocorrelation of errors
            spatial_autocorr = self._calculate_spatial_autocorrelation(spatial_errors, coordinates)
            
            metrics = {
                'spatial_correlation': spatial_corr,
                'spatial_error_mean': error_mean,
                'spatial_error_std': error_std,
                'spatial_autocorrelation': spatial_autocorr
            }
            
            logger.info(f"Calculated spatial correlation: {spatial_corr:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating spatial correlation: {e}")
            raise
    
    def _calculate_spatial_autocorrelation(self, errors: np.ndarray,
                                         coordinates: np.ndarray) -> float:
        """
        Calculate spatial autocorrelation of errors.
        
        Args:
            errors: Error values
            coordinates: Spatial coordinates
            
        Returns:
            Spatial autocorrelation coefficient
        """
        try:
            # Simple spatial autocorrelation using distance weighting
            n_points = len(errors)
            if n_points < 2:
                return 0.0
            
            # Calculate distances between all pairs
            distances = []
            error_products = []
            
            for i in range(n_points):
                for j in range(i+1, n_points):
                    dist = np.sqrt(np.sum((coordinates[i] - coordinates[j])**2))
                    if dist > 0:
                        distances.append(dist)
                        error_products.append(errors[i] * errors[j])
            
            if len(distances) == 0:
                return 0.0
            
            # Calculate weighted correlation
            weights = 1 / np.array(distances)
            weighted_mean = np.average(error_products, weights=weights)
            
            return weighted_mean / (np.std(errors) ** 2)
            
        except Exception as e:
            logger.error(f"Error calculating spatial autocorrelation: {e}")
            return 0.0
    
    def evaluate_regional_performance(self, y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    regions: np.ndarray) -> Dict:
        """
        Evaluate model performance by region.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            regions: Region labels for each sample
            
        Returns:
            Dictionary with regional performance metrics
        """
        try:
            regional_metrics = {}
            unique_regions = np.unique(regions)
            
            for region in unique_regions:
                region_mask = regions == region
                region_true = y_true[region_mask]
                region_pred = y_pred[region_mask]
                
                if len(region_true) > 0:
                    # Calculate metrics for this region
                    region_corr = np.corrcoef(region_true, region_pred)[0, 1]
                    region_rmse = np.sqrt(np.mean((region_true - region_pred)**2))
                    region_mae = np.mean(np.abs(region_true - region_pred))
                    
                    regional_metrics[region] = {
                        'correlation': region_corr,
                        'rmse': region_rmse,
                        'mae': region_mae,
                        'sample_count': len(region_true)
                    }
            
            logger.info(f"Evaluated regional performance for {len(unique_regions)} regions")
            return regional_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating regional performance: {e}")
            raise
    
    def calculate_spatial_accuracy(self, y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 coordinates: np.ndarray,
                                 threshold: Optional[float] = None) -> Dict:
        """
        Calculate spatial accuracy metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            coordinates: Spatial coordinates
            threshold: Classification threshold
            
        Returns:
            Dictionary with spatial accuracy metrics
        """
        try:
            threshold = threshold or self.spatial_threshold
            
            # Convert to binary classification if needed
            if len(np.unique(y_true)) > 2:
                y_true_binary = (y_true > threshold).astype(int)
                y_pred_binary = (y_pred > threshold).astype(int)
            else:
                y_true_binary = y_true
                y_pred_binary = y_pred
            
            # Calculate spatial accuracy
            spatial_accuracy = np.mean(y_true_binary == y_pred_binary)
            
            # Calculate spatial precision and recall
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            
            spatial_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            spatial_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            metrics = {
                'spatial_accuracy': spatial_accuracy,
                'spatial_precision': spatial_precision,
                'spatial_recall': spatial_recall,
                'spatial_f1': 2 * (spatial_precision * spatial_recall) / (spatial_precision + spatial_recall) if (spatial_precision + spatial_recall) > 0 else 0
            }
            
            logger.info(f"Calculated spatial accuracy: {spatial_accuracy:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating spatial accuracy: {e}")
            raise
    
    def analyze_spatial_patterns(self, dataset: xr.Dataset,
                               variable: str,
                               time_dim: str = 'time') -> Dict:
        """
        Analyze spatial patterns in a dataset.
        
        Args:
            dataset: xarray Dataset
            variable: Variable name to analyze
            time_dim: Time dimension name
            
        Returns:
            Dictionary with spatial pattern analysis
        """
        try:
            var_data = dataset[variable]
            
            # Calculate spatial statistics
            spatial_mean = var_data.mean(dim=time_dim)
            spatial_std = var_data.std(dim=time_dim)
            spatial_trend = self._calculate_spatial_trend(var_data, time_dim)
            
            # Calculate spatial gradients
            if 'lat' in var_data.dims and 'lon' in var_data.dims:
                lat_gradient = var_data.diff('lat').mean(dim=time_dim)
                lon_gradient = var_data.diff('lon').mean(dim=time_dim)
            else:
                lat_gradient = None
                lon_gradient = None
            
            analysis = {
                'spatial_mean': spatial_mean,
                'spatial_std': spatial_std,
                'spatial_trend': spatial_trend,
                'lat_gradient': lat_gradient,
                'lon_gradient': lon_gradient
            }
            
            logger.info(f"Analyzed spatial patterns for {variable}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing spatial patterns: {e}")
            raise
    
    def _calculate_spatial_trend(self, data: xr.DataArray,
                               time_dim: str) -> xr.DataArray:
        """
        Calculate spatial trend over time.
        
        Args:
            data: xarray DataArray
            time_dim: Time dimension name
            
        Returns:
            Spatial trend DataArray
        """
        try:
            # Calculate linear trend for each spatial location
            trend = data.polyfit(dim=time_dim, deg=1)
            return trend.polyfit_coefficients.sel(degree=1)
            
        except Exception as e:
            logger.error(f"Error calculating spatial trend: {e}")
            return None 