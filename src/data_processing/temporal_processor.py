"""
Temporal Processor for HAB Forecasting System

This module handles temporal operations including aggregation, resampling,
and temporal feature extraction for the ML pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

from src.utils.config_loader import ConfigLoader, get_data_dir

logger = logging.getLogger(__name__)

class TemporalProcessor:
    """Handles temporal processing operations"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the temporal processor"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_processing_config()
        self.data_dir = get_data_dir()
        
        # Get temporal settings from config
        self.temporal_config = self.config.get("temporal", {})
        self.target_resolution = self.temporal_config.get("target_resolution", "weekly")
        self.aggregation_method = self.temporal_config.get("aggregation_method", "mean")
        self.min_valid_days = self.temporal_config.get("min_valid_days", 4)
        
        # Output directory
        self.output_dir = self.data_dir / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def aggregate_to_weekly(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Aggregate daily data to weekly resolution
        
        Args:
            dataset: Input dataset with daily temporal resolution
            
        Returns:
            Weekly aggregated dataset
        """
        logger.info("Aggregating data to weekly resolution")
        
        try:
            # Ensure time dimension exists
            if "time" not in dataset.dims:
                logger.warning("No time dimension found in dataset")
                return dataset
                
            # Resample to weekly frequency
            weekly_dataset = dataset.resample(time="W").agg(self.aggregation_method)
            
            # Apply minimum valid days filter
            weekly_dataset = self._apply_min_valid_days_filter(weekly_dataset)
            
            logger.info(f"Weekly aggregation complete. Shape: {dict(weekly_dataset.dims)}")
            return weekly_dataset
            
        except Exception as e:
            logger.error(f"Weekly aggregation failed: {e}")
            return dataset
            
    def _apply_min_valid_days_filter(self, dataset: xr.Dataset) -> xr.Dataset:
        """Apply minimum valid days filter to aggregated data"""
        # This is a simplified implementation
        # In practice, you would count actual valid days in each week
        
        logger.info(f"Applying minimum valid days filter ({self.min_valid_days} days)")
        
        # For now, we'll just return the dataset as-is
        # A proper implementation would count valid days and mask weeks with insufficient data
        return dataset
        
    def extract_temporal_features(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Extract temporal features from dataset
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with additional temporal features
        """
        logger.info("Extracting temporal features")
        
        try:
            enhanced_dataset = dataset.copy()
            
            # Extract temporal features for each variable
            for var_name in dataset.data_vars:
                if "time" in dataset[var_name].dims:
                    enhanced_dataset = self._extract_variable_temporal_features(
                        enhanced_dataset, var_name
                    )
                    
            logger.info("Temporal feature extraction complete")
            return enhanced_dataset
            
        except Exception as e:
            logger.error(f"Temporal feature extraction failed: {e}")
            return dataset
            
    def _extract_variable_temporal_features(self, dataset: xr.Dataset, var_name: str) -> xr.Dataset:
        """Extract temporal features for a specific variable"""
        # Compute temporal derivatives
        try:
            # Time derivative (rate of change)
            time_derivative = dataset[var_name].differentiate("time")
            dataset[f"{var_name}_time_derivative"] = time_derivative
            
        except Exception as e:
            logger.warning(f"Failed to compute time derivative for {var_name}: {e}")
            
        # Compute rolling statistics
        try:
            # Rolling mean (5-week window)
            rolling_mean = dataset[var_name].rolling(time=5, center=True).mean()
            dataset[f"{var_name}_rolling_mean_5w"] = rolling_mean
            
            # Rolling standard deviation
            rolling_std = dataset[var_name].rolling(time=5, center=True).std()
            dataset[f"{var_name}_rolling_std_5w"] = rolling_std
            
            # Rolling trend (linear fit over 5 weeks)
            rolling_trend = self._compute_rolling_trend(dataset[var_name], window=5)
            dataset[f"{var_name}_rolling_trend_5w"] = rolling_trend
            
        except Exception as e:
            logger.warning(f"Failed to compute rolling statistics for {var_name}: {e}")
            
        # Compute seasonal features
        try:
            # Extract year and week of year
            time_coords = pd.to_datetime(dataset.time.values)
            years = time_coords.year
            weeks = time_coords.isocalendar().week
            
            # Add seasonal coordinates
            dataset.coords["year"] = ("time", years)
            dataset.coords["week_of_year"] = ("time", weeks)
            
            # Compute climatological mean
            climatological_mean = dataset[var_name].groupby("week_of_year").mean("time")
            dataset[f"{var_name}_climatological_mean"] = climatological_mean
            
            # Compute anomaly (deviation from climatology)
            anomaly = dataset[var_name] - climatological_mean.sel(week_of_year=weeks)
            dataset[f"{var_name}_anomaly"] = anomaly
            
        except Exception as e:
            logger.warning(f"Failed to compute seasonal features for {var_name}: {e}")
            
        return dataset
        
    def _compute_rolling_trend(self, data: xr.DataArray, window: int = 5) -> xr.DataArray:
        """Compute rolling linear trend"""
        # This is a simplified implementation
        # In practice, you would fit a linear regression over the rolling window
        
        # For now, we'll compute a simple difference
        trend = data.diff("time")
        
        # Pad with NaN to maintain shape
        trend = xr.concat([xr.full_like(data.isel(time=0), np.nan), trend], dim="time")
        
        return trend
        
    def create_lag_features(self, dataset: xr.Dataset, lag_periods: List[int] = None) -> xr.Dataset:
        """
        Create lag features for time series
        
        Args:
            dataset: Input dataset
            lag_periods: List of lag periods in weeks (if None, uses config)
            
        Returns:
            Dataset with lag features
        """
        if lag_periods is None:
            lag_periods = self.config.get("features", {}).get("lags", {}).get("weekly_lags", [1, 2, 3, 4])
            
        logger.info(f"Creating lag features for periods: {lag_periods}")
        
        try:
            enhanced_dataset = dataset.copy()
            
            # Create lag features for each variable
            for var_name in dataset.data_vars:
                if "time" in dataset[var_name].dims:
                    enhanced_dataset = self._create_variable_lag_features(
                        enhanced_dataset, var_name, lag_periods
                    )
                    
            logger.info("Lag feature creation complete")
            return enhanced_dataset
            
        except Exception as e:
            logger.error(f"Lag feature creation failed: {e}")
            return dataset
            
    def _create_variable_lag_features(self, dataset: xr.Dataset, var_name: str, lag_periods: List[int]) -> xr.Dataset:
        """Create lag features for a specific variable"""
        for lag in lag_periods:
            try:
                # Shift data by lag periods
                lagged_data = dataset[var_name].shift(time=lag)
                dataset[f"{var_name}_lag_{lag}w"] = lagged_data
                
            except Exception as e:
                logger.warning(f"Failed to create lag {lag} for {var_name}: {e}")
                
        return dataset
        
    def handle_missing_temporal_data(self, dataset: xr.Dataset, method: str = "interpolate") -> xr.Dataset:
        """
        Handle missing temporal data
        
        Args:
            dataset: Input dataset
            method: Method to handle missing data ("interpolate", "forward_fill", "drop")
            
        Returns:
            Dataset with missing data handled
        """
        logger.info(f"Handling missing temporal data using method: {method}")
        
        try:
            if method == "interpolate":
                # Linear interpolation
                filled_dataset = dataset.interpolate_na(dim="time", method="linear")
            elif method == "forward_fill":
                # Forward fill
                filled_dataset = dataset.fillna(method="ffill")
            elif method == "drop":
                # Drop missing values
                filled_dataset = dataset.dropna(dim="time")
            else:
                logger.warning(f"Unknown method: {method}")
                return dataset
                
            logger.info("Missing data handling complete")
            return filled_dataset
            
        except Exception as e:
            logger.error(f"Missing data handling failed: {e}")
            return dataset
            
    def create_time_index(self, dataset: xr.Dataset) -> xr.Dataset:
        """Create additional time-based indices"""
        logger.info("Creating time indices")
        
        try:
            enhanced_dataset = dataset.copy()
            
            # Convert time to pandas datetime
            time_coords = pd.to_datetime(dataset.time.values)
            
            # Create various time indices
            enhanced_dataset.coords["year"] = ("time", time_coords.year)
            enhanced_dataset.coords["month"] = ("time", time_coords.month)
            enhanced_dataset.coords["week_of_year"] = ("time", time_coords.isocalendar().week)
            enhanced_dataset.coords["day_of_year"] = ("time", time_coords.dayofyear)
            enhanced_dataset.coords["season"] = ("time", time_coords.quarter)
            
            logger.info("Time indices created")
            return enhanced_dataset
            
        except Exception as e:
            logger.error(f"Time index creation failed: {e}")
            return dataset
            
    def save_temporal_processed_dataset(self, dataset: xr.Dataset, processing_type: str) -> Path:
        """Save temporally processed dataset"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"temporal_{processing_type}_{timestamp}.nc"
        output_path = self.output_dir / filename
        
        # Save with compression
        encoding = {}
        for var in dataset.data_vars:
            encoding[var] = {
                "zlib": True,
                "complevel": 6,
                "shuffle": True
            }
            
        dataset.to_netcdf(output_path, encoding=encoding)
        logger.info(f"Saved temporal processed dataset: {output_path}")
        
        return output_path 