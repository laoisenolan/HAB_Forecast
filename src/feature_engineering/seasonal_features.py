"""
Seasonal Features for HAB Forecasting System

This module handles seasonal and cyclical feature generation for the ML pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

from src.utils.config_loader import ConfigLoader, get_data_dir

logger = logging.getLogger(__name__)

class SeasonalFeatureGenerator:
    """Generates seasonal and cyclical features"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the seasonal feature generator"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_features_config()
        self.data_dir = get_data_dir()
        
        # Get seasonal settings from config
        self.seasonal_config = self.config.get("computation", {}).get("seasonal", {})
        self.include_cyclical = self.seasonal_config.get("include_cyclical", True)
        self.cyclical_encoding = self.seasonal_config.get("cyclical_encoding", "sin_cos")
        self.seasonal_periods = self.seasonal_config.get("seasonal_periods", ["weekly", "monthly", "quarterly"])
        
        # Output directory
        self.output_dir = self.data_dir / "features"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_seasonal_features(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Create seasonal and cyclical features
        
        Args:
            dataset: Input dataset with time dimension
            
        Returns:
            Dataset with seasonal features added
        """
        logger.info("Creating seasonal features")
        
        enhanced_dataset = dataset.copy()
        
        try:
            # Add basic time-based features
            enhanced_dataset = self._add_time_features(enhanced_dataset)
            
            # Add seasonal features
            enhanced_dataset = self._add_seasonal_features(enhanced_dataset)
            
            # Add cyclical features if requested
            if self.include_cyclical:
                enhanced_dataset = self._add_cyclical_features(enhanced_dataset)
                
            # Add climatological features
            enhanced_dataset = self._add_climatological_features(enhanced_dataset)
            
            logger.info("Seasonal feature creation complete")
            return enhanced_dataset
            
        except Exception as e:
            logger.error(f"Seasonal feature creation failed: {e}")
            return dataset
            
    def _add_time_features(self, dataset: xr.Dataset) -> xr.Dataset:
        """Add basic time-based features"""
        try:
            time_coords = pd.to_datetime(dataset.coords["time"].values)
            
            # Week of year
            if "week_of_year" not in dataset.coords:
                week_of_year = time_coords.isocalendar().week
                dataset = dataset.assign_coords(week_of_year=("time", week_of_year))
                
            # Month
            month = time_coords.month
            dataset = dataset.assign_coords(month=("time", month))
            
            # Day of year
            day_of_year = time_coords.dayofyear
            dataset = dataset.assign_coords(day_of_year=("time", day_of_year))
            
            # Year
            year = time_coords.year
            dataset = dataset.assign_coords(year=("time", year))
            
            # Quarter
            quarter = time_coords.quarter
            dataset = dataset.assign_coords(quarter=("time", quarter))
            
        except Exception as e:
            logger.warning(f"Failed to add time features: {e}")
            
        return dataset
        
    def _add_seasonal_features(self, dataset: xr.Dataset) -> xr.Dataset:
        """Add seasonal features"""
        try:
            time_coords = pd.to_datetime(dataset.coords["time"].values)
            
            # Season (DJF, MAM, JJA, SON)
            seasons = []
            for date in time_coords:
                month = date.month
                if month in [12, 1, 2]:
                    seasons.append("DJF")
                elif month in [3, 4, 5]:
                    seasons.append("MAM")
                elif month in [6, 7, 8]:
                    seasons.append("JJA")
                else:
                    seasons.append("SON")
                    
            # Convert to numeric encoding
            season_encoding = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}
            season_numeric = [season_encoding[season] for season in seasons]
            
            dataset = dataset.assign_coords(season=("time", season_numeric))
            
            # Add season as categorical
            dataset = dataset.assign_coords(season_cat=("time", seasons))
            
        except Exception as e:
            logger.warning(f"Failed to add seasonal features: {e}")
            
        return dataset
        
    def _add_cyclical_features(self, dataset: xr.Dataset) -> xr.Dataset:
        """Add cyclical features using sin/cos encoding"""
        try:
            time_coords = pd.to_datetime(dataset.coords["time"].values)
            
            if self.cyclical_encoding == "sin_cos":
                # Weekly cyclical features
                week_sin = np.sin(2 * np.pi * time_coords.dayofyear / 7)
                week_cos = np.cos(2 * np.pi * time_coords.dayofyear / 7)
                dataset = dataset.assign_coords(week_sin=("time", week_sin))
                dataset = dataset.assign_coords(week_cos=("time", week_cos))
                
                # Monthly cyclical features
                month_sin = np.sin(2 * np.pi * time_coords.month / 12)
                month_cos = np.cos(2 * np.pi * time_coords.month / 12)
                dataset = dataset.assign_coords(month_sin=("time", month_sin))
                dataset = dataset.assign_coords(month_cos=("time", month_cos))
                
                # Annual cyclical features
                year_sin = np.sin(2 * np.pi * time_coords.dayofyear / 365.25)
                year_cos = np.cos(2 * np.pi * time_coords.dayofyear / 365.25)
                dataset = dataset.assign_coords(year_sin=("time", year_sin))
                dataset = dataset.assign_coords(year_cos=("time", year_cos))
                
                # Quarter cyclical features
                quarter_sin = np.sin(2 * np.pi * time_coords.quarter / 4)
                quarter_cos = np.cos(2 * np.pi * time_coords.quarter / 4)
                dataset = dataset.assign_coords(quarter_sin=("time", quarter_sin))
                dataset = dataset.assign_coords(quarter_cos=("time", quarter_cos))
                
        except Exception as e:
            logger.warning(f"Failed to add cyclical features: {e}")
            
        return dataset
        
    def _add_climatological_features(self, dataset: xr.Dataset) -> xr.Dataset:
        """Add climatological features based on historical averages"""
        try:
            # Find environmental variables
            env_vars = []
            for var_name in dataset.data_vars:
                if any(keyword in var_name.lower() for keyword in ["temperature", "chlorophyll", "mld", "wind"]):
                    env_vars.append(var_name)
                    
            for var_name in env_vars:
                var_data = dataset[var_name]
                
                # Compute climatological mean by week of year
                if "week_of_year" in dataset.coords:
                    climatology = var_data.groupby("week_of_year").mean("time")
                    climatology_std = var_data.groupby("week_of_year").std("time")
                    
                    # Add climatological features
                    dataset[f"{var_name}_climatological_mean"] = climatology.sel(week_of_year=dataset.coords["week_of_year"])
                    dataset[f"{var_name}_climatological_std"] = climatology_std.sel(week_of_year=dataset.coords["week_of_year"])
                    
                    # Compute seasonal anomalies
                    seasonal_anomaly = var_data - dataset[f"{var_name}_climatological_mean"]
                    dataset[f"{var_name}_seasonal_anomaly"] = seasonal_anomaly
                    
                # Compute climatological mean by month
                if "month" in dataset.coords:
                    monthly_climatology = var_data.groupby("month").mean("time")
                    dataset[f"{var_name}_monthly_climatology"] = monthly_climatology.sel(month=dataset.coords["month"])
                    
        except Exception as e:
            logger.warning(f"Failed to add climatological features: {e}")
            
        return dataset
        
    def create_holiday_features(self, dataset: xr.Dataset) -> xr.Dataset:
        """Create holiday and special event features (placeholder for future implementation)"""
        logger.info("Creating holiday features (placeholder)")
        
        # This is a placeholder for future implementation
        # Could include Japanese holidays, fishing seasons, etc.
        
        return dataset
        
    def create_trend_features(self, dataset: xr.Dataset, window_sizes: List[int] = None) -> xr.Dataset:
        """
        Create trend features using rolling windows
        
        Args:
            dataset: Input dataset
            window_sizes: List of window sizes for trend computation
            
        Returns:
            Dataset with trend features added
        """
        if window_sizes is None:
            window_sizes = [4, 8, 12]  # 4, 8, 12 weeks
            
        logger.info(f"Creating trend features with window sizes: {window_sizes}")
        
        enhanced_dataset = dataset.copy()
        
        try:
            # Find environmental variables
            env_vars = []
            for var_name in dataset.data_vars:
                if any(keyword in var_name.lower() for keyword in ["temperature", "chlorophyll", "mld", "wind"]):
                    env_vars.append(var_name)
                    
            for var_name in env_vars:
                var_data = dataset[var_name]
                
                for window in window_sizes:
                    # Compute rolling trend (slope)
                    trend = self._compute_rolling_trend(var_data, window)
                    enhanced_dataset[f"{var_name}_trend_{window}w"] = trend
                    
                    # Compute rolling mean
                    rolling_mean = var_data.rolling(time=window, center=True).mean()
                    enhanced_dataset[f"{var_name}_rolling_mean_{window}w"] = rolling_mean
                    
                    # Compute rolling std
                    rolling_std = var_data.rolling(time=window, center=True).std()
                    enhanced_dataset[f"{var_name}_rolling_std_{window}w"] = rolling_std
                    
            logger.info("Trend feature creation complete")
            return enhanced_dataset
            
        except Exception as e:
            logger.error(f"Trend feature creation failed: {e}")
            return dataset
            
    def _compute_rolling_trend(self, data: xr.DataArray, window: int) -> xr.DataArray:
        """Compute rolling trend using linear regression"""
        try:
            # Create time index for regression
            time_index = np.arange(len(data.coords["time"]))
            
            # Initialize trend array
            trend = xr.zeros_like(data)
            
            # Compute trend for each spatial location
            for i in range(data.shape[1]):  # latitude
                for j in range(data.shape[2]):  # longitude
                    series = data[:, i, j]
                    
                    # Compute rolling trend
                    for t in range(window, len(series)):
                        if not series.isnull().values[t-window:t].all():
                            # Get valid data points
                            valid_mask = ~series.isnull().values[t-window:t]
                            if valid_mask.sum() >= window // 2:  # At least half the window
                                y = series.values[t-window:t][valid_mask]
                                x = time_index[t-window:t][valid_mask]
                                
                                # Simple linear regression
                                slope = np.polyfit(x, y, 1)[0]
                                trend[t, i, j] = slope
                                
        except Exception as e:
            logger.warning(f"Failed to compute rolling trend: {e}")
            
        return trend
        
    def save_seasonal_features(self, dataset: xr.Dataset, filename: str = None) -> Path:
        """Save seasonal features to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"seasonal_features_{timestamp}.nc"
            
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
        logger.info(f"Saved seasonal features: {output_path}")
        
        return output_path
        
    def get_seasonal_feature_summary(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """Get summary of seasonal features"""
        summary = {
            "time_features": [],
            "seasonal_features": [],
            "cyclical_features": [],
            "climatological_features": [],
            "trend_features": [],
            "statistics": {}
        }
        
        # Categorize features
        for var_name in dataset.data_vars:
            if any(keyword in var_name for keyword in ["week_of_year", "month", "day_of_year", "year", "quarter"]):
                summary["time_features"].append(var_name)
            elif any(keyword in var_name for keyword in ["season", "DJF", "MAM", "JJA", "SON"]):
                summary["seasonal_features"].append(var_name)
            elif any(keyword in var_name for keyword in ["sin", "cos"]):
                summary["cyclical_features"].append(var_name)
            elif any(keyword in var_name for keyword in ["climatological", "seasonal_anomaly"]):
                summary["climatological_features"].append(var_name)
            elif any(keyword in var_name for keyword in ["trend", "rolling"]):
                summary["trend_features"].append(var_name)
                
            # Compute statistics for seasonal features
            if any(keyword in var_name for keyword in ["seasonal", "cyclical", "climatological", "trend"]):
                var_data = dataset[var_name]
                summary["statistics"][var_name] = {
                    "mean": float(var_data.mean()),
                    "std": float(var_data.std()),
                    "min": float(var_data.min()),
                    "max": float(var_data.max()),
                    "missing_fraction": float(var_data.isnull().sum() / var_data.size)
                }
                
        return summary 