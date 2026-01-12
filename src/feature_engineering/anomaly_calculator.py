"""
Anomaly Calculator for HAB Forecasting System

This module computes environmental anomalies from climatological means
for use as features in the ML pipeline.
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

class AnomalyCalculator:
    """Computes environmental anomalies from climatological means"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the anomaly calculator"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_features_config()
        self.data_dir = get_data_dir()
        
        # Get anomaly settings from config
        self.anomaly_config = self.config.get("computation", {}).get("anomaly", {})
        self.climatology_period = self.anomaly_config.get("climatology_period", "2002-2020")
        self.anomaly_method = self.anomaly_config.get("method", "climatological_mean")
        self.window_size = self.anomaly_config.get("window_size", 5)
        
        # Output directory
        self.output_dir = self.data_dir / "features"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def compute_anomalies(self, dataset: xr.Dataset, variables: List[str] = None) -> xr.Dataset:
        """
        Compute anomalies for specified variables
        
        Args:
            dataset: Input dataset
            variables: List of variables to compute anomalies for (if None, uses all numeric variables)
            
        Returns:
            Dataset with anomaly features added
        """
        logger.info("Computing environmental anomalies")
        
        if variables is None:
            # Get all numeric variables
            variables = [var for var in dataset.data_vars 
                        if dataset[var].dtype.kind in 'fc']
            
        enhanced_dataset = dataset.copy()
        
        try:
            for variable in variables:
                if variable in dataset.data_vars:
                    logger.info(f"Computing anomaly for {variable}")
                    
                    # Compute climatological mean
                    climatology = self._compute_climatology(dataset[variable])
                    
                    # Compute anomaly
                    anomaly = self._compute_anomaly(dataset[variable], climatology)
                    
                    # Add to dataset
                    enhanced_dataset[f"{variable}_anomaly"] = anomaly
                    
                    # Add climatology to dataset
                    enhanced_dataset[f"{variable}_climatology"] = climatology
                    
            logger.info(f"Anomaly computation complete. Added {len(variables)} anomaly features")
            return enhanced_dataset
            
        except Exception as e:
            logger.error(f"Anomaly computation failed: {e}")
            return dataset
            
    def _compute_climatology(self, data: xr.DataArray) -> xr.DataArray:
        """Compute climatological mean for a variable"""
        try:
            # Parse climatology period
            start_year, end_year = self.climatology_period.split("-")
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-12-31"
            
            # Select climatology period
            climatology_data = data.sel(time=slice(start_date, end_date))
            
            if len(climatology_data.time) == 0:
                logger.warning(f"No data found for climatology period {self.climatology_period}")
                # Use all available data as fallback
                climatology_data = data
                
            # Compute climatological mean by week of year
            time_coords = pd.to_datetime(climatology_data.time.values)
            weeks = time_coords.isocalendar().week
            
            # Add week coordinate
            climatology_data = climatology_data.assign_coords(week_of_year=("time", weeks))
            
            # Compute mean by week of year
            climatology = climatology_data.groupby("week_of_year").mean("time")
            
            return climatology
            
        except Exception as e:
            logger.error(f"Failed to compute climatology: {e}")
            # Return zeros as fallback
            return xr.zeros_like(data.isel(time=0))
            
    def _compute_anomaly(self, data: xr.DataArray, climatology: xr.DataArray) -> xr.DataArray:
        """Compute anomaly from climatological mean"""
        try:
            # Get week of year for current data
            time_coords = pd.to_datetime(data.time.values)
            weeks = time_coords.isocalendar().week
            
            # Create climatology values for each time step
            climatology_values = xr.DataArray(
                np.zeros_like(data.values),
                coords=data.coords,
                dims=data.dims
            )
            
            # Map climatology values to each time step
            for i, week in enumerate(weeks):
                if week in climatology.week_of_year:
                    climatology_values[i] = climatology.sel(week_of_year=week)
                else:
                    # Use mean climatology if week not found
                    climatology_values[i] = climatology.mean()
            
            # Compute anomaly
            anomaly = data - climatology_values
            
            return anomaly
            
        except Exception as e:
            logger.error(f"Failed to compute anomaly: {e}")
            # Return zeros as fallback
            return xr.zeros_like(data)
            
    def compute_rolling_anomalies(self, dataset: xr.Dataset, variables: List[str] = None) -> xr.Dataset:
        """
        Compute rolling anomalies using a moving window
        
        Args:
            dataset: Input dataset
            variables: List of variables to compute rolling anomalies for
            
        Returns:
            Dataset with rolling anomaly features added
        """
        logger.info("Computing rolling anomalies")
        
        if variables is None:
            variables = [var for var in dataset.data_vars 
                        if dataset[var].dtype.kind in 'fc']
            
        enhanced_dataset = dataset.copy()
        
        try:
            for variable in variables:
                if variable in dataset.data_vars:
                    logger.info(f"Computing rolling anomaly for {variable}")
                    
                    # Compute rolling mean
                    rolling_mean = dataset[variable].rolling(
                        time=self.window_size, center=True
                    ).mean()
                    
                    # Compute rolling anomaly
                    rolling_anomaly = dataset[variable] - rolling_mean
                    
                    # Add to dataset
                    enhanced_dataset[f"{variable}_rolling_anomaly"] = rolling_anomaly
                    enhanced_dataset[f"{variable}_rolling_mean"] = rolling_mean
                    
            logger.info(f"Rolling anomaly computation complete")
            return enhanced_dataset
            
        except Exception as e:
            logger.error(f"Rolling anomaly computation failed: {e}")
            return dataset
            
    def compute_standardized_anomalies(self, dataset: xr.Dataset, variables: List[str] = None) -> xr.Dataset:
        """
        Compute standardized anomalies (z-scores)
        
        Args:
            dataset: Input dataset
            variables: List of variables to compute standardized anomalies for
            
        Returns:
            Dataset with standardized anomaly features added
        """
        logger.info("Computing standardized anomalies")
        
        if variables is None:
            variables = [var for var in dataset.data_vars 
                        if dataset[var].dtype.kind in 'fc']
            
        enhanced_dataset = dataset.copy()
        
        try:
            for variable in variables:
                if variable in dataset.data_vars:
                    logger.info(f"Computing standardized anomaly for {variable}")
                    
                    # Get anomaly data
                    anomaly_var = f"{variable}_anomaly"
                    if anomaly_var in dataset.data_vars:
                        anomaly_data = dataset[anomaly_var]
                    else:
                        # Compute anomaly if not already present
                        climatology = self._compute_climatology(dataset[variable])
                        anomaly_data = self._compute_anomaly(dataset[variable], climatology)
                        
                    # Compute climatological standard deviation
                    climatology_std = self._compute_climatological_std(dataset[variable])
                    
                    # Compute standardized anomaly
                    standardized_anomaly = anomaly_data / climatology_std
                    
                    # Add to dataset
                    enhanced_dataset[f"{variable}_standardized_anomaly"] = standardized_anomaly
                    
            logger.info(f"Standardized anomaly computation complete")
            return enhanced_dataset
            
        except Exception as e:
            logger.error(f"Standardized anomaly computation failed: {e}")
            return dataset
            
    def _compute_climatological_std(self, data: xr.DataArray) -> xr.DataArray:
        """Compute climatological standard deviation"""
        try:
            # Parse climatology period
            start_year, end_year = self.climatology_period.split("-")
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-12-31"
            
            # Select climatology period
            climatology_data = data.sel(time=slice(start_date, end_date))
            
            if len(climatology_data.time) == 0:
                logger.warning(f"No data found for climatology period {self.climatology_period}")
                climatology_data = data
                
            # Compute climatological std by week of year
            time_coords = pd.to_datetime(climatology_data.time.values)
            weeks = time_coords.isocalendar().week
            
            # Add week coordinate
            climatology_data = climatology_data.assign_coords(week_of_year=("time", weeks))
            
            # Compute std by week of year
            climatology_std = climatology_data.groupby("week_of_year").std("time")
            
            return climatology_std
            
        except Exception as e:
            logger.error(f"Failed to compute climatological std: {e}")
            # Return ones as fallback
            return xr.ones_like(data.isel(time=0))
            
    def save_anomaly_features(self, dataset: xr.Dataset, filename: str = None) -> Path:
        """Save anomaly features to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"anomaly_features_{timestamp}.nc"
            
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
        logger.info(f"Saved anomaly features: {output_path}")
        
        return output_path
        
    def get_anomaly_summary(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """Get summary of anomaly features"""
        summary = {
            "anomaly_variables": [],
            "climatology_variables": [],
            "statistics": {}
        }
        
        # Find anomaly variables
        for var_name in dataset.data_vars:
            if var_name.endswith("_anomaly"):
                summary["anomaly_variables"].append(var_name)
                
                # Compute statistics
                var_data = dataset[var_name]
                summary["statistics"][var_name] = {
                    "mean": float(var_data.mean()),
                    "std": float(var_data.std()),
                    "min": float(var_data.min()),
                    "max": float(var_data.max()),
                    "missing_fraction": float(var_data.isnull().sum() / var_data.size)
                }
                
            elif var_name.endswith("_climatology"):
                summary["climatology_variables"].append(var_name)
                
        return summary 