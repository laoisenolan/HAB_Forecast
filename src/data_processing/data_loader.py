"""
Data Loader for HAB Forecasting System

This module handles loading and initial processing of historical datasets
collected in Phase 2 for use in the machine learning pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.config_loader import ConfigLoader, get_data_dir

logger = logging.getLogger(__name__)

class DataLoader:
    """Loads and prepares historical datasets for ML processing"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the data loader"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_processing_config()
        self.data_dir = get_data_dir()
        
        # Data directories
        self.historical_dir = self.data_dir / "historical"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure directories exist
        for directory in [self.processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
    def get_available_datasets(self) -> Dict[str, List[Path]]:
        """Get list of available historical datasets"""
        available_datasets = {}
        
        # Check historical directory
        if self.historical_dir.exists():
            for data_type_dir in self.historical_dir.iterdir():
                if data_type_dir.is_dir():
                    files = list(data_type_dir.glob("*.nc"))
                    if files:
                        available_datasets[data_type_dir.name] = files
                        
        # Check raw directory
        if self.raw_dir.exists():
            for data_type_dir in self.raw_dir.iterdir():
                if data_type_dir.is_dir():
                    files = list(data_type_dir.glob("*.nc"))
                    if files:
                        available_datasets[f"raw_{data_type_dir.name}"] = files
                        
        return available_datasets
        
    def load_historical_dataset(self, data_type: str, start_date: str = None, end_date: str = None) -> Optional[xr.Dataset]:
        """
        Load historical dataset for a specific data type
        
        Args:
            data_type: Type of data to load ("physics", "ocean_color", "atmospheric")
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            xarray Dataset or None if not found
        """
        logger.info(f"Loading historical dataset: {data_type}")
        
        # Check historical directory first
        historical_path = self.historical_dir / data_type
        if historical_path.exists():
            files = list(historical_path.glob("*.nc"))
            if files:
                # Load the most recent file
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                logger.info(f"Loading from historical: {latest_file}")
                return self._load_and_filter_dataset(latest_file, start_date, end_date)
                
        # Check raw directory as fallback
        raw_path = self.raw_dir / data_type
        if raw_path.exists():
            files = list(raw_path.glob("*.nc"))
            if files:
                # Load the most recent file
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                logger.info(f"Loading from raw: {latest_file}")
                return self._load_and_filter_dataset(latest_file, start_date, end_date)
                
        logger.warning(f"No dataset found for {data_type}")
        return None
        
    def _load_and_filter_dataset(self, file_path: Path, start_date: str = None, end_date: str = None) -> xr.Dataset:
        """Load dataset and apply date filtering"""
        try:
            dataset = xr.open_dataset(file_path)
            
            # Apply date filtering if specified
            if start_date or end_date:
                if "time" in dataset.dims:
                    if start_date:
                        dataset = dataset.sel(time=slice(start_date, None))
                    if end_date:
                        dataset = dataset.sel(time=slice(None, end_date))
                        
            logger.info(f"Loaded dataset: {dict(dataset.dims)}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset {file_path}: {e}")
            return None
            
    def load_all_historical_data(self, start_date: str = None, end_date: str = None) -> Dict[str, xr.Dataset]:
        """
        Load all available historical datasets
        
        Args:
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            Dictionary with loaded datasets
        """
        logger.info("Loading all historical datasets")
        
        datasets = {}
        data_types = ["physics", "ocean_color", "atmospheric"]
        
        for data_type in data_types:
            dataset = self.load_historical_dataset(data_type, start_date, end_date)
            if dataset is not None:
                datasets[data_type] = dataset
                
        logger.info(f"Loaded {len(datasets)} datasets")
        return datasets
        
    def get_dataset_info(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """Get comprehensive information about a dataset"""
        info = {
            "dimensions": dict(dataset.dims),
            "variables": list(dataset.data_vars.keys()),
            "coordinates": list(dataset.coords.keys()),
            "attrs": dict(dataset.attrs)
        }
        
        # Add time range if available
        if "time" in dataset.dims:
            info["time_range"] = {
                "start": str(dataset.time.min().values),
                "end": str(dataset.time.max().values),
                "total_timesteps": len(dataset.time)
            }
            
        # Add spatial information if available
        if "latitude" in dataset.dims and "longitude" in dataset.dims:
            info["spatial_range"] = {
                "lat_min": float(dataset.latitude.min()),
                "lat_max": float(dataset.latitude.max()),
                "lon_min": float(dataset.longitude.min()),
                "lon_max": float(dataset.longitude.max()),
                "spatial_resolution": {
                    "lat": float(dataset.latitude.diff('latitude').mean()),
                    "lon": float(dataset.longitude.diff('longitude').mean())
                }
            }
            
        return info
        
    def validate_dataset(self, dataset: xr.Dataset, data_type: str) -> Dict[str, Any]:
        """Validate dataset quality and completeness"""
        validation_results = {
            "data_type": data_type,
            "is_valid": True,
            "issues": [],
            "statistics": {}
        }
        
        try:
            # Check for required dimensions
            required_dims = ["time", "latitude", "longitude"]
            for dim in required_dims:
                if dim not in dataset.dims:
                    validation_results["is_valid"] = False
                    validation_results["issues"].append(f"Missing required dimension: {dim}")
                    
            # Check for data variables
            if len(dataset.data_vars) == 0:
                validation_results["is_valid"] = False
                validation_results["issues"].append("No data variables found")
                
            # Check for missing data
            for var_name, var_data in dataset.data_vars.items():
                missing_fraction = float(var_data.isnull().sum() / var_data.size)
                validation_results["statistics"][var_name] = {
                    "missing_fraction": missing_fraction,
                    "has_missing": missing_fraction > 0
                }
                
                if missing_fraction > 0.5:  # More than 50% missing
                    validation_results["issues"].append(f"High missing data in {var_name}: {missing_fraction:.2%}")
                    
            # Check time consistency
            if "time" in dataset.dims:
                time_diffs = dataset.time.diff('time')
                if len(time_diffs) > 0:
                    mean_timedelta = time_diffs.mean()
                    validation_results["statistics"]["time_consistency"] = {
                        "mean_timedelta": str(mean_timedelta.values),
                        "is_regular": bool(time_diffs.std() < pd.Timedelta('1D'))
                    }
                    
        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["issues"].append(f"Validation error: {e}")
            
        return validation_results
        
    def save_processed_dataset(self, dataset: xr.Dataset, data_type: str, 
                             processing_info: Dict[str, Any] = None) -> Path:
        """
        Save processed dataset
        
        Args:
            dataset: Dataset to save
            data_type: Type of data
            processing_info: Information about processing steps
            
        Returns:
            Path to saved file
        """
        # Create output directory
        output_dir = self.processed_dir / data_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data_type}_processed_{timestamp}.nc"
        output_path = output_dir / filename
        
        # Add processing metadata
        if processing_info:
            dataset.attrs["processing_info"] = processing_info
        dataset.attrs["processing_date"] = timestamp
        dataset.attrs["data_type"] = data_type
        
        # Save dataset
        dataset.to_netcdf(output_path)
        logger.info(f"Saved processed dataset: {output_path}")
        
        return output_path 