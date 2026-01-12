"""
Spatial Processor for HAB Forecasting System

This module handles spatial operations including regridding, masking,
and spatial feature extraction for the ML pipeline.
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

class SpatialProcessor:
    """Handles spatial processing operations"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the spatial processor"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_processing_config()
        self.data_dir = get_data_dir()
        
        # Get spatial settings from config
        self.spatial_config = self.config.get("spatial", {})
        self.target_resolution_km = self.spatial_config.get("target_resolution_km", 2.0)
        self.interpolation_method = self.spatial_config.get("interpolation_method", "bilinear")
        
        # Output directory
        self.output_dir = self.data_dir / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def regrid_dataset(self, dataset: xr.Dataset, target_resolution: float = None) -> xr.Dataset:
        """
        Regrid dataset to target resolution
        
        Args:
            dataset: Input dataset
            target_resolution: Target resolution in degrees (if None, uses config)
            
        Returns:
            Regridded dataset
        """
        if target_resolution is None:
            target_resolution = self.target_resolution_km / 111.0  # Convert km to degrees
            
        logger.info(f"Regridding dataset to {target_resolution:.4f} degrees")
        
        try:
            # Create target grid
            target_grid = self._create_target_grid(dataset, target_resolution)
            
            # Regrid dataset
            regridded = self._perform_regridding(dataset, target_grid)
            
            logger.info(f"Regridding complete. New shape: {dict(regridded.dims)}")
            return regridded
            
        except Exception as e:
            logger.error(f"Regridding failed: {e}")
            return dataset
            
    def _create_target_grid(self, dataset: xr.Dataset, target_resolution: float) -> xr.Dataset:
        """Create target grid for regridding"""
        # Get spatial bounds
        lat_min = float(dataset.latitude.min())
        lat_max = float(dataset.latitude.max())
        lon_min = float(dataset.longitude.min())
        lon_max = float(dataset.longitude.max())
        
        # Create new coordinates
        new_lat = np.arange(lat_min, lat_max + target_resolution, target_resolution)
        new_lon = np.arange(lon_min, lon_max + target_resolution, target_resolution)
        
        # Create target grid
        target_grid = xr.Dataset(
            coords={
                "latitude": new_lat,
                "longitude": new_lon
            }
        )
        
        return target_grid
        
    def _perform_regridding(self, dataset: xr.Dataset, target_grid: xr.Dataset) -> xr.Dataset:
        """Perform the actual regridding operation"""
        # Use xarray's interp method for regridding
        regridded = dataset.interp(
            latitude=target_grid.latitude,
            longitude=target_grid.longitude,
            method=self.interpolation_method
        )
        
        return regridded
        
    def apply_spatial_mask(self, dataset: xr.Dataset, mask_type: str = "ocean") -> xr.Dataset:
        """
        Apply spatial mask to dataset
        
        Args:
            dataset: Input dataset
            mask_type: Type of mask ("ocean", "land", "custom")
            
        Returns:
            Masked dataset
        """
        logger.info(f"Applying {mask_type} mask")
        
        try:
            if mask_type == "ocean":
                masked_dataset = self._apply_ocean_mask(dataset)
            elif mask_type == "land":
                masked_dataset = self._apply_land_mask(dataset)
            else:
                logger.warning(f"Unknown mask type: {mask_type}")
                return dataset
                
            logger.info("Spatial masking complete")
            return masked_dataset
            
        except Exception as e:
            logger.error(f"Spatial masking failed: {e}")
            return dataset
            
    def _apply_ocean_mask(self, dataset: xr.Dataset) -> xr.Dataset:
        """Apply ocean mask to dataset"""
        # Simple ocean mask based on latitude/longitude bounds
        # In a real implementation, you would use a proper land-sea mask
        
        # Create a simple mask for demonstration
        # This is a placeholder - in practice you'd use a proper land-sea mask file
        mask = xr.ones_like(dataset.latitude) * xr.ones_like(dataset.longitude)
        
        # Apply mask to all variables
        masked_dataset = dataset.copy()
        for var_name in masked_dataset.data_vars:
            if "latitude" in masked_dataset[var_name].dims and "longitude" in masked_dataset[var_name].dims:
                masked_dataset[var_name] = masked_dataset[var_name] * mask
                
        return masked_dataset
        
    def _apply_land_mask(self, dataset: xr.Dataset) -> xr.Dataset:
        """Apply land mask to dataset"""
        # Similar to ocean mask but inverted
        mask = xr.ones_like(dataset.latitude) * xr.ones_like(dataset.longitude)
        
        masked_dataset = dataset.copy()
        for var_name in masked_dataset.data_vars:
            if "latitude" in masked_dataset[var_name].dims and "longitude" in masked_dataset[var_name].dims:
                masked_dataset[var_name] = masked_dataset[var_name] * mask
                
        return masked_dataset
        
    def extract_spatial_features(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Extract spatial features from dataset
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with additional spatial features
        """
        logger.info("Extracting spatial features")
        
        try:
            enhanced_dataset = dataset.copy()
            
            # Extract spatial features for each variable
            for var_name in dataset.data_vars:
                if "latitude" in dataset[var_name].dims and "longitude" in dataset[var_name].dims:
                    enhanced_dataset = self._extract_variable_spatial_features(
                        enhanced_dataset, var_name
                    )
                    
            logger.info("Spatial feature extraction complete")
            return enhanced_dataset
            
        except Exception as e:
            logger.error(f"Spatial feature extraction failed: {e}")
            return dataset
            
    def _extract_variable_spatial_features(self, dataset: xr.Dataset, var_name: str) -> xr.Dataset:
        """Extract spatial features for a specific variable"""
        # Compute spatial gradients
        try:
            # Latitude gradient
            lat_grad = dataset[var_name].differentiate("latitude")
            dataset[f"{var_name}_lat_gradient"] = lat_grad
            
            # Longitude gradient
            lon_grad = dataset[var_name].differentiate("longitude")
            dataset[f"{var_name}_lon_gradient"] = lon_grad
            
            # Gradient magnitude
            grad_magnitude = np.sqrt(lat_grad**2 + lon_grad**2)
            dataset[f"{var_name}_gradient_magnitude"] = grad_magnitude
            
        except Exception as e:
            logger.warning(f"Failed to compute gradients for {var_name}: {e}")
            
        # Compute spatial statistics
        try:
            # Spatial mean (rolling window)
            spatial_mean = dataset[var_name].rolling(
                latitude=3, longitude=3, center=True
            ).mean()
            dataset[f"{var_name}_spatial_mean"] = spatial_mean
            
            # Spatial standard deviation
            spatial_std = dataset[var_name].rolling(
                latitude=3, longitude=3, center=True
            ).std()
            dataset[f"{var_name}_spatial_std"] = spatial_std
            
        except Exception as e:
            logger.warning(f"Failed to compute spatial statistics for {var_name}: {e}")
            
        return dataset
        
    def compute_spatial_averages(self, dataset: xr.Dataset, region_bounds: Dict = None) -> xr.Dataset:
        """
        Compute spatial averages for the region
        
        Args:
            dataset: Input dataset
            region_bounds: Region bounds (if None, uses config)
            
        Returns:
            Dataset with spatial averages
        """
        if region_bounds is None:
            region_bounds = self.spatial_config.get("region_bounds", {})
            
        logger.info("Computing spatial averages")
        
        try:
            # Select region
            region_dataset = dataset.sel(
                latitude=slice(region_bounds.get("south", 34.0), region_bounds.get("north", 35.0)),
                longitude=slice(region_bounds.get("west", 132.0), region_bounds.get("east", 135.0))
            )
            
            # Compute spatial averages
            spatial_avg_dataset = region_dataset.mean(dim=["latitude", "longitude"])
            
            # Add region information
            spatial_avg_dataset.attrs["region_bounds"] = region_bounds
            spatial_avg_dataset.attrs["spatial_averaging"] = "mean_over_region"
            
            logger.info("Spatial averaging complete")
            return spatial_avg_dataset
            
        except Exception as e:
            logger.error(f"Spatial averaging failed: {e}")
            return dataset
            
    def save_spatial_processed_dataset(self, dataset: xr.Dataset, processing_type: str) -> Path:
        """Save spatially processed dataset"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"spatial_{processing_type}_{timestamp}.nc"
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
        logger.info(f"Saved spatial processed dataset: {output_path}")
        
        return output_path 