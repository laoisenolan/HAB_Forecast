"""
Data Regridder for HAB Forecasting System

This module handles regridding of environmental data to common resolutions.
"""

import xarray as xr
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataRegridder:
    """
    Regrids environmental data to common spatial resolution.
    
    This class handles the spatial interpolation and regridding of
    various environmental datasets to a common grid for HAB forecasting.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the DataRegridder.
        
        Args:
            config: Configuration dictionary with regridding parameters
        """
        self.config = config
        self.target_resolution = config.get('target_resolution', 0.1)  # degrees
        self.target_lat_range = config.get('target_lat_range', [30, 50])
        self.target_lon_range = config.get('target_lon_range', [-100, -60])
        self.interpolation_method = config.get('interpolation_method', 'linear')
        
    def regrid_dataset(self, dataset: xr.Dataset, 
                      lat_var: str = 'lat',
                      lon_var: str = 'lon') -> xr.Dataset:
        """
        Regrid a dataset to the target resolution.
        
        Args:
            dataset: xarray Dataset to regrid
            lat_var: Name of latitude variable
            lon_var: Name of longitude variable
            
        Returns:
            Regridded xarray Dataset
        """
        try:
            # Create target grid
            target_lat = np.arange(
                self.target_lat_range[0],
                self.target_lat_range[1] + self.target_resolution,
                self.target_resolution
            )
            target_lon = np.arange(
                self.target_lon_range[0],
                self.target_lon_range[1] + self.target_resolution,
                self.target_resolution
            )
            
            # Create target coordinates
            target_coords = {
                lat_var: target_lat,
                lon_var: target_lon
            }
            
            # Regrid each variable
            regridded_data = {}
            
            for var_name in dataset.data_vars:
                if lat_var in dataset[var_name].dims and lon_var in dataset[var_name].dims:
                    logger.info(f"Regridding variable: {var_name}")
                    
                    # Interpolate to target grid
                    regridded_var = dataset[var_name].interp(
                        coords=target_coords,
                        method=self.interpolation_method
                    )
                    
                    regridded_data[var_name] = regridded_var
            
            # Create new dataset
            result = xr.Dataset(regridded_data)
            
            logger.info(f"Regridded dataset to {len(target_lat)}x{len(target_lon)} grid")
            return result
            
        except Exception as e:
            logger.error(f"Error in dataset regridding: {e}")
            raise
    
    def regrid_dataarray(self, dataarray: xr.DataArray,
                         lat_var: str = 'lat',
                         lon_var: str = 'lon') -> xr.DataArray:
        """
        Regrid a single DataArray to the target resolution.
        
        Args:
            dataarray: xarray DataArray to regrid
            lat_var: Name of latitude variable
            lon_var: Name of longitude variable
            
        Returns:
            Regridded xarray DataArray
        """
        try:
            # Create target grid
            target_lat = np.arange(
                self.target_lat_range[0],
                self.target_lat_range[1] + self.target_resolution,
                self.target_resolution
            )
            target_lon = np.arange(
                self.target_lon_range[0],
                self.target_lon_range[1] + self.target_resolution,
                self.target_resolution
            )
            
            # Create target coordinates
            target_coords = {
                lat_var: target_lat,
                lon_var: target_lon
            }
            
            # Interpolate to target grid
            result = dataarray.interp(
                coords=target_coords,
                method=self.interpolation_method
            )
            
            logger.info(f"Regridded DataArray to {len(target_lat)}x{len(target_lon)} grid")
            return result
            
        except Exception as e:
            logger.error(f"Error in DataArray regridding: {e}")
            raise
    
    def create_common_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create the common target grid.
        
        Returns:
            Tuple of (latitude_array, longitude_array)
        """
        target_lat = np.arange(
            self.target_lat_range[0],
            self.target_lat_range[1] + self.target_resolution,
            self.target_resolution
        )
        target_lon = np.arange(
            self.target_lon_range[0],
            self.target_lon_range[1] + self.target_resolution,
            self.target_resolution
        )
        
        return target_lat, target_lon
    
    def validate_grid(self, dataset: xr.Dataset,
                     lat_var: str = 'lat',
                     lon_var: str = 'lon') -> bool:
        """
        Validate if a dataset matches the target grid.
        
        Args:
            dataset: xarray Dataset to validate
            lat_var: Name of latitude variable
            lon_var: Name of longitude variable
            
        Returns:
            True if grid matches target, False otherwise
        """
        try:
            target_lat, target_lon = self.create_common_grid()
            
            dataset_lat = dataset[lat_var].values
            dataset_lon = dataset[lon_var].values
            
            lat_match = np.allclose(dataset_lat, target_lat, atol=1e-6)
            lon_match = np.allclose(dataset_lon, target_lon, atol=1e-6)
            
            return lat_match and lon_match
            
        except Exception as e:
            logger.error(f"Error in grid validation: {e}")
            return False 