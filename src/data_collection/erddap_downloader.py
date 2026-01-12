"""
ERDDAP Data Downloader for HAB Forecasting System

This module handles data collection from ERDDAP servers for atmospheric data
and fallback ocean color data in the Seto Inland Sea region.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xarray as xr
import numpy as np
import pandas as pd
from erddapy import ERDDAP
from src.utils.config_loader import ConfigLoader, get_data_dir

logger = logging.getLogger(__name__)

class ERDDAPDownloader:
    """Downloader for ERDDAP data"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the ERDDAP downloader"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_data_sources_config()
        self.erddap_config = self.config.get("erddap", {})
        self.spatial_bounds = self.config.get("spatial_bounds", {})
        self.data_dir = get_data_dir() / "raw" / "erddap"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_gfs_data(self, 
                         start_date: str,
                         end_date: str,
                         variables: Optional[List[str]] = None) -> xr.Dataset:
        """
        Download GFS atmospheric data from PacIOOS ERDDAP
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            variables: List of variables to download (default: all configured)
            
        Returns:
            xarray Dataset with GFS data
        """
        logger.info(f"Downloading GFS data from {start_date} to {end_date}")
        
        gfs_config = self.erddap_config.get("gfs", {})
        server_url = gfs_config.get("server")
        dataset_id = gfs_config.get("dataset_id")
        default_variables = gfs_config.get("variables", [])
        
        if variables is None:
            variables = default_variables
            
        try:
            # Initialize ERDDAP connection
            e = ERDDAP(
                server=server_url,
                protocol="griddap",
                response="nc"
            )
            
            e.dataset_id = dataset_id
            
            # First, get the dataset info to understand available constraints
            try:
                e.griddap_initialize()
            except Exception as init_error:
                logger.warning(f"Failed to initialize griddap: {init_error}")
                # Continue with manual constraint setting
            
            # Set spatial constraints with proper formatting
            constraints = {}
            
            # Add time constraints if available
            if hasattr(e, 'constraints') and e.constraints:
                # Use existing constraints as base
                constraints.update(e.constraints)
            
            # Update with our specific constraints
            constraints.update({
                "latitude>=": float(self.spatial_bounds["south"]),
                "latitude<=": float(self.spatial_bounds["north"]),
                "longitude>=": float(self.spatial_bounds["west"]),
                "longitude<=": float(self.spatial_bounds["east"]),
                "time>=": start_date,
                "time<=": end_date
            })
            
            e.constraints = constraints
            e.variables = variables
            
            # Download data
            dataset = e.to_xarray()
            
            # Save to file
            output_file = self.data_dir / f"gfs_{start_date}_{end_date}.nc"
            dataset.to_netcdf(output_file)
            
            logger.info(f"Successfully downloaded GFS data: {dataset}")
            logger.info(f"Saved to: {output_file}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to download GFS data: {e}")
            raise
            
    def download_noaa_coastwatch_data(self,
                                    start_date: str,
                                    end_date: str,
                                    data_type: str = "chlorophyll") -> xr.Dataset:
        """
        Download NOAA CoastWatch fallback data
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            data_type: Type of data ("chlorophyll" or "kd490")
            
        Returns:
            xarray Dataset with NOAA CoastWatch data
        """
        logger.info(f"Downloading NOAA CoastWatch {data_type} data from {start_date} to {end_date}")
        
        noaa_config = self.erddap_config.get("noaa_coastwatch", {})
        server_url = noaa_config.get("server")
        
        if data_type == "chlorophyll":
            dataset_id = noaa_config.get("chlorophyll", {}).get("dataset_id")
            variable = noaa_config.get("chlorophyll", {}).get("variable")
        elif data_type == "kd490":
            dataset_id = noaa_config.get("kd490", {}).get("dataset_id")
            variable = noaa_config.get("kd490", {}).get("variable")
        else:
            raise ValueError(f"Unknown data type: {data_type}")
            
        try:
            # Initialize ERDDAP connection
            e = ERDDAP(
                server=server_url,
                protocol="griddap",
                response="nc"
            )
            
            e.dataset_id = dataset_id
            
            # Initialize the dataset first
            try:
                e.griddap_initialize()
            except Exception as init_error:
                logger.warning(f"Failed to initialize griddap: {init_error}")
                # Continue with manual constraint setting
            
            # Set spatial constraints with proper formatting
            constraints = {}
            
            # Add time constraints if available
            if hasattr(e, 'constraints') and e.constraints:
                # Use existing constraints as base
                constraints.update(e.constraints)
            
            # Update with our specific constraints
            constraints.update({
                "latitude>=": float(self.spatial_bounds["south"]),
                "latitude<=": float(self.spatial_bounds["north"]),
                "longitude>=": float(self.spatial_bounds["west"]),
                "longitude<=": float(self.spatial_bounds["east"]),
                "time>=": start_date,
                "time<=": end_date
            })
            
            e.constraints = constraints
            
            # Set variables
            e.variables = [variable]
            
            # Download data
            dataset = e.to_xarray()
            
            # Save to file
            output_file = self.data_dir / f"noaa_{data_type}_{start_date}_{end_date}.nc"
            dataset.to_netcdf(output_file)
            
            logger.info(f"Successfully downloaded NOAA CoastWatch {data_type} data: {dataset}")
            logger.info(f"Saved to: {output_file}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to download NOAA CoastWatch {data_type} data: {e}")
            raise
            
    def compute_wind_speed(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Compute wind speed from u and v wind components
        
        Args:
            dataset: Dataset containing ugrd10m and vgrd10m variables
            
        Returns:
            Dataset with wind_speed variable added
        """
        logger.info("Computing wind speed from u/v components")
        
        if "ugrd10m" in dataset.data_vars and "vgrd10m" in dataset.data_vars:
            # Compute wind speed: sqrt(u^2 + v^2)
            wind_speed = np.sqrt(dataset["ugrd10m"]**2 + dataset["vgrd10m"]**2)
            dataset["wind_speed"] = wind_speed
            dataset["wind_speed"].attrs["units"] = "m/s"
            dataset["wind_speed"].attrs["long_name"] = "Wind Speed"
            dataset["wind_speed"].attrs["description"] = "Wind speed computed from u and v components"
            
            logger.info("Wind speed computation completed")
        else:
            logger.warning("u and v wind components not found in dataset")
            
        return dataset
        
    def convert_precipitation(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Convert precipitation from kg m^-2 s^-1 to mm day^-1
        
        Args:
            dataset: Dataset containing pratesfc variable
            
        Returns:
            Dataset with precip_mmday variable added
        """
        logger.info("Converting precipitation units")
        
        if "pratesfc" in dataset.data_vars:
            # Convert: kg m^-2 s^-1 * 86400 s/day = mm day^-1
            precip_mmday = dataset["pratesfc"] * 86400
            dataset["precip_mmday"] = precip_mmday
            dataset["precip_mmday"].attrs["units"] = "mm/day"
            dataset["precip_mmday"].attrs["long_name"] = "Precipitation Rate"
            dataset["precip_mmday"].attrs["description"] = "Precipitation rate in mm per day"
            
            logger.info("Precipitation conversion completed")
        else:
            logger.warning("Precipitation variable (pratesfc) not found in dataset")
            
        return dataset
        
    def aggregate_to_weekly(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Aggregate daily data to weekly means
        
        Args:
            dataset: Daily dataset
            
        Returns:
            Weekly aggregated dataset
        """
        logger.info("Aggregating daily data to weekly means")
        
        # Group by week and compute mean
        weekly_dataset = dataset.groupby("time.week").mean()
        
        # Update time coordinates
        weekly_dataset = weekly_dataset.rename({"week": "time"})
        
        logger.info("Weekly aggregation completed")
        return weekly_dataset
        
    def download_weekly_atmospheric_data(self,
                                       start_date: str,
                                       end_date: str) -> xr.Dataset:
        """
        Download and process weekly atmospheric data
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Processed weekly atmospheric dataset
        """
        logger.info(f"Downloading weekly atmospheric data from {start_date} to {end_date}")
        
        try:
            # Download GFS data
            gfs_dataset = self.download_gfs_data(start_date, end_date)
            
            # Compute derived variables
            gfs_dataset = self.compute_wind_speed(gfs_dataset)
            gfs_dataset = self.convert_precipitation(gfs_dataset)
            
            # Aggregate to weekly
            weekly_dataset = self.aggregate_to_weekly(gfs_dataset)
            
            # Save weekly dataset
            output_file = self.data_dir / f"weekly_atmospheric_{start_date}_{end_date}.nc"
            weekly_dataset.to_netcdf(output_file)
            
            logger.info(f"Weekly atmospheric data saved to: {output_file}")
            return weekly_dataset
            
        except Exception as e:
            logger.error(f"Failed to download weekly atmospheric data: {e}")
            # Return None instead of raising to allow fallback
            return None
            
    def get_fallback_data(self,
                         start_date: str,
                         end_date: str,
                         primary_data_failed: bool = False) -> Dict[str, xr.Dataset]:
        """
        Get fallback data when primary sources fail
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            primary_data_failed: Whether primary data collection failed
            
        Returns:
            Dictionary with fallback datasets
        """
        logger.info("Attempting to download fallback data")
        
        datasets = {}
        
        if primary_data_failed:
            try:
                # Try NOAA CoastWatch chlorophyll data
                chl_dataset = self.download_noaa_coastwatch_data(start_date, end_date, "chlorophyll")
                datasets["chlorophyll_fallback"] = chl_dataset
                logger.info("Successfully downloaded chlorophyll fallback data")
            except Exception as e:
                logger.error(f"Failed to download chlorophyll fallback data: {e}")
                
            try:
                # Try NOAA CoastWatch kd490 data
                kd_dataset = self.download_noaa_coastwatch_data(start_date, end_date, "kd490")
                datasets["kd490_fallback"] = kd_dataset
                logger.info("Successfully downloaded kd490 fallback data")
            except Exception as e:
                logger.error(f"Failed to download kd490 fallback data: {e}")
                
        return datasets
        
    def validate_data_quality(self, dataset: xr.Dataset, data_type: str) -> bool:
        """
        Validate downloaded data quality
        
        Args:
            dataset: Downloaded dataset
            data_type: Type of data ("gfs", "noaa_chlorophyll", "noaa_kd490")
            
        Returns:
            True if data quality is acceptable
        """
        logger.info(f"Validating {data_type} data quality")
        
        # Check for missing data
        total_pixels = dataset.dims.get("latitude", 0) * dataset.dims.get("longitude", 0)
        if total_pixels == 0:
            logger.error("No spatial dimensions found in dataset")
            return False
            
        # Check data coverage
        for var in dataset.data_vars:
            missing_ratio = dataset[var].isnull().sum().item() / total_pixels
            if missing_ratio > 0.5:  # More than 50% missing
                logger.warning(f"High missing data ratio for {var}: {missing_ratio:.2%}")
                
        # Check value ranges
        quality_thresholds = self.config.get("quality_thresholds", {})
        
        for var in dataset.data_vars:
            if var in quality_thresholds:
                thresholds = quality_thresholds[var]
                min_val = thresholds.get("min_valid")
                max_val = thresholds.get("max_valid")
                
                if min_val is not None:
                    below_min = (dataset[var] < min_val).sum().item()
                    if below_min > 0:
                        logger.warning(f"{var}: {below_min} values below minimum threshold")
                        
                if max_val is not None:
                    above_max = (dataset[var] > max_val).sum().item()
                    if above_max > 0:
                        logger.warning(f"{var}: {above_max} values above maximum threshold")
                        
        logger.info(f"{data_type} data quality validation completed")
        return True
        
    def get_available_datasets(self) -> Dict[str, Dict]:
        """Get information about available datasets"""
        return {
            "gfs": {
                "server": self.erddap_config.get("gfs", {}).get("server"),
                "dataset_id": self.erddap_config.get("gfs", {}).get("dataset_id"),
                "variables": self.erddap_config.get("gfs", {}).get("variables", []),
                "temporal_resolution": self.erddap_config.get("gfs", {}).get("temporal_resolution")
            },
            "noaa_coastwatch": {
                "server": self.erddap_config.get("noaa_coastwatch", {}).get("server"),
                "chlorophyll": {
                    "dataset_id": self.erddap_config.get("noaa_coastwatch", {}).get("chlorophyll", {}).get("dataset_id"),
                    "variable": self.erddap_config.get("noaa_coastwatch", {}).get("chlorophyll", {}).get("variable")
                },
                "kd490": {
                    "dataset_id": self.erddap_config.get("noaa_coastwatch", {}).get("kd490", {}).get("dataset_id"),
                    "variable": self.erddap_config.get("noaa_coastwatch", {}).get("kd490", {}).get("variable")
                },
                "temporal_resolution": self.erddap_config.get("noaa_coastwatch", {}).get("temporal_resolution"),
                "spatial_resolution": self.erddap_config.get("noaa_coastwatch", {}).get("spatial_resolution")
            }
        }
        
if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    downloader = ERDDAPDownloader()
    
    # Download last week's atmospheric data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    try:
        weekly_data = downloader.download_weekly_atmospheric_data(start_date, end_date)
        print(f"Downloaded weekly atmospheric data: {weekly_data}")
        
    except Exception as e:
        print(f"Download failed: {e}")
        
        # Try fallback data
        fallback_data = downloader.get_fallback_data(start_date, end_date, primary_data_failed=True)
        print(f"Fallback data: {fallback_data}") 