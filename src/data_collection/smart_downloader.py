"""
Smart Data Downloader with Fallback Logic

This module implements intelligent data downloading with fallback strategies
for different date ranges and data availability.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import xarray as xr
import time

from .copernicus_downloader import CopernicusDownloader
from .erddap_downloader import ERDDAPDownloader
from src.utils.config_loader import ConfigLoader, get_data_dir

logger = logging.getLogger(__name__)

class SmartDownloader:
    """Smart downloader with fallback logic for different date ranges"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the smart downloader"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_data_sources_config()
        self.spatial_bounds = self.config.get("spatial_bounds", {})
        
        # Initialize downloaders
        self.copernicus_downloader = CopernicusDownloader(config)
        self.erddap_downloader = ERDDAPDownloader(config)
        
        # Data availability ranges
        self.data_ranges = {
            "physics": {
                "start": "1993-01-01",
                "end": "2021-06-30",
                "dataset_id": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
            },
            "ocean_color": {
                "start": "2002-01-01", 
                "end": "2024-12-31",
                "dataset_id": "cmems_mod_glo_bgc_myint_0.25deg_P1D-m"
            }
        }
        
        # Retry settings
        self.retry_settings = {
            "max_retries": 3,
            "retry_delay": 5,  # seconds
            "timeout": 300     # seconds
        }
        
    def is_date_in_range(self, date_str: str, data_type: str) -> bool:
        """Check if a date is within the available range for a data type"""
        if data_type not in self.data_ranges:
            return False
            
        date = datetime.strptime(date_str, "%Y-%m-%d")
        start_date = datetime.strptime(self.data_ranges[data_type]["start"], "%Y-%m-%d")
        end_date = datetime.strptime(self.data_ranges[data_type]["end"], "%Y-%m-%d")
        
        return start_date <= date <= end_date
        
    def get_available_data_types(self, start_date: str, end_date: str) -> Dict[str, bool]:
        """Get which data types are available for the given date range"""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        availability = {}
        
        for data_type, range_info in self.data_ranges.items():
            range_start = datetime.strptime(range_info["start"], "%Y-%m-%d")
            range_end = datetime.strptime(range_info["end"], "%Y-%m-%d")
            
            # Check if there's any overlap
            overlap_start = max(start_dt, range_start)
            overlap_end = min(end_dt, range_end)
            
            availability[data_type] = overlap_start <= overlap_end
            
        return availability
        
    def download_with_retry(self, download_func, *args, **kwargs) -> Optional[xr.Dataset]:
        """Download data with retry logic"""
        for attempt in range(self.retry_settings["max_retries"]):
            try:
                logger.info(f"Download attempt {attempt + 1}/{self.retry_settings['max_retries']}")
                result = download_func(*args, **kwargs)
                
                if result is not None and hasattr(result, 'dims'):
                    logger.info(f"Download successful on attempt {attempt + 1}")
                    return result
                else:
                    logger.warning(f"Download returned None or invalid dataset on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                
            # Wait before retry (except on last attempt)
            if attempt < self.retry_settings["max_retries"] - 1:
                logger.info(f"Waiting {self.retry_settings['retry_delay']} seconds before retry...")
                time.sleep(self.retry_settings["retry_delay"])
                
        logger.error(f"All {self.retry_settings['max_retries']} download attempts failed")
        return None
        
    def download_physics_data_smart(self, start_date: str, end_date: str) -> Optional[xr.Dataset]:
        """Download physics data with smart fallback"""
        logger.info(f"Smart physics download for {start_date} to {end_date}")
        
        # Check availability
        if not self.is_date_in_range(start_date, "physics") or not self.is_date_in_range(end_date, "physics"):
            logger.warning(f"Physics data not available for {start_date} to {end_date}")
            logger.info(f"Available physics range: {self.data_ranges['physics']['start']} to {self.data_ranges['physics']['end']}")
            return None
            
        # Download with retry
        return self.download_with_retry(
            self.copernicus_downloader.download_physics_data,
            start_date, end_date
        )
        
    def download_ocean_color_data_smart(self, start_date: str, end_date: str) -> Optional[xr.Dataset]:
        """Download ocean color data with smart fallback"""
        logger.info(f"Smart ocean color download for {start_date} to {end_date}")
        
        # Check availability
        if not self.is_date_in_range(start_date, "ocean_color") or not self.is_date_in_range(end_date, "ocean_color"):
            logger.warning(f"Ocean color data not available for {start_date} to {end_date}")
            logger.info(f"Available ocean color range: {self.data_ranges['ocean_color']['start']} to {self.data_ranges['ocean_color']['end']}")
            return None
            
        # Download with retry
        return self.download_with_retry(
            self.copernicus_downloader.download_ocean_color_data,
            start_date, end_date
        )
        
    def download_atmospheric_data_smart(self, start_date: str, end_date: str) -> Optional[xr.Dataset]:
        """Download atmospheric data with smart fallback"""
        logger.info(f"Smart atmospheric download for {start_date} to {end_date}")
        
        # Try ERDDAP with retry
        return self.download_with_retry(
            self.erddap_downloader.download_weekly_atmospheric_data,
            start_date, end_date
        )
        
    def download_weekly_data_smart(self, start_date: str, end_date: str, data_type: str = "both") -> Dict[str, Optional[xr.Dataset]]:
        """Download weekly data with smart fallback logic"""
        logger.info(f"Smart weekly download for {start_date} to {end_date}")
        
        datasets = {}
        
        # Check what's available
        availability = self.get_available_data_types(start_date, end_date)
        logger.info(f"Data availability: {availability}")
        
        # Download physics data if requested and available
        if data_type in ["physics", "both"] and availability.get("physics", False):
            logger.info("Downloading physics data...")
            physics_data = self.download_physics_data_smart(start_date, end_date)
            datasets["physics"] = physics_data
        elif data_type in ["physics", "both"] and not availability.get("physics", False):
            logger.warning("Physics data requested but not available for this date range")
            datasets["physics"] = None
            
        # Download ocean color data if requested and available
        if data_type in ["ocean_color", "both"] and availability.get("ocean_color", False):
            logger.info("Downloading ocean color data...")
            ocean_color_data = self.download_ocean_color_data_smart(start_date, end_date)
            datasets["ocean_color"] = ocean_color_data
        elif data_type in ["ocean_color", "both"] and not availability.get("ocean_color", False):
            logger.warning("Ocean color data requested but not available for this date range")
            datasets["ocean_color"] = None
            
        # Download atmospheric data
        if data_type in ["atmospheric", "both"]:
            logger.info("Downloading atmospheric data...")
            atmospheric_data = self.download_atmospheric_data_smart(start_date, end_date)
            datasets["atmospheric"] = atmospheric_data
            
        return datasets
        
    def get_data_summary(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get a summary of available data for the date range"""
        availability = self.get_available_data_types(start_date, end_date)
        
        summary = {
            "date_range": {"start": start_date, "end": end_date},
            "availability": availability,
            "recommendations": []
        }
        
        # Generate recommendations
        if availability.get("physics", False) and availability.get("ocean_color", False):
            summary["recommendations"].append("Both physics and ocean color data available")
        elif availability.get("ocean_color", False):
            summary["recommendations"].append("Only ocean color data available - physics data out of range")
        elif availability.get("physics", False):
            summary["recommendations"].append("Only physics data available - ocean color data out of range")
        else:
            summary["recommendations"].append("No data available for this date range")
            
        return summary
        
    def validate_downloaded_data(self, datasets: Dict[str, Optional[xr.Dataset]]) -> Dict[str, bool]:
        """Validate downloaded datasets"""
        validation_results = {}
        
        for data_type, dataset in datasets.items():
            if dataset is None:
                validation_results[data_type] = False
                continue
                
            try:
                # Basic validation
                has_dims = hasattr(dataset, 'dims')
                has_vars = len(dataset.data_vars) > 0
                has_time = 'time' in dataset.dims
                
                validation_results[data_type] = all([has_dims, has_vars, has_time])
                
                if validation_results[data_type]:
                    logger.info(f"{data_type} validation: PASS")
                else:
                    logger.warning(f"{data_type} validation: FAIL")
                    
            except Exception as e:
                logger.error(f"{data_type} validation error: {e}")
                validation_results[data_type] = False
                
        return validation_results 