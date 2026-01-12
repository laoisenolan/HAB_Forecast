"""
Data Integrator for HAB Forecasting System

This module handles integration of multiple datasets into a unified format
suitable for machine learning processing.
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

class DataIntegrator:
    """Integrates multiple datasets into unified format"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the data integrator"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_processing_config()
        self.data_dir = get_data_dir()
        
        # Get variable mapping from config
        self.variable_mapping = self.config.get("data_sources", {}).get("variable_mapping", {})
        
        # Output directory
        self.output_dir = self.data_dir / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def integrate_datasets(self, datasets: Dict[str, xr.Dataset]) -> Optional[xr.Dataset]:
        """
        Integrate multiple datasets into a unified dataset
        
        Args:
            datasets: Dictionary of datasets by type
            
        Returns:
            Integrated xarray Dataset or None if integration fails
        """
        logger.info("Starting dataset integration")
        
        if not datasets:
            logger.warning("No datasets provided for integration")
            return None
            
        try:
            # Step 1: Standardize variable names
            standardized_datasets = self._standardize_variables(datasets)
            
            # Step 2: Align spatial coordinates
            aligned_datasets = self._align_spatial_coordinates(standardized_datasets)
            
            # Step 3: Align temporal coordinates
            temporal_datasets = self._align_temporal_coordinates(aligned_datasets)
            
            # Step 4: Merge datasets
            integrated_dataset = self._merge_datasets(temporal_datasets)
            
            # Step 5: Add metadata
            integrated_dataset = self._add_integration_metadata(integrated_dataset, datasets)
            
            logger.info(f"Integration complete. Final dataset: {dict(integrated_dataset.dims)}")
            return integrated_dataset
            
        except Exception as e:
            logger.error(f"Dataset integration failed: {e}")
            return None
            
    def _standardize_variables(self, datasets: Dict[str, xr.Dataset]) -> Dict[str, xr.Dataset]:
        """Standardize variable names across datasets"""
        standardized = {}
        
        for data_type, dataset in datasets.items():
            logger.info(f"Standardizing variables for {data_type}")
            
            # Create a copy to avoid modifying original
            std_dataset = dataset.copy()
            
            # Apply variable mapping
            for standard_name, possible_names in self.variable_mapping.items():
                for var_name in possible_names:
                    if var_name in std_dataset.data_vars:
                        # Rename to standard name
                        std_dataset = std_dataset.rename({var_name: standard_name})
                        logger.info(f"Renamed {var_name} to {standard_name}")
                        break
                        
            # Add data type prefix to remaining variables
            for var_name in list(std_dataset.data_vars.keys()):
                if var_name not in [v for names in self.variable_mapping.values() for v in names]:
                    new_name = f"{data_type}_{var_name}"
                    std_dataset = std_dataset.rename({var_name: new_name})
                    logger.info(f"Added prefix to {var_name} -> {new_name}")
                    
            standardized[data_type] = std_dataset
            
        return standardized
        
    def _align_spatial_coordinates(self, datasets: Dict[str, xr.Dataset]) -> Dict[str, xr.Dataset]:
        """Align spatial coordinates across datasets"""
        aligned = {}
        
        # Get target spatial bounds from config
        bounds = self.config.get("spatial", {}).get("region_bounds", {})
        target_bounds = {
            "lat_min": bounds.get("south", 34.0),
            "lat_max": bounds.get("north", 35.0),
            "lon_min": bounds.get("west", 132.0),
            "lon_max": bounds.get("east", 135.0)
        }
        
        logger.info(f"Aligning spatial coordinates to bounds: {target_bounds}")
        
        for data_type, dataset in datasets.items():
            logger.info(f"Aligning spatial coordinates for {data_type}")
            
            # Select region of interest
            aligned_dataset = dataset.sel(
                latitude=slice(target_bounds["lat_min"], target_bounds["lat_max"]),
                longitude=slice(target_bounds["lon_min"], target_bounds["lon_max"])
            )
            
            # Ensure consistent coordinate names
            if "lat" in aligned_dataset.dims:
                aligned_dataset = aligned_dataset.rename({"lat": "latitude"})
            if "lon" in aligned_dataset.dims:
                aligned_dataset = aligned_dataset.rename({"lon": "longitude"})
                
            aligned[data_type] = aligned_dataset
            
        return aligned
        
    def _align_temporal_coordinates(self, datasets: Dict[str, xr.Dataset]) -> Dict[str, xr.Dataset]:
        """Align temporal coordinates across datasets"""
        aligned = {}
        
        # Find common time range
        time_ranges = []
        for data_type, dataset in datasets.items():
            if "time" in dataset.dims:
                time_ranges.append((dataset.time.min(), dataset.time.max()))
                
        if not time_ranges:
            logger.warning("No temporal dimension found in datasets")
            return datasets
            
        # Find intersection of time ranges
        common_start = max(t[0] for t in time_ranges)
        common_end = min(t[1] for t in time_ranges)
        
        logger.info(f"Common time range: {common_start} to {common_end}")
        
        for data_type, dataset in datasets.items():
            logger.info(f"Aligning temporal coordinates for {data_type}")
            
            if "time" in dataset.dims:
                # Select common time range
                aligned_dataset = dataset.sel(time=slice(common_start, common_end))
                
                # Ensure time is sorted
                aligned_dataset = aligned_dataset.sortby("time")
                
                aligned[data_type] = aligned_dataset
            else:
                aligned[data_type] = dataset
                
        return aligned
        
    def _merge_datasets(self, datasets: Dict[str, xr.Dataset]) -> xr.Dataset:
        """Merge multiple datasets into a single dataset"""
        logger.info("Merging datasets")
        
        # Start with the first dataset
        merged = None
        dataset_list = list(datasets.values())
        
        if not dataset_list:
            raise ValueError("No datasets to merge")
            
        merged = dataset_list[0]
        
        # Merge with remaining datasets
        for dataset in dataset_list[1:]:
            try:
                # Use outer join to keep all data
                merged = xr.merge([merged, dataset], join="outer")
                logger.info(f"Merged dataset. New shape: {dict(merged.dims)}")
            except Exception as e:
                logger.warning(f"Failed to merge dataset: {e}")
                # Try with different join method
                try:
                    merged = xr.merge([merged, dataset], join="left")
                    logger.info("Used left join as fallback")
                except Exception as e2:
                    logger.error(f"Failed to merge even with left join: {e2}")
                    
        return merged
        
    def _add_integration_metadata(self, dataset: xr.Dataset, original_datasets: Dict[str, xr.Dataset]) -> xr.Dataset:
        """Add metadata about the integration process"""
        # Add integration metadata
        dataset.attrs["integration_date"] = datetime.now().isoformat()
        dataset.attrs["integration_method"] = "outer_join"
        dataset.attrs["source_datasets"] = list(original_datasets.keys())
        
        # Add dataset information
        for data_type, orig_dataset in original_datasets.items():
            dataset.attrs[f"{data_type}_variables"] = list(orig_dataset.data_vars.keys())
            dataset.attrs[f"{data_type}_dimensions"] = dict(orig_dataset.dims)
            
        return dataset
        
    def save_integrated_dataset(self, dataset: xr.Dataset, filename: str = None) -> Path:
        """Save the integrated dataset"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"integrated_dataset_{timestamp}.nc"
            
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
        logger.info(f"Saved integrated dataset: {output_path}")
        
        return output_path
        
    def get_integration_summary(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """Get summary of integrated dataset"""
        summary = {
            "dimensions": dict(dataset.dims),
            "variables": list(dataset.data_vars.keys()),
            "coordinates": list(dataset.coords.keys()),
            "attributes": dict(dataset.attrs)
        }
        
        # Add variable statistics
        summary["variable_statistics"] = {}
        for var_name, var_data in dataset.data_vars.items():
            if var_data.dtype.kind in 'fc':  # Float or complex
                summary["variable_statistics"][var_name] = {
                    "min": float(var_data.min()),
                    "max": float(var_data.max()),
                    "mean": float(var_data.mean()),
                    "std": float(var_data.std()),
                    "missing_fraction": float(var_data.isnull().sum() / var_data.size)
                }
                
        return summary 