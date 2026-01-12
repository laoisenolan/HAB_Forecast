"""
Label Generator for HAB Forecasting System

This module handles the generation of HAB event labels from environmental
data, particularly chlorophyll-a anomalies, for the ML pipeline.
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

class LabelGenerator:
    """Generates HAB event labels from environmental data"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the label generator"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_features_config()
        self.data_dir = get_data_dir()
        
        # Get label settings from config
        self.label_config = self.config.get("labels", {})
        self.chlorophyll_threshold = self.label_config.get("chlorophyll_threshold", 2.0)
        self.anomaly_threshold = self.label_config.get("anomaly_threshold", 1.5)
        self.percentile_threshold = self.label_config.get("percentile_threshold", 75)
        self.min_event_duration = self.label_config.get("min_event_duration", 1)  # weeks
        self.min_spatial_coverage = self.label_config.get("min_spatial_coverage", 0.1)  # 10% of region
        
        # Output directory
        self.output_dir = self.data_dir / "labels"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_hab_labels(self, dataset: xr.Dataset, method: str = "anomaly") -> xr.Dataset:
        """
        Generate HAB event labels from dataset
        
        Args:
            dataset: Input dataset with environmental variables
            method: Method for label generation ("anomaly", "threshold", "combined")
            
        Returns:
            Dataset with HAB labels added
        """
        logger.info(f"Generating HAB labels using method: {method}")
        
        enhanced_dataset = dataset.copy()
        
        try:
            if method == "anomaly":
                labels = self._generate_anomaly_labels(dataset)
            elif method == "threshold":
                labels = self._generate_threshold_labels(dataset)
            elif method == "combined":
                labels = self._generate_combined_labels(dataset)
            else:
                logger.warning(f"Unknown method: {method}, using anomaly method")
                labels = self._generate_anomaly_labels(dataset)
                
            # Add labels to dataset
            enhanced_dataset["hab_labels"] = labels
            
            # Add label metadata
            enhanced_dataset.attrs["label_generation_method"] = method
            enhanced_dataset.attrs["label_generation_date"] = datetime.now().isoformat()
            enhanced_dataset.attrs["label_thresholds"] = {
                "chlorophyll_threshold": self.chlorophyll_threshold,
                "anomaly_threshold": self.anomaly_threshold,
                "percentile_threshold": self.percentile_threshold
            }
            
            logger.info("HAB label generation complete")
            return enhanced_dataset
            
        except Exception as e:
            logger.error(f"HAB label generation failed: {e}")
            return dataset
            
    def _generate_anomaly_labels(self, dataset: xr.Dataset) -> xr.DataArray:
        """Generate labels based on chlorophyll anomalies"""
        logger.info("Generating labels based on chlorophyll anomalies")
        
        try:
            # Find chlorophyll variable
            chlorophyll_var = None
            for var_name in dataset.data_vars:
                if "chlorophyll" in var_name.lower() or "chl" in var_name.lower():
                    chlorophyll_var = var_name
                    break
                    
            if chlorophyll_var is None:
                logger.warning("No chlorophyll variable found, using first variable")
                chlorophyll_var = list(dataset.data_vars.keys())[0]
                
            chlorophyll_data = dataset[chlorophyll_var]
            
            # Compute climatological mean and standard deviation
            climatology = chlorophyll_data.groupby("week_of_year").mean("time")
            climatology_std = chlorophyll_data.groupby("week_of_year").std("time")
            
            # Compute anomalies
            anomalies = chlorophyll_data - climatology.sel(week_of_year=dataset.coords["week_of_year"])
            
            # Standardize anomalies
            std_anomalies = anomalies / (climatology_std.sel(week_of_year=dataset.coords["week_of_year"]) + 1e-8)
            
            # Create labels based on anomaly threshold
            labels = (std_anomalies > self.anomaly_threshold).astype(int)
            
            return labels
            
        except Exception as e:
            logger.error(f"Failed to generate anomaly labels: {e}")
            # Return zeros as fallback
            return xr.zeros_like(dataset[list(dataset.data_vars.keys())[0]])
            
    def _generate_threshold_labels(self, dataset: xr.Dataset) -> xr.DataArray:
        """Generate labels based on absolute chlorophyll thresholds"""
        logger.info("Generating labels based on chlorophyll thresholds")
        
        try:
            # Find chlorophyll variable
            chlorophyll_var = None
            for var_name in dataset.data_vars:
                if "chlorophyll" in var_name.lower() or "chl" in var_name.lower():
                    chlorophyll_var = var_name
                    break
                    
            if chlorophyll_var is None:
                logger.warning("No chlorophyll variable found, using first variable")
                chlorophyll_var = list(dataset.data_vars.keys())[0]
                
            chlorophyll_data = dataset[chlorophyll_var]
            
            # Create labels based on absolute threshold
            labels = (chlorophyll_data > self.chlorophyll_threshold).astype(int)
            
            return labels
            
        except Exception as e:
            logger.error(f"Failed to generate threshold labels: {e}")
            # Return zeros as fallback
            return xr.zeros_like(dataset[list(dataset.data_vars.keys())[0]])
            
    def _generate_combined_labels(self, dataset: xr.Dataset) -> xr.DataArray:
        """Generate labels using combined anomaly and threshold approach"""
        logger.info("Generating labels using combined approach")
        
        try:
            # Generate both types of labels
            anomaly_labels = self._generate_anomaly_labels(dataset)
            threshold_labels = self._generate_threshold_labels(dataset)
            
            # Combine labels (HAB if both conditions are met)
            combined_labels = (anomaly_labels & threshold_labels).astype(int)
            
            return combined_labels
            
        except Exception as e:
            logger.error(f"Failed to generate combined labels: {e}")
            # Return zeros as fallback
            return xr.zeros_like(dataset[list(dataset.data_vars.keys())[0]])
            
    def generate_multi_class_labels(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Generate multi-class HAB labels (None, Low, Moderate, High, Severe)
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with multi-class labels added
        """
        logger.info("Generating multi-class HAB labels")
        
        enhanced_dataset = dataset.copy()
        
        try:
            # Find chlorophyll variable
            chlorophyll_var = None
            for var_name in dataset.data_vars:
                if "chlorophyll" in var_name.lower() or "chl" in var_name.lower():
                    chlorophyll_var = var_name
                    break
                    
            if chlorophyll_var is None:
                logger.warning("No chlorophyll variable found, using first variable")
                chlorophyll_var = list(dataset.data_vars.keys())[0]
                
            chlorophyll_data = dataset[chlorophyll_var]
            
            # Compute percentiles for classification
            p50 = chlorophyll_data.quantile(0.5)
            p75 = chlorophyll_data.quantile(0.75)
            p90 = chlorophyll_data.quantile(0.90)
            p95 = chlorophyll_data.quantile(0.95)
            
            # Create multi-class labels
            # 0: None (below 50th percentile)
            # 1: Low (50th-75th percentile)
            # 2: Moderate (75th-90th percentile)
            # 3: High (90th-95th percentile)
            # 4: Severe (above 95th percentile)
            
            multi_class_labels = xr.zeros_like(chlorophyll_data, dtype=int)
            
            # Assign classes
            multi_class_labels = xr.where(chlorophyll_data >= p95, 4, multi_class_labels)  # Severe
            multi_class_labels = xr.where((chlorophyll_data >= p90) & (chlorophyll_data < p95), 3, multi_class_labels)  # High
            multi_class_labels = xr.where((chlorophyll_data >= p75) & (chlorophyll_data < p90), 2, multi_class_labels)  # Moderate
            multi_class_labels = xr.where((chlorophyll_data >= p50) & (chlorophyll_data < p75), 1, multi_class_labels)  # Low
            # 0: None (below p50)
            
            enhanced_dataset["hab_multi_class_labels"] = multi_class_labels
            
            # Add label metadata
            enhanced_dataset.attrs["multi_class_thresholds"] = {
                "low": float(p50),
                "moderate": float(p75),
                "high": float(p90),
                "severe": float(p95)
            }
            
            logger.info("Multi-class HAB label generation complete")
            return enhanced_dataset
            
        except Exception as e:
            logger.error(f"Multi-class label generation failed: {e}")
            return dataset
            
    def generate_lead_time_labels(self, dataset: xr.Dataset, lead_times: List[int] = None) -> xr.Dataset:
        """
        Generate labels for different lead times (1-4 weeks ahead)
        
        Args:
            dataset: Input dataset with HAB labels
            lead_times: List of lead times in weeks
            
        Returns:
            Dataset with lead time labels added
        """
        if lead_times is None:
            lead_times = [1, 2, 3, 4]
            
        logger.info(f"Generating lead time labels for {lead_times} weeks")
        
        enhanced_dataset = dataset.copy()
        
        try:
            # Check if HAB labels exist
            if "hab_labels" not in dataset.data_vars:
                logger.warning("No HAB labels found, generating them first")
                enhanced_dataset = self.generate_hab_labels(enhanced_dataset)
                
            hab_labels = enhanced_dataset["hab_labels"]
            
            # Generate lead time labels
            for lead_time in lead_times:
                # Shift labels forward by lead time
                lead_time_labels = hab_labels.shift(time=-lead_time)
                enhanced_dataset[f"hab_labels_lead_{lead_time}w"] = lead_time_labels
                
            logger.info("Lead time label generation complete")
            return enhanced_dataset
            
        except Exception as e:
            logger.error(f"Lead time label generation failed: {e}")
            return dataset
            
    def apply_spatial_filtering(self, dataset: xr.Dataset, min_coverage: float = None) -> xr.Dataset:
        """
        Apply spatial filtering to remove isolated HAB events
        
        Args:
            dataset: Input dataset with HAB labels
            min_coverage: Minimum spatial coverage threshold
            
        Returns:
            Dataset with filtered labels
        """
        if min_coverage is None:
            min_coverage = self.min_spatial_coverage
            
        logger.info(f"Applying spatial filtering with minimum coverage: {min_coverage}")
        
        enhanced_dataset = dataset.copy()
        
        try:
            # Check if HAB labels exist
            if "hab_labels" not in dataset.data_vars:
                logger.warning("No HAB labels found")
                return dataset
                
            hab_labels = dataset["hab_labels"]
            
            # Compute spatial coverage for each time step
            spatial_coverage = hab_labels.mean(dim=["latitude", "longitude"])
            
            # Create mask for times with sufficient coverage
            coverage_mask = spatial_coverage >= min_coverage
            
            # Apply mask to labels
            filtered_labels = hab_labels.where(coverage_mask, 0)
            
            enhanced_dataset["hab_labels_filtered"] = filtered_labels
            enhanced_dataset["hab_spatial_coverage"] = spatial_coverage
            
            logger.info("Spatial filtering complete")
            return enhanced_dataset
            
        except Exception as e:
            logger.error(f"Spatial filtering failed: {e}")
            return dataset
            
    def apply_temporal_filtering(self, dataset: xr.Dataset, min_duration: int = None) -> xr.Dataset:
        """
        Apply temporal filtering to remove short-lived HAB events
        
        Args:
            dataset: Input dataset with HAB labels
            min_duration: Minimum event duration in weeks
            
        Returns:
            Dataset with filtered labels
        """
        if min_duration is None:
            min_duration = self.min_event_duration
            
        logger.info(f"Applying temporal filtering with minimum duration: {min_duration} weeks")
        
        enhanced_dataset = dataset.copy()
        
        try:
            # Check if HAB labels exist
            if "hab_labels" not in dataset.data_vars:
                logger.warning("No HAB labels found")
                return dataset
                
            hab_labels = dataset["hab_labels"]
            
            # Apply rolling minimum filter
            # An event must persist for at least min_duration weeks
            filtered_labels = hab_labels.rolling(time=min_duration, center=True).min()
            
            enhanced_dataset["hab_labels_temporal_filtered"] = filtered_labels
            
            logger.info("Temporal filtering complete")
            return enhanced_dataset
            
        except Exception as e:
            logger.error(f"Temporal filtering failed: {e}")
            return dataset
            
    def save_labels(self, dataset: xr.Dataset, filename: str = None) -> Path:
        """Save HAB labels to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hab_labels_{timestamp}.nc"
            
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
        logger.info(f"Saved HAB labels: {output_path}")
        
        return output_path
        
    def get_label_summary(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """Get summary of HAB labels"""
        summary = {
            "label_variables": [],
            "label_statistics": {},
            "event_counts": {},
            "spatial_coverage": {}
        }
        
        # Find label variables
        for var_name in dataset.data_vars:
            if "label" in var_name.lower():
                summary["label_variables"].append(var_name)
                
                # Compute label statistics
                var_data = dataset[var_name]
                summary["label_statistics"][var_name] = {
                    "total_pixels": int(var_data.size),
                    "hab_pixels": int(var_data.sum()),
                    "hab_fraction": float(var_data.mean()),
                    "unique_values": list(np.unique(var_data.values))
                }
                
                # Count events over time
                if "time" in var_data.dims:
                    time_series = var_data.mean(dim=["latitude", "longitude"])
                    events = (time_series > 0).sum().item()
                    summary["event_counts"][var_name] = {
                        "total_events": events,
                        "event_fraction": float(events / len(time_series))
                    }
                    
                # Spatial coverage statistics
                if "latitude" in var_data.dims and "longitude" in var_data.dims:
                    spatial_coverage = var_data.mean(dim=["latitude", "longitude"])
                    summary["spatial_coverage"][var_name] = {
                        "mean_coverage": float(spatial_coverage.mean()),
                        "max_coverage": float(spatial_coverage.max()),
                        "coverage_std": float(spatial_coverage.std())
                    }
                    
        return summary 