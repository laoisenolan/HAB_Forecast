"""
Quality Controller for HAB Forecasting System

This module handles data quality control, validation, and cleaning
for the ML pipeline.
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

class QualityController:
    """Handles data quality control and validation"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the quality controller"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_processing_config()
        self.data_dir = get_data_dir()
        
        # Get quality settings from config
        self.quality_config = self.config.get("quality", {})
        self.max_missing_fraction = self.quality_config.get("max_missing_fraction", 0.5)
        self.outlier_threshold = self.quality_config.get("outlier_threshold", 3.0)
        self.value_ranges = self.quality_config.get("value_ranges", {})
        
        # Output directory
        self.output_dir = self.data_dir / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_dataset_quality(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """
        Comprehensive dataset quality validation
        
        Args:
            dataset: Input dataset
            
        Returns:
            Quality validation report
        """
        logger.info("Starting dataset quality validation")
        
        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_quality_score": 0.0,
            "passed_validation": True,
            "issues": [],
            "statistics": {},
            "recommendations": []
        }
        
        try:
            # Check missing data
            missing_report = self._check_missing_data(dataset)
            validation_report["missing_data"] = missing_report
            
            # Check value ranges
            range_report = self._check_value_ranges(dataset)
            validation_report["value_ranges"] = range_report
            
            # Check outliers
            outlier_report = self._check_outliers(dataset)
            validation_report["outliers"] = outlier_report
            
            # Check temporal consistency
            temporal_report = self._check_temporal_consistency(dataset)
            validation_report["temporal_consistency"] = temporal_report
            
            # Check spatial consistency
            spatial_report = self._check_spatial_consistency(dataset)
            validation_report["spatial_consistency"] = spatial_report
            
            # Calculate overall quality score
            validation_report["overall_quality_score"] = self._calculate_quality_score(validation_report)
            
            # Determine if validation passed
            validation_report["passed_validation"] = validation_report["overall_quality_score"] >= 0.7
            
            logger.info(f"Quality validation complete. Score: {validation_report['overall_quality_score']:.2f}")
            return validation_report
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            validation_report["passed_validation"] = False
            validation_report["issues"].append(f"Validation error: {e}")
            return validation_report
            
    def _check_missing_data(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """Check for missing data in dataset"""
        report = {
            "total_missing_fraction": 0.0,
            "variables_with_missing": [],
            "critical_issues": []
        }
        
        total_missing = 0
        total_values = 0
        
        for var_name, var_data in dataset.data_vars.items():
            if var_data.dtype.kind in 'fc':  # Float or complex
                missing_count = var_data.isnull().sum().item()
                total_count = var_data.size
                missing_fraction = missing_count / total_count
                
                total_missing += missing_count
                total_values += total_count
                
                if missing_fraction > 0:
                    report["variables_with_missing"].append({
                        "variable": var_name,
                        "missing_fraction": missing_fraction,
                        "missing_count": missing_count
                    })
                    
                if missing_fraction > self.max_missing_fraction:
                    report["critical_issues"].append(
                        f"High missing data in {var_name}: {missing_fraction:.2%}"
                    )
                    
        if total_values > 0:
            report["total_missing_fraction"] = total_missing / total_values
            
        return report
        
    def _check_value_ranges(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """Check if values are within expected ranges"""
        report = {
            "variables_out_of_range": [],
            "range_violations": []
        }
        
        for var_name, var_data in dataset.data_vars.items():
            if var_data.dtype.kind in 'fc':  # Float or complex
                var_min = float(var_data.min())
                var_max = float(var_data.max())
                
                # Check against configured ranges
                if var_name in self.value_ranges:
                    expected_min = self.value_ranges[var_name].get("min")
                    expected_max = self.value_ranges[var_name].get("max")
                    
                    if expected_min is not None and var_min < expected_min:
                        report["range_violations"].append(
                            f"{var_name} minimum ({var_min}) below expected ({expected_min})"
                        )
                        
                    if expected_max is not None and var_max > expected_max:
                        report["range_violations"].append(
                            f"{var_name} maximum ({var_max}) above expected ({expected_max})"
                        )
                        
                # Check for extreme values
                if var_min < -1000 or var_max > 1000:
                    report["variables_out_of_range"].append({
                        "variable": var_name,
                        "min": var_min,
                        "max": var_max
                    })
                    
        return report
        
    def _check_outliers(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """Check for statistical outliers"""
        report = {
            "variables_with_outliers": [],
            "outlier_counts": {}
        }
        
        for var_name, var_data in dataset.data_vars.items():
            if var_data.dtype.kind in 'fc':  # Float or complex
                # Remove NaN values for outlier detection
                clean_data = var_data.dropna(dim="time")
                
                if len(clean_data) > 0:
                    # Calculate z-scores
                    mean_val = float(clean_data.mean())
                    std_val = float(clean_data.std())
                    
                    if std_val > 0:
                        z_scores = np.abs((clean_data - mean_val) / std_val)
                        outlier_mask = z_scores > self.outlier_threshold
                        outlier_count = outlier_mask.sum().item()
                        
                        if outlier_count > 0:
                            report["variables_with_outliers"].append(var_name)
                            report["outlier_counts"][var_name] = outlier_count
                            
        return report
        
    def _check_temporal_consistency(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """Check temporal consistency of dataset"""
        report = {
            "is_regular": True,
            "time_gaps": [],
            "temporal_issues": []
        }
        
        if "time" in dataset.dims:
            try:
                # Check for regular time intervals
                time_diffs = dataset.time.diff("time")
                if len(time_diffs) > 0:
                    mean_timedelta = time_diffs.mean()
                    std_timedelta = time_diffs.std()
                    
                    # Check if intervals are regular (within 1 day)
                    if std_timedelta > pd.Timedelta('1D'):
                        report["is_regular"] = False
                        report["temporal_issues"].append("Irregular time intervals detected")
                        
                    # Check for large gaps
                    large_gaps = time_diffs[time_diffs > pd.Timedelta('7D')]
                    if len(large_gaps) > 0:
                        report["time_gaps"] = [str(gap) for gap in large_gaps.values]
                        
            except Exception as e:
                report["temporal_issues"].append(f"Temporal analysis error: {e}")
                
        return report
        
    def _check_spatial_consistency(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """Check spatial consistency of dataset"""
        report = {
            "spatial_issues": [],
            "coordinate_consistency": True
        }
        
        # Check for required spatial dimensions
        required_dims = ["latitude", "longitude"]
        for dim in required_dims:
            if dim not in dataset.dims:
                report["spatial_issues"].append(f"Missing required dimension: {dim}")
                report["coordinate_consistency"] = False
                
        # Check coordinate ordering
        if "latitude" in dataset.dims and "longitude" in dataset.dims:
            try:
                lat_coords = dataset.latitude.values
                lon_coords = dataset.longitude.values
                
                # Check if coordinates are monotonically increasing/decreasing
                if not (np.all(np.diff(lat_coords) >= 0) or np.all(np.diff(lat_coords) <= 0)):
                    report["spatial_issues"].append("Non-monotonic latitude coordinates")
                    
                if not (np.all(np.diff(lon_coords) >= 0) or np.all(np.diff(lon_coords) <= 0)):
                    report["spatial_issues"].append("Non-monotonic longitude coordinates")
                    
            except Exception as e:
                report["spatial_issues"].append(f"Spatial analysis error: {e}")
                
        return report
        
    def _calculate_quality_score(self, validation_report: Dict[str, Any]) -> float:
        """Calculate overall quality score from validation report"""
        score = 1.0  # Start with perfect score
        
        # Penalize for missing data
        missing_report = validation_report.get("missing_data", {})
        total_missing_fraction = missing_report.get("total_missing_fraction", 0.0)
        score -= total_missing_fraction * 0.5  # Up to 50% penalty for missing data
        
        # Penalize for range violations
        range_report = validation_report.get("value_ranges", {})
        range_violations = len(range_report.get("range_violations", []))
        score -= range_violations * 0.1  # 10% penalty per range violation
        
        # Penalize for outliers
        outlier_report = validation_report.get("outliers", {})
        variables_with_outliers = len(outlier_report.get("variables_with_outliers", []))
        score -= variables_with_outliers * 0.05  # 5% penalty per variable with outliers
        
        # Penalize for temporal issues
        temporal_report = validation_report.get("temporal_consistency", {})
        if not temporal_report.get("is_regular", True):
            score -= 0.1  # 10% penalty for irregular time intervals
            
        # Penalize for spatial issues
        spatial_report = validation_report.get("spatial_consistency", {})
        if not spatial_report.get("coordinate_consistency", True):
            score -= 0.2  # 20% penalty for coordinate issues
            
        return max(0.0, score)  # Ensure score is non-negative
        
    def clean_dataset(self, dataset: xr.Dataset, cleaning_methods: List[str] = None) -> xr.Dataset:
        """
        Clean dataset based on quality issues
        
        Args:
            dataset: Input dataset
            cleaning_methods: List of cleaning methods to apply
            
        Returns:
            Cleaned dataset
        """
        if cleaning_methods is None:
            cleaning_methods = ["remove_outliers", "fill_missing", "smooth_noise"]
            
        logger.info(f"Cleaning dataset using methods: {cleaning_methods}")
        
        cleaned_dataset = dataset.copy()
        
        try:
            for method in cleaning_methods:
                if method == "remove_outliers":
                    cleaned_dataset = self._remove_outliers(cleaned_dataset)
                elif method == "fill_missing":
                    cleaned_dataset = self._fill_missing_data(cleaned_dataset)
                elif method == "smooth_noise":
                    cleaned_dataset = self._smooth_noise(cleaned_dataset)
                else:
                    logger.warning(f"Unknown cleaning method: {method}")
                    
            logger.info("Dataset cleaning complete")
            return cleaned_dataset
            
        except Exception as e:
            logger.error(f"Dataset cleaning failed: {e}")
            return dataset
            
    def _remove_outliers(self, dataset: xr.Dataset) -> xr.Dataset:
        """Remove statistical outliers from dataset"""
        logger.info("Removing outliers")
        
        cleaned_dataset = dataset.copy()
        
        for var_name, var_data in dataset.data_vars.items():
            if var_data.dtype.kind in 'fc':  # Float or complex
                try:
                    # Calculate z-scores
                    mean_val = float(var_data.mean())
                    std_val = float(var_data.std())
                    
                    if std_val > 0:
                        z_scores = np.abs((var_data - mean_val) / std_val)
                        outlier_mask = z_scores > self.outlier_threshold
                        
                        # Replace outliers with NaN
                        cleaned_dataset[var_name] = var_data.where(~outlier_mask)
                        
                except Exception as e:
                    logger.warning(f"Failed to remove outliers from {var_name}: {e}")
                    
        return cleaned_dataset
        
    def _fill_missing_data(self, dataset: xr.Dataset) -> xr.Dataset:
        """Fill missing data using interpolation"""
        logger.info("Filling missing data")
        
        try:
            # Use linear interpolation for missing values
            filled_dataset = dataset.interpolate_na(dim="time", method="linear")
            
            # Forward fill any remaining NaN values
            filled_dataset = filled_dataset.fillna(method="ffill")
            
            # Backward fill any remaining NaN values
            filled_dataset = filled_dataset.fillna(method="bfill")
            
            return filled_dataset
            
        except Exception as e:
            logger.error(f"Failed to fill missing data: {e}")
            return dataset
            
    def _smooth_noise(self, dataset: xr.Dataset) -> xr.Dataset:
        """Apply smoothing to reduce noise"""
        logger.info("Smoothing noise")
        
        smoothed_dataset = dataset.copy()
        
        for var_name, var_data in dataset.data_vars.items():
            if var_data.dtype.kind in 'fc':  # Float or complex
                try:
                    # Apply rolling mean smoothing
                    if "time" in var_data.dims:
                        smoothed = var_data.rolling(time=3, center=True).mean()
                        smoothed_dataset[var_name] = smoothed
                        
                except Exception as e:
                    logger.warning(f"Failed to smooth {var_name}: {e}")
                    
        return smoothed_dataset
        
    def save_quality_report(self, validation_report: Dict[str, Any], filename: str = None) -> Path:
        """Save quality validation report"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quality_report_{timestamp}.json"
            
        output_path = self.output_dir / filename
        
        # Convert numpy types to native Python types for JSON serialization
        import json
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
            
        # Convert report to JSON-serializable format
        json_report = json.loads(json.dumps(validation_report, default=convert_numpy))
        
        with open(output_path, 'w') as f:
            json.dump(json_report, f, indent=2)
            
        logger.info(f"Saved quality report: {output_path}")
        return output_path 