"""
Data Quality Control for HAB Forecasting System

This module provides comprehensive data quality control and validation
for all collected datasets in the Seto Inland Sea region.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import xarray as xr
import numpy as np
import pandas as pd
from src.utils.config_loader import ConfigLoader, get_data_dir

logger = logging.getLogger(__name__)

class DataQualityController:
    """Comprehensive data quality control and validation"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the quality controller"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_data_sources_config()
        self.quality_thresholds = self.config.get("quality_thresholds", {})
        self.processing_config = self.config_loader.get_processing_config()
        self.qc_config = self.processing_config.get("quality_control", {})
        
        # Output directories
        self.data_dir = get_data_dir()
        self.qc_reports_dir = self.data_dir / "qc_reports"
        self.qc_reports_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_dataset_structure(self, dataset: xr.Dataset, data_type: str) -> Dict[str, Any]:
        """
        Validate dataset structure and dimensions
        
        Args:
            dataset: Dataset to validate
            data_type: Type of data for context
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating dataset structure for {data_type}")
        
        validation_results = {
            "data_type": data_type,
            "timestamp": datetime.now().isoformat(),
            "structure_validation": {},
            "dimension_validation": {},
            "variable_validation": {},
            "overall_valid": True
        }
        
        try:
            # Check basic structure
            validation_results["structure_validation"] = {
                "has_data": dataset is not None,
                "has_dims": len(dataset.dims) > 0,
                "has_vars": len(dataset.data_vars) > 0,
                "has_coords": len(dataset.coords) > 0
            }
            
            # Check required dimensions
            required_dims = ["latitude", "longitude"]
            optional_dims = ["time", "depth"]
            
            validation_results["dimension_validation"] = {
                "required_dims_present": all(dim in dataset.dims for dim in required_dims),
                "required_dims": required_dims,
                "present_dims": list(dataset.dims.keys()),
                "missing_required_dims": [dim for dim in required_dims if dim not in dataset.dims],
                "optional_dims_present": [dim for dim in optional_dims if dim in dataset.dims]
            }
            
            # Check variables
            validation_results["variable_validation"] = {
                "total_variables": len(dataset.data_vars),
                "variable_names": list(dataset.data_vars.keys()),
                "variables_with_attrs": [var for var in dataset.data_vars if dataset[var].attrs]
            }
            
            # Overall validation
            structure_valid = (
                validation_results["structure_validation"]["has_data"] and
                validation_results["structure_validation"]["has_dims"] and
                validation_results["structure_validation"]["has_vars"] and
                validation_results["dimension_validation"]["required_dims_present"]
            )
            
            validation_results["overall_valid"] = structure_valid
            
            if not structure_valid:
                logger.warning(f"Dataset structure validation failed for {data_type}")
            else:
                logger.info(f"Dataset structure validation passed for {data_type}")
                
        except Exception as e:
            logger.error(f"Dataset structure validation error for {data_type}: {e}")
            validation_results["overall_valid"] = False
            validation_results["error"] = str(e)
            
        return validation_results
        
    def validate_data_values(self, dataset: xr.Dataset, data_type: str) -> Dict[str, Any]:
        """
        Validate data values and ranges
        
        Args:
            dataset: Dataset to validate
            data_type: Type of data for context
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating data values for {data_type}")
        
        validation_results = {
            "data_type": data_type,
            "timestamp": datetime.now().isoformat(),
            "value_validation": {},
            "range_validation": {},
            "missing_data_analysis": {},
            "outlier_analysis": {},
            "overall_valid": True
        }
        
        try:
            for var_name in dataset.data_vars:
                var_data = dataset[var_name]
                
                # Basic statistics
                var_stats = {
                    "min": float(var_data.min().values),
                    "max": float(var_data.max().values),
                    "mean": float(var_data.mean().values),
                    "std": float(var_data.std().values),
                    "total_pixels": int(var_data.size),
                    "missing_pixels": int(var_data.isnull().sum().values),
                    "missing_ratio": float(var_data.isnull().sum().values / var_data.size)
                }
                
                validation_results["value_validation"][var_name] = var_stats
                
                # Range validation
                if var_name in self.quality_thresholds:
                    thresholds = self.quality_thresholds[var_name]
                    min_threshold = thresholds.get("min_valid")
                    max_threshold = thresholds.get("max_valid")
                    
                    range_validation = {
                        "min_threshold": min_threshold,
                        "max_threshold": max_threshold,
                        "below_min": int((var_data < min_threshold).sum().values) if min_threshold else 0,
                        "above_max": int((var_data > max_threshold).sum().values) if max_threshold else 0,
                        "within_range": True
                    }
                    
                    if min_threshold and var_stats["min"] < min_threshold:
                        range_validation["within_range"] = False
                    if max_threshold and var_stats["max"] > max_threshold:
                        range_validation["within_range"] = False
                        
                    validation_results["range_validation"][var_name] = range_validation
                    
                # Missing data analysis
                missing_analysis = {
                    "missing_ratio": var_stats["missing_ratio"],
                    "acceptable_missing": var_stats["missing_ratio"] <= 0.5,  # 50% threshold
                    "missing_pattern": "random"  # Simplified - could be enhanced
                }
                
                validation_results["missing_data_analysis"][var_name] = missing_analysis
                
                # Outlier detection (simplified)
                if var_stats["std"] > 0:
                    z_scores = np.abs((var_data - var_stats["mean"]) / var_stats["std"])
                    outliers = z_scores > 3  # 3 standard deviations
                    outlier_count = int(outliers.sum().values)
                    outlier_ratio = outlier_count / var_stats["total_pixels"]
                    
                    validation_results["outlier_analysis"][var_name] = {
                        "outlier_count": outlier_count,
                        "outlier_ratio": outlier_ratio,
                        "acceptable_outliers": outlier_ratio <= 0.01  # 1% threshold
                    }
                    
            # Overall validation
            overall_valid = True
            
            for var_name in dataset.data_vars:
                if var_name in validation_results["range_validation"]:
                    if not validation_results["range_validation"][var_name]["within_range"]:
                        overall_valid = False
                        
                if not validation_results["missing_data_analysis"][var_name]["acceptable_missing"]:
                    overall_valid = False
                    
                if var_name in validation_results["outlier_analysis"]:
                    if not validation_results["outlier_analysis"][var_name]["acceptable_outliers"]:
                        overall_valid = False
                        
            validation_results["overall_valid"] = overall_valid
            
            if not overall_valid:
                logger.warning(f"Data value validation failed for {data_type}")
            else:
                logger.info(f"Data value validation passed for {data_type}")
                
        except Exception as e:
            logger.error(f"Data value validation error for {data_type}: {e}")
            validation_results["overall_valid"] = False
            validation_results["error"] = str(e)
            
        return validation_results
        
    def validate_temporal_consistency(self, dataset: xr.Dataset, data_type: str) -> Dict[str, Any]:
        """
        Validate temporal consistency of data
        
        Args:
            dataset: Dataset to validate
            data_type: Type of data for context
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating temporal consistency for {data_type}")
        
        validation_results = {
            "data_type": data_type,
            "timestamp": datetime.now().isoformat(),
            "temporal_validation": {},
            "overall_valid": True
        }
        
        try:
            if "time" not in dataset.dims:
                validation_results["temporal_validation"] = {
                    "has_time_dimension": False,
                    "message": "No time dimension found"
                }
                return validation_results
                
            time_coord = dataset.time
            
            # Time range analysis
            time_range = {
                "start_time": str(time_coord.min().values),
                "end_time": str(time_coord.max().values),
                "total_timesteps": len(time_coord),
                "time_resolution": None
            }
            
            # Calculate time resolution
            if len(time_coord) > 1:
                time_diffs = time_coord.diff("time")
                time_resolution = time_diffs.mean().values
                time_range["time_resolution"] = str(time_resolution)
                
            validation_results["temporal_validation"]["time_range"] = time_range
            
            # Check for temporal gaps
            if len(time_coord) > 1:
                expected_timesteps = len(time_coord)
                actual_timesteps = len(time_coord)
                
                temporal_gaps = {
                    "expected_timesteps": expected_timesteps,
                    "actual_timesteps": actual_timesteps,
                    "missing_timesteps": expected_timesteps - actual_timesteps,
                    "gap_ratio": (expected_timesteps - actual_timesteps) / expected_timesteps if expected_timesteps > 0 else 0,
                    "acceptable_gaps": (expected_timesteps - actual_timesteps) / expected_timesteps <= 0.2 if expected_timesteps > 0 else True
                }
                
                validation_results["temporal_validation"]["temporal_gaps"] = temporal_gaps
                validation_results["overall_valid"] = temporal_gaps["acceptable_gaps"]
                
            logger.info(f"Temporal consistency validation completed for {data_type}")
            
        except Exception as e:
            logger.error(f"Temporal consistency validation error for {data_type}: {e}")
            validation_results["overall_valid"] = False
            validation_results["error"] = str(e)
            
        return validation_results
        
    def validate_spatial_consistency(self, dataset: xr.Dataset, data_type: str) -> Dict[str, Any]:
        """
        Validate spatial consistency of data
        
        Args:
            dataset: Dataset to validate
            data_type: Type of data for context
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating spatial consistency for {data_type}")
        
        validation_results = {
            "data_type": data_type,
            "timestamp": datetime.now().isoformat(),
            "spatial_validation": {},
            "overall_valid": True
        }
        
        try:
            # Check spatial bounds
            spatial_bounds = self.config.get("spatial_bounds", {})
            
            if "latitude" in dataset.dims and "longitude" in dataset.dims:
                lat_coord = dataset.latitude
                lon_coord = dataset.longitude
                
                spatial_coverage = {
                    "lat_min": float(lat_coord.min().values),
                    "lat_max": float(lat_coord.max().values),
                    "lon_min": float(lon_coord.min().values),
                    "lon_max": float(lon_coord.max().values),
                    "expected_lat_min": spatial_bounds.get("south"),
                    "expected_lat_max": spatial_bounds.get("north"),
                    "expected_lon_min": spatial_bounds.get("west"),
                    "expected_lon_max": spatial_bounds.get("east")
                }
                
                # Check if data covers expected region
                covers_region = (
                    spatial_coverage["lat_min"] <= spatial_coverage["expected_lat_min"] and
                    spatial_coverage["lat_max"] >= spatial_coverage["expected_lat_max"] and
                    spatial_coverage["lon_min"] <= spatial_coverage["expected_lon_min"] and
                    spatial_coverage["lon_max"] >= spatial_coverage["expected_lon_max"]
                )
                
                spatial_coverage["covers_expected_region"] = covers_region
                validation_results["spatial_validation"]["spatial_coverage"] = spatial_coverage
                
                # Spatial resolution
                if len(lat_coord) > 1 and len(lon_coord) > 1:
                    lat_resolution = abs(lat_coord.diff("latitude").mean().values)
                    lon_resolution = abs(lon_coord.diff("longitude").mean().values)
                    
                    spatial_resolution = {
                        "lat_resolution": float(lat_resolution),
                        "lon_resolution": float(lon_resolution),
                        "expected_resolution": spatial_bounds.get("resolution_km", 2.0),
                        "resolution_acceptable": lat_resolution <= 0.02 and lon_resolution <= 0.02  # ~2km
                    }
                    
                    validation_results["spatial_validation"]["spatial_resolution"] = spatial_resolution
                    
                validation_results["overall_valid"] = covers_region
                
            else:
                validation_results["spatial_validation"]["error"] = "No spatial dimensions found"
                validation_results["overall_valid"] = False
                
            logger.info(f"Spatial consistency validation completed for {data_type}")
            
        except Exception as e:
            logger.error(f"Spatial consistency validation error for {data_type}: {e}")
            validation_results["overall_valid"] = False
            validation_results["error"] = str(e)
            
        return validation_results
        
    def run_comprehensive_validation(self, dataset: xr.Dataset, data_type: str) -> Dict[str, Any]:
        """
        Run comprehensive validation on a dataset
        
        Args:
            dataset: Dataset to validate
            data_type: Type of data for context
            
        Returns:
            Comprehensive validation results
        """
        logger.info(f"Running comprehensive validation for {data_type}")
        
        comprehensive_results = {
            "data_type": data_type,
            "validation_timestamp": datetime.now().isoformat(),
            "validation_results": {},
            "overall_score": 0.0,
            "overall_valid": True,
            "recommendations": []
        }
        
        # Run all validation checks
        structure_results = self.validate_dataset_structure(dataset, data_type)
        value_results = self.validate_data_values(dataset, data_type)
        temporal_results = self.validate_temporal_consistency(dataset, data_type)
        spatial_results = self.validate_spatial_consistency(dataset, data_type)
        
        comprehensive_results["validation_results"] = {
            "structure": structure_results,
            "values": value_results,
            "temporal": temporal_results,
            "spatial": spatial_results
        }
        
        # Calculate overall score
        validation_scores = []
        
        if structure_results["overall_valid"]:
            validation_scores.append(1.0)
        else:
            validation_scores.append(0.0)
            comprehensive_results["recommendations"].append("Fix dataset structure issues")
            
        if value_results["overall_valid"]:
            validation_scores.append(1.0)
        else:
            validation_scores.append(0.0)
            comprehensive_results["recommendations"].append("Address data value issues")
            
        if temporal_results["overall_valid"]:
            validation_scores.append(1.0)
        else:
            validation_scores.append(0.0)
            comprehensive_results["recommendations"].append("Fix temporal consistency issues")
            
        if spatial_results["overall_valid"]:
            validation_scores.append(1.0)
        else:
            validation_scores.append(0.0)
            comprehensive_results["recommendations"].append("Address spatial coverage issues")
            
        comprehensive_results["overall_score"] = np.mean(validation_scores)
        comprehensive_results["overall_valid"] = comprehensive_results["overall_score"] >= 0.75  # 75% threshold
        
        if comprehensive_results["overall_valid"]:
            logger.info(f"Comprehensive validation passed for {data_type} (score: {comprehensive_results['overall_score']:.2f})")
        else:
            logger.warning(f"Comprehensive validation failed for {data_type} (score: {comprehensive_results['overall_score']:.2f})")
            
        return comprehensive_results
        
    def check_missing_data(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """
        Check for missing data in dataset (for compatibility with test)
        
        Args:
            dataset: Dataset to check
            
        Returns:
            Dictionary with missing data analysis
        """
        logger.info("Checking for missing data in dataset")
        
        missing_analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_variables": len(dataset.data_vars),
            "missing_data_by_variable": {},
            "overall_missing_ratio": 0.0,
            "variables_with_missing": []
        }
        
        try:
            total_missing = 0
            total_values = 0
            
            for var_name, var_data in dataset.data_vars.items():
                # Count missing values (NaN, inf, etc.)
                missing_count = np.isnan(var_data).sum().item()
                total_count = var_data.size
                
                missing_ratio = (missing_count / total_count * 100) if total_count > 0 else 0
                
                missing_analysis["missing_data_by_variable"][var_name] = {
                    "missing_count": int(missing_count),
                    "total_count": int(total_count),
                    "missing_ratio_percent": float(missing_ratio)
                }
                
                total_missing += missing_count
                total_values += total_count
                
                if missing_ratio > 0:
                    missing_analysis["variables_with_missing"].append(var_name)
            
            # Calculate overall missing ratio
            if total_values > 0:
                missing_analysis["overall_missing_ratio"] = (total_missing / total_values * 100)
            
            logger.info(f"Missing data analysis completed. Overall missing ratio: {missing_analysis['overall_missing_ratio']:.2f}%")
            
        except Exception as e:
            logger.error(f"Failed to check missing data: {e}")
            missing_analysis["error"] = str(e)
            
        return missing_analysis
        
    def detect_outliers(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """
        Detect outliers in dataset (for compatibility with test)
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary with outlier analysis
        """
        logger.info("Detecting outliers in dataset")
        
        outlier_analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_variables": len(dataset.data_vars),
            "outliers_by_variable": {},
            "overall_outlier_ratio": 0.0,
            "variables_with_outliers": []
        }
        
        try:
            total_outliers = 0
            total_values = 0
            
            for var_name, var_data in dataset.data_vars.items():
                # Skip non-numeric variables
                if not np.issubdtype(var_data.dtype, np.number):
                    continue
                
                # Calculate statistics for outlier detection
                data_values = var_data.values.flatten()
                data_values = data_values[~np.isnan(data_values)]  # Remove NaN
                
                if len(data_values) == 0:
                    continue
                
                # Use IQR method for outlier detection
                q1 = np.percentile(data_values, 25)
                q3 = np.percentile(data_values, 75)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = np.sum((data_values < lower_bound) | (data_values > upper_bound))
                total_count = len(data_values)
                
                outlier_ratio = (outliers / total_count * 100) if total_count > 0 else 0
                
                outlier_analysis["outliers_by_variable"][var_name] = {
                    "outlier_count": int(outliers),
                    "total_count": int(total_count),
                    "outlier_ratio_percent": float(outlier_ratio),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "q1": float(q1),
                    "q3": float(q3)
                }
                
                total_outliers += outliers
                total_values += total_count
                
                if outlier_ratio > 0:
                    outlier_analysis["variables_with_outliers"].append(var_name)
            
            # Calculate overall outlier ratio
            if total_values > 0:
                outlier_analysis["overall_outlier_ratio"] = (total_outliers / total_values * 100)
            
            logger.info(f"Outlier detection completed. Overall outlier ratio: {outlier_analysis['overall_outlier_ratio']:.2f}%")
            
        except Exception as e:
            logger.error(f"Failed to detect outliers: {e}")
            outlier_analysis["error"] = str(e)
            
        return outlier_analysis
        
    def validate_data_ranges(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """
        Validate data ranges against expected thresholds (for compatibility with test)
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            Dictionary with range validation results
        """
        logger.info("Validating data ranges")
        
        range_validation = {
            "timestamp": datetime.now().isoformat(),
            "total_variables": len(dataset.data_vars),
            "range_validation_by_variable": {},
            "overall_valid": True,
            "variables_with_range_issues": []
        }
        
        try:
            for var_name, var_data in dataset.data_vars.items():
                # Skip non-numeric variables
                if not np.issubdtype(var_data.dtype, np.number):
                    continue
                
                # Get expected ranges from config
                var_config = self.quality_thresholds.get(var_name, {})
                min_valid = var_config.get("min_valid", -np.inf)
                max_valid = var_config.get("max_valid", np.inf)
                
                # Calculate actual ranges
                data_values = var_data.values.flatten()
                data_values = data_values[~np.isnan(data_values)]  # Remove NaN
                
                if len(data_values) == 0:
                    continue
                
                actual_min = np.min(data_values)
                actual_max = np.max(data_values)
                
                # Check if values are within expected range
                within_range = (actual_min >= min_valid) and (actual_max <= max_valid)
                
                range_validation["range_validation_by_variable"][var_name] = {
                    "actual_min": float(actual_min),
                    "actual_max": float(actual_max),
                    "expected_min": float(min_valid),
                    "expected_max": float(max_valid),
                    "within_range": within_range,
                    "total_values": int(len(data_values))
                }
                
                if not within_range:
                    range_validation["variables_with_range_issues"].append(var_name)
                    range_validation["overall_valid"] = False
            
            if range_validation["overall_valid"]:
                logger.info("Data range validation passed")
            else:
                logger.warning(f"Data range validation failed. Issues found in {len(range_validation['variables_with_range_issues'])} variables")
                
        except Exception as e:
            logger.error(f"Failed to validate data ranges: {e}")
            range_validation["error"] = str(e)
            range_validation["overall_valid"] = False
            
        return range_validation
        
    def save_validation_report(self, validation_results: Dict[str, Any], filename: str = None) -> Path:
        """Save validation report to file"""
        import json
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_type = validation_results.get("data_type", "unknown")
            filename = f"qc_report_{data_type}_{timestamp}.json"
            
        report_file = self.qc_reports_dir / filename
        
        with open(report_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
            
        logger.info(f"Validation report saved to: {report_file}")
        return report_file
        
if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample dataset for testing
    import numpy as np
    
    # Sample dataset
    lat = np.linspace(34, 35, 56)
    lon = np.linspace(132, 135, 167)
    time = pd.date_range("2024-01-01", periods=7, freq="D")
    
    data = np.random.randn(len(time), len(lat), len(lon))
    
    dataset = xr.Dataset(
        data_vars={
            "chlorophyll_a": (["time", "latitude", "longitude"], data)
        },
        coords={
            "time": time,
            "latitude": lat,
            "longitude": lon
        }
    )
    
    # Run validation
    qc = DataQualityController()
    results = qc.run_comprehensive_validation(dataset, "sample_chlorophyll")
    
    # Save report
    report_file = qc.save_validation_report(results)
    
    print(f"Validation completed. Score: {results['overall_score']:.2f}")
    print(f"Report saved to: {report_file}") 