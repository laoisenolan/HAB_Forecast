"""
Feature Scaler for HAB Forecasting System

This module handles feature scaling and normalization for the ML pipeline.
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

class FeatureScaler:
    """Handles feature scaling and normalization"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the feature scaler"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_features_config()
        self.data_dir = get_data_dir()
        
        # Get scaling settings from config
        self.scaling_config = self.config.get("scaling", {})
        self.scaling_methods = self.scaling_config.get("methods", ["standard_scaler"])
        self.default_method = self.scaling_config.get("default_method", "standard_scaler")
        self.exclude_variables = self.scaling_config.get("exclude_variables", ["week_of_year", "month", "season", "year"])
        
        # Store scaling parameters for later use
        self.scaling_params = {}
        
        # Output directory
        self.output_dir = self.data_dir / "features"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def scale_features(self, dataset: xr.Dataset, 
                      method: str = None,
                      exclude_vars: List[str] = None) -> xr.Dataset:
        """
        Scale features using specified method
        
        Args:
            dataset: Input dataset
            method: Scaling method ("standard_scaler", "minmax_scaler", "robust_scaler")
            exclude_vars: Variables to exclude from scaling
            
        Returns:
            Dataset with scaled features
        """
        if method is None:
            method = self.default_method
            
        if exclude_vars is None:
            exclude_vars = self.exclude_variables
            
        logger.info(f"Scaling features using {method} method")
        
        try:
            # Separate features and labels
            feature_vars = [var for var in dataset.data_vars if "label" not in var.lower()]
            label_vars = [var for var in dataset.data_vars if "label" in var.lower()]
            
            feature_dataset = dataset[feature_vars]
            label_dataset = dataset[label_vars]
            
            # Scale features
            scaled_features = self._apply_scaling(feature_dataset, method, exclude_vars)
            
            # Combine scaled features with labels
            scaled_dataset = xr.merge([scaled_features, label_dataset])
            
            # Add scaling metadata
            scaled_dataset.attrs["scaling_method"] = method
            scaled_dataset.attrs["scaling_date"] = datetime.now().isoformat()
            scaled_dataset.attrs["scaling_params"] = str(self.scaling_params)
            
            logger.info("Feature scaling complete")
            return scaled_dataset
            
        except Exception as e:
            logger.error(f"Feature scaling failed: {e}")
            return dataset
            
    def _apply_scaling(self, dataset: xr.Dataset, method: str, exclude_vars: List[str]) -> xr.Dataset:
        """Apply scaling to dataset"""
        scaled_dataset = dataset.copy()
        
        # Get variables to scale
        scale_vars = [var for var in dataset.data_vars if var not in exclude_vars]
        
        for var_name in scale_vars:
            var_data = dataset[var_name]
            
            if method == "standard_scaler":
                scaled_data = self._standard_scaler(var_data, var_name)
            elif method == "minmax_scaler":
                scaled_data = self._minmax_scaler(var_data, var_name)
            elif method == "robust_scaler":
                scaled_data = self._robust_scaler(var_data, var_name)
            else:
                logger.warning(f"Unknown scaling method: {method}, using standard scaler")
                scaled_data = self._standard_scaler(var_data, var_name)
                
            scaled_dataset[var_name] = scaled_data
            
        return scaled_dataset
        
    def _standard_scaler(self, data: xr.DataArray, var_name: str) -> xr.DataArray:
        """Apply standard scaling (z-score normalization)"""
        try:
            # Compute mean and std
            mean_val = float(data.mean())
            std_val = float(data.std())
            
            # Store parameters
            self.scaling_params[var_name] = {
                "method": "standard_scaler",
                "mean": mean_val,
                "std": std_val
            }
            
            # Apply scaling
            if std_val > 0:
                scaled_data = (data - mean_val) / std_val
            else:
                scaled_data = data - mean_val
                
            return scaled_data
            
        except Exception as e:
            logger.warning(f"Standard scaling failed for {var_name}: {e}")
            return data
            
    def _minmax_scaler(self, data: xr.DataArray, var_name: str) -> xr.DataArray:
        """Apply min-max scaling (0-1 normalization)"""
        try:
            # Compute min and max
            min_val = float(data.min())
            max_val = float(data.max())
            
            # Store parameters
            self.scaling_params[var_name] = {
                "method": "minmax_scaler",
                "min": min_val,
                "max": max_val
            }
            
            # Apply scaling
            if max_val > min_val:
                scaled_data = (data - min_val) / (max_val - min_val)
            else:
                scaled_data = data - min_val
                
            return scaled_data
            
        except Exception as e:
            logger.warning(f"Min-max scaling failed for {var_name}: {e}")
            return data
            
    def _robust_scaler(self, data: xr.DataArray, var_name: str) -> xr.DataArray:
        """Apply robust scaling (using median and IQR)"""
        try:
            # Compute median and IQR
            median_val = float(data.median())
            q75 = float(data.quantile(0.75))
            q25 = float(data.quantile(0.25))
            iqr = q75 - q25
            
            # Store parameters
            self.scaling_params[var_name] = {
                "method": "robust_scaler",
                "median": median_val,
                "q25": q25,
                "q75": q75,
                "iqr": iqr
            }
            
            # Apply scaling
            if iqr > 0:
                scaled_data = (data - median_val) / iqr
            else:
                scaled_data = data - median_val
                
            return scaled_data
            
        except Exception as e:
            logger.warning(f"Robust scaling failed for {var_name}: {e}")
            return data
            
    def inverse_scale(self, dataset: xr.Dataset, var_name: str) -> xr.DataArray:
        """Inverse scale a specific variable"""
        if var_name not in self.scaling_params:
            logger.warning(f"No scaling parameters found for {var_name}")
            return dataset[var_name]
            
        params = self.scaling_params[var_name]
        scaled_data = dataset[var_name]
        
        try:
            if params["method"] == "standard_scaler":
                original_data = scaled_data * params["std"] + params["mean"]
            elif params["method"] == "minmax_scaler":
                original_data = scaled_data * (params["max"] - params["min"]) + params["min"]
            elif params["method"] == "robust_scaler":
                original_data = scaled_data * params["iqr"] + params["median"]
            else:
                logger.warning(f"Unknown scaling method for inverse: {params['method']}")
                return scaled_data
                
            return original_data
            
        except Exception as e:
            logger.error(f"Inverse scaling failed for {var_name}: {e}")
            return scaled_data
            
    def scale_with_fit(self, train_dataset: xr.Dataset, test_dataset: xr.Dataset, 
                      method: str = None, exclude_vars: List[str] = None) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Scale features using training data parameters
        
        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
            method: Scaling method
            exclude_vars: Variables to exclude
            
        Returns:
            Tuple of (scaled_train_dataset, scaled_test_dataset)
        """
        if method is None:
            method = self.default_method
            
        if exclude_vars is None:
            exclude_vars = self.exclude_variables
            
        logger.info(f"Scaling features with fit using {method} method")
        
        try:
            # Fit scaling on training data
            self._fit_scaling(train_dataset, method, exclude_vars)
            
            # Apply scaling to both datasets
            scaled_train = self._apply_fitted_scaling(train_dataset, exclude_vars)
            scaled_test = self._apply_fitted_scaling(test_dataset, exclude_vars)
            
            # Add metadata
            for dataset in [scaled_train, scaled_test]:
                dataset.attrs["scaling_method"] = method
                dataset.attrs["scaling_date"] = datetime.now().isoformat()
                dataset.attrs["scaling_params"] = str(self.scaling_params)
                
            logger.info("Feature scaling with fit complete")
            return scaled_train, scaled_test
            
        except Exception as e:
            logger.error(f"Feature scaling with fit failed: {e}")
            return train_dataset, test_dataset
            
    def _fit_scaling(self, dataset: xr.Dataset, method: str, exclude_vars: List[str]):
        """Fit scaling parameters on training data"""
        self.scaling_params = {}
        
        # Get variables to scale
        scale_vars = [var for var in dataset.data_vars if var not in exclude_vars and "label" not in var.lower()]
        
        for var_name in scale_vars:
            var_data = dataset[var_name]
            
            if method == "standard_scaler":
                self._fit_standard_scaler(var_data, var_name)
            elif method == "minmax_scaler":
                self._fit_minmax_scaler(var_data, var_name)
            elif method == "robust_scaler":
                self._fit_robust_scaler(var_data, var_name)
                
    def _fit_standard_scaler(self, data: xr.DataArray, var_name: str):
        """Fit standard scaler parameters"""
        mean_val = float(data.mean())
        std_val = float(data.std())
        
        self.scaling_params[var_name] = {
            "method": "standard_scaler",
            "mean": mean_val,
            "std": std_val
        }
        
    def _fit_minmax_scaler(self, data: xr.DataArray, var_name: str):
        """Fit min-max scaler parameters"""
        min_val = float(data.min())
        max_val = float(data.max())
        
        self.scaling_params[var_name] = {
            "method": "minmax_scaler",
            "min": min_val,
            "max": max_val
        }
        
    def _fit_robust_scaler(self, data: xr.DataArray, var_name: str):
        """Fit robust scaler parameters"""
        median_val = float(data.median())
        q75 = float(data.quantile(0.75))
        q25 = float(data.quantile(0.25))
        iqr = q75 - q25
        
        self.scaling_params[var_name] = {
            "method": "robust_scaler",
            "median": median_val,
            "q25": q25,
            "q75": q75,
            "iqr": iqr
        }
        
    def _apply_fitted_scaling(self, dataset: xr.Dataset, exclude_vars: List[str]) -> xr.Dataset:
        """Apply fitted scaling parameters to dataset"""
        scaled_dataset = dataset.copy()
        
        # Get variables to scale
        scale_vars = [var for var in dataset.data_vars if var not in exclude_vars and "label" not in var.lower()]
        
        for var_name in scale_vars:
            if var_name in self.scaling_params:
                params = self.scaling_params[var_name]
                var_data = dataset[var_name]
                
                if params["method"] == "standard_scaler":
                    if params["std"] > 0:
                        scaled_data = (var_data - params["mean"]) / params["std"]
                    else:
                        scaled_data = var_data - params["mean"]
                elif params["method"] == "minmax_scaler":
                    if params["max"] > params["min"]:
                        scaled_data = (var_data - params["min"]) / (params["max"] - params["min"])
                    else:
                        scaled_data = var_data - params["min"]
                elif params["method"] == "robust_scaler":
                    if params["iqr"] > 0:
                        scaled_data = (var_data - params["median"]) / params["iqr"]
                    else:
                        scaled_data = var_data - params["median"]
                else:
                    scaled_data = var_data
                    
                scaled_dataset[var_name] = scaled_data
                
        return scaled_dataset
        
    def save_scaling_params(self, filename: str = None) -> Path:
        """Save scaling parameters to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scaling_params_{timestamp}.json"
            
        output_path = self.output_dir / filename
        
        import json
        with open(output_path, "w") as f:
            json.dump(self.scaling_params, f, indent=2)
            
        logger.info(f"Saved scaling parameters: {output_path}")
        return output_path
        
    def load_scaling_params(self, filepath: Path):
        """Load scaling parameters from file"""
        import json
        with open(filepath, "r") as f:
            self.scaling_params = json.load(f)
            
        logger.info(f"Loaded scaling parameters from: {filepath}")
        
    def get_scaling_summary(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """Get summary of scaling results"""
        summary = {
            "scaled_variables": [],
            "excluded_variables": [],
            "scaling_methods": {},
            "statistics": {}
        }
        
        # Categorize variables
        for var_name in dataset.data_vars:
            if "label" not in var_name.lower():
                if var_name in self.scaling_params:
                    summary["scaled_variables"].append(var_name)
                    summary["scaling_methods"][var_name] = self.scaling_params[var_name]["method"]
                else:
                    summary["excluded_variables"].append(var_name)
                    
                # Compute statistics
                var_data = dataset[var_name]
                summary["statistics"][var_name] = {
                    "mean": float(var_data.mean()),
                    "std": float(var_data.std()),
                    "min": float(var_data.min()),
                    "max": float(var_data.max())
                }
                
        return summary 