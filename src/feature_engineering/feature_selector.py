"""
Feature Selector for HAB Forecasting System

This module handles feature selection and dimensionality reduction for the ML pipeline.
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

class FeatureSelector:
    """Handles feature selection and dimensionality reduction"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the feature selector"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_features_config()
        self.data_dir = get_data_dir()
        
        # Get selection settings from config
        self.selection_config = self.config.get("selection", {})
        self.correlation_threshold = self.selection_config.get("correlation_threshold", 0.95)
        self.variance_threshold = self.selection_config.get("variance_threshold", 0.01)
        self.importance_methods = self.selection_config.get("importance_methods", ["correlation", "variance"])
        self.min_importance_threshold = self.selection_config.get("min_importance_threshold", 0.01)
        
        # Output directory
        self.output_dir = self.data_dir / "features"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def select_features(self, dataset: xr.Dataset, 
                       method: str = "correlation",
                       threshold: float = None,
                       target_variable: str = None) -> xr.Dataset:
        """
        Select features using specified method
        
        Args:
            dataset: Input dataset
            method: Selection method ("correlation", "variance", "importance")
            threshold: Selection threshold
            target_variable: Target variable for importance-based selection
            
        Returns:
            Dataset with selected features
        """
        if threshold is None:
            if method == "correlation":
                threshold = self.correlation_threshold
            elif method == "variance":
                threshold = self.variance_threshold
            else:
                threshold = self.min_importance_threshold
                
        logger.info(f"Selecting features using {method} method with threshold {threshold}")
        
        try:
            if method == "correlation":
                selected_dataset = self._correlation_selection(dataset, threshold)
            elif method == "variance":
                selected_dataset = self._variance_selection(dataset, threshold)
            elif method == "importance":
                if target_variable is None:
                    logger.warning("Target variable required for importance selection, using correlation")
                    selected_dataset = self._correlation_selection(dataset, threshold)
                else:
                    selected_dataset = self._importance_selection(dataset, target_variable, threshold)
            else:
                logger.warning(f"Unknown selection method: {method}, using correlation")
                selected_dataset = self._correlation_selection(dataset, threshold)
                
            logger.info(f"Feature selection complete: {len(selected_dataset.data_vars)} features selected")
            return selected_dataset
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return dataset
            
    def _correlation_selection(self, dataset: xr.Dataset, threshold: float) -> xr.Dataset:
        """Select features based on correlation"""
        logger.info("Performing correlation-based feature selection")
        
        # Separate features and labels
        feature_vars = [var for var in dataset.data_vars if "label" not in var.lower()]
        label_vars = [var for var in dataset.data_vars if "label" in var.lower()]
        
        feature_dataset = dataset[feature_vars]
        label_dataset = dataset[label_vars]
        
        # Convert to DataFrame for correlation analysis
        df = feature_dataset.to_dataframe().reset_index()
        
        # Compute correlation matrix
        corr_matrix = df.corr().abs()
        
        # Find highly correlated features
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        # Keep features that are not highly correlated
        selected_features = [col for col in feature_dataset.data_vars if col not in to_drop]
        
        logger.info(f"Correlation selection: {len(selected_features)} features selected from {len(feature_dataset.data_vars)}")
        
        # Combine selected features with labels
        selected_dataset = xr.merge([feature_dataset[selected_features], label_dataset])
        
        return selected_dataset
        
    def _variance_selection(self, dataset: xr.Dataset, threshold: float) -> xr.Dataset:
        """Select features based on variance"""
        logger.info("Performing variance-based feature selection")
        
        # Separate features and labels
        feature_vars = [var for var in dataset.data_vars if "label" not in var.lower()]
        label_vars = [var for var in dataset.data_vars if "label" in var.lower()]
        
        feature_dataset = dataset[feature_vars]
        label_dataset = dataset[label_vars]
        
        # Compute variance for each feature
        variances = {}
        for var_name in feature_dataset.data_vars:
            var_data = feature_dataset[var_name]
            variance = float(var_data.var())
            variances[var_name] = variance
            
        # Select features with variance above threshold
        selected_features = [var for var, var_val in variances.items() if var_val > threshold]
        
        logger.info(f"Variance selection: {len(selected_features)} features selected from {len(feature_dataset.data_vars)}")
        
        # Combine selected features with labels
        selected_dataset = xr.merge([feature_dataset[selected_features], label_dataset])
        
        return selected_dataset
        
    def _importance_selection(self, dataset: xr.Dataset, target_variable: str, threshold: float) -> xr.Dataset:
        """Select features based on importance scores (placeholder for ML-based selection)"""
        logger.info("Performing importance-based feature selection")
        
        # This is a placeholder for future implementation
        # In practice, you would use a trained model to compute feature importance scores
        
        # For now, return all features
        logger.info("Importance selection: returning all features (placeholder)")
        return dataset
        
    def compute_feature_importance(self, dataset: xr.Dataset, target_variable: str) -> Dict[str, float]:
        """Compute feature importance scores"""
        logger.info(f"Computing feature importance for target: {target_variable}")
        
        try:
            # Separate features and target
            feature_vars = [var for var in dataset.data_vars if var != target_variable and "label" not in var.lower()]
            feature_dataset = dataset[feature_vars]
            target_data = dataset[target_variable]
            
            # Convert to DataFrame
            df = feature_dataset.to_dataframe().reset_index()
            target_df = target_data.to_dataframe().reset_index()
            
            # Compute correlation with target
            importance_scores = {}
            for var_name in feature_vars:
                correlation = abs(df[var_name].corr(target_df[target_variable]))
                importance_scores[var_name] = correlation
                
            # Sort by importance
            sorted_importance = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
            
            logger.info(f"Feature importance computation complete for {len(sorted_importance)} features")
            return sorted_importance
            
        except Exception as e:
            logger.error(f"Feature importance computation failed: {e}")
            return {}
            
    def select_top_features(self, dataset: xr.Dataset, target_variable: str, n_features: int = 20) -> xr.Dataset:
        """Select top N features based on importance"""
        logger.info(f"Selecting top {n_features} features")
        
        try:
            # Compute feature importance
            importance_scores = self.compute_feature_importance(dataset, target_variable)
            
            # Select top N features
            top_features = list(importance_scores.keys())[:n_features]
            
            # Get label variables
            label_vars = [var for var in dataset.data_vars if "label" in var.lower()]
            
            # Create selected dataset
            selected_vars = top_features + label_vars
            selected_dataset = dataset[selected_vars]
            
            logger.info(f"Top feature selection complete: {len(top_features)} features selected")
            return selected_dataset
            
        except Exception as e:
            logger.error(f"Top feature selection failed: {e}")
            return dataset
            
    def remove_low_variance_features(self, dataset: xr.Dataset, threshold: float = None) -> xr.Dataset:
        """Remove features with low variance"""
        if threshold is None:
            threshold = self.variance_threshold
            
        logger.info(f"Removing low variance features (threshold: {threshold})")
        
        try:
            # Separate features and labels
            feature_vars = [var for var in dataset.data_vars if "label" not in var.lower()]
            label_vars = [var for var in dataset.data_vars if "label" in var.lower()]
            
            feature_dataset = dataset[feature_vars]
            label_dataset = dataset[label_vars]
            
            # Compute variance for each feature
            high_variance_features = []
            for var_name in feature_dataset.data_vars:
                var_data = feature_dataset[var_name]
                variance = float(var_data.var())
                if variance > threshold:
                    high_variance_features.append(var_name)
                    
            logger.info(f"Variance filtering: {len(high_variance_features)} features kept from {len(feature_dataset.data_vars)}")
            
            # Combine high variance features with labels
            filtered_dataset = xr.merge([feature_dataset[high_variance_features], label_dataset])
            
            return filtered_dataset
            
        except Exception as e:
            logger.error(f"Low variance feature removal failed: {e}")
            return dataset
            
    def remove_highly_correlated_features(self, dataset: xr.Dataset, threshold: float = None) -> xr.Dataset:
        """Remove highly correlated features"""
        if threshold is None:
            threshold = self.correlation_threshold
            
        logger.info(f"Removing highly correlated features (threshold: {threshold})")
        
        try:
            # Separate features and labels
            feature_vars = [var for var in dataset.data_vars if "label" not in var.lower()]
            label_vars = [var for var in dataset.data_vars if "label" in var.lower()]
            
            feature_dataset = dataset[feature_vars]
            label_dataset = dataset[label_vars]
            
            # Convert to DataFrame for correlation analysis
            df = feature_dataset.to_dataframe().reset_index()
            
            # Compute correlation matrix
            corr_matrix = df.corr().abs()
            
            # Find highly correlated features
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
            
            # Keep features that are not highly correlated
            low_correlation_features = [col for col in feature_dataset.data_vars if col not in to_drop]
            
            logger.info(f"Correlation filtering: {len(low_correlation_features)} features kept from {len(feature_dataset.data_vars)}")
            
            # Combine low correlation features with labels
            filtered_dataset = xr.merge([feature_dataset[low_correlation_features], label_dataset])
            
            return filtered_dataset
            
        except Exception as e:
            logger.error(f"High correlation feature removal failed: {e}")
            return dataset
            
    def get_feature_selection_summary(self, original_dataset: xr.Dataset, selected_dataset: xr.Dataset) -> Dict[str, Any]:
        """Get summary of feature selection results"""
        summary = {
            "original_features": len([var for var in original_dataset.data_vars if "label" not in var.lower()]),
            "selected_features": len([var for var in selected_dataset.data_vars if "label" not in var.lower()]),
            "reduction_ratio": 0.0,
            "removed_features": [],
            "kept_features": [],
            "selection_method": "unknown"
        }
        
        # Compute reduction ratio
        if summary["original_features"] > 0:
            summary["reduction_ratio"] = 1.0 - (summary["selected_features"] / summary["original_features"])
            
        # Get feature lists
        original_features = [var for var in original_dataset.data_vars if "label" not in var.lower()]
        selected_features = [var for var in selected_dataset.data_vars if "label" not in var.lower()]
        
        summary["removed_features"] = [var for var in original_features if var not in selected_features]
        summary["kept_features"] = selected_features
        
        return summary
        
    def save_selection_results(self, selected_dataset: xr.Dataset, summary: Dict[str, Any], filename: str = None) -> Path:
        """Save feature selection results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"selected_features_{timestamp}.nc"
            
        output_path = self.output_dir / filename
        
        # Save dataset with compression
        encoding = {}
        for var in selected_dataset.data_vars:
            encoding[var] = {
                "zlib": True,
                "complevel": 6,
                "shuffle": True
            }
            
        selected_dataset.to_netcdf(output_path, encoding=encoding)
        
        # Save summary
        summary_file = output_path.with_suffix(".json")
        import json
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Saved feature selection results: {output_path}")
        logger.info(f"Saved selection summary: {summary_file}")
        
        return output_path 