"""
Quality Control for HAB Forecasting System

This module handles data quality control and validation.
"""

import pandas as pd
import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class QualityController:
    """
    Performs quality control on environmental data.
    
    This class handles data validation, outlier detection, and
    quality assessment for HAB forecasting datasets.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the QualityController.
        
        Args:
            config: Configuration dictionary with QC parameters
        """
        self.config = config
        self.outlier_threshold = config.get('outlier_threshold', 3.0)  # standard deviations
        self.missing_threshold = config.get('missing_threshold', 0.5)  # fraction
        self.valid_ranges = config.get('valid_ranges', {
            'temperature': [-5, 40],  # Celsius
            'salinity': [0, 50],      # PSU
            'chlorophyll': [0, 100],  # mg/m³
            'nutrients': [0, 1000],   # μmol/L
            'wind_speed': [0, 50],    # m/s
            'wind_direction': [0, 360], # degrees
            'precipitation': [0, 500], # mm/day
            'solar_radiation': [0, 1000] # W/m²
        })
        
    def check_dataframe_quality(self, df: pd.DataFrame,
                               variable_column: str = 'variable',
                               value_column: str = 'value') -> Dict:
        """
        Perform quality control on a DataFrame.
        
        Args:
            df: DataFrame to check
            variable_column: Name of the variable column
            value_column: Name of the value column
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            quality_report = {
                'total_records': len(df),
                'missing_values': df[value_column].isna().sum(),
                'missing_fraction': df[value_column].isna().mean(),
                'outliers': 0,
                'invalid_values': 0,
                'variables_checked': []
            }
            
            # Check each variable
            for variable in df[variable_column].unique():
                var_data = df[df[variable_column] == variable][value_column]
                
                if variable in self.valid_ranges:
                    min_val, max_val = self.valid_ranges[variable]
                    
                    # Check for values outside valid range
                    invalid_mask = (var_data < min_val) | (var_data > max_val)
                    invalid_count = invalid_mask.sum()
                    
                    # Check for outliers using IQR method
                    Q1 = var_data.quantile(0.25)
                    Q3 = var_data.quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_mask = (var_data < Q1 - 1.5 * IQR) | (var_data > Q3 + 1.5 * IQR)
                    outlier_count = outlier_mask.sum()
                    
                    quality_report['variables_checked'].append({
                        'variable': variable,
                        'records': len(var_data),
                        'missing': var_data.isna().sum(),
                        'invalid': invalid_count,
                        'outliers': outlier_count,
                        'min': var_data.min(),
                        'max': var_data.max(),
                        'mean': var_data.mean(),
                        'std': var_data.std()
                    })
                    
                    quality_report['invalid_values'] += invalid_count
                    quality_report['outliers'] += outlier_count
            
            logger.info(f"Quality check completed: {quality_report['total_records']} records, "
                       f"{quality_report['missing_values']} missing, "
                       f"{quality_report['outliers']} outliers")
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Error in DataFrame quality check: {e}")
            raise
    
    def check_dataset_quality(self, dataset: xr.Dataset) -> Dict:
        """
        Perform quality control on an xarray Dataset.
        
        Args:
            dataset: xarray Dataset to check
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            quality_report = {
                'total_variables': len(dataset.data_vars),
                'variables_checked': [],
                'spatial_coverage': {},
                'temporal_coverage': {}
            }
            
            for var_name in dataset.data_vars:
                var_data = dataset[var_name]
                
                # Basic statistics
                var_stats = {
                    'variable': var_name,
                    'shape': var_data.shape,
                    'missing_fraction': var_data.isnull().mean().values.item(),
                    'min': float(var_data.min().values),
                    'max': float(var_data.max().values),
                    'mean': float(var_data.mean().values),
                    'std': float(var_data.std().values)
                }
                
                # Check for valid ranges if defined
                if var_name in self.valid_ranges:
                    min_val, max_val = self.valid_ranges[var_name]
                    invalid_mask = (var_data < min_val) | (var_data > max_val)
                    var_stats['invalid_fraction'] = float(invalid_mask.mean().values)
                
                quality_report['variables_checked'].append(var_stats)
            
            # Check spatial coverage
            if 'lat' in dataset.coords and 'lon' in dataset.coords:
                quality_report['spatial_coverage'] = {
                    'lat_range': [float(dataset.lat.min()), float(dataset.lat.max())],
                    'lon_range': [float(dataset.lon.min()), float(dataset.lon.max())],
                    'lat_resolution': float(dataset.lat.diff('lat').mean()),
                    'lon_resolution': float(dataset.lon.diff('lon').mean())
                }
            
            # Check temporal coverage
            time_coords = [coord for coord in dataset.coords if 'time' in coord.name.lower()]
            if time_coords:
                time_coord = time_coords[0]
                quality_report['temporal_coverage'] = {
                    'start_time': str(dataset[time_coord].min().values),
                    'end_time': str(dataset[time_coord].max().values),
                    'time_steps': len(dataset[time_coord])
                }
            
            logger.info(f"Dataset quality check completed: {len(dataset.data_vars)} variables")
            return quality_report
            
        except Exception as e:
            logger.error(f"Error in Dataset quality check: {e}")
            raise
    
    def filter_outliers(self, df: pd.DataFrame,
                       variable_column: str = 'variable',
                       value_column: str = 'value',
                       method: str = 'iqr') -> pd.DataFrame:
        """
        Filter outliers from a DataFrame.
        
        Args:
            df: DataFrame to filter
            variable_column: Name of the variable column
            value_column: Name of the value column
            method: Outlier detection method ('iqr' or 'zscore')
            
        Returns:
            DataFrame with outliers removed
        """
        try:
            filtered_df = df.copy()
            
            for variable in df[variable_column].unique():
                var_mask = df[variable_column] == variable
                var_data = df[var_mask][value_column]
                
                if method == 'iqr':
                    # IQR method
                    Q1 = var_data.quantile(0.25)
                    Q3 = var_data.quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_mask = (var_data < Q1 - 1.5 * IQR) | (var_data > Q3 + 1.5 * IQR)
                    
                elif method == 'zscore':
                    # Z-score method
                    z_scores = np.abs((var_data - var_data.mean()) / var_data.std())
                    outlier_mask = z_scores > self.outlier_threshold
                    
                else:
                    logger.warning(f"Unknown outlier detection method: {method}")
                    continue
                
                # Remove outliers
                outlier_indices = var_data[outlier_mask].index
                filtered_df = filtered_df.drop(outlier_indices)
                
                logger.info(f"Removed {len(outlier_indices)} outliers from {variable}")
            
            logger.info(f"Outlier filtering completed: {len(df)} -> {len(filtered_df)} records")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error in outlier filtering: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame,
                     variable_column: str = 'variable',
                     value_column: str = 'value') -> pd.DataFrame:
        """
        Validate data against defined ranges.
        
        Args:
            df: DataFrame to validate
            variable_column: Name of the variable column
            value_column: Name of the value column
            
        Returns:
            DataFrame with invalid values marked as NaN
        """
        try:
            validated_df = df.copy()
            
            for variable in df[variable_column].unique():
                if variable in self.valid_ranges:
                    min_val, max_val = self.valid_ranges[variable]
                    var_mask = df[variable_column] == variable
                    
                    # Mark invalid values as NaN
                    invalid_mask = (df[var_mask][value_column] < min_val) | (df[var_mask][value_column] > max_val)
                    invalid_indices = df[var_mask][invalid_mask].index
                    validated_df.loc[invalid_indices, value_column] = np.nan
                    
                    logger.info(f"Marked {len(invalid_indices)} invalid values as NaN for {variable}")
            
            return validated_df
            
        except Exception as e:
            logger.error(f"Error in data validation: {e}")
            raise 