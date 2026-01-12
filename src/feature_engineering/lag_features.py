"""
Lag Feature Generator for HAB Forecasting System

This module handles lag feature generation for time series data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class LagFeatureGenerator:
    """
    Generates lag features for time series data.
    
    This class handles the creation of lagged variables for
    time series forecasting in HAB prediction.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the LagFeatureGenerator.
        
        Args:
            config: Configuration dictionary with lag parameters
        """
        self.config = config
        self.lag_periods = config.get('lag_periods', [1, 7, 14, 30])  # days
        self.variables = config.get('variables', [
            'temperature', 'salinity', 'chlorophyll', 'nutrients',
            'wind_speed', 'precipitation', 'solar_radiation'
        ])
        
    def generate_lag_features(self, data: pd.DataFrame,
                             date_column: str = 'date',
                             variable_column: str = 'variable',
                             value_column: str = 'value') -> pd.DataFrame:
        """
        Generate lag features for time series data.
        
        Args:
            data: DataFrame with time series data
            date_column: Name of the date column
            variable_column: Name of the variable column
            value_column: Name of the value column
            
        Returns:
            DataFrame with lag features
        """
        try:
            # Ensure date column is datetime and sort
            data[date_column] = pd.to_datetime(data[date_column])
            data = data.sort_values([variable_column, date_column])
            
            # Create lag features for each variable
            lagged_data = []
            
            for variable in data[variable_column].unique():
                var_data = data[data[variable_column] == variable].copy()
                
                # Generate lag features
                for lag in self.lag_periods:
                    lag_col_name = f"{variable}_lag_{lag}d"
                    var_data[lag_col_name] = var_data[value_column].shift(lag)
                
                lagged_data.append(var_data)
            
            result = pd.concat(lagged_data, ignore_index=True)
            
            logger.info(f"Generated lag features for {len(self.lag_periods)} periods")
            return result
            
        except Exception as e:
            logger.error(f"Error in lag feature generation: {e}")
            raise
    
    def generate_rolling_features(self, data: pd.DataFrame,
                                window_sizes: List[int] = [7, 14, 30],
                                date_column: str = 'date',
                                variable_column: str = 'variable',
                                value_column: str = 'value') -> pd.DataFrame:
        """
        Generate rolling window features.
        
        Args:
            data: DataFrame with time series data
            window_sizes: List of window sizes in days
            date_column: Name of the date column
            variable_column: Name of the variable column
            value_column: Name of the value column
            
        Returns:
            DataFrame with rolling features
        """
        try:
            # Ensure date column is datetime and sort
            data[date_column] = pd.to_datetime(data[date_column])
            data = data.sort_values([variable_column, date_column])
            
            # Create rolling features for each variable
            rolling_data = []
            
            for variable in data[variable_column].unique():
                var_data = data[data[variable_column] == variable].copy()
                
                # Generate rolling features
                for window in window_sizes:
                    # Rolling mean
                    mean_col = f"{variable}_rolling_mean_{window}d"
                    var_data[mean_col] = var_data[value_column].rolling(window=window, min_periods=1).mean()
                    
                    # Rolling std
                    std_col = f"{variable}_rolling_std_{window}d"
                    var_data[std_col] = var_data[value_column].rolling(window=window, min_periods=1).std()
                    
                    # Rolling min
                    min_col = f"{variable}_rolling_min_{window}d"
                    var_data[min_col] = var_data[value_column].rolling(window=window, min_periods=1).min()
                    
                    # Rolling max
                    max_col = f"{variable}_rolling_max_{window}d"
                    var_data[max_col] = var_data[value_column].rolling(window=window, min_periods=1).max()
                
                rolling_data.append(var_data)
            
            result = pd.concat(rolling_data, ignore_index=True)
            
            logger.info(f"Generated rolling features for {len(window_sizes)} window sizes")
            return result
            
        except Exception as e:
            logger.error(f"Error in rolling feature generation: {e}")
            raise 