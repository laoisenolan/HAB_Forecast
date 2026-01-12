"""
Weekly Aggregator for HAB Forecasting System

This module handles weekly aggregation of environmental data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class WeeklyAggregator:
    """
    Aggregates daily environmental data to weekly averages.
    
    This class handles the aggregation of various environmental variables
    from daily to weekly time scales for HAB forecasting.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the WeeklyAggregator.
        
        Args:
            config: Configuration dictionary with aggregation parameters
        """
        self.config = config
        self.aggregation_methods = config.get('aggregation_methods', {
            'temperature': 'mean',
            'salinity': 'mean', 
            'chlorophyll': 'mean',
            'nutrients': 'mean',
            'wind_speed': 'mean',
            'wind_direction': 'circular_mean',
            'precipitation': 'sum',
            'solar_radiation': 'mean'
        })
        
    def aggregate_weekly(self, data: pd.DataFrame, 
                        date_column: str = 'date',
                        variable_column: str = 'variable',
                        value_column: str = 'value') -> pd.DataFrame:
        """
        Aggregate daily data to weekly averages.
        
        Args:
            data: DataFrame with daily environmental data
            date_column: Name of the date column
            variable_column: Name of the variable column
            value_column: Name of the value column
            
        Returns:
            DataFrame with weekly aggregated data
        """
        try:
            # Ensure date column is datetime
            data[date_column] = pd.to_datetime(data[date_column])
            
            # Add week column
            data['week'] = data[date_column].dt.to_period('W')
            
            # Group by week and variable, then aggregate
            weekly_data = []
            
            for (week, variable), group in data.groupby(['week', variable_column]):
                if variable in self.aggregation_methods:
                    method = self.aggregation_methods[variable]
                    
                    if method == 'circular_mean':
                        # Handle circular variables like wind direction
                        mean_value = self._circular_mean(group[value_column])
                    elif method == 'sum':
                        mean_value = group[value_column].sum()
                    else:
                        mean_value = group[value_column].mean()
                        
                    weekly_data.append({
                        'week': week,
                        'variable': variable,
                        'value': mean_value,
                        'count': len(group)
                    })
            
            result = pd.DataFrame(weekly_data)
            result['week_start'] = result['week'].dt.start_time
            
            logger.info(f"Aggregated {len(data)} daily records to {len(result)} weekly records")
            return result
            
        except Exception as e:
            logger.error(f"Error in weekly aggregation: {e}")
            raise
    
    def _circular_mean(self, values: pd.Series) -> float:
        """
        Calculate circular mean for directional data.
        
        Args:
            values: Series of directional values (e.g., wind direction)
            
        Returns:
            Circular mean in degrees
        """
        # Convert to radians
        radians = np.radians(values)
        
        # Calculate mean of sin and cos
        mean_sin = np.sin(radians).mean()
        mean_cos = np.cos(radians).mean()
        
        # Convert back to degrees
        mean_angle = np.degrees(np.arctan2(mean_sin, mean_cos))
        
        # Ensure positive angle
        if mean_angle < 0:
            mean_angle += 360
            
        return mean_angle
    
    def aggregate_by_region(self, data: pd.DataFrame, 
                           region_column: str = 'region') -> pd.DataFrame:
        """
        Aggregate data by region and week.
        
        Args:
            data: DataFrame with environmental data
            region_column: Name of the region column
            
        Returns:
            DataFrame with regional weekly aggregated data
        """
        try:
            # Group by region, week, and variable
            grouped = data.groupby([region_column, 'week', 'variable'])
            
            weekly_regional = []
            
            for (region, week, variable), group in grouped:
                if variable in self.aggregation_methods:
                    method = self.aggregation_methods[variable]
                    
                    if method == 'circular_mean':
                        mean_value = self._circular_mean(group['value'])
                    elif method == 'sum':
                        mean_value = group['value'].sum()
                    else:
                        mean_value = group['value'].mean()
                        
                    weekly_regional.append({
                        'region': region,
                        'week': week,
                        'variable': variable,
                        'value': mean_value,
                        'count': len(group)
                    })
            
            result = pd.DataFrame(weekly_regional)
            result['week_start'] = result['week'].dt.start_time
            
            logger.info(f"Aggregated data for {result['region'].nunique()} regions")
            return result
            
        except Exception as e:
            logger.error(f"Error in regional aggregation: {e}")
            raise 