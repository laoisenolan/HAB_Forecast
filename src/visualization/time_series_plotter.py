"""
Time Series Plotter for HAB Forecasting System

This module handles time series visualization for HAB forecasting results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TimeSeriesPlotter:
    """
    Creates time series visualizations for HAB forecasting results.
    
    This class handles the creation of time series plots,
    trend analysis, and temporal pattern visualization.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the TimeSeriesPlotter.
        
        Args:
            config: Configuration dictionary with visualization parameters
        """
        self.config = config
        self.figure_size = config.get('figure_size', (12, 8))
        self.color_palette = config.get('color_palette', 'viridis')
        
    def plot_time_series(self, data: pd.DataFrame,
                        time_column: str = 'date',
                        value_column: str = 'value',
                        title: str = 'Time Series Plot') -> plt.Figure:
        """
        Create a time series plot.
        
        Args:
            data: DataFrame with time series data
            time_column: Name of time column
            value_column: Name of value column
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        try:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Convert time column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
                data[time_column] = pd.to_datetime(data[time_column])
            
            # Sort by time
            data_sorted = data.sort_values(time_column)
            
            # Create line plot
            ax.plot(data_sorted[time_column], data_sorted[value_column], 
                   linewidth=2, alpha=0.8)
            
            # Customize plot
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            logger.info(f"Created time series plot with {len(data)} points")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating time series plot: {e}")
            raise
    
    def plot_prediction_vs_actual(self, data: pd.DataFrame,
                                 time_column: str = 'date',
                                 actual_column: str = 'actual',
                                 predicted_column: str = 'predicted',
                                 title: str = 'Prediction vs Actual') -> plt.Figure:
        """
        Create a prediction vs actual time series plot.
        
        Args:
            data: DataFrame with actual and predicted values
            time_column: Name of time column
            actual_column: Name of actual values column
            predicted_column: Name of predicted values column
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        try:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Convert time column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
                data[time_column] = pd.to_datetime(data[time_column])
            
            # Sort by time
            data_sorted = data.sort_values(time_column)
            
            # Create line plots
            ax.plot(data_sorted[time_column], data_sorted[actual_column], 
                   label='Actual', linewidth=2, alpha=0.8)
            ax.plot(data_sorted[time_column], data_sorted[predicted_column], 
                   label='Predicted', linewidth=2, alpha=0.8, linestyle='--')
            
            # Customize plot
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            logger.info(f"Created prediction vs actual plot")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating prediction vs actual plot: {e}")
            raise
    
    def plot_seasonal_patterns(self, data: pd.DataFrame,
                              time_column: str = 'date',
                              value_column: str = 'value',
                              period: str = 'month') -> plt.Figure:
        """
        Create a seasonal pattern plot.
        
        Args:
            data: DataFrame with time series data
            time_column: Name of time column
            value_column: Name of value column
            period: Seasonal period ('month', 'week', 'day')
            
        Returns:
            Matplotlib figure
        """
        try:
            # Convert time column to datetime
            if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
                data[time_column] = pd.to_datetime(data[time_column])
            
            # Extract seasonal component
            if period == 'month':
                data['season'] = data[time_column].dt.month
            elif period == 'week':
                data['season'] = data[time_column].dt.isocalendar().week
            elif period == 'day':
                data['season'] = data[time_column].dt.dayofyear
            
            # Calculate seasonal statistics
            seasonal_stats = data.groupby('season')[value_column].agg(['mean', 'std']).reset_index()
            
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Create seasonal plot
            ax.plot(seasonal_stats['season'], seasonal_stats['mean'], 
                   linewidth=2, alpha=0.8, label='Mean')
            
            # Add error bars
            ax.fill_between(seasonal_stats['season'], 
                          seasonal_stats['mean'] - seasonal_stats['std'],
                          seasonal_stats['mean'] + seasonal_stats['std'],
                          alpha=0.3, label='±1 Std')
            
            # Customize plot
            ax.set_xlabel(f'{period.capitalize()}')
            ax.set_ylabel('Value')
            ax.set_title(f'Seasonal Patterns ({period})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            logger.info(f"Created seasonal pattern plot for {period}")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating seasonal pattern plot: {e}")
            raise
    
    def plot_trend_analysis(self, data: pd.DataFrame,
                           time_column: str = 'date',
                           value_column: str = 'value',
                           window: int = 30) -> plt.Figure:
        """
        Create a trend analysis plot.
        
        Args:
            data: DataFrame with time series data
            time_column: Name of time column
            value_column: Name of value column
            window: Rolling window size
            
        Returns:
            Matplotlib figure
        """
        try:
            # Convert time column to datetime
            if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
                data[time_column] = pd.to_datetime(data[time_column])
            
            # Sort by time
            data_sorted = data.sort_values(time_column)
            
            # Calculate rolling statistics
            data_sorted['rolling_mean'] = data_sorted[value_column].rolling(window=window).mean()
            data_sorted['rolling_std'] = data_sorted[value_column].rolling(window=window).std()
            
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Plot original data
            ax.plot(data_sorted[time_column], data_sorted[value_column], 
                   alpha=0.5, label='Original', linewidth=1)
            
            # Plot rolling mean
            ax.plot(data_sorted[time_column], data_sorted['rolling_mean'], 
                   linewidth=2, label=f'{window}-period Rolling Mean')
            
            # Add confidence interval
            ax.fill_between(data_sorted[time_column],
                          data_sorted['rolling_mean'] - data_sorted['rolling_std'],
                          data_sorted['rolling_mean'] + data_sorted['rolling_std'],
                          alpha=0.3, label='±1 Std')
            
            # Customize plot
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.set_title('Trend Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            logger.info(f"Created trend analysis plot with {window}-period window")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating trend analysis plot: {e}")
            raise
    
    def save_plot(self, fig: plt.Figure, filepath: str, dpi: int = 300):
        """
        Save plot to file.
        
        Args:
            fig: Matplotlib figure
            filepath: Path to save the plot
            dpi: Resolution for saving
        """
        try:
            fig.savefig(filepath, bbox_inches='tight', dpi=dpi)
            logger.info(f"Plot saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
            raise 