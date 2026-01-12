"""
Map Visualizer for HAB Forecasting System

This module handles map visualization for HAB forecasting results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MapVisualizer:
    """
    Creates map visualizations for HAB forecasting results.
    
    This class handles the creation of probability maps,
    risk level maps, and other spatial visualizations.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the MapVisualizer.
        
        Args:
            config: Configuration dictionary with visualization parameters
        """
        self.config = config
        self.colormap = config.get('colormap', 'RdYlBu_r')
        self.risk_levels = config.get('risk_levels', {
            'low': [0, 0.25],
            'medium': [0.25, 0.5],
            'high': [0.5, 0.75],
            'extreme': [0.75, 1.0]
        })
        
    def create_probability_map(self, data: pd.DataFrame,
                             lat_column: str = 'lat',
                             lon_column: str = 'lon',
                             value_column: str = 'probability',
                             title: str = 'HAB Probability Map') -> plt.Figure:
        """
        Create a probability map visualization.
        
        Args:
            data: DataFrame with spatial data
            lat_column: Name of latitude column
            lon_column: Name of longitude column
            value_column: Name of probability column
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create scatter plot
            scatter = ax.scatter(data[lon_column], data[lat_column], 
                               c=data[value_column], cmap=self.colormap,
                               s=50, alpha=0.7)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Probability')
            
            # Customize plot
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            logger.info(f"Created probability map with {len(data)} points")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating probability map: {e}")
            raise
    
    def create_risk_level_map(self, data: pd.DataFrame,
                             lat_column: str = 'lat',
                             lon_column: str = 'lon',
                             value_column: str = 'probability') -> plt.Figure:
        """
        Create a risk level map visualization.
        
        Args:
            data: DataFrame with spatial data
            lat_column: Name of latitude column
            lon_column: Name of longitude column
            value_column: Name of probability column
            
        Returns:
            Matplotlib figure
        """
        try:
            # Assign risk levels
            data['risk_level'] = pd.cut(data[value_column], 
                                      bins=[0, 0.25, 0.5, 0.75, 1.0],
                                      labels=['Low', 'Medium', 'High', 'Extreme'])
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create scatter plot with risk levels
            colors = ['green', 'yellow', 'orange', 'red']
            for i, level in enumerate(['Low', 'Medium', 'High', 'Extreme']):
                level_data = data[data['risk_level'] == level]
                if len(level_data) > 0:
                    ax.scatter(level_data[lon_column], level_data[lat_column],
                             c=colors[i], s=50, alpha=0.7, label=level)
            
            # Customize plot
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('HAB Risk Level Map')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            logger.info(f"Created risk level map")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating risk level map: {e}")
            raise
    
    def create_interactive_map(self, data: pd.DataFrame,
                             lat_column: str = 'lat',
                             lon_column: str = 'lon',
                             value_column: str = 'probability',
                             center: Tuple[float, float] = (33.5, 133.5)) -> folium.Map:
        """
        Create an interactive map using Folium.
        
        Args:
            data: DataFrame with spatial data
            lat_column: Name of latitude column
            lon_column: Name of longitude column
            value_column: Name of probability column
            center: Center coordinates (lat, lon)
            
        Returns:
            Folium map object
        """
        try:
            # Create base map
            m = folium.Map(location=center, zoom_start=8)
            
            # Add markers for each point
            for idx, row in data.iterrows():
                # Determine color based on probability
                if row[value_column] < 0.25:
                    color = 'green'
                elif row[value_column] < 0.5:
                    color = 'yellow'
                elif row[value_column] < 0.75:
                    color = 'orange'
                else:
                    color = 'red'
                
                # Add marker
                folium.CircleMarker(
                    location=[row[lat_column], row[lon_column]],
                    radius=5,
                    popup=f"Probability: {row[value_column]:.3f}",
                    color=color,
                    fill=True
                ).add_to(m)
            
            logger.info(f"Created interactive map with {len(data)} points")
            return m
            
        except Exception as e:
            logger.error(f"Error creating interactive map: {e}")
            raise
    
    def save_map(self, fig: plt.Figure, filepath: str, dpi: int = 300):
        """
        Save map to file.
        
        Args:
            fig: Matplotlib figure
            filepath: Path to save the map
            dpi: Resolution for saving
        """
        try:
            fig.savefig(filepath, bbox_inches='tight', dpi=dpi)
            logger.info(f"Map saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving map: {e}")
            raise 