"""
SHAP Visualizer for HAB Forecasting System

This module handles SHAP visualization for model interpretability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SHAPVisualizer:
    """
    Creates SHAP visualizations for model interpretability.
    
    This class handles the creation of SHAP plots,
    feature importance visualizations, and model explanations.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the SHAPVisualizer.
        
        Args:
            config: Configuration dictionary with visualization parameters
        """
        self.config = config
        self.figure_size = config.get('figure_size', (12, 8))
        self.max_features = config.get('max_features', 20)
        
    def plot_feature_importance(self, shap_values: np.ndarray,
                              feature_names: List[str],
                              title: str = 'SHAP Feature Importance') -> plt.Figure:
        """
        Create a SHAP feature importance plot.
        
        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        try:
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_shap
            }).sort_values('importance', ascending=True)
            
            # Limit to top features
            if len(importance_df) > self.max_features:
                importance_df = importance_df.tail(self.max_features)
            
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Create horizontal bar plot
            y_pos = np.arange(len(importance_df))
            ax.barh(y_pos, importance_df['importance'])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(importance_df['feature'])
            
            # Customize plot
            ax.set_xlabel('Mean |SHAP value|')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            logger.info(f"Created SHAP feature importance plot for {len(importance_df)} features")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating SHAP feature importance plot: {e}")
            raise
    
    def plot_summary_plot(self, shap_values: np.ndarray,
                         X: np.ndarray,
                         feature_names: Optional[List[str]] = None,
                         max_display: int = 20) -> plt.Figure:
        """
        Create a SHAP summary plot.
        
        Args:
            shap_values: SHAP values array
            X: Feature matrix
            feature_names: List of feature names
            max_display: Maximum number of features to display
            
        Returns:
            Matplotlib figure
        """
        try:
            fig = plt.figure(figsize=self.figure_size)
            
            # Create SHAP summary plot
            shap.summary_plot(shap_values, X, feature_names=feature_names,
                            max_display=max_display, show=False)
            
            logger.info(f"Created SHAP summary plot")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {e}")
            raise
    
    def plot_waterfall_plot(self, shap_values: np.ndarray,
                           X: np.ndarray,
                           sample_idx: int = 0,
                           feature_names: Optional[List[str]] = None) -> plt.Figure:
        """
        Create a SHAP waterfall plot for a specific sample.
        
        Args:
            shap_values: SHAP values array
            X: Feature matrix
            sample_idx: Index of sample to explain
            feature_names: List of feature names
            
        Returns:
            Matplotlib figure
        """
        try:
            fig = plt.figure(figsize=self.figure_size)
            
            # Create SHAP waterfall plot (simplified version)
            # Note: This is a placeholder - actual SHAP waterfall plots require more setup
            logger.warning("SHAP waterfall plot requires additional setup - using placeholder")
            ax = plt.gca()
            ax.text(0.5, 0.5, 'SHAP Waterfall Plot\n(Placeholder)', 
                   ha='center', va='center', transform=ax.transAxes)
            
            logger.info(f"Created SHAP waterfall plot for sample {sample_idx}")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating SHAP waterfall plot: {e}")
            raise
    
    def plot_dependence_plot(self, shap_values: np.ndarray,
                           X: np.ndarray,
                           feature_idx: int,
                           feature_names: Optional[List[str]] = None) -> plt.Figure:
        """
        Create a SHAP dependence plot for a specific feature.
        
        Args:
            shap_values: SHAP values array
            X: Feature matrix
            feature_idx: Index of feature to plot
            feature_names: List of feature names
            
        Returns:
            Matplotlib figure
        """
        try:
            fig = plt.figure(figsize=self.figure_size)
            
            # Create SHAP dependence plot
            shap.dependence_plot(
                feature_idx,
                shap_values,
                X,
                feature_names=feature_names,
                show=False
            )
            
            logger.info(f"Created SHAP dependence plot for feature {feature_idx}")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating SHAP dependence plot: {e}")
            raise
    
    def plot_force_plot(self, shap_values: np.ndarray,
                       X: np.ndarray,
                       sample_idx: int = 0,
                       feature_names: Optional[List[str]] = None) -> plt.Figure:
        """
        Create a SHAP force plot for a specific sample.
        
        Args:
            shap_values: SHAP values array
            X: Feature matrix
            sample_idx: Index of sample to explain
            feature_names: List of feature names
            
        Returns:
            Matplotlib figure
        """
        try:
            fig = plt.figure(figsize=self.figure_size)
            
            # Create SHAP force plot (simplified version)
            # Note: This is a placeholder - actual SHAP force plots require more setup
            logger.warning("SHAP force plot requires additional setup - using placeholder")
            ax = plt.gca()
            ax.text(0.5, 0.5, 'SHAP Force Plot\n(Placeholder)', 
                   ha='center', va='center', transform=ax.transAxes)
            
            logger.info(f"Created SHAP force plot for sample {sample_idx}")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating SHAP force plot: {e}")
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
            logger.info(f"SHAP plot saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving SHAP plot: {e}")
            raise 