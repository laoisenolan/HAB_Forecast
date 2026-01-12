"""
SHAP Analyzer for HAB Forecasting System

This module handles SHAP (SHapley Additive exPlanations) analysis for
model interpretability and feature importance visualization.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP library
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

from src.utils.config_loader import ConfigLoader, get_data_dir

logger = logging.getLogger(__name__)

class SHAPAnalyzer:
    """SHAP analyzer for model interpretability"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the SHAP analyzer"""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library is required for SHAP analysis")
            
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_model_params_config()
        self.data_dir = get_data_dir()
        
        # Get SHAP configuration
        self.shap_config = self.config.get("shap_analysis", {})
        
        # Initialize components
        self.explainer = None
        self.shap_values = None
        self.background_data = None
        
        # Analysis results
        self.feature_importance = {}
        self.interaction_analysis = {}
        self.interpretation_results = {}
        
        # Output directories
        self.plots_dir = self.data_dir / "plots"
        self.results_dir = self.data_dir / "results"
        
        for directory in [self.plots_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
    def create_explainer(self, model, X_train: pd.DataFrame, explainer_type: str = None) -> shap.TreeExplainer:
        """
        Create SHAP explainer
        
        Args:
            model: Trained XGBoost model
            X_train: Training features
            explainer_type: Type of explainer ("tree", "kernel", "linear")
            
        Returns:
            SHAP explainer
        """
        logger.info("Creating SHAP explainer")
        
        try:
            if explainer_type is None:
                explainer_type = self.shap_config.get("explainer_type", "tree")
                
            if explainer_type == "tree":
                # For tree-based models (XGBoost, Random Forest)
                self.explainer = shap.TreeExplainer(model)
            elif explainer_type == "kernel":
                # For any model (slower but more general)
                background_samples = self.shap_config.get("background_samples", 1000)
                background_data = X_train.sample(n=min(background_samples, len(X_train)))
                self.explainer = shap.KernelExplainer(model.predict_proba, background_data)
            elif explainer_type == "linear":
                # For linear models
                self.explainer = shap.LinearExplainer(model, X_train)
            else:
                logger.warning(f"Unknown explainer type: {explainer_type}, using tree explainer")
                self.explainer = shap.TreeExplainer(model)
                
            logger.info(f"SHAP explainer created: {explainer_type}")
            return self.explainer
            
        except Exception as e:
            logger.error(f"SHAP explainer creation failed: {e}")
            raise
            
    def compute_shap_values(self, X_data: pd.DataFrame, max_samples: int = None) -> np.ndarray:
        """
        Compute SHAP values for given data
        
        Args:
            X_data: Input features
            max_samples: Maximum number of samples to analyze
            
        Returns:
            SHAP values
        """
        logger.info("Computing SHAP values")
        
        try:
            if max_samples is None:
                max_samples = self.shap_config.get("max_samples", 5000)
                
            # Sample data if too large
            if len(X_data) > max_samples:
                X_sample = X_data.sample(n=max_samples, random_state=42)
                logger.info(f"Sampled {max_samples} samples for SHAP analysis")
            else:
                X_sample = X_data
                
            # Compute SHAP values
            self.shap_values = self.explainer.shap_values(X_sample)
            
            # For binary classification, get positive class SHAP values
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]  # Positive class
                
            logger.info(f"SHAP values computed: {self.shap_values.shape}")
            return self.shap_values
            
        except Exception as e:
            logger.error(f"SHAP values computation failed: {e}")
            raise
            
    def analyze_feature_importance(self, X_data: pd.DataFrame, top_n: int = None) -> Dict[str, float]:
        """
        Analyze feature importance using SHAP values
        
        Args:
            X_data: Input features
            top_n: Number of top features to return
            
        Returns:
            Feature importance dictionary
        """
        logger.info("Analyzing feature importance")
        
        try:
            if self.shap_values is None:
                self.compute_shap_values(X_data)
                
            if top_n is None:
                top_n = self.shap_config.get("top_n_features", 20)
                
            # Calculate mean absolute SHAP values
            mean_shap_values = np.abs(self.shap_values).mean(axis=0)
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature in enumerate(X_data.columns):
                feature_importance[feature] = float(mean_shap_values[i])
                
            # Sort by importance
            sorted_importance = dict(sorted(feature_importance.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            # Get top N features
            top_features = dict(list(sorted_importance.items())[:top_n])
            
            self.feature_importance = top_features
            
            logger.info(f"Feature importance analysis complete. Top feature: {list(top_features.keys())[0]}")
            return top_features
            
        except Exception as e:
            logger.error(f"Feature importance analysis failed: {e}")
            raise
            
    def analyze_interactions(self, X_data: pd.DataFrame, max_interactions: int = None) -> Dict[str, float]:
        """
        Analyze feature interactions using SHAP
        
        Args:
            X_data: Input features
            max_interactions: Maximum number of interactions to analyze
            
        Returns:
            Interaction analysis results
        """
        logger.info("Analyzing feature interactions")
        
        try:
            if max_interactions is None:
                max_interactions = self.shap_config.get("max_interactions", 10)
                
            if self.shap_values is None:
                self.compute_shap_values(X_data)
                
            # Get top features for interaction analysis
            top_features = list(self.feature_importance.keys())[:min(10, len(self.feature_importance))]
            
            # Compute interaction values
            interaction_values = shap.TreeExplainer.shap_interaction_values(
                self.explainer.model, X_data[top_features]
            )
            
            # Calculate interaction strengths
            interaction_strengths = {}
            for i in range(len(top_features)):
                for j in range(i+1, len(top_features)):
                    interaction_name = f"{top_features[i]} × {top_features[j]}"
                    strength = np.abs(interaction_values[:, i, j]).mean()
                    interaction_strengths[interaction_name] = float(strength)
                    
            # Sort by strength
            sorted_interactions = dict(sorted(interaction_strengths.items(), 
                                            key=lambda x: x[1], reverse=True))
            
            # Get top interactions
            top_interactions = dict(list(sorted_interactions.items())[:max_interactions])
            
            self.interaction_analysis = top_interactions
            
            logger.info(f"Interaction analysis complete. Top interaction: {list(top_interactions.keys())[0]}")
            return top_interactions
            
        except Exception as e:
            logger.error(f"Interaction analysis failed: {e}")
            return {}
            
    def plot_feature_importance(self, save_plot: bool = True) -> plt.Figure:
        """
        Plot feature importance
        
        Args:
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        logger.info("Creating feature importance plot")
        
        try:
            if not self.feature_importance:
                logger.warning("No feature importance data available")
                return None
                
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            features = list(self.feature_importance.keys())
            importance_values = list(self.feature_importance.values())
            
            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            ax.barh(y_pos, importance_values)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Mean |SHAP value|')
            ax.set_title('Feature Importance (SHAP)')
            ax.invert_yaxis()
            
            # Add value labels
            for i, v in enumerate(importance_values):
                ax.text(v + 0.001, i, f'{v:.4f}', va='center')
                
            plt.tight_layout()
            
            if save_plot:
                plot_path = self.plots_dir / "feature_importance_shap.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved: {plot_path}")
                
            return fig
            
        except Exception as e:
            logger.error(f"Feature importance plotting failed: {e}")
            return None
            
    def plot_summary_plot(self, X_data: pd.DataFrame, save_plot: bool = True) -> plt.Figure:
        """
        Create SHAP summary plot
        
        Args:
            X_data: Input features
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        logger.info("Creating SHAP summary plot")
        
        try:
            if self.shap_values is None:
                self.compute_shap_values(X_data)
                
            # Create summary plot
            fig = plt.figure(figsize=(12, 8))
            shap.summary_plot(self.shap_values, X_data, show=False)
            plt.title('SHAP Summary Plot')
            plt.tight_layout()
            
            if save_plot:
                plot_path = self.plots_dir / "shap_summary_plot.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved: {plot_path}")
                
            return fig
            
        except Exception as e:
            logger.error(f"SHAP summary plot failed: {e}")
            return None
            
    def plot_waterfall_plot(self, X_data: pd.DataFrame, sample_idx: int = 0, save_plot: bool = True) -> plt.Figure:
        """
        Create SHAP waterfall plot for a specific sample
        
        Args:
            X_data: Input features
            sample_idx: Index of sample to analyze
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Creating SHAP waterfall plot for sample {sample_idx}")
        
        try:
            if self.shap_values is None:
                self.compute_shap_values(X_data)
                
            # Create waterfall plot
            fig = plt.figure(figsize=(12, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=self.shap_values[sample_idx],
                    base_values=self.explainer.expected_value,
                    data=X_data.iloc[sample_idx]
                ),
                show=False
            )
            plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}')
            plt.tight_layout()
            
            if save_plot:
                plot_path = self.plots_dir / f"shap_waterfall_sample_{sample_idx}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP waterfall plot saved: {plot_path}")
                
            return fig
            
        except Exception as e:
            logger.error(f"SHAP waterfall plot failed: {e}")
            return None
            
    def plot_force_plot(self, X_data: pd.DataFrame, sample_idx: int = 0, save_plot: bool = True) -> plt.Figure:
        """
        Create SHAP force plot for a specific sample
        
        Args:
            X_data: Input features
            sample_idx: Index of sample to analyze
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Creating SHAP force plot for sample {sample_idx}")
        
        try:
            if self.shap_values is None:
                self.compute_shap_values(X_data)
                
            # Create force plot
            fig = plt.figure(figsize=(12, 8))
            shap.force_plot(
                self.explainer.expected_value,
                self.shap_values[sample_idx],
                X_data.iloc[sample_idx],
                show=False
            )
            plt.title(f'SHAP Force Plot - Sample {sample_idx}')
            plt.tight_layout()
            
            if save_plot:
                plot_path = self.plots_dir / f"shap_force_sample_{sample_idx}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP force plot saved: {plot_path}")
                
            return fig
            
        except Exception as e:
            logger.error(f"SHAP force plot failed: {e}")
            return None
            
    def plot_interaction_plot(self, X_data: pd.DataFrame, feature1: str, feature2: str, 
                            save_plot: bool = True) -> plt.Figure:
        """
        Create SHAP interaction plot for two features
        
        Args:
            X_data: Input features
            feature1: First feature name
            feature2: Second feature name
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Creating SHAP interaction plot for {feature1} and {feature2}")
        
        try:
            if self.shap_values is None:
                self.compute_shap_values(X_data)
                
            # Create interaction plot
            fig = plt.figure(figsize=(12, 8))
            shap.dependence_plot(
                feature1, self.shap_values, X_data,
                interaction_index=feature2, show=False
            )
            plt.title(f'SHAP Interaction Plot: {feature1} vs {feature2}')
            plt.tight_layout()
            
            if save_plot:
                plot_path = self.plots_dir / f"shap_interaction_{feature1}_{feature2}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP interaction plot saved: {plot_path}")
                
            return fig
            
        except Exception as e:
            logger.error(f"SHAP interaction plot failed: {e}")
            return None
            
    def create_interpretability_report(self, X_data: pd.DataFrame, y_data: pd.Series = None) -> Dict[str, Any]:
        """
        Create comprehensive interpretability report
        
        Args:
            X_data: Input features
            y_data: Target values (optional)
            
        Returns:
            Interpretability report
        """
        logger.info("Creating interpretability report")
        
        try:
            # Compute SHAP values
            if self.shap_values is None:
                self.compute_shap_values(X_data)
                
            # Analyze feature importance
            feature_importance = self.analyze_feature_importance(X_data)
            
            # Analyze interactions
            interactions = self.analyze_interactions(X_data)
            
            # Create report
            report = {
                "analysis_date": datetime.now().isoformat(),
                "data_shape": X_data.shape,
                "feature_importance": feature_importance,
                "interactions": interactions,
                "shap_values_shape": self.shap_values.shape,
                "expected_value": float(self.explainer.expected_value),
                "top_features": list(feature_importance.keys())[:10],
                "top_interactions": list(interactions.keys())[:5]
            }
            
            # Add target analysis if available
            if y_data is not None:
                report["target_distribution"] = y_data.value_counts().to_dict()
                report["target_balance"] = float(y_data.mean())
                
            self.interpretation_results = report
            
            # Save report
            report_path = self.results_dir / "shap_interpretability_report.json"
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Interpretability report saved: {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"Interpretability report creation failed: {e}")
            raise
            
    def save_analysis_results(self, filename: str = None) -> Path:
        """Save SHAP analysis results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shap_analysis_results_{timestamp}.json"
            
        results_path = self.results_dir / filename
        
        # Prepare results for saving
        results = {
            "feature_importance": self.feature_importance,
            "interaction_analysis": self.interaction_analysis,
            "interpretation_results": self.interpretation_results,
            "analysis_date": datetime.now().isoformat()
        }
        
        # Save results
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"SHAP analysis results saved: {results_path}")
        return results_path 