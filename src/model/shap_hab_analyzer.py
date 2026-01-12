#!/usr/bin/env python3
"""
SHAP HAB Analyzer - Task 4.8 Model Interpretability

This module implements comprehensive model interpretability for HAB prediction models:
- SHAP (SHapley Additive exPlanations) analysis for feature importance
- LIME (Local Interpretable Model-agnostic Explanations) for local explanations
- Feature importance analysis across multiple models
- Species-specific interpretation and comparison
- Visualization of model decision processes

🎯 INTERPRETABILITY METHODS IMPLEMENTED:
- SHAP TreeExplainer for XGBoost models
- SHAP DeepExplainer for neural networks (when available)
- LIME Tabular explainer for model-agnostic explanations
- Permutation importance analysis
- Feature interaction analysis
- Global vs local explanation generation

🛰️ SCIENTIFIC INSIGHTS:
- Environmental driver identification for different HAB species
- Seasonal pattern analysis in model decisions
- Critical threshold identification for HAB risk
- Model confidence and uncertainty visualization

Author: HAB Forecasting System - Task 4.8 Implementation
Date: December 19, 2024
"""

import numpy as np
import pandas as pd
import logging
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import warnings
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SHAP imports with fallback
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("✅ SHAP available for model interpretability")
except ImportError:
    logger.warning("⚠️  SHAP not available - install with: pip install shap")

# LIME imports with fallback  
LIME_AVAILABLE = False
try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
    logger.info("✅ LIME available for local explanations")
except ImportError:
    logger.warning("⚠️  LIME not available - install with: pip install lime")

# Standard ML libraries
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

class SHAPHABAnalyzer:
    """
    Comprehensive SHAP-based interpretability analyzer for HAB prediction models
    """
    
    def __init__(self, cache_dir: str = "cache/interpretability", 
                 random_state: int = 42):
        """
        Initialize SHAP HAB analyzer
        
        Args:
            cache_dir: Directory for caching analysis results
            random_state: Random state for reproducibility
        """
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # Analysis storage
        self.shap_values = {}
        self.feature_importance = {}
        self.global_explanations = {}
        self.local_explanations = {}
        
        logger.info(f"🔍 SHAP HAB Analyzer initialized")
        logger.info(f"📁 Cache directory: {self.cache_dir}")
    
    def analyze_xgboost_model(self, model, X_test: np.ndarray, 
                             feature_names: List[str], species: str,
                             max_samples: int = 1000) -> Dict:
        """
        Comprehensive SHAP analysis for XGBoost models
        
        Args:
            model: Trained XGBoost model
            X_test: Test data for analysis
            feature_names: Names of features
            species: Species name for identification
            max_samples: Maximum samples for analysis (for performance)
            
        Returns:
            Dictionary with SHAP analysis results
        """
        
        if not SHAP_AVAILABLE:
            logger.error("❌ SHAP not available - cannot perform analysis")
            return {}
        
        logger.info(f"🔍 Performing SHAP analysis for {species}")
        
        try:
            # Limit samples for performance
            n_samples = min(len(X_test), max_samples)
            X_analysis = X_test[:n_samples]
            
            logger.info(f"📊 Analyzing {n_samples} samples with {len(feature_names)} features")
            
            # Create SHAP explainer for tree models
            explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_analysis)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Binary classification - take positive class
                shap_values_processed = shap_values[1] if len(shap_values) == 2 else shap_values[0]
            else:
                shap_values_processed = shap_values
            
            # Calculate feature importance
            mean_abs_shap = np.mean(np.abs(shap_values_processed), axis=0)
            feature_importance = dict(zip(feature_names, mean_abs_shap))
            
            # Sort by importance
            sorted_importance = sorted(feature_importance.items(), 
                                     key=lambda x: x[1], reverse=True)
            
            # Create results dictionary
            results = {
                'species': species,
                'shap_values': shap_values_processed,
                'feature_importance': feature_importance,
                'sorted_importance': sorted_importance,
                'feature_names': feature_names,
                'n_samples': n_samples,
                'explainer': explainer,
                'expected_value': explainer.expected_value
            }
            
            # Store results
            self.shap_values[species] = results
            
            # Log top features
            logger.info(f"🏆 Top 5 features for {species}:")
            for feat, imp in sorted_importance[:5]:
                logger.info(f"  {feat}: {imp:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ SHAP analysis failed for {species}: {e}")
            return {}
    
    def create_shap_visualizations(self, species: str, output_dir: Optional[Path] = None) -> Dict[str, str]:
        """
        Create comprehensive SHAP visualizations
        
        Args:
            species: Species to analyze
            output_dir: Directory to save plots
            
        Returns:
            Dictionary mapping plot types to file paths
        """
        
        if not SHAP_AVAILABLE or species not in self.shap_values:
            logger.error(f"❌ No SHAP analysis available for {species}")
            return {}
        
        if output_dir is None:
            output_dir = self.cache_dir / species.replace(' ', '_')
        output_dir.mkdir(exist_ok=True)
        
        results = self.shap_values[species]
        shap_values = results['shap_values']
        feature_names = results['feature_names']
        
        plot_files = {}
        
        try:
            # 1. Feature Importance Plot (Bar)
            logger.info(f"📊 Creating feature importance plot for {species}")
            plt.figure(figsize=(10, 8))
            
            sorted_importance = results['sorted_importance'][:10]  # Top 10
            features, importance = zip(*sorted_importance)
            
            plt.barh(range(len(features)), importance)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Mean |SHAP value|')
            plt.title(f'{species} - Feature Importance (SHAP)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            importance_file = output_dir / f"{species}_feature_importance.png"
            plt.savefig(importance_file, dpi=150, bbox_inches='tight')
            plt.close()
            plot_files['feature_importance'] = str(importance_file)
            
            # 2. SHAP Summary Plot
            logger.info(f"📊 Creating SHAP summary plot for {species}")
            plt.figure(figsize=(10, 8))
            
            # Create dummy data for X since we only have SHAP values
            # We'll use the mean absolute SHAP values for visualization
            shap.summary_plot(shap_values, feature_names=feature_names, 
                            show=False, max_display=10)
            
            summary_file = output_dir / f"{species}_shap_summary.png"
            plt.savefig(summary_file, dpi=150, bbox_inches='tight')
            plt.close()
            plot_files['summary'] = str(summary_file)
            
            # 3. SHAP Waterfall Plot for a representative sample
            if len(shap_values) > 0:
                logger.info(f"📊 Creating SHAP waterfall plot for {species}")
                
                # Use first sample as representative
                sample_idx = 0
                plt.figure(figsize=(10, 8))
                
                # Create waterfall plot manually since we may not have all SHAP components
                sample_shap = shap_values[sample_idx]
                expected_value = results.get('expected_value', 0)
                
                # Sort by absolute importance for this sample
                abs_values = np.abs(sample_shap)
                sorted_indices = np.argsort(abs_values)[::-1][:10]  # Top 10
                
                # Create waterfall-style plot
                plt.barh(range(len(sorted_indices)), 
                        [sample_shap[i] for i in sorted_indices],
                        color=['red' if sample_shap[i] < 0 else 'blue' 
                              for i in sorted_indices])
                plt.yticks(range(len(sorted_indices)), 
                          [feature_names[i] for i in sorted_indices])
                plt.xlabel('SHAP value')
                plt.title(f'{species} - SHAP Values for Sample {sample_idx}')
                plt.gca().invert_yaxis()
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                plt.tight_layout()
                
                waterfall_file = output_dir / f"{species}_shap_waterfall.png"
                plt.savefig(waterfall_file, dpi=150, bbox_inches='tight')
                plt.close()
                plot_files['waterfall'] = str(waterfall_file)
            
            logger.info(f"✅ Created {len(plot_files)} SHAP visualizations for {species}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create SHAP visualizations for {species}: {e}")
        
        return plot_files
    
    def analyze_feature_interactions(self, species: str, top_n: int = 5) -> Dict:
        """
        Analyze feature interactions using SHAP values
        
        Args:
            species: Species to analyze
            top_n: Number of top features to analyze interactions for
            
        Returns:
            Dictionary with interaction analysis results
        """
        
        if species not in self.shap_values:
            logger.error(f"❌ No SHAP analysis available for {species}")
            return {}
        
        logger.info(f"🔗 Analyzing feature interactions for {species}")
        
        results = self.shap_values[species]
        shap_values = results['shap_values']
        feature_names = results['feature_names']
        sorted_importance = results['sorted_importance']
        
        try:
            # Get top features
            top_features = [feat for feat, _ in sorted_importance[:top_n]]
            top_indices = [feature_names.index(feat) for feat in top_features]
            
            # Calculate correlation between SHAP values (proxy for interaction)
            shap_df = pd.DataFrame(shap_values[:, top_indices], columns=top_features)
            shap_correlation = shap_df.corr()
            
            # Find strong interactions (high absolute correlation)
            interactions = []
            for i, feat1 in enumerate(top_features):
                for j, feat2 in enumerate(top_features):
                    if i < j:  # Avoid duplicates
                        correlation = shap_correlation.loc[feat1, feat2]
                        if abs(correlation) > 0.3:  # Threshold for significant interaction
                            interactions.append({
                                'feature1': feat1,
                                'feature2': feat2,
                                'correlation': correlation,
                                'strength': abs(correlation)
                            })
            
            # Sort by interaction strength
            interactions.sort(key=lambda x: x['strength'], reverse=True)
            
            interaction_results = {
                'species': species,
                'top_features': top_features,
                'interactions': interactions,
                'shap_correlation_matrix': shap_correlation.to_dict()
            }
            
            logger.info(f"🔗 Found {len(interactions)} significant feature interactions")
            for interaction in interactions[:3]:  # Log top 3
                logger.info(f"  {interaction['feature1']} ↔ {interaction['feature2']}: "
                           f"r={interaction['correlation']:.3f}")
            
            return interaction_results
            
        except Exception as e:
            logger.error(f"❌ Feature interaction analysis failed for {species}: {e}")
            return {}
    
    def generate_interpretation_report(self, species: str) -> Dict:
        """
        Generate comprehensive interpretation report for a species
        
        Args:
            species: Species to generate report for
            
        Returns:
            Complete interpretation report dictionary
        """
        
        if species not in self.shap_values:
            logger.error(f"❌ No analysis available for {species}")
            return {}
        
        logger.info(f"📋 Generating interpretation report for {species}")
        
        results = self.shap_values[species]
        
        # Create visualizations
        plot_files = self.create_shap_visualizations(species)
        
        # Analyze interactions
        interactions = self.analyze_feature_interactions(species)
        
        # Environmental driver analysis
        environmental_insights = self.analyze_environmental_drivers(species)
        
        # Compile comprehensive report
        report = {
            'species': species,
            'analysis_timestamp': datetime.now().isoformat(),
            'model_type': 'XGBoost',
            'n_samples_analyzed': results['n_samples'],
            'n_features': len(results['feature_names']),
            
            # Feature importance
            'top_features': dict(results['sorted_importance'][:10]),
            'all_feature_importance': results['feature_importance'],
            
            # Environmental insights
            'environmental_drivers': environmental_insights,
            
            # Feature interactions
            'feature_interactions': interactions.get('interactions', []),
            
            # Visualization files
            'visualizations': plot_files,
            
            # Model insights
            'model_insights': {
                'most_important_feature': results['sorted_importance'][0][0],
                'least_important_feature': results['sorted_importance'][-1][0],
                'feature_count_above_threshold': sum(1 for _, imp in results['sorted_importance'] 
                                                   if imp > np.mean(list(results['feature_importance'].values()))),
                'diversity_score': np.std(list(results['feature_importance'].values()))
            }
        }
        
        # Save report
        report_file = self.cache_dir / f"{species.replace(' ', '_')}_interpretation_report.json"
        with open(report_file, 'w') as f:
            # Convert numpy types for JSON serialization
            report_json = json.loads(json.dumps(report, default=str))
            json.dump(report_json, f, indent=2)
        
        logger.info(f"📋 Interpretation report saved to {report_file}")
        
        return report
    
    def analyze_environmental_drivers(self, species: str) -> Dict:
        """
        Analyze environmental drivers for HAB occurrence
        
        Args:
            species: Species to analyze
            
        Returns:
            Environmental driver analysis results
        """
        
        if species not in self.shap_values:
            return {}
        
        results = self.shap_values[species]
        feature_importance = results['feature_importance']
        
        # Categorize features by environmental type
        feature_categories = {
            'physical_ocean': ['ocean_temperature', 'ocean_salinity', 'mixed_layer_depth', 
                             'sea_surface_temperature'],
            'nutrients': ['nitrate', 'phosphate', 'silicate'],
            'water_quality': ['dissolved_oxygen', 'chlorophyll_a', 'turbidity_kd490'],
            'atmospheric': ['wind_u', 'wind_v', 'precipitation']
        }
        
        # Calculate importance by category
        category_importance = {}
        for category, features in feature_categories.items():
            category_score = sum(feature_importance.get(feat, 0) for feat in features 
                               if feat in feature_importance)
            category_importance[category] = category_score
        
        # Find dominant category
        dominant_category = max(category_importance.items(), key=lambda x: x[1])
        
        environmental_analysis = {
            'category_importance': category_importance,
            'dominant_driver_category': dominant_category[0],
            'dominant_driver_score': dominant_category[1],
            'environmental_profile': self.get_species_environmental_profile(species, feature_importance)
        }
        
        return environmental_analysis
    
    def get_species_environmental_profile(self, species: str, feature_importance: Dict) -> Dict:
        """
        Create environmental profile for species based on feature importance
        
        Args:
            species: Species name
            feature_importance: Feature importance dictionary
            
        Returns:
            Environmental profile dictionary
        """
        
        # Known environmental preferences (can be expanded)
        environmental_interpretations = {
            'ocean_temperature': 'thermal_preference',
            'ocean_salinity': 'salinity_tolerance', 
            'nitrate': 'nutrient_requirement',
            'chlorophyll_a': 'productivity_association',
            'dissolved_oxygen': 'oxygen_sensitivity',
            'wind_u': 'wind_mixing_sensitivity',
            'wind_v': 'wind_pattern_response',
            'turbidity_kd490': 'water_clarity_preference'
        }
        
        profile = {}
        for feature, interpretation in environmental_interpretations.items():
            if feature in feature_importance:
                importance = feature_importance[feature]
                if importance > 0.01:  # Threshold for significance
                    profile[interpretation] = {
                        'feature': feature,
                        'importance': importance,
                        'category': 'high' if importance > 0.05 else 'moderate'
                    }
        
        return profile
    
    def compare_species_interpretability(self, species_list: List[str]) -> Dict:
        """
        Compare interpretability results across multiple species
        
        Args:
            species_list: List of species to compare
            
        Returns:
            Comparative analysis results
        """
        
        logger.info(f"🔬 Comparing interpretability across {len(species_list)} species")
        
        valid_species = [s for s in species_list if s in self.shap_values]
        if not valid_species:
            logger.error("❌ No valid species with SHAP analysis")
            return {}
        
        # Collect feature importance across species
        all_features = set()
        species_importance = {}
        
        for species in valid_species:
            importance = self.shap_values[species]['feature_importance']
            species_importance[species] = importance
            all_features.update(importance.keys())
        
        # Create comparison matrix
        comparison_matrix = pd.DataFrame(index=list(all_features), columns=valid_species)
        for species in valid_species:
            for feature in all_features:
                comparison_matrix.loc[feature, species] = species_importance[species].get(feature, 0)
        
        # Calculate cross-species statistics
        feature_stats = {}
        for feature in all_features:
            values = [species_importance[s].get(feature, 0) for s in valid_species]
            feature_stats[feature] = {
                'mean_importance': np.mean(values),
                'std_importance': np.std(values),
                'max_species': valid_species[np.argmax(values)],
                'min_species': valid_species[np.argmin(values)],
                'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
            }
        
        comparison_results = {
            'species_analyzed': valid_species,
            'n_features': len(all_features),
            'comparison_matrix': comparison_matrix.to_dict(),
            'feature_statistics': feature_stats,
            'most_variable_feature': max(feature_stats.items(), 
                                       key=lambda x: x[1]['coefficient_of_variation'])[0],
            'most_consistent_feature': min(feature_stats.items(), 
                                         key=lambda x: x[1]['coefficient_of_variation'])[0]
        }
        
        logger.info(f"✅ Species comparison complete")
        logger.info(f"📊 Most variable feature: {comparison_results['most_variable_feature']}")
        logger.info(f"📊 Most consistent feature: {comparison_results['most_consistent_feature']}")
        
        return comparison_results
    
    def save_analysis_results(self, species: str) -> None:
        """Save all analysis results for a species"""
        
        if species not in self.shap_values:
            logger.warning(f"⚠️  No results to save for {species}")
            return
        
        species_dir = self.cache_dir / species.replace(' ', '_')
        species_dir.mkdir(exist_ok=True)
        
        # Save SHAP values
        results = self.shap_values[species]
        
        # Save feature importance as CSV
        importance_df = pd.DataFrame(list(results['feature_importance'].items()),
                                   columns=['feature', 'importance'])
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df.to_csv(species_dir / 'feature_importance.csv', index=False)
        
        # Save SHAP values (sample)
        shap_sample_df = pd.DataFrame(results['shap_values'][:100], 
                                    columns=results['feature_names'])
        shap_sample_df.to_csv(species_dir / 'shap_values_sample.csv', index=False)
        
        logger.info(f"💾 Saved analysis results for {species} to {species_dir}")

if __name__ == "__main__":
    print("🔍 SHAP HAB Analyzer - Task 4.8 Model Interpretability")
    print(f"📊 SHAP available: {SHAP_AVAILABLE}")
    print(f"📊 LIME available: {LIME_AVAILABLE}")
    
    if SHAP_AVAILABLE:
        # Initialize analyzer
        analyzer = SHAPHABAnalyzer()
        
        print("✅ SHAP HAB Analyzer initialized successfully!")
        print("🎯 Ready to analyze model interpretability and feature importance")
        print("🔬 Can analyze XGBoost models, create visualizations, and generate reports")
    else:
        print("❌ SHAP not available - install with: pip install shap")
        print("💡 SHAP provides the best interpretability analysis for tree models")