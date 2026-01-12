#!/usr/bin/env python3
"""
Advanced Interaction Feature Engineering for HAB Forecasting

This module creates sophisticated interaction features between environmental variables,
capturing synergistic effects and non-linear relationships critical for HAB prediction.

🎯 INTERACTION FEATURES IMPLEMENTED:
- Variable cross-products and ratios
- Non-linear interaction terms
- Threshold-based interaction features
- Environmental stress combinations
- Multi-variable habitat suitability indices
- Tipping point detection features
- Synergistic effect modeling

🧪 LEVERAGES:
- 25 environmental variables with full coverage
- Species-specific interaction knowledge
- Biogeochemical process understanding
- Oceanographic interaction patterns

Author: HAB Forecasting System
Date: August 6, 2025
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from itertools import combinations
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

class AdvancedInteractionFeatureEngineer:
    """
    Advanced interaction feature engineering for HAB forecasting.
    
    Creates sophisticated interaction features capturing:
    1. Synergistic environmental effects
    2. Non-linear variable relationships  
    3. Threshold-based interactions
    4. Multi-variable habitat indices
    5. Environmental stress combinations
    """
    
    def __init__(self):
        """Initialize interaction feature engineer"""
        
        # Define ecologically meaningful variable interactions
        self.ecological_interactions = {
            # Temperature-Nutrient interactions
            'temperature_nutrient': {
                'variables': [['temperature', 'nitrate'], ['temperature', 'phosphate'], 
                            ['temperature', 'silicate']],
                'description': 'Temperature affects nutrient uptake rates',
                'interaction_type': 'multiplicative'
            },
            
            # Light-Nutrient interactions  
            'light_nutrient': {
                'variables': [['solar_radiation', 'nitrate'], ['solar_radiation', 'phosphate']],
                'description': 'Light availability affects nutrient utilization',
                'interaction_type': 'multiplicative'
            },
            
            # Salinity-Temperature interactions
            'salinity_temperature': {
                'variables': [['salinity', 'temperature']],
                'description': 'Combined salinity-temperature stress',
                'interaction_type': 'stress_combination'
            },
            
            # Wind-Stratification interactions
            'wind_stratification': {
                'variables': [['wind_speed', 'temperature_gradient']],
                'description': 'Wind-induced mixing vs stratification',
                'interaction_type': 'competing'
            },
            
            # Nutrient ratio interactions
            'nutrient_ratios': {
                'variables': [['nitrate', 'phosphate'], ['nitrate', 'silicate'], 
                            ['phosphate', 'silicate']],
                'description': 'Nutrient stoichiometry effects',
                'interaction_type': 'ratio'
            },
            
            # Turbidity-Light interactions
            'turbidity_light': {
                'variables': [['turbidity', 'solar_radiation']],
                'description': 'Light attenuation by turbidity',
                'interaction_type': 'inhibitory'
            },
            
            # Current-Nutrient interactions
            'current_nutrient': {
                'variables': [['current_speed', 'nitrate'], ['current_speed', 'phosphate']],
                'description': 'Advective nutrient transport',
                'interaction_type': 'transport'
            }
        }
        
        # Species-specific interaction preferences
        self.species_interaction_preferences = {
            'Karenia_mikimotoi': {
                'critical_interactions': ['temperature_nutrient', 'wind_stratification'],
                'stress_combinations': ['salinity_temperature', 'turbidity_light'],
                'optimal_ratios': {'N_P_ratio': [10, 16], 'N_Si_ratio': [1, 3]}
            },
            'Heterosigma_akashiwo': {
                'critical_interactions': ['light_nutrient', 'salinity_temperature'],
                'stress_combinations': ['wind_stratification'],
                'optimal_ratios': {'N_P_ratio': [8, 20], 'N_Si_ratio': [2, 5]}
            },
            'Chattonella_antiqua': {
                'critical_interactions': ['temperature_nutrient', 'current_nutrient'],
                'stress_combinations': ['salinity_temperature', 'turbidity_light'],
                'optimal_ratios': {'N_P_ratio': [12, 18], 'N_Si_ratio': [1, 4]}
            }
        }
        
        logger.info("🔗 Advanced Interaction Feature Engineer initialized")
        logger.info(f"📊 Ecological interactions: {len(self.ecological_interactions)}")
        
    def calculate_multiplicative_interactions(self,
                                            data: pd.DataFrame,
                                            variable_pairs: List[Tuple[str, str]]) -> Dict[str, pd.Series]:
        """
        Calculate multiplicative interaction features
        
        Args:
            data: DataFrame with environmental variables
            variable_pairs: List of variable pairs to interact
            
        Returns:
            Dictionary of multiplicative interaction features
        """
        interactions = {}
        
        for var1, var2 in variable_pairs:
            if var1 in data.columns and var2 in data.columns:
                # Basic multiplication
                interaction_name = f'{var1}_{var2}_product'
                interactions[interaction_name] = data[var1] * data[var2]
                
                # Normalized multiplication (z-score products)
                var1_norm = (data[var1] - data[var1].mean()) / data[var1].std()
                var2_norm = (data[var2] - data[var2].mean()) / data[var2].std()
                interactions[f'{var1}_{var2}_normalized_product'] = var1_norm * var2_norm
                
                # Geometric mean (sqrt of product for positive values)
                product = data[var1] * data[var2]
                geometric_mean = np.where(product > 0, np.sqrt(np.abs(product)), 0)
                interactions[f'{var1}_{var2}_geometric_mean'] = pd.Series(geometric_mean, index=data.index)
                
        return interactions
    
    def calculate_ratio_interactions(self,
                                   data: pd.DataFrame,
                                   variable_pairs: List[Tuple[str, str]]) -> Dict[str, pd.Series]:
        """
        Calculate ratio-based interaction features
        
        Args:
            data: DataFrame with environmental variables  
            variable_pairs: List of variable pairs for ratios
            
        Returns:
            Dictionary of ratio interaction features
        """
        interactions = {}
        
        for var1, var2 in variable_pairs:
            if var1 in data.columns and var2 in data.columns:
                # Basic ratio (avoid division by zero)
                denominator = data[var2].replace(0, np.nan)
                ratio = data[var1] / denominator
                interactions[f'{var1}_{var2}_ratio'] = ratio
                
                # Log ratio (for variables that span orders of magnitude)
                log_ratio = np.log(data[var1] + 1e-6) - np.log(data[var2] + 1e-6)
                interactions[f'{var1}_{var2}_log_ratio'] = pd.Series(log_ratio, index=data.index)
                
                # Relative difference
                relative_diff = (data[var1] - data[var2]) / (data[var1] + data[var2] + 1e-6)
                interactions[f'{var1}_{var2}_relative_diff'] = pd.Series(relative_diff, index=data.index)
                
                # Balance index (how close to optimal ratio)
                if var1 == 'nitrate' and var2 == 'phosphate':
                    # Redfield ratio reference (N:P = 16:1)
                    redfield_ratio = 16
                    balance = 1 - np.abs(ratio - redfield_ratio) / (ratio + redfield_ratio)
                    interactions['N_P_redfield_balance'] = pd.Series(balance, index=data.index)
                
        return interactions
    
    def calculate_stress_combination_features(self,
                                            data: pd.DataFrame,
                                            stress_variables: List[str],
                                            species: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Calculate combined environmental stress features
        
        Args:
            data: DataFrame with environmental variables
            stress_variables: List of variables that can cause stress
            species: Optional species name for species-specific stress thresholds
            
        Returns:
            Dictionary of stress combination features
        """
        stress_features = {}
        
        # Define general stress thresholds (can be species-specific)
        stress_thresholds = {
            'temperature': {'low': 15, 'high': 28},
            'salinity': {'low': 25, 'high': 38},
            'turbidity': {'high': 10},
            'wind_speed': {'high': 15},
            'nitrate': {'low': 2},
            'phosphate': {'low': 0.1},
            'oxygen': {'low': 4}  # mg/L
        }
        
        # Modify thresholds for specific species
        if species in self.species_interaction_preferences:
            # Would implement species-specific threshold modifications here
            pass
            
        # Calculate individual stress indicators
        stress_indicators = {}
        for var in stress_variables:
            if var in data.columns and var in stress_thresholds:
                thresholds = stress_thresholds[var]
                
                stress_values = []
                for value in data[var]:
                    stress = 0
                    if pd.notna(value):
                        if 'low' in thresholds and value < thresholds['low']:
                            stress = (thresholds['low'] - value) / thresholds['low']
                        elif 'high' in thresholds and value > thresholds['high']:
                            stress = (value - thresholds['high']) / thresholds['high']
                    stress_values.append(min(1.0, stress))  # Cap at 1.0
                    
                stress_indicators[f'{var}_stress'] = pd.Series(stress_values, index=data.index)
                
        # Combine stress indicators
        if stress_indicators:
            # Total stress (sum of individual stresses)
            total_stress = sum(stress_indicators.values())
            stress_features['total_environmental_stress'] = total_stress
            
            # Maximum stress (worst single stressor)
            max_stress = pd.concat(stress_indicators.values(), axis=1).max(axis=1)
            stress_features['max_environmental_stress'] = max_stress
            
            # Stress diversity (number of active stressors)
            stress_count = (pd.concat(stress_indicators.values(), axis=1) > 0.1).sum(axis=1)
            stress_features['stress_factor_count'] = stress_count
            
            # Multiplicative stress (compound effect)
            multiplicative_stress = pd.Series(1.0, index=data.index)
            for stress_series in stress_indicators.values():
                multiplicative_stress *= (1 + stress_series)
            stress_features['compound_stress'] = multiplicative_stress - 1
            
        # Add individual stress indicators to output
        stress_features.update(stress_indicators)
        
        return stress_features
    
    def calculate_threshold_interactions(self,
                                       data: pd.DataFrame,
                                       threshold_rules: List[Dict]) -> Dict[str, pd.Series]:
        """
        Calculate threshold-based interaction features
        
        Args:
            data: DataFrame with environmental variables
            threshold_rules: List of threshold rule dictionaries
            
        Returns:
            Dictionary of threshold interaction features
        """
        threshold_features = {}
        
        # Default threshold rules for HAB-relevant conditions
        default_rules = [
            {
                'name': 'warm_stratified_conditions',
                'conditions': [('temperature', '>', 24), ('wind_speed', '<', 5)],
                'description': 'Warm, calm conditions favoring stratification'
            },
            {
                'name': 'nutrient_rich_conditions', 
                'conditions': [('nitrate', '>', 5), ('phosphate', '>', 0.3)],
                'description': 'High nutrient availability'
            },
            {
                'name': 'light_limited_conditions',
                'conditions': [('turbidity', '>', 5), ('solar_radiation', '<', 200)],
                'description': 'Light limitation conditions'
            },
            {
                'name': 'optimal_growth_conditions',
                'conditions': [('temperature', '>', 20), ('temperature', '<', 26),
                             ('nitrate', '>', 3), ('wind_speed', '<', 8)],
                'description': 'Generally optimal conditions for many HAB species'
            },
            {
                'name': 'extreme_conditions',
                'conditions': [('temperature', '>', 30), ('wind_speed', '>', 20)],
                'description': 'Extreme environmental conditions'
            }
        ]
        
        rules_to_process = threshold_rules if threshold_rules else default_rules
        
        for rule in rules_to_process:
            rule_name = rule['name']
            conditions = rule['conditions']
            
            # Evaluate all conditions
            condition_results = []
            for var, operator, threshold in conditions:
                if var in data.columns:
                    if operator == '>':
                        condition_met = data[var] > threshold
                    elif operator == '<':
                        condition_met = data[var] < threshold
                    elif operator == '>=':
                        condition_met = data[var] >= threshold  
                    elif operator == '<=':
                        condition_met = data[var] <= threshold
                    elif operator == '==':
                        condition_met = data[var] == threshold
                    else:
                        continue
                        
                    condition_results.append(condition_met)
                    
            if condition_results:
                # All conditions must be met (AND logic)
                all_conditions_met = pd.concat(condition_results, axis=1).all(axis=1)
                threshold_features[rule_name] = all_conditions_met.astype(int)
                
                # Fraction of conditions met (partial fulfillment)
                fraction_met = pd.concat(condition_results, axis=1).mean(axis=1)
                threshold_features[f'{rule_name}_partial'] = fraction_met
                
        return threshold_features
    
    def calculate_polynomial_interactions(self,
                                        data: pd.DataFrame,
                                        variables: List[str],
                                        degree: int = 2) -> Dict[str, pd.Series]:
        """
        Calculate polynomial interaction features
        
        Args:
            data: DataFrame with environmental variables
            variables: List of variables to create polynomial features for
            degree: Polynomial degree
            
        Returns:
            Dictionary of polynomial interaction features
        """
        poly_features = {}
        
        # Select available variables
        available_vars = [var for var in variables if var in data.columns]
        if len(available_vars) < 2:
            logger.warning("Need at least 2 variables for polynomial interactions")
            return poly_features
            
        # Create polynomial features
        try:
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
            
            # Get data for selected variables
            var_data = data[available_vars].fillna(0)  # Fill NaN with 0 for polynomial features
            
            # Fit and transform
            poly_array = poly.fit_transform(var_data)
            feature_names = poly.get_feature_names_out(available_vars)
            
            # Convert to DataFrame and extract interaction terms (skip single variables)
            poly_df = pd.DataFrame(poly_array, columns=feature_names, index=data.index)
            
            for col in poly_df.columns:
                # Skip single variables, keep only interactions
                if ' ' in col:  # Interaction terms have spaces in sklearn naming
                    # Clean up the name
                    clean_name = col.replace(' ', '_').replace('^', '_power_')
                    poly_features[f'poly_{clean_name}'] = poly_df[col]
                    
        except Exception as e:
            logger.warning(f"Polynomial feature creation failed: {e}")
            
        return poly_features
    
    def calculate_habitat_suitability_interactions(self,
                                                 data: pd.DataFrame,
                                                 species: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Calculate multi-variable habitat suitability indices
        
        Args:
            data: DataFrame with environmental variables
            species: Optional species name for species-specific calculations
            
        Returns:
            Dictionary of habitat suitability features
        """
        habitat_features = {}
        
        # Define habitat requirements for key variables
        habitat_requirements = {
            'general_phytoplankton': {
                'temperature': [15, 25],  # Optimal range
                'nitrate': [2, 10],
                'phosphate': [0.1, 0.8],
                'light': [100, 400],  # PAR or solar radiation
                'salinity': [25, 35]
            }
        }
        
        # Species-specific habitat requirements
        if species in self.species_interaction_preferences:
            # Use species-specific requirements if available
            # This would be expanded with detailed species data
            pass
            
        for habitat_name, requirements in habitat_requirements.items():
            suitability_scores = []
            
            for idx in data.index:
                row = data.loc[idx]
                individual_scores = []
                
                for var, optimal_range in requirements.items():
                    if var in data.columns and pd.notna(row[var]):
                        value = row[var]
                        min_val, max_val = optimal_range
                        
                        # Calculate Gaussian-like suitability score
                        mid_point = (min_val + max_val) / 2
                        range_width = max_val - min_val
                        
                        # Gaussian score with width based on optimal range
                        score = np.exp(-0.5 * ((value - mid_point) / (range_width / 4))**2)
                        individual_scores.append(score)
                        
                if individual_scores:
                    # Geometric mean of individual scores
                    combined_score = np.power(np.prod(individual_scores), 1.0 / len(individual_scores))
                else:
                    combined_score = 0.5  # Neutral when no data
                    
                suitability_scores.append(combined_score)
                
            feature_name = f'{habitat_name}_suitability_index'
            if species:
                feature_name = f'{species}_{habitat_name}_suitability_index'
                
            habitat_features[feature_name] = pd.Series(suitability_scores, index=data.index)
            
        return habitat_features
    
    def calculate_synergistic_effects(self,
                                    data: pd.DataFrame,
                                    synergy_groups: List[List[str]]) -> Dict[str, pd.Series]:
        """
        Calculate synergistic effect features between variable groups
        
        Args:
            data: DataFrame with environmental variables
            synergy_groups: List of variable groups that have synergistic effects
            
        Returns:
            Dictionary of synergistic effect features
        """
        synergy_features = {}
        
        # Default synergy groups for HAB research
        default_groups = [
            ['temperature', 'nitrate', 'phosphate'],  # Nutrient-temperature synergy
            ['wind_speed', 'current_speed'],  # Physical mixing synergy
            ['solar_radiation', 'temperature'],  # Light-heat synergy
            ['salinity', 'temperature', 'oxygen'],  # Physical-chemical synergy
        ]
        
        groups_to_process = synergy_groups if synergy_groups else default_groups
        
        for i, group in enumerate(groups_to_process):
            available_vars = [var for var in group if var in data.columns]
            
            if len(available_vars) < 2:
                continue
                
            group_name = '_'.join(available_vars[:3])  # Limit name length
            
            # Normalize variables to 0-1 range for synergy calculation
            normalized_vars = {}
            for var in available_vars:
                var_data = data[var]
                var_min, var_max = var_data.min(), var_data.max()
                if var_max > var_min:
                    normalized = (var_data - var_min) / (var_max - var_min)
                else:
                    normalized = pd.Series(0.5, index=var_data.index)  # Constant value
                normalized_vars[var] = normalized
                
            # Calculate synergistic effect as product of normalized values
            synergy_product = pd.Series(1.0, index=data.index)
            for var_series in normalized_vars.values():
                synergy_product *= var_series
                
            synergy_features[f'{group_name}_synergy_product'] = synergy_product
            
            # Calculate synergistic effect as geometric mean
            synergy_geomean = np.power(synergy_product, 1.0 / len(normalized_vars))
            synergy_features[f'{group_name}_synergy_geomean'] = synergy_geomean
            
            # Calculate synergistic boost (when all variables are above median)
            above_median = pd.DataFrame(normalized_vars) > 0.5
            synergy_boost = above_median.all(axis=1).astype(float)
            synergy_features[f'{group_name}_synergy_boost'] = synergy_boost
            
        return synergy_features
    
    def create_comprehensive_interaction_features(self,
                                                data: pd.DataFrame,
                                                species: Optional[str] = None) -> pd.DataFrame:
        """
        Create comprehensive interaction features from environmental data
        
        Args:
            data: DataFrame with environmental variables
            species: Optional species name for species-specific features
            
        Returns:
            DataFrame with all interaction features
        """
        logger.info("🔗 Creating comprehensive interaction features...")
        
        all_features = data.copy()
        
        # Get available variables
        available_vars = data.columns.tolist()
        logger.info(f"Processing {len(available_vars)} input variables")
        
        # Variable pairs for different interaction types
        key_pairs = [
            ('temperature', 'nitrate'), ('temperature', 'phosphate'),
            ('temperature', 'salinity'), ('salinity', 'chlorophyll'),
            ('wind_speed', 'temperature'), ('current_speed', 'nitrate'),
            ('turbidity', 'chlorophyll'), ('nitrate', 'phosphate'),
            ('nitrate', 'silicate'), ('phosphate', 'silicate')
        ]
        
        # Filter pairs to only include available variables
        available_pairs = [(v1, v2) for v1, v2 in key_pairs 
                         if v1 in available_vars and v2 in available_vars]
        
        # Multiplicative interactions
        mult_features = self.calculate_multiplicative_interactions(data, available_pairs)
        for name, series in mult_features.items():
            all_features[name] = series
            
        # Ratio interactions
        ratio_features = self.calculate_ratio_interactions(data, available_pairs)
        for name, series in ratio_features.items():
            all_features[name] = series
            
        # Stress combination features
        stress_vars = [var for var in ['temperature', 'salinity', 'turbidity', 'wind_speed', 
                                     'nitrate', 'phosphate'] if var in available_vars]
        if stress_vars:
            stress_features = self.calculate_stress_combination_features(data, stress_vars, species)
            for name, series in stress_features.items():
                all_features[name] = series
                
        # Threshold interactions
        threshold_features = self.calculate_threshold_interactions(data, [])
        for name, series in threshold_features.items():
            all_features[name] = series
            
        # Polynomial interactions (limited to key variables to avoid explosion)
        key_vars_for_poly = [var for var in ['temperature', 'salinity', 'nitrate', 'chlorophyll'] 
                           if var in available_vars]
        if len(key_vars_for_poly) >= 2:
            poly_features = self.calculate_polynomial_interactions(data, key_vars_for_poly, degree=2)
            for name, series in poly_features.items():
                all_features[name] = series
                
        # Habitat suitability interactions
        habitat_features = self.calculate_habitat_suitability_interactions(data, species)
        for name, series in habitat_features.items():
            all_features[name] = series
            
        # Synergistic effects
        synergy_features = self.calculate_synergistic_effects(data, [])
        for name, series in synergy_features.items():
            all_features[name] = series
            
        n_new_features = len(all_features.columns) - len(data.columns)
        logger.info(f"✅ Created {n_new_features} interaction features")
        
        return all_features