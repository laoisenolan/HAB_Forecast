#!/usr/bin/env python3
"""
Advanced Biogeochemical Ratio Features for HAB Forecasting

This module creates sophisticated biogeochemical ratio features critical for
understanding nutrient dynamics and HAB occurrence patterns.

🎯 BIOGEOCHEMICAL FEATURES IMPLEMENTED:
- Redfield ratio analysis (N:P:Si = 16:1:16)
- Nutrient limitation indicators
- Stoichiometric imbalance features
- Nutrient availability indices
- Biogeochemical process indicators
- Nutrient competition features
- Ecosystem health metrics

🧪 LEVERAGES:
- Complete biogeochemical suite (8/8 variables working)
- Nutrient data: Nitrate, Phosphate, Silicate, Iron
- Biological data: Oxygen, Primary Production, Chlorophyll
- Phytoplankton carbon biomass data

Author: HAB Forecasting System
Date: August 6, 2025
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

class AdvancedBiogeochemicalFeatureEngineer:
    """
    Advanced biogeochemical feature engineering for HAB forecasting.
    
    Creates sophisticated nutrient and biogeochemical features including:
    1. Stoichiometric ratios and imbalances
    2. Nutrient limitation indicators
    3. Biogeochemical process metrics
    4. Ecosystem health indices
    5. HAB-relevant nutrient conditions
    """
    
    def __init__(self):
        """Initialize biogeochemical feature engineer"""
        
        # Redfield ratios (molar basis)
        self.redfield_ratios = {
            'C_N': 106/16,    # Carbon to Nitrogen
            'C_P': 106/1,     # Carbon to Phosphorus
            'N_P': 16/1,      # Nitrogen to Phosphorus (classic Redfield)
            'N_Si': 16/16,    # Nitrogen to Silicate (for diatoms)
            'Si_P': 16/1      # Silicate to Phosphorus
        }
        
        # Nutrient limitation thresholds (μM or μg/L depending on measurement)
        self.limitation_thresholds = {
            'nitrate': {'severe': 0.5, 'moderate': 2.0, 'sufficient': 5.0},
            'phosphate': {'severe': 0.05, 'moderate': 0.15, 'sufficient': 0.3},
            'silicate': {'severe': 1.0, 'moderate': 3.0, 'sufficient': 8.0},
            'iron': {'severe': 0.01, 'moderate': 0.05, 'sufficient': 0.2},
            'oxygen': {'severe': 2.0, 'moderate': 4.0, 'sufficient': 6.0}  # mg/L
        }
        
        # HAB species nutrient preferences
        self.species_nutrient_preferences = {
            'Karenia_mikimotoi': {
                'preferred_N_P': [12, 20],
                'N_limitation_sensitive': True,
                'P_limitation_tolerant': False,
                'Si_requirement': 'low'
            },
            'Heterosigma_akashiwo': {
                'preferred_N_P': [10, 18],
                'N_limitation_sensitive': True,
                'P_limitation_tolerant': True,
                'Si_requirement': 'low'
            },
            'Chattonella_antiqua': {
                'preferred_N_P': [14, 22],
                'N_limitation_sensitive': False,
                'P_limitation_tolerant': True,
                'Si_requirement': 'none'
            },
            'Skeletonema_costatum': {
                'preferred_N_P': [15, 25],
                'N_limitation_sensitive': True,
                'P_limitation_tolerant': False,
                'Si_requirement': 'high'
            }
        }
        
        logger.info("🧪 Advanced Biogeochemical Feature Engineer initialized")
        logger.info("⚗️ Using complete biogeochemical suite (8/8 variables)")
        
    def calculate_stoichiometric_ratios(self,
                                      nutrient_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Calculate stoichiometric ratios and deviations from Redfield ratios
        
        Args:
            nutrient_data: Dictionary of nutrient concentration series
            
        Returns:
            Dictionary of stoichiometric ratio features
        """
        ratio_features = {}
        
        # Basic nutrient ratios
        if 'nitrate' in nutrient_data and 'phosphate' in nutrient_data:
            n_data = nutrient_data['nitrate']
            p_data = nutrient_data['phosphate']
            
            # N:P ratio
            np_ratio = n_data / (p_data + 1e-6)  # Avoid division by zero
            ratio_features['N_P_ratio'] = np_ratio
            
            # Deviation from Redfield N:P ratio
            redfield_np = self.redfield_ratios['N_P']
            np_deviation = (np_ratio - redfield_np) / redfield_np
            ratio_features['N_P_deviation_from_redfield'] = np_deviation
            
            # Log-transformed ratio (for better distribution)
            ratio_features['log_N_P_ratio'] = np.log(np_ratio + 1e-6)
            
            # N:P balance index (0 = perfect Redfield, 1 = maximum imbalance)
            np_balance = 1 - np.exp(-np.abs(np_deviation))
            ratio_features['N_P_imbalance_index'] = np_balance
            
        # N:Si ratio (important for diatoms vs dinoflagellates)
        if 'nitrate' in nutrient_data and 'silicate' in nutrient_data:
            n_data = nutrient_data['nitrate']
            si_data = nutrient_data['silicate']
            
            nsi_ratio = n_data / (si_data + 1e-6)
            ratio_features['N_Si_ratio'] = nsi_ratio
            
            # Diatom favorability (lower N:Si favors diatoms)
            diatom_favorability = 1 / (1 + nsi_ratio / 2)  # Sigmoid-like function
            ratio_features['diatom_favorability_index'] = diatom_favorability
            
            # Dinoflagellate favorability (higher N:Si favors dinoflagellates)
            dinoflagellate_favorability = nsi_ratio / (nsi_ratio + 2)
            ratio_features['dinoflagellate_favorability_index'] = dinoflagellate_favorability
            
        # Si:P ratio
        if 'silicate' in nutrient_data and 'phosphate' in nutrient_data:
            si_data = nutrient_data['silicate']
            p_data = nutrient_data['phosphate']
            
            sip_ratio = si_data / (p_data + 1e-6)
            ratio_features['Si_P_ratio'] = sip_ratio
            
        # Calculate DIN:DIP ratio (Dissolved Inorganic Nitrogen : Dissolved Inorganic Phosphorus)
        # This is a key ecological indicator
        if 'nitrate' in nutrient_data and 'phosphate' in nutrient_data:
            # Assuming nitrate represents most of DIN in this system
            din_dip = np_ratio  # Same as N:P ratio in this case
            
            # Classify nutrient limitation based on DIN:DIP
            limitation_class = []
            for ratio_val in din_dip:
                if pd.isna(ratio_val):
                    limitation_class.append('unknown')
                elif ratio_val < 10:
                    limitation_class.append('N_limited')
                elif ratio_val > 20:
                    limitation_class.append('P_limited')
                else:
                    limitation_class.append('balanced')
                    
            ratio_features['nutrient_limitation_class'] = pd.Series(limitation_class, index=din_dip.index)
            
            # Binary indicators
            ratio_features['is_N_limited'] = (din_dip < 10).astype(int)
            ratio_features['is_P_limited'] = (din_dip > 20).astype(int)
            ratio_features['is_nutrient_balanced'] = ((din_dip >= 10) & (din_dip <= 20)).astype(int)
            
        return ratio_features
    
    def calculate_nutrient_limitation_indices(self,
                                            nutrient_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Calculate nutrient limitation indices for each major nutrient
        
        Args:
            nutrient_data: Dictionary of nutrient concentration series
            
        Returns:
            Dictionary of nutrient limitation features
        """
        limitation_features = {}
        
        for nutrient, thresholds in self.limitation_thresholds.items():
            if nutrient not in nutrient_data:
                continue
                
            conc_data = nutrient_data[nutrient]
            
            # Limitation severity index (0 = no limitation, 1 = severe limitation)
            limitation_severity = []
            for conc in conc_data:
                if pd.isna(conc):
                    limitation_severity.append(0.5)  # Neutral when unknown
                elif conc <= thresholds['severe']:
                    limitation_severity.append(1.0)  # Severe limitation
                elif conc <= thresholds['moderate']:
                    # Linear interpolation between severe and moderate
                    severity = 1.0 - (conc - thresholds['severe']) / (thresholds['moderate'] - thresholds['severe'])
                    limitation_severity.append(severity * 0.7)  # Scale to 0.7 max for moderate
                elif conc <= thresholds['sufficient']:
                    # Linear interpolation between moderate and sufficient
                    severity = 1.0 - (conc - thresholds['moderate']) / (thresholds['sufficient'] - thresholds['moderate'])
                    limitation_severity.append(severity * 0.3)  # Scale to 0.3 max for slight
                else:
                    limitation_severity.append(0.0)  # No limitation
                    
            limitation_features[f'{nutrient}_limitation_index'] = pd.Series(limitation_severity, index=conc_data.index)
            
            # Binary limitation indicators
            limitation_features[f'{nutrient}_severe_limitation'] = (conc_data <= thresholds['severe']).astype(int)
            limitation_features[f'{nutrient}_moderate_limitation'] = ((conc_data > thresholds['severe']) & 
                                                                   (conc_data <= thresholds['moderate'])).astype(int)
            limitation_features[f'{nutrient}_sufficient'] = (conc_data > thresholds['sufficient']).astype(int)
            
            # Relative availability (0-1 scale)
            max_sufficient = thresholds['sufficient'] * 2  # Define "abundant" as 2x sufficient
            relative_availability = np.minimum(1.0, conc_data / max_sufficient)
            limitation_features[f'{nutrient}_relative_availability'] = relative_availability
            
        # Overall limitation index (worst single nutrient)
        individual_limitations = [v for k, v in limitation_features.items() if k.endswith('_limitation_index')]
        if individual_limitations:
            overall_limitation = pd.concat(individual_limitations, axis=1).max(axis=1)
            limitation_features['overall_nutrient_limitation'] = overall_limitation
            
            # Count of limiting nutrients
            severe_limitations = [v for k, v in limitation_features.items() if k.endswith('_severe_limitation')]
            if severe_limitations:
                limiting_nutrient_count = pd.concat(severe_limitations, axis=1).sum(axis=1)
                limitation_features['limiting_nutrient_count'] = limiting_nutrient_count
                
        return limitation_features
    
    def calculate_biogeochemical_process_indicators(self,
                                                  biogeochem_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Calculate indicators of biogeochemical processes
        
        Args:
            biogeochem_data: Dictionary including nutrients, oxygen, primary production, etc.
            
        Returns:
            Dictionary of biogeochemical process features
        """
        process_features = {}
        
        # Primary production efficiency
        if 'primary_production' in biogeochem_data and 'chlorophyll' in biogeochem_data:
            pp = biogeochem_data['primary_production']
            chl = biogeochem_data['chlorophyll']
            
            # Production per unit chlorophyll (efficiency indicator)
            production_efficiency = pp / (chl + 1e-6)
            process_features['production_efficiency'] = production_efficiency
            
            # High efficiency indicator (may indicate stress conditions)
            efficiency_threshold = production_efficiency.quantile(0.8)
            process_features['high_production_efficiency'] = (production_efficiency > efficiency_threshold).astype(int)
            
        # Oxygen production/consumption balance
        if 'oxygen' in biogeochem_data:
            o2 = biogeochem_data['oxygen']
            
            # Oxygen saturation level (assuming saturation ~8 mg/L at typical conditions)
            o2_saturation = 8.0  # mg/L (approximate for coastal waters at 20°C)
            oxygen_deficit = o2_saturation - o2
            process_features['oxygen_deficit'] = oxygen_deficit
            
            # Hypoxic conditions
            process_features['is_hypoxic'] = (o2 < 2.0).astype(int)
            process_features['is_severely_hypoxic'] = (o2 < 1.0).astype(int)
            
            # Oxygen trends (production vs consumption)
            o2_change = o2.diff()
            process_features['oxygen_change_rate'] = o2_change
            process_features['oxygen_increasing'] = (o2_change > 0.1).astype(int)
            process_features['oxygen_decreasing'] = (o2_change < -0.1).astype(int)
            
        # Carbon to chlorophyll ratio
        if 'phytoplankton_carbon' in biogeochem_data and 'chlorophyll' in biogeochem_data:
            carbon = biogeochem_data['phytoplankton_carbon']
            chl = biogeochem_data['chlorophyll']
            
            c_chl_ratio = carbon / (chl + 1e-6)
            process_features['carbon_chlorophyll_ratio'] = c_chl_ratio
            
            # High C:Chl ratio may indicate nutrient stress
            high_c_chl_threshold = c_chl_ratio.quantile(0.75)
            process_features['high_carbon_chlorophyll_ratio'] = (c_chl_ratio > high_c_chl_threshold).astype(int)
            
        # Nutrient uptake potential
        if all(nutrient in biogeochem_data for nutrient in ['nitrate', 'phosphate']):
            n = biogeochem_data['nitrate']
            p = biogeochem_data['phosphate']
            
            # Total inorganic nutrient pool
            total_nutrients = n + p * 16  # Convert P to N-equivalent using Redfield ratio
            process_features['total_nutrient_pool'] = total_nutrients
            
            # Nutrient depletion index
            max_nutrients = total_nutrients.quantile(0.95)  # Use 95th percentile as "replete"
            depletion_index = 1 - (total_nutrients / max_nutrients)
            process_features['nutrient_depletion_index'] = np.maximum(0, depletion_index)
            
        # Remineralization vs uptake balance
        if 'primary_production' in biogeochem_data and 'nitrate' in biogeochem_data:
            pp = biogeochem_data['primary_production']
            n = biogeochem_data['nitrate']
            
            # Production to nutrient ratio (uptake pressure)
            uptake_pressure = pp / (n + 1e-6)
            process_features['nutrient_uptake_pressure'] = uptake_pressure
            
            # High uptake pressure may lead to rapid nutrient depletion
            high_uptake_threshold = uptake_pressure.quantile(0.8)
            process_features['high_uptake_pressure'] = (uptake_pressure > high_uptake_threshold).astype(int)
            
        return process_features
    
    def calculate_ecosystem_health_metrics(self,
                                         biogeochem_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Calculate ecosystem health and stability metrics
        
        Args:
            biogeochem_data: Dictionary of biogeochemical variables
            
        Returns:
            Dictionary of ecosystem health features
        """
        health_features = {}
        
        # Nutrient balance score (based on how close ratios are to Redfield)
        if all(nutrient in biogeochem_data for nutrient in ['nitrate', 'phosphate']):
            n = biogeochem_data['nitrate']
            p = biogeochem_data['phosphate']
            
            np_ratio = n / (p + 1e-6)
            redfield_np = self.redfield_ratios['N_P']
            
            # Balance score: 1 = perfect Redfield, 0 = maximum imbalance
            balance_score = np.exp(-np.abs(np_ratio - redfield_np) / redfield_np)
            health_features['nutrient_balance_score'] = balance_score
            
        # Biodiversity proxy (based on nutrient and production diversity)
        if 'chlorophyll' in biogeochem_data and 'primary_production' in biogeochem_data:
            chl = biogeochem_data['chlorophyll']
            pp = biogeochem_data['primary_production']
            
            # Calculate coefficient of variation over time windows
            for window in [7, 30, 90]:  # days
                chl_cv = chl.rolling(window=window).std() / (chl.rolling(window=window).mean() + 1e-6)
                pp_cv = pp.rolling(window=window).std() / (pp.rolling(window=window).mean() + 1e-6)
                
                # Average CV as biodiversity proxy
                avg_cv = (chl_cv + pp_cv) / 2
                health_features[f'biodiversity_proxy_{window}d'] = avg_cv
                
                # Stability index (inverse of variability)
                stability = 1 / (1 + avg_cv)
                health_features[f'ecosystem_stability_{window}d'] = stability
                
        # Eutrophication indicators
        if 'chlorophyll' in biogeochem_data:
            chl = biogeochem_data['chlorophyll']
            
            # Trophic state based on chlorophyll
            trophic_state = []
            for chl_val in chl:
                if pd.isna(chl_val):
                    trophic_state.append('unknown')
                elif chl_val <= 2.0:
                    trophic_state.append('oligotrophic')
                elif chl_val <= 6.0:
                    trophic_state.append('mesotrophic')
                elif chl_val <= 20.0:
                    trophic_state.append('eutrophic')
                else:
                    trophic_state.append('hypereutrophic')
                    
            health_features['trophic_state'] = pd.Series(trophic_state, index=chl.index)
            
            # Binary trophic indicators
            health_features['is_oligotrophic'] = (chl <= 2.0).astype(int)
            health_features['is_eutrophic'] = (chl > 6.0).astype(int)
            health_features['is_hypereutrophic'] = (chl > 20.0).astype(int)
            
        # Stress indicators
        stress_indicators = []
        
        if 'oxygen' in biogeochem_data:
            o2 = biogeochem_data['oxygen']
            stress_indicators.append((o2 < 4.0).astype(float))  # Oxygen stress
            
        if 'nitrate' in biogeochem_data:
            n = biogeochem_data['nitrate']
            stress_indicators.append((n < 1.0).astype(float))  # Nitrogen stress
            
        if 'phosphate' in biogeochem_data:
            p = biogeochem_data['phosphate']
            stress_indicators.append((p < 0.1).astype(float))  # Phosphorus stress
            
        if stress_indicators:
            # Overall stress index
            overall_stress = sum(stress_indicators) / len(stress_indicators)
            health_features['ecosystem_stress_index'] = overall_stress
            
            # Multiple stress conditions
            multiple_stress = (sum(stress_indicators) >= 2).astype(int)
            health_features['multiple_stress_conditions'] = multiple_stress
            
        return health_features
    
    def calculate_species_specific_nutrient_features(self,
                                                   nutrient_data: Dict[str, pd.Series],
                                                   species: str) -> Dict[str, pd.Series]:
        """
        Calculate species-specific nutrient features
        
        Args:
            nutrient_data: Dictionary of nutrient data
            species: HAB species name
            
        Returns:
            Dictionary of species-specific nutrient features
        """
        if species not in self.species_nutrient_preferences:
            logger.warning(f"No nutrient preferences for species {species}")
            return {}
            
        prefs = self.species_nutrient_preferences[species]
        species_features = {}
        
        # N:P ratio suitability for species
        if 'nitrate' in nutrient_data and 'phosphate' in nutrient_data:
            n = nutrient_data['nitrate']
            p = nutrient_data['phosphate']
            np_ratio = n / (p + 1e-6)
            
            preferred_np = prefs['preferred_N_P']
            
            # Suitability based on preferred N:P range
            suitability = []
            for ratio in np_ratio:
                if pd.isna(ratio):
                    suitability.append(0.5)
                elif preferred_np[0] <= ratio <= preferred_np[1]:
                    suitability.append(1.0)  # Optimal
                else:
                    # Linear decrease outside optimal range
                    if ratio < preferred_np[0]:
                        distance = preferred_np[0] - ratio
                        score = max(0, 1 - distance / preferred_np[0])
                    else:
                        distance = ratio - preferred_np[1]
                        score = max(0, 1 - distance / preferred_np[1])
                    suitability.append(score)
                    
            species_features[f'{species}_np_ratio_suitability'] = pd.Series(suitability, index=np_ratio.index)
            
        # Nitrogen limitation sensitivity
        if prefs['N_limitation_sensitive'] and 'nitrate' in nutrient_data:
            n = nutrient_data['nitrate']
            n_limitation = (n < 2.0).astype(float)  # Sensitive species threshold
            species_features[f'{species}_n_limitation_stress'] = n_limitation
            
        # Phosphorus limitation tolerance
        if not prefs['P_limitation_tolerant'] and 'phosphate' in nutrient_data:
            p = nutrient_data['phosphate']
            p_limitation = (p < 0.1).astype(float)
            species_features[f'{species}_p_limitation_stress'] = p_limitation
            
        # Silicon requirement
        if 'silicate' in nutrient_data:
            si = nutrient_data['silicate']
            si_requirement = prefs['Si_requirement']
            
            if si_requirement == 'high':
                si_adequacy = (si > 5.0).astype(int)
                species_features[f'{species}_si_adequacy'] = si_adequacy
            elif si_requirement == 'low':
                si_inhibition = (si > 10.0).astype(int)  # Too much Si may favor diatoms over this species
                species_features[f'{species}_si_competition'] = si_inhibition
                
        return species_features
    
    def create_comprehensive_biogeochemical_features(self,
                                                   biogeochem_data: Dict[str, pd.Series],
                                                   species: Optional[str] = None) -> pd.DataFrame:
        """
        Create comprehensive biogeochemical features from nutrient and biological data
        
        Args:
            biogeochem_data: Dictionary of biogeochemical variables
            species: Optional HAB species for species-specific features
            
        Returns:
            DataFrame with all biogeochemical features
        """
        logger.info("🧪 Creating comprehensive biogeochemical features...")
        logger.info(f"⚗️ Processing {len(biogeochem_data)} biogeochemical variables")
        
        # Determine common index
        if not biogeochem_data:
            logger.warning("No biogeochemical data provided")
            return pd.DataFrame()
            
        common_index = next(iter(biogeochem_data.values())).index
        feature_df = pd.DataFrame(index=common_index)
        
        # Stoichiometric ratios
        ratio_features = self.calculate_stoichiometric_ratios(biogeochem_data)
        for name, series in ratio_features.items():
            feature_df[name] = series
            
        # Nutrient limitation indices
        limitation_features = self.calculate_nutrient_limitation_indices(biogeochem_data)
        for name, series in limitation_features.items():
            feature_df[name] = series
            
        # Biogeochemical process indicators
        process_features = self.calculate_biogeochemical_process_indicators(biogeochem_data)
        for name, series in process_features.items():
            feature_df[name] = series
            
        # Ecosystem health metrics
        health_features = self.calculate_ecosystem_health_metrics(biogeochem_data)
        for name, series in health_features.items():
            feature_df[name] = series
            
        # Species-specific nutrient features
        if species:
            species_features = self.calculate_species_specific_nutrient_features(biogeochem_data, species)
            for name, series in species_features.items():
                feature_df[name] = series
                
        logger.info(f"✅ Created {len(feature_df.columns)} biogeochemical features")
        logger.info("⚗️ Complete nutrient stoichiometry and limitation analysis")
        
        return feature_df