#!/usr/bin/env python3
"""
Advanced Turbidity-Based Water Quality Features for HAB Forecasting

This module leverages the BREAKTHROUGH turbidity data discovery to create sophisticated
water quality indices and features for HAB prediction.

🎯 TURBIDITY FEATURES IMPLEMENTED:
- Multi-source turbidity integration (3 breakthrough sources)
- Water quality classification indices  
- Light attenuation and availability features
- Sediment resuspension indicators
- Coastal runoff and pollution proxies
- Optical water type classification
- Turbidity-based stratification features

🛰️ BREAKTHROUGH DATA SOURCES:
- 🥇 Copernicus Daily Gap-Free Transparency (KD490) - 4km resolution
- 🥈 Copernicus Reflectance-Derived Turbidity (RRS bands) - 4km resolution  
- 🥉 Sentinel-2 High-Resolution Turbidity (B4/B2 ratio) - 10-20m resolution

Author: HAB Forecasting System
Date: August 6, 2025
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import ndimage, interpolate
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

logger = logging.getLogger(__name__)

class AdvancedTurbidityFeatureEngineer:
    """
    Advanced turbidity feature engineering leveraging breakthrough data sources.
    
    Creates sophisticated turbidity-based features using:
    1. Multi-source turbidity integration
    2. Water quality classification
    3. Light availability modeling  
    4. Optical water type analysis
    5. Turbidity-based environmental indicators
    """
    
    def __init__(self):
        """Initialize turbidity feature engineer"""
        
        # Turbidity data source specifications (from our breakthrough discovery)
        self.turbidity_sources = {
            'copernicus_transparency': {
                'variable': 'KD490',  # Diffuse attenuation coefficient
                'resolution': '4km',
                'coverage': '2025-07-17 to present',
                'description': 'Daily gap-free water transparency',
                'quality': 'highest',
                'unit': 'm^-1'
            },
            
            'copernicus_reflectance': {
                'variables': ['RRS443', 'RRS490', 'RRS555'],
                'resolution': '4km', 
                'coverage': '2025-05-01 to 2025-07-01',
                'description': 'Remote sensing reflectance for turbidity derivation',
                'quality': 'high',
                'unit': 'sr^-1'
            },
            
            'sentinel2_coastal': {
                'variables': ['B2', 'B4'],  # Blue and Red bands
                'resolution': '10-20m',
                'coverage': '2017-present',
                'description': 'Ultra-high resolution coastal turbidity',
                'quality': 'highest_resolution',
                'unit': 'reflectance'
            }
        }
        
        # Water quality classification thresholds based on literature
        self.water_quality_thresholds = {
            'kd490': {
                'oligotrophic': [0, 0.1],      # Very clear water
                'mesotrophic': [0.1, 0.3],     # Moderately clear
                'eutrophic': [0.3, 0.8],       # Turbid water
                'hypertrophic': [0.8, 2.0],    # Very turbid
                'sediment_laden': [2.0, np.inf] # Extremely turbid
            },
            
            'turbidity_ntu': {
                'excellent': [0, 4],
                'good': [4, 10], 
                'moderate': [10, 25],
                'poor': [25, 50],
                'very_poor': [50, np.inf]
            }
        }
        
        # HAB species turbidity preferences (based on literature)
        self.species_turbidity_preferences = {
            'Karenia_mikimotoi': {
                'optimal_kd490': [0.05, 0.2],
                'tolerance': 'low_turbidity',
                'light_requirement': 'high'
            },
            'Heterosigma_akashiwo': {
                'optimal_kd490': [0.1, 0.4], 
                'tolerance': 'moderate_turbidity',
                'light_requirement': 'moderate'
            },
            'Noctiluca_scintillans': {
                'optimal_kd490': [0.2, 0.6],
                'tolerance': 'high_turbidity',
                'light_requirement': 'low'
            },
            'Chattonella_antiqua': {
                'optimal_kd490': [0.08, 0.3],
                'tolerance': 'low_to_moderate',
                'light_requirement': 'high'
            }
        }
        
        logger.info("🌊 Advanced Turbidity Feature Engineer initialized")
        logger.info(f"📊 Turbidity sources: {len(self.turbidity_sources)}")
        logger.info("🥇 Using BREAKTHROUGH turbidity data (3 sources)")
        
    def integrate_multi_source_turbidity(self,
                                       kd490_data: Optional[pd.Series] = None,
                                       rrs_data: Optional[Dict[str, pd.Series]] = None,
                                       sentinel2_data: Optional[Dict[str, pd.Series]] = None) -> Dict[str, pd.Series]:
        """
        Integrate turbidity data from multiple breakthrough sources
        
        Args:
            kd490_data: KD490 diffuse attenuation coefficient data
            rrs_data: Dictionary of remote sensing reflectance data
            sentinel2_data: Dictionary of Sentinel-2 band data
            
        Returns:
            Dictionary of integrated turbidity features
        """
        turbidity_features = {}
        
        # Primary source: KD490 (best quality)
        if kd490_data is not None:
            turbidity_features['kd490_primary'] = kd490_data
            turbidity_features['water_transparency'] = 1 / (kd490_data + 0.01)  # Inverse relationship
            
            # Secchi depth estimation (empirical relationship)
            secchi_depth = 1.7 / (kd490_data + 0.1)  # Empirical formula
            turbidity_features['estimated_secchi_depth'] = secchi_depth
            
            # Light attenuation at different depths
            for depth in [1, 3, 5, 10]:  # meters
                light_fraction = np.exp(-kd490_data * depth)
                turbidity_features[f'light_at_{depth}m'] = light_fraction
                
        # Secondary source: RRS-derived turbidity
        if rrs_data is not None:
            if 'RRS490' in rrs_data and 'RRS555' in rrs_data:
                # Turbidity index from reflectance ratios
                rrs_490 = rrs_data['RRS490']
                rrs_555 = rrs_data['RRS555']
                
                # Simple turbidity proxy using reflectance ratio
                turbidity_proxy = rrs_555 / (rrs_490 + 1e-6)  # Avoid division by zero
                turbidity_features['rrs_turbidity_proxy'] = turbidity_proxy
                
                # Normalized difference turbidity index
                ndti = (rrs_555 - rrs_490) / (rrs_555 + rrs_490 + 1e-6)
                turbidity_features['normalized_difference_turbidity'] = ndti
                
            if 'RRS443' in rrs_data and 'RRS490' in rrs_data:
                # Blue-green ratio (sensitive to chlorophyll and turbidity)
                bg_ratio = rrs_data['RRS443'] / (rrs_data['RRS490'] + 1e-6)
                turbidity_features['blue_green_reflectance_ratio'] = bg_ratio
                
        # Tertiary source: Sentinel-2 high-resolution
        if sentinel2_data is not None:
            if 'B2' in sentinel2_data and 'B4' in sentinel2_data:
                # Red/Blue ratio for turbidity (the breakthrough method)
                b2_blue = sentinel2_data['B2']
                b4_red = sentinel2_data['B4']
                
                turbidity_rb_ratio = b4_red / (b2_blue + 1e-6)
                turbidity_features['sentinel2_turbidity_ratio'] = turbidity_rb_ratio
                
                # High-resolution turbidity index
                turbidity_index = (b4_red - b2_blue) / (b4_red + b2_blue + 1e-6)
                turbidity_features['sentinel2_turbidity_index'] = turbidity_index
                
                # Turbidity magnitude (total signal)
                turbidity_magnitude = np.sqrt(b4_red**2 + b2_blue**2)
                turbidity_features['sentinel2_turbidity_magnitude'] = turbidity_magnitude
                
        # Create unified turbidity estimate
        available_turbidity = []
        
        if 'kd490_primary' in turbidity_features:
            available_turbidity.append(turbidity_features['kd490_primary'])
            
        if 'rrs_turbidity_proxy' in turbidity_features:
            # Normalize RRS proxy to approximate KD490 scale
            rrs_normalized = turbidity_features['rrs_turbidity_proxy'] * 0.2  # Scaling factor
            available_turbidity.append(rrs_normalized)
            
        if 'sentinel2_turbidity_ratio' in turbidity_features:
            # Normalize Sentinel-2 ratio to approximate KD490 scale  
            s2_normalized = np.log(turbidity_features['sentinel2_turbidity_ratio'] + 1) * 0.3
            available_turbidity.append(s2_normalized)
            
        # Multi-source turbidity consensus
        if available_turbidity:
            if len(available_turbidity) == 1:
                turbidity_features['unified_turbidity'] = available_turbidity[0]
            else:
                # Weighted average (KD490 gets highest weight)
                weights = [0.7, 0.2, 0.1][:len(available_turbidity)]  # Prioritize KD490
                weighted_turbidity = sum(w * turb for w, turb in zip(weights, available_turbidity))
                turbidity_features['unified_turbidity'] = weighted_turbidity / sum(weights)
                
            # Data source diversity indicator
            turbidity_features['turbidity_data_sources'] = pd.Series(
                len(available_turbidity), index=available_turbidity[0].index
            )
            
        return turbidity_features
    
    def classify_water_quality(self,
                             turbidity_data: pd.Series,
                             turbidity_type: str = 'kd490') -> Dict[str, pd.Series]:
        """
        Classify water quality based on turbidity thresholds
        
        Args:
            turbidity_data: Turbidity measurement series
            turbidity_type: Type of turbidity data ('kd490' or 'turbidity_ntu')
            
        Returns:
            Dictionary of water quality classification features
        """
        if turbidity_type not in self.water_quality_thresholds:
            logger.warning(f"Unknown turbidity type: {turbidity_type}")
            return {}
            
        thresholds = self.water_quality_thresholds[turbidity_type]
        quality_features = {}
        
        # Create classification features
        for quality_class, (min_val, max_val) in thresholds.items():
            if max_val == np.inf:
                is_class = turbidity_data >= min_val
            else:
                is_class = (turbidity_data >= min_val) & (turbidity_data < max_val)
                
            quality_features[f'is_{quality_class}_water'] = is_class.astype(int)
            
        # Numeric quality score (0=best, higher=worse)
        quality_scores = []
        for value in turbidity_data:
            score = 0
            for i, (quality_class, (min_val, max_val)) in enumerate(thresholds.items()):
                if max_val == np.inf:
                    if value >= min_val:
                        score = i
                        break
                else:
                    if min_val <= value < max_val:
                        score = i
                        break
            quality_scores.append(score)
            
        quality_features['water_quality_score'] = pd.Series(quality_scores, index=turbidity_data.index)
        
        # Water clarity percentage (0-100%)
        max_good_turbidity = thresholds['good'][1] if 'good' in thresholds else 0.3
        clarity_percentage = np.maximum(0, 100 * (1 - turbidity_data / max_good_turbidity))
        quality_features['water_clarity_percentage'] = clarity_percentage
        
        return quality_features
    
    def calculate_optical_water_types(self,
                                    turbidity_data: pd.Series,
                                    chlorophyll_data: Optional[pd.Series] = None,
                                    rrs_data: Optional[Dict[str, pd.Series]] = None) -> Dict[str, pd.Series]:
        """
        Classify optical water types using turbidity and other optical data
        
        Args:
            turbidity_data: Turbidity measurement series
            chlorophyll_data: Optional chlorophyll data
            rrs_data: Optional remote sensing reflectance data
            
        Returns:
            Dictionary of optical water type features
        """
        optical_features = {}
        
        # Basic optical water type classification
        # Based on turbidity and chlorophyll (if available)
        
        if chlorophyll_data is not None:
            # Case 1 vs Case 2 water classification
            # Case 1: Dominated by phytoplankton
            # Case 2: Dominated by non-algal particles and CDOM
            
            case_1_indicator = []
            case_2_indicator = []
            
            for i in turbidity_data.index:
                turb = turbidity_data.loc[i] if pd.notna(turbidity_data.loc[i]) else 0
                chl = chlorophyll_data.loc[i] if i in chlorophyll_data.index and pd.notna(chlorophyll_data.loc[i]) else 0
                
                # Simple heuristic for water type classification
                if chl > 1.0 and turb < 0.5:
                    case_1_indicator.append(1)  # Phytoplankton-dominated
                    case_2_indicator.append(0)
                elif turb > 0.5 and chl < 2.0:
                    case_1_indicator.append(0)
                    case_2_indicator.append(1)  # Sediment/CDOM-dominated
                else:
                    # Mixed case
                    case_1_indicator.append(0.5)
                    case_2_indicator.append(0.5)
                    
            optical_features['case1_water_probability'] = pd.Series(case_1_indicator, index=turbidity_data.index)
            optical_features['case2_water_probability'] = pd.Series(case_2_indicator, index=turbidity_data.index)
            
            # Trophic state based on chlorophyll and turbidity
            trophic_scores = []
            for i in turbidity_data.index:
                turb = turbidity_data.loc[i] if pd.notna(turbidity_data.loc[i]) else 0
                chl = chlorophyll_data.loc[i] if i in chlorophyll_data.index and pd.notna(chlorophyll_data.loc[i]) else 0
                
                # Combined trophic state index
                trophic = np.log10(chl + 0.1) + np.log10(turb + 0.1)  # Log scale combination
                trophic_scores.append(trophic)
                
            optical_features['combined_trophic_index'] = pd.Series(trophic_scores, index=turbidity_data.index)
            
        # Turbidity-based optical characteristics
        optical_features['light_penetration_depth'] = 1.7 / (turbidity_data + 0.1)  # Approximate euphotic depth
        optical_features['underwater_visibility'] = np.exp(-turbidity_data * 2)  # Visibility factor
        
        # Optical complexity index (higher values indicate more complex optical properties)
        if rrs_data is not None and len(rrs_data) >= 2:
            # Calculate spectral slope and curvature as complexity measures
            wavelengths = []
            reflectances = []
            
            # Map common RRS bands to wavelengths
            wavelength_map = {
                'RRS443': 443, 'RRS490': 490, 'RRS510': 510, 
                'RRS555': 555, 'RRS670': 670
            }
            
            for band, rrs_series in rrs_data.items():
                if band in wavelength_map:
                    wavelengths.append(wavelength_map[band])
                    reflectances.append(rrs_series)
                    
            if len(wavelengths) >= 2:
                # Calculate spectral complexity for each time point
                complexity_scores = []
                
                for i in turbidity_data.index:
                    spec_values = []
                    for rrs_series in reflectances:
                        if i in rrs_series.index and pd.notna(rrs_series.loc[i]):
                            spec_values.append(rrs_series.loc[i])
                        else:
                            spec_values.append(0)
                            
                    if len(spec_values) >= 2:
                        # Spectral variability as complexity measure
                        complexity = np.std(spec_values) / (np.mean(spec_values) + 1e-6)
                    else:
                        complexity = 0
                        
                    complexity_scores.append(complexity)
                    
                optical_features['spectral_complexity'] = pd.Series(complexity_scores, index=turbidity_data.index)
                
        return optical_features
    
    def calculate_turbidity_based_hab_features(self,
                                             turbidity_data: pd.Series,
                                             species: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Calculate HAB-relevant features based on turbidity conditions
        
        Args:
            turbidity_data: Turbidity measurement series
            species: Optional HAB species name for species-specific features
            
        Returns:
            Dictionary of HAB-relevant turbidity features
        """
        hab_features = {}
        
        # Light availability for photosynthesis
        # Assume surface irradiance of 2000 umol/m2/s (typical sunny day)
        surface_light = 2000  
        
        # Light availability at different depths
        for depth in [0.5, 1, 2, 5]:  # meters
            light_fraction = np.exp(-turbidity_data * depth)
            available_light = surface_light * light_fraction
            
            # Photosynthetically available radiation (PAR) threshold
            par_threshold = 50  # umol/m2/s minimum for photosynthesis
            is_sufficient_light = (available_light > par_threshold).astype(int)
            
            hab_features[f'sufficient_light_{depth}m'] = is_sufficient_light
            hab_features[f'available_light_{depth}m'] = available_light
            
        # Euphotic zone depth (1% surface light level)
        euphotic_depth = -np.log(0.01) / (turbidity_data + 0.001)
        hab_features['euphotic_zone_depth'] = euphotic_depth
        
        # Light limitation index
        critical_depth = 3  # meters (typical mixed layer depth)
        light_at_critical = np.exp(-turbidity_data * critical_depth)
        hab_features['light_limitation_index'] = 1 - light_at_critical  # Higher = more limited
        
        # Species-specific turbidity suitability
        if species and species in self.species_turbidity_preferences:
            prefs = self.species_turbidity_preferences[species]
            optimal_range = prefs['optimal_kd490']
            
            # Suitability score based on distance from optimal range
            suitability_scores = []
            for turb_val in turbidity_data:
                if pd.isna(turb_val):
                    suitability_scores.append(0.5)
                    continue
                    
                if optimal_range[0] <= turb_val <= optimal_range[1]:
                    score = 1.0  # Optimal
                else:
                    # Linear decrease outside optimal range
                    if turb_val < optimal_range[0]:
                        # Too clear
                        distance = optimal_range[0] - turb_val
                        score = max(0, 1 - distance / optimal_range[0])
                    else:
                        # Too turbid
                        distance = turb_val - optimal_range[1]
                        score = max(0, 1 - distance / optimal_range[1])
                        
                suitability_scores.append(score)
                
            hab_features[f'{species}_turbidity_suitability'] = pd.Series(
                suitability_scores, index=turbidity_data.index
            )
            
            # Binary optimal condition indicator
            is_optimal = (turbidity_data >= optimal_range[0]) & (turbidity_data <= optimal_range[1])
            hab_features[f'{species}_optimal_turbidity'] = is_optimal.astype(int)
            
        # General HAB risk indicators
        # Very clear water (may favor some toxic species)
        hab_features['very_clear_water_risk'] = (turbidity_data < 0.05).astype(int)
        
        # Moderate turbidity (optimal for many species)
        hab_features['moderate_turbidity_optimal'] = ((turbidity_data >= 0.1) & (turbidity_data <= 0.4)).astype(int)
        
        # High turbidity (may stress some species but favor others)
        hab_features['high_turbidity_stress'] = (turbidity_data > 0.8).astype(int)
        
        return hab_features
    
    def calculate_sediment_and_runoff_indicators(self,
                                               turbidity_data: pd.Series,
                                               precipitation_data: Optional[pd.Series] = None,
                                               wind_data: Optional[pd.Series] = None) -> Dict[str, pd.Series]:
        """
        Calculate sediment resuspension and runoff indicators
        
        Args:
            turbidity_data: Turbidity measurement series
            precipitation_data: Optional precipitation data
            wind_data: Optional wind speed data
            
        Returns:
            Dictionary of sediment and runoff features
        """
        sediment_features = {}
        
        # Sediment load classification
        sediment_features['low_sediment'] = (turbidity_data < 0.2).astype(int)
        sediment_features['moderate_sediment'] = ((turbidity_data >= 0.2) & (turbidity_data < 0.8)).astype(int)
        sediment_features['high_sediment'] = (turbidity_data >= 0.8).astype(int)
        
        # Sediment resuspension potential (based on turbidity changes)
        turbidity_change = turbidity_data.diff().abs()
        sediment_features['turbidity_variability'] = turbidity_change
        
        # Rolling statistics for sediment dynamics
        for window in [3, 7, 14]:  # days
            rolling_mean = turbidity_data.rolling(window=window).mean()
            rolling_std = turbidity_data.rolling(window=window).std()
            
            sediment_features[f'turbidity_stability_{window}d'] = rolling_std / (rolling_mean + 0.01)
            
            # Recent increase in turbidity (potential runoff event)
            recent_increase = turbidity_data > (rolling_mean + rolling_std)
            sediment_features[f'elevated_turbidity_{window}d'] = recent_increase.astype(int)
            
        # Precipitation-driven runoff (if precipitation data available)
        if precipitation_data is not None:
            # Align indices
            common_index = turbidity_data.index.intersection(precipitation_data.index)
            turb_aligned = turbidity_data.loc[common_index]
            precip_aligned = precipitation_data.loc[common_index]
            
            # Lag correlation between precipitation and turbidity
            # Check different lag periods
            for lag_days in [1, 2, 3, 5]:
                precip_lagged = precip_aligned.shift(lag_days)
                
                # Create windows for correlation calculation
                correlation_values = []
                window_size = 30  # days
                
                for i in range(window_size, len(turb_aligned)):
                    turb_window = turb_aligned.iloc[i-window_size:i]
                    precip_window = precip_lagged.iloc[i-window_size:i]
                    
                    # Remove NaN values
                    valid_indices = ~(pd.isna(turb_window) | pd.isna(precip_window))
                    
                    if valid_indices.sum() > 10:  # Need at least 10 valid points
                        corr = np.corrcoef(turb_window[valid_indices], 
                                         precip_window[valid_indices])[0, 1]
                        if not np.isnan(corr):
                            correlation_values.append(corr)
                        else:
                            correlation_values.append(0)
                    else:
                        correlation_values.append(0)
                        
                # Pad with zeros for initial period
                full_correlation = [0] * window_size + correlation_values
                
                sediment_features[f'precip_turbidity_correlation_{lag_days}d'] = pd.Series(
                    full_correlation, index=common_index
                )
                
        # Wind-driven resuspension (if wind data available)
        if wind_data is not None:
            common_index = turbidity_data.index.intersection(wind_data.index)
            turb_aligned = turbidity_data.loc[common_index]
            wind_aligned = wind_data.loc[common_index]
            
            # Wind stress proxy (wind speed squared)
            wind_stress = wind_aligned ** 2
            
            # Immediate wind-turbidity relationship
            sediment_features['wind_turbidity_product'] = turb_aligned * wind_stress
            
            # High wind conditions that could cause resuspension
            high_wind_threshold = wind_aligned.quantile(0.8)  # 80th percentile
            high_wind_events = (wind_aligned > high_wind_threshold).astype(int)
            
            # Turbidity response to high wind (next day)
            turb_response = turb_aligned.shift(-1) > turb_aligned  # Increase next day
            wind_resuspension_indicator = high_wind_events & turb_response.shift(1).fillna(False)
            
            sediment_features['wind_resuspension_events'] = wind_resuspension_indicator.astype(int)
            
        return sediment_features
    
    def create_comprehensive_turbidity_features(self,
                                              turbidity_data_dict: Dict[str, Any],
                                              environmental_data: Optional[pd.DataFrame] = None,
                                              species: Optional[str] = None) -> pd.DataFrame:
        """
        Create comprehensive turbidity-based features using all breakthrough data sources
        
        Args:
            turbidity_data_dict: Dictionary of turbidity data from different sources
            environmental_data: Optional DataFrame with other environmental variables
            species: Optional species name for species-specific features
            
        Returns:
            DataFrame with all turbidity-based features
        """
        logger.info("🌊 Creating comprehensive turbidity features using BREAKTHROUGH data...")
        
        # Extract turbidity data from different sources
        kd490_data = turbidity_data_dict.get('kd490')
        rrs_data = turbidity_data_dict.get('rrs', {})
        sentinel2_data = turbidity_data_dict.get('sentinel2', {})
        
        # Determine common index
        available_series = []
        if kd490_data is not None:
            available_series.append(kd490_data)
        if rrs_data:
            available_series.extend(rrs_data.values())
        if sentinel2_data:
            available_series.extend(sentinel2_data.values())
            
        if not available_series:
            logger.warning("No turbidity data provided")
            return pd.DataFrame()
            
        common_index = available_series[0].index
        feature_df = pd.DataFrame(index=common_index)
        
        # Multi-source integration
        integration_features = self.integrate_multi_source_turbidity(kd490_data, rrs_data, sentinel2_data)
        for name, series in integration_features.items():
            feature_df[name] = series
            
        # Get primary turbidity series for further analysis
        if 'unified_turbidity' in feature_df.columns:
            primary_turbidity = feature_df['unified_turbidity']
        elif kd490_data is not None:
            primary_turbidity = kd490_data
        elif 'rrs_turbidity_proxy' in integration_features:
            primary_turbidity = integration_features['rrs_turbidity_proxy']
        else:
            logger.warning("No primary turbidity series available")
            return feature_df
            
        # Water quality classification
        quality_features = self.classify_water_quality(primary_turbidity, 'kd490')
        for name, series in quality_features.items():
            feature_df[name] = series
            
        # Optical water type analysis
        chlorophyll_data = None
        if environmental_data is not None and 'chlorophyll' in environmental_data.columns:
            chlorophyll_data = environmental_data['chlorophyll']
            
        optical_features = self.calculate_optical_water_types(primary_turbidity, chlorophyll_data, rrs_data)
        for name, series in optical_features.items():
            feature_df[name] = series
            
        # HAB-specific turbidity features
        hab_features = self.calculate_turbidity_based_hab_features(primary_turbidity, species)
        for name, series in hab_features.items():
            feature_df[name] = series
            
        # Sediment and runoff indicators
        precipitation_data = None
        wind_data = None
        if environmental_data is not None:
            if 'precipitation' in environmental_data.columns:
                precipitation_data = environmental_data['precipitation']
            if 'wind_speed' in environmental_data.columns:
                wind_data = environmental_data['wind_speed']
                
        sediment_features = self.calculate_sediment_and_runoff_indicators(
            primary_turbidity, precipitation_data, wind_data
        )
        for name, series in sediment_features.items():
            feature_df[name] = series
            
        logger.info(f"✅ Created {len(feature_df.columns)} turbidity-based features")
        logger.info("🥇 Leveraged all 3 BREAKTHROUGH turbidity sources")
        
        return feature_df