"""
Smart Dataset Selection System for HAB Forecasting

This module provides deterministic dataset selection based on temporal coverage,
data quality, and availability for optimal HAB variable collection.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class DatasetSelector:
    """
    Intelligent dataset selector for HAB forecasting variables.
    
    Selects optimal datasets based on:
    1. Temporal coverage
    2. Data quality (NRT vs MY vs Analysis/Forecast)
    3. Spatial resolution
    4. Temporal resolution
    5. Data latency
    """
    
    def __init__(self, config_path: str = "config/data_sources.yaml"):
        """Initialize the dataset selector."""
        self.config_path = config_path
        self.config = self._load_config()
        self.dataset_catalog = self._build_dataset_catalog()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError as e:
            # Try with different encodings if UTF-8 fails
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(self.config_path, 'r', encoding=encoding) as f:
                        return yaml.safe_load(f)
                except (UnicodeDecodeError, yaml.YAMLError):
                    continue
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return {}
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return {}
    
    def _build_dataset_catalog(self) -> Dict:
        """Build comprehensive dataset catalog with temporal coverage for all 25 HAB variables."""
        return {
            # ============ CORE HAB INDICATORS (3/3) ============
            'sea_surface_temperature': {
                'datasets': [
                    {
                        'id': 'METOFFICE-GLO-SST-L4-REP-OBS-SST',
                        'product': 'SST_GLO_SST_L4_REP_OBSERVATIONS_010_011',
                        'source': 'copernicus',
                        'type': 'reprocessed',
                        'variable': 'analysed_sst',
                        'temporal_range': ['1981-09-01', '2021-12-31'],
                        'resolution': '0.05°',
                        'quality_score': 10.0,
                        'latency_days': 365,
                        'selection_criteria': {
                            'historical': True,
                            'recent': False,
                            'real_time': False
                        }
                    },
                    {
                        'id': 'METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2',
                        'product': 'SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001',
                        'source': 'copernicus',
                        'type': 'near_real_time',
                        'variable': 'analysed_sst',
                        'temporal_range': ['2007-01-01', 'present'],
                        'resolution': '0.05°',
                        'quality_score': 9.5,
                        'latency_days': 1,
                        'selection_criteria': {
                            'historical': True,
                            'recent': True,
                            'real_time': True
                        }
                    },
                    {
                        'id': 'HYCOM/sea_temp_salinity',
                        'product': 'GEE',
                        'source': 'gee',
                        'type': 'reanalysis',
                        'variable': 'water_temp_0',
                        'temporal_range': ['1992-10-02', 'present'],
                        'resolution': 'variable',
                        'quality_score': 8.0,
                        'latency_days': 7,
                        'selection_criteria': {
                            'historical': True,
                            'recent': True,
                            'real_time': True
                        }
                    }
                ]
            },
            'chlorophyll_a': {
                'datasets': [
                    {
                        'id': 'cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D',
                        'product': 'OCEANCOLOUR_GLO_BGC_L3_MY_009_103',
                        'source': 'copernicus',
                        'type': 'multi_year',
                        'variable': 'CHL',
                        'temporal_range': ['1997-09-04', '2023-12-31'],
                        'resolution': '4km',
                        'quality_score': 9.8,
                        'latency_days': 180,
                        'selection_criteria': {
                            'historical': True,
                            'recent': False,
                            'real_time': False
                        }
                    },
                    {
                        'id': 'cmems_obs-oc_glo_bgc-plankton_nrt_l3-multi-4km_P1D',
                        'product': 'OCEANCOLOUR_GLO_BGC_L3_NRT_009_101',
                        'source': 'copernicus',
                        'type': 'near_real_time',
                        'variable': 'CHL',
                        'temporal_range': ['2025-07-17', 'present'],
                        'resolution': '4km',
                        'quality_score': 9.2,
                        'latency_days': 3,
                        'selection_criteria': {
                            'historical': False,
                            'recent': True,
                            'real_time': True
                        }
                    },
                    {
                        'id': 'JAXA/GCOM-C/L3/OCEAN/CHLA/V3',
                        'product': 'GEE',
                        'source': 'gee',
                        'type': 'near_real_time',
                        'variable': 'CHLA_AVE',
                        'temporal_range': ['2017-12-23', 'present'],
                        'resolution': '1km',
                        'quality_score': 9.0,
                        'latency_days': 7,
                        'selection_criteria': {
                            'historical': False,
                            'recent': True,
                            'real_time': True,
                            'high_resolution': True,
                            'coastal': True
                        }
                    }
                ]
            },
            'turbidity': {
                'datasets': [
                    {
                        'id': 'cmems_obs-oc_glo_bgc-transp_nrt_l4-gapfree-multi-4km_P1D',
                        'product': 'OCEANCOLOUR_GLO_BGC_L4_NRT_009_102',
                        'source': 'copernicus',
                        'type': 'near_real_time',
                        'variable': 'KD490',
                        'temporal_range': ['2025-07-17', 'present'],
                        'resolution': '4km',
                        'quality_score': 9.5,
                        'latency_days': 1,
                        'selection_criteria': {
                            'historical': False,
                            'recent': True,
                            'real_time': True
                        }
                    },
                    {
                        'id': 'cmems_obs-oc_glo_bgc-reflectance_nrt_l4-multi-4km_P1M',
                        'product': 'OCEANCOLOUR_GLO_BGC_L4_NRT_009_102',
                        'source': 'copernicus',
                        'type': 'near_real_time',
                        'variable': ['RRS443', 'RRS490', 'RRS555'],
                        'temporal_range': ['2025-05-01', '2025-07-01'],
                        'resolution': '4km',
                        'quality_score': 8.5,
                        'latency_days': 30,
                        'selection_criteria': {
                            'historical': False,
                            'recent': True,
                            'real_time': True
                        }
                    },
                    {
                        'id': 'COPERNICUS/S2_SR_HARMONIZED',
                        'product': 'GEE',
                        'source': 'gee',
                        'type': 'near_real_time',
                        'variable': ['B2', 'B4'],
                        'temporal_range': ['2017-03-28', 'present'],
                        'resolution': '10-20m',
                        'quality_score': 9.0,
                        'latency_days': 5,
                        'selection_criteria': {
                            'historical': False,
                            'recent': True,
                            'real_time': True,
                            'high_resolution': True,
                            'coastal': True
                        }
                    }
                ]
            },
            
            # ============ PHYSICAL OCEANOGRAPHY (8/8) ============
            'mixed_layer_depth': {
                'datasets': [
                    {
                        'id': 'cmems_mod_glo_phy_anfc_0.083deg_P1D-m',
                        'product': 'GLOBAL_ANALYSISFORECAST_PHY_001_024',
                        'source': 'copernicus',
                        'type': 'analysis_forecast',
                        'variable': 'mlotst',
                        'temporal_range': ['2019-01-01', 'present'],
                        'resolution': '1/12°',
                        'quality_score': 8.5,
                        'latency_days': 1,
                        'selection_criteria': {
                            'historical': False,
                            'recent': True,
                            'real_time': True
                        }
                    }
                ]
            },
            'ocean_currents': {
                'datasets': [
                    {
                        'id': 'HYCOM/sea_water_velocity',
                        'product': 'GEE',
                        'source': 'gee',
                        'type': 'reanalysis',
                        'variable': ['velocity_u_0', 'velocity_v_0'],
                        'temporal_range': ['1992-10-02', 'present'],
                        'resolution': 'variable',
                        'quality_score': 8.0,
                        'latency_days': 7,
                        'selection_criteria': {
                            'historical': True,
                            'recent': True,
                            'real_time': True
                        }
                    }
                ]
            },
            'sea_level_anomaly': {
                'datasets': [
                    {
                        'id': 'cmems_mod_glo_phy_anfc_0.083deg_P1D-m',
                        'product': 'GLOBAL_ANALYSISFORECAST_PHY_001_024',
                        'source': 'copernicus',
                        'type': 'analysis_forecast',
                        'variable': 'zos',
                        'temporal_range': ['2019-01-01', 'present'],
                        'resolution': '1/12°',
                        'quality_score': 8.5,
                        'latency_days': 1,
                        'selection_criteria': {
                            'historical': False,
                            'recent': True,
                            'real_time': True
                        }
                    }
                ]
            },
            'ocean_temperature': {
                'datasets': [
                    {
                        'id': 'HYCOM/sea_temp_salinity',
                        'product': 'GEE',
                        'source': 'gee',
                        'type': 'reanalysis',
                        'variable': 'water_temp_0',
                        'temporal_range': ['1992-10-02', 'present'],
                        'resolution': 'variable',
                        'quality_score': 8.0,
                        'latency_days': 7,
                        'selection_criteria': {
                            'historical': True,
                            'recent': True,
                            'real_time': True
                        }
                    }
                ]
            },
            'ocean_salinity': {
                'datasets': [
                    {
                        'id': 'HYCOM/sea_temp_salinity',
                        'product': 'GEE',
                        'source': 'gee',
                        'type': 'reanalysis',
                        'variable': 'salinity_0',
                        'temporal_range': ['1992-10-02', 'present'],
                        'resolution': 'variable',
                        'quality_score': 8.0,
                        'latency_days': 7,
                        'selection_criteria': {
                            'historical': True,
                            'recent': True,
                            'real_time': True
                        }
                    }
                ]
            },
            'sea_surface_elevation': {
                'datasets': [
                    {
                        'id': 'HYCOM/sea_surface_elevation',
                        'product': 'GEE',
                        'source': 'gee',
                        'type': 'reanalysis',
                        'variable': 'surface_elevation',
                        'temporal_range': ['1992-10-02', 'present'],
                        'resolution': 'variable',
                        'quality_score': 8.0,
                        'latency_days': 7,
                        'selection_criteria': {
                            'historical': True,
                            'recent': True,
                            'real_time': True
                        }
                    }
                ]
            },
            
            # ============ ATMOSPHERIC VARIABLES (3/3) ============
            'wind_components': {
                'datasets': [
                    {
                        'id': 'ECMWF/ERA5/DAILY',
                        'product': 'GEE',
                        'source': 'gee',
                        'type': 'reanalysis',
                        'variable': ['u_component_of_wind_10m', 'v_component_of_wind_10m'],
                        'temporal_range': ['1940-01-01', 'present'],
                        'resolution': '0.25°',
                        'quality_score': 9.5,
                        'latency_days': 30,
                        'selection_criteria': {
                            'historical': True,
                            'recent': True,
                            'real_time': True
                        }
                    }
                ]
            },
            'precipitation': {
                'datasets': [
                    {
                        'id': 'ECMWF/ERA5/DAILY',
                        'product': 'GEE',
                        'source': 'gee',
                        'type': 'reanalysis',
                        'variable': 'total_precipitation',
                        'temporal_range': ['1940-01-01', 'present'],
                        'resolution': '0.25°',
                        'quality_score': 9.5,
                        'latency_days': 30,
                        'selection_criteria': {
                            'historical': True,
                            'recent': True,
                            'real_time': True
                        }
                    }
                ]
            },
            
            # ============ BIOGEOCHEMISTRY & NUTRIENTS (8/8) ============
            'nitrate': {
                'datasets': [
                    {
                        'id': 'cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m',
                        'product': 'GLOBAL_ANALYSISFORECAST_BGC_001_028',
                        'source': 'copernicus',
                        'type': 'analysis_forecast',
                        'variable': 'no3',
                        'temporal_range': ['2019-01-01', 'present'],
                        'resolution': '0.25°',
                        'quality_score': 7.5,
                        'latency_days': 1,
                        'selection_criteria': {
                            'historical': False,
                            'recent': True,
                            'real_time': True
                        }
                    }
                ]
            },
            'phosphate': {
                'datasets': [
                    {
                        'id': 'cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m',
                        'product': 'GLOBAL_ANALYSISFORECAST_BGC_001_028',
                        'source': 'copernicus',
                        'type': 'analysis_forecast',
                        'variable': 'po4',
                        'temporal_range': ['2019-01-01', 'present'],
                        'resolution': '0.25°',
                        'quality_score': 7.5,
                        'latency_days': 1,
                        'selection_criteria': {
                            'historical': False,
                            'recent': True,
                            'real_time': True
                        }
                    }
                ]
            },
            'silicate': {
                'datasets': [
                    {
                        'id': 'cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m',
                        'product': 'GLOBAL_ANALYSISFORECAST_BGC_001_028',
                        'source': 'copernicus',
                        'type': 'analysis_forecast',
                        'variable': 'si',
                        'temporal_range': ['2019-01-01', 'present'],
                        'resolution': '0.25°',
                        'quality_score': 7.5,
                        'latency_days': 1,
                        'selection_criteria': {
                            'historical': False,
                            'recent': True,
                            'real_time': True
                        }
                    }
                ]
            },
            'dissolved_iron': {
                'datasets': [
                    {
                        'id': 'cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m',
                        'product': 'GLOBAL_ANALYSISFORECAST_BGC_001_028',
                        'source': 'copernicus',
                        'type': 'analysis_forecast',
                        'variable': 'fe',
                        'temporal_range': ['2019-01-01', 'present'],
                        'resolution': '0.25°',
                        'quality_score': 7.5,
                        'latency_days': 1,
                        'selection_criteria': {
                            'historical': False,
                            'recent': True,
                            'real_time': True
                        }
                    }
                ]
            },
            'dissolved_oxygen': {
                'datasets': [
                    {
                        'id': 'cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m',
                        'product': 'GLOBAL_ANALYSISFORECAST_BGC_001_028',
                        'source': 'copernicus',
                        'type': 'analysis_forecast',
                        'variable': 'o2',
                        'temporal_range': ['2019-01-01', 'present'],
                        'resolution': '0.25°',
                        'quality_score': 7.5,
                        'latency_days': 1,
                        'selection_criteria': {
                            'historical': False,
                            'recent': True,
                            'real_time': True
                        }
                    }
                ]
            },
            'primary_production': {
                'datasets': [
                    {
                        'id': 'cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m',
                        'product': 'GLOBAL_ANALYSISFORECAST_BGC_001_028',
                        'source': 'copernicus',
                        'type': 'analysis_forecast',
                        'variable': 'nppv',
                        'temporal_range': ['2019-01-01', 'present'],
                        'resolution': '0.25°',
                        'quality_score': 7.5,
                        'latency_days': 1,
                        'selection_criteria': {
                            'historical': False,
                            'recent': True,
                            'real_time': True
                        }
                    }
                ]
            },
            'total_chlorophyll': {
                'datasets': [
                    {
                        'id': 'cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m',
                        'product': 'GLOBAL_ANALYSISFORECAST_BGC_001_028',
                        'source': 'copernicus',
                        'type': 'analysis_forecast',
                        'variable': 'chl',
                        'temporal_range': ['2019-01-01', 'present'],
                        'resolution': '0.25°',
                        'quality_score': 7.5,
                        'latency_days': 1,
                        'selection_criteria': {
                            'historical': False,
                            'recent': True,
                            'real_time': True
                        }
                    }
                ]
            },
            'phytoplankton_carbon': {
                'datasets': [
                    {
                        'id': 'cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m',
                        'product': 'GLOBAL_ANALYSISFORECAST_BGC_001_028',
                        'source': 'copernicus',
                        'type': 'analysis_forecast',
                        'variable': 'phyc',
                        'temporal_range': ['2019-01-01', 'present'],
                        'resolution': '0.25°',
                        'quality_score': 7.5,
                        'latency_days': 1,
                        'selection_criteria': {
                            'historical': False,
                            'recent': True,
                            'real_time': True
                        }
                    }
                ]
            },
            
            # ============ WATER QUALITY & OPTICS (3/3) ============
            'diffuse_attenuation': {
                'datasets': [
                    {
                        'id': 'cmems_mod_glo_bgc-optics_anfc_0.25deg_P1D-m',
                        'product': 'GLOBAL_ANALYSISFORECAST_BGC_001_028',
                        'source': 'copernicus',
                        'type': 'analysis_forecast',
                        'variable': 'kd',
                        'temporal_range': ['2019-01-01', 'present'],
                        'resolution': '0.25°',
                        'quality_score': 7.5,
                        'latency_days': 1,
                        'selection_criteria': {
                            'historical': False,
                            'recent': True,
                            'real_time': True
                        }
                    }
                ]
            }
        }
    
    def select_optimal_dataset(self, 
                             variable: str, 
                             start_date: str, 
                             end_date: str,
                             preferences: Optional[Dict] = None) -> Dict:
        """
        Select optimal dataset for a variable and time period.
        
        Args:
            variable: Variable name (e.g., 'sea_surface_temperature')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            preferences: Optional preferences dict with keys:
                - prefer_high_resolution: bool
                - prefer_low_latency: bool
                - prefer_historical_quality: bool
                - coastal_focus: bool
        
        Returns:
            Dict with selected dataset information
        """
        if variable not in self.dataset_catalog:
            raise ValueError(f"Variable '{variable}' not found in dataset catalog")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Set default preferences
        prefs = preferences or {}
        prefer_high_res = prefs.get('prefer_high_resolution', False)
        prefer_low_latency = prefs.get('prefer_low_latency', True)
        prefer_historical_quality = prefs.get('prefer_historical_quality', True)
        coastal_focus = prefs.get('coastal_focus', False)
        
        datasets = self.dataset_catalog[variable]['datasets']
        valid_datasets = []
        
        # Filter datasets by temporal coverage
        for dataset in datasets:
            if self._dataset_covers_period(dataset, start_date, end_date):
                valid_datasets.append(dataset)
        
        if not valid_datasets:
            raise ValueError(f"No datasets found covering period {start_date} to {end_date}")
        
        # Score and rank datasets
        scored_datasets = []
        for dataset in valid_datasets:
            score = self._calculate_dataset_score(
                dataset, start_dt, end_dt, 
                prefer_high_res, prefer_low_latency, 
                prefer_historical_quality, coastal_focus
            )
            scored_datasets.append((score, dataset))
        
        # Sort by score (highest first)
        scored_datasets.sort(key=lambda x: x[0], reverse=True)
        
        best_dataset = scored_datasets[0][1]
        
        logger.info(f"Selected dataset for {variable} ({start_date} to {end_date}): "
                   f"{best_dataset['id']} (score: {scored_datasets[0][0]:.2f})")
        
        return {
            'variable': variable,
            'dataset_id': best_dataset['id'],
            'product_id': best_dataset['product'],
            'source': best_dataset['source'],
            'variable_name': best_dataset['variable'],
            'resolution': best_dataset['resolution'],
            'type': best_dataset['type'],
            'quality_score': best_dataset['quality_score'],
            'selection_score': scored_datasets[0][0],
            'temporal_range': best_dataset['temporal_range'],
            'reason': self._get_selection_reason(best_dataset, start_dt, end_dt)
        }
    
    def _dataset_covers_period(self, dataset: Dict, start_date: str, end_date: str) -> bool:
        """Check if dataset covers the requested time period."""
        try:
            dataset_start = dataset['temporal_range'][0]
            dataset_end = dataset['temporal_range'][1]
            
            if dataset_start == 'present' or dataset_end == 'present':
                dataset_end_dt = datetime.now()
            else:
                dataset_end_dt = datetime.strptime(dataset_end, '%Y-%m-%d')
            
            dataset_start_dt = datetime.strptime(dataset_start, '%Y-%m-%d')
            request_start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            request_end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            return (dataset_start_dt <= request_start_dt and 
                    dataset_end_dt >= request_end_dt)
        except:
            return False
    
    def _calculate_dataset_score(self, 
                               dataset: Dict, 
                               start_dt: datetime, 
                               end_dt: datetime,
                               prefer_high_res: bool = False,
                               prefer_low_latency: bool = True,
                               prefer_historical_quality: bool = True,
                               coastal_focus: bool = False) -> float:
        """Calculate a score for dataset selection."""
        score = dataset['quality_score']  # Base score (0-10)
        
        # Time period preferences
        time_span = (end_dt - start_dt).days
        current_date = datetime.now()
        days_from_present = (current_date - end_dt).days
        
        # Historical vs recent preference
        if days_from_present > 365 and prefer_historical_quality:
            if dataset['type'] in ['reprocessed', 'multi_year']:
                score += 2.0  # Prefer reprocessed for historical
        elif days_from_present < 30 and prefer_low_latency:
            if dataset['type'] in ['near_real_time', 'analysis_forecast']:
                score += 2.0  # Prefer NRT for recent data
        
        # Resolution preference
        if prefer_high_res:
            if '1km' in dataset['resolution'] or 'high' in dataset.get('selection_criteria', {}):
                score += 1.5
        
        # Coastal focus
        if coastal_focus and dataset.get('selection_criteria', {}).get('coastal', False):
            score += 1.0
        
        # Latency penalty for near real-time needs
        if prefer_low_latency:
            latency_penalty = dataset.get('latency_days', 1) / 10.0
            score -= min(latency_penalty, 2.0)
        
        return score
    
    def _get_selection_reason(self, dataset: Dict, start_dt: datetime, end_dt: datetime) -> str:
        """Generate human-readable reason for dataset selection."""
        reasons = []
        
        dataset_type = dataset['type']
        if dataset_type == 'reprocessed':
            reasons.append("Highest quality reprocessed data")
        elif dataset_type == 'near_real_time':
            reasons.append("Low latency near real-time data")
        elif dataset_type == 'analysis_forecast':
            reasons.append("Model-based analysis/forecast")
        elif dataset_type == 'multi_year':
            reasons.append("Long time series with consistent processing")
        
        if '1km' in dataset['resolution']:
            reasons.append("High spatial resolution")
        
        days_from_present = (datetime.now() - end_dt).days
        if days_from_present > 365:
            reasons.append("Historical period - prioritizing quality")
        elif days_from_present < 30:
            reasons.append("Recent period - prioritizing timeliness")
        
        return "; ".join(reasons)
    
    def get_temporal_coverage_map(self) -> Dict:
        """Return comprehensive temporal coverage for all variables."""
        coverage_map = {}
        
        for variable, info in self.dataset_catalog.items():
            coverage_map[variable] = {
                'datasets': [],
                'earliest_start': None,
                'latest_end': None
            }
            
            earliest_start = None
            latest_end = None
            
            for dataset in info['datasets']:
                start_str = dataset['temporal_range'][0]
                end_str = dataset['temporal_range'][1]
                
                try:
                    start_dt = datetime.strptime(start_str, '%Y-%m-%d')
                    end_dt = datetime.now() if end_str == 'present' else datetime.strptime(end_str, '%Y-%m-%d')
                    
                    if earliest_start is None or start_dt < earliest_start:
                        earliest_start = start_dt
                    if latest_end is None or end_dt > latest_end:
                        latest_end = end_dt
                    
                    coverage_map[variable]['datasets'].append({
                        'id': dataset['id'],
                        'start': start_str,
                        'end': end_str, 
                        'type': dataset['type'],
                        'quality': dataset['quality_score']
                    })
                except:
                    continue
            
            coverage_map[variable]['earliest_start'] = earliest_start.strftime('%Y-%m-%d') if earliest_start else None
            coverage_map[variable]['latest_end'] = latest_end.strftime('%Y-%m-%d') if latest_end else None
        
        return coverage_map
    
    def get_available_variables(self) -> List[str]:
        """Get list of all available variables in the dataset catalog."""
        return list(self.dataset_catalog.keys())
    
    def validate_dataset_availability(self, dataset_id: str, date_range: Tuple[str, str]) -> Dict:
        """Check if a specific dataset is available for a date range."""
        start_date, end_date = date_range
        
        # Find dataset in catalog
        dataset = None
        variable = None
        
        for var, info in self.dataset_catalog.items():
            for ds in info['datasets']:
                if ds['id'] == dataset_id:
                    dataset = ds
                    variable = var
                    break
            if dataset:
                break
        
        if not dataset:
            return {
                'available': False,
                'reason': f'Dataset {dataset_id} not found in catalog'
            }
        
        if self._dataset_covers_period(dataset, start_date, end_date):
            return {
                'available': True,
                'dataset': dataset,
                'variable': variable,
                'coverage': dataset['temporal_range']
            }
        else:
            return {
                'available': False,
                'reason': f'Dataset {dataset_id} does not cover period {start_date} to {end_date}',
                'dataset_coverage': dataset['temporal_range']
            }

def main():
    """Example usage of the DatasetSelector."""
    selector = DatasetSelector()
    
    # Example: Select SST dataset for recent period
    try:
        result = selector.select_optimal_dataset(
            'sea_surface_temperature',
            '2024-01-01',
            '2024-12-31',
            preferences={'prefer_low_latency': True}
        )
        print(f"Selected dataset: {result['dataset_id']}")
        print(f"Reason: {result['reason']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Get temporal coverage overview
    coverage = selector.get_temporal_coverage_map()
    print("\nTemporal Coverage Summary:")
    for var, info in coverage.items():
        print(f"{var}: {info['earliest_start']} to {info['latest_end']}")

if __name__ == "__main__":
    main()