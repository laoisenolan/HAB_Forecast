#!/usr/bin/env python3
"""
Enhanced Dataset Selector with Historical Satellite Integration

This module intelligently selects the best available datasets for any given
time period, automatically choosing historical satellites for older data
and current sources for recent data.

COVERAGE MATRIX:
- 1950-1992: ERA5 Atmospheric only
- 1992-2000: HYCOM Ocean + ERA5 Atmospheric  
- 2000-2015: HYCOM + ERA5 + MODIS Ocean Color
- 2015-2016: + Sentinel-2 Coastal High-Resolution
- 2016-2022: + Sentinel-3 Ocean Color High-Resolution
- 2022-present: + Copernicus Marine + ERDDAP + GEE
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json

from .historical_satellite_integrator import HistoricalSatelliteIntegrator
from .dataset_selector import DatasetSelector

logger = logging.getLogger(__name__)

class EnhancedDatasetSelector:
    """
    Enhanced dataset selector with historical satellite integration
    
    Automatically selects optimal data sources based on requested time period
    and variable requirements.
    """
    
    def __init__(self):
        """Initialize enhanced dataset selector"""
        
        # Initialize component selectors
        try:
            self.historical_integrator = HistoricalSatelliteIntegrator()
        except Exception as e:
            logger.warning(f"Historical integrator initialization failed: {e}")
            self.historical_integrator = None
            
        try:
            self.current_selector = DatasetSelector()
        except Exception as e:
            logger.warning(f"Current selector initialization failed: {e}")
            self.current_selector = None
        
        # Define temporal coverage epochs
        self.coverage_epochs = {
            'prehistoric': {
                'period': '1950-1992',
                'description': 'Atmospheric data only (ERA5)',
                'available_data_types': ['atmospheric'],
                'primary_sources': ['era5_extended'],
                'variable_count': 7,
                'completeness': 'LOW'
            },
            'early_satellite': {
                'period': '1992-2000', 
                'description': 'Ocean physics + Atmospheric',
                'available_data_types': ['ocean_physics', 'atmospheric'],
                'primary_sources': ['hycom_historical', 'era5_extended'],
                'variable_count': 11,
                'completeness': 'MEDIUM'
            },
            'ocean_color_era': {
                'period': '2000-2015',
                'description': 'Ocean physics + Color + Atmospheric',
                'available_data_types': ['ocean_physics', 'ocean_color', 'atmospheric'],
                'primary_sources': ['hycom_historical', 'modis_aqua_ocean', 'era5_extended'],
                'variable_count': 18,
                'completeness': 'GOOD'
            },
            'high_resolution': {
                'period': '2015-2016',
                'description': 'Previous + High-resolution coastal',
                'available_data_types': ['ocean_physics', 'ocean_color', 'atmospheric', 'coastal_high_res'],
                'primary_sources': ['hycom_historical', 'modis_aqua_ocean', 'era5_extended', 'sentinel2_coastal'],
                'variable_count': 24,
                'completeness': 'VERY_GOOD'
            },
            'multi_sensor': {
                'period': '2016-2022',
                'description': 'Multi-sensor ocean color + High-resolution',
                'available_data_types': ['ocean_physics', 'ocean_color', 'atmospheric', 'coastal_high_res', 'multi_spectral'],
                'primary_sources': ['hycom_historical', 'sentinel3_ocean', 'era5_extended', 'sentinel2_coastal', 'modis_aqua_ocean'],
                'variable_count': 30,
                'completeness': 'EXCELLENT'
            },
            'comprehensive': {
                'period': '2022-present',
                'description': 'All sources including real-time operational',
                'available_data_types': ['ocean_physics', 'ocean_color', 'atmospheric', 'coastal_high_res', 'multi_spectral', 'operational_nrt'],
                'primary_sources': ['hycom_historical', 'sentinel3_ocean', 'era5_extended', 'sentinel2_coastal', 'copernicus_marine', 'erddap_atmospheric'],
                'variable_count': 35,
                'completeness': 'MAXIMUM'
            }
        }
        
        logger.info("🎯 Enhanced Dataset Selector initialized")
        logger.info(f"📊 Coverage epochs: {len(self.coverage_epochs)}")
        logger.info(f"🕰️ Temporal range: 1950 to present (74+ years)")
        
    def determine_optimal_epoch(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Determine the optimal data collection epoch for the given date range
        """
        start_year = pd.to_datetime(start_date).year
        end_year = pd.to_datetime(end_date).year
        
        # Find which epoch(s) the date range spans
        applicable_epochs = []
        
        for epoch_name, epoch_info in self.coverage_epochs.items():
            period_start, period_end = epoch_info['period'].split('-')
            period_start_year = int(period_start)
            
            # Handle 'present' end dates
            if period_end == 'present':
                period_end_year = datetime.now().year
            else:
                period_end_year = int(period_end)
            
            # Check if date range overlaps with this epoch
            if not (end_year < period_start_year or start_year > period_end_year):
                applicable_epochs.append({
                    'name': epoch_name,
                    'info': epoch_info,
                    'overlap_start': max(start_year, period_start_year),
                    'overlap_end': min(end_year, period_end_year)
                })
        
        if not applicable_epochs:
            return {
                'error': 'No applicable epochs found',
                'requested_range': f"{start_date} to {end_date}"
            }
        
        # If spanning multiple epochs, choose the most comprehensive one
        best_epoch = max(applicable_epochs, key=lambda x: x['info']['variable_count'])
        
        return {
            'optimal_epoch': best_epoch['name'],
            'epoch_info': best_epoch['info'],
            'applicable_epochs': [e['name'] for e in applicable_epochs],
            'total_variables_available': best_epoch['info']['variable_count'],
            'data_completeness': best_epoch['info']['completeness'],
            'recommended_sources': best_epoch['info']['primary_sources']
        }
    
    def create_intelligent_collection_plan(self, 
                                         start_date: str, 
                                         end_date: str,
                                         required_variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create an intelligent data collection plan based on temporal requirements
        """
        logger.info(f"🧠 Creating intelligent collection plan: {start_date} to {end_date}")
        
        # Determine optimal epoch
        epoch_analysis = self.determine_optimal_epoch(start_date, end_date)
        
        if 'error' in epoch_analysis:
            return epoch_analysis
        
        optimal_epoch = epoch_analysis['optimal_epoch']
        epoch_info = epoch_analysis['epoch_info']
        
        # Create temporal segments for optimal data collection
        segments = self._create_temporal_segments(start_date, end_date)
        
        # Build collection plan
        collection_plan = {
            'overview': {
                'requested_period': f"{start_date} to {end_date}",
                'optimal_epoch': optimal_epoch,
                'total_duration_years': (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25,
                'data_completeness': epoch_info['completeness'],
                'estimated_variables': epoch_info['variable_count']
            },
            'temporal_segments': segments,
            'data_sources': self._plan_data_sources_by_segment(segments),
            'variable_coverage': self._analyze_variable_coverage(segments, required_variables),
            'collection_strategy': self._determine_collection_strategy(segments),
            'estimated_metrics': self._estimate_collection_metrics(segments)
        }
        
        return collection_plan
    
    def _create_temporal_segments(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Create temporal segments with different data source strategies"""
        
        segments = []
        current_date = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Define segment boundaries based on data availability changes
        boundaries = [
            ('1950-01-01', '1992-01-01', 'prehistoric'),
            ('1992-01-01', '2000-01-01', 'early_satellite'), 
            ('2000-01-01', '2015-06-01', 'ocean_color_era'),
            ('2015-06-01', '2016-10-01', 'high_resolution'),
            ('2016-10-01', '2022-01-01', 'multi_sensor'),
            ('2022-01-01', '2030-01-01', 'comprehensive')  # Future-proofed
        ]
        
        for i, (seg_start, seg_end, epoch_name) in enumerate(boundaries):
            seg_start_dt = pd.to_datetime(seg_start)
            seg_end_dt = pd.to_datetime(seg_end)
            
            # Check if this segment overlaps with requested range
            if not (end_dt <= seg_start_dt or current_date >= seg_end_dt):
                # Calculate overlap
                overlap_start = max(current_date, seg_start_dt)
                overlap_end = min(end_dt, seg_end_dt)
                
                if overlap_start < overlap_end:
                    segment = {
                        'segment_id': f"segment_{i+1}",
                        'period': f"{overlap_start.strftime('%Y-%m-%d')} to {overlap_end.strftime('%Y-%m-%d')}",
                        'start_date': overlap_start.strftime('%Y-%m-%d'),
                        'end_date': overlap_end.strftime('%Y-%m-%d'),
                        'duration_days': (overlap_end - overlap_start).days,
                        'epoch': epoch_name,
                        'epoch_info': self.coverage_epochs[epoch_name]
                    }
                    segments.append(segment)
        
        return segments
    
    def _plan_data_sources_by_segment(self, segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Plan optimal data sources for each temporal segment"""
        
        source_plan = {}
        
        for segment in segments:
            epoch_name = segment['epoch']
            epoch_info = segment['epoch_info']
            
            segment_sources = {
                'primary_sources': epoch_info['primary_sources'],
                'data_types': epoch_info['available_data_types'],
                'collection_method': self._determine_collection_method(epoch_name),
                'expected_variables': epoch_info['variable_count'],
                'data_quality': epoch_info['completeness']
            }
            
            source_plan[segment['segment_id']] = segment_sources
        
        return source_plan
    
    def _determine_collection_method(self, epoch_name: str) -> str:
        """Determine the best collection method for an epoch"""
        
        methods = {
            'prehistoric': 'historical_satellite_only',
            'early_satellite': 'historical_satellite_primary',
            'ocean_color_era': 'historical_satellite_primary',
            'high_resolution': 'historical_satellite_primary', 
            'multi_sensor': 'hybrid_historical_current',
            'comprehensive': 'current_operational_primary'
        }
        
        return methods.get(epoch_name, 'hybrid_historical_current')
    
    def _analyze_variable_coverage(self, 
                                 segments: List[Dict[str, Any]], 
                                 required_variables: Optional[List[str]]) -> Dict[str, Any]:
        """Analyze variable coverage across temporal segments"""
        
        # Collect all available variables across segments
        all_variables = set()
        segment_coverage = {}
        
        for segment in segments:
            epoch_info = segment['epoch_info']
            available_types = epoch_info['available_data_types']
            
            # Map data types to variable categories
            type_variables = {
                'atmospheric': ['air_temperature', 'wind_u', 'wind_v', 'precipitation', 'pressure'],
                'ocean_physics': ['sea_temperature', 'sea_salinity', 'mixed_layer_depth'],
                'ocean_color': ['chlorophyll', 'turbidity', 'ocean_reflectance'],
                'coastal_high_res': ['coastal_turbidity', 'coastal_reflectance'],
                'multi_spectral': ['multi_band_radiance', 'fluorescence_bands'],
                'operational_nrt': ['real_time_sst', 'nrt_chlorophyll', 'operational_physics']
            }
            
            segment_variables = set()
            for data_type in available_types:
                if data_type in type_variables:
                    segment_variables.update(type_variables[data_type])
            
            all_variables.update(segment_variables)
            segment_coverage[segment['segment_id']] = {
                'available_variables': list(segment_variables),
                'variable_count': len(segment_variables)
            }
        
        # Analyze coverage for required variables if specified
        coverage_analysis = {
            'total_unique_variables': len(all_variables),
            'all_available_variables': sorted(list(all_variables)),
            'segment_coverage': segment_coverage
        }
        
        if required_variables:
            missing_variables = [var for var in required_variables if var not in all_variables]
            coverage_analysis.update({
                'required_variables': required_variables,
                'available_required': [var for var in required_variables if var in all_variables],
                'missing_required': missing_variables,
                'requirement_satisfaction': (len(required_variables) - len(missing_variables)) / len(required_variables) if required_variables else 1.0
            })
        
        return coverage_analysis
    
    def _determine_collection_strategy(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine overall collection strategy"""
        
        total_duration = sum(segment['duration_days'] for segment in segments)
        
        # Categorize segments by method
        method_counts = {}
        for segment in segments:
            epoch = segment['epoch']
            method = self._determine_collection_method(epoch)
            method_counts[method] = method_counts.get(method, 0) + segment['duration_days']
        
        # Determine primary strategy
        primary_method = max(method_counts.items(), key=lambda x: x[1])[0]
        
        strategy = {
            'primary_strategy': primary_method,
            'method_distribution': {
                method: days/total_duration for method, days in method_counts.items()
            },
            'collection_phases': len(segments),
            'requires_historical_integration': any('historical' in self._determine_collection_method(seg['epoch']) for seg in segments),
            'requires_current_integration': any('current' in self._determine_collection_method(seg['epoch']) for seg in segments)
        }
        
        return strategy
    
    def _estimate_collection_metrics(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate collection metrics and requirements"""
        
        total_days = sum(segment['duration_days'] for segment in segments)
        total_variables = sum(segment['epoch_info']['variable_count'] for segment in segments) / len(segments)  # Average
        
        # Rough estimates based on experience
        estimated_records = total_days * 5 * total_variables  # 5 locations average
        estimated_size_mb = estimated_records * 0.1  # ~0.1 MB per 1000 records
        estimated_collection_hours = len(segments) * 2  # 2 hours per segment average
        
        return {
            'total_duration_days': total_days,
            'average_variables_per_segment': int(total_variables),
            'estimated_total_records': int(estimated_records),
            'estimated_data_size_mb': round(estimated_size_mb, 1),
            'estimated_collection_time_hours': estimated_collection_hours,
            'complexity_score': len(segments) * 2 + (1 if total_days > 3650 else 0)  # Complexity based on segments and duration
        }
    
    def select_optimal_datasets_for_period(self, 
                                         start_date: str, 
                                         end_date: str,
                                         variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main method: Select optimal datasets for a given period
        """
        logger.info(f"🎯 Selecting optimal datasets for {start_date} to {end_date}")
        
        # Create intelligent collection plan
        plan = self.create_intelligent_collection_plan(start_date, end_date, variables)
        
        if 'error' in plan:
            return plan
        
        # Convert plan to actionable dataset selection
        dataset_selection = {
            'collection_plan': plan,
            'recommended_datasets': self._convert_plan_to_datasets(plan),
            'collection_order': self._determine_collection_order(plan),
            'fallback_strategies': self._create_fallback_strategies(plan),
            'data_integration_strategy': self._plan_data_integration(plan)
        }
        
        return dataset_selection
    
    def _convert_plan_to_datasets(self, plan: Dict[str, Any]) -> Dict[str, List[str]]:
        """Convert collection plan to specific dataset IDs"""
        
        datasets_by_segment = {}
        
        for segment_id, source_info in plan['data_sources'].items():
            datasets_by_segment[segment_id] = source_info['primary_sources']
        
        return datasets_by_segment
    
    def _determine_collection_order(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Determine optimal collection order for segments"""
        
        segments = plan['temporal_segments']
        
        # Sort by start date
        sorted_segments = sorted(segments, key=lambda x: x['start_date'])
        
        collection_order = []
        for i, segment in enumerate(sorted_segments):
            collection_order.append({
                'order': i + 1,
                'segment_id': segment['segment_id'],
                'period': segment['period'],
                'primary_method': self._determine_collection_method(segment['epoch']),
                'priority': 'HIGH' if segment['duration_days'] > 365 else 'MEDIUM'
            })
        
        return collection_order
    
    def _create_fallback_strategies(self, plan: Dict[str, Any]) -> List[str]:
        """Create fallback strategies if primary collection fails"""
        
        fallback_strategies = [
            "If historical satellites fail: Use ERA5 for atmospheric data + interpolated ocean climatology",
            "If ocean color unavailable: Use turbidity proxies from high-resolution optical data",
            "If high-resolution coastal data unavailable: Use lower resolution but longer temporal coverage",
            "If real-time data unavailable: Use reanalysis or delayed-mode products",
            "If specific satellites unavailable: Use alternative sensors from same time period"
        ]
        
        return fallback_strategies
    
    def _plan_data_integration(self, plan: Dict[str, Any]) -> Dict[str, str]:
        """Plan how to integrate data from different sources and time periods"""
        
        strategy = plan['collection_strategy']
        
        integration_strategy = {
            'temporal_alignment': 'Resample all data to daily resolution with consistent timestamps',
            'spatial_alignment': 'Regrid to common 0.05° resolution centered on Seto Inland Sea',
            'variable_standardization': 'Apply consistent units and naming conventions across all sources',
            'quality_harmonization': 'Apply source-specific quality flags and bias corrections',
            'gap_filling': 'Use temporal interpolation for short gaps, climatology for longer gaps'
        }
        
        if strategy['requires_historical_integration']:
            integration_strategy['historical_integration'] = 'Apply inter-calibration between historical and current sensors'
        
        if len(plan['temporal_segments']) > 1:
            integration_strategy['multi_segment_merging'] = 'Merge segments with overlap analysis and bias correction'
        
        return integration_strategy
    
    def generate_collection_summary(self, selection: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the collection plan"""
        
        plan = selection['collection_plan']
        overview = plan['overview']
        
        summary = f"""
🎯 ENHANCED DATA COLLECTION PLAN SUMMARY

📅 Period: {overview['requested_period']} ({overview['total_duration_years']:.1f} years)
🏆 Optimal Epoch: {overview['optimal_epoch']} 
📊 Data Completeness: {overview['data_completeness']}
🔢 Estimated Variables: {overview['estimated_variables']}

📊 TEMPORAL SEGMENTS: {len(plan['temporal_segments'])}
"""
        
        for i, segment in enumerate(plan['temporal_segments'], 1):
            summary += f"   {i}. {segment['period']} ({segment['epoch']})\n"
        
        summary += f"""
🛰️ COLLECTION STRATEGY: {plan['collection_strategy']['primary_strategy']}
⏱️ Estimated Time: {plan['estimated_metrics']['estimated_collection_time_hours']} hours
💾 Estimated Size: {plan['estimated_metrics']['estimated_data_size_mb']} MB
📈 Complexity: {plan['estimated_metrics']['complexity_score']}/10

✅ Ready for execution with {len(plan['temporal_segments'])} collection phases
"""
        
        return summary