"""
Task 2.4: Temporal Gap Filling using Climatology and Interpolation

This module implements comprehensive temporal gap filling strategies for HAB
forecasting datasets, using climatological baselines, statistical interpolation,
and machine learning approaches to fill missing data points.

Features:
- Multiple gap filling strategies (climatology, interpolation, ML)
- Seasonal and trend-aware gap filling
- Quality-weighted interpolation
- Gap filling validation and uncertainty estimation
- Automatic strategy selection based on gap characteristics
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
from dataclasses import dataclass
from scipy.interpolate import interp1d, UnivariateSpline
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

@dataclass
class GapInfo:
    """Information about a data gap"""
    gap_id: str
    variable: str
    start_date: datetime
    end_date: datetime
    duration_days: int
    gap_type: str  # 'short', 'medium', 'long', 'extreme'
    surrounding_data_quality: str  # 'good', 'fair', 'poor'
    recommended_strategy: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'gap_id': self.gap_id,
            'variable': self.variable,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'duration_days': self.duration_days,
            'gap_type': self.gap_type,
            'surrounding_data_quality': self.surrounding_data_quality,
            'recommended_strategy': self.recommended_strategy
        }

@dataclass
class GapFillingResult:
    """Result of gap filling operation"""
    gap_id: str
    strategy_used: str
    filled_points: int
    uncertainty_mean: float
    uncertainty_std: float
    validation_score: Optional[float]
    quality_flag: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'gap_id': self.gap_id,
            'strategy_used': self.strategy_used,
            'filled_points': self.filled_points,
            'uncertainty_mean': self.uncertainty_mean,
            'uncertainty_std': self.uncertainty_std,
            'validation_score': self.validation_score,
            'quality_flag': self.quality_flag
        }

class TemporalGapFiller:
    """
    Comprehensive temporal gap filling system for HAB datasets
    
    Gap Filling Strategies:
    1. Linear Interpolation - for short gaps (<7 days)
    2. Spline Interpolation - for medium gaps (7-30 days) 
    3. Climatological Mean - for seasonal patterns
    4. Climatological Anomaly - preserves trends + seasonal patterns
    5. Machine Learning - for complex gaps with auxiliary variables
    6. Hybrid Approach - combines multiple methods
    """
    
    def __init__(self, output_dir: str = "data/gap_filled"):
        """Initialize temporal gap filler"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Gap classification thresholds (days)
        self.gap_thresholds = {
            'short': 7,      # ≤7 days
            'medium': 30,    # 8-30 days
            'long': 90,      # 31-90 days
            'extreme': 999   # >90 days
        }
        
        # Strategy preferences by gap type
        self.strategy_preferences = {
            'short': ['linear_interpolation', 'spline_interpolation'],
            'medium': ['spline_interpolation', 'climatological_anomaly'],
            'long': ['climatological_anomaly', 'machine_learning'],
            'extreme': ['climatological_mean', 'machine_learning']
        }
        
        # Seasonal periods for climatology
        self.seasonal_periods = {
            'monthly': 30,
            'quarterly': 91,
            'yearly': 365
        }
        
        logger.info("Temporal gap filler initialized")
    
    def fill_gaps(self, 
                  data: pd.DataFrame, 
                  gap_filling_config: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fill temporal gaps in HAB dataset
        
        Args:
            data: Historical data with gaps
            gap_filling_config: Configuration for gap filling
            
        Returns:
            Tuple of (gap_filled_data, filling_report)
        """
        logger.info("🔄 Starting temporal gap filling process")
        
        # Prepare data
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values(['variable_name', 'timestamp', 'latitude', 'longitude'])
        
        # Initialize filling report
        filling_report = {
            "start_time": datetime.now().isoformat(),
            "original_records": len(data),
            "gaps_identified": {},
            "gaps_filled": {},
            "strategies_used": {},
            "quality_metrics": {},
            "summary": {}
        }
        
        # Process each variable separately
        filled_data_list = []
        
        for variable in data['variable_name'].unique():
            logger.info(f"🔍 Processing variable: {variable}")
            
            var_data = data[data['variable_name'] == variable].copy()
            
            # Identify gaps
            gaps = self._identify_gaps(var_data, variable)
            filling_report["gaps_identified"][variable] = [gap.to_dict() for gap in gaps]
            
            # Fill gaps
            if gaps:
                filled_var_data, var_results = self._fill_variable_gaps(var_data, gaps, variable)
                filling_report["gaps_filled"][variable] = [result.to_dict() for result in var_results]
            else:
                filled_var_data = var_data
                filling_report["gaps_filled"][variable] = []
            
            filled_data_list.append(filled_var_data)
        
        # Combine all filled data
        filled_data = pd.concat(filled_data_list, ignore_index=True)
        
        # Generate summary statistics
        filling_report["summary"] = self._generate_filling_summary(data, filled_data, filling_report)
        filling_report["end_time"] = datetime.now().isoformat()
        
        # Save gap filling report
        report_path = self._save_gap_filling_report(filling_report)
        filling_report["report_path"] = str(report_path)
        
        logger.info(f"✅ Gap filling completed: {len(filled_data) - len(data)} points added")
        
        return filled_data, filling_report
    
    def _identify_gaps(self, var_data: pd.DataFrame, variable: str) -> List[GapInfo]:
        """Identify temporal gaps in variable data"""
        logger.info(f"🔍 Identifying gaps in {variable}")
        
        gaps = []
        gap_counter = 1
        
        # Group by location to handle spatial variations
        location_groups = var_data.groupby(['latitude', 'longitude'])
        
        for (lat, lon), location_data in location_groups:
            location_data = location_data.sort_values('timestamp')
            
            if len(location_data) < 2:
                continue
                
            # Calculate time differences
            time_diffs = location_data['timestamp'].diff().dropna()
            
            # Identify gaps larger than expected frequency (>1 day typically)
            expected_interval = pd.Timedelta(days=1)  # Assume daily data
            gap_threshold = expected_interval * 1.5  # 1.5x expected interval
            
            large_gaps = time_diffs[time_diffs > gap_threshold]
            
            for i, (gap_idx, gap_duration) in enumerate(large_gaps.items()):
                gap_start_idx = gap_idx - 1
                gap_start = location_data.loc[gap_start_idx, 'timestamp']
                gap_end = location_data.loc[gap_idx, 'timestamp']
                duration_days = gap_duration.total_seconds() / (24 * 3600)
                
                # Classify gap type
                gap_type = self._classify_gap_type(duration_days)
                
                # Assess surrounding data quality
                surrounding_quality = self._assess_surrounding_quality(
                    location_data, gap_start_idx, gap_idx
                )
                
                # Recommend strategy
                recommended_strategy = self._recommend_strategy(gap_type, surrounding_quality, duration_days)
                
                gap = GapInfo(
                    gap_id=f"{variable}_{lat:.3f}_{lon:.3f}_gap_{gap_counter}",
                    variable=variable,
                    start_date=gap_start,
                    end_date=gap_end,
                    duration_days=int(duration_days),
                    gap_type=gap_type,
                    surrounding_data_quality=surrounding_quality,
                    recommended_strategy=recommended_strategy
                )
                
                gaps.append(gap)
                gap_counter += 1
        
        logger.info(f"   Found {len(gaps)} gaps in {variable}")
        
        return gaps
    
    def _classify_gap_type(self, duration_days: float) -> str:
        """Classify gap based on duration"""
        if duration_days <= self.gap_thresholds['short']:
            return 'short'
        elif duration_days <= self.gap_thresholds['medium']:
            return 'medium'
        elif duration_days <= self.gap_thresholds['long']:
            return 'long'
        else:
            return 'extreme'
    
    def _assess_surrounding_quality(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> str:
        """Assess quality of data surrounding a gap"""
        # Look at data quality flags around the gap
        before_data = data.iloc[max(0, start_idx-5):start_idx]
        after_data = data.iloc[end_idx:min(len(data), end_idx+5)]
        
        surrounding_data = pd.concat([before_data, after_data])
        
        if len(surrounding_data) == 0:
            return 'poor'
        
        good_quality = (surrounding_data['quality_flag'] == 'good').mean()
        
        if good_quality >= 0.8:
            return 'good'
        elif good_quality >= 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def _recommend_strategy(self, gap_type: str, surrounding_quality: str, duration_days: float) -> str:
        """Recommend gap filling strategy"""
        preferences = self.strategy_preferences.get(gap_type, ['climatological_mean'])
        
        # Adjust based on surrounding quality
        if surrounding_quality == 'good':
            if gap_type == 'short':
                return 'linear_interpolation'
            elif gap_type == 'medium':
                return 'spline_interpolation'
            else:
                return 'climatological_anomaly'
        elif surrounding_quality == 'fair':
            return 'climatological_anomaly'
        else:
            return 'climatological_mean'
    
    def _fill_variable_gaps(self, 
                           var_data: pd.DataFrame, 
                           gaps: List[GapInfo], 
                           variable: str) -> Tuple[pd.DataFrame, List[GapFillingResult]]:
        """Fill gaps for a specific variable"""
        logger.info(f"🔧 Filling {len(gaps)} gaps in {variable}")
        
        filled_data = var_data.copy()
        filling_results = []
        
        # Build climatology for this variable
        climatology = self._build_climatology(var_data, variable)
        
        for gap in gaps:
            try:
                # Fill gap using recommended strategy
                filled_points, uncertainty, validation_score = self._apply_gap_filling_strategy(
                    filled_data, gap, climatology, variable
                )
                
                # Create result record
                result = GapFillingResult(
                    gap_id=gap.gap_id,
                    strategy_used=gap.recommended_strategy,
                    filled_points=filled_points,
                    uncertainty_mean=uncertainty.get('mean', np.nan),
                    uncertainty_std=uncertainty.get('std', np.nan),
                    validation_score=validation_score,
                    quality_flag='gap_filled'
                )
                
                filling_results.append(result)
                
                logger.info(f"   ✅ Filled gap {gap.gap_id}: {filled_points} points using {gap.recommended_strategy}")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to fill gap {gap.gap_id}: {e}")
                
                # Create failure record
                result = GapFillingResult(
                    gap_id=gap.gap_id,
                    strategy_used="failed",
                    filled_points=0,
                    uncertainty_mean=np.nan,
                    uncertainty_std=np.nan,
                    validation_score=None,
                    quality_flag='gap_fill_failed'
                )
                filling_results.append(result)
        
        return filled_data, filling_results
    
    def _build_climatology(self, var_data: pd.DataFrame, variable: str) -> Dict[str, Any]:
        """Build climatology for gap filling"""
        logger.info(f"📊 Building climatology for {variable}")
        
        # Ensure we have clean data for climatology
        clean_data = var_data[var_data['quality_flag'] == 'good'].copy()
        clean_data['month'] = clean_data['timestamp'].dt.month
        clean_data['day_of_year'] = clean_data['timestamp'].dt.dayofyear
        
        climatology = {
            'monthly_means': {},
            'monthly_stds': {},
            'daily_means': {},
            'overall_stats': {},
            'seasonal_cycle': {}
        }
        
        if len(clean_data) > 0:
            # Monthly climatology
            monthly_stats = clean_data.groupby('month')['value'].agg(['mean', 'std', 'count'])
            climatology['monthly_means'] = monthly_stats['mean'].to_dict()
            climatology['monthly_stds'] = monthly_stats['std'].fillna(0).to_dict()
            
            # Daily climatology (if enough data)
            if len(clean_data) > 365:
                daily_stats = clean_data.groupby('day_of_year')['value'].agg(['mean', 'std'])
                climatology['daily_means'] = daily_stats['mean'].to_dict()
            
            # Overall statistics
            climatology['overall_stats'] = {
                'mean': clean_data['value'].mean(),
                'std': clean_data['value'].std(),
                'min': clean_data['value'].min(),
                'max': clean_data['value'].max(),
                'median': clean_data['value'].median()
            }
            
            # Seasonal cycle (fit harmonic)
            try:
                climatology['seasonal_cycle'] = self._fit_seasonal_cycle(clean_data)
            except Exception as e:
                logger.warning(f"Could not fit seasonal cycle for {variable}: {e}")
                climatology['seasonal_cycle'] = {}
        
        return climatology
    
    def _fit_seasonal_cycle(self, data: pd.DataFrame) -> Dict[str, float]:
        """Fit harmonic seasonal cycle to data"""
        # Convert timestamps to day of year
        data['day_of_year'] = data['timestamp'].dt.dayofyear
        
        # Fit harmonic model: value = a0 + a1*cos(2π*day/365) + b1*sin(2π*day/365)
        X = np.column_stack([
            np.ones(len(data)),
            np.cos(2 * np.pi * data['day_of_year'] / 365),
            np.sin(2 * np.pi * data['day_of_year'] / 365)
        ])
        
        # Robust fitting using least squares
        coeffs, residuals, rank, s = np.linalg.lstsq(X, data['value'].values, rcond=None)
        
        return {
            'a0': coeffs[0],  # Annual mean
            'a1': coeffs[1],  # Cosine coefficient
            'b1': coeffs[2],  # Sine coefficient
            'rmse': np.sqrt(residuals[0] / len(data)) if len(residuals) > 0 else np.nan
        }
    
    def _apply_gap_filling_strategy(self, 
                                  data: pd.DataFrame, 
                                  gap: GapInfo, 
                                  climatology: Dict[str, Any],
                                  variable: str) -> Tuple[int, Dict[str, float], Optional[float]]:
        """Apply specific gap filling strategy"""
        
        if gap.recommended_strategy == 'linear_interpolation':
            return self._fill_linear_interpolation(data, gap, variable)
        
        elif gap.recommended_strategy == 'spline_interpolation':
            return self._fill_spline_interpolation(data, gap, variable)
        
        elif gap.recommended_strategy == 'climatological_mean':
            return self._fill_climatological_mean(data, gap, climatology, variable)
        
        elif gap.recommended_strategy == 'climatological_anomaly':
            return self._fill_climatological_anomaly(data, gap, climatology, variable)
        
        elif gap.recommended_strategy == 'machine_learning':
            return self._fill_machine_learning(data, gap, variable)
        
        else:
            # Fallback to climatological mean
            return self._fill_climatological_mean(data, gap, climatology, variable)
    
    def _fill_linear_interpolation(self, 
                                 data: pd.DataFrame, 
                                 gap: GapInfo,
                                 variable: str) -> Tuple[int, Dict[str, float], Optional[float]]:
        """Fill gap using linear interpolation"""
        
        # Find data points around the gap for the same location
        gap_data = data[
            (data['variable_name'] == variable) & 
            (data['timestamp'] >= gap.start_date - timedelta(days=30)) &
            (data['timestamp'] <= gap.end_date + timedelta(days=30))
        ].copy()
        
        if len(gap_data) < 4:  # Need at least 4 points for interpolation
            raise ValueError("Insufficient data for linear interpolation")
        
        # Sort by timestamp
        gap_data = gap_data.sort_values('timestamp')
        
        # Remove gap period
        gap_data = gap_data[
            (gap_data['timestamp'] < gap.start_date) |
            (gap_data['timestamp'] > gap.end_date)
        ]
        
        # Create interpolation function
        timestamps = gap_data['timestamp'].astype(np.int64) // 10**9  # Convert to seconds
        values = gap_data['value'].values
        
        # Remove NaN values
        valid_mask = ~np.isnan(values)
        timestamps = timestamps[valid_mask]
        values = values[valid_mask]
        
        if len(timestamps) < 2:
            raise ValueError("Insufficient valid data for interpolation")
        
        interp_func = interp1d(timestamps, values, kind='linear', fill_value='extrapolate')
        
        # Generate timestamps for gap filling
        gap_start_sec = gap.start_date.timestamp()
        gap_end_sec = gap.end_date.timestamp()
        
        # Daily interpolation points
        gap_timestamps = np.arange(
            gap_start_sec + 86400,  # Start from next day
            gap_end_sec,           # Until end date (exclusive)
            86400                  # Daily interval
        )
        
        # Interpolate values
        interpolated_values = interp_func(gap_timestamps)
        
        # Add interpolated points to data
        filled_points = 0
        uncertainties = []
        
        for ts, value in zip(gap_timestamps, interpolated_values):
            # Find matching location info
            location_data = data[
                (data['variable_name'] == variable) & 
                (data['timestamp'] == gap.start_date)
            ]
            
            if len(location_data) > 0:
                # Get location coordinates
                lat = location_data.iloc[0]['latitude']
                lon = location_data.iloc[0]['longitude']
                
                # Create new row
                new_row = {
                    'timestamp': pd.to_datetime(ts, unit='s'),
                    'latitude': lat,
                    'longitude': lon,
                    'variable_name': variable,
                    'value': value,
                    'unit': location_data.iloc[0]['unit'],
                    'data_source': 'gap_filled_linear',
                    'dataset_id': f'gap_filled_{gap.gap_id}',
                    'quality_flag': 'gap_filled',
                    'collection_method': 'linear_interpolation',
                    'spatial_resolution': location_data.iloc[0]['spatial_resolution'],
                    'temporal_resolution': 'daily'
                }
                
                # Add to data
                data.loc[len(data)] = new_row
                filled_points += 1
                
                # Estimate uncertainty (simple approach)
                uncertainty = abs(value) * 0.1  # 10% uncertainty for linear interpolation
                uncertainties.append(uncertainty)
        
        # Calculate uncertainty statistics
        uncertainty_stats = {
            'mean': np.mean(uncertainties) if uncertainties else 0,
            'std': np.std(uncertainties) if len(uncertainties) > 1 else 0
        }
        
        # Validation score (R² if possible to calculate)
        validation_score = None
        
        return filled_points, uncertainty_stats, validation_score
    
    def _fill_spline_interpolation(self, 
                                 data: pd.DataFrame, 
                                 gap: GapInfo,
                                 variable: str) -> Tuple[int, Dict[str, float], Optional[float]]:
        """Fill gap using spline interpolation"""
        # Similar to linear but using UnivariateSpline
        gap_data = data[
            (data['variable_name'] == variable) & 
            (data['timestamp'] >= gap.start_date - timedelta(days=60)) &
            (data['timestamp'] <= gap.end_date + timedelta(days=60))
        ].copy()
        
        if len(gap_data) < 6:  # Need more points for spline
            raise ValueError("Insufficient data for spline interpolation")
        
        gap_data = gap_data.sort_values('timestamp')
        
        # Remove gap period
        gap_data = gap_data[
            (gap_data['timestamp'] < gap.start_date) |
            (gap_data['timestamp'] > gap.end_date)
        ]
        
        timestamps = gap_data['timestamp'].astype(np.int64) // 10**9
        values = gap_data['value'].values
        
        # Remove NaN values
        valid_mask = ~np.isnan(values)
        timestamps = timestamps[valid_mask]
        values = values[valid_mask]
        
        if len(timestamps) < 4:
            raise ValueError("Insufficient valid data for spline interpolation")
        
        # Create spline
        spline = UnivariateSpline(timestamps, values, s=0.5)  # Some smoothing
        
        # Generate gap timestamps
        gap_start_sec = gap.start_date.timestamp()
        gap_end_sec = gap.end_date.timestamp()
        gap_timestamps = np.arange(gap_start_sec + 86400, gap_end_sec, 86400)
        
        # Interpolate
        interpolated_values = spline(gap_timestamps)
        
        # Add to data (similar to linear)
        filled_points = 0
        uncertainties = []
        
        for ts, value in zip(gap_timestamps, interpolated_values):
            # Similar process as linear interpolation...
            location_data = data[
                (data['variable_name'] == variable) & 
                (data['timestamp'] == gap.start_date)
            ]
            
            if len(location_data) > 0:
                lat = location_data.iloc[0]['latitude']
                lon = location_data.iloc[0]['longitude']
                
                new_row = {
                    'timestamp': pd.to_datetime(ts, unit='s'),
                    'latitude': lat,
                    'longitude': lon,
                    'variable_name': variable,
                    'value': value,
                    'unit': location_data.iloc[0]['unit'],
                    'data_source': 'gap_filled_spline',
                    'dataset_id': f'gap_filled_{gap.gap_id}',
                    'quality_flag': 'gap_filled',
                    'collection_method': 'spline_interpolation',
                    'spatial_resolution': location_data.iloc[0]['spatial_resolution'],
                    'temporal_resolution': 'daily'
                }
                
                data.loc[len(data)] = new_row
                filled_points += 1
                
                # Higher uncertainty for spline
                uncertainty = abs(value) * 0.15
                uncertainties.append(uncertainty)
        
        uncertainty_stats = {
            'mean': np.mean(uncertainties) if uncertainties else 0,
            'std': np.std(uncertainties) if len(uncertainties) > 1 else 0
        }
        
        return filled_points, uncertainty_stats, None
    
    def _fill_climatological_mean(self, 
                                data: pd.DataFrame, 
                                gap: GapInfo, 
                                climatology: Dict[str, Any],
                                variable: str) -> Tuple[int, Dict[str, float], Optional[float]]:
        """Fill gap using climatological mean values"""
        
        if not climatology['monthly_means']:
            raise ValueError("No climatology available for gap filling")
        
        filled_points = 0
        uncertainties = []
        
        # Generate daily timestamps in gap
        current_date = gap.start_date + timedelta(days=1)
        
        while current_date < gap.end_date:
            month = current_date.month
            
            # Get climatological value
            if month in climatology['monthly_means']:
                clim_value = climatology['monthly_means'][month]
                clim_std = climatology['monthly_stds'].get(month, clim_value * 0.2)
            else:
                clim_value = climatology['overall_stats']['mean']
                clim_std = climatology['overall_stats']['std']
            
            # Add some daily variation
            daily_variation = np.random.normal(0, clim_std * 0.1)
            filled_value = clim_value + daily_variation
            
            # Find location info
            location_data = data[
                (data['variable_name'] == variable) & 
                (data['timestamp'] == gap.start_date)
            ]
            
            if len(location_data) > 0:
                lat = location_data.iloc[0]['latitude']
                lon = location_data.iloc[0]['longitude']
                
                new_row = {
                    'timestamp': current_date,
                    'latitude': lat,
                    'longitude': lon,
                    'variable_name': variable,
                    'value': filled_value,
                    'unit': location_data.iloc[0]['unit'],
                    'data_source': 'gap_filled_climatology',
                    'dataset_id': f'gap_filled_{gap.gap_id}',
                    'quality_flag': 'gap_filled',
                    'collection_method': 'climatological_mean',
                    'spatial_resolution': location_data.iloc[0]['spatial_resolution'],
                    'temporal_resolution': 'daily'
                }
                
                data.loc[len(data)] = new_row
                filled_points += 1
                
                # Uncertainty is climatological standard deviation
                uncertainties.append(clim_std)
            
            current_date += timedelta(days=1)
        
        uncertainty_stats = {
            'mean': np.mean(uncertainties) if uncertainties else 0,
            'std': np.std(uncertainties) if len(uncertainties) > 1 else 0
        }
        
        return filled_points, uncertainty_stats, None
    
    def _fill_climatological_anomaly(self, 
                                   data: pd.DataFrame, 
                                   gap: GapInfo, 
                                   climatology: Dict[str, Any],
                                   variable: str) -> Tuple[int, Dict[str, float], Optional[float]]:
        """Fill gap using climatological anomaly approach"""
        
        # This preserves trends by using climatology + recent anomaly pattern
        # Get recent data to estimate anomaly pattern
        recent_data = data[
            (data['variable_name'] == variable) & 
            (data['timestamp'] >= gap.start_date - timedelta(days=90)) &
            (data['timestamp'] < gap.start_date) &
            (data['quality_flag'] == 'good')
        ]
        
        if len(recent_data) == 0:
            # Fallback to climatological mean
            return self._fill_climatological_mean(data, gap, climatology, variable)
        
        # Calculate recent anomalies
        recent_data = recent_data.copy()
        recent_data['month'] = recent_data['timestamp'].dt.month
        recent_anomalies = []
        
        for _, row in recent_data.iterrows():
            month = row['month']
            clim_value = climatology['monthly_means'].get(month, climatology['overall_stats']['mean'])
            anomaly = row['value'] - clim_value
            recent_anomalies.append(anomaly)
        
        # Estimate trend in anomalies
        if len(recent_anomalies) > 1:
            anomaly_trend = np.polyfit(range(len(recent_anomalies)), recent_anomalies, 1)[0]
            base_anomaly = recent_anomalies[-1]  # Most recent anomaly
        else:
            anomaly_trend = 0
            base_anomaly = recent_anomalies[0] if recent_anomalies else 0
        
        # Fill gap
        filled_points = 0
        uncertainties = []
        
        current_date = gap.start_date + timedelta(days=1)
        days_since_start = 1
        
        while current_date < gap.end_date:
            month = current_date.month
            
            # Get climatological value
            clim_value = climatology['monthly_means'].get(month, climatology['overall_stats']['mean'])
            
            # Project anomaly
            projected_anomaly = base_anomaly + anomaly_trend * days_since_start
            
            # Combine climatology + anomaly
            filled_value = clim_value + projected_anomaly
            
            # Add location and save
            location_data = data[
                (data['variable_name'] == variable) & 
                (data['timestamp'] == gap.start_date)
            ]
            
            if len(location_data) > 0:
                lat = location_data.iloc[0]['latitude']
                lon = location_data.iloc[0]['longitude']
                
                new_row = {
                    'timestamp': current_date,
                    'latitude': lat,
                    'longitude': lon,
                    'variable_name': variable,
                    'value': filled_value,
                    'unit': location_data.iloc[0]['unit'],
                    'data_source': 'gap_filled_anomaly',
                    'dataset_id': f'gap_filled_{gap.gap_id}',
                    'quality_flag': 'gap_filled',
                    'collection_method': 'climatological_anomaly',
                    'spatial_resolution': location_data.iloc[0]['spatial_resolution'],
                    'temporal_resolution': 'daily'
                }
                
                data.loc[len(data)] = new_row
                filled_points += 1
                
                # Uncertainty combines climatological + trend uncertainty
                clim_std = climatology['monthly_stds'].get(month, clim_value * 0.2)
                trend_uncertainty = abs(projected_anomaly) * 0.2
                total_uncertainty = np.sqrt(clim_std**2 + trend_uncertainty**2)
                uncertainties.append(total_uncertainty)
            
            current_date += timedelta(days=1)
            days_since_start += 1
        
        uncertainty_stats = {
            'mean': np.mean(uncertainties) if uncertainties else 0,
            'std': np.std(uncertainties) if len(uncertainties) > 1 else 0
        }
        
        return filled_points, uncertainty_stats, None
    
    def _fill_machine_learning(self, 
                             data: pd.DataFrame, 
                             gap: GapInfo,
                             variable: str) -> Tuple[int, Dict[str, float], Optional[float]]:
        """Fill gap using machine learning approach"""
        
        # This is a simplified ML approach using other variables as predictors
        # Get data for ML training
        training_data = data[
            (data['timestamp'] >= gap.start_date - timedelta(days=365)) &
            (data['timestamp'] <= gap.end_date + timedelta(days=365)) &
            (data['quality_flag'] == 'good')
        ].copy()
        
        if len(training_data) < 100:
            # Fallback to climatology if insufficient training data
            climatology = self._build_climatology(data[data['variable_name'] == variable], variable)
            return self._fill_climatological_mean(data, gap, climatology, variable)
        
        # Prepare features and target
        # Pivot to get variables as columns
        ml_data = training_data.pivot_table(
            values='value', 
            index=['timestamp', 'latitude', 'longitude'], 
            columns='variable_name', 
            aggfunc='mean'
        ).reset_index()
        
        # Check if target variable exists
        if variable not in ml_data.columns:
            climatology = self._build_climatology(data[data['variable_name'] == variable], variable)
            return self._fill_climatological_mean(data, gap, climatology, variable)
        
        # Remove rows where target variable is missing
        ml_data = ml_data.dropna(subset=[variable])
        
        if len(ml_data) < 50:
            climatology = self._build_climatology(data[data['variable_name'] == variable], variable)
            return self._fill_climatological_mean(data, gap, climatology, variable)
        
        # Prepare features
        feature_columns = [col for col in ml_data.columns if col not in ['timestamp', 'latitude', 'longitude', variable]]
        
        if len(feature_columns) < 2:
            # Not enough predictor variables
            climatology = self._build_climatology(data[data['variable_name'] == variable], variable)
            return self._fill_climatological_mean(data, gap, climatology, variable)
        
        # Add temporal features
        ml_data['month'] = ml_data['timestamp'].dt.month
        ml_data['day_of_year'] = ml_data['timestamp'].dt.dayofyear
        
        feature_columns.extend(['month', 'day_of_year'])
        
        # Prepare training data
        X = ml_data[feature_columns].fillna(ml_data[feature_columns].mean())
        y = ml_data[variable]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_scaled, y)
        
        # Predict gap values
        filled_points = 0
        uncertainties = []
        predictions = []
        
        current_date = gap.start_date + timedelta(days=1)
        
        while current_date < gap.end_date:
            # Get feature values for this date
            # This requires having other variables available during the gap period
            # For simplicity, we'll use climatological values for missing predictors
            
            feature_values = []
            for feature in feature_columns:
                if feature == 'month':
                    feature_values.append(current_date.month)
                elif feature == 'day_of_year':
                    feature_values.append(current_date.timetuple().tm_yday)
                else:
                    # Use mean value for this feature (simplified)
                    feature_values.append(X[feature].mean())
            
            # Scale and predict
            feature_array = np.array([feature_values])
            feature_scaled = scaler.transform(feature_array)
            
            predicted_value = rf_model.predict(feature_scaled)[0]
            predictions.append(predicted_value)
            
            # Estimate uncertainty using prediction interval
            # Use out-of-bag predictions for uncertainty estimation
            tree_predictions = [tree.predict(feature_scaled)[0] for tree in rf_model.estimators_]
            prediction_std = np.std(tree_predictions)
            
            # Add to data
            location_data = data[
                (data['variable_name'] == variable) & 
                (data['timestamp'] == gap.start_date)
            ]
            
            if len(location_data) > 0:
                lat = location_data.iloc[0]['latitude']
                lon = location_data.iloc[0]['longitude']
                
                new_row = {
                    'timestamp': current_date,
                    'latitude': lat,
                    'longitude': lon,
                    'variable_name': variable,
                    'value': predicted_value,
                    'unit': location_data.iloc[0]['unit'],
                    'data_source': 'gap_filled_ml',
                    'dataset_id': f'gap_filled_{gap.gap_id}',
                    'quality_flag': 'gap_filled',
                    'collection_method': 'machine_learning',
                    'spatial_resolution': location_data.iloc[0]['spatial_resolution'],
                    'temporal_resolution': 'daily'
                }
                
                data.loc[len(data)] = new_row
                filled_points += 1
                uncertainties.append(prediction_std)
            
            current_date += timedelta(days=1)
        
        # Calculate model performance score (R²)
        validation_score = rf_model.score(X_scaled, y)
        
        uncertainty_stats = {
            'mean': np.mean(uncertainties) if uncertainties else 0,
            'std': np.std(uncertainties) if len(uncertainties) > 1 else 0
        }
        
        return filled_points, uncertainty_stats, validation_score
    
    def _generate_filling_summary(self, 
                                original_data: pd.DataFrame, 
                                filled_data: pd.DataFrame, 
                                filling_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of gap filling results"""
        
        summary = {
            "original_records": len(original_data),
            "filled_records": len(filled_data),
            "added_records": len(filled_data) - len(original_data),
            "variables_processed": len(filling_report["gaps_identified"]),
            "total_gaps_identified": sum(len(gaps) for gaps in filling_report["gaps_identified"].values()),
            "total_gaps_filled": sum(len(results) for results in filling_report["gaps_filled"].values()),
            "strategies_summary": {},
            "quality_summary": {}
        }
        
        # Strategy usage summary
        all_results = []
        for var_results in filling_report["gaps_filled"].values():
            all_results.extend(var_results)
        
        strategy_counts = {}
        for result in all_results:
            strategy = result['strategy_used']
            if strategy not in strategy_counts:
                strategy_counts[strategy] = {'count': 0, 'total_points': 0}
            strategy_counts[strategy]['count'] += 1
            strategy_counts[strategy]['total_points'] += result['filled_points']
        
        summary["strategies_summary"] = strategy_counts
        
        # Quality summary
        filled_records = filled_data[filled_data['collection_method'].str.contains('gap_fill', na=False)]
        if len(filled_records) > 0:
            summary["quality_summary"] = {
                "gap_filled_records": len(filled_records),
                "gap_filled_percentage": len(filled_records) / len(filled_data) * 100,
                "gap_filled_variables": filled_records['variable_name'].nunique()
            }
        
        return summary
    
    def _save_gap_filling_report(self, report: Dict[str, Any]) -> Path:
        """Save gap filling report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"gap_filling_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Gap filling report saved: {report_path}")
        
        return report_path
    
    def validate_gap_filling_quality(self, 
                                   original_data: pd.DataFrame, 
                                   filled_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate the quality of gap filling results"""
        logger.info("🔍 Validating gap filling quality")
        
        validation = {
            "data_continuity": {},
            "statistical_consistency": {},
            "temporal_patterns": {},
            "overall_assessment": {}
        }
        
        # Check data continuity
        for variable in original_data['variable_name'].unique():
            orig_var = original_data[original_data['variable_name'] == variable]
            filled_var = filled_data[filled_data['variable_name'] == variable]
            
            # Time series continuity
            filled_var_sorted = filled_var.sort_values('timestamp')
            time_diffs = filled_var_sorted['timestamp'].diff().dropna()
            
            validation["data_continuity"][variable] = {
                "original_records": len(orig_var),
                "filled_records": len(filled_var),
                "added_records": len(filled_var) - len(orig_var),
                "max_gap_days": time_diffs.max().total_seconds() / (24 * 3600) if len(time_diffs) > 0 else 0,
                "median_interval_days": time_diffs.dt.total_seconds().median() / (24 * 3600) if len(time_diffs) > 0 else 0
            }
            
            # Statistical consistency
            orig_stats = orig_var['value'].describe()
            filled_stats = filled_var['value'].describe()
            
            validation["statistical_consistency"][variable] = {
                "mean_change_percent": ((filled_stats['mean'] - orig_stats['mean']) / orig_stats['mean']) * 100,
                "std_change_percent": ((filled_stats['std'] - orig_stats['std']) / orig_stats['std']) * 100,
                "range_preserved": filled_stats['min'] >= orig_stats['min'] * 0.8 and filled_stats['max'] <= orig_stats['max'] * 1.2
            }
        
        # Overall assessment
        total_added = sum(v["added_records"] for v in validation["data_continuity"].values())
        total_original = sum(v["original_records"] for v in validation["data_continuity"].values())
        
        validation["overall_assessment"] = {
            "fill_percentage": (total_added / total_original) * 100 if total_original > 0 else 0,
            "continuity_improved": sum(1 for v in validation["data_continuity"].values() if v["max_gap_days"] < 7),
            "statistical_stability": sum(1 for v in validation["statistical_consistency"].values() if abs(v["mean_change_percent"]) < 10)
        }
        
        logger.info(f"✅ Gap filling validation completed")
        logger.info(f"   Added {total_added:,} records ({validation['overall_assessment']['fill_percentage']:.1f}% increase)")
        
        return validation