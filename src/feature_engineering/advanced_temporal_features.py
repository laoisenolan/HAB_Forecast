#!/usr/bin/env python3
"""
Advanced Temporal Feature Engineering for HAB Forecasting

This module implements sophisticated temporal feature extraction leveraging
32+ years of historical satellite data for HAB prediction.

🎯 TEMPORAL FEATURES IMPLEMENTED:
- Long-term trends and climate patterns (1950-2024)
- Multi-scale seasonality (daily, monthly, annual cycles)
- Temporal lag features for environmental drivers
- Anomaly detection from climatological baselines
- Time-to-event features for HAB occurrence
- Temporal autocorrelation features
- Seasonal decomposition and harmonic analysis
- Climate oscillation indices (ENSO, PDO proxies)

🛰️ LEVERAGES:
- 75+ years ERA5 atmospheric data (1950-2024)
- 32+ years HYCOM ocean physics (1992-2024) 
- 24+ years ocean color data (2000-2024)
- Historical HAB event patterns

Author: HAB Forecasting System
Date: August 6, 2025
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from scipy import stats, signal
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import warnings
# Optional dependencies - gracefully handle if not available
try:
    from statsmodels.tsa import seasonal
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.stattools import acf, pacf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

logger = logging.getLogger(__name__)

class AdvancedTemporalFeatureEngineer:
    """
    Advanced temporal feature engineering for HAB forecasting.
    
    Leverages extensive historical data (32+ years ocean, 75+ years atmosphere)
    to create sophisticated temporal features including:
    1. Multi-scale trends and patterns
    2. Seasonal and cyclic components
    3. Temporal lag relationships
    4. Anomaly detection from historical baselines
    5. Climate oscillation influences
    """
    
    def __init__(self, historical_baseline_years: int = 30):
        """Initialize temporal feature engineer"""
        
        self.baseline_years = historical_baseline_years
        
        # Temporal analysis parameters
        self.temporal_params = {
            'seasonal_periods': {
                'daily': 365.25,
                'semi_annual': 365.25/2, 
                'seasonal': 365.25/4,
                'monthly': 365.25/12,
                'lunar': 29.53  # Lunar cycle (tidal influence)
            },
            'lag_periods': [1, 3, 7, 14, 30, 60, 90, 180],  # days
            'trend_window_years': [1, 5, 10, 20],  # trend calculation windows
            'anomaly_threshold_std': 2.0,  # standard deviations for anomaly detection
            'climate_oscillation_periods': [2, 3, 5, 7, 11],  # years for climate oscillations
        }
        
        # HAB seasonal patterns (from literature and historical analysis)
        self.hab_seasonal_patterns = {
            'Karenia_mikimotoi': {'peak_months': [7, 8, 9], 'strength': 'high'},
            'Heterosigma_akashiwo': {'peak_months': [5, 6, 7], 'strength': 'medium'},
            'Noctiluca_scintillans': {'peak_months': [1, 2, 3, 11, 12], 'strength': 'low'},
            'Chattonella_antiqua': {'peak_months': [7, 8], 'strength': 'high'},
            'Chattonella_marina': {'peak_months': [6, 7, 8], 'strength': 'medium'},
            'Cochlodinium_polykrikoides': {'peak_months': [8, 9], 'strength': 'high'},
            'Alexandrium_tamarense': {'peak_months': [4, 5, 6], 'strength': 'medium'},
        }
        
        logger.info("⏰ Advanced Temporal Feature Engineer initialized")
        logger.info(f"📊 Historical baseline: {historical_baseline_years} years")
        logger.info(f"🔄 Seasonal periods: {len(self.temporal_params['seasonal_periods'])}")
        
    def calculate_long_term_trends(self,
                                 time_series: pd.Series,
                                 window_years: List[int] = None) -> Dict[str, pd.Series]:
        """
        Calculate multi-scale long-term trends
        
        Args:
            time_series: Time series data with datetime index
            window_years: List of window sizes in years for trend calculation
            
        Returns:
            Dictionary of trend features
        """
        if window_years is None:
            window_years = self.temporal_params['trend_window_years']
            
        trend_features = {}
        
        for window_years_val in window_years:
            window_days = int(window_years_val * 365.25)
            
            if len(time_series) < window_days:
                logger.warning(f"Time series too short for {window_years_val}-year trend")
                continue
                
            # Rolling linear regression for trend
            trend_values = []
            slope_values = []
            r_squared_values = []
            
            for i in range(window_days, len(time_series)):
                # Get window data
                window_data = time_series.iloc[i-window_days:i]
                valid_data = window_data.dropna()
                
                if len(valid_data) < window_days * 0.5:  # Need at least 50% valid data
                    trend_values.append(np.nan)
                    slope_values.append(np.nan)
                    r_squared_values.append(np.nan)
                    continue
                    
                # Fit linear trend
                x = np.arange(len(valid_data))
                y = valid_data.values
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                # Current trend value (extrapolated)
                trend_val = intercept + slope * (len(valid_data) - 1)
                
                trend_values.append(trend_val)
                slope_values.append(slope)
                r_squared_values.append(r_value**2)
                
            # Pad with NaN for the first window_days values
            trend_values = [np.nan] * window_days + trend_values
            slope_values = [np.nan] * window_days + slope_values
            r_squared_values = [np.nan] * window_days + r_squared_values
            
            var_name = time_series.name or 'variable'
            trend_features[f'{var_name}_trend_{window_years_val}y'] = pd.Series(trend_values, index=time_series.index)
            trend_features[f'{var_name}_trend_slope_{window_years_val}y'] = pd.Series(slope_values, index=time_series.index)
            trend_features[f'{var_name}_trend_r2_{window_years_val}y'] = pd.Series(r_squared_values, index=time_series.index)
            
        return trend_features
    
    def calculate_seasonal_features(self,
                                  time_series: pd.Series,
                                  include_harmonics: bool = True) -> Dict[str, pd.Series]:
        """
        Calculate comprehensive seasonal features
        
        Args:
            time_series: Time series data with datetime index
            include_harmonics: Whether to include harmonic analysis
            
        Returns:
            Dictionary of seasonal features
        """
        seasonal_features = {}
        var_name = time_series.name or 'variable'
        
        # Basic calendar features
        dates = pd.to_datetime(time_series.index)
        
        seasonal_features[f'{var_name}_month'] = dates.month
        seasonal_features[f'{var_name}_day_of_year'] = dates.day_of_year
        seasonal_features[f'{var_name}_season'] = dates.month.map(
            {1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 12:0}  # 0=winter, 1=spring, 2=summer, 3=autumn
        )
        seasonal_features[f'{var_name}_is_summer'] = (dates.month.isin([6, 7, 8])).astype(int)
        seasonal_features[f'{var_name}_is_winter'] = (dates.month.isin([12, 1, 2])).astype(int)
        
        # Harmonic seasonal components
        if include_harmonics:
            day_of_year = dates.day_of_year
            
            for period_name, period_days in self.temporal_params['seasonal_periods'].items():
                # Calculate harmonic components
                phase = 2 * np.pi * day_of_year / period_days
                
                seasonal_features[f'{var_name}_cos_{period_name}'] = np.cos(phase)
                seasonal_features[f'{var_name}_sin_{period_name}'] = np.sin(phase)
                
        # Seasonal decomposition (if enough data and statsmodels available)
        if HAS_STATSMODELS and len(time_series) >= 2 * 365:  # At least 2 years of data
            try:
                # Resample to daily if higher frequency
                if len(time_series) > 3 * 365:  # Downsample if too high frequency
                    daily_series = time_series.resample('D').mean()
                else:
                    daily_series = time_series
                    
                # Remove NaN values for decomposition
                clean_series = daily_series.dropna()
                
                if len(clean_series) >= 2 * 365:
                    decomposition = seasonal.seasonal_decompose(
                        clean_series, 
                        model='additive', 
                        period=365,
                        extrapolate_trend='freq'
                    )
                    
                    # Interpolate back to original index
                    seasonal_component = decomposition.seasonal.reindex(time_series.index).interpolate()
                    trend_component = decomposition.trend.reindex(time_series.index).interpolate()
                    residual_component = decomposition.resid.reindex(time_series.index).interpolate()
                    
                    seasonal_features[f'{var_name}_seasonal_component'] = seasonal_component
                    seasonal_features[f'{var_name}_trend_component'] = trend_component
                    seasonal_features[f'{var_name}_residual_component'] = residual_component
                    seasonal_features[f'{var_name}_seasonal_strength'] = np.abs(seasonal_component)
                    
            except Exception as e:
                logger.warning(f"Seasonal decomposition failed for {var_name}: {e}")
        elif not HAS_STATSMODELS:
            logger.info("Statsmodels not available, skipping seasonal decomposition")
                
        # Convert all to pandas Series with proper index
        for key, values in seasonal_features.items():
            if not isinstance(values, pd.Series):
                seasonal_features[key] = pd.Series(values, index=time_series.index)
                
        return seasonal_features
    
    def calculate_lag_features(self,
                             time_series: pd.Series,
                             lag_periods: List[int] = None,
                             include_rolling_stats: bool = True) -> Dict[str, pd.Series]:
        """
        Calculate temporal lag features and rolling statistics
        
        Args:
            time_series: Time series data
            lag_periods: List of lag periods in days
            include_rolling_stats: Whether to include rolling statistics
            
        Returns:
            Dictionary of lag features
        """
        if lag_periods is None:
            lag_periods = self.temporal_params['lag_periods']
            
        lag_features = {}
        var_name = time_series.name or 'variable'
        
        # Basic lag features
        for lag in lag_periods:
            if lag < len(time_series):
                lag_series = time_series.shift(lag)
                lag_features[f'{var_name}_lag_{lag}d'] = lag_series
                
                # Lag differences
                diff_series = time_series - lag_series
                lag_features[f'{var_name}_diff_lag_{lag}d'] = diff_series
                
                # Lag ratios (avoid division by zero)
                ratio_series = time_series / (lag_series.replace(0, np.nan))
                lag_features[f'{var_name}_ratio_lag_{lag}d'] = ratio_series
        
        # Rolling statistics
        if include_rolling_stats:
            for window in [7, 14, 30, 60, 90]:  # days
                if window < len(time_series):
                    rolling_mean = time_series.rolling(window=window).mean()
                    rolling_std = time_series.rolling(window=window).std()
                    rolling_min = time_series.rolling(window=window).min()
                    rolling_max = time_series.rolling(window=window).max()
                    
                    lag_features[f'{var_name}_rolling_mean_{window}d'] = rolling_mean
                    lag_features[f'{var_name}_rolling_std_{window}d'] = rolling_std
                    lag_features[f'{var_name}_rolling_range_{window}d'] = rolling_max - rolling_min
                    
                    # Current value relative to rolling statistics
                    lag_features[f'{var_name}_zscore_{window}d'] = (time_series - rolling_mean) / rolling_std
                    lag_features[f'{var_name}_percentile_{window}d'] = (
                        time_series.rolling(window=window).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]))
                    ) / 100.0
                    
        return lag_features
    
    def calculate_anomaly_features(self,
                                 time_series: pd.Series,
                                 baseline_period: Optional[Tuple[str, str]] = None) -> Dict[str, pd.Series]:
        """
        Calculate anomaly features based on historical climatology
        
        Args:
            time_series: Time series data
            baseline_period: Tuple of (start_date, end_date) for baseline period
            
        Returns:
            Dictionary of anomaly features
        """
        var_name = time_series.name or 'variable'
        anomaly_features = {}
        
        # If no baseline period specified, use the first N years of data
        if baseline_period is None:
            start_date = time_series.index.min()
            end_date = start_date + pd.DateOffset(years=self.baseline_years)
            if end_date > time_series.index.max():
                end_date = time_series.index.max()
        else:
            start_date, end_date = baseline_period
            
        # Get baseline data
        baseline_data = time_series.loc[start_date:end_date]
        
        if len(baseline_data) < 365:  # Need at least 1 year of baseline
            logger.warning(f"Insufficient baseline data for {var_name} anomaly calculation")
            return anomaly_features
            
        # Calculate climatological mean and standard deviation by day of year
        baseline_df = pd.DataFrame({'values': baseline_data, 'day_of_year': baseline_data.index.day_of_year})
        climatology = baseline_df.groupby('day_of_year')['values'].agg(['mean', 'std']).reset_index()
        
        # Smooth climatology to avoid sharp transitions
        climatology['mean_smooth'] = signal.savgol_filter(climatology['mean'], 15, 3, mode='wrap')
        climatology['std_smooth'] = signal.savgol_filter(climatology['std'], 15, 3, mode='wrap')
        
        # Map climatology to full time series
        full_climatology = pd.DataFrame({
            'day_of_year': time_series.index.day_of_year,
        })
        full_climatology = full_climatology.merge(climatology, on='day_of_year', how='left')
        full_climatology.index = time_series.index
        
        # Calculate anomalies
        anomalies = time_series - full_climatology['mean_smooth']
        std_anomalies = anomalies / full_climatology['std_smooth']
        
        anomaly_features[f'{var_name}_anomaly'] = anomalies
        anomaly_features[f'{var_name}_standardized_anomaly'] = std_anomalies
        anomaly_features[f'{var_name}_is_extreme_anomaly'] = (
            np.abs(std_anomalies) > self.temporal_params['anomaly_threshold_std']
        ).astype(int)
        anomaly_features[f'{var_name}_positive_anomaly'] = (std_anomalies > 0).astype(int)
        anomaly_features[f'{var_name}_climatology_mean'] = full_climatology['mean_smooth']
        anomaly_features[f'{var_name}_climatology_std'] = full_climatology['std_smooth']
        
        # Percentile ranking based on historical distribution
        percentiles = []
        for i, value in enumerate(time_series):
            if pd.isna(value):
                percentiles.append(np.nan)
                continue
                
            day_of_year = time_series.index[i].day_of_year
            baseline_values = baseline_df[baseline_df['day_of_year'] == day_of_year]['values']
            
            if len(baseline_values) > 0:
                percentile = stats.percentileofscore(baseline_values, value)
            else:
                percentile = 50.0  # Default to median
                
            percentiles.append(percentile)
            
        anomaly_features[f'{var_name}_historical_percentile'] = pd.Series(percentiles, index=time_series.index)
        
        return anomaly_features
    
    def calculate_autocorrelation_features(self,
                                         time_series: pd.Series,
                                         max_lags: int = 30) -> Dict[str, float]:
        """
        Calculate temporal autocorrelation features
        
        Args:
            time_series: Time series data
            max_lags: Maximum number of lags for autocorrelation
            
        Returns:
            Dictionary of autocorrelation features (scalar values)
        """
        var_name = time_series.name or 'variable'
        autocorr_features = {}
        
        # Remove NaN values
        clean_series = time_series.dropna()
        
        if len(clean_series) < max_lags + 10:  # Need sufficient data
            return autocorr_features
        
        if not HAS_STATSMODELS:
            logger.info("Statsmodels not available, skipping autocorrelation analysis")
            return autocorr_features
            
        try:
            # Calculate autocorrelation function
            autocorr_values = acf(clean_series, nlags=max_lags, fft=True)
            
            # Key autocorrelation features
            autocorr_features[f'{var_name}_autocorr_lag1'] = autocorr_values[1]
            autocorr_features[f'{var_name}_autocorr_lag7'] = autocorr_values[min(7, max_lags)]
            autocorr_features[f'{var_name}_autocorr_lag30'] = autocorr_values[min(30, max_lags)]
            
            # First zero crossing (indicates memory timescale)
            zero_crossing = np.where(autocorr_values[1:] <= 0)[0]
            if len(zero_crossing) > 0:
                autocorr_features[f'{var_name}_autocorr_zero_crossing'] = zero_crossing[0] + 1
            else:
                autocorr_features[f'{var_name}_autocorr_zero_crossing'] = max_lags
                
            # Integral timescale (e-folding time)
            e_folding_idx = np.where(autocorr_values[1:] <= 1/np.e)[0]
            if len(e_folding_idx) > 0:
                autocorr_features[f'{var_name}_e_folding_time'] = e_folding_idx[0] + 1
            else:
                autocorr_features[f'{var_name}_e_folding_time'] = max_lags
                
        except Exception as e:
            logger.warning(f"Autocorrelation calculation failed for {var_name}: {e}")
            
        return autocorr_features
    
    def calculate_hab_seasonal_timing_features(self,
                                             dates: pd.DatetimeIndex,
                                             species: str) -> Dict[str, pd.Series]:
        """
        Calculate HAB season timing features for specific species
        
        Args:
            dates: DateTime index
            species: HAB species name
            
        Returns:
            Dictionary of seasonal timing features
        """
        timing_features = {}
        
        if species not in self.hab_seasonal_patterns:
            logger.warning(f"No seasonal pattern data for species {species}")
            return timing_features
            
        pattern = self.hab_seasonal_patterns[species]
        peak_months = pattern['peak_months']
        strength = pattern['strength']
        
        # Calculate seasonal preference score
        seasonal_scores = []
        for date in dates:
            month = date.month
            
            if month in peak_months:
                if strength == 'high':
                    score = 1.0
                elif strength == 'medium':
                    score = 0.7
                else:  # low
                    score = 0.4
            else:
                # Gradual decrease for adjacent months
                min_distance = min([abs(month - pm) if abs(month - pm) <= 6 
                                  else 12 - abs(month - pm) for pm in peak_months])
                if min_distance == 1:
                    score = 0.3
                elif min_distance == 2:
                    score = 0.1
                else:
                    score = 0.0
                    
            seasonal_scores.append(score)
            
        timing_features[f'{species}_seasonal_suitability'] = pd.Series(seasonal_scores, index=dates)
        
        # Distance to nearest peak season
        peak_distances = []
        for date in dates:
            month = date.month
            distances = [min(abs(month - pm), 12 - abs(month - pm)) for pm in peak_months]
            peak_distances.append(min(distances))
            
        timing_features[f'{species}_months_to_peak_season'] = pd.Series(peak_distances, index=dates)
        
        # Binary indicators
        timing_features[f'{species}_is_peak_season'] = pd.Series(
            [date.month in peak_months for date in dates], index=dates
        ).astype(int)
        
        return timing_features
    
    def calculate_climate_oscillation_features(self,
                                             time_series: pd.Series,
                                             oscillation_periods: List[int] = None) -> Dict[str, pd.Series]:
        """
        Calculate features related to climate oscillations (ENSO, PDO proxies)
        
        Args:
            time_series: Time series data (should be long-term)
            oscillation_periods: List of oscillation periods in years
            
        Returns:
            Dictionary of climate oscillation features
        """
        if oscillation_periods is None:
            oscillation_periods = self.temporal_params['climate_oscillation_periods']
            
        var_name = time_series.name or 'variable'
        oscillation_features = {}
        
        # Need at least 5 years of data for meaningful climate analysis
        if len(time_series) < 5 * 365:
            logger.warning(f"Time series too short for climate oscillation analysis: {len(time_series)} days")
            return oscillation_features
            
        # Resample to monthly means for climate analysis
        monthly_series = time_series.resample('M').mean().dropna()
        
        if len(monthly_series) < 60:  # Need at least 5 years of monthly data
            return oscillation_features
            
        for period_years in oscillation_periods:
            period_months = period_years * 12
            
            if len(monthly_series) < 2 * period_months:  # Need at least 2 full cycles
                continue
                
            # Calculate moving correlation with sine wave of the given period
            phase = 2 * np.pi * np.arange(len(monthly_series)) / period_months
            sine_wave = np.sin(phase)
            cosine_wave = np.cos(phase)
            
            # Rolling correlation over 2-year windows
            window_months = 24
            correlations_sin = []
            correlations_cos = []
            
            for i in range(window_months, len(monthly_series)):
                window_data = monthly_series.iloc[i-window_months:i]
                window_sin = sine_wave[i-window_months:i]
                window_cos = cosine_wave[i-window_months:i]
                
                # Calculate correlations
                corr_sin = np.corrcoef(window_data, window_sin)[0, 1]
                corr_cos = np.corrcoef(window_data, window_cos)[0, 1]
                
                correlations_sin.append(corr_sin if not np.isnan(corr_sin) else 0)
                correlations_cos.append(corr_cos if not np.isnan(corr_cos) else 0)
                
            # Pad with zeros for initial period
            correlations_sin = [0] * window_months + correlations_sin
            correlations_cos = [0] * window_months + correlations_cos
            
            # Interpolate back to original frequency
            monthly_index = monthly_series.index
            corr_sin_series = pd.Series(correlations_sin, index=monthly_index)
            corr_cos_series = pd.Series(correlations_cos, index=monthly_index)
            
            # Resample to original frequency
            corr_sin_daily = corr_sin_series.resample('D').interpolate().reindex(time_series.index)
            corr_cos_daily = corr_cos_series.resample('D').interpolate().reindex(time_series.index)
            
            oscillation_features[f'{var_name}_oscillation_{period_years}y_sin'] = corr_sin_daily
            oscillation_features[f'{var_name}_oscillation_{period_years}y_cos'] = corr_cos_daily
            
            # Oscillation strength (magnitude of correlation)
            oscillation_strength = np.sqrt(corr_sin_daily**2 + corr_cos_daily**2)
            oscillation_features[f'{var_name}_oscillation_{period_years}y_strength'] = oscillation_strength
            
        return oscillation_features
    
    def create_comprehensive_temporal_features(self,
                                             time_series_dict: Dict[str, pd.Series],
                                             target_species: Optional[str] = None) -> pd.DataFrame:
        """
        Create comprehensive temporal features for all environmental variables
        
        Args:
            time_series_dict: Dictionary of time series data
            target_species: Species name for species-specific timing features
            
        Returns:
            DataFrame with all temporal features
        """
        logger.info(f"⏰ Creating comprehensive temporal features for {len(time_series_dict)} variables")
        
        all_features = {}
        
        # Get common index
        common_index = next(iter(time_series_dict.values())).index
        
        # Process each time series
        for var_name, time_series in time_series_dict.items():
            if time_series is None or len(time_series.dropna()) < 10:
                logger.warning(f"Skipping {var_name} - insufficient data")
                continue
                
            logger.info(f"Processing temporal features for {var_name}")
            
            # Ensure the series has a name
            if time_series.name is None:
                time_series.name = var_name
                
            # Long-term trends
            trend_features = self.calculate_long_term_trends(time_series)
            all_features.update(trend_features)
            
            # Seasonal features
            seasonal_features = self.calculate_seasonal_features(time_series)
            all_features.update(seasonal_features)
            
            # Lag features
            lag_features = self.calculate_lag_features(time_series)
            all_features.update(lag_features)
            
            # Anomaly features
            anomaly_features = self.calculate_anomaly_features(time_series)
            all_features.update(anomaly_features)
            
            # Autocorrelation features (scalar values - add to each row)
            autocorr_features = self.calculate_autocorrelation_features(time_series)
            for feature_name, feature_value in autocorr_features.items():
                all_features[feature_name] = pd.Series(feature_value, index=common_index)
                
            # Climate oscillation features (for longer time series)
            if len(time_series) >= 5 * 365:  # At least 5 years
                climate_features = self.calculate_climate_oscillation_features(time_series)
                all_features.update(climate_features)
                
        # Species-specific seasonal timing features
        if target_species and isinstance(common_index, pd.DatetimeIndex):
            timing_features = self.calculate_hab_seasonal_timing_features(common_index, target_species)
            all_features.update(timing_features)
            
        # Convert to DataFrame
        feature_df = pd.DataFrame(all_features, index=common_index)
        
        logger.info(f"✅ Created {len(feature_df.columns)} temporal features")
        return feature_df