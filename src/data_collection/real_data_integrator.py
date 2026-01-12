"""
Real Data Integrator for HAB Forecasting System

This module integrates real environmental data from:
- Copernicus Marine Service (ocean physics and color)
- Google Earth Engine (climatology and historical data)
- OpenWeather API (atmospheric conditions)
"""

import os
import logging
import requests
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Import our existing modules
from src.data_collection.copernicus_downloader import CopernicusDownloader
from src.data_collection.erddap_downloader import ERDDAPDownloader
from src.data_collection.gee_integrator import GEEIntegrator
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class RealDataIntegrator:
    """
    Integrates real environmental data from multiple sources for HAB forecasting.
    """
    
    def __init__(self):
        """Initialize the real data integrator."""
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_data_sources_config()
        
        # Initialize data sources
        self.copernicus_downloader = CopernicusDownloader()
        self.erddap_downloader = ERDDAPDownloader()
        self.gee_integrator = GEEIntegrator()
        
        # OpenWeather API configuration
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
        self.openweather_base_url = "http://api.openweathermap.org/data/2.5"
        
        # Seto Inland Sea coordinates
        self.region_bounds = {
            'lat_min': 34.0, 'lat_max': 35.0,
            'lon_min': 132.0, 'lon_max': 135.0
        }
        
        logger.info("Real Data Integrator initialized")
    
    def get_copernicus_marine_data(self, start_date: str, end_date: str) -> Dict[str, xr.Dataset]:
        """
        Get real ocean data from Copernicus Marine Service.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary containing ocean datasets
        """
        logger.info(f"Collecting Copernicus Marine data from {start_date} to {end_date}")
        
        ocean_data = {}
        
        try:
            # Get available datasets
            available_datasets = self.copernicus_downloader.get_available_datasets()
            logger.info(f"Available Copernicus datasets: {list(available_datasets.keys())}")
            
            # Try to download physics data (SST, currents, etc.)
            try:
                physics_data = self.copernicus_downloader.download_physics_data(
                    start_date=start_date,
                    end_date=end_date
                )
                if physics_data is not None:
                    ocean_data['physics'] = physics_data
                    logger.info("✅ Copernicus physics data collected")
            except Exception as e:
                logger.warning(f"Failed to get physics data: {e}")
            
            # Try to download ocean color data (chlorophyll, etc.)
            try:
                ocean_color_data = self.copernicus_downloader.download_ocean_color_data(
                    start_date=start_date,
                    end_date=end_date
                )
                if ocean_color_data is not None:
                    ocean_data['ocean_color'] = ocean_color_data
                    logger.info("✅ Copernicus ocean color data collected")
            except Exception as e:
                logger.warning(f"Failed to get ocean color data: {e}")
            
            return ocean_data
            
        except Exception as e:
            logger.error(f"Failed to collect Copernicus Marine data: {e}")
            return {}
    
    def get_openweather_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Get real atmospheric data from OpenWeather API.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary containing atmospheric datasets
        """
        logger.info(f"Collecting OpenWeather data from {start_date} to {end_date}")
        
        if not self.openweather_api_key:
            logger.warning("OpenWeather API key not found. Using synthetic data.")
            return self._generate_synthetic_atmospheric_data(start_date, end_date)
        
        atmospheric_data = {}
        
        try:
            # Get current weather for Seto Inland Sea region
            # Using Hiroshima as a representative location
            lat, lon = 34.3853, 132.4553  # Hiroshima coordinates
            
            # Get current weather
            current_url = f"{self.openweather_base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.openweather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(current_url, params=params, timeout=10)
            if response.status_code == 200:
                current_data = response.json()
                
                # Extract relevant atmospheric data
                atmospheric_data['current'] = {
                    'temperature': current_data['main']['temp'],
                    'humidity': current_data['main']['humidity'],
                    'pressure': current_data['main']['pressure'],
                    'wind_speed': current_data['wind']['speed'],
                    'wind_direction': current_data['wind'].get('deg', 0),
                    'precipitation': current_data.get('rain', {}).get('1h', 0),
                    'visibility': current_data.get('visibility', 10000),
                    'clouds': current_data['clouds']['all']
                }
                
                logger.info("✅ OpenWeather current data collected")
                
                # Get 5-day forecast
                forecast_url = f"{self.openweather_base_url}/forecast"
                forecast_response = requests.get(forecast_url, params=params, timeout=10)
                
                if forecast_response.status_code == 200:
                    forecast_data = forecast_response.json()
                    
                    # Process forecast data
                    forecast_list = []
                    for item in forecast_data['list']:
                        forecast_list.append({
                            'datetime': item['dt_txt'],
                            'temperature': item['main']['temp'],
                            'humidity': item['main']['humidity'],
                            'pressure': item['main']['pressure'],
                            'wind_speed': item['wind']['speed'],
                            'wind_direction': item['wind'].get('deg', 0),
                            'precipitation': item.get('rain', {}).get('3h', 0),
                            'clouds': item['clouds']['all']
                        })
                    
                    atmospheric_data['forecast'] = pd.DataFrame(forecast_list)
                    logger.info("✅ OpenWeather forecast data collected")
                
            else:
                logger.warning(f"OpenWeather API error: {response.status_code}")
                atmospheric_data = self._generate_synthetic_atmospheric_data(start_date, end_date)
                
        except Exception as e:
            logger.error(f"Failed to collect OpenWeather data: {e}")
            atmospheric_data = self._generate_synthetic_atmospheric_data(start_date, end_date)
        
        return atmospheric_data
    
    def get_erddap_data(self, start_date: str, end_date: str) -> Dict[str, xr.Dataset]:
        """
        Get real oceanographic data from ERDDAP servers.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary containing ERDDAP datasets
        """
        logger.info(f"Collecting ERDDAP data from {start_date} to {end_date}")
        
        erddap_data = {}
        
        try:
            # Get GFS atmospheric data
            gfs_data = self.erddap_downloader.download_gfs_data(
                start_date=start_date,
                end_date=end_date,
                variables=['ugrd10m', 'vgrd10m', 'pratesfc']
            )
            
            if gfs_data is not None:
                erddap_data['gfs'] = gfs_data
                logger.info("✅ ERDDAP GFS data collected")
            
            # Get NOAA CoastWatch data as fallback
            noaa_data = self.erddap_downloader.download_noaa_data(
                start_date=start_date,
                end_date=end_date
            )
            
            if noaa_data is not None:
                erddap_data['noaa'] = noaa_data
                logger.info("✅ ERDDAP NOAA data collected")
            
            return erddap_data
            
        except Exception as e:
            logger.error(f"Failed to collect ERDDAP data: {e}")
            return {}
    
    def get_gee_data(self, start_date: str, end_date: str) -> Dict[str, any]:
        """
        Get real environmental data from Google Earth Engine.
        
        Args:
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
            
        Returns:
            Dictionary containing GEE datasets
        """
        logger.info(f"Collecting Google Earth Engine data from {start_date} to {end_date}")
        
        try:
            # Get all environmental data from GEE
            gee_data = self.gee_integrator.get_all_environmental_data(start_date, end_date)
            
            if gee_data and len(gee_data) > 1:  # More than just metadata
                logger.info(f"✅ Collected {len(gee_data) - 1} environmental datasets from GEE")
                return gee_data
            else:
                logger.warning("No GEE data available")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to collect GEE data: {e}")
            return {}
    
    def _generate_synthetic_atmospheric_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic atmospheric data when real data is unavailable.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary containing synthetic atmospheric data
        """
        logger.info("Generating synthetic atmospheric data...")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Generate realistic atmospheric data
        atmospheric_data = {
            'current': {
                'temperature': 25.0 + np.random.normal(0, 3),
                'humidity': 70 + np.random.normal(0, 10),
                'pressure': 1013 + np.random.normal(0, 5),
                'wind_speed': 5.0 + np.random.exponential(2),
                'wind_direction': np.random.uniform(0, 360),
                'precipitation': np.random.exponential(1),
                'visibility': 10000 + np.random.normal(0, 2000),
                'clouds': np.random.uniform(0, 100)
            }
        }
        
        # Generate forecast data
        forecast_data = []
        for date in dates:
            forecast_data.append({
                'datetime': date.strftime('%Y-%m-%d %H:%M:%S'),
                'temperature': 25.0 + 5 * np.sin(2 * np.pi * date.hour / 24) + np.random.normal(0, 2),
                'humidity': 70 + 10 * np.sin(2 * np.pi * date.hour / 24) + np.random.normal(0, 5),
                'pressure': 1013 + np.random.normal(0, 3),
                'wind_speed': 5.0 + np.random.exponential(1.5),
                'wind_direction': np.random.uniform(0, 360),
                'precipitation': np.random.exponential(0.5),
                'clouds': np.random.uniform(0, 100)
            })
        
        atmospheric_data['forecast'] = pd.DataFrame(forecast_data)
        
        return atmospheric_data
    
    def integrate_all_data_sources(self, start_date: str, end_date: str) -> Dict[str, any]:
        """
        Integrate data from all sources into a unified dataset.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary containing integrated environmental data
        """
        logger.info(f"Integrating all data sources from {start_date} to {end_date}")
        
        integrated_data = {
            'metadata': {
                'start_date': start_date,
                'end_date': end_date,
                'region_bounds': self.region_bounds,
                'data_sources': [],
                'collection_timestamp': datetime.now().isoformat()
            }
        }
        
        # 1. Collect Copernicus Marine data
        copernicus_data = self.get_copernicus_marine_data(start_date, end_date)
        if copernicus_data:
            integrated_data['copernicus'] = copernicus_data
            integrated_data['metadata']['data_sources'].append('copernicus_marine')
            logger.info(f"✅ Copernicus data: {list(copernicus_data.keys())}")
        
        # 2. Collect OpenWeather data
        openweather_data = self.get_openweather_data(start_date, end_date)
        if openweather_data:
            integrated_data['openweather'] = openweather_data
            integrated_data['metadata']['data_sources'].append('openweather')
            logger.info(f"✅ OpenWeather data: {list(openweather_data.keys())}")
        
        # 3. Collect ERDDAP data
        erddap_data = self.get_erddap_data(start_date, end_date)
        if erddap_data:
            integrated_data['erddap'] = erddap_data
            integrated_data['metadata']['data_sources'].append('erddap')
            logger.info(f"✅ ERDDAP data: {list(erddap_data.keys())}")
        
        # 4. Collect Google Earth Engine data
        gee_data = self.get_gee_data(start_date, end_date)
        if gee_data:
            integrated_data['gee'] = gee_data
            integrated_data['metadata']['data_sources'].append('google_earth_engine')
            logger.info(f"✅ Google Earth Engine data: {list(gee_data.keys())}")
        
        # 5. Create unified environmental features
        environmental_features = self._create_unified_features(
            copernicus_data, openweather_data, erddap_data, gee_data, start_date, end_date
        )
        
        if environmental_features is not None:
            integrated_data['environmental_features'] = environmental_features
            logger.info(f"✅ Created unified features with {len(environmental_features)} samples")
        
        logger.info(f"Data integration complete. Sources: {integrated_data['metadata']['data_sources']}")
        return integrated_data
    
    def _create_unified_features(self, copernicus_data: Dict, openweather_data: Dict, 
                               erddap_data: Dict, gee_data: Dict, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Create unified environmental features from all data sources.
        
        Args:
            copernicus_data: Copernicus Marine data
            openweather_data: OpenWeather data
            erddap_data: ERDDAP data
            gee_data: Google Earth Engine data
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with unified environmental features
        """
        try:
            # Create date range
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            features_list = []
            
            for date in dates:
                features = {
                    'date': date,
                    'month': date.month,
                    'day_of_year': date.dayofyear,
                    'seasonal_factor': np.sin(2 * np.pi * date.dayofyear / 365.25)
                }
                
                # Extract ocean data (from Copernicus)
                if 'physics' in copernicus_data:
                    try:
                        sst_value = copernicus_data['physics'].sel(time=date, method='nearest')['thetao'].mean().values
                        features['sst'] = float(sst_value) if not np.isnan(sst_value) else 20.0
                    except:
                        features['sst'] = 20.0
                else:
                    features['sst'] = 20.0
                
                if 'ocean_color' in copernicus_data:
                    try:
                        chl_value = copernicus_data['ocean_color'].sel(time=date, method='nearest')['chl'].mean().values
                        features['chlorophyll'] = float(chl_value) if not np.isnan(chl_value) else 2.0
                    except:
                        features['chlorophyll'] = 2.0
                else:
                    features['chlorophyll'] = 2.0
                
                # Extract atmospheric data (from OpenWeather)
                if 'current' in openweather_data:
                    current = openweather_data['current']
                    features['wind_speed'] = current.get('wind_speed', 5.0)
                    features['precipitation'] = current.get('precipitation', 2.0)
                    features['temperature_air'] = current.get('temperature', 25.0)
                    features['humidity'] = current.get('humidity', 70.0)
                else:
                    features['wind_speed'] = 5.0
                    features['precipitation'] = 2.0
                    features['temperature_air'] = 25.0
                    features['humidity'] = 70.0
                
                # Extract additional ocean data (from ERDDAP)
                if 'gfs' in erddap_data:
                    try:
                        wind_u = erddap_data['gfs'].sel(time=date, method='nearest')['ugrd10m'].mean().values
                        wind_v = erddap_data['gfs'].sel(time=date, method='nearest')['vgrd10m'].mean().values
                        wind_magnitude = np.sqrt(wind_u**2 + wind_v**2)
                        features['wind_speed_erddap'] = float(wind_magnitude) if not np.isnan(wind_magnitude) else 5.0
                    except:
                        features['wind_speed_erddap'] = 5.0
                else:
                    features['wind_speed_erddap'] = 5.0
                
                # Extract GEE data
                if gee_data:
                    if 'sst' in gee_data:
                        features['sst_gee'] = gee_data['sst'].get('sst_mean', 20.0)
                    if 'chlorophyll' in gee_data:
                        features['chlorophyll_gee'] = gee_data['chlorophyll'].get('chlorophyll_mean', 2.0)
                    if 'wind' in gee_data:
                        features['wind_speed_gee'] = gee_data['wind'].get('wind_speed_mean', 5.0)
                    if 'precipitation' in gee_data:
                        features['precipitation_gee'] = gee_data['precipitation'].get('precipitation_mean', 2.0)
                
                # Add nitrate (synthetic if not available)
                features['nitrate'] = 5.0 - 0.5 * features['chlorophyll'] + np.random.normal(0, 0.5)
                features['nitrate'] = max(features['nitrate'], 0.1)
                
                features_list.append(features)
            
            features_df = pd.DataFrame(features_list)
            
            # Fill missing values
            for col in features_df.columns:
                if col != 'date':
                    features_df[col] = features_df[col].fillna(features_df[col].mean())
            
            return features_df
            
        except Exception as e:
            logger.error(f"Failed to create unified features: {e}")
            return None
    
    def save_integrated_data(self, integrated_data: Dict, output_path: str):
        """
        Save integrated data to file.
        
        Args:
            integrated_data: Integrated data dictionary
            output_path: Path to save the data
        """
        try:
            # Convert xarray datasets to dictionaries for JSON serialization
            json_data = {}
            for key, value in integrated_data.items():
                if key == 'environmental_features':
                    json_data[key] = value.to_dict()
                elif isinstance(value, dict):
                    json_data[key] = value
                else:
                    json_data[key] = str(value)
            
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            logger.info(f"✅ Integrated data saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save integrated data: {e}")
            raise 