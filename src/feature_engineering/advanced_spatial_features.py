#!/usr/bin/env python3
"""
Advanced Spatial Feature Engineering for HAB Forecasting

This module implements sophisticated spatial feature extraction for HAB prediction,
including gradients, fronts, upwelling indices, and oceanographic features.

🎯 SPATIAL FEATURES IMPLEMENTED:
- Temperature and salinity gradients
- Oceanographic fronts detection
- Upwelling and downwelling indices
- Stratification and mixing indices  
- Coastal proximity features
- Bathymetric influence features
- Current shear and vorticity
- Land-sea interaction features

🛰️ LEVERAGES:
- Multi-resolution satellite data (10m to 9km)
- HYCOM ocean physics (32+ years, 1992-2024)
- Sentinel-3 high-resolution ocean color (300m)
- Coastal bathymetry and topography

Author: HAB Forecasting System
Date: August 6, 2025
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import xarray as xr
from scipy import ndimage, spatial
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler
# from skimage import feature, filters, measure  # Optional dependency
# from geopy.distance import geodesic  # Optional dependency
import warnings

logger = logging.getLogger(__name__)

class AdvancedSpatialFeatureEngineer:
    """
    Advanced spatial feature engineering for HAB forecasting.
    
    Implements sophisticated spatial analysis techniques including:
    1. Multi-scale gradient calculations
    2. Oceanographic front detection
    3. Upwelling/downwelling analysis
    4. Stratification and mixing parameters
    5. Coastal and bathymetric influences
    """
    
    def __init__(self, region_bounds: Dict[str, float] = None):
        """Initialize spatial feature engineer"""
        
        # Default to Seto Inland Sea region
        if region_bounds is None:
            region_bounds = {
                'lat_min': 33.5, 'lat_max': 35.0,
                'lon_min': 132.0, 'lon_max': 135.5
            }
            
        self.region_bounds = region_bounds
        
        # Spatial analysis parameters
        self.spatial_params = {
            'gradient_scales': [1, 3, 5, 10],  # km scales for gradient calculation
            'front_detection_threshold': 0.5,  # Standard deviations for front detection
            'upwelling_smoothing': 3,  # Smoothing scale for upwelling calculation
            'coastal_distance_threshold': 20,  # km for coastal influence
            'bathymetry_influence_scale': 50,  # km for bathymetric influence
        }
        
        # Pre-defined HAB hotspot locations in Seto Inland Sea
        self.hab_hotspots = {
            'Osaka Bay': {'lat': 34.6, 'lon': 135.2, 'severity': 'high'},
            'Harima-Nada': {'lat': 34.7, 'lon': 134.5, 'severity': 'high'},
            'Hiroshima Bay': {'lat': 34.3, 'lon': 132.4, 'severity': 'medium'},
            'Suo-Nada': {'lat': 34.0, 'lon': 131.8, 'severity': 'medium'},
            'Iyo-Nada': {'lat': 33.8, 'lon': 132.5, 'severity': 'medium'},
            'Kagawa Coast': {'lat': 34.3, 'lon': 134.0, 'severity': 'low'},
            'Okayama Coast': {'lat': 34.6, 'lon': 133.9, 'severity': 'medium'},
        }
        
        logger.info("🗺️ Advanced Spatial Feature Engineer initialized")
        logger.info(f"📍 Region: {region_bounds}")
        logger.info(f"🎯 HAB hotspots: {len(self.hab_hotspots)} locations")
        
    def calculate_spatial_gradients(self, 
                                  data: Union[np.ndarray, xr.DataArray],
                                  variable_name: str,
                                  lat_coords: np.ndarray,
                                  lon_coords: np.ndarray,
                                  scales: List[int] = None) -> Dict[str, np.ndarray]:
        """
        Calculate multi-scale spatial gradients for a variable
        
        Args:
            data: 2D array of variable values
            variable_name: Name of the variable
            lat_coords: Latitude coordinates
            lon_coords: Longitude coordinates  
            scales: List of scales (km) for gradient calculation
            
        Returns:
            Dictionary of gradient arrays at different scales
        """
        if scales is None:
            scales = self.spatial_params['gradient_scales']
            
        if isinstance(data, xr.DataArray):
            data = data.values
            
        gradients = {}
        
        # Calculate pixel size in km (approximate)
        lat_res = np.mean(np.diff(lat_coords)) * 111  # km per degree latitude
        lon_res = np.mean(np.diff(lon_coords)) * 111 * np.cos(np.radians(np.mean(lat_coords)))
        pixel_size = np.mean([lat_res, lon_res])
        
        for scale in scales:
            # Convert scale from km to pixels
            scale_pixels = max(1, int(scale / pixel_size))
            
            # Smooth data at the appropriate scale
            smoothed = ndimage.gaussian_filter(data, sigma=scale_pixels)
            
            # Calculate gradients
            grad_y, grad_x = np.gradient(smoothed)
            
            # Convert to physical units (per km)
            grad_y = grad_y / lat_res
            grad_x = grad_x / lon_res  
            
            # Calculate gradient magnitude and direction
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            grad_direction = np.arctan2(grad_y, grad_x)
            
            gradients[f'{variable_name}_gradient_magnitude_{scale}km'] = grad_magnitude
            gradients[f'{variable_name}_gradient_direction_{scale}km'] = grad_direction
            gradients[f'{variable_name}_gradient_x_{scale}km'] = grad_x
            gradients[f'{variable_name}_gradient_y_{scale}km'] = grad_y
            
        return gradients
    
    def detect_oceanographic_fronts(self,
                                  temperature: np.ndarray,
                                  salinity: Optional[np.ndarray] = None,
                                  lat_coords: np.ndarray = None,
                                  lon_coords: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Detect oceanographic fronts using temperature and salinity gradients
        
        Args:
            temperature: Sea surface temperature array
            salinity: Sea surface salinity array (optional)
            lat_coords: Latitude coordinates
            lon_coords: Longitude coordinates
            
        Returns:
            Dictionary of front detection features
        """
        fronts = {}
        
        # Temperature fronts
        temp_gradients = self.calculate_spatial_gradients(
            temperature, 'temperature', lat_coords, lon_coords, [5]  # 5km scale for fronts
        )
        temp_grad_mag = temp_gradients['temperature_gradient_magnitude_5km']
        
        # Detect strong temperature gradients (fronts)
        temp_front_threshold = np.nanstd(temp_grad_mag) * self.spatial_params['front_detection_threshold']
        temp_fronts = temp_grad_mag > temp_front_threshold
        
        fronts['temperature_front_strength'] = temp_grad_mag
        fronts['temperature_front_binary'] = temp_fronts.astype(float)
        
        # Enhanced front detection using simple edge detection (alternative to Canny)
        temp_normalized = (temperature - np.nanmin(temperature)) / (np.nanmax(temperature) - np.nanmin(temperature))
        # Simple edge detection using gradient magnitude
        grad_y, grad_x = np.gradient(temp_normalized)
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edge_threshold = np.nanstd(edge_magnitude) * 0.5
        temp_edges = (edge_magnitude > edge_threshold).astype(float)
        fronts['temperature_front_edges'] = temp_edges
        
        # Salinity fronts (if available)
        if salinity is not None:
            sal_gradients = self.calculate_spatial_gradients(
                salinity, 'salinity', lat_coords, lon_coords, [5]
            )
            sal_grad_mag = sal_gradients['salinity_gradient_magnitude_5km']
            
            sal_front_threshold = np.nanstd(sal_grad_mag) * self.spatial_params['front_detection_threshold']
            sal_fronts = sal_grad_mag > sal_front_threshold
            
            fronts['salinity_front_strength'] = sal_grad_mag
            fronts['salinity_front_binary'] = sal_fronts.astype(float)
            
            # Combined density fronts
            # Approximate density using temperature and salinity
            density_proxy = 1025 + 0.8 * (salinity - 35) - 0.2 * (temperature - 15)
            density_gradients = self.calculate_spatial_gradients(
                density_proxy, 'density', lat_coords, lon_coords, [5]
            )
            fronts['density_front_strength'] = density_gradients['density_gradient_magnitude_5km']
            
        return fronts
    
    def calculate_upwelling_indices(self,
                                  u_velocity: np.ndarray,
                                  v_velocity: np.ndarray,
                                  temperature: Optional[np.ndarray] = None,
                                  lat_coords: np.ndarray = None,
                                  lon_coords: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Calculate upwelling and downwelling indices
        
        Args:
            u_velocity: Eastward velocity component
            v_velocity: Northward velocity component  
            temperature: Sea surface temperature (optional, for validation)
            lat_coords: Latitude coordinates
            lon_coords: Longitude coordinates
            
        Returns:
            Dictionary of upwelling/downwelling features
        """
        upwelling_features = {}
        
        # Calculate divergence (upwelling indicator)
        du_dx = np.gradient(u_velocity, axis=1)
        dv_dy = np.gradient(v_velocity, axis=0)
        divergence = du_dx + dv_dy
        
        # Smooth divergence to reduce noise
        smooth_scale = self.spatial_params['upwelling_smoothing']
        divergence_smooth = ndimage.gaussian_filter(divergence, sigma=smooth_scale)
        
        upwelling_features['velocity_divergence'] = divergence_smooth
        upwelling_features['upwelling_index'] = np.maximum(divergence_smooth, 0)  # Positive divergence
        upwelling_features['downwelling_index'] = np.abs(np.minimum(divergence_smooth, 0))  # Negative divergence
        
        # Calculate vorticity (rotational flow)
        du_dy = np.gradient(u_velocity, axis=0)
        dv_dx = np.gradient(v_velocity, axis=1)
        vorticity = dv_dx - du_dy
        vorticity_smooth = ndimage.gaussian_filter(vorticity, sigma=smooth_scale)
        
        upwelling_features['vorticity'] = vorticity_smooth
        upwelling_features['cyclonic_vorticity'] = np.maximum(vorticity_smooth, 0)
        upwelling_features['anticyclonic_vorticity'] = np.abs(np.minimum(vorticity_smooth, 0))
        
        # Current speed and direction
        current_speed = np.sqrt(u_velocity**2 + v_velocity**2)
        current_direction = np.arctan2(v_velocity, u_velocity)
        
        upwelling_features['current_speed'] = current_speed
        upwelling_features['current_direction'] = current_direction
        
        # Current shear
        speed_gradients = self.calculate_spatial_gradients(
            current_speed, 'current_speed', lat_coords, lon_coords, [3]
        )
        upwelling_features['current_shear'] = speed_gradients['current_speed_gradient_magnitude_3km']
        
        # Ekman upwelling (if wind data available - placeholder)
        # This would require wind stress curl calculation
        
        return upwelling_features
    
    def calculate_stratification_features(self,
                                        temperature: np.ndarray,
                                        salinity: Optional[np.ndarray] = None,
                                        depth_layers: Optional[List[np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """
        Calculate water column stratification and mixing features
        
        Args:
            temperature: Surface temperature (or temperature profile)
            salinity: Surface salinity (or salinity profile)
            depth_layers: List of temperature/salinity at different depths
            
        Returns:
            Dictionary of stratification features
        """
        stratification_features = {}
        
        if depth_layers is not None and len(depth_layers) > 1:
            # Multi-layer stratification analysis
            # Calculate density for each layer
            density_layers = []
            for i, (temp_layer, sal_layer) in enumerate(depth_layers):
                if sal_layer is not None:
                    # Approximate density calculation
                    density = 1025 + 0.8 * (sal_layer - 35) - 0.2 * (temp_layer - 15)
                else:
                    # Temperature-only density approximation
                    density = 1025 - 0.2 * (temp_layer - 15)
                density_layers.append(density)
            
            # Calculate stratification index (density difference between layers)
            if len(density_layers) >= 2:
                stratification_index = density_layers[1] - density_layers[0]  # Bottom - Top
                stratification_features['stratification_index'] = stratification_index
                stratification_features['is_stratified'] = (stratification_index > 0.1).astype(float)
                
        else:
            # Single-layer analysis - use spatial variability as proxy
            temp_std = ndimage.uniform_filter(temperature**2, size=5) - ndimage.uniform_filter(temperature, size=5)**2
            stratification_features['temperature_spatial_variance'] = np.sqrt(np.abs(temp_std))
            
            if salinity is not None:
                sal_std = ndimage.uniform_filter(salinity**2, size=5) - ndimage.uniform_filter(salinity, size=5)**2
                stratification_features['salinity_spatial_variance'] = np.sqrt(np.abs(sal_std))
                
                # Mixed layer depth proxy using temperature-salinity relationship
                temp_sal_correlation = np.corrcoef(temperature.flatten(), salinity.flatten())[0,1]
                stratification_features['temp_sal_correlation'] = np.full_like(temperature, temp_sal_correlation)
        
        return stratification_features
    
    def calculate_coastal_proximity_features(self,
                                           lat_coords: np.ndarray,
                                           lon_coords: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate distance-based features from coastline and bathymetric features
        
        Args:
            lat_coords: Latitude coordinates
            lon_coords: Longitude coordinates
            
        Returns:
            Dictionary of coastal proximity features  
        """
        coastal_features = {}
        
        # Create coordinate grids
        lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
        
        # Simplified coastline approximation for Seto Inland Sea
        # In practice, this would use detailed coastline data
        coastline_points = [
            (34.7, 135.3),  # Osaka Bay
            (34.6, 134.8),  # Kii Channel
            (34.4, 134.0),  # Central Seto
            (34.2, 133.2),  # Hiroshima
            (34.0, 132.0),  # Western entrance
        ]
        
        # Calculate minimum distance to coastline for each point
        min_coastal_distance = np.full_like(lat_grid, np.inf)
        
        for coast_lat, coast_lon in coastline_points:
            distances = np.sqrt((lat_grid - coast_lat)**2 + (lon_grid - coast_lon)**2) * 111  # Approximate km
            min_coastal_distance = np.minimum(min_coastal_distance, distances)
            
        coastal_features['coastal_distance'] = min_coastal_distance
        coastal_features['is_coastal'] = (min_coastal_distance < self.spatial_params['coastal_distance_threshold']).astype(float)
        
        # Bathymetric influence (simplified)
        # Approximate depth based on distance from coast (deeper offshore)
        depth_approx = np.minimum(50, min_coastal_distance * 2)  # Max 50m depth
        coastal_features['depth_approximation'] = depth_approx
        coastal_features['is_shallow'] = (depth_approx < 20).astype(float)
        
        # Land-sea interaction features
        # River influence (simplified - major rivers in Seto Inland Sea)
        river_mouths = [
            (34.7, 135.1),  # Yodo River (Osaka)
            (34.4, 133.9),  # Yoshii River (Okayama)
        ]
        
        min_river_distance = np.full_like(lat_grid, np.inf)
        for river_lat, river_lon in river_mouths:
            distances = np.sqrt((lat_grid - river_lat)**2 + (lon_grid - river_lon)**2) * 111
            min_river_distance = np.minimum(min_river_distance, distances)
            
        coastal_features['river_distance'] = min_river_distance
        coastal_features['river_influence'] = np.exp(-min_river_distance / 10)  # Exponential decay over 10km
        
        return coastal_features
    
    def calculate_hab_hotspot_features(self,
                                     lat_coords: np.ndarray,
                                     lon_coords: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate distance-based features from known HAB hotspots
        
        Args:
            lat_coords: Latitude coordinates
            lon_coords: Longitude coordinates
            
        Returns:
            Dictionary of HAB hotspot proximity features
        """
        hotspot_features = {}
        
        # Create coordinate grids
        lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
        
        # Calculate distances to each hotspot
        for hotspot_name, hotspot_info in self.hab_hotspots.items():
            hotspot_lat = hotspot_info['lat']
            hotspot_lon = hotspot_info['lon']
            severity = hotspot_info['severity']
            
            # Calculate distance in km
            distances = np.sqrt((lat_grid - hotspot_lat)**2 + (lon_grid - hotspot_lon)**2) * 111
            
            # Create influence field (exponential decay)
            if severity == 'high':
                decay_scale = 20  # km
                max_influence = 1.0
            elif severity == 'medium':
                decay_scale = 15  # km
                max_influence = 0.7
            else:  # low
                decay_scale = 10  # km
                max_influence = 0.4
                
            influence = max_influence * np.exp(-distances / decay_scale)
            
            feature_name = f'hotspot_{hotspot_name.lower().replace(" ", "_")}_influence'
            hotspot_features[feature_name] = influence
            
        # Overall hotspot influence (maximum across all hotspots)
        all_influences = [v for k, v in hotspot_features.items() if k.endswith('_influence')]
        if all_influences:
            hotspot_features['max_hotspot_influence'] = np.maximum.reduce(all_influences)
            hotspot_features['total_hotspot_influence'] = np.sum(all_influences, axis=0)
            
        # Distance to nearest high-severity hotspot
        high_severity_distances = []
        for hotspot_name, hotspot_info in self.hab_hotspots.items():
            if hotspot_info['severity'] == 'high':
                distances = np.sqrt((lat_grid - hotspot_info['lat'])**2 + (lon_grid - hotspot_info['lon'])**2) * 111
                high_severity_distances.append(distances)
                
        if high_severity_distances:
            hotspot_features['nearest_high_hotspot_distance'] = np.minimum.reduce(high_severity_distances)
            
        return hotspot_features
    
    def create_comprehensive_spatial_features(self,
                                            data_dict: Dict[str, np.ndarray],
                                            lat_coords: np.ndarray,
                                            lon_coords: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create comprehensive spatial features from environmental data
        
        Args:
            data_dict: Dictionary of environmental variables (2D arrays)
            lat_coords: Latitude coordinate array
            lon_coords: Longitude coordinate array
            
        Returns:
            Dictionary of all spatial features
        """
        logger.info("🗺️ Creating comprehensive spatial features...")
        
        all_features = {}
        
        # Basic coordinate features
        lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
        all_features['latitude'] = lat_grid
        all_features['longitude'] = lon_grid
        
        # Coastal proximity features
        coastal_features = self.calculate_coastal_proximity_features(lat_coords, lon_coords)
        all_features.update(coastal_features)
        
        # HAB hotspot features
        hotspot_features = self.calculate_hab_hotspot_features(lat_coords, lon_coords)
        all_features.update(hotspot_features)
        
        # Process each environmental variable
        for var_name, var_data in data_dict.items():
            if var_data is None or np.all(np.isnan(var_data)):
                logger.warning(f"Skipping {var_name} - no valid data")
                continue
                
            logger.info(f"Processing spatial features for {var_name}")
            
            # Spatial gradients
            gradients = self.calculate_spatial_gradients(var_data, var_name, lat_coords, lon_coords)
            all_features.update(gradients)
            
        # Oceanographic fronts (requires temperature, optionally salinity)
        if 'temperature' in data_dict:
            salinity_data = data_dict.get('salinity', None)
            front_features = self.detect_oceanographic_fronts(
                data_dict['temperature'], salinity_data, lat_coords, lon_coords
            )
            all_features.update(front_features)
            
        # Upwelling indices (requires velocity components)
        if 'u_velocity' in data_dict and 'v_velocity' in data_dict:
            upwelling_features = self.calculate_upwelling_indices(
                data_dict['u_velocity'], data_dict['v_velocity'],
                data_dict.get('temperature', None), lat_coords, lon_coords
            )
            all_features.update(upwelling_features)
            
        # Stratification features
        if 'temperature' in data_dict:
            stratification_features = self.calculate_stratification_features(
                data_dict['temperature'], data_dict.get('salinity', None)
            )
            all_features.update(stratification_features)
            
        logger.info(f"✅ Created {len(all_features)} spatial features")
        return all_features