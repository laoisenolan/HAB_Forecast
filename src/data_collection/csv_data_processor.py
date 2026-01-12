"""
CSV Data Processor for HAB Forecasting System

This module standardizes all data collection to use CSV format,
making data processing, analysis, and integration much simpler.
"""

import logging
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json

logger = logging.getLogger(__name__)

class CSVDataProcessor:
    """
    Processes and standardizes environmental data in CSV format.
    
    Features:
    - Converts xarray datasets to standardized CSV format
    - Handles all 25 HAB variables consistently
    - Maintains spatial and temporal coordinates
    - Adds metadata and quality flags
    - Supports data merging and aggregation
    """
    
    def __init__(self, output_dir: str = "data/csv"):
        """Initialize CSV data processor"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Standardized column structure for all HAB data
        self.standard_columns = [
            'timestamp',      # ISO format datetime
            'date',           # YYYY-MM-DD (data point date)
            'latitude',       # Decimal degrees
            'longitude',      # Decimal degrees
            'variable_name',  # HAB variable name
            'value',         # Measured/modeled value
            'unit',          # Unit of measurement
            'data_source',   # Source (copernicus, gee, erddap)
            'dataset_id',    # Original dataset identifier
            'quality_flag',  # Quality assessment flag
            'collection_method', # Collection method info
            'spatial_resolution', # Original spatial resolution
            'temporal_resolution' # Original temporal resolution
        ]
        
        # HAB variable standardization
        self.variable_mapping = {
            # Core HAB Indicators
            'analysed_sst': {
                'standard_name': 'sea_surface_temperature',
                'unit': 'degrees_celsius',
                'description': 'Sea Surface Temperature'
            },
            'CHL': {
                'standard_name': 'chlorophyll_a',
                'unit': 'mg_m3',
                'description': 'Chlorophyll-a concentration'
            },
            'CHLA_AVE': {
                'standard_name': 'chlorophyll_a_jaxa',
                'unit': 'mg_m3',
                'description': 'High-resolution Chlorophyll-a (JAXA)'
            },
            'KD490': {
                'standard_name': 'turbidity_kd490',
                'unit': 'm-1',
                'description': 'Diffuse attenuation coefficient (turbidity)'
            },
            'B4_B2_ratio': {
                'standard_name': 'turbidity_sentinel2',
                'unit': 'dimensionless',
                'description': 'Sentinel-2 turbidity index (B4/B2 ratio)'
            },
            
            # Physical Oceanography
            'water_temp_0': {
                'standard_name': 'ocean_temperature_hycom',
                'unit': 'degrees_celsius',
                'description': 'Ocean temperature (HYCOM surface)'
            },
            'salinity_0': {
                'standard_name': 'ocean_salinity',
                'unit': 'psu',
                'description': 'Ocean salinity (HYCOM surface)'
            },
            'velocity_u_0': {
                'standard_name': 'ocean_current_u',
                'unit': 'm_s',
                'description': 'Eastward ocean current velocity'
            },
            'velocity_v_0': {
                'standard_name': 'ocean_current_v',
                'unit': 'm_s',
                'description': 'Northward ocean current velocity'
            },
            'surface_elevation': {
                'standard_name': 'sea_surface_elevation',
                'unit': 'm',
                'description': 'Sea surface elevation'
            },
            'mlotst': {
                'standard_name': 'mixed_layer_depth',
                'unit': 'm',
                'description': 'Ocean mixed layer depth'
            },
            'zos': {
                'standard_name': 'sea_level_anomaly',
                'unit': 'm',
                'description': 'Sea surface height anomaly'
            },
            
            # Atmospheric Variables
            'u_component_of_wind_10m': {
                'standard_name': 'wind_u',
                'unit': 'm_s',
                'description': '10m U wind component'
            },
            'v_component_of_wind_10m': {
                'standard_name': 'wind_v',
                'unit': 'm_s',
                'description': '10m V wind component'
            },
            'total_precipitation': {
                'standard_name': 'precipitation',
                'unit': 'm',
                'description': 'Total precipitation'
            },
            
            # Biogeochemistry
            'no3': {
                'standard_name': 'nitrate',
                'unit': 'mmol_m3',
                'description': 'Nitrate concentration'
            },
            'po4': {
                'standard_name': 'phosphate',
                'unit': 'mmol_m3',
                'description': 'Phosphate concentration'
            },
            'si': {
                'standard_name': 'silicate',
                'unit': 'mmol_m3',
                'description': 'Silicate concentration'
            },
            'fe': {
                'standard_name': 'dissolved_iron',
                'unit': 'mmol_m3',
                'description': 'Dissolved iron concentration'
            },
            'o2': {
                'standard_name': 'dissolved_oxygen',
                'unit': 'mmol_m3',
                'description': 'Dissolved oxygen concentration'
            },
            'nppv': {
                'standard_name': 'primary_production',
                'unit': 'mg_c_m3_day',
                'description': 'Net primary production'
            },
            'chl': {
                'standard_name': 'total_chlorophyll_bgc',
                'unit': 'mg_m3',
                'description': 'Total chlorophyll (BGC model)'
            },
            'phyc': {
                'standard_name': 'phytoplankton_carbon',
                'unit': 'mg_c_m3',
                'description': 'Phytoplankton carbon biomass'
            },
            'kd': {
                'standard_name': 'diffuse_attenuation',
                'unit': 'm-1',
                'description': 'Diffuse attenuation coefficient (model)'
            }
        }
        
        logger.info(f"CSV data processor initialized at: {self.output_dir}")
    
    def convert_xarray_to_csv(self, 
                             dataset: xr.Dataset, 
                             data_source: str,
                             dataset_id: str,
                             timestamp: Optional[str] = None) -> pd.DataFrame:
        """
        Convert xarray Dataset to standardized CSV format
        
        Args:
            dataset: xarray Dataset to convert
            data_source: Data source name (copernicus, gee, erddap)
            dataset_id: Original dataset identifier
            timestamp: Collection timestamp
            
        Returns:
            Standardized pandas DataFrame
        """
        logger.info(f"Converting xarray dataset to CSV format from {data_source}")
        
        try:
            # Initialize list to collect rows
            rows = []
            
            # Get coordinates
            lat_coords = None
            lon_coords = None
            time_coords = None
            
            # Handle different coordinate naming conventions
            for coord_name in dataset.coords:
                coord_lower = coord_name.lower()
                if 'lat' in coord_lower and lat_coords is None:
                    lat_coords = dataset.coords[coord_name]
                elif 'lon' in coord_lower and lon_coords is None:
                    lon_coords = dataset.coords[coord_name]
                elif 'time' in coord_lower and time_coords is None:
                    time_coords = dataset.coords[coord_name]
            
            # Process each data variable
            for var_name in dataset.data_vars:
                data_array = dataset[var_name]
                
                # Get variable metadata
                var_info = self.variable_mapping.get(var_name, {
                    'standard_name': var_name,
                    'unit': 'unknown',
                    'description': f'Variable: {var_name}'
                })
                
                # Handle different dimensionalities
                if data_array.dims == ():
                    # Scalar value
                    ts_val = pd.to_datetime(timestamp or datetime.now())
                    rows.append({
                        'timestamp': ts_val.isoformat(),
                        'date': ts_val.strftime('%Y-%m-%d'),
                        'latitude': None,
                        'longitude': None,
                        'variable_name': var_info['standard_name'],
                        'value': float(data_array.values),
                        'unit': var_info['unit'],
                        'data_source': data_source,
                        'dataset_id': dataset_id,
                        'quality_flag': 'good',
                        'collection_method': 'direct',
                        'spatial_resolution': 'point',
                        'temporal_resolution': 'instant'
                    })
                else:
                    # Multi-dimensional data
                    # Create a flattened version
                    values = data_array.values
                    
                    # Handle different dimension structures
                    if 'time' in data_array.dims:
                        if time_coords is not None:
                            time_values = time_coords.values
                        else:
                            time_values = [timestamp or datetime.now()]
                    else:
                        time_values = [timestamp or datetime.now()]
                    
                    # Spatial coordinates
                    if lat_coords is not None and lon_coords is not None:
                        lat_values = lat_coords.values
                        lon_values = lon_coords.values
                        
                        # Handle 1D vs 2D coordinate grids
                        if len(lat_values.shape) == 1 and len(lon_values.shape) == 1:
                            # 1D coordinates - create meshgrid
                            lat_grid, lon_grid = np.meshgrid(lat_values, lon_values, indexing='ij')
                        else:
                            # 2D coordinates
                            lat_grid, lon_grid = lat_values, lon_values
                    else:
                        # No spatial coordinates - use points or assume regional
                        lat_grid = np.array([[34.5]])  # Default Seto Inland Sea center
                        lon_grid = np.array([[133.5]])
                    
                    # Flatten and create rows
                    if len(values.shape) == 3:  # time, lat, lon
                        for t_idx, time_val in enumerate(time_values):
                            ts = pd.to_datetime(time_val)
                            for i in range(values.shape[1]):
                                for j in range(values.shape[2]):
                                    if not np.isnan(values[t_idx, i, j]):
                                        rows.append({
                                            'timestamp': ts.isoformat(),
                                            'date': ts.strftime('%Y-%m-%d'),
                                            'latitude': float(lat_grid[i, j] if len(lat_grid.shape) == 2 else lat_grid[i]),
                                            'longitude': float(lon_grid[i, j] if len(lon_grid.shape) == 2 else lon_grid[j]),
                                            'variable_name': var_info['standard_name'],
                                            'value': float(values[t_idx, i, j]),
                                            'unit': var_info['unit'],
                                            'data_source': data_source,
                                            'dataset_id': dataset_id,
                                            'quality_flag': 'good',
                                            'collection_method': 'gridded',
                                            'spatial_resolution': self._estimate_spatial_resolution(lat_grid, lon_grid),
                                            'temporal_resolution': 'daily'
                                        })
                    elif len(values.shape) == 2:  # lat, lon (single time)
                        time_val = time_values[0] if time_values else datetime.now()
                        ts = pd.to_datetime(time_val)
                        for i in range(values.shape[0]):
                            for j in range(values.shape[1]):
                                if not np.isnan(values[i, j]):
                                    rows.append({
                                        'timestamp': ts.isoformat(),
                                        'date': ts.strftime('%Y-%m-%d'),
                                        'latitude': float(lat_grid[i, j] if len(lat_grid.shape) == 2 else lat_grid[i]),
                                        'longitude': float(lon_grid[i, j] if len(lon_grid.shape) == 2 else lon_grid[j]),
                                        'variable_name': var_info['standard_name'],
                                        'value': float(values[i, j]),
                                        'unit': var_info['unit'],
                                        'data_source': data_source,
                                        'dataset_id': dataset_id,
                                        'quality_flag': 'good',
                                        'collection_method': 'gridded',
                                        'spatial_resolution': self._estimate_spatial_resolution(lat_grid, lon_grid),
                                        'temporal_resolution': 'snapshot'
                                    })
                    elif len(values.shape) == 1:  # point data (time series or spatial points)
                        for idx, value in enumerate(values):
                            if not np.isnan(value):
                                # Determine if this is time series or spatial points
                                if len(time_values) == len(values):
                                    # Time series
                                    time_val = time_values[idx]
                                    lat_val = lat_grid.flat[0]
                                    lon_val = lon_grid.flat[0]
                                else:
                                    # Spatial points
                                    time_val = time_values[0] if time_values else datetime.now()
                                    lat_val = lat_grid.flat[idx] if len(lat_grid.flat) > idx else lat_grid.flat[0]
                                    lon_val = lon_grid.flat[idx] if len(lon_grid.flat) > idx else lon_grid.flat[0]
                                ts = pd.to_datetime(time_val)
                                
                                rows.append({
                                    'timestamp': ts.isoformat(),
                                    'date': ts.strftime('%Y-%m-%d'),
                                    'latitude': float(lat_val),
                                    'longitude': float(lon_val),
                                    'variable_name': var_info['standard_name'],
                                    'value': float(value),
                                    'unit': var_info['unit'],
                                    'data_source': data_source,
                                    'dataset_id': dataset_id,
                                    'quality_flag': 'good',
                                    'collection_method': 'point',
                                    'spatial_resolution': 'point',
                                    'temporal_resolution': 'time_series'
                                })
            
            # Create DataFrame
            df = pd.DataFrame(rows)
            
            # Ensure all standard columns are present
            for col in self.standard_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Reorder columns
            df = df[self.standard_columns]
            
            # Sort by timestamp and location
            df = df.sort_values(['timestamp', 'latitude', 'longitude', 'variable_name'])
            
            logger.info(f"Converted to CSV: {len(df)} rows with {len(df['variable_name'].unique())} variables")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to convert xarray to CSV: {e}")
            return pd.DataFrame(columns=self.standard_columns)
    
    def convert_gee_data_to_csv(self, 
                               gee_data: Dict, 
                               data_source: str = "gee",
                               timestamp: Optional[str] = None) -> pd.DataFrame:
        """Convert Google Earth Engine data to standardized CSV format"""
        logger.info("Converting GEE data to CSV format")
        
        try:
            rows = []
            
            for data_type, data in gee_data.items():
                if data_type == 'metadata':
                    continue
                    
                if isinstance(data, pd.DataFrame):
                    # Process DataFrame data
                    for _, row in data.iterrows():
                        for col in data.columns:
                            if col in ['latitude', 'longitude', 'date']:
                                continue
                            
                            # Get variable info
                            var_info = self.variable_mapping.get(col, {
                                'standard_name': col,
                                'unit': 'unknown',
                                'description': f'GEE variable: {col}'
                            })
                            
                            # Prefer explicit per-row date if present; else fall back to provided timestamp
                            row_date = pd.to_datetime(row.get('date', timestamp or datetime.now()))
                            rows.append({
                                'timestamp': row_date.isoformat(),
                                'date': row_date.strftime('%Y-%m-%d'),
                                'latitude': row.get('latitude', 34.5),
                                'longitude': row.get('longitude', 133.5),
                                'variable_name': var_info['standard_name'],
                                'value': row[col],
                                'unit': var_info['unit'],
                                'data_source': data_source,
                                'dataset_id': f"gee_{data_type}",
                                'quality_flag': 'good',
                                'collection_method': 'gee_processed',
                                'spatial_resolution': 'variable',
                                'temporal_resolution': 'daily'
                            })
                
                elif isinstance(data, dict):
                    # Process dictionary data (includes GEE statistics)
                    logger.debug(f"Processing dict data: {data}")
                    
                    # Handle GEE statistics structure
                    if 'statistics' in data and isinstance(data['statistics'], dict):
                        stats_data = data['statistics']
                        for var_name, value in stats_data.items():
                            if isinstance(value, (int, float, np.number)) and value is not None:
                                # Clean up variable names
                                clean_var_name = var_name.replace('_mean', '').replace('_median', '')
                                
                                var_info = self.variable_mapping.get(clean_var_name, {
                                    'standard_name': clean_var_name,
                                    'unit': 'unknown',
                                    'description': f'GEE variable: {clean_var_name}'
                                })
                                ts = pd.to_datetime(timestamp or datetime.now())
                                
                                rows.append({
                                    'timestamp': ts.isoformat(),
                                    'date': ts.strftime('%Y-%m-%d'),
                                    'latitude': 34.5,  # Seto Inland Sea center
                                    'longitude': 133.5,
                                    'variable_name': var_info['standard_name'],
                                    'value': float(value),
                                    'unit': var_info['unit'],
                                    'data_source': data_source,
                                    'dataset_id': data.get('dataset_id', f"gee_{data_type}"),
                                    'quality_flag': 'good',
                                    'collection_method': 'gee_processed',
                                    'spatial_resolution': 'regional_mean',
                                    'temporal_resolution': 'period_mean'
                                })
                    else:
                        # Regular dictionary processing
                        for var_name, value in data.items():
                            if isinstance(value, (int, float, np.number)) and value is not None:
                                var_info = self.variable_mapping.get(var_name, {
                                    'standard_name': var_name,
                                    'unit': 'unknown',
                                    'description': f'GEE variable: {var_name}'
                                })
                                ts = pd.to_datetime(timestamp or datetime.now())
                                
                                rows.append({
                                'timestamp': ts.isoformat(),
                                'date': ts.strftime('%Y-%m-%d'),
                                'latitude': 34.5,  # Default Seto Inland Sea
                                'longitude': 133.5,
                                'variable_name': var_info['standard_name'],
                                'value': float(value),
                                'unit': var_info['unit'],
                                'data_source': data_source,
                                'dataset_id': f"gee_{data_type}",
                                'quality_flag': 'good',
                                'collection_method': 'gee_processed',
                                'spatial_resolution': 'regional',
                                'temporal_resolution': 'snapshot'
                            })
            
            # Create DataFrame
            df = pd.DataFrame(rows)
            
            # Ensure all standard columns are present
            for col in self.standard_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Reorder columns
            df = df[self.standard_columns]
            
            logger.info(f"Converted GEE data to CSV: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to convert GEE data to CSV: {e}")
            return pd.DataFrame(columns=self.standard_columns)
    
    def save_csv(self, 
                 df: pd.DataFrame, 
                 filename: str,
                 subdirectory: Optional[str] = None) -> Path:
        """Save DataFrame to CSV file"""
        if subdirectory:
            save_dir = self.output_dir / subdirectory
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.output_dir
            
        file_path = save_dir / f"{filename}.csv"
        
        try:
            df.to_csv(file_path, index=False)
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Saved CSV file: {file_path} ({file_size:.2f} MB, {len(df)} rows)")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save CSV file {file_path}: {e}")
            raise

    def append_to_master_long(self,
                              df: pd.DataFrame,
                              master_filename: str = "historical_master_long.csv",
                              subdirectory: Optional[str] = "historical",
                              dedup_keys: Optional[List[str]] = None) -> Path:
        """Append rows to a single long-format master CSV with de-duplication.
        Ensures a 'date' column (YYYY-MM-DD) is present for every row.
        """
        if subdirectory:
            save_dir = self.output_dir / subdirectory
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.output_dir
        master_path = save_dir / master_filename

        try:
            # Ensure date column exists
            if 'date' not in df.columns:
                if 'timestamp' in df.columns:
                    df['date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.strftime('%Y-%m-%d')
                else:
                    df['date'] = None

            # Default dedup keys
            if dedup_keys is None:
                dedup_keys = ['date', 'latitude', 'longitude', 'variable_name', 'dataset_id']

            # Write new or append+dedup
            if not master_path.exists():
                df.to_csv(master_path, index=False)
            else:
                # Load existing with safe memory usage for moderate size
                existing = pd.read_csv(master_path)
                merged = pd.concat([existing, df], ignore_index=True)
                # Only keep columns that exist in either to avoid misalignment
                common_cols = list(merged.columns)
                merged = merged[common_cols]
                # Deduplicate
                existing_keys = [k for k in dedup_keys if k in merged.columns]
                if existing_keys:
                    merged = merged.drop_duplicates(subset=existing_keys, keep='first')
                merged.to_csv(master_path, index=False)

            logger.info(f"Appended to master CSV: {master_path}")
            return master_path
        except Exception as e:
            logger.error(f"Failed to append to master CSV {master_path}: {e}")
            raise
    
    def merge_csv_files(self, csv_files: List[Path]) -> pd.DataFrame:
        """Merge multiple CSV files with the same structure"""
        logger.info(f"Merging {len(csv_files)} CSV files")
        
        try:
            dataframes = []
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    dataframes.append(df)
                    logger.info(f"Loaded {csv_file.name}: {len(df)} rows")
                except Exception as e:
                    logger.warning(f"Failed to load {csv_file}: {e}")
            
            if dataframes:
                merged_df = pd.concat(dataframes, ignore_index=True)
                
                # Remove duplicates (same timestamp, location, variable)
                merged_df = merged_df.drop_duplicates(
                    subset=['timestamp', 'latitude', 'longitude', 'variable_name', 'data_source'],
                    keep='first'
                )
                
                # Sort by timestamp and location
                merged_df = merged_df.sort_values(['timestamp', 'variable_name', 'latitude', 'longitude'])
                
                logger.info(f"Merged CSV: {len(merged_df)} rows from {len(dataframes)} files")
                return merged_df
            else:
                logger.warning("No valid CSV files to merge")
                return pd.DataFrame(columns=self.standard_columns)
                
        except Exception as e:
            logger.error(f"Failed to merge CSV files: {e}")
            return pd.DataFrame(columns=self.standard_columns)

    def build_wide_feature_matrix(
        self,
        master_filename: str = "historical_2017_2024_master.csv",
        output_filename: str = "historical_2017_2024_master_wide.csv",
        subdirectory: Optional[str] = "historical",
    ) -> Path:
        """Create a wide pivoted feature matrix from the long master CSV.

        Index: ['date','latitude','longitude']
        Columns: distinct 'variable_name'
        Values: 'value' (mean if duplicates)
        """
        try:
            save_dir = self.output_dir / subdirectory if subdirectory else self.output_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            master_path = save_dir / master_filename
            out_path = save_dir / output_filename

            if not master_path.exists():
                raise FileNotFoundError(f"Master CSV not found: {master_path}")

            df = pd.read_csv(master_path)
            required = {'date', 'latitude', 'longitude', 'variable_name', 'value'}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Master CSV missing columns: {missing}")

            # Remove obvious duplicates before pivot
            df = df.drop_duplicates(subset=['date','latitude','longitude','variable_name'])

            wide = df.pivot_table(
                index=['date', 'latitude', 'longitude'],
                columns='variable_name',
                values='value',
                aggfunc='mean'
            ).reset_index()

            # Sort columns for stability
            non_feat_cols = ['date', 'latitude', 'longitude']
            feat_cols = [c for c in wide.columns if c not in non_feat_cols]
            wide = wide[non_feat_cols + sorted(feat_cols)]

            wide.to_csv(out_path, index=False)
            logger.info(f"Updated wide feature matrix: {out_path} (rows={len(wide)})")
            return out_path
        except Exception as e:
            logger.error(f"Failed to build wide feature matrix: {e}")
            raise
    
    def _estimate_spatial_resolution(self, lat_grid: np.ndarray, lon_grid: np.ndarray) -> str:
        """Estimate spatial resolution from coordinate grids"""
        try:
            if len(lat_grid.shape) == 2 and lat_grid.shape[0] > 1:
                lat_diff = abs(lat_grid[1, 0] - lat_grid[0, 0])
                lon_diff = abs(lon_grid[0, 1] - lon_grid[0, 0])
                avg_diff = (lat_diff + lon_diff) / 2
                
                if avg_diff < 0.01:
                    return "high_res_<1km"
                elif avg_diff < 0.05:
                    return "medium_res_1-5km" 
                elif avg_diff < 0.25:
                    return "low_res_5-25km"
                else:
                    return "coarse_res_>25km"
            else:
                return "point"
        except:
            return "unknown"
    
    def get_variable_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics for variables in CSV data"""
        try:
            summary = {
                'total_records': len(df),
                'variables': df['variable_name'].unique().tolist(),
                'variable_count': df['variable_name'].nunique(),
                'data_sources': df['data_source'].unique().tolist(),
                'time_range': {
                    'start': df['timestamp'].min(),
                    'end': df['timestamp'].max()
                },
                'spatial_extent': {
                    'lat_min': df['latitude'].min(),
                    'lat_max': df['latitude'].max(),
                    'lon_min': df['longitude'].min(),
                    'lon_max': df['longitude'].max()
                },
                'variable_stats': {}
            }
            
            # Per-variable statistics
            for var in df['variable_name'].unique():
                var_data = df[df['variable_name'] == var]
                summary['variable_stats'][var] = {
                    'count': len(var_data),
                    'mean': var_data['value'].mean(),
                    'std': var_data['value'].std(),
                    'min': var_data['value'].min(),
                    'max': var_data['value'].max(),
                    'unit': var_data['unit'].iloc[0] if len(var_data) > 0 else 'unknown'
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate variable summary: {e}")
            return {}