"""
Spatial Environmental Data Mapper

Maps environmental data to the spatial grid system for spatial-temporal HAB prediction.
Handles interpolation and aggregation of environmental variables to grid cells.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata, NearestNDInterpolator
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
import numpy as np

from .spatial_temporal_grid import SetoInlandSeaSpatialGrid, GridCell

warnings.filterwarnings('ignore', category=RuntimeWarning)
logger = logging.getLogger(__name__)

class SpatialEnvironmentalMapper:
    """
    Maps environmental data to spatial grid cells for spatial-temporal modeling
    
    This class handles:
    - Interpolation of point environmental data to grid cells
    - Temporal alignment of environmental data with grid labels
    - Feature engineering for spatial-temporal modeling
    - Missing data handling and quality control
    """
    
    def __init__(self, 
                 spatial_grid: SetoInlandSeaSpatialGrid,
                 interpolation_method: str = 'nearest',
                 max_interpolation_distance: float = 50.0):  # km
        """
        Initialize spatial environmental mapper
        
        Args:
            spatial_grid: Spatial grid system
            interpolation_method: Method for spatial interpolation ('nearest', 'linear', 'cubic')
            max_interpolation_distance: Maximum distance for interpolation (km)
        """
        self.spatial_grid = spatial_grid
        self.interpolation_method = interpolation_method
        self.max_interpolation_distance = max_interpolation_distance
        
        # Environmental variable categories
        self.variable_categories = {
            'physical': [
                'sst', 'sst_mean', 'ocean_temperature_hycom', 'sea_surface_elevation',
                'ocean_current_u', 'ocean_current_v', 'wind_u', 'wind_v'
            ],
            'optical': [
                'CHLA_AVE_mean', 'chlor_a', 'chlor_a_mean', 'chlorophyll_a_jaxa',
                'Rrs_443', 'Rrs_443_mean', 'Rrs_555', 'Rrs_555_mean', 
                'Rrs_667', 'Rrs_667_mean'
            ],
            'water_quality': [
                'turbidity_index', 'turbidity_index_max', 'turbidity_index_mean', 
                'turbidity_index_min', 'turbidity_index_stdDev', 'turbidity_modis',
                'ocean_salinity'
            ],
            'meteorological': [
                'precipitation'
            ]
        }
        
        # All environmental variables
        self.all_variables = []
        for category_vars in self.variable_categories.values():
            self.all_variables.extend(category_vars)
        
        logger.info(f"🌊 Initialized spatial environmental mapper")
        logger.info(f"   Interpolation method: {interpolation_method}")
        logger.info(f"   Max interpolation distance: {max_interpolation_distance} km")
        logger.info(f"   Environmental variables: {len(self.all_variables)}")
    
    def map_environmental_data_to_grid(self, 
                                     environmental_df: pd.DataFrame,
                                     point_coordinates: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Map environmental data to spatial grid cells
        
        Args:
            environmental_df: Environmental data DataFrame with date, lat, lon, and variables
            point_coordinates: If provided, treats all data as from this single point
            
        Returns:
            DataFrame with environmental data mapped to grid cells
        """
        logger.info(f"🗺️ Mapping environmental data to spatial grid...")
        logger.info(f"   Input data: {len(environmental_df)} records")
        
        # Prepare environmental data
        env_df = environmental_df.copy()
        
        # Handle date column
        if 'date' not in env_df.columns:
            logger.error("Environmental data must have a 'date' column")
            raise ValueError("Missing 'date' column in environmental data")
        
        env_df['date'] = pd.to_datetime(env_df['date'])
        
        # Handle coordinates
        if point_coordinates is not None:
            # Single point data - assign coordinates to all records
            lat, lon = point_coordinates
            env_df['latitude'] = lat
            env_df['longitude'] = lon
            logger.info(f"   Using single point coordinates: {lat:.3f}, {lon:.3f}")
        else:
            # Multi-point data - expect lat/lon columns
            if 'latitude' not in env_df.columns or 'longitude' not in env_df.columns:
                logger.error("Environmental data must have 'latitude' and 'longitude' columns")
                raise ValueError("Missing coordinate columns in environmental data")
        
        # Get available environmental variables
        available_vars = [col for col in self.all_variables if col in env_df.columns]
        missing_vars = [col for col in self.all_variables if col not in env_df.columns]
        
        logger.info(f"   Available variables: {len(available_vars)}")
        if missing_vars:
            logger.warning(f"   Missing variables: {missing_vars[:5]}{'...' if len(missing_vars) > 5 else ''}")
        
        # Map to grid cells
        grid_mapped_data = []
        
        # Group by date for efficient processing
        for date, date_group in env_df.groupby('date'):
            logger.debug(f"Processing date: {date.strftime('%Y-%m-%d')}")
            
            # Get coordinates and values for this date
            coords = date_group[['latitude', 'longitude']].values
            
            # Map environmental data to each grid cell for this date
            for cell_id, cell in self.spatial_grid.grid_cells.items():
                if not cell.is_valid:
                    continue
                
                # Get environmental values for this cell
                cell_env_data = self._interpolate_to_cell(
                    cell, coords, date_group, available_vars
                )
                
                if cell_env_data is not None:
                    # Create record for this cell and date
                    record = {
                        'date': date,
                        'grid_cell_id': cell_id,
                        'lat': cell.lat_center,
                        'lon': cell.lon_center,
                        'bay_region': cell.bay_region,
                        **cell_env_data
                    }
                    
                    grid_mapped_data.append(record)
        
        # Create DataFrame
        if not grid_mapped_data:
            logger.error("No environmental data could be mapped to grid cells")
            return pd.DataFrame()
        
        grid_env_df = pd.DataFrame(grid_mapped_data)
        
        logger.info(f"✅ Environmental data mapped to grid:")
        logger.info(f"   Output records: {len(grid_env_df):,}")
        logger.info(f"   Date range: {grid_env_df['date'].min()} to {grid_env_df['date'].max()}")
        logger.info(f"   Grid cells with data: {grid_env_df['grid_cell_id'].nunique()}")
        
        return grid_env_df
    
    def _interpolate_to_cell(self, 
                           cell: GridCell, 
                           coords: np.ndarray, 
                           data_group: pd.DataFrame, 
                           variables: List[str]) -> Optional[Dict[str, float]]:
        """
        Interpolate environmental data to a specific grid cell
        
        Args:
            cell: Target grid cell
            coords: Array of coordinate points
            data_group: Environmental data for the date
            variables: List of environmental variables to interpolate
            
        Returns:
            Dictionary of interpolated values or None if interpolation fails
        """
        cell_coords = np.array([[cell.lat_center, cell.lon_center]])
        
        # Calculate distances to all data points
        distances = cdist(cell_coords, coords, metric='euclidean')[0]
        
        # Convert distance in degrees to km (rough approximation)
        distances_km = distances * 111.0  # 1 degree ≈ 111 km
        
        # Check if any points are within interpolation distance
        within_distance = distances_km <= self.max_interpolation_distance
        
        if not np.any(within_distance):
            # No points within interpolation distance
            return None
        
        # Filter to points within distance
        nearby_coords = coords[within_distance]
        nearby_data = data_group.iloc[np.where(within_distance)[0]]
        
        interpolated_values = {}
        
        for var in variables:
            if var not in nearby_data.columns:
                continue
            
            # Get values for this variable
            values = nearby_data[var].values
            
            # Handle missing values
            valid_mask = ~pd.isna(values)
            if not np.any(valid_mask):
                interpolated_values[var] = np.nan
                continue
            
            valid_coords = nearby_coords[valid_mask]
            valid_values = values[valid_mask]
            
            if len(valid_values) == 1:
                # Only one valid point - use its value
                interpolated_values[var] = valid_values[0]
            else:
                # Multiple points - interpolate
                try:
                    if self.interpolation_method == 'nearest':
                        # Find nearest valid point
                        point_distances = cdist(cell_coords, valid_coords)[0]
                        nearest_idx = np.argmin(point_distances)
                        interpolated_values[var] = valid_values[nearest_idx]
                    
                    elif self.interpolation_method in ['linear', 'cubic']:
                        # Use griddata for interpolation
                        interpolated_value = griddata(
                            valid_coords, valid_values, cell_coords,
                            method=self.interpolation_method
                        )[0]
                        
                        if np.isnan(interpolated_value):
                            # Fallback to nearest if linear/cubic fails
                            point_distances = cdist(cell_coords, valid_coords)[0]
                            nearest_idx = np.argmin(point_distances)
                            interpolated_values[var] = valid_values[nearest_idx]
                        else:
                            interpolated_values[var] = interpolated_value
                    
                    else:
                        # Unknown method - use nearest
                        point_distances = cdist(cell_coords, valid_coords)[0]
                        nearest_idx = np.argmin(point_distances)
                        interpolated_values[var] = valid_values[nearest_idx]
                        
                except Exception as e:
                    logger.debug(f"Interpolation failed for {var}: {e}")
                    # Fallback to nearest valid value
                    point_distances = cdist(cell_coords, valid_coords)[0]
                    nearest_idx = np.argmin(point_distances)
                    interpolated_values[var] = valid_values[nearest_idx]
        
        return interpolated_values if interpolated_values else None
    
    def create_spatial_features(self, grid_env_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create spatial features for each grid cell
        
        Args:
            grid_env_df: Environmental data mapped to grid cells
            
        Returns:
            DataFrame with additional spatial features
        """
        logger.info("🧮 Creating spatial features...")
        
        enhanced_df = grid_env_df.copy()
        
        # Get available environmental variables
        env_vars = [col for col in self.all_variables if col in enhanced_df.columns]
        
        # Group by date for spatial feature calculation
        spatial_features_list = []
        
        for date, date_group in enhanced_df.groupby('date'):
            logger.debug(f"Creating spatial features for {date.strftime('%Y-%m-%d')}")
            
            date_features = []
            
            for _, row in date_group.iterrows():
                cell_id = row['grid_cell_id']
                cell = self.spatial_grid.grid_cells[cell_id]
                
                # Get neighboring cells
                neighbors = self.spatial_grid.get_neighboring_cells(cell_id, distance=1)
                neighbor_data = date_group[date_group['grid_cell_id'].isin(neighbors)]
                
                spatial_features = {'grid_cell_id': cell_id, 'date': date}
                
                # Calculate spatial features for each environmental variable
                for var in env_vars:
                    if var in row and not pd.isna(row[var]):
                        # Current cell value
                        current_value = row[var]
                        spatial_features[var] = current_value
                        
                        # Spatial gradients and neighbors
                        if len(neighbor_data) > 0:
                            neighbor_values = neighbor_data[var].dropna()
                            
                            if len(neighbor_values) > 0:
                                # Spatial gradient (difference from neighbors)
                                mean_neighbor = neighbor_values.mean()
                                spatial_features[f'{var}_spatial_gradient'] = current_value - mean_neighbor
                                
                                # Spatial variability
                                spatial_features[f'{var}_neighbor_std'] = neighbor_values.std()
                                spatial_features[f'{var}_neighbor_mean'] = mean_neighbor
                                spatial_features[f'{var}_neighbor_max'] = neighbor_values.max()
                                spatial_features[f'{var}_neighbor_min'] = neighbor_values.min()
                            else:
                                spatial_features[f'{var}_spatial_gradient'] = 0
                                spatial_features[f'{var}_neighbor_std'] = 0
                                spatial_features[f'{var}_neighbor_mean'] = current_value
                                spatial_features[f'{var}_neighbor_max'] = current_value
                                spatial_features[f'{var}_neighbor_min'] = current_value
                        else:
                            # No neighbors - use current value
                            spatial_features[f'{var}_spatial_gradient'] = 0
                            spatial_features[f'{var}_neighbor_std'] = 0
                            spatial_features[f'{var}_neighbor_mean'] = current_value
                            spatial_features[f'{var}_neighbor_max'] = current_value
                            spatial_features[f'{var}_neighbor_min'] = current_value
                
                # Bay region features
                bay_region = cell.bay_region or 'Open_Water'
                spatial_features['bay_region'] = bay_region
                
                # Distance-based features
                spatial_features['distance_to_coast'] = cell.distance_to_coast_km or 0
                spatial_features['cell_area_km2'] = cell.area_km2
                
                date_features.append(spatial_features)
            
            spatial_features_list.extend(date_features)
        
        # Create enhanced DataFrame
        if spatial_features_list:
            spatial_df = pd.DataFrame(spatial_features_list)
            
            # Merge with original data
            merge_cols = ['date', 'grid_cell_id']
            enhanced_df = enhanced_df.merge(
                spatial_df, on=merge_cols, how='left', suffixes=('', '_spatial')
            )
            
            # Remove duplicate columns
            enhanced_df = enhanced_df.loc[:, ~enhanced_df.columns.duplicated()]
        
        logger.info(f"✅ Spatial features created: {enhanced_df.shape[1]} total columns")
        
        return enhanced_df
    
    def integrate_with_hab_labels(self, 
                                grid_env_df: pd.DataFrame,
                                hab_labels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate environmental data with HAB labels for model training
        
        Args:
            grid_env_df: Environmental data mapped to grid cells
            hab_labels_df: HAB labels for grid cells and dates
            
        Returns:
            Integrated DataFrame ready for model training
        """
        logger.info("🔗 Integrating environmental data with HAB labels (memory-optimized)...")

        # Ensure key dtypes align and are efficient
        merge_cols = ['date', 'grid_cell_id']
        grid_env_df = grid_env_df.copy()
        hab_labels_df = hab_labels_df.copy()

        # Standardize date dtype
        if grid_env_df['date'].dtype != 'datetime64[ns]':
            grid_env_df['date'] = pd.to_datetime(grid_env_df['date'])
        if hab_labels_df['date'].dtype != 'datetime64[ns]':
            hab_labels_df['date'] = pd.to_datetime(hab_labels_df['date'])

        # Reduce label columns to only what's needed
        keep_label_cols = [c for c in ['date', 'grid_cell_id', 'hab_occurrence'] if c in hab_labels_df.columns]
        hab_labels_slim = hab_labels_df[keep_label_cols]

        # Filter labels to only relevant keys present in environmental data
        unique_keys = grid_env_df[merge_cols].drop_duplicates()
        # Use merge to filter efficiently without exploding memory
        hab_labels_filtered = unique_keys.merge(hab_labels_slim, on=merge_cols, how='left', sort=False)

        # Merge with environmental data using LEFT join on the smaller left table (grid_env_df)
        integrated_df = grid_env_df.merge(
            hab_labels_filtered, on=merge_cols, how='left', sort=False, copy=False
        )

        # Fill missing labels with 0 and downcast
        if 'hab_occurrence' in integrated_df.columns:
            integrated_df['hab_occurrence'] = (
                integrated_df['hab_occurrence']
                .fillna(0)
                .astype('uint8')
            )

        # Downcast float columns to float32 to reduce memory footprint
        float_cols = integrated_df.select_dtypes(include=['float64']).columns.tolist()
        if float_cols:
            integrated_df[float_cols] = integrated_df[float_cols].astype('float32')

        logger.info("✅ Integration complete (memory-optimized):")
        logger.info(f"   HAB labels (original): {len(hab_labels_df):,} records")
        logger.info(f"   HAB labels (filtered): {len(hab_labels_filtered):,} records")
        logger.info(f"   Environmental data: {len(grid_env_df):,} records")
        logger.info(f"   Integrated: {len(integrated_df):,} records")

        # Show class distribution in integrated data
        if 'hab_occurrence' in integrated_df.columns:
            positive_records = (integrated_df['hab_occurrence'] == 1).sum()
            total_records = len(integrated_df)
            logger.info(f"   HAB positive: {positive_records:,} ({positive_records/total_records*100:.1f}%)")
            logger.info(f"   HAB negative: {total_records-positive_records:,} ({(total_records-positive_records)/total_records*100:.1f}%)")

        return integrated_df
    
    def get_feature_importance_by_category(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """
        Categorize features for analysis and interpretation
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary of features by category
        """
        categorized_features = {
            'physical_base': [],
            'physical_spatial': [],
            'optical_base': [],
            'optical_spatial': [],
            'water_quality_base': [],
            'water_quality_spatial': [],
            'meteorological_base': [],
            'meteorological_spatial': [],
            'spatial_location': [],
            'hab_labels': []
        }
        
        for feature in feature_names:
            categorized = False
            
            # Check base environmental variables
            for category, vars_list in self.variable_categories.items():
                for var in vars_list:
                    if feature == var:
                        categorized_features[f'{category}_base'].append(feature)
                        categorized = True
                        break
                    elif feature.startswith(var + '_'):
                        categorized_features[f'{category}_spatial'].append(feature)
                        categorized = True
                        break
                if categorized:
                    break
            
            # Check spatial/location features
            if not categorized:
                if any(keyword in feature for keyword in ['bay_region', 'distance', 'area', 'lat', 'lon']):
                    categorized_features['spatial_location'].append(feature)
                elif any(keyword in feature for keyword in ['hab_', 'grid_cell']):
                    categorized_features['hab_labels'].append(feature)
        
        # Remove empty categories
        return {k: v for k, v in categorized_features.items() if v}


def create_training_ready_dataset(hab_csv_path: str,
                                environmental_csv_path: str,
                                output_dir: str,
                                grid_resolution: float = 0.05,
                                point_coordinates: Optional[Tuple[float, float]] = None) -> str:
    """
    Create a complete training-ready dataset with spatial-temporal features
    
    Args:
        hab_csv_path: Path to HAB events CSV
        environmental_csv_path: Path to environmental data CSV  
        output_dir: Output directory for processed data
        grid_resolution: Spatial grid resolution in degrees
        point_coordinates: Single point coordinates if environmental data is from one location
        
    Returns:
        Path to the created training dataset
    """
    logger.info("🚀 Creating spatial-temporal training dataset...")
    
    # Import required classes
    from .spatial_temporal_grid import SetoInlandSeaSpatialGrid, SpatialHABEventProcessor
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create spatial grid
    logger.info("1️⃣ Creating spatial grid...")
    spatial_grid = SetoInlandSeaSpatialGrid(grid_resolution=grid_resolution)
    
    # Save grid for reference
    grid_file = output_path / "spatial_grid.json"
    spatial_grid.save_grid(str(grid_file))
    
    # Step 2: Process HAB events
    logger.info("2️⃣ Processing HAB events...")
    hab_processor = SpatialHABEventProcessor(spatial_grid)
    spatial_events = hab_processor.process_hab_events_from_csv(hab_csv_path)
    
    # Step 3: Create spatial-temporal labels
    logger.info("3️⃣ Creating spatial-temporal labels...")
    
    # Get date range from environmental data
    env_df = pd.read_csv(environmental_csv_path)
    env_df['date'] = pd.to_datetime(env_df['date'])
    start_date = env_df['date'].min().strftime('%Y-%m-%d')
    end_date = env_df['date'].max().strftime('%Y-%m-%d')
    
    hab_labels = hab_processor.create_spatial_temporal_labels(
        spatial_events, start_date, end_date
    )
    
    # Step 4: Map environmental data to grid
    logger.info("4️⃣ Mapping environmental data to grid...")
    env_mapper = SpatialEnvironmentalMapper(spatial_grid)
    grid_env_data = env_mapper.map_environmental_data_to_grid(
        env_df, point_coordinates=point_coordinates
    )
    
    # Step 5: Create spatial features
    logger.info("5️⃣ Creating spatial features...")
    enhanced_env_data = env_mapper.create_spatial_features(grid_env_data)
    
    # Step 6: Integrate with HAB labels
    logger.info("6️⃣ Integrating environmental data with HAB labels...")
    training_dataset = env_mapper.integrate_with_hab_labels(
        enhanced_env_data, hab_labels
    )
    
    # Add compatibility alias for label column expected by helper scripts
    if 'hab_occurrence' in training_dataset.columns:
        training_dataset['has_hab'] = training_dataset['hab_occurrence'].astype('uint8')

    # Step 7: Save training dataset (CSV + Parquet)
    training_csv = output_path / "spatial_temporal_training_dataset.csv"
    training_parquet = output_path / "spatial_temporal_training_dataset.parquet"

    # Persist
    training_dataset.to_csv(training_csv, index=False)
    try:
        training_dataset.to_parquet(training_parquet, index=False)
    except Exception as e:
        logger.warning(f"Parquet save failed ({e}), CSV will be used")
        training_parquet = None
    
    # Save metadata
    metadata = {
        'creation_date': datetime.now().isoformat(),
        'grid_resolution': grid_resolution,
        'point_coordinates': point_coordinates,
        'total_records': int(len(training_dataset)),
        'positive_samples': (training_dataset['hab_occurrence'] == 1).sum(),
        'negative_samples': (training_dataset['hab_occurrence'] == 0).sum(),
        'date_range': {
            'start': start_date,
            'end': end_date
        },
        'grid_cells': spatial_grid.get_grid_summary(),
        'feature_categories': env_mapper.get_feature_importance_by_category(training_dataset.columns.tolist())
    }
    
    # Helper: JSON default to handle numpy/pandas types
    def _json_default(o):
        try:
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (pd.Timestamp,)):
                return o.isoformat()
        except Exception:
            pass
        return str(o)

    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=_json_default)
    
    logger.info(f"✅ Spatial-temporal training dataset created!")
    if training_parquet is not None:
        logger.info(f"   Training data (parquet): {training_parquet}")
    logger.info(f"   Training data (csv): {training_csv}")
    logger.info(f"   Metadata: {metadata_file}")
    logger.info(f"   Grid definition: {grid_file}")

    # Prefer parquet if available
    return str(training_parquet if training_parquet is not None else training_csv)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Example: Create training dataset
    # This would be called with your actual file paths
    """
    training_file = create_training_ready_dataset(
        hab_csv_path="test_data/HAB_2017_2023.csv",
        environmental_csv_path="data/csv/historical/historical_2017_2024_master_wide.csv",
        output_dir="cache/spatial_temporal_training",
        grid_resolution=0.05,
        point_coordinates=(34.5, 133.5)  # Center of Seto Inland Sea
    )
    print(f"Training dataset created: {training_file}")
    """
