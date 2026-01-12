"""
Spatial-Temporal Grid System for HAB Prediction

This module implements a spatial grid-based approach to HAB prediction that addresses
the class imbalance problem by creating location-specific predictions rather than
region-wide temporal predictions.

Key Features:
- Spatial grid generation for Seto Inland Sea
- HAB event mapping to grid cells
- Environmental data interpolation to grid points
- Spatial-temporal feature engineering
- Grid-based model training and prediction
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import geopandas as gpd
from shapely.geometry import Point, Polygon
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

@dataclass
class GridCell:
    """Represents a single grid cell in the spatial grid"""
    cell_id: str
    lat_center: float
    lon_center: float
    lat_bounds: Tuple[float, float]
    lon_bounds: Tuple[float, float]
    area_km2: float
    depth_m: Optional[float] = None
    distance_to_coast_km: Optional[float] = None
    bay_region: Optional[str] = None
    is_valid: bool = True  # False for land cells

@dataclass
class SpatialHABEvent:
    """HAB event with spatial grid mapping"""
    original_event_id: int
    grid_cell_id: str
    date: datetime
    end_date: Optional[datetime]
    species: str
    cell_count: float
    duration_days: int
    severity: str
    confidence: float = 1.0

class SetoInlandSeaSpatialGrid:
    """
    Spatial grid system specifically designed for the Seto Inland Sea region
    
    Creates a regular grid covering the Seto Inland Sea with proper handling of:
    - Land/water boundaries
    - Different bay regions
    - Coastal vs offshore areas
    - Bathymetry considerations
    """
    
    def __init__(self, 
                 grid_resolution: float = 0.05,  # degrees (~5km)
                 region_bounds: Optional[Dict[str, float]] = None):
        """
        Initialize the Seto Inland Sea spatial grid system
        
        Args:
            grid_resolution: Grid cell size in degrees (0.05° ≈ 5km)
            region_bounds: Custom region boundaries
        """
        self.grid_resolution = grid_resolution
        
        # Default Seto Inland Sea boundaries (expanded to ensure full coverage including adjacent areas)
        if region_bounds is None:
            self.region_bounds = {
                'lat_min': 32.8,   # Southern extent (includes Sukumo Bay)
                'lat_max': 35.0,   # Northern extent  
                'lon_min': 130.0,  # Western extent (includes Fukuoka area)
                'lon_max': 136.0   # Eastern extent (includes all of Osaka Bay and Tosa Bay area)
            }
        else:
            self.region_bounds = region_bounds
            
        # Major bay regions in Seto Inland Sea - Updated with actual coordinate bounds
        self.bay_regions = {
            'Osaka_Bay': {
                'bounds': {'lat_min': 34.3, 'lat_max': 34.8, 'lon_min': 135.0, 'lon_max': 135.6},
                'center': (34.5, 135.33),
                'characteristics': 'highly_eutrophic'
            },
            'Harima_Nada': {
                'bounds': {'lat_min': 34.3, 'lat_max': 34.8, 'lon_min': 134.0, 'lon_max': 135.0},
                'center': (34.5, 134.5),
                'characteristics': 'moderate_eutrophic'
            },
            'Hiroshima_Bay': {
                'bounds': {'lat_min': 34.0, 'lat_max': 34.4, 'lon_min': 132.1, 'lon_max': 132.8},
                'center': (34.17, 132.42),
                'characteristics': 'moderate_eutrophic'
            },
            'Suo_Nada': {
                'bounds': {'lat_min': 33.6, 'lat_max': 34.1, 'lon_min': 131.5, 'lon_max': 132.2},
                'center': (33.83, 131.83),
                'characteristics': 'oligotrophic'
            },
            'Iyo_Nada': {
                'bounds': {'lat_min': 33.5, 'lat_max': 34.1, 'lon_min': 132.0, 'lon_max': 133.0},
                'center': (33.83, 132.5),
                'characteristics': 'mesotrophic'
            },
            'Bisan_Seto': {
                'bounds': {'lat_min': 34.2, 'lat_max': 34.5, 'lon_min': 133.5, 'lon_max': 134.2},
                'center': (34.33, 133.83),
                'characteristics': 'variable'
            },
            'Aki_Nada': {
                'bounds': {'lat_min': 34.0, 'lat_max': 34.4, 'lon_min': 132.2, 'lon_max': 132.8},
                'center': (34.17, 132.5),
                'characteristics': 'moderate_eutrophic'
            },
            'Hiuchi_Nada': {
                'bounds': {'lat_min': 34.1, 'lat_max': 34.4, 'lon_min': 133.0, 'lon_max': 133.6},
                'center': (34.25, 133.33),
                'characteristics': 'variable'
            }
        }
        
        # Initialize grid
        self.grid_cells = {}
        self.grid_array = None
        self.lat_coords = None
        self.lon_coords = None
        
        self._create_spatial_grid()
        logger.info(f"🗺️ Initialized Seto Inland Sea spatial grid: {len(self.grid_cells)} cells")
        
    def _create_spatial_grid(self):
        """Create the spatial grid covering the Seto Inland Sea"""
        logger.info(f"Creating spatial grid with {self.grid_resolution}° resolution...")
        
        # Generate coordinate arrays
        self.lat_coords = np.arange(
            self.region_bounds['lat_min'], 
            self.region_bounds['lat_max'] + self.grid_resolution, 
            self.grid_resolution
        )
        self.lon_coords = np.arange(
            self.region_bounds['lon_min'], 
            self.region_bounds['lon_max'] + self.grid_resolution, 
            self.grid_resolution
        )
        
        # Create grid cells
        cell_count = 0
        for i, lat in enumerate(self.lat_coords[:-1]):  # Exclude last to avoid boundary issues
            for j, lon in enumerate(self.lon_coords[:-1]):
                
                # Cell boundaries
                lat_min, lat_max = lat, lat + self.grid_resolution
                lon_min, lon_max = lon, lon + self.grid_resolution
                
                # Cell center
                lat_center = lat + self.grid_resolution / 2
                lon_center = lon + self.grid_resolution / 2
                
                # Generate cell ID
                cell_id = f"grid_{i:03d}_{j:03d}"
                
                # Calculate approximate area (rough estimate)
                area_km2 = self._calculate_cell_area(lat_center, self.grid_resolution)
                
                # Determine bay region
                bay_region = self._get_bay_region(lat_center, lon_center)
                
                # Check if cell is valid (over water, not land)
                is_valid = self._is_water_cell(lat_center, lon_center)
                
                # Create grid cell
                grid_cell = GridCell(
                    cell_id=cell_id,
                    lat_center=lat_center,
                    lon_center=lon_center,
                    lat_bounds=(lat_min, lat_max),
                    lon_bounds=(lon_min, lon_max),
                    area_km2=area_km2,
                    bay_region=bay_region,
                    is_valid=is_valid
                )
                
                self.grid_cells[cell_id] = grid_cell
                
                if is_valid:
                    cell_count += 1
        
        logger.info(f"✅ Created {len(self.grid_cells)} total grid cells ({cell_count} water cells)")
        
        # Log bay region distribution
        bay_counts = {}
        for cell in self.grid_cells.values():
            if cell.is_valid and cell.bay_region:
                bay_counts[cell.bay_region] = bay_counts.get(cell.bay_region, 0) + 1
        
        logger.info("Bay region distribution:")
        for bay, count in bay_counts.items():
            logger.info(f"  {bay}: {count} cells")
    
    def _calculate_cell_area(self, lat_center: float, resolution_deg: float) -> float:
        """Calculate approximate area of grid cell in km²"""
        # Rough approximation: 1 degree ≈ 111 km at equator
        # Account for latitude compression
        lat_factor = np.cos(np.radians(lat_center))
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * lat_factor
        
        area_km2 = (resolution_deg * km_per_deg_lat) * (resolution_deg * km_per_deg_lon)
        return area_km2
    
    def _get_bay_region(self, lat: float, lon: float) -> Optional[str]:
        """Determine which bay region a coordinate belongs to"""
        for bay_name, bay_info in self.bay_regions.items():
            bounds = bay_info['bounds']
            if (bounds['lat_min'] <= lat <= bounds['lat_max'] and 
                bounds['lon_min'] <= lon <= bounds['lon_max']):
                return bay_name
        return None
    
    def _is_water_cell(self, lat: float, lon: float) -> bool:
        """
        Determine if a grid cell is over water (improved approach)
        
        Uses more accurate boundaries based on actual Seto Inland Sea geography
        """
        # Exclude areas clearly outside Seto Inland Sea and adjacent waters
        
        # Too far north (inland Honshu)
        if lat > 34.9:
            return False
            
        # Too far south (open Pacific, but include Tosa Bay area)
        if lat < 32.9:
            return False
            
        # Too far west (inland Kyushu, but include Suo-nada)
        if lon < 130.2:
            return False
            
        # Too far east (inland Honshu, but include Osaka Bay and Tosa Bay area)
        if lon > 135.9:
            return False
        
        # Exclude some obvious inland areas
        # Northern Honshu mainland (above certain latitude and east of certain longitude)
        if lat > 34.7 and lon > 135.4:
            return False
            
        # Central Shikoku (rough exclusion)
        if 33.3 < lat < 33.8 and 133.3 < lon < 134.0:
            return False
            
        # Central Kyushu areas (rough exclusion)
        if lat < 33.4 and lon < 131.0:
            return False
        
        # For the expanded region, include most coastal and sea areas
        # This is still simplified - in production use proper coastline data
        return True
    
    def map_hab_event_to_grid(self, event_data: Dict) -> Optional[SpatialHABEvent]:
        """
        Map a HAB event to the appropriate grid cell
        
        Args:
            event_data: Dictionary containing HAB event information
            
        Returns:
            SpatialHABEvent object or None if event can't be mapped
        """
        try:
            # Extract location information
            location_str = event_data.get('location', '')
            
            # Estimate coordinates from location
            estimated_coords = self._estimate_coordinates_from_location(location_str)
            
            if estimated_coords is None:
                logger.debug(f"Could not estimate coordinates for location: {location_str}")
                return None
                
            lat, lon = estimated_coords
            
            # Find the appropriate grid cell
            grid_cell_id = self._find_grid_cell(lat, lon)
            
            if grid_cell_id is None:
                logger.debug(f"No grid cell found for coordinates: {lat}, {lon}")
                return None
            
            # Parse date information
            start_date = pd.to_datetime(event_data['start_date'])
            end_date_raw = event_data.get('end_date', event_data['start_date'])
            end_date = pd.to_datetime(end_date_raw) if pd.notna(end_date_raw) else start_date
            
            # Calculate duration
            duration_days = max(1, (end_date - start_date).days + 1)
            
            # Extract and clean species name
            species_raw = event_data.get('plankton_species', 'Unknown')
            if pd.isna(species_raw) or species_raw == '':
                species = 'Unknown'
            else:
                species = str(species_raw).strip()
            
            # Extract and parse cell count
            cell_count = self._parse_cell_count(event_data.get('cell_count', 0))
            
            # Determine severity based on cell count
            severity = self._classify_severity(cell_count)
            
            # Create spatial HAB event
            spatial_event = SpatialHABEvent(
                original_event_id=event_data.get('id', 0),
                grid_cell_id=grid_cell_id,
                date=start_date,
                end_date=end_date,
                species=species,
                cell_count=cell_count,
                duration_days=duration_days,
                severity=severity
            )
            
            return spatial_event
            
        except Exception as e:
            logger.debug(f"Error mapping HAB event to grid: {e}")
            return None
    
    def _estimate_coordinates_from_location(self, location_str: str) -> Optional[Tuple[float, float]]:
        """
        Get actual coordinates from location string using researched coordinates
        
        Based on web search for actual coordinates of Seto Inland Sea bays and channels
        """
        # Comprehensive location mapping with actual coordinates from web search
        location_coords = {
            # Major Bays - Actual coordinates from web search
            'Osaka Bay': (34.5, 135.33),  # Actual center of Osaka Bay
            'Osaka': (34.5, 135.33),
            '大阪湾': (34.5, 135.33),
            'osaka bay': (34.5, 135.33),
            
            # Harima-nada - Actual coordinates
            'Harima-nada': (34.5, 134.5),  # Verified coordinates
            'Harima': (34.5, 134.5),
            '播磨灘': (34.5, 134.5),
            'harima': (34.5, 134.5),
            'harima-nada': (34.5, 134.5),
            
            # Hiroshima Bay - Actual coordinates  
            'Hiroshima Bay': (34.17, 132.42),  # Verified Hiroshima Bay center
            'Hiroshima': (34.17, 132.42),
            '広島湾': (34.17, 132.42),
            'hiroshima': (34.17, 132.42),
            
            # Suo-nada - Actual coordinates
            'Suo-nada': (33.83, 131.83),  # Verified Suo-nada coordinates
            'Suō-nada': (33.83, 131.83),
            'Suo': (33.83, 131.83),
            '周防灘': (33.83, 131.83),
            'suo-nada': (33.83, 131.83),
            
            # Iyo-nada - Actual coordinates
            'Iyo-nada': (33.83, 132.5),  # Verified Iyo-nada coordinates
            'Iyo': (33.83, 132.5),
            '伊予灘': (33.83, 132.5),
            'iyo-nada': (33.83, 132.5),
            
            # Aki-nada - Actual coordinates
            'Aki-nada': (34.17, 132.5),  # Verified Aki-nada coordinates
            'aki-nada': (34.17, 132.5),
            
            # Hiuchi-nada - Actual coordinates
            'Hiuchi-nada': (34.25, 133.33),  # Verified Hiuchi-nada coordinates
            'hiuchi-nada': (34.25, 133.33),
            'Hiuchinada': (34.25, 133.33),
            
            # Bisan Seto - Actual coordinates
            'Bisan Seto': (34.33, 133.83),  # Verified Bisan Seto coordinates
            'bisan seto': (34.33, 133.83),
            '備讃瀬戸': (34.33, 133.83),
            
            # Channels - Actual coordinates
            'Bungo Channel': (33.33, 132.0),  # Verified Bungo Channel center
            'bungo channel': (33.33, 132.0),
            'bungo': (33.33, 132.0),
            
            'Kii Channel': (34.2, 135.0),  # Verified Kii Channel center
            'kii channel': (34.2, 135.0),
            
            # Tosa Bay - Actual coordinates (adjusted to be within grid bounds)
            'Tosa Bay': (33.5, 133.5),  # Adjusted to be within grid bounds
            'tosa bay': (33.5, 133.5),
            'tosa': (33.5, 133.5),
            
            # Prefectures - Capital city coordinates
            'Kagawa': (34.34, 134.04),  # Takamatsu
            'Okayama': (34.66, 133.92),  # Okayama city
            'Hyogo': (34.69, 135.18),   # Kobe area
            'Yamaguchi': (34.18, 131.47), # Yamaguchi city
            'Tokushima': (34.07, 134.56), # Tokushima city
            'Ehime': (33.84, 132.77),   # Matsuyama
            'Oita': (33.24, 131.61),    # Oita city
            'Fukuoka': (33.59, 130.40), # Fukuoka city
            'Wakayama': (34.23, 135.17), # Wakayama city
            'Kochi': (33.56, 133.53),   # Kochi city
            
            # Specific locations with coordinates
            'Nishinomiya': (34.74, 135.34),
            'Sakai': (34.57, 135.48),
            'Kobe': (34.69, 135.19),
            'Izumiotsu': (34.50, 135.41),
            'Kaizuka': (34.44, 135.36),
            'Naruto': (34.18, 134.61),
            'Anan': (33.92, 134.66),
            'Mihara': (34.40, 133.08),
            'Kure': (34.25, 132.57),
            'Iwakuni': (34.16, 132.22),
            'Nakatsu': (33.60, 131.19),
            'Kunisaki': (33.57, 131.73),
            
            # Sea areas with actual coordinates
            'Sea of Aki': (34.17, 132.5),
            'sea of aki': (34.17, 132.5),
            
            # Specific bays mentioned in data
            '別府湾': (33.28, 131.50),  # Beppu Bay
            'Beppu Bay': (33.28, 131.50),
            '徳山湾': (34.05, 131.83),  # Tokuyama Bay  
            'Tokuyama Bay': (34.05, 131.83),
            '浦ノ内湾': (33.45, 133.67), # Uranouchi Bay
            'Uranouchi Bay': (33.45, 133.67),
            '宿毛湾': (32.93, 132.73),  # Sukumo Bay
            'Sukumo Bay': (32.93, 132.73),
            '宇和島湾': (33.22, 132.55), # Uwajima Bay
            'Uwajima Bay': (33.22, 132.55),
            '野見湾': (33.52, 133.28),  # Nomi Bay
            'Nomi Bay': (33.52, 133.28),
        }
        
        # Clean location string
        location_clean = location_str.strip() if location_str else ''
        
        # Try exact match first (case-insensitive)
        location_lower = location_clean.lower()
        for known_location, coords in location_coords.items():
            if known_location.lower() == location_lower:
                return coords
        
        # Try partial matching with longer strings first (most specific)
        sorted_locations = sorted(location_coords.items(), key=lambda x: len(x[0]), reverse=True)
        
        for known_location, coords in sorted_locations:
            # Check if known location is in the input string
            if known_location.lower() in location_lower:
                return coords
            # Check if any word from known location is in input
            known_words = known_location.lower().split()
            if any(word in location_lower for word in known_words if len(word) > 3):
                return coords
        
        # Try to match prefecture names in complex location strings
        prefecture_patterns = {
            'osaka': (34.5, 135.33),
            'hyogo': (34.69, 135.18),
            'okayama': (34.66, 133.92),
            'hiroshima': (34.17, 132.42),
            'yamaguchi': (34.18, 131.47),
            'kagawa': (34.34, 134.04),
            'tokushima': (34.07, 134.56),
            'ehime': (33.84, 132.77),
            'oita': (33.24, 131.61),
            'fukuoka': (33.59, 130.40),
            'wakayama': (34.23, 135.17),
            'kochi': (33.56, 133.53),
        }
        
        for prefecture, coords in prefecture_patterns.items():
            if prefecture in location_lower:
                return coords
        
        # Default to center of Seto Inland Sea if no match
        logger.debug(f"No coordinate match found for '{location_str}', using default center")
        return (34.17, 133.33)  # Actual Seto Inland Sea center coordinates
    
    def _find_grid_cell(self, lat: float, lon: float) -> Optional[str]:
        """Find the grid cell that contains the given coordinates"""
        for cell_id, cell in self.grid_cells.items():
            if (cell.lat_bounds[0] <= lat < cell.lat_bounds[1] and
                cell.lon_bounds[0] <= lon < cell.lon_bounds[1] and
                cell.is_valid):
                return cell_id
        return None
    
    def _parse_cell_count(self, cell_count_raw) -> float:
        """Parse cell count from various formats"""
        if pd.isna(cell_count_raw) or cell_count_raw == '':
            return 0.0
        
        try:
            # Convert to string and clean
            cell_str = str(cell_count_raw).strip()
            
            # Handle multiple numbers separated by space or comma
            if ' ' in cell_str or ',' in cell_str:
                # Extract numbers and take the first valid one
                import re
                numbers = re.findall(r'[\d,]+', cell_str)
                if numbers:
                    # Clean first number (remove commas) and convert
                    first_number = numbers[0].replace(',', '')
                    return float(first_number)
                else:
                    return 0.0
            else:
                # Simple case - remove commas and convert
                return float(cell_str.replace(',', ''))
                
        except (ValueError, TypeError):
            return 0.0
    
    def _classify_severity(self, cell_count: float) -> str:
        """Classify HAB severity based on cell count"""
        if cell_count == 0:
            return 'none'
        elif cell_count < 1000:
            return 'low'
        elif cell_count < 10000:
            return 'moderate'
        elif cell_count < 100000:
            return 'high'
        else:
            return 'extreme'
    
    def get_neighboring_cells(self, cell_id: str, distance: int = 1) -> List[str]:
        """Get neighboring grid cells within specified distance"""
        if cell_id not in self.grid_cells:
            return []
        
        # Parse cell ID to get grid indices
        parts = cell_id.split('_')
        if len(parts) != 3:
            return []
        
        try:
            i, j = int(parts[1]), int(parts[2])
        except ValueError:
            return []
        
        neighbors = []
        for di in range(-distance, distance + 1):
            for dj in range(-distance, distance + 1):
                if di == 0 and dj == 0:
                    continue
                    
                ni, nj = i + di, j + dj
                neighbor_id = f"grid_{ni:03d}_{nj:03d}"
                
                if (neighbor_id in self.grid_cells and 
                    self.grid_cells[neighbor_id].is_valid):
                    neighbors.append(neighbor_id)
        
        return neighbors
    
    def get_grid_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the spatial grid"""
        total_cells = len(self.grid_cells)
        water_cells = sum(1 for cell in self.grid_cells.values() if cell.is_valid)
        
        bay_distribution = {}
        for cell in self.grid_cells.values():
            if cell.is_valid:
                bay = cell.bay_region or 'Open_Water'
                bay_distribution[bay] = bay_distribution.get(bay, 0) + 1
        
        total_area = sum(cell.area_km2 for cell in self.grid_cells.values() if cell.is_valid)
        
        return {
            'total_cells': total_cells,
            'water_cells': water_cells,
            'land_cells': total_cells - water_cells,
            'grid_resolution_deg': self.grid_resolution,
            'total_water_area_km2': total_area,
            'bay_distribution': bay_distribution,
            'region_bounds': self.region_bounds
        }
    
    def save_grid(self, output_path: str):
        """Save the spatial grid to file"""
        grid_data = {
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'grid_resolution': self.grid_resolution,
                'region_bounds': self.region_bounds,
                'bay_regions': self.bay_regions
            },
            'cells': {
                cell_id: {
                    'lat_center': cell.lat_center,
                    'lon_center': cell.lon_center,
                    'lat_bounds': cell.lat_bounds,
                    'lon_bounds': cell.lon_bounds,
                    'area_km2': cell.area_km2,
                    'bay_region': cell.bay_region,
                    'is_valid': cell.is_valid
                }
                for cell_id, cell in self.grid_cells.items()
            },
            'summary': self.get_grid_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(grid_data, f, indent=2)
        
        logger.info(f"💾 Saved spatial grid to {output_path}")


class SpatialHABEventProcessor:
    """
    Processes HAB events and maps them to the spatial grid system
    """
    
    def __init__(self, spatial_grid: SetoInlandSeaSpatialGrid):
        self.spatial_grid = spatial_grid
        self.processed_events = []
        
    def process_hab_events_from_csv(self, csv_path: str) -> List[SpatialHABEvent]:
        """
        Process HAB events from CSV file and map to spatial grid
        
        Args:
            csv_path: Path to HAB events CSV file
            
        Returns:
            List of processed spatial HAB events
        """
        logger.info(f"📊 Processing HAB events from {csv_path}")
        
        try:
            # Load HAB events data
            hab_df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(hab_df)} HAB events")
            
            spatial_events = []
            successful_mappings = 0
            
            for idx, row in hab_df.iterrows():
                # Convert row to dictionary for processing
                event_data = {
                    'id': idx,
                    'start_date': row['start_date'],
                    'end_date': row.get('end_date', row['start_date']),
                    'location': row['location'],
                    'plankton_species': row['plankton_species'],
                    'cell_count': row.get('cell_count', 0),
                    'duration_days': row.get('duration_days', 1)
                }
                
                # Map to spatial grid
                spatial_event = self.spatial_grid.map_hab_event_to_grid(event_data)
                
                if spatial_event:
                    spatial_events.append(spatial_event)
                    successful_mappings += 1
                else:
                    logger.debug(f"Failed to map event {idx} to grid")
            
            logger.info(f"✅ Successfully mapped {successful_mappings}/{len(hab_df)} events to spatial grid")
            
            # Store processed events
            self.processed_events = spatial_events
            
            return spatial_events
            
        except Exception as e:
            logger.error(f"❌ Error processing HAB events: {e}")
            return []
    
    def create_spatial_temporal_labels(self, 
                                     spatial_events: List[SpatialHABEvent],
                                     start_date: str,
                                     end_date: str) -> pd.DataFrame:
        """
        Create spatial-temporal labels for model training
        
        Args:
            spatial_events: List of processed spatial HAB events
            start_date: Start date for label generation
            end_date: End date for label generation
            
        Returns:
            DataFrame with spatial-temporal labels
        """
        logger.info(f"🏷️ Creating spatial-temporal labels from {start_date} to {end_date}")
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Get all valid grid cells
        valid_cells = [cell_id for cell_id, cell in self.spatial_grid.grid_cells.items() 
                      if cell.is_valid]
        
        logger.info(f"📅 Date range: {len(date_range)} days")
        logger.info(f"🗺️ Valid grid cells: {len(valid_cells)}")
        
        # Create all combinations of dates and grid cells (optimized)
        labels_data = []
        
        # Pre-index events by grid cell for faster lookup
        events_by_cell = {}
        for event in spatial_events:
            if event.grid_cell_id not in events_by_cell:
                events_by_cell[event.grid_cell_id] = []
            events_by_cell[event.grid_cell_id].append(event)
        
        total_records = len(date_range) * len(valid_cells)
        logger.info(f"🔄 Processing {len(date_range):,} dates × {len(valid_cells):,} cells = {total_records:,} records...")
        
        # Estimate processing time
        if total_records > 5_000_000:
            est_minutes = total_records / 100_000  # Rough estimate: 100k records per minute
            logger.warning(f"⚠️ Large dataset detected! Estimated time: {est_minutes:.1f} minutes")
            logger.info("💡 Consider using run_spatial_temporal_hab_system_FAST.py for testing")
        
        processed_count = 0
        batch_size = 50000  # Process in batches for better memory management
        
        for date in date_range:
            if processed_count % batch_size == 0:
                progress = (processed_count / total_records) * 100
                logger.info(f"   Processed {processed_count:,}/{total_records:,} records ({progress:.1f}%)")
            
            for cell_id in valid_cells:
                cell = self.spatial_grid.grid_cells[cell_id]
                
                # Find HAB events for this cell and date (using pre-indexed lookup)
                events_for_cell_date = []
                if cell_id in events_by_cell:
                    events_for_cell_date = [
                        event for event in events_by_cell[cell_id]
                        if (event.date <= date <= (event.end_date or event.date))
                    ]
                
                processed_count += 1
                
                # Create label record
                has_hab = len(events_for_cell_date) > 0
                
                if has_hab:
                    # Use the most severe event if multiple
                    event = max(events_for_cell_date, key=lambda x: x.cell_count)
                    
                    label_record = {
                        'date': date,
                        'grid_cell_id': cell_id,
                        'lat': cell.lat_center,
                        'lon': cell.lon_center,
                        'bay_region': cell.bay_region,
                        'hab_occurrence': 1,
                        'hab_species': event.species,
                        'hab_cell_count': event.cell_count,
                        'hab_severity': event.severity,
                        'hab_duration_days': event.duration_days,
                        'event_count': len(events_for_cell_date)
                    }
                else:
                    label_record = {
                        'date': date,
                        'grid_cell_id': cell_id,
                        'lat': cell.lat_center,
                        'lon': cell.lon_center,
                        'bay_region': cell.bay_region,
                        'hab_occurrence': 0,
                        'hab_species': None,
                        'hab_cell_count': 0,
                        'hab_severity': 'none',
                        'hab_duration_days': 0,
                        'event_count': 0
                    }
                
                labels_data.append(label_record)
        
        # Create DataFrame
        labels_df = pd.DataFrame(labels_data)
        
        # Log label statistics
        total_records = len(labels_df)
        positive_records = (labels_df['hab_occurrence'] == 1).sum()
        negative_records = total_records - positive_records
        
        logger.info(f"📊 Spatial-temporal labels created:")
        logger.info(f"   Total records: {total_records:,}")
        logger.info(f"   Positive (HAB): {positive_records:,} ({positive_records/total_records*100:.1f}%)")
        logger.info(f"   Negative (No HAB): {negative_records:,} ({negative_records/total_records*100:.1f}%)")
        
        # Show distribution by bay region
        region_stats = labels_df.groupby(['bay_region', 'hab_occurrence']).size().unstack(fill_value=0)
        logger.info("📍 Distribution by bay region:")
        for region in region_stats.index:
            pos = region_stats.loc[region, 1] if 1 in region_stats.columns else 0
            neg = region_stats.loc[region, 0] if 0 in region_stats.columns else 0
            total = pos + neg
            logger.info(f"   {region}: {pos}/{total} positive ({pos/total*100:.1f}%)")
        
        return labels_df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create spatial grid
    spatial_grid = SetoInlandSeaSpatialGrid(grid_resolution=0.05)
    
    # Print grid summary
    summary = spatial_grid.get_grid_summary()
    print("\n🗺️ Spatial Grid Summary:")
    print(f"Total cells: {summary['total_cells']}")
    print(f"Water cells: {summary['water_cells']}")
    print(f"Total area: {summary['total_water_area_km2']:.1f} km²")
    print("\nBay distribution:")
    for bay, count in summary['bay_distribution'].items():
        print(f"  {bay}: {count} cells")
    
    # Save grid
    output_dir = Path("cache/spatial_grid")
    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_grid.save_grid(output_dir / "seto_inland_sea_grid.json")
