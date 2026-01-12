"""
Task 2.2: HAB Event Data Integration for Historical Analysis

This module integrates historical HAB event data from Japanese monitoring stations
with environmental data for comprehensive HAB forecasting model training.

Features:
- Integrates HAB event data from multiple Japanese monitoring sources
- Creates species-specific HAB occurrence datasets
- Links environmental conditions with HAB events
- Generates labels for supervised machine learning
- Supports temporal and spatial matching with environmental data
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HABEvent:
    """Represents a single HAB event"""
    event_id: str
    date: datetime
    latitude: float
    longitude: float
    species: str
    cell_density: float  # cells/mL
    toxin_level: Optional[float]  # µg/L
    severity: str  # 'low', 'medium', 'high', 'extreme'
    water_body: str
    monitoring_station: str
    data_source: str
    confidence_level: str  # 'confirmed', 'probable', 'suspected'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'event_id': self.event_id,
            'date': self.date.isoformat(),
            'latitude': self.latitude,
            'longitude': self.longitude,
            'species': self.species,
            'cell_density': self.cell_density,
            'toxin_level': self.toxin_level,
            'severity': self.severity,
            'water_body': self.water_body,
            'monitoring_station': self.monitoring_station,
            'data_source': self.data_source,
            'confidence_level': self.confidence_level
        }

class HABEventIntegrator:
    """
    Integrates HAB event data with environmental data for model training
    
    Data Sources:
    - Japan Fisheries Research and Education Agency (FRA)
    - Kagawa Prefecture Red Tide Monitoring
    - Seto Inland Sea Environmental Conservation Association
    - Local fisheries cooperatives monitoring data
    - Research literature HAB event databases
    """
    
    def __init__(self, data_dir: str = "data/hab_events"):
        """Initialize HAB event integrator"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Seto Inland Sea monitoring stations (example locations)
        self.monitoring_stations = {
            "SETOUCHI_01": {"lat": 34.3, "lon": 133.2, "name": "Hiroshima Bay"},
            "SETOUCHI_02": {"lat": 34.4, "lon": 133.8, "name": "Bingo-Nada"},
            "SETOUCHI_03": {"lat": 34.5, "lon": 134.0, "name": "Harima-Nada"}, 
            "SETOUCHI_04": {"lat": 34.2, "lon": 134.5, "name": "Osaka Bay"},
            "SETOUCHI_05": {"lat": 34.3, "lon": 134.8, "name": "Kii Channel"},
            "SETOUCHI_06": {"lat": 34.1, "lon": 133.5, "name": "Suo-Nada"},
            "SETOUCHI_07": {"lat": 34.4, "lon": 133.0, "name": "Iyo-Nada"},
            "KAGAWA_01": {"lat": 34.2, "lon": 133.9, "name": "Kagawa Coast"},
            "OKAYAMA_01": {"lat": 34.5, "lon": 133.7, "name": "Okayama Coast"},
            "HYOGO_01": {"lat": 34.6, "lon": 134.3, "name": "Hyogo Coast"}
        }
        
        # HAB species in Seto Inland Sea (from project species models)
        self.hab_species_mapping = {
            "Karenia mikimotoi": "karenia_mikimotoi",
            "Heterosigma akashiwo": "heterosigma_akashiwo", 
            "Noctiluca scintillans": "noctiluca_scintillans",
            "Chattonella antiqua": "chattonella_antiqua",
            "Chattonella marina": "chattonella_marina",
            "Cochlodinium polykrikoides": "cochlodinium_polykrikoides",
            "Heterocapsa circularisquama": "heterocapsa_circularisquama",
            "Prorocentrum minimum": "prorocentrum_minimum",
            "Alexandrium tamarense": "alexandrium_tamarense",
            "Skeletonema costatum": "skeletonema_costatum"
        }
        
        # Cell density thresholds for severity classification
        self.severity_thresholds = {
            "karenia_mikimotoi": {"low": 100, "medium": 1000, "high": 5000, "extreme": 20000},
            "heterosigma_akashiwo": {"low": 1000, "medium": 10000, "high": 50000, "extreme": 200000},
            "noctiluca_scintillans": {"low": 10, "medium": 100, "high": 500, "extreme": 2000},
            "chattonella_antiqua": {"low": 500, "medium": 5000, "high": 20000, "extreme": 100000},
            "cochlodinium_polykrikoides": {"low": 1000, "medium": 10000, "high": 50000, "extreme": 200000}
        }
        
        logger.info(f"HAB event integrator initialized with {len(self.monitoring_stations)} stations")
    
    def generate_synthetic_hab_events(self, start_date: str, end_date: str, num_events: int = 500) -> List[HABEvent]:
        """
        Generate synthetic HAB events for model development and testing
        
        Note: In production, this would be replaced with real data from:
        - Japanese Fisheries Research Agency databases
        - Prefecture monitoring systems
        - Research literature
        - Fisheries cooperative reports
        """
        logger.info(f"Generating {num_events} synthetic HAB events for {start_date} to {end_date}")
        
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        events = []
        
        for i in range(num_events):
            # Random date within period
            random_days = np.random.randint(0, (end_dt - start_dt).days)
            event_date = start_dt + timedelta(days=random_days)
            
            # Select random station and species
            station_id = np.random.choice(list(self.monitoring_stations.keys()))
            station = self.monitoring_stations[station_id]
            species = np.random.choice(list(self.hab_species_mapping.keys()))
            species_code = self.hab_species_mapping[species]
            
            # Add some spatial variation around station
            lat_offset = np.random.normal(0, 0.05)  # ~5km variation
            lon_offset = np.random.normal(0, 0.05)
            
            # Generate cell density based on species and seasonal patterns
            cell_density = self._generate_realistic_cell_density(species_code, event_date)
            
            # Determine severity
            severity = self._classify_severity(species_code, cell_density)
            
            # Generate toxin level (if applicable)
            toxin_level = self._generate_toxin_level(species_code, cell_density) if self._is_toxic_species(species_code) else None
            
            # Create event
            event = HABEvent(
                event_id=f"SYN_{i+1:06d}",
                date=event_date,
                latitude=station["lat"] + lat_offset,
                longitude=station["lon"] + lon_offset,
                species=species,
                cell_density=cell_density,
                toxin_level=toxin_level,
                severity=severity,
                water_body="Seto Inland Sea",
                monitoring_station=station["name"],
                data_source="synthetic_data",
                confidence_level="confirmed"
            )
            
            events.append(event)
        
        # Sort by date
        events.sort(key=lambda x: x.date)
        
        logger.info(f"Generated {len(events)} HAB events")
        
        # Show species distribution
        species_counts = {}
        for event in events:
            species_counts[event.species] = species_counts.get(event.species, 0) + 1
        
        logger.info("Species distribution:")
        for species, count in sorted(species_counts.items()):
            logger.info(f"  {species}: {count} events")
        
        return events
    
    def _generate_realistic_cell_density(self, species_code: str, event_date: datetime) -> float:
        """Generate realistic cell density based on species and seasonal patterns"""
        
        # Base cell densities by species (cells/mL)
        base_densities = {
            "karenia_mikimotoi": 2000,
            "heterosigma_akashiwo": 15000,
            "noctiluca_scintillans": 200,
            "chattonella_antiqua": 8000,
            "cochlodinium_polykrikoides": 12000
        }
        
        base_density = base_densities.get(species_code, 5000)
        
        # Seasonal variation (summer peaks)
        month = event_date.month
        if month in [7, 8, 9]:  # Summer peak
            seasonal_factor = np.random.uniform(2.0, 5.0)
        elif month in [5, 6, 10]:  # Transition periods
            seasonal_factor = np.random.uniform(1.2, 2.5)
        else:  # Lower activity
            seasonal_factor = np.random.uniform(0.3, 1.0)
        
        # Random variation
        variation = np.random.lognormal(0, 0.8)  # Log-normal distribution
        
        cell_density = base_density * seasonal_factor * variation
        
        # Ensure minimum threshold
        return max(cell_density, 10)
    
    def _classify_severity(self, species_code: str, cell_density: float) -> str:
        """Classify HAB event severity based on cell density"""
        thresholds = self.severity_thresholds.get(species_code, 
                                                self.severity_thresholds["karenia_mikimotoi"])
        
        if cell_density >= thresholds["extreme"]:
            return "extreme"
        elif cell_density >= thresholds["high"]:
            return "high"
        elif cell_density >= thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def _is_toxic_species(self, species_code: str) -> bool:
        """Check if species produces toxins"""
        toxic_species = [
            "karenia_mikimotoi", "heterosigma_akashiwo", 
            "chattonella_antiqua", "cochlodinium_polykrikoides"
        ]
        return species_code in toxic_species
    
    def _generate_toxin_level(self, species_code: str, cell_density: float) -> float:
        """Generate toxin level based on species and cell density"""
        # Toxin production varies by species and cell density
        toxin_factors = {
            "karenia_mikimotoi": 0.02,  # µg/L per 1000 cells/mL
            "heterosigma_akashiwo": 0.01,
            "chattonella_antiqua": 0.03,
            "cochlodinium_polykrikoides": 0.015
        }
        
        factor = toxin_factors.get(species_code, 0.02)
        base_toxin = (cell_density / 1000) * factor
        
        # Add variation
        variation = np.random.lognormal(0, 0.3)
        
        return max(base_toxin * variation, 0.01)
    
    def save_hab_events_csv(self, events: List[HABEvent], filename: str = None) -> Path:
        """Save HAB events to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hab_events_{timestamp}.csv"
        
        # Convert to pandas DataFrame
        event_data = [event.to_dict() for event in events]
        df = pd.DataFrame(event_data)
        
        # Save to CSV
        csv_path = self.data_dir / filename
        df.to_csv(csv_path, index=False)
        
        file_size = csv_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Saved {len(events)} HAB events to: {csv_path} ({file_size:.2f} MB)")
        
        return csv_path
    
    def load_hab_events_csv(self, csv_path: Path) -> List[HABEvent]:
        """Load HAB events from CSV file"""
        logger.info(f"Loading HAB events from: {csv_path}")
        
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        
        events = []
        for _, row in df.iterrows():
            event = HABEvent(
                event_id=row['event_id'],
                date=row['date'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                species=row['species'],
                cell_density=row['cell_density'],
                toxin_level=row['toxin_level'] if pd.notna(row['toxin_level']) else None,
                severity=row['severity'],
                water_body=row['water_body'],
                monitoring_station=row['monitoring_station'],
                data_source=row['data_source'],
                confidence_level=row['confidence_level']
            )
            events.append(event)
        
        logger.info(f"Loaded {len(events)} HAB events")
        return events
    
    def create_hab_labels_for_environmental_data(self, 
                                               environmental_data: pd.DataFrame,
                                               hab_events: List[HABEvent],
                                               temporal_window_days: int = 3,
                                               spatial_window_km: int = 10) -> pd.DataFrame:
        """
        Create HAB occurrence labels for environmental data points
        
        This links environmental conditions with HAB events to create
        supervised learning labels for HAB prediction models.
        """
        logger.info("Creating HAB labels for environmental data...")
        logger.info(f"Environmental data: {len(environmental_data)} records")
        logger.info(f"HAB events: {len(hab_events)} events")
        logger.info(f"Matching windows: ±{temporal_window_days} days, ±{spatial_window_km} km")
        
        # Prepare environmental data
        env_data = environmental_data.copy()
        env_data['timestamp'] = pd.to_datetime(env_data['timestamp'])
        
        # Initialize label columns
        env_data['hab_occurrence'] = 0  # Binary: HAB present/absent
        env_data['hab_species'] = None  # Species name if present
        env_data['hab_severity'] = None  # Severity level
        env_data['hab_cell_density'] = 0.0  # Cell density
        env_data['hab_confidence'] = None  # Confidence level
        env_data['days_to_hab_event'] = None  # Days to nearest HAB event
        env_data['distance_to_hab_km'] = None  # Distance to nearest HAB event
        
        matched_events = 0
        
        # For each environmental data point, check for nearby HAB events
        for idx, row in env_data.iterrows():
            env_date = row['timestamp']
            env_lat = row['latitude']
            env_lon = row['longitude']
            
            # Find HAB events within temporal and spatial windows
            matching_events = []
            
            for event in hab_events:
                # Temporal matching
                days_diff = abs((event.date - env_date).days)
                if days_diff <= temporal_window_days:
                    
                    # Spatial matching (approximate distance in km)
                    lat_diff_km = abs(event.latitude - env_lat) * 111  # ~111 km per degree
                    lon_diff_km = abs(event.longitude - env_lon) * 111 * np.cos(np.radians(env_lat))
                    distance_km = np.sqrt(lat_diff_km**2 + lon_diff_km**2)
                    
                    if distance_km <= spatial_window_km:
                        matching_events.append({
                            'event': event,
                            'days_diff': days_diff,
                            'distance_km': distance_km
                        })
            
            # If matching events found, label this environmental data point
            if matching_events:
                # Use the closest event (by time, then by distance)
                closest_match = min(matching_events, 
                                  key=lambda x: (x['days_diff'], x['distance_km']))
                
                event = closest_match['event']
                
                env_data.loc[idx, 'hab_occurrence'] = 1
                env_data.loc[idx, 'hab_species'] = event.species
                env_data.loc[idx, 'hab_severity'] = event.severity
                env_data.loc[idx, 'hab_cell_density'] = event.cell_density
                env_data.loc[idx, 'hab_confidence'] = event.confidence_level
                env_data.loc[idx, 'days_to_hab_event'] = closest_match['days_diff']
                env_data.loc[idx, 'distance_to_hab_km'] = closest_match['distance_km']
                
                matched_events += 1
        
        logger.info(f"✅ HAB labeling complete:")
        logger.info(f"   Labeled records: {matched_events:,} ({matched_events/len(env_data)*100:.1f}%)")
        logger.info(f"   HAB positive: {(env_data['hab_occurrence'] == 1).sum():,}")
        logger.info(f"   HAB negative: {(env_data['hab_occurrence'] == 0).sum():,}")
        
        # Show species distribution in labeled data
        species_counts = env_data[env_data['hab_occurrence'] == 1]['hab_species'].value_counts()
        logger.info(f"Species in labeled data:")
        for species, count in species_counts.items():
            logger.info(f"   {species}: {count} records")
        
        return env_data
    
    def get_hab_event_summary(self, hab_events: List[HABEvent]) -> Dict[str, Any]:
        """Generate summary statistics for HAB events"""
        if not hab_events:
            return {"error": "No HAB events provided"}
        
        # Convert to DataFrame for analysis
        event_data = [event.to_dict() for event in hab_events]
        df = pd.DataFrame(event_data)
        df['date'] = pd.to_datetime(df['date'])
        
        summary = {
            "total_events": len(hab_events),
            "date_range": {
                "start": df['date'].min().isoformat(),
                "end": df['date'].max().isoformat(),
                "duration_days": (df['date'].max() - df['date'].min()).days
            },
            "spatial_extent": {
                "lat_range": [df['latitude'].min(), df['latitude'].max()],
                "lon_range": [df['longitude'].min(), df['longitude'].max()]
            },
            "species_distribution": dict(df['species'].value_counts()),
            "severity_distribution": dict(df['severity'].value_counts()),
            "monthly_distribution": dict(df['date'].dt.month.value_counts().sort_index()),
            "yearly_distribution": dict(df['date'].dt.year.value_counts().sort_index()),
            "cell_density_stats": {
                "mean": df['cell_density'].mean(),
                "median": df['cell_density'].median(),
                "std": df['cell_density'].std(),
                "min": df['cell_density'].min(),
                "max": df['cell_density'].max()
            },
            "monitoring_stations": dict(df['monitoring_station'].value_counts())
        }
        
        return summary
    
    def export_hab_training_dataset(self, 
                                  environmental_data: pd.DataFrame,
                                  output_path: str = None) -> Path:
        """Export HAB-labeled environmental data for machine learning training"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"hab_training_dataset_{timestamp}.csv"
        
        output_file = Path(output_path)
        
        # Save the labeled dataset
        environmental_data.to_csv(output_file, index=False)
        
        file_size = output_file.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Exported HAB training dataset: {output_file} ({file_size:.2f} MB)")
        
        # Generate dataset summary
        hab_positive = (environmental_data['hab_occurrence'] == 1).sum()
        hab_negative = (environmental_data['hab_occurrence'] == 0).sum()
        
        logger.info(f"Training dataset summary:")
        logger.info(f"   Total records: {len(environmental_data):,}")
        logger.info(f"   HAB positive: {hab_positive:,} ({hab_positive/len(environmental_data)*100:.1f}%)")
        logger.info(f"   HAB negative: {hab_negative:,} ({hab_negative/len(environmental_data)*100:.1f}%)")
        logger.info(f"   Variables: {len(environmental_data.columns)}")
        
        return output_file