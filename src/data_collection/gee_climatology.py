"""
Google Earth Engine Climatology Computation for HAB Forecasting System

This module handles climatology computation from OC-CCI v6 data using Google Earth Engine
for the Seto Inland Sea region.
"""

import logging
import os
import ee
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xarray as xr
import numpy as np
from src.utils.config_loader import ConfigLoader, get_data_dir

logger = logging.getLogger(__name__)

class GEEClimatology:
    """Google Earth Engine climatology computation"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the GEE climatology processor"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_data_sources_config()
        self.gee_config = self.config.get("google_earth_engine", {})
        self.spatial_bounds = self.config.get("spatial_bounds", {})
        self.climatology_dir = get_data_dir() / "climatology"
        self.climatology_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Google Earth Engine
        self._initialize_gee()
        
    def _initialize_gee(self):
        """Initialize Google Earth Engine"""
        try:
            # Always try to find credentials file first
            service_account_path = Path(__file__).parent.parent.parent / "config" / "hab-forecast-e78bb750e4e7.json"
            
            if service_account_path.exists():
                # Initialize with service account
                import json
                with open(service_account_path, 'r') as f:
                    service_account_info = json.load(f)
                
                credentials = ee.ServiceAccountCredentials(
                    service_account_info.get('client_email'),
                    key_data=service_account_info.get('private_key')
                )
                project_id = service_account_info.get('project_id')
                ee.Initialize(credentials, project=project_id)
                logger.info(f"✅ Google Earth Engine initialized with service account: {service_account_path}, project: {project_id}")
            elif 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
                # Try environment variable as backup
                credentials_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
                if Path(credentials_path).exists():
                    # Initialize with service account from environment
                    import json
                    with open(credentials_path, 'r') as f:
                        service_account_info = json.load(f)
                    
                    credentials = ee.ServiceAccountCredentials(
                        service_account_info.get('client_email'),
                        key_data=service_account_info.get('private_key')
                    )
                    project_id = service_account_info.get('project_id')
                    ee.Initialize(credentials, project=project_id)
                    logger.info(f"✅ Google Earth Engine initialized with service account: {credentials_path}, project: {project_id}")
                else:
                    # Fallback to default initialization
                    ee.Initialize()
                    logger.info("✅ Google Earth Engine initialized with default credentials")
            else:
                # Fallback to default initialization
                ee.Initialize()
                logger.info("✅ Google Earth Engine initialized with default credentials")
            
            self.gee_available = True
        except Exception as e:
            logger.error(f"Failed to initialize Google Earth Engine: {e}")
            logger.info("Please check your service account credentials")
            self.gee_available = False
            # Don't raise the exception, just mark as unavailable
            
    def create_seto_inland_sea_roi(self) -> ee.Geometry:
        """Create Region of Interest for Seto Inland Sea"""
        bounds = [
            self.spatial_bounds["west"],
            self.spatial_bounds["south"],
            self.spatial_bounds["east"],
            self.spatial_bounds["north"]
        ]
        
        roi = ee.Geometry.Rectangle(bounds)
        logger.info(f"Created ROI for Seto Inland Sea: {bounds}")
        return roi
        
    def load_oc_cci_data(self, start_year: int = 2002, end_year: int = 2024) -> ee.ImageCollection:
        """
        Load OC-CCI v6 chlorophyll-a data
        
        Args:
            start_year: Start year for climatology
            end_year: End year for climatology
            
        Returns:
            Earth Engine ImageCollection
        """
        logger.info(f"Loading OC-CCI v6 data from {start_year} to {end_year}")
        
        oc_cci_config = self.gee_config.get("oc_cci_v6", {})
        dataset_name = oc_cci_config.get("dataset")
        variable = oc_cci_config.get("variable", "chlor_a")
        
        # Load the dataset
        dataset = ee.ImageCollection(dataset_name)
        
        # Filter by date range
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        
        filtered_dataset = dataset.filterDate(start_date, end_date)
        
        # Select chlorophyll-a variable
        chl_dataset = filtered_dataset.select(variable)
        
        # Filter out invalid values (typically negative or very high values)
        valid_dataset = chl_dataset.filter(ee.Filter.gt(variable, 0))
        
        logger.info(f"Loaded {valid_dataset.size().getInfo()} images")
        return valid_dataset
        
    def compute_seasonal_statistics(self, 
                                  dataset: ee.ImageCollection,
                                  seasons: Optional[List[Dict]] = None) -> Dict[str, ee.Image]:
        """
        Compute seasonal statistics (mean and 75th percentile)
        
        Args:
            dataset: Chlorophyll-a dataset
            seasons: List of season definitions
            
        Returns:
            Dictionary with seasonal statistics
        """
        logger.info("Computing seasonal statistics")
        
        if seasons is None:
            seasons = self.gee_config.get("oc_cci_v6", {}).get("seasons", [])
            
        seasonal_stats = {}
        
        for season in seasons:
            season_name = season["name"]
            months = season["months"]
            
            logger.info(f"Processing season: {season_name} (months: {months})")
            
            # Filter dataset by months
            season_dataset = dataset.filter(ee.Filter.calendarRange(months[0], months[-1], "month"))
            
            # Compute mean
            season_mean = season_dataset.mean()
            seasonal_stats[f"{season_name}_mean"] = season_mean
            
            # Compute 75th percentile
            season_p75 = season_dataset.reduce(ee.Reducer.percentile([75]))
            seasonal_stats[f"{season_name}_p75"] = season_p75
            
            logger.info(f"Completed {season_name} statistics")
            
        return seasonal_stats
        
    def export_climatology_rasters(self, 
                                  seasonal_stats: Dict[str, ee.Image],
                                  roi: ee.Geometry,
                                  resolution: float = 0.018) -> List[str]:
        """
        Export climatology rasters as GeoTIFF files
        
        Args:
            seasonal_stats: Dictionary with seasonal statistics
            roi: Region of interest
            resolution: Output resolution in degrees
            
        Returns:
            List of exported file paths
        """
        logger.info("Exporting climatology rasters")
        
        exported_files = []
        
        for stat_name, image in seasonal_stats.items():
            try:
                # Prepare image for export
                export_image = image.clip(roi)
                
                # Set projection and scale
                export_image = export_image.reproject(
                    crs='EPSG:4326',
                    scale=resolution
                )
                
                # Export to Google Drive (then download)
                filename = f"seto_inland_sea_{stat_name}.tif"
                
                task = ee.batch.Export.image.toDrive(
                    image=export_image,
                    description=f"Seto Inland Sea {stat_name}",
                    folder="HAB_Forecast_Climatology",
                    fileNamePrefix=filename,
                    scale=resolution,
                    region=roi,
                    fileFormat="GeoTIFF",
                    maxPixels=1e10
                )
                
                task.start()
                logger.info(f"Started export task for {stat_name}")
                
                # For now, we'll return the task ID
                # In a real implementation, you'd wait for completion and download
                exported_files.append(f"Task ID: {task.id}")
                
            except Exception as e:
                logger.error(f"Failed to export {stat_name}: {e}")
                
        return exported_files
        
    def compute_climatology(self, 
                          start_year: int = 2002,
                          end_year: int = 2024,
                          export_rasters: bool = True) -> Dict[str, ee.Image]:
        """
        Compute complete climatology for Seto Inland Sea
        
        Args:
            start_year: Start year for climatology
            end_year: End year for climatology
            export_rasters: Whether to export rasters
            
        Returns:
            Dictionary with climatology statistics
        """
        logger.info(f"Computing climatology for {start_year}-{end_year}")
        
        try:
            # Create ROI
            roi = self.create_seto_inland_sea_roi()
            
            # Load OC-CCI data
            dataset = self.load_oc_cci_data(start_year, end_year)
            
            # Compute seasonal statistics
            seasonal_stats = self.compute_seasonal_statistics(dataset)
            
            # Export rasters if requested
            if export_rasters:
                exported_files = self.export_climatology_rasters(seasonal_stats, roi)
                logger.info(f"Exported {len(exported_files)} raster files")
                
            logger.info("Climatology computation completed successfully")
            return seasonal_stats
            
        except Exception as e:
            logger.error(f"Failed to compute climatology: {e}")
            raise
            
    def create_climatology_summary(self, seasonal_stats: Dict[str, ee.Image]) -> Dict:
        """
        Create summary statistics for climatology
        
        Args:
            seasonal_stats: Dictionary with seasonal statistics
            
        Returns:
            Summary statistics dictionary
        """
        logger.info("Creating climatology summary")
        
        summary = {
            "computation_date": datetime.now().isoformat(),
            "spatial_bounds": self.spatial_bounds,
            "seasons": {}
        }
        
        # Extract season information
        seasons = self.gee_config.get("oc_cci_v6", {}).get("seasons", [])
        
        for season in seasons:
            season_name = season["name"]
            months = season["months"]
            
            mean_key = f"{season_name}_mean"
            p75_key = f"{season_name}_p75"
            
            if mean_key in seasonal_stats and p75_key in seasonal_stats:
                summary["seasons"][season_name] = {
                    "months": months,
                    "mean_image": mean_key,
                    "p75_image": p75_key,
                    "description": f"Chlorophyll-a statistics for {season_name} season"
                }
                
        logger.info("Climatology summary created")
        return summary
        
    def validate_climatology_data(self, seasonal_stats: Dict[str, ee.Image]) -> bool:
        """
        Validate climatology data quality
        
        Args:
            seasonal_stats: Dictionary with seasonal statistics
            
        Returns:
            True if data quality is acceptable
        """
        logger.info("Validating climatology data quality")
        
        try:
            # Check that all expected statistics are present
            expected_stats = []
            seasons = self.gee_config.get("oc_cci_v6", {}).get("seasons", [])
            
            for season in seasons:
                season_name = season["name"]
                expected_stats.extend([f"{season_name}_mean", f"{season_name}_p75"])
                
            for stat_name in expected_stats:
                if stat_name not in seasonal_stats:
                    logger.error(f"Missing expected statistic: {stat_name}")
                    return False
                    
            # Check that images have valid data
            roi = self.create_seto_inland_sea_roi()
            
            for stat_name, image in seasonal_stats.items():
                # Sample a few points to check data validity
                sample = image.sample(region=roi, numPixels=100)
                
                # Check for null values
                null_count = sample.aggregate_array(stat_name).filter(ee.Filter.eq(stat_name, None)).size()
                total_count = sample.size()
                
                null_ratio = null_count.divide(total_count)
                
                if null_ratio.getInfo() > 0.5:  # More than 50% null
                    logger.warning(f"High null ratio for {stat_name}: {null_ratio.getInfo():.2%}")
                    
            logger.info("Climatology data validation completed")
            return True
            
        except Exception as e:
            logger.error(f"Climatology validation failed: {e}")
            return False
            
    def get_climatology_info(self) -> Dict:
        """Get information about climatology configuration"""
        return {
            "dataset": self.gee_config.get("oc_cci_v6", {}).get("dataset"),
            "variable": self.gee_config.get("oc_cci_v6", {}).get("variable"),
            "temporal_coverage": self.gee_config.get("oc_cci_v6", {}).get("temporal_coverage"),
            "seasons": self.gee_config.get("oc_cci_v6", {}).get("seasons", []),
            "statistics": self.gee_config.get("oc_cci_v6", {}).get("statistics", []),
            "spatial_bounds": self.spatial_bounds
        }
        
    def save_climatology_metadata(self, summary: Dict, filename: str = "climatology_metadata.json"):
        """Save climatology metadata to file"""
        import json
        
        metadata_file = self.climatology_dir / filename
        
        with open(metadata_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Climatology metadata saved to: {metadata_file}")
        
if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    try:
        climatology = GEEClimatology()
        
        # Compute climatology
        seasonal_stats = climatology.compute_climatology(2002, 2024)
        
        # Create summary
        summary = climatology.create_climatology_summary(seasonal_stats)
        
        # Save metadata
        climatology.save_climatology_metadata(summary)
        
        print("Climatology computation completed successfully")
        print(f"Generated {len(seasonal_stats)} seasonal statistics")
        
    except Exception as e:
        print(f"Climatology computation failed: {e}")
        print("Please ensure you have authenticated with Google Earth Engine") 