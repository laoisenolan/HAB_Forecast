"""
Historical Data Collector for HAB Forecasting System

This module handles the collection of historical data for the Seto Inland Sea region,
including chunked downloads, progress tracking, and data validation.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import xarray as xr
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
from dataclasses import dataclass, asdict
import sys

# Handle imports for both direct execution and module import
try:
    # Try relative imports first (when run as module)
    from .copernicus_downloader import CopernicusDownloader
    from .erddap_downloader import ERDDAPDownloader
    from .gee_climatology import GEEClimatology
    from .gee_integrator import GEEIntegrator
    from .smart_downloader import SmartDownloader
    from .csv_data_processor import CSVDataProcessor
    from ..utils.config_loader import ConfigLoader, get_data_dir
except ImportError:
    # Fall back to absolute imports (when run directly)
    # Add src to path
    current_dir = Path(__file__).parent
    src_dir = current_dir.parent
    project_root = src_dir.parent
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(project_root))
    
    from data_collection.copernicus_downloader import CopernicusDownloader
    from data_collection.erddap_downloader import ERDDAPDownloader
    from data_collection.gee_climatology import GEEClimatology
    from data_collection.gee_integrator import GEEIntegrator
    from data_collection.smart_downloader import SmartDownloader
    from data_collection.csv_data_processor import CSVDataProcessor
    from utils.config_loader import ConfigLoader, get_data_dir

logger = logging.getLogger(__name__)

@dataclass
class HistoricalCollectionPeriod:
    """Represents a historical data collection period"""
    name: str
    start_date: str
    end_date: str
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class HistoricalDataCollector:
    """Handles historical data collection for the Seto Inland Sea region"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the historical data collector"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_data_sources_config()
        self.historical_config = self.config.get("historical", {})
        self.spatial_bounds = self.config.get("spatial_bounds", {})
        
        # Initialize downloaders
        self.copernicus_downloader = CopernicusDownloader(config)
        self.erddap_downloader = ERDDAPDownloader(config)
        self.gee_climatology = GEEClimatology(config)
        self.gee_integrator = GEEIntegrator()
        self.smart_downloader = SmartDownloader(config)
        
        # Initialize CSV processor for standardized data format
        self.csv_processor = CSVDataProcessor()
        
        # Data storage (CSV format)
        self.data_dir = get_data_dir()
        self.historical_dir = self.data_dir / "historical_csv"
        self.raw_dir = self.data_dir / "raw_csv"
        
        # Ensure directories exist
        for directory in [self.historical_dir, self.raw_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Chunking settings
        self.chunk_settings = self.historical_config.get("chunk_settings", {})
        self.time_chunk_days = self.chunk_settings.get("time_chunk_days", 30)
        self.max_concurrent_downloads = self.chunk_settings.get("max_concurrent_downloads", 3)
        self.retry_attempts = self.chunk_settings.get("retry_attempts", 3)
        
    def get_available_collection_periods(self) -> List[HistoricalCollectionPeriod]:
        """Get available historical collection periods"""
        periods = []
        collection_periods = self.historical_config.get("collection_periods", [])
        
        for period_config in collection_periods:
            period = HistoricalCollectionPeriod(
                name=period_config["name"],
                start_date=period_config["start_date"],
                end_date=period_config["end_date"],
                description=period_config["description"]
            )
            periods.append(period)
            
        return periods
        
    def get_available_periods(self) -> Dict[str, Dict]:
        """Get available historical collection periods as dictionary (for compatibility)"""
        periods = self.get_available_collection_periods()
        return {period.name: period.to_dict() for period in periods}
        
    def create_time_chunks(self, start_date: str, end_date: str) -> List[Tuple[str, str]]:
        """
        Create time chunks for downloading large historical datasets
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of (chunk_start, chunk_end) tuples
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        chunks = []
        current_start = start_dt
        
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=self.time_chunk_days), end_dt)
            chunks.append((
                current_start.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d")
            ))
            current_start = current_end
            
        return chunks
        
    def collect_period_data(self, start_date: str, end_date: str, chunk_days: int = 30) -> Dict[str, Any]:
        """
        Collect data for a specific period (for compatibility with test)
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            chunk_days: Number of days per chunk
            
        Returns:
            Dictionary with collection results
        """
        try:
            # Create a temporary period for this collection
            period = HistoricalCollectionPeriod(
                name="custom_period",
                start_date=start_date,
                end_date=end_date,
                description=f"Custom period from {start_date} to {end_date}"
            )
            
            # Run the historical collection
            result = self.run_historical_collection(
                period_name="custom_period",
                start_date=start_date,
                end_date=end_date,
                include_climatology=False
            )
            
            return {
                "success": result.get("success", False),
                "period": period.to_dict(),
                "collected_data": result.get("collected_data", {}),
                "saved_files": result.get("saved_files", {}),
                "chunk_stats": result.get("chunk_stats", {})
            }
            
        except Exception as e:
            logger.error(f"Failed to collect period data: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        
    def process_collected_data(self) -> Dict[str, Any]:
        """
        Process collected historical data (for compatibility with test)
        
        Returns:
            Dictionary with processing results
        """
        try:
            # Get the latest historical data files
            historical_files = list(self.historical_dir.glob("*.nc"))
            
            if not historical_files:
                logger.warning("No historical data files found to process")
                return {
                    "success": False,
                    "message": "No historical data files found"
                }
            
            # Process the most recent file
            latest_file = max(historical_files, key=lambda f: f.stat().st_mtime)
            
            # Load and validate the data
            dataset = xr.open_dataset(latest_file)
            
            # Basic processing - could be expanded
            processing_result = {
                "success": True,
                "file": str(latest_file),
                "dataset_info": {
                    "dimensions": dict(dataset.dims),
                    "variables": list(dataset.data_vars.keys()),
                    "size_mb": latest_file.stat().st_size / (1024 * 1024)
                },
                "message": "Historical data processed successfully"
            }
            
            return processing_result
            
        except Exception as e:
            logger.error(f"Failed to process collected data: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        
    def collect_historical_copernicus_data(self, 
                                         start_date: str, 
                                         end_date: str,
                                         data_type: str = "both") -> Dict[str, List[xr.Dataset]]:
        """
        Collect historical Copernicus Marine data in chunks using smart downloader
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            data_type: Type of data to collect ("physics", "ocean_color", or "both")
            
        Returns:
            Dictionary with collected datasets organized by type
        """
        logger.info(f"Starting historical Copernicus data collection from {start_date} to {end_date}")
        
        # Get data availability summary
        summary = self.smart_downloader.get_data_summary(start_date, end_date)
        logger.info(f"Data availability: {summary['availability']}")
        for recommendation in summary['recommendations']:
            logger.info(f"Recommendation: {recommendation}")
        
        # Create time chunks
        chunks = self.create_time_chunks(start_date, end_date)
        logger.info(f"Created {len(chunks)} time chunks for download")
        
        collected_data = {"physics": [], "ocean_color": []}
        failed_chunks = []
        
        # Process chunks with progress tracking
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk_start} to {chunk_end}")
            
            try:
                # Download chunk data using smart downloader
                chunk_data = self.smart_downloader.download_weekly_data_smart(
                    chunk_start, chunk_end, data_type
                )
                
                # Organize by data type
                for data_type_key, dataset in chunk_data.items():
                    if dataset is not None and hasattr(dataset, 'dims'):
                        collected_data[data_type_key].append(dataset)
                        
                logger.info(f"Successfully downloaded chunk {i+1}")
                
            except Exception as e:
                logger.error(f"Failed to download chunk {i+1}: {e}")
                failed_chunks.append((chunk_start, chunk_end, str(e)))
                
            # Add small delay between chunks to avoid overwhelming the server
            time.sleep(1)
            
        # Report results
        total_chunks = len(chunks)
        successful_chunks = total_chunks - len(failed_chunks)
        logger.info(f"Historical collection completed: {successful_chunks}/{total_chunks} chunks successful")
        
        if failed_chunks:
            logger.warning(f"Failed chunks: {failed_chunks}")
            
        return collected_data
        
    def collect_historical_erddap_data(self, 
                                     start_date: str, 
                                     end_date: str) -> Dict[str, List[xr.Dataset]]:
        """
        Collect historical ERDDAP data in chunks
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with collected datasets
        """
        logger.info(f"Starting historical ERDDAP data collection from {start_date} to {end_date}")
        
        # Create time chunks
        chunks = self.create_time_chunks(start_date, end_date)
        logger.info(f"Created {len(chunks)} time chunks for download")
        
        collected_data = {"atmospheric": [], "gfs": []}
        failed_chunks = []
        
        # Process chunks
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            logger.info(f"Processing ERDDAP chunk {i+1}/{len(chunks)}: {chunk_start} to {chunk_end}")
            
            try:
                # Download chunk data
                chunk_data = self.erddap_downloader.download_weekly_atmospheric_data(
                    chunk_start, chunk_end
                )
                
                if chunk_data is not None and hasattr(chunk_data, 'dims'):
                    collected_data["atmospheric"].append(chunk_data)
                    
                logger.info(f"Successfully downloaded ERDDAP chunk {i+1}")
                
            except Exception as e:
                logger.error(f"Failed to download ERDDAP chunk {i+1}: {e}")
                failed_chunks.append((chunk_start, chunk_end, str(e)))
                
            # Add delay between chunks
            time.sleep(1)
            
        # Report results
        total_chunks = len(chunks)
        successful_chunks = total_chunks - len(failed_chunks)
        logger.info(f"Historical ERDDAP collection completed: {successful_chunks}/{total_chunks} chunks successful")
        
        if failed_chunks:
            logger.warning(f"Failed ERDDAP chunks: {failed_chunks}")
            
        return collected_data
        
    def collect_historical_gee_data(self, 
                                  start_date: str, 
                                  end_date: str) -> Dict[str, List[xr.Dataset]]:
        """
        Collect historical Google Earth Engine data in chunks
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with collected GEE datasets organized by type
        """
        logger.info(f"Starting historical GEE data collection from {start_date} to {end_date}")
        
        # Create time chunks
        chunks = self.create_time_chunks(start_date, end_date)
        logger.info(f"Created {len(chunks)} time chunks for GEE download")
        
        # Organize data by GEE categories
        collected_data = {
            "atmospheric": [],      # ERA5: wind, precipitation
            "ocean_physics": [],    # HYCOM: temperature, salinity, currents, elevation
            "chlorophyll": [],      # JAXA GCOM-C: high-resolution chlorophyll
            "turbidity": []         # Sentinel-2: high-resolution turbidity
        }
        failed_chunks = []
        
        # Process chunks with progress tracking
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            logger.info(f"Processing GEE chunk {i+1}/{len(chunks)}: {chunk_start} to {chunk_end}")
            
            try:
                # Download comprehensive GEE data for this chunk
                chunk_data = self.gee_integrator.get_comprehensive_gee_data(
                    chunk_start, chunk_end
                )
                
                if chunk_data:
                    # Convert GEE data directly to CSV format for better standardization
                    gee_csv_df = self.csv_processor.convert_gee_data_to_csv(
                        chunk_data, data_source="gee", timestamp=chunk_start
                    )
                    
                    if len(gee_csv_df) > 0:
                        # Convert back to xarray for compatibility with existing merge logic
                        # Group by variable and create datasets
                        for var_name in gee_csv_df['variable_name'].unique():
                            var_data = gee_csv_df[gee_csv_df['variable_name'] == var_name]
                            
                            # Create xarray dataset from CSV data
                            ds = xr.Dataset()
                            
                            # Add coordinates
                            ds.coords['time'] = ('point', pd.to_datetime(var_data['timestamp']))
                            ds.coords['lat'] = ('point', var_data['latitude'])
                            ds.coords['lon'] = ('point', var_data['longitude'])
                            
                            # Add data variable
                            ds[var_name] = ('point', var_data['value'])
                            ds[var_name].attrs['units'] = var_data['unit'].iloc[0]
                            ds[var_name].attrs['standard_name'] = var_name
                            
                            # Categorize by data type
                            if var_name in ['wind_u', 'wind_v', 'precipitation']:
                                collected_data["atmospheric"].append(ds)
                            elif var_name in ['ocean_temperature_hycom', 'ocean_salinity', 'ocean_current_u', 'ocean_current_v', 'sea_surface_elevation']:
                                collected_data["ocean_physics"].append(ds)
                            elif var_name in ['chlorophyll_a_jaxa']:
                                collected_data["chlorophyll"].append(ds)
                            elif var_name in ['turbidity_sentinel2']:
                                collected_data["turbidity"].append(ds)
                        
                logger.info(f"Successfully downloaded GEE chunk {i+1}")
                
            except Exception as e:
                logger.error(f"Failed to download GEE chunk {i+1}: {e}")
                failed_chunks.append((chunk_start, chunk_end, str(e)))
                
            # Add delay between chunks to avoid overwhelming GEE
            time.sleep(2)
            
        # Report results
        total_chunks = len(chunks)
        successful_chunks = total_chunks - len(failed_chunks)
        logger.info(f"Historical GEE collection completed: {successful_chunks}/{total_chunks} chunks successful")
        
        if failed_chunks:
            logger.warning(f"Failed GEE chunks: {failed_chunks}")
            
        return collected_data
    
    def _combine_gee_atmospheric_data(self, chunk_data: Dict) -> Optional[xr.Dataset]:
        """Combine atmospheric data from GEE chunk into xarray Dataset"""
        try:
            datasets = []
            
            # Wind data
            if 'wind' in chunk_data:
                wind_ds = self._convert_gee_to_xarray(chunk_data['wind'], 'wind', None)
                if wind_ds is not None:
                    datasets.append(wind_ds)
            
            # Precipitation data
            if 'precipitation' in chunk_data:
                precip_ds = self._convert_gee_to_xarray(chunk_data['precipitation'], 'precipitation', None)
                if precip_ds is not None:
                    datasets.append(precip_ds)
            
            # Merge datasets if we have any
            if datasets:
                if len(datasets) == 1:
                    return datasets[0]
                else:
                    return xr.merge(datasets)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to combine atmospheric data: {e}")
            return None
    
    def _combine_gee_ocean_data(self, chunk_data: Dict) -> Optional[xr.Dataset]:
        """Combine ocean physics data from GEE chunk into xarray Dataset"""
        try:
            datasets = []
            
            # Check for ocean data in the chunk
            ocean_keys = ['ocean', 'hycom_temperature_salinity', 'hycom_velocity', 'hycom_elevation']
            
            for key in ocean_keys:
                if key in chunk_data:
                    ocean_ds = self._convert_gee_to_xarray(chunk_data[key], key, None)
                    if ocean_ds is not None:
                        datasets.append(ocean_ds)
            
            # Merge datasets if we have any
            if datasets:
                if len(datasets) == 1:
                    return datasets[0]
                else:
                    return xr.merge(datasets)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to combine ocean data: {e}")
            return None
    
    def _convert_gee_to_xarray(self, gee_data: any, data_type: str, timestamp: str) -> Optional[xr.Dataset]:
        """Convert GEE data to xarray Dataset"""
        try:
            if gee_data is None:
                return None
            
            # If it's already an xarray Dataset, return it
            if isinstance(gee_data, xr.Dataset):
                return gee_data
            
            # If it's a pandas DataFrame, convert to xarray
            if isinstance(gee_data, pd.DataFrame):
                # Create xarray Dataset from DataFrame
                ds = xr.Dataset()
                
                # Add coordinates
                if 'latitude' in gee_data.columns and 'longitude' in gee_data.columns:
                    ds.coords['lat'] = ('point', gee_data['latitude'].values)
                    ds.coords['lon'] = ('point', gee_data['longitude'].values)
                
                # Add time coordinate if available
                if 'date' in gee_data.columns:
                    ds.coords['time'] = ('point', pd.to_datetime(gee_data['date']))
                elif timestamp:
                    ds.coords['time'] = pd.to_datetime(timestamp)
                
                # Add data variables (exclude coordinate columns)
                coord_cols = ['latitude', 'longitude', 'date']
                for col in gee_data.columns:
                    if col not in coord_cols:
                        ds[col] = ('point', gee_data[col].values)
                
                return ds
            
            # If it's a dictionary, try to convert
            if isinstance(gee_data, dict):
                ds = xr.Dataset()
                
                for key, value in gee_data.items():
                    if isinstance(value, (list, np.ndarray)):
                        ds[key] = ('point', value)
                    elif isinstance(value, (int, float)):
                        ds[key] = value
                
                return ds
            
            logger.warning(f"Unknown GEE data type for {data_type}: {type(gee_data)}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to convert GEE data to xarray for {data_type}: {e}")
            return None
        
    def merge_historical_datasets(self, 
                                datasets_by_type: Dict[str, List[xr.Dataset]]) -> Dict[str, xr.Dataset]:
        """
        Merge chunked datasets into complete historical datasets
        
        Args:
            datasets_by_type: Dictionary with datasets organized by type
            
        Returns:
            Dictionary with merged datasets
        """
        logger.info("Merging historical datasets")
        
        merged_datasets = {}
        
        for data_type, datasets in datasets_by_type.items():
            if not datasets:
                logger.warning(f"No datasets to merge for {data_type}")
                continue
                
            try:
                # Concatenate datasets along time dimension
                merged_dataset = xr.concat(datasets, dim="time")
                
                # Sort by time
                merged_dataset = merged_dataset.sortby("time")
                
                # Remove duplicates if any
                merged_dataset = merged_dataset.drop_duplicates("time")
                
                merged_datasets[data_type] = merged_dataset
                logger.info(f"Successfully merged {len(datasets)} {data_type} datasets")
                
            except Exception as e:
                logger.error(f"Failed to merge {data_type} datasets: {e}")
                
        return merged_datasets
        
    def save_historical_data(self, 
                           merged_datasets: Dict[str, xr.Dataset],
                           period_name: str,
                           timestamp: str = None) -> Dict[str, Path]:
        """
        Save merged historical datasets as CSV files
        
        Args:
            merged_datasets: Dictionary with merged datasets
            period_name: Name of the collection period
            timestamp: Timestamp for file naming
            
        Returns:
            Dictionary with saved CSV file paths
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        logger.info(f"📊 Saving historical data as CSV for period: {period_name}")
        
        saved_files = {}
        all_dataframes = []
        
        for data_type, dataset in merged_datasets.items():
            try:
                # Determine data source based on data type
                if data_type in ['physics', 'ocean_color', 'biogeochemistry', 'nutrients', 'biology', 'optics']:
                    data_source = 'copernicus'
                    dataset_id = f'copernicus_{data_type}'
                elif data_type in ['atmospheric', 'ocean_physics', 'chlorophyll', 'turbidity']:
                    data_source = 'gee'
                    dataset_id = f'gee_{data_type}'
                else:
                    data_source = 'erddap'
                    dataset_id = f'erddap_{data_type}'
                
                # Convert xarray dataset to standardized CSV format
                logger.info(f"🔄 Converting {data_type} data to CSV format...")
                csv_df = self.csv_processor.convert_xarray_to_csv(
                    dataset=dataset,
                    data_source=data_source,
                    dataset_id=dataset_id,
                    timestamp=timestamp
                )
                
                if len(csv_df) > 0:
                    # Save individual CSV file for this data type
                    filename = f"{data_type}_{period_name}_{timestamp}"
                    file_path = self.csv_processor.save_csv(
                        df=csv_df,
                        filename=filename,
                        subdirectory="by_source"
                    )
                    
                    saved_files[data_type] = file_path
                    all_dataframes.append(csv_df)
                    
                    logger.info(f"✅ Saved {data_type} CSV: {len(csv_df)} records to {file_path}")
                else:
                    logger.warning(f"⚠️ No data to save for {data_type}")
                
            except Exception as e:
                logger.error(f"❌ Failed to save {data_type} historical data: {e}")
        
        # Create combined CSV file with all data
        try:
            if all_dataframes:
                logger.info("🔄 Creating combined historical CSV file...")
                combined_df = pd.concat(all_dataframes, ignore_index=True)
                
                # Remove duplicates and sort
                combined_df = combined_df.drop_duplicates(
                    subset=['timestamp', 'latitude', 'longitude', 'variable_name', 'data_source'],
                    keep='first'
                ).sort_values(['timestamp', 'variable_name', 'latitude', 'longitude'])
                
                # Save combined CSV
                combined_filename = f"combined_historical_{period_name}_{timestamp}"
                combined_path = self.csv_processor.save_csv(
                    df=combined_df,
                    filename=combined_filename,
                    subdirectory=None
                )
                
                saved_files['combined'] = combined_path
                
                # Generate and save data summary
                summary = self.csv_processor.get_variable_summary(combined_df)
                summary_path = self.historical_dir / f"data_summary_{period_name}_{timestamp}.json"
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                
                saved_files['summary'] = summary_path
                
                logger.info(f"✅ Combined historical CSV: {len(combined_df)} records, {len(combined_df['variable_name'].unique())} variables")
                logger.info(f"📊 Variables: {', '.join(combined_df['variable_name'].unique())}")
                
        except Exception as e:
            logger.error(f"❌ Failed to create combined CSV file: {e}")
                
        return saved_files
        
    def create_historical_collection_report(self,
                                          period: HistoricalCollectionPeriod,
                                          collected_data: Dict[str, List[xr.Dataset]],
                                          merged_datasets: Dict[str, xr.Dataset],
                                          saved_files: Dict[str, Path],
                                          chunk_stats: Dict[str, Any]) -> Dict:
        """
        Create a comprehensive historical collection report
        
        Args:
            period: Collection period information
            collected_data: Original chunked data
            merged_datasets: Merged datasets
            saved_files: Saved file paths
            chunk_stats: Chunking statistics
            
        Returns:
            Historical collection report
        """
        logger.info("Creating historical collection report")
        
        report = {
            "collection_period": period.to_dict(),
            "collection_date": datetime.now().isoformat(),
            "spatial_bounds": self.spatial_bounds,
            "chunking_settings": self.chunk_settings,
            "data_sources": {},
            "file_locations": {},
            "statistics": {},
            "chunk_statistics": chunk_stats
        }
        
        # Data sources information
        for data_type, datasets in collected_data.items():
            if datasets:
                # Get time range from first and last datasets
                first_dataset = datasets[0]
                last_dataset = datasets[-1]
                
                report["data_sources"][data_type] = {
                    "total_chunks": len(datasets),
                    "time_range": {
                        "start": str(first_dataset.time.min().values) if "time" in first_dataset.dims else None,
                        "end": str(last_dataset.time.max().values) if "time" in last_dataset.dims else None
                    },
                    "variables": list(first_dataset.data_vars.keys()),
                    "dimensions": dict(first_dataset.dims)
                }
                
        # Merged dataset information
        for data_type, dataset in merged_datasets.items():
            report["data_sources"][f"{data_type}_merged"] = {
                "dimensions": dict(dataset.dims),
                "variables": list(dataset.data_vars.keys()),
                "time_range": {
                    "start": str(dataset.time.min().values) if "time" in dataset.dims else None,
                    "end": str(dataset.time.max().values) if "time" in dataset.dims else None
                }
            }
            
        # File locations
        report["file_locations"] = {k: str(v) for k, v in saved_files.items()}
        
        # Statistics
        total_datasets = len(merged_datasets)
        total_files = len(saved_files)
        report["statistics"] = {
            "total_datasets": total_datasets,
            "total_files_saved": total_files,
            "collection_success": total_files > 0
        }
        
        return report
        
    def run_historical_collection(self,
                                period_name: str,
                                start_date: str = None,
                                end_date: str = None,
                                include_climatology: bool = False) -> Dict:
        """
        Run complete historical data collection for a specific period
        
        Args:
            period_name: Name of the collection period
            start_date: Start date (if not using predefined period)
            end_date: End date (if not using predefined period)
            include_climatology: Whether to include climatology computation
            
        Returns:
            Complete collection results
        """
        logger.info(f"Starting historical collection for period: {period_name}")
        
        collection_start_time = time.time()
        
        # Get period information
        if start_date and end_date:
            period = HistoricalCollectionPeriod(
                name=period_name,
                start_date=start_date,
                end_date=end_date,
                description=f"Custom period: {start_date} to {end_date}"
            )
        else:
            # Find predefined period
            available_periods = self.get_available_collection_periods()
            period = None
            for p in available_periods:
                if p.name == period_name:
                    period = p
                    break
                    
            if not period:
                raise ValueError(f"Period '{period_name}' not found in configuration")
                
        logger.info(f"Collection period: {period.description}")
        
        try:
            # Collect Copernicus Marine data (15 variables)
            logger.info("🌊 Collecting historical Copernicus Marine data (15 variables)...")
            copernicus_data = self.collect_historical_copernicus_data(
                period.start_date, period.end_date, "both"
            )
            
            # Collect Google Earth Engine data (10 variables) 
            logger.info("🌍 Collecting historical Google Earth Engine data (10 variables)...")
            gee_data = self.collect_historical_gee_data(
                period.start_date, period.end_date
            )
            
            # Collect ERDDAP data (backup)
            logger.info("📡 Collecting historical ERDDAP data (backup)...")
            erddap_data = self.collect_historical_erddap_data(
                period.start_date, period.end_date
            )
            
            # Merge datasets from all sources
            logger.info("🔄 Merging historical datasets from all sources...")
            all_collected_data = {**copernicus_data, **gee_data, **erddap_data}
            merged_datasets = self.merge_historical_datasets(all_collected_data)
            
            # Save merged datasets
            logger.info("Saving historical datasets...")
            saved_files = self.save_historical_data(merged_datasets, period.name)
            
            # Create report
            chunk_stats = {
                "total_chunks": len(self.create_time_chunks(period.start_date, period.end_date)),
                "chunk_size_days": self.time_chunk_days,
                "max_concurrent": self.max_concurrent_downloads
            }
            
            report = self.create_historical_collection_report(
                period, all_collected_data, merged_datasets, saved_files, chunk_stats
            )
            
            # Save report
            report_file = self.save_historical_report(report, period.name)
            
            collection_end_time = time.time()
            collection_duration = collection_end_time - collection_start_time
            
            # Add collection metadata
            report["collection_metadata"] = {
                "start_time": collection_start_time,
                "end_time": collection_end_time,
                "duration_seconds": collection_duration,
                "success": True
            }
            
            logger.info(f"Historical collection completed in {collection_duration:.2f} seconds")
            
            return {
                "period": period,
                "collected_data": all_collected_data,
                "merged_datasets": merged_datasets,
                "saved_files": saved_files,
                "report": report,
                "report_file": report_file,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Historical collection failed: {e}")
            
            collection_end_time = time.time()
            collection_duration = collection_end_time - collection_start_time
            
            return {
                "period": period,
                "collected_data": {},
                "merged_datasets": {},
                "saved_files": {},
                "report": {
                    "collection_metadata": {
                        "start_time": collection_start_time,
                        "end_time": collection_end_time,
                        "duration_seconds": collection_duration,
                        "success": False,
                        "error": str(e)
                    }
                },
                "success": False,
                "error": str(e)
            }
            
    def save_historical_report(self, report: Dict, period_name: str) -> Path:
        """Save historical collection report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"historical_collection_report_{period_name}_{timestamp}.json"
        
        report_file = self.historical_dir / filename
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Historical collection report saved to: {report_file}")
        return report_file
        
    def get_historical_collection_status(self) -> Dict:
        """Get status of historical data collections"""
        status = {
            "available_periods": [p.to_dict() for p in self.get_available_collection_periods()],
            "existing_collections": [],
            "storage_info": {}
        }
        
        # Check for existing historical collections
        if self.historical_dir.exists():
            for data_type_dir in self.historical_dir.iterdir():
                if data_type_dir.is_dir():
                    files = list(data_type_dir.glob("*.nc"))
                    if files:
                        status["existing_collections"].append({
                            "data_type": data_type_dir.name,
                            "file_count": len(files),
                            "files": [f.name for f in files]
                        })
                        
        # Storage information
        status["storage_info"] = {
            "historical_dir": str(self.historical_dir),
            "total_size_mb": self._get_directory_size(self.historical_dir)
        }
        
        return status
        
    def _get_directory_size(self, directory: Path) -> float:
        """Calculate directory size in MB"""
        total_size = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        return total_size / (1024 * 1024)  # Convert to MB 