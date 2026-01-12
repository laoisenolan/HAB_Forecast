"""
Data Collection Orchestrator for HAB Forecasting System

This module orchestrates data collection from all sources and manages
the complete data collection pipeline for the Seto Inland Sea region.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import xarray as xr
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .copernicus_downloader import CopernicusDownloader
from .erddap_downloader import ERDDAPDownloader
from .gee_climatology import GEEClimatology
from .historical_collector import HistoricalDataCollector
from .dataset_selector import DatasetSelector
from .quality_monitor import QualityMonitor
from .parallel_downloader import ParallelDownloadManager, BatchDownloadScheduler, DownloadTask
from .data_cache import DataCache
from src.utils.config_loader import ConfigLoader, get_data_dir

logger = logging.getLogger(__name__)

class DataCollectionOrchestrator:
    """Orchestrates data collection from all sources"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the data collection orchestrator"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_data_sources_config()
        self.spatial_bounds = self.config.get("spatial_bounds", {})
        self.temporal_settings = self.config.get("temporal", {})
        
        # Initialize dataset selector for intelligent dataset selection
        self.dataset_selector = DatasetSelector()
        
        # Initialize quality monitor for automated quality monitoring and alerting
        self.quality_monitor = QualityMonitor(config)
        
        # Initialize data cache for performance optimization
        self.data_cache = DataCache(cache_dir="data/cache", ttl_days=30)
        
        # Initialize downloaders
        self.copernicus_downloader = CopernicusDownloader(config)
        self.erddap_downloader = ERDDAPDownloader(config)
        
        # Initialize GEE climatology (optional - may fail if not authenticated)
        self.gee_climatology = None
        self.gee_available = False
        
        try:
            # Try to initialize GEE with available credentials
            self.gee_climatology = GEEClimatology(config)
            self.gee_available = True
            logger.info("GEE initialization successful")
        except Exception as e:
            logger.warning(f"GEE initialization failed (will be disabled): {e}")
            # Don't let GEE failure break the entire system
            pass
            
        self.historical_collector = HistoricalDataCollector(config)
        
        # Initialize parallel download manager for optimized performance
        self.parallel_download_manager = ParallelDownloadManager(max_workers=5)
        self.batch_scheduler = BatchDownloadScheduler(self.parallel_download_manager)
        
        # Register downloader functions with parallel manager
        self._register_parallel_downloaders()
        
        # Data storage
        self.data_dir = get_data_dir()
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.climatology_dir = self.data_dir / "climatology"
        
        # Ensure directories exist
        for directory in [self.raw_dir, self.processed_dir, self.climatology_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Start quality monitoring
        self._start_quality_monitoring()
    
    def _register_parallel_downloaders(self):
        """Register downloader functions with the parallel download manager"""
        # Register Copernicus downloader
        self.parallel_download_manager.register_downloader(
            'copernicus',
            self._parallel_copernicus_download
        )
        
        # Register GEE downloader
        self.parallel_download_manager.register_downloader(
            'gee',
            self._parallel_gee_download
        )
        
        # Register ERDDAP downloader
        self.parallel_download_manager.register_downloader(
            'erddap',
            self._parallel_erddap_download
        )
        
        logger.info("Registered parallel downloaders for all data sources")
    
    def _parallel_copernicus_download(self, variable: str, start_date: str, end_date: str):
        """Parallel download function for Copernicus data"""
        try:
            # Map variable to appropriate download method
            if variable in ['sst', 'sea_surface_temperature']:
                return self.copernicus_downloader.download_sst_data(start_date, end_date)
            elif variable in ['chl', 'chlorophyll_a']:
                return self.copernicus_downloader.download_ocean_color_data(start_date, end_date)
            elif variable in ['turbidity', 'kd490']:
                return self.copernicus_downloader.download_turbidity_data(start_date, end_date)
            elif variable in ['physics', 'ocean_physics']:
                return self.copernicus_downloader.download_physics_data(start_date, end_date)
            elif variable in ['biogeochemistry', 'nutrients']:
                return self.copernicus_downloader.download_biogeochemistry_data(start_date, end_date)
            else:
                # Default to weekly data
                return self.copernicus_downloader.download_weekly_data(start_date, end_date, "both")
        except Exception as e:
            logger.error(f"Parallel Copernicus download failed for {variable}: {e}")
            raise
    
    def _parallel_gee_download(self, variable: str, start_date: str, end_date: str):
        """Parallel download function for GEE data"""
        if not self.gee_available:
            logger.warning(f"GEE not available for {variable} download")
            return None
            
        try:
            # Map variable to appropriate GEE method
            if variable in ['wind', 'wind_components']:
                return self.gee_climatology.get_wind_data(start_date, end_date)
            elif variable in ['precipitation', 'rainfall']:
                return self.gee_climatology.get_precipitation_data(start_date, end_date)
            elif variable in ['sst', 'sea_surface_temperature']:
                return self.gee_climatology.get_sst_data(start_date, end_date)
            elif variable in ['chl', 'chlorophyll_a']:
                return self.gee_climatology.get_chlorophyll_data(start_date, end_date)
            elif variable in ['turbidity', 'sentinel2_turbidity']:
                return self.gee_climatology.get_sentinel2_turbidity_data(start_date, end_date)
            elif variable in ['hycom', 'ocean_physics']:
                return self.gee_climatology.get_hycom_data(start_date, end_date)
            else:
                # Default to comprehensive data
                return self.gee_climatology.get_comprehensive_data(start_date, end_date)
        except Exception as e:
            logger.error(f"Parallel GEE download failed for {variable}: {e}")
            return None
    
    def _parallel_erddap_download(self, variable: str, start_date: str, end_date: str):
        """Parallel download function for ERDDAP data"""
        try:
            # ERDDAP primarily provides atmospheric data
            return self.erddap_downloader.download_weekly_atmospheric_data(start_date, end_date)
        except Exception as e:
            logger.error(f"Parallel ERDDAP download failed for {variable}: {e}")
            raise
            
    def collect_weekly_data(self, 
                          start_date: str,
                          end_date: str,
                          include_climatology: bool = False) -> Dict[str, xr.Dataset]:
        """
        Collect all weekly data from all sources
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            include_climatology: Whether to include climatology computation
            
        Returns:
            Dictionary with all collected datasets
        """
        logger.info(f"Starting weekly data collection from {start_date} to {end_date}")
        
        collected_data = {}
        
        try:
            # Collect Copernicus Marine data
            logger.info("Collecting Copernicus Marine data...")
            copernicus_data = self.copernicus_downloader.download_weekly_data(
                start_date, end_date, "both"
            )
            collected_data.update(copernicus_data)
            
            # Collect ERDDAP atmospheric data
            logger.info("Collecting ERDDAP atmospheric data...")
            atmospheric_data = self.erddap_downloader.download_weekly_atmospheric_data(
                start_date, end_date
            )
            collected_data["atmospheric"] = atmospheric_data
            
            # Collect climatology if requested
            if include_climatology and self.gee_available:
                logger.info("Computing climatology...")
                climatology_data = self.gee_climatology.compute_climatology()
                collected_data["climatology"] = climatology_data
            elif include_climatology and not self.gee_available:
                logger.warning("Climatology requested but GEE not available")
                
            logger.info(f"Successfully collected data from {len(collected_data)} sources")
            return collected_data
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            raise
            
    def collect_data_with_fallback(self,
                                 start_date: str,
                                 end_date: str) -> Dict[str, xr.Dataset]:
        """
        Collect data with fallback mechanisms
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with collected datasets (including fallbacks)
        """
        logger.info("Starting data collection with fallback mechanisms")
        
        collected_data = {}
        
        # Try primary data sources
        try:
            # Copernicus Marine data
            copernicus_data = self.copernicus_downloader.download_weekly_data(
                start_date, end_date, "both"
            )
            collected_data.update(copernicus_data)
            logger.info("Successfully collected Copernicus Marine data")
            
        except Exception as e:
            logger.warning(f"Copernicus Marine data collection failed: {e}")
            logger.info("Attempting fallback data collection...")
            
            # Try fallback data
            fallback_data = self.erddap_downloader.get_fallback_data(
                start_date, end_date, primary_data_failed=True
            )
            collected_data.update(fallback_data)
            
        # Atmospheric data (ERDDAP)
        try:
            atmospheric_data = self.erddap_downloader.download_weekly_atmospheric_data(
                start_date, end_date
            )
            collected_data["atmospheric"] = atmospheric_data
            logger.info("Successfully collected atmospheric data")
            
        except Exception as e:
            logger.error(f"Atmospheric data collection failed: {e}")
            
        return collected_data
        
    def validate_collected_data(self, collected_data: Dict[str, xr.Dataset]) -> Dict[str, bool]:
        """
        Validate all collected data
        
        Args:
            collected_data: Dictionary with collected datasets
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating collected data")
        
        validation_results = {}
        
        for data_type, dataset in collected_data.items():
            try:
                if data_type in ["physics", "ocean_color"]:
                    is_valid = self.copernicus_downloader.validate_data_quality(dataset, data_type)
                elif data_type in ["atmospheric", "gfs"]:
                    is_valid = self.erddap_downloader.validate_data_quality(dataset, data_type)
                elif data_type == "climatology":
                    is_valid = self.gee_climatology.validate_climatology_data(dataset)
                else:
                    # For fallback data, use basic validation
                    is_valid = self._basic_data_validation(dataset)
                    
                validation_results[data_type] = is_valid
                logger.info(f"{data_type} validation: {'PASS' if is_valid else 'FAIL'}")
                
            except Exception as e:
                logger.error(f"Validation failed for {data_type}: {e}")
                validation_results[data_type] = False
                
        return validation_results
        
    def _basic_data_validation(self, dataset: xr.Dataset) -> bool:
        """Basic data validation for fallback datasets"""
        try:
            # Check if dataset has data
            if dataset is None:
                return False
                
            # Check for spatial dimensions
            if "latitude" not in dataset.dims or "longitude" not in dataset.dims:
                return False
                
            # Check for data variables
            if len(dataset.data_vars) == 0:
                return False
                
            return True
            
        except Exception:
            return False
            
    def save_collected_data(self, 
                          collected_data: Dict[str, xr.Dataset],
                          timestamp: str = None) -> Dict[str, Path]:
        """
        Save collected data to files
        
        Args:
            collected_data: Dictionary with collected datasets
            timestamp: Timestamp for file naming
            
        Returns:
            Dictionary with saved file paths
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        logger.info(f"Saving collected data with timestamp: {timestamp}")
        
        saved_files = {}
        
        for data_type, dataset in collected_data.items():
            try:
                # Create filename
                filename = f"{data_type}_{timestamp}.nc"
                file_path = self.raw_dir / data_type / filename
                
                # Ensure directory exists
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save dataset
                dataset.to_netcdf(file_path)
                saved_files[data_type] = file_path
                
                logger.info(f"Saved {data_type} data to: {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to save {data_type} data: {e}")
                
        return saved_files
        
    def create_data_collection_report(self,
                                    collected_data: Dict[str, xr.Dataset],
                                    validation_results: Dict[str, bool],
                                    saved_files: Dict[str, Path]) -> Dict:
        """
        Create a comprehensive data collection report
        
        Args:
            collected_data: Dictionary with collected datasets
            validation_results: Dictionary with validation results
            saved_files: Dictionary with saved file paths
            
        Returns:
            Data collection report
        """
        logger.info("Creating data collection report")
        
        report = {
            "collection_date": datetime.now().isoformat(),
            "spatial_bounds": self.spatial_bounds,
            "temporal_settings": self.temporal_settings,
            "data_sources": {},
            "validation_summary": {},
            "file_locations": {},
            "statistics": {}
        }
        
        # Data sources information
        for data_type, dataset in collected_data.items():
            report["data_sources"][data_type] = {
                "dimensions": dict(dataset.dims),
                "variables": list(dataset.data_vars.keys()),
                "time_range": {
                    "start": str(dataset.time.min().values) if "time" in dataset.dims else None,
                    "end": str(dataset.time.max().values) if "time" in dataset.dims else None
                }
            }
            
        # Validation summary
        report["validation_summary"] = validation_results
        report["validation_summary"]["overall_success"] = all(validation_results.values())
        
        # File locations
        report["file_locations"] = {k: str(v) for k, v in saved_files.items()}
        
        # Statistics
        total_datasets = len(collected_data)
        successful_validations = sum(validation_results.values())
        report["statistics"] = {
            "total_datasets": total_datasets,
            "successful_validations": successful_validations,
            "validation_success_rate": successful_validations / total_datasets if total_datasets > 0 else 0
        }
        
        return report
        
    def save_collection_report(self, report: Dict, filename: str = None) -> Path:
        """Save data collection report to file"""
        import json
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_collection_report_{timestamp}.json"
            
        report_file = self.data_dir / filename
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Data collection report saved to: {report_file}")
        return report_file
        
    def run_complete_collection_pipeline(self,
                                       start_date: str,
                                       end_date: str,
                                       include_climatology: bool = False,
                                       use_fallback: bool = True) -> Dict:
        """
        Run complete data collection pipeline
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            include_climatology: Whether to include climatology computation
            use_fallback: Whether to use fallback mechanisms
            
        Returns:
            Complete pipeline results
        """
        logger.info("Starting complete data collection pipeline")
        
        pipeline_start_time = time.time()
        
        try:
            # Collect data
            if use_fallback:
                collected_data = self.collect_data_with_fallback(start_date, end_date)
            else:
                collected_data = self.collect_weekly_data(start_date, end_date, include_climatology)
                
            # Validate data
            validation_results = self.validate_collected_data(collected_data)
            
            # Save data
            saved_files = self.save_collected_data(collected_data)
            
            # Create report
            report = self.create_data_collection_report(
                collected_data, validation_results, saved_files
            )
            
            # Save report
            report_file = self.save_collection_report(report)
            
            pipeline_end_time = time.time()
            pipeline_duration = pipeline_end_time - pipeline_start_time
            
            # Add pipeline metadata
            report["pipeline_metadata"] = {
                "start_time": pipeline_start_time,
                "end_time": pipeline_end_time,
                "duration_seconds": pipeline_duration,
                "success": True
            }
            
            logger.info(f"Data collection pipeline completed in {pipeline_duration:.2f} seconds")
            
            return {
                "collected_data": collected_data,
                "validation_results": validation_results,
                "saved_files": saved_files,
                "report": report,
                "report_file": report_file,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Data collection pipeline failed: {e}")
            
            pipeline_end_time = time.time()
            pipeline_duration = pipeline_end_time - pipeline_start_time
            
            return {
                "collected_data": {},
                "validation_results": {},
                "saved_files": {},
                "report": {
                    "pipeline_metadata": {
                        "start_time": pipeline_start_time,
                        "end_time": pipeline_end_time,
                        "duration_seconds": pipeline_duration,
                        "success": False,
                        "error": str(e)
                    }
                },
                "success": False,
                "error": str(e)
            }
            
    def get_available_data_sources(self) -> Dict:
        """Get information about all available data sources"""
        return {
            "copernicus_marine": self.copernicus_downloader.get_available_datasets(),
            "erddap": self.erddap_downloader.get_available_datasets(),
            "google_earth_engine": self.gee_climatology.get_climatology_info()
        }
        
    def run_historical_collection(self,
                                period_name: str,
                                start_date: str = None,
                                end_date: str = None,
                                include_climatology: bool = False) -> Dict:
        """
        Run historical data collection for Seto Inland Sea
        
        Args:
            period_name: Name of the collection period
            start_date: Start date (if not using predefined period)
            end_date: End date (if not using predefined period)
            include_climatology: Whether to include climatology computation
            
        Returns:
            Historical collection results
        """
        logger.info(f"Starting historical collection for Seto Inland Sea: {period_name}")
        
        return self.historical_collector.run_historical_collection(
            period_name=period_name,
            start_date=start_date,
            end_date=end_date,
            include_climatology=include_climatology
        )
        
    def get_historical_collection_status(self) -> Dict:
        """Get status of historical data collections"""
        return self.historical_collector.get_historical_collection_status()
        
    def collect_data_with_parallel_processing(self,
                                            start_date: str,
                                            end_date: str,
                                            variables: Optional[List[str]] = None,
                                            preferences: Optional[Dict] = None,
                                            max_workers: int = 5) -> Dict[str, Any]:
        """
        Collect data using parallel processing for optimal performance
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            variables: List of variables to collect (None = all available)
            preferences: Collection preferences (resolution, quality, etc.)
            max_workers: Maximum number of concurrent download threads
            
        Returns:
            Dictionary with collected data and performance metrics
        """
        logger.info(f"Starting parallel data collection from {start_date} to {end_date}")
        
        # Update parallel download manager workers if needed
        if max_workers != self.parallel_download_manager.max_workers:
            self.parallel_download_manager.max_workers = max_workers
            logger.info(f"Updated parallel download manager to {max_workers} workers")
        
        # Get variables to collect
        if variables is None:
            variables = self.dataset_selector.get_all_variables()
        
        # Create batch configuration
        batch_id = f"parallel_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_config = self.batch_scheduler.create_batch_config(
            batch_id=batch_id,
            max_concurrent=max_workers,
            priority_order=['copernicus', 'gee', 'erddap']
        )
        
        # Create download tasks
        download_tasks = []
        for variable in variables:
            # Get optimal dataset for this variable
            selected_dataset = self.dataset_selector.select_optimal_dataset(
                variable, start_date, end_date, preferences
            )
            
            if selected_dataset:
                task = DownloadTask(
                    task_id=f"{variable}_{selected_dataset['source']}_{start_date}",
                    data_source=selected_dataset['source'],
                    variable=variable,
                    start_date=start_date,
                    end_date=end_date,
                    priority=selected_dataset.get('priority', 1),
                    timeout=selected_dataset.get('timeout', 300)
                )
                download_tasks.append(task)
                logger.debug(f"Created download task for {variable} from {selected_dataset['source']}")
            else:
                logger.warning(f"No optimal dataset found for variable {variable}")
        
        # Schedule batch download
        if not self.batch_scheduler.schedule_batch_download(batch_id, download_tasks):
            raise RuntimeError(f"Failed to schedule batch download {batch_id}")
        
        # Process downloads
        start_time = time.time()
        results = self.parallel_download_manager.process_download_queue(timeout=1800)  # 30 minutes timeout
        processing_time = time.time() - start_time
        
        # Collect successful downloads
        collected_data = {}
        for task_id, result in self.parallel_download_manager.completed_tasks.items():
            if result.success and result.data is not None:
                # Extract variable name from task_id
                variable = task_id.split('_')[0]
                collected_data[variable] = result.data
        
        # Generate collection report
        collection_report = {
            'batch_id': batch_id,
            'start_date': start_date,
            'end_date': end_date,
            'variables_requested': len(variables),
            'tasks_created': len(download_tasks),
            'successful_downloads': len(collected_data),
            'failed_downloads': len(self.parallel_download_manager.failed_tasks),
            'total_processing_time': processing_time,
            'average_download_time': results.get('average_download_time', 0),
            'success_rate': results.get('success_rate', 0),
            'performance_metrics': self.parallel_download_manager.get_performance_metrics(),
            'batch_status': self.batch_scheduler.get_batch_status(batch_id)
        }
        
        logger.info(f"Parallel collection completed: {len(collected_data)}/{len(variables)} variables in {processing_time:.2f}s")
        
        return {
            'collected_data': collected_data,
            'collection_report': collection_report
        }
    
    def collect_data_with_intelligent_selection(self,
                                              start_date: str,
                                              end_date: str,
                                              variables: Optional[List[str]] = None,
                                              preferences: Optional[Dict] = None) -> Dict[str, xr.Dataset]:
        """
        Collect data using intelligent dataset selection based on temporal coverage.
        
        This method uses the DatasetSelector to choose optimal datasets for each variable
        based on temporal coverage, data quality, and user preferences.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            variables: List of variables to collect (if None, collects all available)
            preferences: Optional preferences dict with keys:
                - prefer_high_resolution: bool
                - prefer_low_latency: bool
                - prefer_historical_quality: bool
                - coastal_focus: bool
                
        Returns:
            Dictionary with collected datasets
        """
        logger.info(f"Starting intelligent data collection from {start_date} to {end_date}")
        
        # Get temporal coverage map to understand available datasets
        coverage_map = self.dataset_selector.get_temporal_coverage_map()
        logger.info(f"Available variables: {list(coverage_map.keys())}")
        
        # If no specific variables requested, use all available
        if variables is None:
            variables = list(coverage_map.keys())
            
        collected_data = {}
        selection_log = {}
        
        for variable in variables:
            try:
                logger.info(f"Selecting optimal dataset for variable: {variable}")
                
                # Use dataset selector to choose optimal dataset
                selected_dataset = self.dataset_selector.select_optimal_dataset(
                    variable=variable,
                    start_date=start_date,
                    end_date=end_date,
                    preferences=preferences
                )
                
                selection_log[variable] = selected_dataset
                logger.info(f"Selected dataset for {variable}: {selected_dataset['dataset_id']}")
                
                # Collect data based on the selected dataset
                dataset_data = self._collect_variable_data(
                    variable=variable,
                    selected_dataset=selected_dataset,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if dataset_data is not None:
                    collected_data[variable] = dataset_data
                    logger.info(f"Successfully collected data for {variable}")
                else:
                    logger.warning(f"Failed to collect data for {variable}")
                    
            except Exception as e:
                logger.error(f"Error collecting data for {variable}: {e}")
                selection_log[variable] = {"error": str(e)}
                
        # Create collection report
        collection_report = {
            "collection_metadata": {
                "start_date": start_date,
                "end_date": end_date,
                "variables_requested": variables,
                "variables_collected": list(collected_data.keys()),
                "success_rate": len(collected_data) / len(variables) if variables else 0,
                "preferences_used": preferences
            },
            "dataset_selections": selection_log,
            "temporal_coverage": coverage_map
        }
        
        logger.info(f"Intelligent data collection completed. Success rate: {collection_report['collection_metadata']['success_rate']:.2%}")
        
        return collected_data, collection_report
        
    def _collect_variable_data(self,
                              variable: str,
                              selected_dataset: Dict,
                              start_date: str,
                              end_date: str) -> Optional[xr.Dataset]:
        """
        Collect data for a specific variable using the selected dataset.
        
        Args:
            variable: Variable name
            selected_dataset: Dataset selection result from DatasetSelector
            start_date: Start date
            end_date: End date
            
        Returns:
            Collected dataset or None if collection failed
        """
        dataset_info = selected_dataset
        source = dataset_info.get('source', 'unknown')
        
        try:
            if source == 'copernicus':
                return self._collect_copernicus_data(
                    variable, dataset_info, start_date, end_date
                )
            elif source == 'gee':
                return self._collect_gee_data(
                    variable, dataset_info, start_date, end_date
                )
            elif source == 'erddap':
                return self._collect_erddap_data(
                    variable, dataset_info, start_date, end_date
                )
            else:
                logger.warning(f"Unknown data source: {source}")
                return None
                
        except Exception as e:
            logger.error(f"Error collecting {variable} from {source}: {e}")
            return None
            
    def _collect_copernicus_data(self,
                                variable: str,
                                dataset_info: Dict,
                                start_date: str,
                                end_date: str) -> Optional[xr.Dataset]:
        """Collect data from Copernicus Marine service."""
        if isinstance(dataset_info, str):
            logger.error(f"dataset_info is a string: {dataset_info}")
            return None
            
        # Check if required keys exist
        if 'dataset_id' not in dataset_info:
            logger.error(f"Missing 'dataset_id' in dataset_info: {dataset_info}")
            return None
        if 'variable_name' not in dataset_info:
            logger.error(f"Missing 'variable_name' in dataset_info: {dataset_info}")
            return None
            
        dataset_id = dataset_info['dataset_id']
        var_name = dataset_info['variable_name']
        
        # Map variable names to downloader methods
        variable_methods = {
            'sea_surface_temperature': self.copernicus_downloader.download_sst_data,
            'chlorophyll_a': self.copernicus_downloader.download_ocean_color_data,
            'turbidity': self.copernicus_downloader.download_turbidity_data,
            'mixed_layer_depth': self.copernicus_downloader.download_physics_data,
            'sea_level_anomaly': self.copernicus_downloader.download_physics_data,
            'nitrate': self.copernicus_downloader.download_biogeochemistry_data,
            'phosphate': self.copernicus_downloader.download_biogeochemistry_data,
            'silicate': self.copernicus_downloader.download_biogeochemistry_data,
            'dissolved_iron': self.copernicus_downloader.download_biogeochemistry_data,
            'dissolved_oxygen': self.copernicus_downloader.download_biogeochemistry_data,
            'primary_production': self.copernicus_downloader.download_biogeochemistry_data,
            'total_chlorophyll': self.copernicus_downloader.download_biogeochemistry_data,
            'phytoplankton_carbon': self.copernicus_downloader.download_biogeochemistry_data,
            'diffuse_attenuation': self.copernicus_downloader.download_biogeochemistry_data
        }
        
        if variable in variable_methods:
            method = variable_methods[variable]
            return method(start_date, end_date)
        else:
            logger.warning(f"No collection method found for Copernicus variable: {variable}")
            return None
            
    def _collect_gee_data(self,
                         variable: str,
                         dataset_info: Dict,
                         start_date: str,
                         end_date: str) -> Optional[xr.Dataset]:
        """Collect data from Google Earth Engine."""
        if not self.gee_available:
            logger.warning(f"GEE not available for {variable} collection")
            return None
            
        try:
            # Import GEE integrator
            from .gee_integrator import GEEIntegrator
            
            # Initialize GEE integrator
            gee_integrator = GEEIntegrator()
            
            # Map variable names to GEE integrator methods
            variable_methods = {
                'wind_components': gee_integrator.get_wind_data,
                'precipitation': gee_integrator.get_precipitation_data,
                'ocean_temperature': gee_integrator.get_sst_data,
                'ocean_salinity': lambda s, e: gee_integrator.get_comprehensive_gee_data(s, e).get('hycom_temperature_salinity'),
                'ocean_currents': lambda s, e: gee_integrator.get_comprehensive_gee_data(s, e).get('hycom_velocity'),
                'sea_surface_elevation': lambda s, e: gee_integrator.get_comprehensive_gee_data(s, e).get('hycom_elevation'),
                'chlorophyll_a': gee_integrator.get_chlorophyll_data,
                'turbidity': gee_integrator.get_sentinel2_turbidity_data
            }
            
            if variable in variable_methods:
                method = variable_methods[variable]
                result = method(start_date, end_date)
                
                # Convert result to xarray Dataset if it's not already
                if isinstance(result, dict) and 'data' in result:
                    # GEE integrator returns dict with 'data' key
                    return result['data']
                elif hasattr(result, 'to_dataset'):
                    # If it's a pandas DataFrame, convert to xarray
                    return result.to_dataset()
                else:
                    return result
            else:
                logger.warning(f"No collection method found for GEE variable: {variable}")
                return None
                
        except Exception as e:
            logger.error(f"Error collecting GEE data for {variable}: {e}")
            return None
        
    def _collect_erddap_data(self,
                            variable: str,
                            dataset_info: Dict,
                            start_date: str,
                            end_date: str) -> Optional[xr.Dataset]:
        """Collect data from ERDDAP services."""
        try:
            # Map variable names to ERDDAP downloader methods
            variable_methods = {
                'wind_components': self.erddap_downloader.download_weekly_atmospheric_data,
                'precipitation': self.erddap_downloader.download_weekly_atmospheric_data,
                'atmospheric': self.erddap_downloader.download_weekly_atmospheric_data
            }
            
            if variable in variable_methods:
                method = variable_methods[variable]
                result = method(start_date, end_date)
                
                # Convert result to xarray Dataset if it's not already
                if isinstance(result, dict) and 'data' in result:
                    return result['data']
                elif hasattr(result, 'to_dataset'):
                    return result.to_dataset()
                else:
                    return result
            else:
                logger.warning(f"No collection method found for ERDDAP variable: {variable}")
                return None
                
        except Exception as e:
            logger.error(f"Error collecting ERDDAP data for {variable}: {e}")
            return None
    
    def _start_quality_monitoring(self):
        """Start the quality monitoring system"""
        try:
            logger.info("Starting quality monitoring system")
            success = self.quality_monitor.start_monitoring()
            if success:
                logger.info("Quality monitoring system started successfully")
            else:
                logger.warning("Failed to start quality monitoring system")
        except Exception as e:
            logger.error(f"Error starting quality monitoring: {e}")
    
    def stop_quality_monitoring(self):
        """Stop the quality monitoring system"""
        try:
            logger.info("Stopping quality monitoring system")
            success = self.quality_monitor.stop_monitoring()
            if success:
                logger.info("Quality monitoring system stopped successfully")
            else:
                logger.warning("Failed to stop quality monitoring system")
        except Exception as e:
            logger.error(f"Error stopping quality monitoring: {e}")
    
    def monitor_dataset_quality(self, dataset: xr.Dataset, data_source: str, variable: str = None) -> Dict[str, Any]:
        """
        Monitor quality of a specific dataset
        
        Args:
            dataset: Dataset to monitor
            data_source: Source of the data (copernicus, gee, erddap)
            variable: Specific variable being monitored
            
        Returns:
            Quality monitoring results
        """
        try:
            return self.quality_monitor.monitor_dataset_quality(dataset, data_source, variable)
        except Exception as e:
            logger.error(f"Error monitoring dataset quality: {e}")
            return {"error": str(e)}
    
    def get_quality_monitoring_status(self) -> Dict[str, Any]:
        """Get current quality monitoring status"""
        try:
            return self.quality_monitor.get_monitoring_status()
        except Exception as e:
            logger.error(f"Error getting quality monitoring status: {e}")
            return {"error": str(e)}
    
    def acknowledge_quality_alert(self, alert_index: int) -> bool:
        """Acknowledge a quality alert"""
        try:
            return self.quality_monitor.acknowledge_alert(alert_index)
        except Exception as e:
            logger.error(f"Error acknowledging quality alert: {e}")
            return False
    
    def resolve_quality_alert(self, alert_index: int) -> bool:
        """Resolve a quality alert"""
        try:
            return self.quality_monitor.resolve_alert(alert_index)
        except Exception as e:
            logger.error(f"Error resolving quality alert: {e}")
            return False
    
    def export_quality_monitoring_data(self, filepath: str):
        """Export quality monitoring data to file"""
        try:
            self.quality_monitor.export_monitoring_data(filepath)
        except Exception as e:
            logger.error(f"Error exporting quality monitoring data: {e}")
            raise
    
    def collect_data_with_caching(self, 
                                 start_date: str, 
                                 end_date: str, 
                                 variables: Optional[List[str]] = None,
                                 use_cache: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Collect data with intelligent caching for improved performance
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            variables: List of variables to collect (None = all available)
            use_cache: Whether to use caching (default: True)
            
        Returns:
            Tuple of (collected_data, collection_report)
        """
        logger.info(f"Starting cached data collection from {start_date} to {end_date}")
        
        collected_data = {}
        collection_report = {
            "start_date": start_date,
            "end_date": end_date,
            "variables_requested": variables,
            "cache_used": use_cache,
            "cache_hits": 0,
            "cache_misses": 0,
            "download_times": {},
            "errors": [],
            "total_size_mb": 0
        }
        
        # Get variables to collect
        if variables is None:
            variables = list(self.dataset_selector.get_available_variables())
        
        for variable in variables:
            try:
                start_time = time.time()
                
                # Select optimal dataset for this variable
                dataset_info = self.dataset_selector.select_optimal_dataset(variable, start_date, end_date)
                if not dataset_info:
                    logger.warning(f"No optimal dataset found for {variable}")
                    collection_report["errors"].append(f"No dataset for {variable}")
                    continue
                
                data_source = dataset_info["source"]
                
                if use_cache:
                    # Try to get from cache first
                    cached_data = self.data_cache.get(
                        data_source=data_source,
                        variable=variable,
                        start_date=start_date,
                        end_date=end_date,
                        dataset_id=dataset_info.get("dataset_id")
                    )
                    
                    if cached_data is not None:
                        collected_data[variable] = cached_data
                        collection_report["cache_hits"] += 1
                        download_time = time.time() - start_time
                        collection_report["download_times"][variable] = download_time
                        logger.info(f"Cache hit for {variable} (loaded in {download_time:.2f}s)")
                        continue
                    else:
                        collection_report["cache_misses"] += 1
                
                # Download fresh data
                logger.info(f"Downloading fresh data for {variable} from {data_source}")
                data = self._collect_variable_data(variable, dataset_info, start_date, end_date)
                
                if data is not None:
                    collected_data[variable] = data
                    
                    # Cache the result if caching is enabled
                    if use_cache:
                        self.data_cache.set(
                            data=data,
                            data_source=data_source,
                            variable=variable,
                            start_date=start_date,
                            end_date=end_date,
                            dataset_id=dataset_info.get("dataset_id")
                        )
                    
                    download_time = time.time() - start_time
                    collection_report["download_times"][variable] = download_time
                    
                    # Estimate data size
                    if hasattr(data, 'nbytes'):
                        size_mb = data.nbytes / (1024 * 1024)
                        collection_report["total_size_mb"] += size_mb
                    
                    logger.info(f"Downloaded {variable} in {download_time:.2f}s")
                else:
                    collection_report["errors"].append(f"Failed to download {variable}")
                    
            except Exception as e:
                error_msg = f"Error collecting {variable}: {str(e)}"
                logger.error(error_msg)
                collection_report["errors"].append(error_msg)
        
        # Add cache statistics to report
        if use_cache:
            cache_stats = self.data_cache.get_cache_stats()
            collection_report["cache_stats"] = cache_stats
        
        logger.info(f"Data collection completed: {len(collected_data)} variables, "
                   f"{collection_report['cache_hits']} cache hits, "
                   f"{collection_report['cache_misses']} cache misses")
        
        return collected_data, collection_report

    def get_cache_status(self) -> Dict[str, Any]:
        """
        Get current cache status and statistics
        
        Returns:
            Dictionary with cache statistics
        """
        return self.data_cache.get_cache_stats()
    
    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear expired cache files
        
        Args:
            older_than_days: Clear files older than this many days
            
        Returns:
            Number of files cleared
        """
        return self.data_cache.clear(older_than_days)
    
    def list_cached_variables(self) -> Dict[str, list]:
        """
        List all cached variables by data source
        
        Returns:
            Dictionary mapping data sources to lists of cached variables
        """
        return self.data_cache.list_cached_variables()
        
if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    orchestrator = DataCollectionOrchestrator()
    
    # Run complete pipeline for last week
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    try:
        results = orchestrator.run_complete_collection_pipeline(
            start_date, end_date, include_climatology=False, use_fallback=True
        )
        
        if results["success"]:
            print("Data collection pipeline completed successfully")
            print(f"Collected {len(results['collected_data'])} datasets")
            print(f"Validation success rate: {results['report']['statistics']['validation_success_rate']:.2%}")
        else:
            print(f"Pipeline failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Pipeline execution failed: {e}") 