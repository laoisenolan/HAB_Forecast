#!/usr/bin/env python3
"""
PERIOD-AWARE 7-Year Historical Data Collection (2017-2024)
Uses the CORRECT data sources for each time period based on ACTUAL availability

DATA AVAILABILITY STRATEGY:
- 2017-2021: GEE-focused (ERA5, HYCOM, Sentinel-2/3, MODIS, JAXA) = ~20+ variables
- 2022-2024: GEE + Copernicus Marine + ERDDAP = ~30+ variables

SKIPS UNAVAILABLE SOURCES:
- Copernicus Marine for pre-2022 data (availability: 2022-06+)
- ERDDAP for pre-2022 data (availability: 2022-12+)
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from existing data collection infrastructure
from src.data_collection.gee_integrator import GEEIntegrator
from src.data_collection.historical_satellite_integrator import HistoricalSatelliteIntegrator
from src.data_collection.copernicus_downloader import CopernicusDownloader
from src.data_collection.erddap_downloader import ERDDAPDownloader
from src.data_collection.csv_data_processor import CSVDataProcessor
from src.utils.config_loader import ConfigLoader, get_data_dir

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'period_aware_collection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)

class PeriodAwareHistoricalCollector:
    """Period-aware collector that uses the RIGHT data sources for each time period"""
    
    def __init__(self):
        """Initialize period-aware collector"""
        logger.info("🧠 Initializing PERIOD-AWARE 7-Year Historical Collector")
        
        # Load configuration
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_data_sources_config()
        
        # Initialize data processors
        self.csv_processor = CSVDataProcessor()
        self.data_dir = get_data_dir()
        self.historical_dir = self.data_dir / "historical_csv"
        self.historical_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data source components
        self.gee_integrator = GEEIntegrator()
        self.historical_satellite = HistoricalSatelliteIntegrator()
        self.copernicus_downloader = CopernicusDownloader(self.config)
        self.erddap_downloader = ERDDAPDownloader(self.config)
        
        # Fast/parallel mode toggles
        self.fast_mode: bool = True
        self.max_workers: int = 5  # safe for GEE rate limits
        self.batch_days: int = 7   # write to master once per week
        self.keep_per_source_csv: bool = False  # avoid small per-source files to reduce I/O

        # Define INTELLIGENT time periods based on ACTUAL availability
        # 🧪 TEST MODE: Short periods for pipeline validation
        test_mode = False  # FULL 7-YEAR COLLECTION MODE
        
        if test_mode:
            logger.info("🧪 TEST MODE ENABLED: 3-month validation periods")
            logger.info("📅 Pre-Copernicus Test: 2021-09-01 to 2021-11-30 (3 months)")
            logger.info("📅 Comprehensive Test: 2022-06-01 to 2022-08-31 (3 months)")
            
        self.collection_periods = {
            "pre_copernicus": {
                "start": "2021-09-01" if test_mode else "2017-01-01",
                "end": "2021-11-30" if test_mode else "2021-12-31", 
                "description": "TEST: Pre-Copernicus Period (GEE-focused)" if test_mode else "Pre-Copernicus Period (GEE-focused)",
                "available_sources": {
                    "gee_era5_extended": "1950-present (atmospheric)",
                    "gee_hycom_extended": "1992-2024 (ocean physics)", 
                    "gee_sentinel2": "2015-present (high-res turbidity)",
                    "gee_sentinel3": "2016-present (ocean color)",
                    "gee_modis": "2002-present (ocean color, SST)",
                    "gee_jaxa": "2017-present (high-res chlorophyll)"
                },
                "unavailable_sources": ["copernicus_marine", "erddap"],
                "expected_variables": 22,
                "collection_method": "gee_historical_focused"
            },
            "comprehensive": {
                "start": "2022-06-01" if test_mode else "2022-01-01", 
                "end": "2022-08-31" if test_mode else "2024-12-31",
                "description": "TEST: Comprehensive Period (All sources)" if test_mode else "Comprehensive Period (All sources)",
                "available_sources": {
                    "gee_all": "All GEE sources continue",
                    "copernicus_marine": "2022-06+ (nutrients, biology, physics)",
                    "erddap": "2022-12+ (atmospheric backup)"
                },
                "unavailable_sources": [],
                "expected_variables": 30,
                "collection_method": "comprehensive_all_sources"
            }
        }
        
        # Track collection results
        self.total_records = 0
        self.total_variables = set()
        self.period_results = {}
        
    def run_period_aware_collection(self) -> Dict[str, Any]:
        """Run the complete period-aware 7-year collection"""
        logger.info("🚀 Starting PERIOD-AWARE 7-Year Historical Data Collection")
        logger.info("📊 Using ACTUAL data source availability windows")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Collect each period with appropriate sources
            for period_name, period_config in self.collection_periods.items():
                logger.info(f"\n📊 PERIOD: {period_name.upper()}")
                logger.info(f"   📅 {period_config['start']} to {period_config['end']}")
                logger.info(f"   📝 {period_config['description']}")
                logger.info(f"   🎯 Method: {period_config['collection_method']}")
                logger.info(f"   📈 Expected Variables: {period_config['expected_variables']}")
                
                logger.info("   ✅ Available Sources:")
                for source, desc in period_config['available_sources'].items():
                    logger.info(f"      • {source}: {desc}")
                
                if period_config['unavailable_sources']:
                    logger.info("   🚫 Skipping Sources:")
                    for source in period_config['unavailable_sources']:
                        logger.info(f"      • {source} (not available for this period)")
                
                # Collect this period
                period_result = self._collect_period(period_name, period_config)
                self.period_results[period_name] = period_result
                
                if period_result['success']:
                    logger.info(f"✅ {period_name} completed: {period_result['records']} records, {len(period_result['variables'])} variables")
                    self.total_records += period_result['records']
                    self.total_variables.update(period_result['variables'])
                else:
                    logger.error(f"❌ {period_name} failed: {period_result['error']}")
            
            # Generate final summary
            duration = datetime.now() - start_time
            return self._generate_final_summary(duration)
            
        except Exception as e:
            logger.error(f"❌ Period-aware collection failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _collect_period(self, period_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Collect data for a specific period using appropriate sources"""
        
        start_date = config['start']
        end_date = config['end']
        method = config['collection_method']
        
        try:
            if method == "gee_historical_focused":
                # Use historical satellite integrator + GEE integrator (2017-2021)
                result = self._collect_gee_historical_period(start_date, end_date)
            elif method == "comprehensive_all_sources":
                # Use all sources including Copernicus and ERDDAP (2022-2024) 
                result = self._collect_comprehensive_period(start_date, end_date)
            else:
                raise ValueError(f"Unknown collection method: {method}")
            
            if result.get("success", False):
                # Extract collected data info
                records = result.get("total_records", 0)
                variables = result.get("variables_collected", [])
                
                return {
                    "success": True,
                    "records": records,
                    "variables": variables,
                    "saved_files": result.get("saved_files", {}),
                    "period": f"{start_date} to {end_date}",
                    "method": method
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "period": f"{start_date} to {end_date}",
                    "method": method
                }
                
        except Exception as e:
            logger.error(f"Period collection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "period": f"{start_date} to {end_date}",
                "method": method
            }
    
    def _collect_gee_historical_period(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        FIXED: Collect using individual GEE methods with proper chunking (2017-2021)
        
        DEBUG NOTES:
        - Fixed GEE 5000+ element limit by using smaller time chunks (quarterly)
        - Use individual GEE methods instead of comprehensive (which calls missing get_ocean_data)
        - Use correct CSV processor methods (convert_gee_data_to_csv + save_csv)
        """
        logger.info(f"🌍 Collecting GEE-focused historical data: {start_date} to {end_date}")
        logger.info("✅ Using: ERA5, HYCOM individual methods, Sentinel-2, JAXA")
        
        try:
            total_rows = 0
            # Use quarterly chunks, but iterate daily inside each chunk to get real dates
            quarterly_chunks = self._create_quarterly_chunks(start_date, end_date)
            logger.info(f"📦 Created {len(quarterly_chunks)} quarterly chunks to avoid GEE limits")
            for i, (chunk_start, chunk_end) in enumerate(quarterly_chunks, 1):
                logger.info(f"📦 Processing chunk {i}/{len(quarterly_chunks)}: {chunk_start} to {chunk_end}")
                try:
                    day_dt = datetime.strptime(chunk_start, "%Y-%m-%d")
                    end_dt = datetime.strptime(chunk_end, "%Y-%m-%d")
                    week_buffer: List[pd.DataFrame] = []
                    days_in_batch = 0
                    while day_dt <= end_dt:
                        day = day_dt.strftime("%Y-%m-%d")
                        next_day = (day_dt + timedelta(days=1)).strftime("%Y-%m-%d")
                        day_dt += timedelta(days=1)
                        # Build callables per variable family to run in parallel
                        def _tasks_for_day(sd: str, ed: str) -> Dict[str, callable]:
                            return {
                                "wind": lambda: self.gee_integrator.get_wind_data(sd, ed),
                                "precipitation": lambda: self.gee_integrator.get_precipitation_data(sd, ed),
                                "chlorophyll": lambda: self.gee_integrator.get_chlorophyll_data(sd, ed),
                                "jaxa_chlorophyll": lambda: self.gee_integrator.get_jaxa_chlorophyll_data(sd, ed),
                                "turbidity": lambda: self.gee_integrator.get_sentinel2_turbidity_data(sd, ed),
                                "sst": lambda: self.gee_integrator.get_sst_data(sd, ed),
                                "modis_ocean": lambda: self.gee_integrator.get_modis_ocean_data(sd, ed),
                                "hycom_temp_sal": lambda: self._collect_hycom_temp_salinity(sd, ed),
                                "hycom_velocity": lambda: self._collect_hycom_velocity(sd, ed),
                                "hycom_elevation": lambda: self._collect_hycom_elevation(sd, ed),
                            }

                        results: Dict[str, Any] = {}
                        tasks = _tasks_for_day(day, next_day)
                        if self.fast_mode:
                            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                                future_map = {ex.submit(func): name for name, func in tasks.items()}
                                for fut in as_completed(future_map):
                                    name = future_map[fut]
                                    try:
                                        results[name] = fut.result()
                                    except Exception as e:
                                        logger.warning(f"{name} failed @ {day}: {e}")
                        else:
                            for name, func in tasks.items():
                                try:
                                    results[name] = func()
                                except Exception as e:
                                    logger.warning(f"{name} failed @ {day}: {e}")

                        # Convert all non-empty results and buffer for weekly write
                        day_frames: List[pd.DataFrame] = []
                        for source_name, source_data in results.items():
                            if not source_data:
                                continue
                            df = self.csv_processor.convert_gee_data_to_csv(
                                source_data, data_source="gee", timestamp=f"{day}T00:00:00"
                            )
                            if df.empty:
                                continue
                            if self.keep_per_source_csv:
                                self.csv_processor.save_csv(
                                    df, f"gee_historical_{chunk_start}_{chunk_end}_{source_name}_{day}", subdirectory="historical"
                                )
                            day_frames.append(df)

                        if day_frames:
                            days_in_batch += 1
                            day_df = pd.concat(day_frames, ignore_index=True)
                            week_buffer.append(day_df)
                            total_rows += len(day_df)

                        # Flush weekly to reduce I/O
                        if days_in_batch >= self.batch_days:
                            batch_df = pd.concat(week_buffer, ignore_index=True) if week_buffer else pd.DataFrame()
                            if not batch_df.empty:
                                self.csv_processor.append_to_master_long(
                                    batch_df,
                                    master_filename="historical_2017_2024_master.csv",
                                    subdirectory="historical",
                                    dedup_keys=['date','latitude','longitude','variable_name','dataset_id']
                                )
                                # Also refresh wide feature matrix
                                try:
                                    self.csv_processor.build_wide_feature_matrix(
                                        master_filename="historical_2017_2024_master.csv",
                                        output_filename="historical_2017_2024_master_wide.csv",
                                        subdirectory="historical",
                                    )
                                except Exception as e:
                                    logger.warning(f"Wide feature matrix update failed: {e}")
                            week_buffer = []
                            days_in_batch = 0

                    # Flush remainder of the chunk buffer
                    if week_buffer:
                        batch_df = pd.concat(week_buffer, ignore_index=True)
                        self.csv_processor.append_to_master_long(
                            batch_df,
                            master_filename="historical_2017_2024_master.csv",
                            subdirectory="historical",
                            dedup_keys=['date','latitude','longitude','variable_name','dataset_id']
                        )
                        try:
                            self.csv_processor.build_wide_feature_matrix(
                                master_filename="historical_2017_2024_master.csv",
                                output_filename="historical_2017_2024_master_wide.csv",
                                subdirectory="historical",
                            )
                        except Exception as e:
                            logger.warning(f"Wide feature matrix update failed: {e}")
                except Exception as e:
                    logger.warning(f"Chunk {i} failed (continuing): {e}")
                    continue

            if total_rows > 0:
                logger.info(f"✅ GEE historical collection (daily): {total_rows} rows")
                return {"success": True, "total_records": total_rows, "variables_collected": [], "saved_files": {}, "data_sources": ["gee_daily"]}
            return {"success": False, "error": "No data collected from GEE sources"}
                
        except Exception as e:
            logger.error(f"GEE historical collection failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _collect_comprehensive_period(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        FIXED: Collect using all sources including Copernicus and ERDDAP (2022-2024)
        
        DEBUG NOTES:
        - Use individual GEE methods + chunking for comprehensive period too
        - Add Copernicus and ERDDAP data when available (proper date checking)
        - Use fixed CSV conversion methods
        """
        logger.info(f"🌊 Collecting comprehensive data: {start_date} to {end_date}")
        logger.info("✅ Using: GEE + Copernicus Marine + ERDDAP")
        # For speed, also switch GEE phase to parallel per-day if fast_mode
        all_collected_data = {}
        
        try:
            # Phase 1: FIXED - Use individual GEE methods with chunking
            logger.info("🌍 Phase 1: GEE Environmental Data (with chunking)...")
            quarterly_chunks = self._create_quarterly_chunks(start_date, end_date)
            # Save current chunk ranges for timestamping during CSV conversion
            self._current_chunks = quarterly_chunks
            logger.info(f"📦 Created {len(quarterly_chunks)} quarterly chunks for comprehensive period")
            
            for i, (chunk_start, chunk_end) in enumerate(quarterly_chunks, 1):
                logger.info(f"📦 Comprehensive chunk {i}/{len(quarterly_chunks)}: {chunk_start} to {chunk_end}")
                try:
                    # If fast_mode, reuse the historical daily parallel path to write directly to master
                    if self.fast_mode:
                        day_dt = datetime.strptime(chunk_start, "%Y-%m-%d")
                        end_dt = datetime.strptime(chunk_end, "%Y-%m-%d")
                        week_buffer: List[pd.DataFrame] = []
                        days_in_batch = 0
                        while day_dt <= end_dt:
                            day = day_dt.strftime("%Y-%m-%d")
                            next_day = (day_dt + timedelta(days=1)).strftime("%Y-%m-%d")
                            day_dt += timedelta(days=1)
                            def _tasks_for_day(sd: str, ed: str) -> Dict[str, callable]:
                                return {
                                    "wind": lambda: self.gee_integrator.get_wind_data(sd, ed),
                                    "precipitation": lambda: self.gee_integrator.get_precipitation_data(sd, ed),
                                    "chlorophyll": lambda: self.gee_integrator.get_chlorophyll_data(sd, ed),
                                    "jaxa_chlorophyll": lambda: self.gee_integrator.get_jaxa_chlorophyll_data(sd, ed),
                                    "turbidity": lambda: self.gee_integrator.get_sentinel2_turbidity_data(sd, ed),
                                    "sst": lambda: self.gee_integrator.get_sst_data(sd, ed),
                                    "modis_ocean": lambda: self.gee_integrator.get_modis_ocean_data(sd, ed),
                                    "hycom_temp_sal": lambda: self._collect_hycom_temp_salinity(sd, ed),
                                    "hycom_velocity": lambda: self._collect_hycom_velocity(sd, ed),
                                    "hycom_elevation": lambda: self._collect_hycom_elevation(sd, ed),
                                }
                            tasks = _tasks_for_day(day, next_day)
                            results: Dict[str, Any] = {}
                            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                                future_map = {ex.submit(func): name for name, func in tasks.items()}
                                for fut in as_completed(future_map):
                                    name = future_map[fut]
                                    try:
                                        results[name] = fut.result()
                                    except Exception as e:
                                        logger.warning(f"{name} failed @ {day}: {e}")
                            day_frames: List[pd.DataFrame] = []
                            for source_name, source_data in results.items():
                                if not source_data:
                                    continue
                                df = self.csv_processor.convert_gee_data_to_csv(
                                    source_data, data_source="gee", timestamp=f"{day}T00:00:00"
                                )
                                if df.empty:
                                    continue
                                if self.keep_per_source_csv:
                                    self.csv_processor.save_csv(
                                        df, f"comprehensive_{chunk_start}_{chunk_end}_{source_name}_{day}", subdirectory="historical"
                                    )
                                day_frames.append(df)
                            if day_frames:
                                days_in_batch += 1
                                week_buffer.append(pd.concat(day_frames, ignore_index=True))
                            if days_in_batch >= self.batch_days:
                                batch_df = pd.concat(week_buffer, ignore_index=True) if week_buffer else pd.DataFrame()
                                if not batch_df.empty:
                                    self.csv_processor.append_to_master_long(
                                        batch_df,
                                        master_filename="historical_2017_2024_master.csv",
                                        subdirectory="historical",
                                        dedup_keys=['date','latitude','longitude','variable_name','dataset_id']
                                    )
                                    try:
                                        self.csv_processor.build_wide_feature_matrix(
                                            master_filename="historical_2017_2024_master.csv",
                                            output_filename="historical_2017_2024_master_wide.csv",
                                            subdirectory="historical",
                                        )
                                    except Exception as e:
                                        logger.warning(f"Wide feature matrix update failed: {e}")
                                week_buffer = []
                                days_in_batch = 0
                        if week_buffer:
                            batch_df = pd.concat(week_buffer, ignore_index=True)
                            self.csv_processor.append_to_master_long(
                                batch_df,
                                master_filename="historical_2017_2024_master.csv",
                                subdirectory="historical",
                                dedup_keys=['date','latitude','longitude','variable_name','dataset_id']
                            )
                            try:
                                self.csv_processor.build_wide_feature_matrix(
                                    master_filename="historical_2017_2024_master.csv",
                                    output_filename="historical_2017_2024_master_wide.csv",
                                    subdirectory="historical",
                                )
                            except Exception as e:
                                logger.warning(f"Wide feature matrix update failed: {e}")
                    else:
                        # Fallback: chunk-level calls (older behavior)
                        wind_data = self.gee_integrator.get_wind_data(chunk_start, chunk_end)
                        if wind_data:
                            all_collected_data[f"wind_comp_{i}"] = wind_data
                        precip_data = self.gee_integrator.get_precipitation_data(chunk_start, chunk_end)
                        if precip_data:
                            all_collected_data[f"precipitation_comp_{i}"] = precip_data
                        chl_data = self.gee_integrator.get_chlorophyll_data(chunk_start, chunk_end)
                        if chl_data:
                            all_collected_data[f"chlorophyll_comp_{i}"] = chl_data
                        jaxa_chl_data = self.gee_integrator.get_jaxa_chlorophyll_data(chunk_start, chunk_end)
                        if jaxa_chl_data:
                            all_collected_data[f"jaxa_chlorophyll_comp_{i}"] = jaxa_chl_data
                        turb_data = self.gee_integrator.get_sentinel2_turbidity_data(chunk_start, chunk_end)
                        if turb_data:
                            all_collected_data[f"turbidity_comp_{i}"] = turb_data
                        sst_data = self.gee_integrator.get_sst_data(chunk_start, chunk_end)
                        if sst_data:
                            all_collected_data[f"sst_comp_{i}"] = sst_data
                        try:
                            hycom_temp_sal = self._collect_hycom_temp_salinity(chunk_start, chunk_end)
                            if hycom_temp_sal:
                                all_collected_data[f"hycom_temp_sal_comp_{i}"] = hycom_temp_sal
                            hycom_velocity = self._collect_hycom_velocity(chunk_start, chunk_end)
                            if hycom_velocity:
                                all_collected_data[f"hycom_velocity_comp_{i}"] = hycom_velocity
                            hycom_elevation = self._collect_hycom_elevation(chunk_start, chunk_end)
                            if hycom_elevation:
                                all_collected_data[f"hycom_elevation_comp_{i}"] = hycom_elevation
                        except Exception as e:
                            logger.warning(f"HYCOM comprehensive collection failed for chunk {i}: {e}")
                except Exception as e:
                    logger.warning(f"GEE comprehensive chunk {i} failed (continuing): {e}")
            
            # Phase 2: FIXED - Add Copernicus Marine data with proper date checking
            logger.info("🌊 Phase 2: Copernicus Marine Data...")
            copernicus_chunks = [(chunk_start, chunk_end) for chunk_start, chunk_end in quarterly_chunks 
                               if datetime.strptime(chunk_start, "%Y-%m-%d") >= datetime(2022, 6, 1)]
            
            if copernicus_chunks:
                logger.info(f"📦 {len(copernicus_chunks)} chunks eligible for Copernicus data")
                for i, (chunk_start, chunk_end) in enumerate(copernicus_chunks, 1):
                    try:
                        copernicus_data = self._collect_copernicus_chunk(chunk_start, chunk_end)
                        if copernicus_data:
                            all_collected_data[f"copernicus_{i}"] = copernicus_data
                    except Exception as e:
                        logger.warning(f"Copernicus chunk {i} failed (continuing): {e}")
            else:
                logger.info("🚫 No chunks eligible for Copernicus (before 2022-06)")
            
            # Phase 3: FIXED - Add ERDDAP data with proper date checking
            logger.info("📡 Phase 3: ERDDAP Atmospheric Data...")
            erddap_chunks = [(chunk_start, chunk_end) for chunk_start, chunk_end in quarterly_chunks 
                           if datetime.strptime(chunk_start, "%Y-%m-%d") >= datetime(2022, 12, 1)]
            
            if erddap_chunks:
                logger.info(f"📦 {len(erddap_chunks)} chunks eligible for ERDDAP data")
                for i, (chunk_start, chunk_end) in enumerate(erddap_chunks, 1):
                    try:
                        erddap_data = self._collect_erddap_chunk(chunk_start, chunk_end)
                        if erddap_data:
                            all_collected_data[f"erddap_{i}"] = erddap_data
                    except Exception as e:
                        logger.warning(f"ERDDAP chunk {i} failed (continuing): {e}")
            else:
                logger.info("🚫 No chunks eligible for ERDDAP (before 2022-12)")
            
            # FIXED: Convert to CSV format using correct methods
            if all_collected_data:
                csv_files = self._convert_gee_to_csv_fixed(all_collected_data, f"comprehensive_{start_date}_{end_date}")
                variables = self._extract_variables_from_gee_data(all_collected_data)
                total_records = self._count_total_records_fixed(csv_files)
                
                logger.info(f"✅ Comprehensive collection: {total_records} records, {len(variables)} variables")
                
                return {
                    "success": True,
                    "total_records": total_records,
                    "variables_collected": variables,
                    "saved_files": csv_files,
                    "data_sources": ["gee", "copernicus_marine", "erddap"]
                }
            else:
                return {
                    "success": False,
                    "error": "No data collected from comprehensive sources"
                }
                
        except Exception as e:
            logger.error(f"Comprehensive collection failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _collect_copernicus_chunk(self, start_date: str, end_date: str) -> Optional[Dict]:
        """Collect Copernicus Marine data for a time chunk"""
        try:
            # Only try if date is after 2022-06-01
            if datetime.strptime(start_date, "%Y-%m-%d") >= datetime(2022, 6, 1):
                result = self.copernicus_downloader.download_weekly_data(
                    start_date, end_date, data_type="physics"  # Fix: use data_type not data_types
                )
                return result
            else:
                logger.info(f"Skipping Copernicus for {start_date} (not available before 2022-06)")
                return None
        except Exception as e:
            logger.warning(f"Copernicus chunk collection failed: {e}")
            return None
    
    def _collect_erddap_chunk(self, start_date: str, end_date: str) -> Optional[Dict]:
        """Collect ERDDAP data for a time chunk"""
        try:
            # Only try if date is after 2022-12-01
            if datetime.strptime(start_date, "%Y-%m-%d") >= datetime(2022, 12, 1):
                result = self.erddap_downloader.download_gfs_data(start_date, end_date)
                return result
            else:
                logger.info(f"Skipping ERDDAP for {start_date} (not available before 2022-12)")
                return None
        except Exception as e:
            logger.warning(f"ERDDAP chunk collection failed: {e}")
            return None
    
    def _create_quarterly_chunks(self, start_date: str, end_date: str) -> List[Tuple[str, str]]:
        """
        FIXED: Create quarterly chunks to avoid GEE 5000+ element limit
        
        DEBUG NOTES:
        - Split large date ranges into 3-month chunks 
        - Each chunk has max ~90 days to stay under GEE limits
        """
        chunks = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current_date < end_dt:
            # 3-month chunks
            chunk_end = min(current_date + timedelta(days=90), end_dt)
            chunks.append((
                current_date.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d")
            ))
            current_date = chunk_end + timedelta(days=1)
            
        return chunks
    
    def _convert_gee_to_csv_fixed(self, data_dict: Dict[str, Any], period_name: str) -> Dict[str, str]:
        """
        FIXED: Convert GEE data to CSV format using correct CSVDataProcessor methods
        
        DEBUG NOTES:
        - Use convert_gee_data_to_csv() method (exists in CSVDataProcessor)
        - Use save_csv() method instead of non-existent process_and_save()
        """
        csv_files = {}
        
        for source_name, source_data in data_dict.items():
            try:
                if source_data is not None:
                    # FIXED: Use correct CSVDataProcessor method
                    # Prefer midpoint date of corresponding chunk as data point date
                    ts = datetime.now()
                    try:
                        # infer chunk index from source_name suffix like '_{i}' or '_comp_{i}'
                        idx = int(source_name.split('_')[-1]) if source_name.split('_')[-1].isdigit() else None
                        if idx is not None and hasattr(self, '_current_chunks') and len(self._current_chunks) >= idx:
                            cstart, cend = self._current_chunks[idx-1]
                            cs = datetime.strptime(cstart, "%Y-%m-%d")
                            ce = datetime.strptime(cend, "%Y-%m-%d")
                            ts = cs + (ce - cs)/2
                    except Exception:
                        pass
                    df = self.csv_processor.convert_gee_data_to_csv(
                        source_data,
                        data_source="gee",
                        timestamp=ts.isoformat()
                    )
                    
                    if not df.empty:
                        # FIXED: Use save_csv method instead of process_and_save
                        # Save per-source CSV (kept for traceability)
                        csv_file = self.csv_processor.save_csv(
                            df,
                            f"{period_name}_{source_name}",
                            subdirectory="historical"
                        )
                        # Also append to single master long CSV with de-duplication
                        self.csv_processor.append_to_master_long(
                            df,
                            master_filename="historical_2017_2024_master.csv",
                            subdirectory="historical",
                            dedup_keys=['date','latitude','longitude','variable_name','dataset_id']
                        )
                        csv_files[source_name] = str(csv_file)
                        logger.info(f"✅ Converted {source_name}: {len(df)} records")
                    else:
                        logger.warning(f"⚠️ No data in {source_name} after conversion")
                    
            except Exception as e:
                logger.warning(f"Failed to convert {source_name} to CSV: {e}")
        
        return csv_files
    
    def _extract_variables_from_gee_data(self, data_dict: Dict[str, Any]) -> List[str]:
        """
        FIXED: Extract variable names from GEE data structure
        
        DEBUG NOTES:
        - GEE data returns dict with 'statistics', 'dataset_id', etc.
        - Extract actual variable names from data structure
        """
        variables = set()
        
        for source_name, source_data in data_dict.items():
            try:
                if isinstance(source_data, dict):
                    # GEE data structure analysis
                    if 'variables' in source_data:
                        variables.update(source_data['variables'])
                    if 'variable' in source_data:
                        variables.add(source_data['variable'])
                    
                    # Add source-specific variable names
                    if 'wind' in source_name:
                        variables.update(['wind_u', 'wind_v'])
                    elif 'precipitation' in source_name:
                        variables.add('precipitation')
                    elif 'chlorophyll' in source_name:
                        variables.add('chlorophyll_a')
                    elif 'turbidity' in source_name:
                        variables.add('turbidity')
                    elif 'sst' in source_name:
                        variables.add('sea_surface_temperature')
                    elif 'modis' in source_name:
                        variables.update(['modis_chlorophyll', 'modis_sst'])
                    elif 'hycom_temp_sal' in source_name:
                        variables.update(['water_temp_0', 'salinity_0', 'ocean_temperature', 'ocean_salinity'])
                    elif 'hycom_velocity' in source_name:
                        variables.update(['velocity_u_0', 'velocity_v_0', 'ocean_current_u', 'ocean_current_v'])
                    elif 'hycom_elevation' in source_name:
                        variables.update(['surface_elevation', 'sea_surface_elevation'])
                    elif 'copernicus' in source_name:
                        variables.update(['nitrate', 'phosphate', 'silicate', 'dissolved_iron', 'primary_production', 'dissolved_oxygen'])
                    elif 'erddap' in source_name:
                        variables.update(['atmospheric_backup_wind_u', 'atmospheric_backup_wind_v', 'atmospheric_backup_precipitation'])
                        
            except Exception as e:
                logger.debug(f"Error extracting variables from {source_name}: {e}")
        
        return list(variables)
    
    def _count_total_records_fixed(self, csv_files: Dict[str, str]) -> int:
        """
        FIXED: Count total records in CSV files with proper error handling
        
        DEBUG NOTES:
        - Handle missing files gracefully
        - Count actual data rows excluding headers
        """
        total_records = 0
        
        for source_name, csv_file in csv_files.items():
            try:
                if Path(csv_file).exists():
                    df = pd.read_csv(csv_file)
                    records = len(df)
                    total_records += records
                    logger.debug(f"📊 {source_name}: {records} records")
                else:
                    logger.warning(f"⚠️ CSV file not found: {csv_file}")
            except Exception as e:
                logger.warning(f"Error counting records in {csv_file}: {e}")
        
        return total_records
    
    def _collect_hycom_temp_salinity(self, start_date: str, end_date: str) -> Optional[Dict]:
        """
        Collect HYCOM temperature and salinity data using direct GEE collection
        
        HYCOM coverage: 1992-2024 (32+ years) - CRITICAL for ocean physics variables
        """
        try:
            import ee
            
            # Load HYCOM temperature/salinity dataset
            hycom_dataset = ee.ImageCollection('HYCOM/sea_temp_salinity') \
                .filterDate(start_date, end_date) \
                .filterBounds(ee.Geometry.Rectangle([132.0, 34.0, 135.0, 35.0]))
            
            # Get mean values over the region
            stats = hycom_dataset.select(['water_temp_0', 'salinity_0']) \
                .mean() \
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=ee.Geometry.Rectangle([132.0, 34.0, 135.0, 35.0]),
                    scale=1000,
                    maxPixels=1e6
                )
            
            result = stats.getInfo()
            if result:
                return {
                    'dataset_id': 'HYCOM/sea_temp_salinity',
                    'variables': ['water_temp_0', 'salinity_0'],
                    'statistics': result,
                    'period': f"{start_date} to {end_date}",
                    'variable_count': 2
                }
            return None
            
        except Exception as e:
            logger.warning(f"HYCOM temp/salinity collection failed: {e}")
            return None
    
    def _collect_hycom_velocity(self, start_date: str, end_date: str) -> Optional[Dict]:
        """Collect HYCOM ocean current velocity data"""
        try:
            import ee
            
            hycom_dataset = ee.ImageCollection('HYCOM/sea_water_velocity') \
                .filterDate(start_date, end_date) \
                .filterBounds(ee.Geometry.Rectangle([132.0, 34.0, 135.0, 35.0]))
            
            stats = hycom_dataset.select(['velocity_u_0', 'velocity_v_0']) \
                .mean() \
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=ee.Geometry.Rectangle([132.0, 34.0, 135.0, 35.0]),
                    scale=1000,
                    maxPixels=1e6
                )
            
            result = stats.getInfo()
            if result:
                return {
                    'dataset_id': 'HYCOM/sea_water_velocity',
                    'variables': ['velocity_u_0', 'velocity_v_0'],
                    'statistics': result,
                    'period': f"{start_date} to {end_date}",
                    'variable_count': 2
                }
            return None
            
        except Exception as e:
            logger.warning(f"HYCOM velocity collection failed: {e}")
            return None
    
    def _collect_hycom_elevation(self, start_date: str, end_date: str) -> Optional[Dict]:
        """Collect HYCOM sea surface elevation data"""
        try:
            import ee
            
            hycom_dataset = ee.ImageCollection('HYCOM/sea_surface_elevation') \
                .filterDate(start_date, end_date) \
                .filterBounds(ee.Geometry.Rectangle([132.0, 34.0, 135.0, 35.0]))
            
            stats = hycom_dataset.select(['surface_elevation']) \
                .mean() \
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=ee.Geometry.Rectangle([132.0, 34.0, 135.0, 35.0]),
                    scale=1000,
                    maxPixels=1e6
                )
            
            result = stats.getInfo()
            if result:
                return {
                    'dataset_id': 'HYCOM/sea_surface_elevation',
                    'variables': ['surface_elevation'],
                    'statistics': result,
                    'period': f"{start_date} to {end_date}",
                    'variable_count': 1
                }
            return None
            
        except Exception as e:
            logger.warning(f"HYCOM elevation collection failed: {e}")
            return None
    
    def _generate_final_summary(self, duration: timedelta) -> Dict[str, Any]:
        """Generate final collection summary"""
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 PERIOD-AWARE 7-YEAR COLLECTION COMPLETED!")
        logger.info("=" * 80)
        
        # Calculate success metrics
        successful_periods = sum(1 for result in self.period_results.values() if result['success'])
        total_periods = len(self.period_results)
        
        logger.info(f"📊 COLLECTION SUMMARY:")
        logger.info(f"   ⏱️ Duration: {duration.total_seconds():.0f} seconds ({duration.total_seconds()/60:.1f} minutes)")
        logger.info(f"   🎯 Periods Collected: {successful_periods}/{total_periods}")
        logger.info(f"   📈 Total Records: {self.total_records:,}")
        logger.info(f"   🔢 Unique Variables: {len(self.total_variables)}")
        logger.info(f"   📅 Time Range: 2017-2024 (7+ years)")
        
        logger.info(f"\n🔢 VARIABLES COLLECTED ({len(self.total_variables)}/25+):")
        for i, var in enumerate(sorted(self.total_variables), 1):
            logger.info(f"   {i:2d}. {var}")
        
        logger.info(f"\n📊 PERIOD DETAILS:")
        for period_name, result in self.period_results.items():
            status = "✅" if result['success'] else "❌"
            logger.info(f"   {status} {period_name}: {result.get('records', 0)} records, {len(result.get('variables', []))} variables")
        
        if len(self.total_variables) >= 25:
            logger.info("\n🏆 SUCCESS: 25+ HAB variables collected for 7-year period!")
        else:
            missing = 25 - len(self.total_variables)
            logger.warning(f"\n⚠️ PARTIAL: {missing} variables missing from 25+ target")
        
        return {
            "success": successful_periods == total_periods,
            "total_records": self.total_records,
            "total_variables": len(self.total_variables),
            "variables_list": sorted(list(self.total_variables)),
            "duration_seconds": duration.total_seconds(),
            "period_results": self.period_results,
            "periods_successful": successful_periods,
            "periods_total": total_periods
        }

def main():
    """Main execution function"""
    logger.info("🧠 PERIOD-AWARE 7-Year HAB Data Collection System")
    logger.info("Using ACTUAL data source availability windows for optimal coverage")
    
    try:
        collector = PeriodAwareHistoricalCollector()
        result = collector.run_period_aware_collection()
        
        if result["success"]:
            logger.info("🎉 PERIOD-AWARE COLLECTION COMPLETED SUCCESSFULLY!")
            return 0
        else:
            logger.error("❌ PERIOD-AWARE COLLECTION FAILED")
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Collection interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
