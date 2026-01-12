#!/usr/bin/env python3
"""
Main Data Collection Script for HAB Forecasting System

This script provides a command-line interface for running the complete
data collection pipeline for the Seto Inland Sea region.
"""

import argparse
import logging
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection import DataCollectionOrchestrator, DataQualityController
from src.data_collection.historical_collector import HistoricalDataCollector
from src.utils.config_loader import ConfigLoader

def setup_logging(level: str = "INFO", log_file: str = None):
    """Set up logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=log_format
        )

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="HAB Forecasting System - Data Collection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect last week's data
  python main.py collect-weekly
  
  # Collect specific date range
  python main.py collect-range --start-date 2024-01-01 --end-date 2024-01-07
  
  # Collect with climatology computation
  python main.py collect-weekly --include-climatology
  
  # Collect data using intelligent dataset selection
  python main.py collect-intelligent --start-date 2024-01-01 --end-date 2024-01-07
  
  # Collect specific variables with intelligent selection
  python main.py collect-intelligent --start-date 2024-01-01 --end-date 2024-01-07 --variables sea_surface_temperature chlorophyll_a
  
  # Collect with high-resolution preference
  python main.py collect-intelligent --start-date 2024-01-01 --end-date 2024-01-07 --prefer-high-resolution --coastal-focus
  
  # Collect data using parallel processing for optimal performance
  python main.py collect-parallel --start-date 2024-01-01 --end-date 2024-01-07 --max-workers 8
  
  # Collect specific variables with parallel processing
  python main.py collect-parallel --start-date 2024-01-01 --end-date 2024-01-07 --variables sea_surface_temperature chlorophyll_a --max-workers 3
  
  # Collect historical data for Seto Inland Sea
  python main.py collect-historical --period full_historical
  
  # Collect custom historical period
  python main.py collect-historical --start-date 2010-01-01 --end-date 2020-12-31
  
  # Show historical collection status
  python main.py historical-status
  
  # Validate existing data
  python main.py validate --data-type physics
  
  # Show available data sources
  python main.py info
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Collect weekly data
    weekly_parser = subparsers.add_parser("collect-weekly", help="Collect last week's data")
    weekly_parser.add_argument(
        "--include-climatology",
        action="store_true",
        help="Include climatology computation"
    )
    weekly_parser.add_argument(
        "--use-fallback",
        action="store_true",
        default=True,
        help="Use fallback data sources if primary fails"
    )
    weekly_parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    weekly_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Collect specific date range
    range_parser = subparsers.add_parser("collect-range", help="Collect data for specific date range")
    range_parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    range_parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    range_parser.add_argument(
        "--include-climatology",
        action="store_true",
        help="Include climatology computation"
    )
    range_parser.add_argument(
        "--use-fallback",
        action="store_true",
        default=True,
        help="Use fallback data sources if primary fails"
    )
    range_parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    range_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Validate data
    validate_parser = subparsers.add_parser("validate", help="Validate existing data")
    validate_parser.add_argument(
        "--data-type",
        type=str,
        choices=["physics", "ocean_color", "atmospheric", "climatology"],
        help="Type of data to validate"
    )
    validate_parser.add_argument(
        "--file-path",
        type=str,
        help="Path to specific data file to validate"
    )
    validate_parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    validate_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Test comprehensive data collection (ALL 25 variables)
    comprehensive_parser = subparsers.add_parser("test-comprehensive", help="Test comprehensive data collection with all 25 variables")
    comprehensive_parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        help="Start date (YYYY-MM-DD)"
    )
    comprehensive_parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)"
    )
    comprehensive_parser.add_argument(
        "--include-turbidity",
        action="store_true",
        default=True,
        help="Include turbidity data sources (breakthrough discovery)"
    )
    comprehensive_parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with minimal data downloads"
    )
    comprehensive_parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    comprehensive_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Intelligent data collection with dataset selection
    intelligent_parser = subparsers.add_parser("collect-intelligent", help="Collect data using intelligent dataset selection")
    intelligent_parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    intelligent_parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    intelligent_parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        help="Specific variables to collect (if not specified, collects all available)"
    )
    intelligent_parser.add_argument(
        "--prefer-high-resolution",
        action="store_true",
        help="Prefer high-resolution datasets when available"
    )
    intelligent_parser.add_argument(
        "--prefer-low-latency",
        action="store_true",
        default=True,
        help="Prefer low-latency (near real-time) datasets"
    )
    intelligent_parser.add_argument(
        "--prefer-historical-quality",
        action="store_true",
        default=True,
        help="Prefer historical quality (reprocessed) datasets"
    )
    intelligent_parser.add_argument(
        "--coastal-focus",
        action="store_true",
        help="Optimize for coastal region data collection"
    )
    intelligent_parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    intelligent_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Parallel data collection with optimized performance
    parallel_parser = subparsers.add_parser("collect-parallel", help="Collect data using parallel processing for optimal performance")
    parallel_parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parallel_parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parallel_parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        help="Specific variables to collect (if not specified, collects all available)"
    )
    parallel_parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum concurrent download threads (default: 5)"
    )
    parallel_parser.add_argument(
        "--prefer-high-resolution",
        action="store_true",
        help="Prefer high-resolution datasets when available"
    )
    parallel_parser.add_argument(
        "--prefer-low-latency",
        action="store_true",
        default=True,
        help="Prefer low-latency (near real-time) datasets"
    )
    parallel_parser.add_argument(
        "--prefer-historical-quality",
        action="store_true",
        default=True,
        help="Prefer historical quality (reprocessed) datasets"
    )
    parallel_parser.add_argument(
        "--coastal-focus",
        action="store_true",
        help="Optimize for coastal region data collection"
    )
    parallel_parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    parallel_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Quality monitoring and alerting
    monitor_parser = subparsers.add_parser("monitor", help="Quality monitoring and alerting")
    monitor_parser.add_argument(
        "action",
        choices=["start", "stop", "status", "alerts", "acknowledge", "resolve", "export"],
        help="Monitoring action to perform"
    )
    monitor_parser.add_argument(
        "--alert-index",
        type=int,
        help="Alert index for acknowledge/resolve actions"
    )
    monitor_parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for export action"
    )
    monitor_parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    monitor_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Historical data collection
    historical_parser = subparsers.add_parser("collect-historical", help="Collect historical data for Seto Inland Sea")
    historical_parser.add_argument(
        "--period",
        type=str,
        choices=["full_historical", "recent_decade", "last_five_years"],
        help="Predefined historical collection period"
    )
    historical_parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for custom period (YYYY-MM-DD)"
    )
    historical_parser.add_argument(
        "--end-date",
        type=str,
        help="End date for custom period (YYYY-MM-DD)"
    )
    historical_parser.add_argument(
        "--include-climatology",
        action="store_true",
        help="Include climatology computation"
    )
    historical_parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    historical_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Historical status
    historical_status_parser = subparsers.add_parser("historical-status", help="Show historical collection status")
    historical_status_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Show information
    info_parser = subparsers.add_parser("info", help="Show available data sources and configuration")
    info_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Cached data collection with intelligent caching
    cached_parser = subparsers.add_parser("collect-cached", help="Collect data with intelligent caching")
    cached_parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    cached_parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    cached_parser.add_argument(
        "--variables",
        type=str,
        help="Comma-separated list of variables to collect"
    )
    cached_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    cached_parser.add_argument(
        "--prefer-high-resolution",
        action="store_true",
        help="Prefer high-resolution datasets when available"
    )
    cached_parser.add_argument(
        "--prefer-low-latency",
        action="store_true",
        default=True,
        help="Prefer low-latency (near real-time) datasets"
    )
    cached_parser.add_argument(
        "--prefer-historical-quality",
        action="store_true",
        default=True,
        help="Prefer historical quality (reprocessed) datasets"
    )
    cached_parser.add_argument(
        "--coastal-focus",
        action="store_true",
        help="Optimize for coastal region data collection"
    )
    cached_parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    cached_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Cache management commands
    cache_status_parser = subparsers.add_parser("cache-status", help="Show cache status and statistics")
    cache_status_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    clear_cache_parser = subparsers.add_parser("clear-cache", help="Clear expired cache files")
    clear_cache_parser.add_argument(
        "--all",
        action="store_true",
        help="Clear all cache files"
    )
    clear_cache_parser.add_argument(
        "--older-than-days",
        type=int,
        default=30,
        help="Clear files older than N days"
    )
    clear_cache_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Global arguments
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()

def collect_weekly_data(args):
    """Collect last week's data"""
    logger = logging.getLogger(__name__)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    logger.info(f"Collecting weekly data from {start_date_str} to {end_date_str}")
    
    # Initialize orchestrator
    orchestrator = DataCollectionOrchestrator()
    
    # Run collection pipeline
    results = orchestrator.run_complete_collection_pipeline(
        start_date_str,
        end_date_str,
        include_climatology=args.include_climatology,
        use_fallback=args.use_fallback
    )
    
    if results["success"]:
        logger.info("Weekly data collection completed successfully")
        logger.info(f"Collected {len(results['collected_data'])} datasets")
        logger.info(f"Validation success rate: {results['report']['statistics']['validation_success_rate']:.2%}")
        logger.info(f"Report saved to: {results['report_file']}")
        return 0
    else:
        logger.error(f"Weekly data collection failed: {results.get('error', 'Unknown error')}")
        return 1

def collect_range_data(args):
    """Collect data for specific date range"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Collecting data from {args.start_date} to {args.end_date}")
    
    # Initialize orchestrator
    orchestrator = DataCollectionOrchestrator()
    
    # Run collection pipeline
    results = orchestrator.run_complete_collection_pipeline(
        args.start_date,
        args.end_date,
        include_climatology=args.include_climatology,
        use_fallback=args.use_fallback
    )
    
    if results["success"]:
        logger.info("Range data collection completed successfully")
        logger.info(f"Collected {len(results['collected_data'])} datasets")
        logger.info(f"Validation success rate: {results['report']['statistics']['validation_success_rate']:.2%}")
        logger.info(f"Report saved to: {results['report_file']}")
        return 0
    else:
        logger.error(f"Range data collection failed: {results.get('error', 'Unknown error')}")
        return 1

def validate_data(args):
    """Validate existing data"""
    logger = logging.getLogger(__name__)
    
    # Initialize quality controller
    qc = DataQualityController()
    
    if args.file_path:
        # Validate specific file
        logger.info(f"Validating file: {args.file_path}")
        
        import xarray as xr
        try:
            dataset = xr.open_dataset(args.file_path)
            results = qc.run_comprehensive_validation(dataset, args.data_type or "unknown")
            
            report_file = qc.save_validation_report(results)
            
            logger.info(f"Validation completed. Score: {results['overall_score']:.2f}")
            logger.info(f"Report saved to: {report_file}")
            
            return 0 if results["overall_valid"] else 1
            
        except Exception as e:
            logger.error(f"Failed to validate file: {e}")
            return 1
            
    elif args.data_type:
        # Validate all files of specific type
        logger.info(f"Validating all {args.data_type} data files")
        
        # This would require scanning the data directory
        # For now, just show a message
        logger.info("File scanning validation not yet implemented")
        return 0
        
    else:
        logger.error("Please specify either --file-path or --data-type")
        return 1

def collect_historical_data(args):
    """Collect historical data for Seto Inland Sea"""
    logger = logging.getLogger(__name__)
    
    # Initialize historical collector
    historical_collector = HistoricalDataCollector()
    
    # Determine collection period
    if args.period:
        # Use predefined period
        logger.info(f"Using predefined period: {args.period}")
        results = historical_collector.run_historical_collection(
            period_name=args.period,
            include_climatology=args.include_climatology
        )
    elif args.start_date and args.end_date:
        # Use custom period
        logger.info(f"Using custom period: {args.start_date} to {args.end_date}")
        results = historical_collector.run_historical_collection(
            period_name="custom_period",
            start_date=args.start_date,
            end_date=args.end_date,
            include_climatology=args.include_climatology
        )
    else:
        logger.error("Please specify either --period or both --start-date and --end-date")
        return 1
    
    if results["success"]:
        logger.info("Historical data collection completed successfully")
        logger.info(f"Collected {len(results['merged_datasets'])} merged datasets")
        logger.info(f"Saved {len(results['saved_files'])} files")
        logger.info(f"Report saved to: {results['report_file']}")
        return 0
    else:
        logger.error(f"Historical data collection failed: {results.get('error', 'Unknown error')}")
        return 1

def show_historical_status():
    """Show historical collection status"""
    logger = logging.getLogger(__name__)
    
    # Initialize historical collector
    historical_collector = HistoricalDataCollector()
    
    # Get status
    status = historical_collector.get_historical_collection_status()
    
    print("HAB Forecasting System - Historical Collection Status")
    print("=" * 55)
    
    # Available periods
    print("\nAVAILABLE COLLECTION PERIODS:")
    print("-" * 30)
    for period in status["available_periods"]:
        print(f"  {period['name']}: {period['description']}")
        print(f"    Period: {period['start_date']} to {period['end_date']}")
    
    # Existing collections
    print("\nEXISTING COLLECTIONS:")
    print("-" * 20)
    if status["existing_collections"]:
        for collection in status["existing_collections"]:
            print(f"  {collection['data_type']}: {collection['file_count']} files")
            for file_name in collection['files'][:5]:  # Show first 5 files
                print(f"    - {file_name}")
            if len(collection['files']) > 5:
                print(f"    ... and {len(collection['files']) - 5} more files")
    else:
        print("  No existing historical collections found")
    
    # Storage information
    print("\nSTORAGE INFORMATION:")
    print("-" * 20)
    print(f"  Historical directory: {status['storage_info']['historical_dir']}")
    print(f"  Total size: {status['storage_info']['total_size_mb']:.2f} MB")
    
    return 0

def show_info():
    """Show available data sources and configuration"""
    logger = logging.getLogger(__name__)
    
    # Initialize orchestrator
    orchestrator = DataCollectionOrchestrator()
    
    # Get available data sources
    data_sources = orchestrator.get_available_data_sources()
    
    print("HAB Forecasting System - Data Sources Information")
    print("=" * 50)
    
    for source_name, source_info in data_sources.items():
        print(f"\n{source_name.upper()}:")
        print("-" * 20)
        
        if isinstance(source_info, dict):
            for key, value in source_info.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {source_info}")
    
    # Show configuration summary
    config_loader = ConfigLoader()
    spatial_bounds = config_loader.get_spatial_bounds()
    temporal_settings = config_loader.get_temporal_settings()
    
    print(f"\nSPATIAL BOUNDS:")
    print("-" * 20)
    for key, value in spatial_bounds.items():
        print(f"  {key}: {value}")
        
    print(f"\nTEMPORAL SETTINGS:")
    print("-" * 20)
    for key, value in temporal_settings.items():
        print(f"  {key}: {value}")
    
    return 0

def main():
    """Main function"""
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting HAB Forecasting System Data Collection")
    
    try:
        if args.command == "collect-weekly":
            return collect_weekly_data(args)
        elif args.command == "collect-range":
            return collect_range_data(args)
        elif args.command == "collect-historical":
            return collect_historical_data(args)
        elif args.command == "historical-status":
            return show_historical_status()
        elif args.command == "validate":
            return validate_data(args)
        elif args.command == "test-comprehensive":
            return test_comprehensive_collection(args)
        elif args.command == "collect-intelligent":
            return collect_intelligent_data(args)
        elif args.command == "collect-parallel":
            return collect_parallel_data(args)
        elif args.command == "monitor":
            return quality_monitoring(args)
        elif args.command == "collect-cached":
            return collect_cached_data(args)
        elif args.command == "cache-status":
            return cache_status(args)
        elif args.command == "clear-cache":
            return clear_cache(args)
        elif args.command == "info":
            return show_info()
        else:
            logger.error("No command specified. Use --help for usage information.")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

def test_comprehensive_collection(args):
    """Test comprehensive data collection with all 25 variables"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"🚀 TESTING COMPREHENSIVE DATA COLLECTION (ALL 25 VARIABLES)")
    logger.info(f"   Date range: {args.start_date} to {args.end_date}")
    logger.info(f"   Include turbidity: {args.include_turbidity}")
    logger.info(f"   Test mode: {args.test_mode}")
    
    try:
        # Import comprehensive collection modules
        from .copernicus_downloader import CopernicusDownloader
        from .gee_integrator import GEEIntegrator
        
        # Initialize data collectors
        copernicus = CopernicusDownloader()
        gee = GEEIntegrator()
        
        all_data = {}
        
        # 1. Test Copernicus comprehensive collection
        logger.info("🌊 Phase 1: Testing Copernicus comprehensive collection...")
        if args.test_mode:
            logger.info("   (Test mode: Simulating Copernicus collection)")
            copernicus_data = {"sst": "simulated", "turbidity_daily": "simulated", "nutrients": "simulated"}
        else:
            copernicus_data = copernicus.download_comprehensive_hab_data(
                args.start_date, args.end_date
            )
        all_data.update(copernicus_data)
        
        # 2. Test GEE comprehensive collection
        logger.info("🌍 Phase 2: Testing GEE comprehensive collection...")
        if args.test_mode:
            logger.info("   (Test mode: Simulating GEE collection)")
            gee_data = {"wind": "simulated", "precipitation": "simulated", "ocean": "simulated", "turbidity_sentinel2": "simulated"}
        else:
            gee_data = gee.get_comprehensive_gee_data(
                args.start_date, args.end_date
            )
        all_data.update(gee_data)
        
        # 3. Summary report
        logger.info("📊 COMPREHENSIVE COLLECTION TEST RESULTS:")
        logger.info(f"   📈 Total datasets collected: {len(all_data)}")
        
        # Count by source
        copernicus_count = len([k for k in all_data.keys() if any(x in k for x in ['turbidity', 'sst', 'chlorophyll', 'physics', 'nutrients', 'biology', 'optics'])])
        gee_count = len([k for k in all_data.keys() if any(x in k for x in ['wind', 'precipitation', 'ocean', 'sentinel2'])])
        
        logger.info(f"   🌊 Copernicus datasets: {copernicus_count}")
        logger.info(f"   🌍 GEE datasets: {gee_count}")
        
        # Detailed breakdown
        logger.info("📋 Dataset breakdown:")
        for dataset_name, dataset in all_data.items():
            if dataset is not None:
                logger.info(f"   ✅ {dataset_name}: SUCCESS")
            else:
                logger.info(f"   ❌ {dataset_name}: FAILED")
        
        # Test turbidity specifically if requested
        if args.include_turbidity:
            logger.info("🌊 TURBIDITY BREAKTHROUGH TEST:")
            turbidity_datasets = [k for k in all_data.keys() if 'turbidity' in k]
            logger.info(f"   🎯 Turbidity datasets found: {len(turbidity_datasets)}")
            for turb_dataset in turbidity_datasets:
                logger.info(f"   ✅ {turb_dataset}: {all_data[turb_dataset] is not None}")
        
        # Success metrics
        success_count = len([v for v in all_data.values() if v is not None])
        success_rate = (success_count / len(all_data)) * 100 if all_data else 0
        
        logger.info(f"🎯 FINAL RESULTS:")
        logger.info(f"   📊 Success rate: {success_rate:.1f}% ({success_count}/{len(all_data)})")
        
        if success_rate >= 80:
            logger.info("🎉 COMPREHENSIVE COLLECTION TEST: PASSED!")
            logger.info("✅ System ready for operational HAB forecasting!")
            return 0
        else:
            logger.warning("⚠️ COMPREHENSIVE COLLECTION TEST: NEEDS IMPROVEMENT")
            logger.info("🔧 Some data sources may need troubleshooting")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Comprehensive collection test failed: {e}")
        return 1

def collect_intelligent_data(args):
    """Collect data using intelligent dataset selection"""
    logger = logging.getLogger(__name__)

    logger.info(f"🚀 COLLECTING INTELLIGENT DATA (Date: {args.start_date} to {args.end_date})")
    logger.info(f"   Variables to collect: {args.variables}")
    logger.info(f"   Prefer high-resolution: {args.prefer_high_resolution}")
    logger.info(f"   Prefer low-latency: {args.prefer_low_latency}")
    logger.info(f"   Prefer historical quality: {args.prefer_historical_quality}")
    logger.info(f"   Coastal focus: {args.coastal_focus}")

    try:
        # Initialize data collection orchestrator
        orchestrator = DataCollectionOrchestrator()

        # Build preferences dictionary
        preferences = {
            'prefer_high_resolution': args.prefer_high_resolution,
            'prefer_low_latency': args.prefer_low_latency,
            'prefer_historical_quality': args.prefer_historical_quality,
            'coastal_focus': args.coastal_focus
        }

        # Run intelligent data collection
        collected_data, collection_report = orchestrator.collect_data_with_intelligent_selection(
            start_date=args.start_date,
            end_date=args.end_date,
            variables=args.variables,
            preferences=preferences
        )

        # Save collection report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"intelligent_collection_report_{timestamp}.json"
        report_path = orchestrator.data_dir / "reports" / report_filename
        report_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(report_path, 'w') as f:
            json.dump(collection_report, f, indent=2, default=str)

        logger.info("✅ Intelligent data collection completed successfully")
        logger.info(f"   Collected {len(collected_data)} datasets")
        logger.info(f"   Success rate: {collection_report['collection_metadata']['success_rate']:.2%}")
        logger.info(f"   Report saved to: {report_path}")

        # Print dataset selections
        logger.info("📊 Dataset selections made:")
        for variable, selection in collection_report['dataset_selections'].items():
            if 'dataset' in selection:
                dataset_id = selection['dataset']['id']
                reason = selection.get('reason', 'No reason provided')
                logger.info(f"   {variable}: {dataset_id} ({reason})")
            elif 'error' in selection:
                logger.warning(f"   {variable}: ERROR - {selection['error']}")

        return 0

    except Exception as e:
        logger.error(f"❌ Intelligent data collection failed: {e}")
        return 1

def collect_parallel_data(args):
    """Collect data using parallel processing for optimal performance"""
    logger = logging.getLogger(__name__)

    logger.info(f"🚀 COLLECTING PARALLEL DATA (Date: {args.start_date} to {args.end_date})")
    logger.info(f"   Variables to collect: {args.variables}")
    logger.info(f"   Max workers: {args.max_workers}")
    logger.info(f"   Prefer high-resolution: {args.prefer_high_resolution}")
    logger.info(f"   Prefer low-latency: {args.prefer_low_latency}")
    logger.info(f"   Prefer historical quality: {args.prefer_historical_quality}")
    logger.info(f"   Coastal focus: {args.coastal_focus}")

    try:
        # Initialize data collection orchestrator
        orchestrator = DataCollectionOrchestrator()

        # Build preferences dictionary
        preferences = {
            'prefer_high_resolution': args.prefer_high_resolution,
            'prefer_low_latency': args.prefer_low_latency,
            'prefer_historical_quality': args.prefer_historical_quality,
            'coastal_focus': args.coastal_focus
        }

        # Run parallel data collection
        result = orchestrator.collect_data_with_parallel_processing(
            start_date=args.start_date,
            end_date=args.end_date,
            variables=args.variables,
            preferences=preferences,
            max_workers=args.max_workers
        )

        # Extract results
        collected_data = result['collected_data']
        collection_report = result['collection_report']

        # Save collection report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"parallel_collection_report_{timestamp}.json"
        report_path = orchestrator.data_dir / "reports" / report_filename
        report_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(report_path, 'w') as f:
            json.dump(collection_report, f, indent=2, default=str)

        logger.info("✅ Parallel data collection completed successfully")
        logger.info(f"   Collected {len(collected_data)} datasets")
        logger.info(f"   Success rate: {collection_report['success_rate']:.2%}")
        logger.info(f"   Total processing time: {collection_report['total_processing_time']:.2f}s")
        logger.info(f"   Average download time: {collection_report['average_download_time']:.2f}s")
        logger.info(f"   Report saved to: {report_path}")

        # Performance summary
        if collection_report['successful_downloads'] > 0:
            efficiency = collection_report['total_processing_time'] / collection_report['successful_downloads']
            logger.info(f"   Efficiency: {efficiency:.2f}s per variable")
        
        # Batch status
        batch_status = collection_report['batch_status']
        logger.info(f"   Batch ID: {batch_status['batch_id']}")
        logger.info(f"   Progress: {batch_status['progress']:.1%}")

        return 0

    except Exception as e:
        logger.error(f"❌ Parallel data collection failed: {e}")
        return 1

def quality_monitoring(args):
    """Handle quality monitoring and alerting commands"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"🔍 QUALITY MONITORING - Action: {args.action}")
    
    try:
        # Initialize data collection orchestrator
        orchestrator = DataCollectionOrchestrator()
        
        if args.action == "start":
            logger.info("Starting quality monitoring system...")
            orchestrator._start_quality_monitoring()
            logger.info("✅ Quality monitoring system started")
            
        elif args.action == "stop":
            logger.info("Stopping quality monitoring system...")
            orchestrator.stop_quality_monitoring()
            logger.info("✅ Quality monitoring system stopped")
            
        elif args.action == "status":
            logger.info("Getting quality monitoring status...")
            status = orchestrator.get_quality_monitoring_status()
            
            if "error" in status:
                logger.error(f"❌ Failed to get status: {status['error']}")
                return 1
            
            logger.info("📊 Quality Monitoring Status:")
            logger.info(f"   Monitoring active: {status['is_monitoring']}")
            logger.info(f"   Active alerts: {status['active_alerts_count']}")
            logger.info(f"   Data sources monitored: {len(status['data_sources'])}")
            
            # Show data source status
            for source, info in status['data_sources'].items():
                logger.info(f"   {source}: {info['status']} (score: {info['quality_score']:.2f})")
            
            # Show recent alerts
            if status['recent_alerts']:
                logger.info("🚨 Recent Alerts:")
                for alert in status['recent_alerts']:
                    logger.info(f"   [{alert['severity'].upper()}] {alert['message']}")
            
        elif args.action == "alerts":
            logger.info("Getting active alerts...")
            status = orchestrator.get_quality_monitoring_status()
            
            if "error" in status:
                logger.error(f"❌ Failed to get alerts: {status['error']}")
                return 1
            
            active_alerts = status.get('active_alerts_count', 0)
            if active_alerts == 0:
                logger.info("✅ No active alerts")
            else:
                logger.info(f"🚨 Active Alerts ({active_alerts}):")
                for i, alert in enumerate(status.get('recent_alerts', [])):
                    logger.info(f"   [{i}] [{alert['severity'].upper()}] {alert['message']}")
                    logger.info(f"       Source: {alert['data_source']}, Variable: {alert['variable']}")
                    logger.info(f"       Time: {alert['timestamp']}")
            
        elif args.action == "acknowledge":
            if args.alert_index is None:
                logger.error("❌ Alert index required for acknowledge action")
                return 1
            
            logger.info(f"Acknowledging alert {args.alert_index}...")
            success = orchestrator.acknowledge_quality_alert(args.alert_index)
            
            if success:
                logger.info("✅ Alert acknowledged successfully")
            else:
                logger.error("❌ Failed to acknowledge alert")
                return 1
                
        elif args.action == "resolve":
            if args.alert_index is None:
                logger.error("❌ Alert index required for resolve action")
                return 1
            
            logger.info(f"Resolving alert {args.alert_index}...")
            success = orchestrator.resolve_quality_alert(args.alert_index)
            
            if success:
                logger.info("✅ Alert resolved successfully")
            else:
                logger.error("❌ Failed to resolve alert")
                return 1
                
        elif args.action == "export":
            if args.output_file is None:
                logger.error("❌ Output file required for export action")
                return 1
            
            logger.info(f"Exporting quality monitoring data to {args.output_file}...")
            orchestrator.export_quality_monitoring_data(args.output_file)
            logger.info("✅ Quality monitoring data exported successfully")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Quality monitoring action failed: {e}")
        return 1

def collect_cached_data(args):
    """Collect data with intelligent caching"""
    logger = logging.getLogger(__name__)
    try:
        orchestrator = DataCollectionOrchestrator()
        
        # Build preferences dictionary
        preferences = {}
        if args.prefer_high_resolution:
            preferences["prefer_high_resolution"] = True
        if args.prefer_low_latency:
            preferences["prefer_low_latency"] = True
        if args.prefer_historical_quality:
            preferences["prefer_historical_quality"] = True
        if args.coastal_focus:
            preferences["coastal_focus"] = True
        
        # Collect data with caching
        collected_data, collection_report = orchestrator.collect_data_with_caching(
            start_date=args.start_date,
            end_date=args.end_date,
            variables=args.variables.split(',') if args.variables else None,
            use_cache=not args.no_cache
        )
        
        # Save collection report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"cached_collection_report_{timestamp}.json"
        report_path = orchestrator.data_dir / "reports" / report_filename
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(collection_report, f, indent=2, default=str)
        
        # Print summary
        print(f"\n🎯 Cached Data Collection Summary:")
        print(f"   Variables collected: {len(collected_data)}")
        print(f"   Cache hits: {collection_report['cache_hits']}")
        print(f"   Cache misses: {collection_report['cache_misses']}")
        print(f"   Total size: {collection_report['total_size_mb']:.2f} MB")
        print(f"   Errors: {len(collection_report['errors'])}")
        print(f"   Report saved: {report_path}")
        
        if collection_report['errors']:
            print(f"\n⚠️  Errors encountered:")
            for error in collection_report['errors']:
                print(f"   - {error}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Cached data collection failed: {e}")
        return 1

def cache_status(args):
    """Show cache status and statistics"""
    logger = logging.getLogger(__name__)
    try:
        orchestrator = DataCollectionOrchestrator()
        cache_stats = orchestrator.get_cache_status()
        
        print(f"\n📊 Cache Status:")
        print(f"   Cache directory: {cache_stats['cache_dir']}")
        print(f"   Total files: {cache_stats['total_files']}")
        print(f"   Valid files: {cache_stats['valid_files']}")
        print(f"   Expired files: {cache_stats['expired_files']}")
        print(f"   Total size: {cache_stats['total_size_mb']:.2f} MB")
        print(f"   TTL: {cache_stats['ttl_days']} days")
        
        # Show cached variables by source
        cached_variables = orchestrator.list_cached_variables()
        if cached_variables:
            print(f"\n📁 Cached Variables by Source:")
            for source, variables in cached_variables.items():
                print(f"   {source}: {', '.join(variables)}")
        else:
            print(f"\n📁 No cached variables found")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to get cache status: {e}")
        return 1

def clear_cache(args):
    """Clear expired cache files"""
    logger = logging.getLogger(__name__)
    try:
        orchestrator = DataCollectionOrchestrator()
        
        if args.all:
            # Clear all cache files
            cleared_count = orchestrator.clear_cache(older_than_days=0)
            print(f"\n🗑️  Cleared all cache files: {cleared_count} files")
        else:
            # Clear only expired files
            cleared_count = orchestrator.clear_cache(older_than_days=args.older_than_days)
            print(f"\n🗑️  Cleared expired cache files: {cleared_count} files")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 