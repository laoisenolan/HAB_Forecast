#!/usr/bin/env python3
"""
Standalone Historical Data Collector Runner

This script runs the historical data collector without import issues.
Run this from the project root directory.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the historical data collector"""
    parser = argparse.ArgumentParser(description='Historical Data Collector for HAB Forecasting')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--region', default='seto_inland_sea', help='Region to collect data for')
    parser.add_argument('--include-climatology', action='store_true', help='Include climatology data')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting Historical Data Collection")
        logger.info(f"Period: {args.start_date} to {args.end_date}")
        logger.info(f"Region: {args.region}")
        logger.info("=" * 60)
        
        # Import after path setup
        from data_collection.historical_collector import HistoricalDataCollector
        
        # Initialize collector
        collector = HistoricalDataCollector()
        
        # Run collection
        result = collector.run_historical_collection(
            period_name="custom_period",
            start_date=args.start_date,
            end_date=args.end_date,
            include_climatology=args.include_climatology
        )
        
        logger.info("Historical data collection completed successfully!")
        logger.info(f"Results: {result}")
        
    except Exception as e:
        logger.error(f"Historical data collection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
