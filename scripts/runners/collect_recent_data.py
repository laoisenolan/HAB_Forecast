#!/usr/bin/env python3
"""
Realistic Historical Data Collection (2022-2024)
Optimized for actual data availability and HAB event coverage

Based on analysis of:
- 201 HAB events from 2022-2023 
- Actual data source availability windows
- Realistic storage constraints (2GB available)
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.historical_collector import HistoricalDataCollector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'realistic_collection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)

class RealisticHistoricalCollector:
    """Optimized historical collector based on actual data availability"""
    
    def __init__(self):
        """Initialize with realistic parameters"""
        self.collector = HistoricalDataCollector()
        
        # Realistic collection parameters - aligned with data availability
        self.start_date = "2022-06-01"  # When Copernicus physics becomes available
        self.end_date = "2024-06-30"    # 2 years of data + buffer for training
        self.period_name = "realistic_2022_2024"
        
        # Optimize chunk size for storage constraints
        self.collector.time_chunk_days = 20  # Smaller chunks for better memory management
        
        # Expected working variables (based on config analysis)
        self.working_variables = [
            # Copernicus Marine (available from 2022-06-01) - 11 variables
            'mixed_layer_depth', 'sea_surface_height',           # Physics: 2 vars
            'total_chlorophyll_bgc', 'phytoplankton_carbon',     # Biology: 2 vars
            'nitrate', 'phosphate', 'silicate', 'dissolved_iron', # Nutrients: 4 vars
            'primary_production', 'dissolved_oxygen',             # Biology: 2 vars
            'diffuse_attenuation',                               # Optics: 1 var
            
            # Google Earth Engine (excellent coverage) - 9 variables
            'wind_u', 'wind_v', 'precipitation',                 # ERA5: 3 vars
            'ocean_temperature_hycom', 'ocean_salinity',         # HYCOM: 2 vars
            'ocean_current_u', 'ocean_current_v', 'sea_surface_elevation', # HYCOM: 3 vars
            'chlorophyll_a_jaxa',                                # JAXA: 1 var
            
            # Note: Turbidity from Sentinel-2 and ERDDAP will be attempted
            # but may have coverage limitations
        ]
        
        logger.info(f"Target collection: {self.start_date} to {self.end_date}")
        logger.info(f"Expected variables: {len(self.working_variables)}")
        
    def run_realistic_collection(self):
        """Run the realistic historical collection"""
        logger.info("Starting Realistic Historical Data Collection")
        logger.info("="*70)
        
        collection_start = datetime.now()
        
        try:
            # Display collection strategy
            self._display_collection_strategy()
            
            # Run collection with optimized settings
            logger.info("Starting optimized data collection...")
            result = self.collector.run_historical_collection(
                period_name=self.period_name,
                start_date=self.start_date,
                end_date=self.end_date,
                include_climatology=False  # Skip climatology for faster collection
            )
            
            if result.get("success", False):
                logger.info("Realistic collection completed successfully!")
                
                # Analyze results
                analysis = self._analyze_results(result)
                
                # Generate final report
                final_report = self._create_final_report(result, analysis, collection_start)
                
                logger.info("REALISTIC COLLECTION COMPLETED!")
                return final_report
                
            else:
                logger.error(f"Collection failed: {result.get('error', 'Unknown error')}")
                return {"success": False, "error": result.get('error', 'Collection failed')}
                
        except Exception as e:
            logger.error(f"Critical error: {e}")
            return {"success": False, "error": str(e)}
    
    def _display_collection_strategy(self):
        """Display the collection strategy"""
        logger.info("Realistic Collection Strategy:")
        logger.info(f"   Period: {self.start_date} to {self.end_date} (2 years)")
        logger.info(f"   Focus: HAB event coverage + Data availability alignment")
        logger.info(f"   Expected variables: {len(self.working_variables)}")
        logger.info(f"   Chunks: ~37 chunks (20-day chunks)")
        logger.info(f"   Est. Time: 1-2 hours")
        logger.info(f"   Est. Size: ~800MB-1.5GB")
        
        logger.info("Data Source Strategy:")
        logger.info("   Copernicus Marine (2022-06-01+):")
        logger.info("      • Ocean Physics: Mixed layer, Sea level") 
        logger.info("      • Biogeochemistry: N,P,Si,Fe,O2,Production")
        logger.info("      • Biology: Chlorophyll, Phytoplankton")
        logger.info("   Google Earth Engine (Full coverage):")
        logger.info("      • Atmospheric: ERA5 wind, precipitation")
        logger.info("      • Ocean: HYCOM temp, salinity, currents")
        logger.info("      • High-res: JAXA chlorophyll")
        logger.info("   ERDDAP (2022-12-01+):")
        logger.info("      • Atmospheric backup: GFS weather data")
    
    def _analyze_results(self, collection_result):
        """Analyze collection results"""
        logger.info("Analyzing collection results...")
        
        # Basic analysis
        analysis = {
            "collection_success": collection_result.get("success", False),
            "files_created": len(collection_result.get("saved_files", {})),
            "data_sources_used": list(collection_result.get("saved_files", {}).keys()),
        }
        
        return analysis
    
    def _create_final_report(self, collection_result, analysis, start_time):
        """Create final collection report"""
        end_time = datetime.now()
        duration = end_time - start_time
        
        report = {
            "collection_info": {
                "period": self.period_name,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "collection_start": start_time.isoformat(),
                "collection_end": end_time.isoformat(),
                "duration_minutes": duration.total_seconds() / 60,
                "success": True
            },
            "files": collection_result.get("saved_files", {}),
            "analysis": analysis
        }
        
        return report

def main():
    """Main function"""
    logger.info("Starting Realistic Historical Data Collection")
    
    try:
        collector = RealisticHistoricalCollector()
        result = collector.run_realistic_collection()
        
        if result.get("success", False):
            logger.info("SUCCESS! Ready for HAB model training with real events")
            return 0
        else:
            logger.error(f"FAILED: {result.get('error', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Critical error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
