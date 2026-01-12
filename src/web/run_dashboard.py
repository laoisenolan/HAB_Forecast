"""
Simple Dashboard Runner - No Authentication Required

This script runs the HAB forecasting dashboard without authentication
for easy development and testing access.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path - Fix the path to point to the correct src directory
current_dir = Path(__file__).parent
src_dir = current_dir.parent  # Go up one level from web/ to src/
project_root = src_dir.parent  # Go up one level from src/ to project root/

# Add both src and project root to path
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root))

from web.dashboard_app import HABDashboardApp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the dashboard without authentication"""
    try:
        logger.info("Starting HAB Forecasting Dashboard (No Auth Required)")
        logger.info("=" * 60)
        
        # Initialize dashboard
        dashboard = HABDashboardApp()
        
        # Start background updates
        dashboard.start_background_updates()
        
        logger.info("Dashboard initialized successfully!")
        logger.info("Access the dashboard at: http://localhost:5000")
        logger.info("Press Ctrl+C to stop the server")
        logger.info("=" * 60)
        
        # Run the dashboard
        dashboard.run(host='0.0.0.0', port=5000, debug=True)
        
    except KeyboardInterrupt:
        logger.info("Shutting down dashboard...")
        if 'dashboard' in locals():
            dashboard.stop_background_updates()
        logger.info("Dashboard stopped")
    except Exception as e:
        logger.error(f"Dashboard failed to start: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 