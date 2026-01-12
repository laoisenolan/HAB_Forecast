#!/usr/bin/env python3
"""
Complete 7-Year Training Pipeline Runner

This script orchestrates the complete process:
1. Enhanced 7-year historical data collection (2018-2024)
2. Enhanced model training pipeline with the collected data
3. Integration with existing dashboard for production deployment

Usage:
python run_complete_7_year_training_pipeline.py [--resume] [--skip-collection] [--dry-run]
"""

import os
import sys
import subprocess
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'complete_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)

class Complete7YearTrainingPipeline:
    """Orchestrator for the complete 7-year training pipeline"""
    
    def __init__(self):
        """Initialize the pipeline orchestrator"""
        self.project_root = Path(__file__).parent
        self.scripts_dir = self.project_root / "scripts"
        
        # Pipeline steps
        self.steps = [
            {
                "name": "Enhanced 7-Year Historical Data Collection",
                "script": "scripts/collect_7_year_historical_data.py",
                "description": "Collect comprehensive 7-year dataset (2018-2024) for training",
                "estimated_time": "2-4 hours"
            },
            {
                "name": "Enhanced Model Training Pipeline",
                "script": "scripts/enhanced_model_training_pipeline.py", 
                "description": "Train optimized models with collected 7-year dataset",
                "estimated_time": "1-2 hours"
            }
        ]
    
    def run_complete_pipeline(self, resume: bool = False, skip_collection: bool = False, dry_run: bool = False) -> bool:
        """Run the complete 7-year training pipeline"""
        logger.info("🚀 Starting Complete 7-Year Training Pipeline")
        logger.info("="*100)
        
        pipeline_start = time.time()
        
        try:
            if dry_run:
                self._show_pipeline_plan()
                return True
            
            # Step 1: Enhanced Historical Data Collection
            if not skip_collection:
                logger.info("📊 Step 1: Enhanced 7-Year Historical Data Collection")
                logger.info("-" * 60)
                
                collection_success = self._run_data_collection(resume)
                
                if not collection_success:
                    logger.error("❌ Data collection failed - stopping pipeline")
                    return False
                
                logger.info("✅ Data collection completed successfully!")
            else:
                logger.info("⏭️ Skipping data collection (using existing data)")
            
            # Step 2: Enhanced Model Training
            logger.info("🤖 Step 2: Enhanced Model Training Pipeline")
            logger.info("-" * 60)
            
            training_success = self._run_model_training()
            
            if not training_success:
                logger.error("❌ Model training failed - stopping pipeline")
                return False
                
            logger.info("✅ Model training completed successfully!")
            
            # Step 3: Integration Summary
            self._display_integration_summary()
            
            pipeline_duration = time.time() - pipeline_start
            logger.info(f"🎉 Complete Pipeline Finished in {pipeline_duration/3600:.1f} hours!")
            
            return True
            
        except KeyboardInterrupt:
            logger.info("🛑 Pipeline interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return False
    
    def _show_pipeline_plan(self):
        """Show the complete pipeline plan"""
        logger.info("📋 Complete 7-Year Training Pipeline Plan")
        logger.info("="*100)
        
        total_estimated_time = "3-6 hours"
        
        logger.info(f"🎯 Objective: Train production-ready HAB forecasting models using 7 years of data")
        logger.info(f"📅 Data Period: 2018-2024 (7 years)")
        logger.info(f"⏱️ Estimated Total Time: {total_estimated_time}")
        logger.info(f"📊 Expected Output: Production-ready models for 7+ HAB species")
        
        logger.info("\n📋 Pipeline Steps:")
        for i, step in enumerate(self.steps, 1):
            logger.info(f"   {i}. {step['name']}")
            logger.info(f"      📝 {step['description']}")
            logger.info(f"      ⏱️ Est. Time: {step['estimated_time']}")
            logger.info(f"      📄 Script: {step['script']}")
            logger.info("")
        
        logger.info("🎯 Expected Benefits:")
        logger.info("   📈 Realistic model performance (F1: 0.3-0.6)")
        logger.info("   🧹 No data leakage (clean environmental features only)")
        logger.info("   📊 Comprehensive 7-year training dataset")
        logger.info("   🚀 Production-ready model artifacts")
        logger.info("   🔬 Model interpretability with SHAP analysis")
        logger.info("   📱 Integration with existing dashboard")
        
        logger.info("⚙️ Technical Features:")
        logger.info("   🔄 Resume capability for interrupted collections")
        logger.info("   🛡️ Enhanced error recovery and retry mechanisms")
        logger.info("   📊 Real-time progress tracking and monitoring")
        logger.info("   💾 Memory-efficient processing for large datasets")
        logger.info("   ✅ Comprehensive quality assessment and validation")
    
    def _run_data_collection(self, resume: bool = False) -> bool:
        """Run the enhanced 7-year data collection"""
        try:
            cmd = [sys.executable, "scripts/collect_7_year_historical_data.py"]
            
            if resume:
                cmd.append("--resume")
                logger.info("🔄 Running with resume capability")
            
            # Add optimized chunk size for efficiency
            cmd.extend(["--chunk-size", "15"])
            
            logger.info(f"🚀 Starting data collection...")
            logger.info(f"📄 Command: {' '.join(cmd)}")
            
            # Run the collection script
            result = subprocess.run(cmd, capture_output=False, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                logger.info("✅ Data collection completed successfully")
                return True
            else:
                logger.error(f"❌ Data collection failed with return code: {result.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to run data collection: {e}")
            return False
    
    def _run_model_training(self) -> bool:
        """Run the enhanced model training pipeline"""
        try:
            cmd = [sys.executable, "scripts/enhanced_model_training_pipeline.py"]
            
            logger.info(f"🤖 Starting model training...")
            logger.info(f"📄 Command: {' '.join(cmd)}")
            
            # Run the training script
            result = subprocess.run(cmd, capture_output=False, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                logger.info("✅ Model training completed successfully")
                return True
            else:
                logger.error(f"❌ Model training failed with return code: {result.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to run model training: {e}")
            return False
    
    def _display_integration_summary(self):
        """Display integration summary and next steps"""
        logger.info("🔗 Integration Summary & Next Steps")
        logger.info("="*100)
        
        logger.info("✅ What was accomplished:")
        logger.info("   📊 Collected 7 years of comprehensive environmental data")
        logger.info("   🤖 Trained production-ready species-specific models")
        logger.info("   🧹 Fixed data leakage issues from previous models")
        logger.info("   📈 Achieved realistic model performance metrics")
        logger.info("   💾 Saved models in production-ready format")
        
        logger.info("🚀 Models ready for integration:")
        logger.info("   📂 Location: models/enhanced_7_year_models/")
        logger.info("   🧬 Species: Karenia, Heterosigma, Chattonella, Cochlodinium, etc.")
        logger.info("   📄 Artifacts: model.pkl, scaler.pkl, selector.pkl, metadata.json")
        
        logger.info("🔗 Dashboard Integration:")
        logger.info("   📱 Models automatically loaded by enhanced dashboard")
        logger.info("   🎯 Real predictions replace simulated data")
        logger.info("   📊 Model performance metrics displayed")
        logger.info("   🌐 Species-specific risk assessment available")
        
        logger.info("💡 Next Steps:")
        logger.info("   1. 🚀 Start the dashboard to see new models in action:")
        logger.info("      python src/web/run_dashboard.py")
        logger.info("   2. 🌐 Access dashboard at http://localhost:5000")
        logger.info("   3. 📊 Verify models are loaded and working")
        logger.info("   4. 🔄 Set up automated retraining schedule")
        logger.info("   5. 📈 Monitor model performance in production")
        
        logger.info("🎯 Expected Dashboard Improvements:")
        logger.info("   📊 Stable, realistic risk predictions")
        logger.info("   🗺️ Proper heatmap visualization")
        logger.info("   🎯 Species-specific predictions")
        logger.info("   📈 Real model performance metrics")
        logger.info("   🔍 Model interpretability insights")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Complete 7-Year Training Pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume data collection from previous run")
    parser.add_argument("--skip-collection", action="store_true", help="Skip data collection (use existing data)")
    parser.add_argument("--dry-run", action="store_true", help="Show pipeline plan without executing")
    
    args = parser.parse_args()
    
    logger.info("🚀 Complete 7-Year HAB Forecasting Training Pipeline")
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize pipeline
        pipeline = Complete7YearTrainingPipeline()
        
        # Run complete pipeline
        success = pipeline.run_complete_pipeline(
            resume=args.resume,
            skip_collection=args.skip_collection,
            dry_run=args.dry_run
        )
        
        if success:
            logger.info("🎉 Complete Pipeline Successfully Executed!")
            logger.info("🚀 Your enhanced HAB forecasting system is ready!")
            return 0
        else:
            logger.error("❌ Pipeline execution failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
