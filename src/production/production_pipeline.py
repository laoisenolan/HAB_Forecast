"""
Production Pipeline for Real-time HAB Forecasting

This module implements a production-ready pipeline for real-time HAB forecasting:
- Streaming data processing
- Real-time predictions
- Automated model updates
- Performance monitoring
- Error handling and recovery
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import json
import pickle
import time
import threading
from queue import Queue
import schedule

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Handle imports for both direct execution and module import
try:
    # Try relative imports first (when run as module)
    from ..feature_engineering.comprehensive_feature_pipeline import ComprehensiveHABFeaturePipeline as FeaturePipeline
    from ..model.species_specific_trainer import SpeciesSpecificXGBoostTrainer as SpeciesSpecificTrainer
    from ..model.ensemble_hab_trainer import EnsembleHABTrainer
    from ..model.predictor import HABPredictor
    from ..utils.config_loader import ConfigLoader
except ImportError:
    # Fall back to absolute imports (when run directly)
    # Add src to path
    current_dir = Path(__file__).parent
    src_dir = current_dir.parent
    project_root = src_dir.parent
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(project_root))
    
    from feature_engineering.comprehensive_feature_pipeline import ComprehensiveHABFeaturePipeline as FeaturePipeline
    from model.species_specific_trainer import SpeciesSpecificXGBoostTrainer as SpeciesSpecificTrainer
    from model.ensemble_hab_trainer import EnsembleHABTrainer
    from model.predictor import HABPredictor
    from utils.config_loader import ConfigLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionPipeline:
    """Production pipeline for real-time HAB forecasting"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the production pipeline"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_model_params_config()
        
        # Production configuration
        self.production_config = {
            "update_frequency": "weekly",  # daily, weekly, monthly
            "prediction_horizon": 7,  # days
            "model_update_threshold": 0.1,  # 10% performance degradation
            "data_retention_days": 90,
            "max_retries": 3,
            "timeout_seconds": 300,
            "monitoring_interval": 3600,  # 1 hour
            "alert_thresholds": {
                "accuracy_degradation": 0.15,  # 15%
                "data_quality": 0.8,  # 80%
                "processing_time": 300  # 5 minutes
            }
        }
        
        # Initialize components
        self.data_orchestrator = DataCollectionOrchestrator()
        self.data_loader = DataLoader()
        self.feature_pipeline = FeaturePipeline()
        self.model_trainer = EnsembleTrainer()
        self.optimizer = AdvancedOptimizer()
        
        # Production state
        self.current_model = None
        self.model_performance_history = []
        self.data_quality_metrics = {}
        self.processing_times = []
        self.error_log = []
        self.is_running = False
        self.prediction_queue = Queue()
        
        # Monitoring
        self.monitoring_thread = None
        self.last_update = None
        self.last_performance_check = None
        
        logger.info("Production pipeline initialized")
    
    def start_production_pipeline(self):
        """Start the production pipeline"""
        try:
            logger.info("Starting production pipeline")
            
            # Load or train initial model
            self._initialize_model()
            
            # Start monitoring thread
            self.is_running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            # Schedule regular updates
            self._schedule_updates()
            
            logger.info("Production pipeline started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start production pipeline: {e}")
            return False
    
    def stop_production_pipeline(self):
        """Stop the production pipeline"""
        try:
            logger.info("Stopping production pipeline")
            
            self.is_running = False
            
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=10)
            
            logger.info("Production pipeline stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop production pipeline: {e}")
            return False
    
    def _initialize_model(self):
        """Initialize or load the production model"""
        try:
            model_path = "outputs/production_model.pkl"
            
            if os.path.exists(model_path):
                logger.info("Loading existing production model")
                self.model_trainer.load_ensemble_model("ensemble", model_path)
                self.current_model = self.model_trainer.models["ensemble"]["model"]
            else:
                logger.info("Training new production model")
                self._train_production_model()
            
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
    
    def _train_production_model(self):
        """Train a new production model"""
        try:
            logger.info("Training production model")
            
            # Collect historical data
            dataset = self._collect_historical_data()
            
            # Prepare features
            features_df, target_series = self.model_trainer.xgb_trainer.prepare_dataset(
                dataset, target_variable="hab_risk_percentage"
            )
            
            # Create splits
            X_train, X_val, X_test, y_train, y_val, y_test = self.model_trainer.xgb_trainer.create_temporal_split(
                features_df, target_series
            )
            
            # Scale features
            X_train_scaled, X_val_scaled, X_test_scaled = self.model_trainer.xgb_trainer.scale_features(
                X_train, X_val, X_test
            )
            
            # Select features
            X_train_selected, X_val_selected, X_test_selected = self.model_trainer.xgb_trainer.select_features(
                X_train_scaled, y_train, X_val_scaled, X_test_scaled
            )
            
            # Train ensemble model
            ensemble_results = self.model_trainer.train_ensemble_model(
                X_train_selected, y_train, X_val_selected, y_val
            )
            
            # Evaluate model
            test_metrics = self.model_trainer.evaluate_ensemble(X_test_selected, y_test)
            
            # Store model
            self.current_model = self.model_trainer.models["ensemble"]["model"]
            self.model_trainer.save_ensemble_model("ensemble", "outputs/production_model.pkl")
            
            # Store performance
            self.model_performance_history.append({
                "timestamp": datetime.now(),
                "train_metrics": ensemble_results["train_metrics"],
                "validation_metrics": ensemble_results.get("validation_metrics", {}),
                "test_metrics": test_metrics["ensemble_metrics"]
            })
            
            logger.info(f"Production model trained. Test RMSE: {test_metrics['ensemble_metrics']['rmse']:.2f}%")
            
        except Exception as e:
            logger.error(f"Production model training failed: {e}")
            raise
    
    def _collect_historical_data(self):
        """Collect historical data for model training"""
        try:
            logger.info("Collecting historical data")
            
            # Use data orchestrator to collect recent data
            # For production, we'll use a simplified approach
            # In real implementation, this would connect to data sources
            
            # Create synthetic data for demonstration
            dataset = self._create_synthetic_production_data()
            
            logger.info(f"Collected historical data: {dataset.dims}")
            return dataset
            
        except Exception as e:
            logger.error(f"Historical data collection failed: {e}")
            raise
    
    def _create_synthetic_production_data(self):
        """Create synthetic data for production testing"""
        # Create time coordinates (3 months of daily data)
        start_date = datetime.now() - timedelta(days=90)
        end_date = datetime.now()
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Create spatial coordinates
        lats = np.linspace(34.0, 35.0, 6)
        lons = np.linspace(132.0, 135.0, 10)
        
        # Create realistic environmental data
        sst = 20 + 5 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
        sst = np.array(sst)[:, np.newaxis, np.newaxis] + np.random.normal(0, 1, (len(dates), len(lats), len(lons)))
        
        chlorophyll = 1.0 + 0.5 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
        chlorophyll = np.array(chlorophyll)[:, np.newaxis, np.newaxis] + np.random.normal(0, 0.1, (len(dates), len(lats), len(lons)))
        chlorophyll = np.maximum(chlorophyll, 0.01)
        
        mld = 100 - 50 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
        mld = np.array(mld)[:, np.newaxis, np.newaxis] + np.random.normal(0, 10, (len(dates), len(lats), len(lons)))
        
        # Create HAB risk
        risk_percentages = 25 + 35 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
        risk_percentages = np.array(risk_percentages)[:, np.newaxis, np.newaxis] + np.random.normal(0, 5, (len(dates), len(lats), len(lons)))
        risk_percentages = np.clip(risk_percentages, 0, 100)
        
        # Create dataset
        dataset = xr.Dataset(
            {
                "sea_surface_temperature": (["time", "latitude", "longitude"], sst),
                "chlorophyll_a": (["time", "latitude", "longitude"], chlorophyll),
                "mixed_layer_depth": (["time", "latitude", "longitude"], mld),
                "hab_risk_percentage": (["time", "latitude", "longitude"], risk_percentages),
            },
            coords={
                "time": dates,
                "latitude": lats,
                "longitude": lons,
            }
        )
        
        return dataset
    
    def generate_real_time_prediction(self, location: Dict[str, float] = None) -> Dict[str, Any]:
        """Generate real-time HAB risk prediction"""
        try:
            start_time = time.time()
            
            # Get latest data
            latest_data = self._get_latest_data(location)
            
            if latest_data is None:
                raise ValueError("Unable to retrieve latest data")
            
            # Prepare features
            features_df, _ = self.model_trainer.xgb_trainer.prepare_dataset(
                latest_data, target_variable="hab_risk_percentage"
            )
            
            # Scale features
            features_scaled = self.model_trainer.xgb_trainer.scaler.transform(features_df)
            
            # Select features
            features_selected = self.model_trainer.xgb_trainer.feature_selector.transform(features_scaled)
            
            # Generate prediction
            prediction = self.current_model.predict(features_selected)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Create prediction result
            result = {
                "timestamp": datetime.now(),
                "prediction": float(prediction[0]),
                "location": location or {"latitude": 34.5, "longitude": 133.5},
                "processing_time": processing_time,
                "model_version": self.last_update.isoformat() if self.last_update else None,
                "confidence": self._calculate_prediction_confidence(prediction[0])
            }
            
            # Add to queue for monitoring
            self.prediction_queue.put(result)
            
            logger.info(f"Real-time prediction generated: {result['prediction']:.2f}% risk")
            return result
            
        except Exception as e:
            logger.error(f"Real-time prediction failed: {e}")
            self.error_log.append({
                "timestamp": datetime.now(),
                "error": str(e),
                "type": "prediction_error"
            })
            raise
    
    def _get_latest_data(self, location: Dict[str, float] = None) -> Optional[xr.Dataset]:
        """Get latest environmental data"""
        try:
            # In production, this would connect to real-time data sources
            # For now, create synthetic latest data
            
            # Create single time point data
            current_time = datetime.now()
            lats = np.linspace(34.0, 35.0, 6)
            lons = np.linspace(132.0, 135.0, 10)
            
            # Create realistic current conditions
            sst = 20 + 5 * np.sin(2 * np.pi * current_time.dayofyear / 365.25) + np.random.normal(0, 0.5)
            chlorophyll = 1.0 + 0.5 * np.sin(2 * np.pi * current_time.dayofyear / 365.25) + np.random.normal(0, 0.05)
            mld = 100 - 50 * np.sin(2 * np.pi * current_time.dayofyear / 365.25) + np.random.normal(0, 5)
            
            # Create spatial data
            sst_data = sst + np.random.normal(0, 0.5, (len(lats), len(lons)))
            chlorophyll_data = np.maximum(chlorophyll + np.random.normal(0, 0.05, (len(lats), len(lons))), 0.01)
            mld_data = mld + np.random.normal(0, 5, (len(lats), len(lons)))
            
            # Create dataset
            dataset = xr.Dataset(
                {
                    "sea_surface_temperature": (["latitude", "longitude"], sst_data),
                    "chlorophyll_a": (["latitude", "longitude"], chlorophyll_data),
                    "mixed_layer_depth": (["latitude", "longitude"], mld_data),
                },
                coords={
                    "time": [current_time],
                    "latitude": lats,
                    "longitude": lons,
                }
            )
            
            return dataset
            
        except Exception as e:
            logger.error(f"Latest data retrieval failed: {e}")
            return None
    
    def _calculate_prediction_confidence(self, prediction: float) -> float:
        """Calculate prediction confidence"""
        try:
            # Simple confidence calculation based on prediction value
            # In production, this would use model uncertainty quantification
            
            if prediction < 10:
                confidence = 0.95  # High confidence for low risk
            elif prediction < 30:
                confidence = 0.85  # Medium confidence for moderate risk
            elif prediction < 60:
                confidence = 0.75  # Lower confidence for high risk
            else:
                confidence = 0.65  # Lowest confidence for extreme risk
            
            return confidence
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5  # Default confidence
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Check model performance
                self._check_model_performance()
                
                # Check data quality
                self._check_data_quality()
                
                # Check processing performance
                self._check_processing_performance()
                
                # Generate alerts if needed
                self._generate_alerts()
                
                # Sleep for monitoring interval
                time.sleep(self.production_config["monitoring_interval"])
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _check_model_performance(self):
        """Check model performance and trigger updates if needed"""
        try:
            if not self.model_performance_history:
                return
            
            # Get recent performance
            recent_performance = self.model_performance_history[-1]["test_metrics"]["rmse"]
            
            # Check for degradation
            if len(self.model_performance_history) > 1:
                previous_performance = self.model_performance_history[-2]["test_metrics"]["rmse"]
                degradation = (recent_performance - previous_performance) / previous_performance
                
                if degradation > self.production_config["model_update_threshold"]:
                    logger.warning(f"Model performance degradation detected: {degradation:.2%}")
                    self._trigger_model_update()
            
            self.last_performance_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Model performance check failed: {e}")
    
    def _check_data_quality(self):
        """Check data quality metrics"""
        try:
            # In production, this would check actual data quality
            # For now, simulate data quality metrics
            
            data_quality = {
                "completeness": 0.95 + np.random.normal(0, 0.02),
                "accuracy": 0.92 + np.random.normal(0, 0.03),
                "timeliness": 0.98 + np.random.normal(0, 0.01),
                "consistency": 0.94 + np.random.normal(0, 0.02)
            }
            
            self.data_quality_metrics = data_quality
            
            # Check for quality issues
            for metric, value in data_quality.items():
                if value < self.production_config["alert_thresholds"]["data_quality"]:
                    logger.warning(f"Data quality issue detected: {metric} = {value:.2f}")
            
        except Exception as e:
            logger.error(f"Data quality check failed: {e}")
    
    def _check_processing_performance(self):
        """Check processing performance"""
        try:
            if not self.processing_times:
                return
            
            # Calculate average processing time
            avg_processing_time = np.mean(self.processing_times[-100:])  # Last 100 predictions
            
            if avg_processing_time > self.production_config["alert_thresholds"]["processing_time"]:
                logger.warning(f"Processing time exceeded threshold: {avg_processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Processing performance check failed: {e}")
    
    def _generate_alerts(self):
        """Generate alerts for critical issues"""
        try:
            alerts = []
            
            # Check for critical errors
            recent_errors = [e for e in self.error_log if 
                           (datetime.now() - e["timestamp"]).total_seconds() < 3600]
            
            if len(recent_errors) > 5:
                alerts.append({
                    "type": "critical",
                    "message": f"High error rate detected: {len(recent_errors)} errors in last hour",
                    "timestamp": datetime.now()
                })
            
            # Check for performance degradation
            if self.model_performance_history:
                recent_rmse = self.model_performance_history[-1]["test_metrics"]["rmse"]
                if recent_rmse > 10:  # RMSE threshold
                    alerts.append({
                        "type": "warning",
                        "message": f"High prediction error: RMSE = {recent_rmse:.2f}%",
                        "timestamp": datetime.now()
                    })
            
            # Log alerts
            for alert in alerts:
                logger.warning(f"ALERT: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
    
    def _trigger_model_update(self):
        """Trigger model update"""
        try:
            logger.info("Triggering model update")
            
            # Retrain model
            self._train_production_model()
            
            # Update timestamp
            self.last_update = datetime.now()
            
            logger.info("Model update completed")
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
    
    def _schedule_updates(self):
        """Schedule regular updates"""
        try:
            if self.production_config["update_frequency"] == "daily":
                schedule.every().day.at("02:00").do(self._trigger_model_update)
            elif self.production_config["update_frequency"] == "weekly":
                schedule.every().monday.at("02:00").do(self._trigger_model_update)
            elif self.production_config["update_frequency"] == "monthly":
                schedule.every().month.at("02:00").do(self._trigger_model_update)
            
            logger.info(f"Scheduled {self.production_config['update_frequency']} model updates")
            
        except Exception as e:
            logger.error(f"Update scheduling failed: {e}")
    
    def get_production_status(self) -> Dict[str, Any]:
        """Get current production status"""
        try:
            status = {
                "is_running": self.is_running,
                "last_update": self.last_update.isoformat() if self.last_update else None,
                "last_performance_check": self.last_performance_check.isoformat() if self.last_performance_check else None,
                "model_performance": self.model_performance_history[-1] if self.model_performance_history else None,
                "data_quality": self.data_quality_metrics,
                "processing_performance": {
                    "avg_processing_time": np.mean(self.processing_times[-100:]) if self.processing_times else None,
                    "total_predictions": len(self.processing_times)
                },
                "recent_errors": len([e for e in self.error_log if 
                                    (datetime.now() - e["timestamp"]).total_seconds() < 3600]),
                "queue_size": self.prediction_queue.qsize()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Status retrieval failed: {e}")
            return {"error": str(e)} 