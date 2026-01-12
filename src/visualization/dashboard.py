"""
HAB Forecasting Dashboard

A simple dashboard interface for the HAB forecasting system that provides:
- Risk map visualization
- Time series plots
- System status monitoring
- Configuration management
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
import xarray as xr
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.main_orchestrator import HABForecastOrchestrator
from src.utils.config_loader import ConfigLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HABForecastDashboard:
    """
    Dashboard interface for the HAB forecasting system
    """
    
    def __init__(self):
        """Initialize the dashboard"""
        self.orchestrator = HABForecastOrchestrator()
        self.config_loader = ConfigLoader()
        
        # Dashboard state
        self.current_status = {}
        self.latest_predictions = None
        self.system_health = {}
        
        logger.info("HAB Forecast Dashboard initialized")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        logger.info("Getting system status")
        
        try:
            # Get pipeline status
            pipeline_status = self.orchestrator.get_pipeline_status()
            
            # Check data availability
            data_status = self._check_data_availability()
            
            # Check model status
            model_status = self._check_model_status()
            
            # Check latest predictions
            prediction_status = self._check_prediction_status()
            
            # Compile system status
            system_status = {
                "timestamp": datetime.now().isoformat(),
                "pipeline_status": pipeline_status,
                "data_status": data_status,
                "model_status": model_status,
                "prediction_status": prediction_status,
                "system_health": self._calculate_system_health()
            }
            
            self.current_status = system_status
            return system_status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}
    
    def _check_data_availability(self) -> Dict[str, Any]:
        """Check data availability"""
        try:
            data_dir = Path("data")
            raw_dir = data_dir / "raw"
            processed_dir = data_dir / "processed"
            
            data_status = {
                "raw_data_available": raw_dir.exists() and any(raw_dir.iterdir()),
                "processed_data_available": processed_dir.exists() and any(processed_dir.iterdir()),
                "latest_data_date": None,
                "data_sources": []
            }
            
            # Check for latest data
            if processed_dir.exists():
                nc_files = list(processed_dir.glob("*.nc"))
                if nc_files:
                    latest_file = max(nc_files, key=lambda x: x.stat().st_mtime)
                    data_status["latest_data_date"] = datetime.fromtimestamp(
                        latest_file.stat().st_mtime
                    ).strftime("%Y-%m-%d")
            
            # Check data sources
            if raw_dir.exists():
                for source_dir in raw_dir.iterdir():
                    if source_dir.is_dir():
                        data_status["data_sources"].append(source_dir.name)
            
            return data_status
            
        except Exception as e:
            logger.error(f"Data availability check failed: {e}")
            return {"error": str(e)}
    
    def _check_model_status(self) -> Dict[str, Any]:
        """Check model status"""
        try:
            model_dir = Path("outputs/models")
            model_status = {
                "model_available": False,
                "model_path": None,
                "last_training_date": None,
                "model_performance": None
            }
            
            if model_dir.exists():
                model_files = list(model_dir.glob("*.json"))
                if model_files:
                    model_status["model_available"] = True
                    model_status["model_path"] = str(model_files[0])
                    model_status["last_training_date"] = datetime.fromtimestamp(
                        model_files[0].stat().st_mtime
                    ).strftime("%Y-%m-%d %H:%M:%S")
            
            return model_status
            
        except Exception as e:
            logger.error(f"Model status check failed: {e}")
            return {"error": str(e)}
    
    def _check_prediction_status(self) -> Dict[str, Any]:
        """Check prediction status"""
        try:
            prediction_dir = Path("outputs/predictions")
            prediction_status = {
                "predictions_available": False,
                "latest_prediction_date": None,
                "prediction_summary": None
            }
            
            if prediction_dir.exists():
                prediction_files = list(prediction_dir.glob("*.nc"))
                if prediction_files:
                    prediction_status["predictions_available"] = True
                    latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
                    prediction_status["latest_prediction_date"] = datetime.fromtimestamp(
                        latest_file.stat().st_mtime
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Load prediction summary
                    try:
                        pred_data = xr.open_dataset(latest_file)
                        if "predicted_risk_percentage" in pred_data:
                            risk_data = pred_data["predicted_risk_percentage"].values
                            prediction_status["prediction_summary"] = {
                                "mean_risk": float(np.mean(risk_data)),
                                "max_risk": float(np.max(risk_data)),
                                "min_risk": float(np.min(risk_data)),
                                "risk_range": f"{float(np.min(risk_data)):.1f}% - {float(np.max(risk_data)):.1f}%"
                            }
                    except Exception as e:
                        logger.warning(f"Could not load prediction summary: {e}")
            
            return prediction_status
            
        except Exception as e:
            logger.error(f"Prediction status check failed: {e}")
            return {"error": str(e)}
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health"""
        try:
            health_score = 0
            health_issues = []
            
            # Check data availability
            if self.current_status.get("data_status", {}).get("processed_data_available", False):
                health_score += 25
            else:
                health_issues.append("No processed data available")
            
            # Check model availability
            if self.current_status.get("model_status", {}).get("model_available", False):
                health_score += 25
            else:
                health_issues.append("No trained model available")
            
            # Check predictions availability
            if self.current_status.get("prediction_status", {}).get("predictions_available", False):
                health_score += 25
            else:
                health_issues.append("No predictions available")
            
            # Check pipeline status
            pipeline_state = self.current_status.get("pipeline_status", {}).get("pipeline_state", {})
            if pipeline_state.get("model_trained", False):
                health_score += 25
            else:
                health_issues.append("Pipeline not fully executed")
            
            # Determine health status
            if health_score >= 75:
                health_status = "Healthy"
            elif health_score >= 50:
                health_status = "Warning"
            else:
                health_status = "Critical"
            
            return {
                "health_score": health_score,
                "health_status": health_status,
                "health_issues": health_issues,
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"System health calculation failed: {e}")
            return {"error": str(e)}
    
    def run_quick_pipeline(self, days_back: int = 30) -> Dict[str, Any]:
        """Run a quick pipeline execution"""
        logger.info(f"Running quick pipeline for last {days_back} days")
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Run pipeline
            results = self.orchestrator.run_complete_pipeline(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                force_retrain=False,
                generate_predictions=True
            )
            
            # Update status
            self.get_system_status()
            
            return {
                "success": True,
                "pipeline_results": results,
                "execution_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Quick pipeline execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": datetime.now().isoformat()
            }
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk summary"""
        logger.info("Getting risk summary")
        
        try:
            prediction_dir = Path("outputs/predictions")
            if not prediction_dir.exists():
                return {"error": "No predictions directory found"}
            
            prediction_files = list(prediction_dir.glob("*.nc"))
            if not prediction_files:
                return {"error": "No prediction files found"}
            
            # Load latest prediction
            latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
            pred_data = xr.open_dataset(latest_file)
            
            if "predicted_risk_percentage" not in pred_data:
                return {"error": "No risk predictions found in data"}
            
            risk_data = pred_data["predicted_risk_percentage"].values
            
            # Calculate risk statistics
            risk_stats = {
                "mean_risk": float(np.mean(risk_data)),
                "median_risk": float(np.median(risk_data)),
                "std_risk": float(np.std(risk_data)),
                "min_risk": float(np.min(risk_data)),
                "max_risk": float(np.max(risk_data)),
                "low_risk_count": int((risk_data <= 25).sum()),
                "moderate_risk_count": int(((risk_data > 25) & (risk_data <= 50)).sum()),
                "high_risk_count": int(((risk_data > 50) & (risk_data <= 75)).sum()),
                "extreme_risk_count": int((risk_data > 75).sum()),
                "total_points": int(risk_data.size)
            }
            
            # Calculate percentages
            total = risk_stats["total_points"]
            risk_stats.update({
                "low_risk_percentage": risk_stats["low_risk_count"] / total * 100,
                "moderate_risk_percentage": risk_stats["moderate_risk_count"] / total * 100,
                "high_risk_percentage": risk_stats["high_risk_count"] / total * 100,
                "extreme_risk_percentage": risk_stats["extreme_risk_count"] / total * 100
            })
            
            return {
                "prediction_date": datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat(),
                "risk_statistics": risk_stats,
                "risk_assessment": self._assess_risk_level(risk_stats["mean_risk"])
            }
            
        except Exception as e:
            logger.error(f"Risk summary generation failed: {e}")
            return {"error": str(e)}
    
    def _assess_risk_level(self, mean_risk: float) -> Dict[str, Any]:
        """Assess overall risk level"""
        if mean_risk <= 25:
            risk_level = "Low"
            risk_description = "Minimal HAB risk conditions"
            risk_color = "green"
        elif mean_risk <= 50:
            risk_level = "Moderate"
            risk_description = "Moderate HAB risk conditions"
            risk_color = "yellow"
        elif mean_risk <= 75:
            risk_level = "High"
            risk_description = "High HAB risk conditions"
            risk_color = "orange"
        else:
            risk_level = "Extreme"
            risk_description = "Extreme HAB risk conditions"
            risk_color = "red"
        
        return {
            "risk_level": risk_level,
            "risk_description": risk_description,
            "risk_color": risk_color,
            "mean_risk_percentage": mean_risk
        }
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        logger.info("Getting configuration summary")
        
        try:
            configs = {
                "data_sources": self.config_loader.get_data_sources_config(),
                "processing": self.config_loader.get_processing_config(),
                "features": self.config_loader.get_features_config(),
                "model_params": self.config_loader.get_model_params_config()
            }
            
            # Create summary
            summary = {}
            for config_name, config_data in configs.items():
                if config_data:
                    summary[config_name] = {
                        "sections": list(config_data.keys()),
                        "last_modified": datetime.now().isoformat()  # Placeholder
                    }
                else:
                    summary[config_name] = {"error": "Configuration not loaded"}
            
            return summary
            
        except Exception as e:
            logger.error(f"Configuration summary generation failed: {e}")
            return {"error": str(e)} 