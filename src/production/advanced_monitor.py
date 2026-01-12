"""
Advanced Monitoring System for HAB Forecasting

This module implements advanced monitoring capabilities:
- Model drift detection
- Performance tracking
- Automated retraining triggers
- Alert system for accuracy degradation
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
from collections import deque
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedMonitor:
    """Advanced monitoring system for HAB forecasting"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the advanced monitor"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_model_params_config()
        
        # Monitoring configuration
        self.monitoring_config = {
            "drift_detection": {
                "window_size": 100,  # Number of predictions for drift detection
                "detection_threshold": 0.1,  # 10% change threshold
                "statistical_test": "ks_test",  # ks_test, ad_test, cvm_test
                "confidence_level": 0.95
            },
            "performance_tracking": {
                "metrics": ["rmse", "mae", "r2_score", "accuracy"],
                "tracking_window": 30,  # days
                "degradation_threshold": 0.15,  # 15% degradation
                "improvement_threshold": 0.05  # 5% improvement
            },
            "alert_system": {
                "alert_levels": ["info", "warning", "critical"],
                "notification_channels": ["log", "email", "webhook"],
                "alert_cooldown": 3600,  # 1 hour between alerts
                "max_alerts_per_hour": 10
            },
            "retraining_triggers": {
                "performance_degradation": 0.15,  # 15% degradation
                "drift_detected": True,
                "data_quality_issues": 0.8,  # 80% quality threshold
                "time_based": "weekly"  # daily, weekly, monthly
            }
        }
        
        # Monitoring state
        self.prediction_history = deque(maxlen=self.monitoring_config["drift_detection"]["window_size"])
        self.performance_history = []
        self.drift_alerts = []
        self.performance_alerts = []
        self.retraining_triggers = []
        self.alert_history = deque(maxlen=1000)
        
        # Statistical tracking
        self.baseline_stats = {}
        self.current_stats = {}
        self.drift_scores = {}
        
        # Threading
        self.is_monitoring = False
        self.monitoring_thread = None
        
        logger.info("Advanced monitor initialized")
    
    def start_monitoring(self):
        """Start the monitoring system"""
        try:
            logger.info("Starting advanced monitoring system")
            
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            logger.info("Advanced monitoring system started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        try:
            logger.info("Stopping advanced monitoring system")
            
            self.is_monitoring = False
            
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=10)
            
            logger.info("Advanced monitoring system stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
            return False
    
    def add_prediction(self, prediction: float, actual: float = None, timestamp: datetime = None):
        """Add a new prediction to the monitoring system"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            prediction_record = {
                "timestamp": timestamp,
                "prediction": prediction,
                "actual": actual,
                "error": actual - prediction if actual is not None else None
            }
            
            self.prediction_history.append(prediction_record)
            
            # Update current statistics
            self._update_current_stats()
            
            # Check for drift if we have enough data
            if len(self.prediction_history) >= self.monitoring_config["drift_detection"]["window_size"]:
                self._check_for_drift()
            
            # Check performance if we have actual values
            if actual is not None:
                self._check_performance_degradation()
            
        except Exception as e:
            logger.error(f"Failed to add prediction: {e}")
    
    def add_performance_metrics(self, metrics: Dict[str, float], timestamp: datetime = None):
        """Add performance metrics to the monitoring system"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            performance_record = {
                "timestamp": timestamp,
                "metrics": metrics
            }
            
            self.performance_history.append(performance_record)
            
            # Check for performance degradation
            self._check_performance_degradation()
            
        except Exception as e:
            logger.error(f"Failed to add performance metrics: {e}")
    
    def _update_current_stats(self):
        """Update current statistics"""
        try:
            if len(self.prediction_history) < 10:
                return
            
            predictions = [p["prediction"] for p in self.prediction_history]
            
            self.current_stats = {
                "mean": np.mean(predictions),
                "std": np.std(predictions),
                "min": np.min(predictions),
                "max": np.max(predictions),
                "median": np.median(predictions),
                "q25": np.percentile(predictions, 25),
                "q75": np.percentile(predictions, 75)
            }
            
        except Exception as e:
            logger.error(f"Failed to update current stats: {e}")
    
    def _check_for_drift(self):
        """Check for model drift using statistical tests"""
        try:
            if len(self.prediction_history) < self.monitoring_config["drift_detection"]["window_size"]:
                return
            
            # Split data into baseline and current windows
            window_size = self.monitoring_config["drift_detection"]["window_size"]
            baseline_data = list(self.prediction_history)[:window_size//2]
            current_data = list(self.prediction_history)[window_size//2:]
            
            baseline_predictions = [p["prediction"] for p in baseline_data]
            current_predictions = [p["prediction"] for p in current_data]
            
            # Perform statistical test
            drift_detected = self._statistical_drift_test(baseline_predictions, current_predictions)
            
            if drift_detected:
                drift_alert = {
                    "timestamp": datetime.now(),
                    "type": "drift_detected",
                    "severity": "warning",
                    "message": "Model drift detected using statistical test",
                    "baseline_stats": self._calculate_stats(baseline_predictions),
                    "current_stats": self._calculate_stats(current_predictions)
                }
                
                self.drift_alerts.append(drift_alert)
                self._generate_alert(drift_alert)
                
                # Trigger retraining if configured
                if self.monitoring_config["retraining_triggers"]["drift_detected"]:
                    self._trigger_retraining("drift_detected", drift_alert)
            
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
    
    def _statistical_drift_test(self, baseline_data: List[float], current_data: List[float]) -> bool:
        """Perform statistical drift test"""
        try:
            from scipy import stats
            
            test_type = self.monitoring_config["drift_detection"]["statistical_test"]
            confidence_level = self.monitoring_config["drift_detection"]["confidence_level"]
            
            if test_type == "ks_test":
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(baseline_data, current_data)
                drift_detected = p_value < (1 - confidence_level)
                
            elif test_type == "ad_test":
                # Anderson-Darling test (approximation)
                statistic, p_value = stats.anderson_ksamp([baseline_data, current_data])
                drift_detected = p_value < (1 - confidence_level)
                
            elif test_type == "cvm_test":
                # Cramér-von Mises test
                statistic, p_value = stats.cramervonmises_2samp(baseline_data, current_data)
                drift_detected = p_value < (1 - confidence_level)
                
            else:
                # Default to KS test
                statistic, p_value = stats.ks_2samp(baseline_data, current_data)
                drift_detected = p_value < (1 - confidence_level)
            
            # Store drift score
            self.drift_scores[datetime.now()] = {
                "test_type": test_type,
                "statistic": statistic,
                "p_value": p_value,
                "drift_detected": drift_detected
            }
            
            return drift_detected
            
        except Exception as e:
            logger.error(f"Statistical drift test failed: {e}")
            return False
    
    def _check_performance_degradation(self):
        """Check for performance degradation"""
        try:
            if len(self.performance_history) < 2:
                return
            
            # Get recent performance metrics
            recent_metrics = self.performance_history[-1]["metrics"]
            previous_metrics = self.performance_history[-2]["metrics"]
            
            degradation_detected = False
            degradation_details = {}
            
            for metric in self.monitoring_config["performance_tracking"]["metrics"]:
                if metric in recent_metrics and metric in previous_metrics:
                    recent_value = recent_metrics[metric]
                    previous_value = previous_metrics[metric]
                    
                    # Calculate degradation (higher is worse for RMSE, MAE; lower is worse for R², accuracy)
                    if metric in ["rmse", "mae"]:
                        degradation = (recent_value - previous_value) / previous_value
                    else:
                        degradation = (previous_value - recent_value) / previous_value
                    
                    threshold = self.monitoring_config["performance_tracking"]["degradation_threshold"]
                    
                    if degradation > threshold:
                        degradation_detected = True
                        degradation_details[metric] = {
                            "previous": previous_value,
                            "current": recent_value,
                            "degradation": degradation
                        }
            
            if degradation_detected:
                performance_alert = {
                    "timestamp": datetime.now(),
                    "type": "performance_degradation",
                    "severity": "warning",
                    "message": "Performance degradation detected",
                    "degradation_details": degradation_details
                }
                
                self.performance_alerts.append(performance_alert)
                self._generate_alert(performance_alert)
                
                # Check if retraining should be triggered
                if degradation_detected and any(
                    degradation > self.monitoring_config["retraining_triggers"]["performance_degradation"]
                    for degradation in [d["degradation"] for d in degradation_details.values()]
                ):
                    self._trigger_retraining("performance_degradation", performance_alert)
            
        except Exception as e:
            logger.error(f"Performance degradation check failed: {e}")
    
    def _calculate_stats(self, data: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for data"""
        try:
            return {
                "mean": np.mean(data),
                "std": np.std(data),
                "min": np.min(data),
                "max": np.max(data),
                "median": np.median(data),
                "count": len(data)
            }
        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return {}
    
    def _generate_alert(self, alert: Dict[str, Any]):
        """Generate and log alert"""
        try:
            # Add to alert history
            self.alert_history.append(alert)
            
            # Log alert
            logger.warning(f"ALERT [{alert['severity'].upper()}]: {alert['message']}")
            
            # Check alert cooldown
            recent_alerts = [a for a in self.alert_history if 
                           (datetime.now() - a["timestamp"]).total_seconds() < 
                           self.monitoring_config["alert_system"]["alert_cooldown"]]
            
            if len(recent_alerts) > self.monitoring_config["alert_system"]["max_alerts_per_hour"]:
                logger.warning("Alert rate limit exceeded")
            
        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
    
    def _trigger_retraining(self, trigger_type: str, alert: Dict[str, Any]):
        """Trigger model retraining"""
        try:
            retraining_trigger = {
                "timestamp": datetime.now(),
                "trigger_type": trigger_type,
                "alert": alert,
                "status": "pending"
            }
            
            self.retraining_triggers.append(retraining_trigger)
            
            logger.warning(f"Retraining triggered: {trigger_type}")
            
            # In production, this would call the retraining pipeline
            # For now, just log the trigger
            
        except Exception as e:
            logger.error(f"Retraining trigger failed: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Check for time-based retraining triggers
                self._check_time_based_triggers()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Sleep for monitoring interval
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _check_time_based_triggers(self):
        """Check for time-based retraining triggers"""
        try:
            time_based = self.monitoring_config["retraining_triggers"]["time_based"]
            
            if time_based == "daily":
                # Check if it's time for daily retraining (e.g., 2 AM)
                if datetime.now().hour == 2 and datetime.now().minute < 10:
                    self._trigger_retraining("time_based", {
                        "type": "time_based",
                        "message": "Daily retraining trigger"
                    })
                    
            elif time_based == "weekly":
                # Check if it's Monday and time for weekly retraining
                if datetime.now().weekday() == 0 and datetime.now().hour == 2 and datetime.now().minute < 10:
                    self._trigger_retraining("time_based", {
                        "type": "time_based",
                        "message": "Weekly retraining trigger"
                    })
                    
            elif time_based == "monthly":
                # Check if it's the first day of the month and time for monthly retraining
                if datetime.now().day == 1 and datetime.now().hour == 2 and datetime.now().minute < 10:
                    self._trigger_retraining("time_based", {
                        "type": "time_based",
                        "message": "Monthly retraining trigger"
                    })
                    
        except Exception as e:
            logger.error(f"Time-based trigger check failed: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            # Keep only recent performance history (30 days)
            cutoff_date = datetime.now() - timedelta(days=30)
            self.performance_history = [
                p for p in self.performance_history 
                if p["timestamp"] > cutoff_date
            ]
            
            # Keep only recent drift scores (7 days)
            cutoff_date = datetime.now() - timedelta(days=7)
            self.drift_scores = {
                k: v for k, v in self.drift_scores.items()
                if k > cutoff_date
            }
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        try:
            status = {
                "is_monitoring": self.is_monitoring,
                "prediction_history_size": len(self.prediction_history),
                "performance_history_size": len(self.performance_history),
                "drift_alerts_count": len(self.drift_alerts),
                "performance_alerts_count": len(self.performance_alerts),
                "retraining_triggers_count": len(self.retraining_triggers),
                "recent_alerts_count": len([a for a in self.alert_history if 
                                          (datetime.now() - a["timestamp"]).total_seconds() < 3600]),
                "current_stats": self.current_stats,
                "baseline_stats": self.baseline_stats,
                "recent_drift_scores": dict(list(self.drift_scores.items())[-5:])  # Last 5 drift scores
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Status retrieval failed: {e}")
            return {"error": str(e)}
    
    def set_baseline_stats(self, baseline_data: List[float]):
        """Set baseline statistics for drift detection"""
        try:
            self.baseline_stats = self._calculate_stats(baseline_data)
            logger.info("Baseline statistics set for drift detection")
            
        except Exception as e:
            logger.error(f"Failed to set baseline stats: {e}")
    
    def export_monitoring_data(self, filepath: str):
        """Export monitoring data to file"""
        try:
            monitoring_data = {
                "prediction_history": list(self.prediction_history),
                "performance_history": self.performance_history,
                "drift_alerts": self.drift_alerts,
                "performance_alerts": self.performance_alerts,
                "retraining_triggers": self.retraining_triggers,
                "drift_scores": self.drift_scores,
                "current_stats": self.current_stats,
                "baseline_stats": self.baseline_stats,
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(monitoring_data, f, default=str, indent=2)
            
            logger.info(f"Monitoring data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Monitoring data export failed: {e}")
            raise 