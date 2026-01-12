"""
Automated Data Quality Monitoring and Alerting System

This module provides automated data quality monitoring and alerting capabilities
for the HAB forecasting system. It integrates with existing quality control
components and provides real-time monitoring, alerting, and reporting.

Features:
- Real-time quality monitoring for all data sources
- Automated alerting for quality issues
- Quality trend analysis and reporting
- Integration with existing quality control systems
- Configurable alert thresholds and notification channels
"""

import logging
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict, deque
import xarray as xr
import numpy as np
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

from src.utils.config_loader import ConfigLoader, get_data_dir
from .quality_control import DataQualityController

logger = logging.getLogger(__name__)

class QualityAlert:
    """Represents a quality alert"""
    
    def __init__(self, alert_type: str, severity: str, message: str, 
                 data_source: str, variable: str = None, details: Dict = None):
        self.alert_type = alert_type
        self.severity = severity  # info, warning, critical
        self.message = message
        self.data_source = data_source
        self.variable = variable
        self.details = details or {}
        self.timestamp = datetime.now()
        self.acknowledged = False
        self.resolved = False
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "data_source": self.data_source,
            "variable": self.variable,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved
        }

class QualityMonitor:
    """Automated data quality monitoring and alerting system"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the quality monitor"""
        self.config_loader = ConfigLoader()
        self.config = config or self.config_loader.get_data_sources_config()
        
        # Initialize quality controller
        self.quality_controller = DataQualityController(config)
        
        # Monitoring configuration
        self.monitoring_config = self._load_monitoring_config()
        
        # Alert configuration
        self.alert_config = self._load_alert_config()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.quality_history = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts = []
        self.alert_history = deque(maxlen=10000)
        self.quality_trends = {}
        
        # Data source tracking
        self.data_sources = {
            'copernicus': {'status': 'unknown', 'last_check': None, 'quality_score': 0.0},
            'gee': {'status': 'unknown', 'last_check': None, 'quality_score': 0.0},
            'erddap': {'status': 'unknown', 'last_check': None, 'quality_score': 0.0}
        }
        
        # Output directories
        self.data_dir = get_data_dir()
        self.monitoring_dir = self.data_dir / "quality_monitoring"
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Alert cooldown tracking
        self.alert_cooldowns = {}
        
        logger.info("Quality monitor initialized")
    
    def _load_monitoring_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        return {
            "monitoring_interval": 3600,  # 1 hour
            "quality_thresholds": {
                "overall_score_min": 0.7,
                "missing_data_max": 0.5,
                "outlier_ratio_max": 0.1,
                "range_violations_max": 0.05
            },
            "trend_analysis": {
                "window_size": 24,  # hours
                "trend_threshold": 0.1,  # 10% change
                "minimum_data_points": 10
            },
            "data_source_checks": {
                "availability_check": True,
                "freshness_check": True,
                "completeness_check": True
            }
        }
    
    def _load_alert_config(self) -> Dict[str, Any]:
        """Load alert configuration"""
        return {
            "alert_levels": {
                "info": {"cooldown": 3600, "max_per_hour": 10},
                "warning": {"cooldown": 1800, "max_per_hour": 20},
                "critical": {"cooldown": 300, "max_per_hour": 50}
            },
            "notification_channels": {
                "log": True,
                "email": False,
                "webhook": False,
                "file": True
            },
            "email_config": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "recipients": []
            },
            "webhook_config": {
                "url": "",
                "headers": {"Content-Type": "application/json"}
            }
        }
    
    def start_monitoring(self) -> bool:
        """Start the quality monitoring system"""
        try:
            logger.info("Starting quality monitoring system")
            
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            logger.info("Quality monitoring system started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start quality monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop the quality monitoring system"""
        try:
            logger.info("Stopping quality monitoring system")
            
            self.is_monitoring = False
            
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=10)
            
            logger.info("Quality monitoring system stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop quality monitoring: {e}")
            return False
    
    def monitor_dataset_quality(self, dataset: xr.Dataset, data_source: str, 
                              variable: str = None) -> Dict[str, Any]:
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
            logger.info(f"Monitoring quality for {data_source} - {variable or 'all variables'}")
            
            # Run comprehensive quality validation
            validation_results = self.quality_controller.run_comprehensive_validation(
                dataset, f"{data_source}_{variable or 'dataset'}"
            )
            
            # Store quality history
            quality_record = {
                "timestamp": datetime.now(),
                "data_source": data_source,
                "variable": variable,
                "quality_score": validation_results["overall_score"],
                "validation_results": validation_results,
                "dataset_info": {
                    "dims": dict(dataset.dims),
                    "variables": list(dataset.data_vars.keys()),
                    "size": dataset.nbytes
                }
            }
            
            self.quality_history[f"{data_source}_{variable or 'overall'}"].append(quality_record)
            
            # Update data source status
            self._update_data_source_status(data_source, quality_record)
            
            # Check for quality issues and generate alerts
            self._check_quality_issues(quality_record)
            
            # Update quality trends
            self._update_quality_trends(data_source, variable)
            
            return quality_record
            
        except Exception as e:
            logger.error(f"Dataset quality monitoring failed: {e}")
            self._generate_alert("monitoring_error", "critical", 
                               f"Quality monitoring failed for {data_source}: {e}", 
                               data_source, variable)
            return {"error": str(e)}
    
    def _update_data_source_status(self, data_source: str, quality_record: Dict[str, Any]):
        """Update data source status"""
        try:
            self.data_sources[data_source]["status"] = "healthy" if quality_record["quality_score"] >= 0.7 else "degraded"
            self.data_sources[data_source]["last_check"] = datetime.now()
            self.data_sources[data_source]["quality_score"] = quality_record["quality_score"]
            
        except Exception as e:
            logger.error(f"Failed to update data source status: {e}")
    
    def _check_quality_issues(self, quality_record: Dict[str, Any]):
        """Check for quality issues and generate alerts"""
        try:
            data_source = quality_record["data_source"]
            variable = quality_record["variable"]
            quality_score = quality_record["quality_score"]
            validation_results = quality_record["validation_results"]
            
            # Check overall quality score
            if quality_score < self.monitoring_config["quality_thresholds"]["overall_score_min"]:
                self._generate_alert("low_quality_score", "warning",
                                   f"Low quality score ({quality_score:.2f}) for {data_source}",
                                   data_source, variable,
                                   {"quality_score": quality_score, "threshold": self.monitoring_config["quality_thresholds"]["overall_score_min"]})
            
            # Check missing data
            missing_data = validation_results["validation_results"]["values"]["missing_data_analysis"]
            for var_name, missing_info in missing_data.items():
                if missing_info["missing_ratio"] > self.monitoring_config["quality_thresholds"]["missing_data_max"]:
                    self._generate_alert("high_missing_data", "warning",
                                       f"High missing data ratio ({missing_info['missing_ratio']:.2%}) for {var_name}",
                                       data_source, var_name,
                                       {"missing_ratio": missing_info["missing_ratio"], "threshold": self.monitoring_config["quality_thresholds"]["missing_data_max"]})
            
            # Check outliers
            outlier_data = validation_results["validation_results"]["values"]["outlier_analysis"]
            for var_name, outlier_info in outlier_data.items():
                if outlier_info["outlier_ratio"] > self.monitoring_config["quality_thresholds"]["outlier_ratio_max"]:
                    self._generate_alert("high_outlier_ratio", "warning",
                                       f"High outlier ratio ({outlier_info['outlier_ratio']:.2%}) for {var_name}",
                                       data_source, var_name,
                                       {"outlier_ratio": outlier_info["outlier_ratio"], "threshold": self.monitoring_config["quality_thresholds"]["outlier_ratio_max"]})
            
            # Check range violations
            range_violations = validation_results["validation_results"]["values"]["range_validation"]
            violation_count = len([v for v in range_violations.values() if not v["within_range"]])
            if violation_count > 0:
                self._generate_alert("range_violations", "warning",
                                   f"Range violations detected in {violation_count} variables",
                                   data_source, variable,
                                   {"violation_count": violation_count})
            
            # Check for critical issues
            if quality_score < 0.5:
                self._generate_alert("critical_quality_issue", "critical",
                                   f"Critical quality issue detected (score: {quality_score:.2f})",
                                   data_source, variable,
                                   {"quality_score": quality_score})
            
        except Exception as e:
            logger.error(f"Quality issue check failed: {e}")
    
    def _update_quality_trends(self, data_source: str, variable: str = None):
        """Update quality trends analysis"""
        try:
            key = f"{data_source}_{variable or 'overall'}"
            history = self.quality_history[key]
            
            if len(history) < self.monitoring_config["trend_analysis"]["minimum_data_points"]:
                return
            
            # Calculate trend over the specified window
            window_size = self.monitoring_config["trend_analysis"]["window_size"]
            recent_scores = [record["quality_score"] for record in list(history)[-window_size:]]
            
            if len(recent_scores) >= 2:
                # Calculate trend (positive = improving, negative = degrading)
                trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                trend_percentage = (trend / np.mean(recent_scores)) * 100
                
                self.quality_trends[key] = {
                    "trend": trend,
                    "trend_percentage": trend_percentage,
                    "is_improving": trend > 0,
                    "is_degrading": trend < -self.monitoring_config["trend_analysis"]["trend_threshold"],
                    "last_updated": datetime.now()
                }
                
                # Alert on significant degradation
                if self.quality_trends[key]["is_degrading"]:
                    self._generate_alert("quality_degradation", "warning",
                                       f"Quality degradation detected for {key} (trend: {trend_percentage:.1f}%)",
                                       data_source, variable,
                                       {"trend_percentage": trend_percentage})
            
        except Exception as e:
            logger.error(f"Quality trend update failed: {e}")
    
    def _generate_alert(self, alert_type: str, severity: str, message: str,
                       data_source: str, variable: str = None, details: Dict = None):
        """Generate and send quality alert"""
        try:
            # Check cooldown
            alert_key = f"{alert_type}_{data_source}_{variable or 'overall'}"
            if self._is_alert_in_cooldown(alert_key, severity):
                return
            
            # Create alert
            alert = QualityAlert(alert_type, severity, message, data_source, variable, details)
            
            # Add to active alerts and history
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            
            # Send notifications
            self._send_notifications(alert)
            
            # Set cooldown
            self._set_alert_cooldown(alert_key, severity)
            
            logger.warning(f"QUALITY ALERT [{severity.upper()}]: {message}")
            
        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
    
    def _is_alert_in_cooldown(self, alert_key: str, severity: str) -> bool:
        """Check if alert is in cooldown period"""
        try:
            if alert_key not in self.alert_cooldowns:
                return False
            
            cooldown_until = self.alert_cooldowns[alert_key]
            return datetime.now() < cooldown_until
            
        except Exception as e:
            logger.error(f"Cooldown check failed: {e}")
            return False
    
    def _set_alert_cooldown(self, alert_key: str, severity: str):
        """Set alert cooldown period"""
        try:
            cooldown_seconds = self.alert_config["alert_levels"][severity]["cooldown"]
            cooldown_until = datetime.now() + timedelta(seconds=cooldown_seconds)
            self.alert_cooldowns[alert_key] = cooldown_until
            
        except Exception as e:
            logger.error(f"Cooldown setting failed: {e}")
    
    def _send_notifications(self, alert: QualityAlert):
        """Send alert notifications through configured channels"""
        try:
            # Log notification
            if self.alert_config["notification_channels"]["log"]:
                self._log_notification(alert)
            
            # Email notification
            if self.alert_config["notification_channels"]["email"]:
                self._send_email_notification(alert)
            
            # Webhook notification
            if self.alert_config["notification_channels"]["webhook"]:
                self._send_webhook_notification(alert)
            
            # File notification
            if self.alert_config["notification_channels"]["file"]:
                self._save_notification_to_file(alert)
            
        except Exception as e:
            logger.error(f"Notification sending failed: {e}")
    
    def _log_notification(self, alert: QualityAlert):
        """Log notification"""
        logger.warning(f"QUALITY ALERT [{alert.severity.upper()}]: {alert.message}")
        if alert.details:
            logger.warning(f"Alert details: {alert.details}")
    
    def _send_email_notification(self, alert: QualityAlert):
        """Send email notification"""
        try:
            email_config = self.alert_config["email_config"]
            
            if not email_config["username"] or not email_config["recipients"]:
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config["username"]
            msg['To'] = ", ".join(email_config["recipients"])
            msg['Subject'] = f"HAB Forecasting - Quality Alert [{alert.severity.upper()}]"
            
            # Create email body
            body = f"""
            Quality Alert Detected
            
            Type: {alert.alert_type}
            Severity: {alert.severity.upper()}
            Data Source: {alert.data_source}
            Variable: {alert.variable or 'All variables'}
            Message: {alert.message}
            Timestamp: {alert.timestamp}
            
            Details: {json.dumps(alert.details, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
            server.starttls()
            server.login(email_config["username"], email_config["password"])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert: {alert.alert_type}")
            
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
    
    def _send_webhook_notification(self, alert: QualityAlert):
        """Send webhook notification"""
        try:
            webhook_config = self.alert_config["webhook_config"]
            
            if not webhook_config["url"]:
                return
            
            # Prepare payload
            payload = {
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "data_source": alert.data_source,
                "variable": alert.variable,
                "timestamp": alert.timestamp.isoformat(),
                "details": alert.details
            }
            
            # Send webhook
            response = requests.post(
                webhook_config["url"],
                json=payload,
                headers=webhook_config["headers"],
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook notification sent for alert: {alert.alert_type}")
            else:
                logger.warning(f"Webhook notification failed with status {response.status_code}")
            
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
    
    def _save_notification_to_file(self, alert: QualityAlert):
        """Save notification to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quality_alert_{alert.severity}_{timestamp}.json"
            filepath = self.monitoring_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(alert.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Alert saved to file: {filepath}")
            
        except Exception as e:
            logger.error(f"File notification failed: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Check data source availability
                self._check_data_source_availability()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Generate monitoring report
                self._generate_monitoring_report()
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_config["monitoring_interval"])
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _check_data_source_availability(self):
        """Check data source availability"""
        try:
            for data_source, status_info in self.data_sources.items():
                if status_info["last_check"] is None:
                    continue
                
                # Check if data source is stale (no updates in 24 hours)
                stale_threshold = datetime.now() - timedelta(hours=24)
                if status_info["last_check"] < stale_threshold:
                    self._generate_alert("data_source_stale", "warning",
                                       f"Data source {data_source} has not been updated recently",
                                       data_source)
                
                # Check if data source is consistently low quality
                if status_info["quality_score"] < 0.5:
                    self._generate_alert("data_source_low_quality", "critical",
                                       f"Data source {data_source} has consistently low quality",
                                       data_source,
                                       {"quality_score": status_info["quality_score"]})
            
        except Exception as e:
            logger.error(f"Data source availability check failed: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            # Clean up old quality history (keep last 30 days)
            cutoff_date = datetime.now() - timedelta(days=30)
            for key in list(self.quality_history.keys()):
                self.quality_history[key] = deque(
                    [record for record in self.quality_history[key] 
                     if record["timestamp"] > cutoff_date],
                    maxlen=1000
                )
            
            # Clean up resolved alerts (older than 7 days)
            cutoff_date = datetime.now() - timedelta(days=7)
            self.active_alerts = [
                alert for alert in self.active_alerts
                if not alert.resolved or alert.timestamp > cutoff_date
            ]
            
            # Clean up old cooldowns
            current_time = datetime.now()
            self.alert_cooldowns = {
                key: cooldown for key, cooldown in self.alert_cooldowns.items()
                if cooldown > current_time
            }
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    def _generate_monitoring_report(self):
        """Generate periodic monitoring report"""
        try:
            timestamp = datetime.now()
            report = {
                "timestamp": timestamp.isoformat(),
                "monitoring_status": {
                    "is_monitoring": self.is_monitoring,
                    "data_sources": self.data_sources,
                    "active_alerts_count": len(self.active_alerts),
                    "total_alerts_today": len([a for a in self.alert_history 
                                             if a.timestamp.date() == timestamp.date()])
                },
                "quality_summary": self._generate_quality_summary(),
                "alert_summary": self._generate_alert_summary(),
                "trends": self.quality_trends
            }
            
            # Save report
            filename = f"quality_monitoring_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.monitoring_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Monitoring report generated: {filepath}")
            
        except Exception as e:
            logger.error(f"Monitoring report generation failed: {e}")
    
    def _generate_quality_summary(self) -> Dict[str, Any]:
        """Generate quality summary"""
        try:
            summary = {
                "overall_quality_score": 0.0,
                "data_sources_healthy": 0,
                "data_sources_degraded": 0,
                "variables_monitored": 0
            }
            
            # Calculate overall quality score
            all_scores = []
            for data_source, status in self.data_sources.items():
                if status["quality_score"] > 0:
                    all_scores.append(status["quality_score"])
                
                if status["status"] == "healthy":
                    summary["data_sources_healthy"] += 1
                elif status["status"] == "degraded":
                    summary["data_sources_degraded"] += 1
            
            if all_scores:
                summary["overall_quality_score"] = np.mean(all_scores)
            
            # Count monitored variables
            summary["variables_monitored"] = len(self.quality_history)
            
            return summary
            
        except Exception as e:
            logger.error(f"Quality summary generation failed: {e}")
            return {}
    
    def _generate_alert_summary(self) -> Dict[str, Any]:
        """Generate alert summary"""
        try:
            summary = {
                "total_alerts": len(self.alert_history),
                "active_alerts": len(self.active_alerts),
                "alerts_by_severity": defaultdict(int),
                "alerts_by_type": defaultdict(int),
                "recent_alerts": []
            }
            
            # Count alerts by severity and type
            for alert in self.alert_history:
                summary["alerts_by_severity"][alert.severity] += 1
                summary["alerts_by_type"][alert.alert_type] += 1
            
            # Get recent alerts (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_alerts = [alert for alert in self.alert_history if alert.timestamp > cutoff_time]
            summary["recent_alerts"] = [alert.to_dict() for alert in recent_alerts[-10:]]  # Last 10
            
            return summary
            
        except Exception as e:
            logger.error(f"Alert summary generation failed: {e}")
            return {}
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        try:
            return {
                "is_monitoring": self.is_monitoring,
                "data_sources": self.data_sources,
                "active_alerts_count": len(self.active_alerts),
                "quality_history_size": {k: len(v) for k, v in self.quality_history.items()},
                "recent_alerts": [alert.to_dict() for alert in list(self.alert_history)[-5:]],
                "quality_trends": self.quality_trends
            }
            
        except Exception as e:
            logger.error(f"Status retrieval failed: {e}")
            return {"error": str(e)}
    
    def acknowledge_alert(self, alert_index: int) -> bool:
        """Acknowledge an alert"""
        try:
            if 0 <= alert_index < len(self.active_alerts):
                self.active_alerts[alert_index].acknowledged = True
                logger.info(f"Alert acknowledged: {self.active_alerts[alert_index].message}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Alert acknowledgment failed: {e}")
            return False
    
    def resolve_alert(self, alert_index: int) -> bool:
        """Resolve an alert"""
        try:
            if 0 <= alert_index < len(self.active_alerts):
                self.active_alerts[alert_index].resolved = True
                logger.info(f"Alert resolved: {self.active_alerts[alert_index].message}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Alert resolution failed: {e}")
            return False
    
    def export_monitoring_data(self, filepath: str):
        """Export monitoring data to file"""
        try:
            monitoring_data = {
                "quality_history": {k: list(v) for k, v in self.quality_history.items()},
                "active_alerts": [alert.to_dict() for alert in self.active_alerts],
                "alert_history": [alert.to_dict() for alert in self.alert_history],
                "data_sources": self.data_sources,
                "quality_trends": self.quality_trends,
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(monitoring_data, f, default=str, indent=2)
            
            logger.info(f"Monitoring data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Monitoring data export failed: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize quality monitor
    monitor = QualityMonitor()
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Keep running for demonstration
        time.sleep(60)
    finally:
        monitor.stop_monitoring() 