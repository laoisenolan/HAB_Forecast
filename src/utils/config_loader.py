"""
Configuration Loader for HAB Forecasting System

This module provides utilities for loading and managing configuration files
throughout the project.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Configuration loader for HAB forecasting system"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._configs = {}
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a configuration file by name"""
        if config_name in self._configs:
            return self._configs[config_name]
            
        config_file = self.config_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self._configs[config_name] = config
            logger.info(f"Loaded configuration: {config_name}")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_file}: {e}")
        except UnicodeDecodeError as e:
            # Try with different encodings if UTF-8 fails
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(config_file, 'r', encoding=encoding) as f:
                        config = yaml.safe_load(f)
                    self._configs[config_name] = config
                    logger.info(f"Loaded configuration: {config_name} with {encoding} encoding")
                    return config
                except (UnicodeDecodeError, yaml.YAMLError):
                    continue
            raise ValueError(f"Could not decode {config_file} with any supported encoding: {e}")
            
    def get_data_sources_config(self) -> Dict[str, Any]:
        """Get data sources configuration"""
        return self.load_config("data_sources")
        
    def get_model_params_config(self) -> Dict[str, Any]:
        """Get model parameters configuration"""
        return self.load_config("model_params")
        
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration"""
        return self.load_config("processing")
        
    def get_features_config(self) -> Dict[str, Any]:
        """Get features configuration"""
        return self.load_config("features")
            
    def get_model_params_config(self) -> Dict[str, Any]:
        """Get model parameters configuration"""
        return self.load_config("model_params")
            
    def get_spatial_bounds(self) -> Dict[str, float]:
        """Get spatial boundaries for Seto Inland Sea"""
        config = self.get_data_sources_config()
        return config.get("spatial_bounds", {})
        
    def get_temporal_settings(self) -> Dict[str, Any]:
        """Get temporal settings"""
        config = self.get_data_sources_config()
        return config.get("temporal", {})
        
    def get_xgboost_params(self) -> Dict[str, Any]:
        """Get XGBoost hyperparameters"""
        config = self.get_model_params_config()
        return config.get("xgboost_params", {})
        
    def get_grid_settings(self) -> Dict[str, Any]:
        """Get grid and spatial settings"""
        config = self.get_processing_config()
        return config.get("grid", {})
        
    def validate_configs(self) -> bool:
        """Validate all configuration files"""
        try:
            # Load all configs
            self.get_data_sources_config()
            self.get_model_params_config()
            self.get_processing_config()
            
            # Validate spatial bounds
            bounds = self.get_spatial_bounds()
            required_bounds = ["north", "south", "east", "west"]
            for bound in required_bounds:
                if bound not in bounds:
                    raise ValueError(f"Missing spatial bound: {bound}")
                    
            # Validate temporal settings
            temporal = self.get_temporal_settings()
            if "climatology_start" not in temporal or "climatology_end" not in temporal:
                raise ValueError("Missing climatology temporal settings")
                
            logger.info("All configuration files validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

def load_env_vars() -> Dict[str, str]:
    """Load environment variables from .env file"""
    env_vars = {}
    
    # Try to load from .env file
    env_file = Path(".env")
    if env_file.exists():
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key] = value
            logger.info("Loaded environment variables from .env file")
        except UnicodeDecodeError as e:
            # Try with different encodings if UTF-8 fails
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(env_file, 'r', encoding=encoding) as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                env_vars[key] = value
                    logger.info(f"Loaded environment variables from .env file with {encoding} encoding")
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            else:
                logger.warning(f"Failed to load .env file with any supported encoding: {e}")
        except Exception as e:
            logger.warning(f"Failed to load .env file: {e}")
    
    # Also check system environment variables
    important_vars = [
        'COPERNICUS_USERNAME',
        'COPERNICUS_PASSWORD',
        'GOOGLE_APPLICATION_CREDENTIALS'
    ]
    
    for var in important_vars:
        if var not in env_vars:
            env_vars[var] = os.getenv(var, '')
            
    return env_vars

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent.parent

def get_data_dir() -> Path:
    """Get the data directory"""
    return get_project_root() / "data"

def get_outputs_dir() -> Path:
    """Get the outputs directory"""
    return get_project_root() / "outputs"

def get_logs_dir() -> Path:
    """Get the logs directory"""
    return get_project_root() / "logs" 