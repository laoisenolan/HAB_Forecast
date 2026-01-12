#!/usr/bin/env python3
"""
Environment Setup Script for HAB Forecasting System
Seto Inland Sea Project

This script automates the setup of the development environment,
including package installation, directory creation, and configuration.
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentSetup:
    def __init__(self, project_root: str = "hab_forecast"):
        self.project_root = Path(project_root)
        self.requirements_file = Path("requirements.txt")
        
    def check_python_version(self):
        """Check if Python version is compatible (3.8+)"""
        logger.info("Checking Python version...")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            raise RuntimeError(f"Python 3.8+ required, found {version.major}.{version.minor}")
        logger.info(f"Python version {version.major}.{version.minor}.{version.micro} is compatible")
        
    def create_directories(self):
        """Create project directory structure"""
        logger.info("Creating project directory structure...")
        
        directories = [
            "data/raw",
            "data/processed", 
            "data/climatology",
            "data/labels",
            "src/data_collection",
            "src/data_processing",
            "src/feature_engineering",
            "src/model",
            "src/evaluation",
            "src/visualization",
            "config",
            "notebooks",
            "tests",
            "outputs",
            "docs",
            "logs",
            "cache"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
            
    def install_requirements(self):
        """Install required packages"""
        logger.info("Installing required packages...")
        
        if not self.requirements_file.exists():
            logger.error(f"Requirements file not found: {self.requirements_file}")
            return False
            
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
            ], check=True)
            logger.info("Successfully installed required packages")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install packages: {e}")
            return False
            
    def create_init_files(self):
        """Create __init__.py files for Python packages"""
        logger.info("Creating __init__.py files...")
        
        init_dirs = [
            "src",
            "src/data_collection",
            "src/data_processing", 
            "src/feature_engineering",
            "src/model",
            "src/evaluation",
            "src/visualization",
            "tests"
        ]
        
        for directory in init_dirs:
            init_file = self.project_root / directory / "__init__.py"
            init_file.touch(exist_ok=True)
            logger.info(f"Created: {init_file}")
            
    def validate_config_files(self):
        """Validate configuration files"""
        logger.info("Validating configuration files...")
        
        config_files = [
            "config/data_sources.yaml",
            "config/model_params.yaml", 
            "config/processing.yaml"
        ]
        
        for config_file in config_files:
            file_path = self.project_root / config_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        yaml.safe_load(f)
                    logger.info(f"Validated: {config_file}")
                except UnicodeDecodeError as e:
                    # Try with different encodings if UTF-8 fails
                    for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                yaml.safe_load(f)
                            logger.info(f"Validated: {config_file} with {encoding} encoding")
                            break
                        except (UnicodeDecodeError, yaml.YAMLError):
                            continue
                    else:
                        logger.error(f"Could not decode {config_file} with any supported encoding: {e}")
                except yaml.YAMLError as e:
                    logger.error(f"Invalid YAML in {config_file}: {e}")
            else:
                logger.warning(f"Config file not found: {config_file}")
                
    def create_env_template(self):
        """Create environment variables template"""
        logger.info("Creating environment variables template...")
        
        env_template = """# Environment Variables for HAB Forecasting System
# Copy this file to .env and fill in your credentials

# Copernicus Marine credentials
COPERNICUS_USERNAME=your_username_here
COPERNICUS_PASSWORD=your_password_here

# Google Earth Engine credentials (optional)
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json

# Database settings (if using)
DATABASE_URL=postgresql://user:password@localhost:5432/hab_forecast

# API keys (if needed)
WEATHER_API_KEY=your_api_key_here

# Logging
LOG_LEVEL=INFO
"""
        
        env_file = self.project_root / ".env.template"
        with open(env_file, 'w') as f:
            f.write(env_template)
        logger.info(f"Created: {env_file}")
        
    def create_readme(self):
        """Create project README"""
        logger.info("Creating project README...")
        
        readme_content = """# HAB Forecasting System for Seto Inland Sea

## Overview
This project implements a supervised binary classification model using XGBoost to forecast Harmful Algal Blooms (HABs) in the Seto Inland Sea, Japan, with up to 7-day lead time.

## Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv hab_env
source hab_env/bin/activate  # On Windows: hab_env\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
1. Copy `.env.template` to `.env`
2. Fill in your API credentials
3. Configure data sources in `config/data_sources.yaml`

### 3. Data Collection
```bash
python src/data_collection/download_copernicus.py
python src/data_collection/download_erddap.py
```

### 4. Model Training
```bash
python src/model/train_xgboost.py
```

### 5. Forecasting
```bash
python src/model/predict.py
```

## Project Structure
```
hab_forecast/
├── data/               # Data storage
├── src/               # Source code
├── config/            # Configuration files
├── notebooks/         # Jupyter notebooks
├── tests/             # Unit tests
├── outputs/           # Model outputs
└── docs/              # Documentation
```

## Data Sources
- **Copernicus Marine**: Physics and ocean color data
- **ERDDAP**: Atmospheric data via PacIOOS
- **Google Earth Engine**: OC-CCI climatology
- **NOAA CoastWatch**: Gap-filled fallback data

## Model Details
- **Algorithm**: XGBoost (Gradient Boosted Decision Trees)
- **Features**: SST, wind speed, precipitation, MLD, Kd490, chlorophyll-a
- **Output**: Probability maps and risk categorization
- **Interpretability**: SHAP analysis

## Performance Targets
- Accuracy: >80%
- Lead Time: 7 days
- False Alarm Rate: <20%
- Detection Rate: >85%

## Documentation
See `docs/` directory for detailed documentation.

## License
[Add your license here]
"""
        
        readme_file = self.project_root / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        logger.info(f"Created: {readme_file}")
        
    def run_setup(self):
        """Run complete environment setup"""
        logger.info("Starting HAB Forecasting System environment setup...")
        
        try:
            self.check_python_version()
            self.create_directories()
            self.install_requirements()
            self.create_init_files()
            self.validate_config_files()
            self.create_env_template()
            self.create_readme()
            
            logger.info("Environment setup completed successfully!")
            logger.info(f"Project root: {self.project_root.absolute()}")
            logger.info("Next steps:")
            logger.info("1. Copy .env.template to .env and fill in credentials")
            logger.info("2. Configure data sources in config/data_sources.yaml")
            logger.info("3. Start with data collection scripts")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False
            
        return True

def main():
    """Main function"""
    setup = EnvironmentSetup()
    success = setup.run_setup()
    
    if success:
        print("\n✅ Environment setup completed successfully!")
        print("📁 Project created in: hab_forecast/")
        print("📋 Next steps:")
        print("   1. Copy .env.template to .env and add your credentials")
        print("   2. Review configuration files in config/")
        print("   3. Start with Phase 2: Data Collection")
    else:
        print("\n❌ Environment setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 