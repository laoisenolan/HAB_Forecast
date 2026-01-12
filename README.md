# HAB Forecast - Harmful Algal Bloom Prediction System

A machine learning-based system for predicting harmful algal blooms (HABs) in marine environments, specifically designed for the Seto Inland Sea region of Japan.

## Overview

HAB Forecast integrates environmental data from multiple sources (satellite imagery, ocean sensors, atmospheric data) to predict the occurrence and distribution of harmful algal blooms. The system uses advanced machine learning techniques including XGBoost, spatial-temporal modeling, and species-specific prediction models.

**Key Capabilities:**
- Multi-source data integration (Google Earth Engine, Copernicus Marine, ERDDAP)
- Species-specific HAB prediction for 12+ phytoplankton species
- Spatial-temporal forecasting with grid-based risk assessment
- Real-time data collection and model training pipelines
- Production-ready model deployment with uncertainty quantification

## Features

### Data Collection
- **25+ environmental variables** including SST, chlorophyll-a, nutrients, ocean currents, salinity, wind patterns
- **Multiple data sources** with automatic fallback and error handling
- **Historical data collection** spanning 2017-2024 (7+ years)
- **Caching system** for efficient data retrieval and processing

### Machine Learning Models
- **XGBoost-based predictors** with hyperparameter optimization
- **Species-specific models** for targeted bloom prediction
- **Spatial-temporal modeling** accounting for location and time dynamics
- **Ensemble methods** combining multiple model predictions
- **Transfer learning** for improved performance across species

### Analysis & Evaluation
- Feature importance analysis and SHAP interpretability
- Spatial distribution mapping and temporal pattern detection
- Model performance evaluation and stability analysis
- Dataset quality assessment and validation

## Installation

### Requirements
- Python 3.8+
- 8GB+ RAM recommended
- Internet connection for data collection

### Setup

```bash
# Clone the repository
git clone https://github.com/laoisenolan/HAB_Forecast.git
cd HAB_Forecast

# Install dependencies
pip install -r requirements.txt

# Configure authentication
python scripts/setup/setup_copernicus_auth.py
python scripts/setup/setup_environment.py
```

### Authentication

**Google Earth Engine:**
Place service account credentials in `config/hab-forecast-e78bb750e4e7.json`

**Copernicus Marine:**
Run setup script to configure credentials in `~/.copernicusmarine/`

## Quick Start

### Collect Environmental Data

```bash
# Collect recent data (2022-2024)
python scripts/runners/collect_recent_data.py

# Collect full historical data (2017-2024)
python scripts/runners/collect_historical_data_full.py
```

### Train Models

```bash
# Quick demo training (~30 seconds)
python scripts/runners/train_demo_quick.py

# Smart sampling approach (~15 minutes)
python scripts/runners/train_smart_sampling.py

# Full production training (~1-2 hours)
python scripts/runners/train_production_batch.py
```

### Run Analysis

```bash
# Analyze dataset coverage
python scripts/analysis/analyze_dataset_coverage.py

# Analyze bloom spatial patterns
python scripts/analysis/analyze_bloom_spatial_distribution.py

# Extract feature importance
python scripts/analysis/extract_feature_importance.py
```

## Project Structure

```
HAB_Forecast/
├── config/                 # Configuration files and credentials
├── src/                    # Core source code
│   ├── data_collection/    # Data collection modules
│   ├── data_processing/    # Data processing pipelines
│   ├── feature_engineering/# Feature generation
│   ├── model/              # ML model training and prediction
│   ├── evaluation/         # Model evaluation
│   ├── visualization/      # Plotting and dashboards
│   └── web/                # Web interface
├── scripts/                # Executable scripts
│   ├── analysis/           # Data analysis tools
│   ├── data_processing/    # Data preparation scripts
│   ├── runners/            # Main execution scripts
│   ├── setup/              # Environment setup
│   ├── testing/            # Validation utilities
│   └── training/           # Model training scripts
├── docs/                   # Documentation
│   ├── guides/             # User guides
│   ├── results/            # Analysis results
│   └── setup/              # Setup instructions
├── data/                   # Data storage
├── models/                 # Trained models
└── logs/                   # Training logs
```

## Usage

### Data Collection Pipeline

```python
# Using Python API
from src.data_collection import main

main.collect_cached(
    start_date="2023-01-01",
    end_date="2023-12-31",
    variables=["sea_surface_temperature", "chlorophyll_a", "salinity"]
)
```

### Model Training

```python
# Using Python API
from src.model import spatial_temporal_trainer

trainer = spatial_temporal_trainer.SpatialTemporalTrainer()
trainer.train(
    data_path="data/processed/training_data.parquet",
    model_type="xgboost",
    species="Karenia_mikimotoi"
)
```

### Making Predictions

```python
# Using trained models
from src.model import predictor

model = predictor.load_model("models/species_specific/Karenia_mikimotoi.pkl")
predictions = model.predict(environmental_data)
```

## Configuration

Edit configuration files in `config/`:
- `data_sources.yaml` - Data source settings and endpoints
- `features.yaml` - Feature engineering parameters
- `model_params.yaml` - ML model hyperparameters
- `processing.yaml` - Data processing configuration

## Documentation

- **[Quick Start Guide](QUICK_START.md)** - Get started quickly
- **[Deployment Options](docs/guides/DEPLOYMENT_OPTIONS.md)** - Training strategies
- **[Pipeline Guide](docs/guides/HAB_FORECASTING_PIPELINE_GUIDE.md)** - Complete workflow
- **[Setup Instructions](docs/setup/SETUP_INSTRUCTIONS.md)** - Detailed configuration
- **[Project Updates](docs/PROJECT_UPDATES_2025.md)** - Development history

## Performance

**Model Accuracy:**
- AUC scores: 0.90-0.99 across species
- F1 scores: 0.70-0.95 for balanced datasets
- Spatial resolution: 0.05° (~5km grid cells)
- Temporal resolution: Daily predictions

**System Performance:**
- Data collection: ~10 variables per minute
- Model training: 30 seconds (demo) to 2 hours (production)
- Prediction: Real-time (<1 second per location)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## Testing

```bash
# Run validation tests
python scripts/testing/validate_training_features.py

# Check data quality
python scripts/analysis/analyze_dataset_quality.py
```

## Troubleshooting

**Common Issues:**

1. **Authentication errors**: Verify credentials in `config/` directory
2. **Memory errors**: Reduce dataset size or use batch processing
3. **Missing data**: Check data source availability and date ranges
4. **Import errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`)

**For detailed troubleshooting**, see [Setup Guide](docs/SETUP_GUIDE.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{hab_forecast_2026,
  title={HAB Forecast: Machine Learning System for Harmful Algal Bloom Prediction},
  author={Laoise Nolan},
  year={2026},
  url={https://github.com/laoisenolan/HAB_Forecast}
}
```

## Support

- **Issues**: Report bugs via [GitHub Issues](https://github.com/laoisenolan/HAB_Forecast/issues)
- **Documentation**: See `docs/` directory
- **Updates**: Check [Project Updates](docs/PROJECT_UPDATES_2025.md)

## Acknowledgments

- Google Earth Engine for satellite data access
- Copernicus Marine Service for ocean data
- ERDDAP data providers
- Open-source ML community (scikit-learn, XGBoost, PyTorch)

---

**Status**: Production Ready | **Version**: 4.0 | **Last Updated**: January 2026
