# HAB Forecasting System - Setup Instructions

## 📋 Quick Start Guide for Team Members

### 🎯 What's New in This Version
- ✅ **Task 4.0 ML Breakthrough**: Species-specific XGBoost, Deep Learning (LSTM/CNN), and Ensemble models completed
- ✅ **Clean Features Only**: Data leakage fixed - using only environmental predictors for realistic HAB forecasting  
- ✅ **Production-Ready**: Real-world tested ML pipeline with 21,880→2,735 processed records
- ✅ **Comprehensive Framework**: Multiple ML architectures ready for operational deployment

---

## 🚀 Installation Guide

### Option 1: Minimal Installation (Recommended for most users)
```bash
# Clone the repository
git clone <your-repo-url>
cd HAB_Forecast

# Install core dependencies only
pip install -r requirements-core.txt
```

### Option 2: Full Installation (For advanced ML development)
```bash
# Install all dependencies including deep learning
pip install -r requirements.txt
```

### Option 3: Conda Environment (Recommended for data scientists)
```bash
# Create conda environment
conda create -n hab_forecast python=3.9
conda activate hab_forecast

# Install from requirements
pip install -r requirements-core.txt

# Optional: Add PyTorch for deep learning
conda install pytorch torchvision torchaudio -c pytorch
```

---

## 🔑 Authentication Setup

### 1. Google Earth Engine (Required)
```bash
# The service account credentials are already configured in:
# config/hab-forecast-e78bb750e4e7.json

# Set environment variable (if not already set)
export GOOGLE_APPLICATION_CREDENTIALS="config/hab-forecast-e78bb750e4e7.json"
```

### 2. Copernicus Marine (Required)
```bash
# Set up Copernicus Marine credentials
# Run this once to store credentials:
python -c "
import copernicusmarine
# Follow the prompts to enter your credentials
# Username: lnolan1 (or your username)
# Password: [your password]
"
```

---

## 🧪 Test the Installation

### Quick System Test
```bash
# Test core data collection (should work immediately)
python -c "
import pandas as pd
import xgboost as xgb
print('✅ Core libraries working!')
print(f'Pandas: {pd.__version__}')
print(f'XGBoost: {xgb.__version__}')
"
```

### Test Machine Learning Models
```bash
# Test species-specific XGBoost models (Task 4.1)
python species_trainer_standalone.py

# Test ensemble models (Task 4.3)  
python test_ensemble_standalone.py
```

### Test Deep Learning (Optional - requires PyTorch)
```bash
# Only if you installed PyTorch
python test_deep_learning_standalone.py
```

---

## 📊 Available Datasets & Features

### Training Data Available
- **Real HAB Dataset**: 21,880 records with HAB events
- **Environmental Variables**: 13 clean predictors (no data leakage)
- **Species Coverage**: 10+ HAB species from Seto Inland Sea
- **Temporal Coverage**: 2023-2024 with seasonal patterns

### Clean Environmental Features (No Data Leakage!)
```python
CLEAN_FEATURES = [
    'chlorophyll_a',         # Primary productivity indicator
    'dissolved_oxygen',      # Water quality measure  
    'mixed_layer_depth',     # Physical oceanography
    'nitrate',              # Nutrient availability
    'ocean_salinity',       # Water mass properties
    'ocean_temperature',    # Thermal conditions
    'phosphate',            # Nutrient limitation
    'precipitation',        # Atmospheric forcing
    'sea_surface_temperature', # Surface conditions
    'silicate',             # Diatom growth factor
    'turbidity_kd490',      # Water clarity
    'wind_u',               # Wind forcing (east-west)
    'wind_v'                # Wind forcing (north-south)
]
```

---

## 🧠 Machine Learning Models Ready

### 1. Species-Specific XGBoost (Task 4.1) ✅
```bash
# Train models for specific HAB species
python species_trainer_standalone.py

# Features:
# - Balanced training data (positive/negative samples)
# - Hyperparameter optimization
# - Cross-validation
# - Feature importance analysis
# - Realistic performance (F1 ~0.6-0.65)
```

### 2. Deep Learning Models (Task 4.2) ✅ 
```bash
# Requires PyTorch installation
pip install torch torchvision torchaudio

# Run deep learning models
python test_deep_learning_standalone.py

# Features:
# - LSTM for temporal patterns
# - CNN for spatial patterns  
# - Attention mechanisms
# - Sequence processing for time series
```

### 3. Ensemble Models (Task 4.3) ✅
```bash
# Multiple ensemble strategies
python test_ensemble_standalone.py

# Features:
# - Soft/Hard voting classifiers
# - Performance-weighted ensembles
# - Stacked ensembles with meta-learners
# - 5 diverse base models (XGBoost, RF, GB, ET)
```

---

## 📁 Key File Structure

```
HAB_Forecast/
├── 📊 Machine Learning (NEW!)
│   ├── src/model/species_specific_trainer.py    # Species XGBoost models
│   ├── src/model/deep_learning_hab_trainer.py   # PyTorch LSTM/CNN framework  
│   ├── src/model/ensemble_hab_trainer.py        # Advanced ensemble methods
│   ├── species_trainer_standalone.py            # Test species models
│   ├── test_ensemble_standalone.py              # Test ensemble methods
│   └── cache/                                   # Trained model storage
│
├── 🗂️ Data Collection (Completed)
│   ├── src/data_collection/                     # Multi-source data APIs
│   └── data/real_collection/                    # Real HAB training data
│
├── 🔧 Feature Engineering (Completed) 
│   ├── src/feature_engineering/                 # 1,168 features from 17 variables
│   └── Advanced modules for all feature types
│
├── ⚙️ Configuration  
│   ├── config/hab-forecast-e78bb750e4e7.json   # GEE credentials
│   ├── config/data_sources.yaml                # Data source definitions
│   ├── requirements.txt                        # Full dependencies
│   └── requirements-core.txt                   # Core dependencies
│
└── 📚 Documentation
    ├── PROJECT_UPDATES_2025.md                 # Complete project status
    ├── HAB_FORECASTING_TASKS.md                # Implementation tasks  
    └── SETUP_INSTRUCTIONS.md                   # This file
```

---

## 🎯 What You Can Do Right Now

### For Data Scientists:
1. **Train Species Models**: Run `python species_trainer_standalone.py` to see XGBoost in action
2. **Explore Ensemble Methods**: Test `python test_ensemble_standalone.py` for advanced ML
3. **Analyze Feature Importance**: Check which environmental variables drive HAB predictions
4. **Experiment with Deep Learning**: Install PyTorch and try neural networks

### For Developers:
1. **Study the ML Pipeline**: Review how 21,880 records → 2,735 processed samples
2. **Understand Data Processing**: See long-format → wide-format conversion logic  
3. **Explore Model Architectures**: Check PyTorch LSTM/CNN implementations
4. **Build New Features**: Add to the ensemble or create new model types

### For Marine Scientists:
1. **Validate Results**: Check if F1 ~0.6-0.65 performance makes sense for HAB prediction
2. **Review Clean Features**: Ensure environmental variables are scientifically sound
3. **Species Analysis**: Examine species-specific model differences  
4. **Environmental Drivers**: Study which variables are most important for different species

---

## ⚠️  Important Notes

### Data Leakage Fixed!
- **Previous Issue**: Models had perfect scores (F1=1.0) using HAB indicators in training
- **Resolution**: Now using only environmental predictors for realistic forecasting
- **Result**: More realistic performance (F1 ~0.6-0.65) suitable for operational use

### Deep Learning Requirements
- PyTorch models require separate installation: `pip install torch`
- GPU acceleration available if CUDA installed
- CPU-only mode works fine for testing and small datasets

### System Requirements
- **Python**: 3.8+ (tested with 3.9, 3.10, 3.11)
- **Memory**: 8GB+ RAM recommended for full ML pipeline
- **Storage**: ~2GB for datasets and trained models
- **Network**: Internet connection for data collection APIs

---

## 🆘 Troubleshooting

### Common Issues:

**1. "ModuleNotFoundError" for deep learning**
```bash
# Install PyTorch for deep learning models
pip install torch torchvision torchaudio
```

**2. "GEE authentication error"**  
```bash
# Check environment variable
export GOOGLE_APPLICATION_CREDENTIALS="config/hab-forecast-e78bb750e4e7.json"
```

**3. "Copernicus Marine login required"**
```bash
# Set up credentials once
python -c "import copernicusmarine; copernicusmarine.login()"
```

**4. "Data file not found"**
```bash
# Make sure you're in the project root directory
cd HAB_Forecast
python species_trainer_standalone.py
```

---

## 🎉 Ready to Contribute!

The system is now ready for:
- ✅ Species-specific HAB prediction models  
- ✅ Real-time environmental data processing
- ✅ Advanced ensemble machine learning
- ✅ Deep learning temporal pattern analysis
- 🔄 Additional ML model development (Tasks 4.4-4.12)
- 🔄 Production deployment (Task 5.0)
- 🔄 Web interface development (Task 6.0)

### Next Development Priorities:
1. **Task 4.4-4.12**: Remaining ML subtasks (transfer learning, uncertainty quantification, etc.)
2. **Task 5.0**: Production deployment and operational systems
3. **Task 6.0**: Web interface and visualization dashboard

**Happy HAB Forecasting! 🌊🤖**