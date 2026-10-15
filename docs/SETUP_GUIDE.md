# HAB Forecasting System - Setup Guide

## Quick Setup for Data Collection Testing

### 1. Prerequisites

You need:
- **Python 3.8+** (already have)
- **Copernicus Marine credentials** (already have)
- **Google Earth Engine credentials** (already have)
- **OpenWeather API key** (already have)

### 2. Environment Setup

You've already created the `.env` file with your credentials.

### 3. Install Required Packages

Run this command to install all required packages:

```bash
pip install -r requirements.txt
```

**Key packages needed:**
- `copernicusmarine` - For Copernicus Marine data
- `erddapy` - For ERDDAP data (atmospheric)
- `earthengine-api` - For Google Earth Engine
- `xarray` - For data handling
- `numpy`, `pandas` - For data processing
- `rasterio`, `geopandas` - For geospatial data

### 4. Google Earth Engine Authentication

Since you have the GEE credentials file, authenticate with:

```bash
earthengine authenticate
```

This will use your service account credentials.

### 5. Test the System

Run the requirements check:

```bash
python requirements_check.py
```

This will verify everything is set up correctly.

### 6. Test Data Collection

Once everything is set up, test the data collection:

```bash
# Show available data sources
python src/data_collection/main.py info

# Test weekly data collection (small test)
python src/data_collection/main.py collect-weekly --log-level DEBUG
```

## What Each Data Source Provides

### 1. Copernicus Marine
- **Physics Data**: SST, MLD, currents (daily, 1/12° resolution)
- **Ocean Color**: Chlorophyll-a, Kd490 (daily, 4km resolution)
- **Coverage**: Seto Inland Sea region (34-35°N, 132-135°E)

### 2. ERDDAP (PacIOOS)
- **Atmospheric Data**: Wind speed, precipitation (3-hourly)
- **Fallback**: NOAA CoastWatch gap-filled data
- **Processing**: Automatic unit conversions and weekly aggregation

### 3. Google Earth Engine
- **Climatology**: OC-CCI v6 seasonal statistics (2002-2024)
- **Output**: Seasonal means and 75th percentiles
- **Format**: GeoTIFF files for Seto Inland Sea

## Expected Output

When you run the data collection, you'll get:

1. **Raw Data Files**: NetCDF files in `data/raw/`
2. **Quality Reports**: JSON reports in `data/qc_reports/`
3. **Collection Reports**: Summary reports with statistics
4. **Logs**: Detailed logs in `logs/`

## Troubleshooting

### Common Issues:

1. **Python not found**: Make sure Python is in your PATH
2. **Missing packages**: Run `pip install -r requirements.txt`
3. **GEE authentication**: Run `earthengine authenticate`
4. **Copernicus credentials**: Check your `.env` file
5. **Network issues**: Some data sources require internet access

### Test Commands:

```bash
# Check Python
python --version

# Check packages
python -c "import copernicusmarine, erddapy, earthengine; print('All packages OK')"

# Test configuration
python -c "from src.utils.config_loader import ConfigLoader; print('Config OK')"

# Test data collection
python src/data_collection/main.py info
```

## Next Steps After Testing

Once the data collection is working:

1. **Phase 3**: Data Processing and Feature Engineering
2. **Phase 4**: Model Development
3. **Phase 5**: Evaluation and Validation

## File Structure

```
hab_forecast/
├── .env                    # Your credentials (already created)
├── requirements.txt        # Python packages
├── config/                 # Configuration files
├── src/data_collection/    # Data collection modules
├── data/                   # Data storage
│   ├── raw/               # Raw downloaded data
│   ├── processed/         # Processed data
│   ├── climatology/       # GEE climatology
│   └── qc_reports/        # Quality control reports
└── logs/                  # System logs
```

## Support

If you encounter issues:

1. Check the logs in `logs/` directory
2. Run `python requirements_check.py` to diagnose issues
3. Verify your credentials in `.env` file
4. Ensure all packages are installed correctly 
