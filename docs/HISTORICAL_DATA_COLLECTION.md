# Historical Data Collection for Seto Inland Sea

This document describes the historical data collection functionality for the HAB Forecasting System, specifically designed for the Seto Inland Sea region in Japan.

## Overview

The historical data collection system allows you to download and store large amounts of historical oceanographic and atmospheric data for the Seto Inland Sea region. This data is essential for:

- Training machine learning models
- Creating climatological baselines
- Understanding long-term trends in harmful algal blooms
- Validating forecast models

## Geographic Coverage

The system is configured for the Seto Inland Sea region with the following boundaries:
- **North**: 35.0°N
- **South**: 34.0°N  
- **East**: 135.0°E
- **West**: 132.0°E
- **Resolution**: 2.0 km

## Available Data Sources

### 1. Copernicus Marine Service
- **Physics Data**: Sea surface temperature, mixed layer depth, ocean currents
- **Ocean Color Data**: Chlorophyll-a concentration, nitrate levels
- **Temporal Coverage**: 2002-present
- **Spatial Resolution**: 1/12° (physics), 0.25° (ocean color)

### 2. ERDDAP Data Sources
- **Atmospheric Data**: Wind components, precipitation
- **NOAA CoastWatch**: Chlorophyll-a, light attenuation
- **Temporal Resolution**: 3-hourly to daily

### 3. Google Earth Engine
- **OC-CCI Climatology**: Long-term chlorophyll-a averages
- **Seasonal Analysis**: DJF, MAM, JJA, SON periods

## Predefined Collection Periods

The system includes three predefined collection periods:

1. **Full Historical** (2002-2024)
   - Complete dataset from 2002 to 2024
   - Maximum historical coverage
   - Requires significant storage space

2. **Recent Decade** (2014-2024)
   - Last 10 years of data
   - Good balance of coverage and storage
   - Recommended for most applications

3. **Last Five Years** (2019-2024)
   - Most recent 5 years
   - Minimal storage requirements
   - Suitable for recent trend analysis

## Usage

### Command Line Interface

#### 1. Check Historical Collection Status
```bash
python src/data_collection/main.py historical-status
```

#### 2. Collect Historical Data Using Predefined Period
```bash
# Full historical dataset (2002-2024)
python src/data_collection/main.py collect-historical --period full_historical

# Recent decade (2014-2024)
python src/data_collection/main.py collect-historical --period recent_decade

# Last five years (2019-2024)
python src/data_collection/main.py collect-historical --period last_five_years
```

#### 3. Collect Custom Historical Period
```bash
python src/data_collection/main.py collect-historical \
    --start-date 2010-01-01 \
    --end-date 2020-12-31
```

#### 4. Include Climatology Computation
```bash
python src/data_collection/main.py collect-historical \
    --period recent_decade \
    --include-climatology
```

### Programmatic Usage

```python
from src.data_collection.historical_collector import HistoricalDataCollector

# Initialize collector
collector = HistoricalDataCollector()

# Run historical collection
results = collector.run_historical_collection(
    period_name="recent_decade",
    include_climatology=True
)

if results["success"]:
    print(f"Collected {len(results['merged_datasets'])} datasets")
    print(f"Saved {len(results['saved_files'])} files")
else:
    print(f"Collection failed: {results.get('error')}")
```

## Data Processing

### Chunking Strategy
Large historical datasets are downloaded in manageable chunks:
- **Default chunk size**: 30 days
- **Concurrent downloads**: 3 maximum
- **Retry attempts**: 3 per failed chunk

### Data Merging
After downloading, the system:
1. Concatenates time chunks along the time dimension
2. Sorts data chronologically
3. Removes duplicate timestamps
4. Adds metadata and attributes

### Storage Format
- **Format**: NetCDF4
- **Compression**: LZW compression enabled
- **Metadata**: Collection period, spatial bounds, data source
- **Directory Structure**: `data/historical/{data_type}/`

## Configuration

### Historical Collection Settings
```yaml
historical:
  collection_periods:
    - name: "full_historical"
      start_date: "2002-01-01"
      end_date: "2024-12-31"
      description: "Complete historical dataset from 2002 to 2024"
  
  chunk_settings:
    time_chunk_days: 30
    max_concurrent_downloads: 3
    retry_attempts: 3
  
  storage:
    compress_historical: true
    historical_data_format: "netcdf4"
    metadata_included: true
```

### Customizing Collection Periods
Edit `config/data_sources.yaml` to add custom collection periods:

```yaml
historical:
  collection_periods:
    - name: "custom_period"
      start_date: "2015-01-01"
      end_date: "2023-12-31"
      description: "Custom 9-year period"
```

## Storage Requirements

### Estimated Storage Sizes
- **Full Historical** (2002-2024): ~50-100 GB
- **Recent Decade** (2014-2024): ~20-40 GB
- **Last Five Years** (2019-2024): ~10-20 GB

### Storage Location
Historical data is stored in:
```
hab_forecast/data/historical/
├── physics/
├── ocean_color/
├── atmospheric/
└── climatology/
```

## Monitoring and Reporting

### Collection Reports
Each historical collection generates a comprehensive report including:
- Collection period and metadata
- Data source information
- File locations and sizes
- Validation statistics
- Chunking statistics

### Progress Tracking
The system provides real-time progress updates:
- Chunk download progress
- Success/failure rates
- Estimated completion time
- Storage usage

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Ensure Copernicus Marine credentials are configured
   - Check ERDDAP server accessibility

2. **Storage Space**
   - Monitor available disk space
   - Use smaller collection periods for testing

3. **Network Issues**
   - Check internet connectivity
   - Verify firewall settings
   - Consider using VPN if needed

4. **Memory Issues**
   - Reduce concurrent downloads
   - Use smaller chunk sizes
   - Close other applications

### Error Recovery
- Failed chunks are logged and reported
- System continues with remaining chunks
- Partial results are saved
- Retry mechanisms handle temporary failures

## Best Practices

### For Large Collections
1. Start with smaller periods to test
2. Monitor system resources
3. Use off-peak hours for large downloads
4. Ensure stable internet connection

### For Production Use
1. Schedule collections during low-usage periods
2. Set up automated monitoring
3. Implement backup strategies
4. Regular validation of collected data

### For Development
1. Use test periods for development
2. Validate data quality after collection
3. Document any custom configurations
4. Share successful configurations with team

## Integration with HAB Forecasting

The collected historical data feeds into:
- **Model Training**: Machine learning model development
- **Climatology**: Baseline environmental conditions
- **Validation**: Model performance assessment
- **Research**: Long-term trend analysis

## Support

For issues or questions:
1. Check the logs in `hab_forecast/logs/`
2. Review collection reports in `hab_forecast/data/`
3. Test with smaller periods first
4. Verify configuration settings

## Future Enhancements

Planned improvements include:
- Parallel processing for faster downloads
- Incremental updates for new data
- Quality control automation
- Integration with cloud storage
- Real-time monitoring dashboard 