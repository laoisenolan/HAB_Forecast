"""
CSV Data Loader for HAB Forecasting System

This module provides utilities for loading and working with CSV-formatted
HAB forecasting data, making data analysis much simpler.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json

logger = logging.getLogger(__name__)

class CSVDataLoader:
    """
    Loads and processes CSV-formatted HAB forecasting data.
    
    Features:
    - Easy loading of historical CSV datasets
    - Variable filtering and selection
    - Temporal and spatial filtering
    - Data quality assessment
    - Simple data analysis functions
    """
    
    def __init__(self, data_dir: str = "data/csv"):
        """Initialize CSV data loader"""
        self.data_dir = Path(data_dir)
        logger.info(f"CSV data loader initialized for: {self.data_dir}")
    
    def load_historical_data(self, 
                           period_name: Optional[str] = None,
                           variables: Optional[List[str]] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           data_sources: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load historical HAB data from CSV files
        
        Args:
            period_name: Specific period name (e.g., 'last_five_years')
            variables: List of variable names to include
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            data_sources: List of data sources to include
            
        Returns:
            Filtered pandas DataFrame
        """
        logger.info("Loading historical HAB data from CSV files...")
        
        try:
            # Find available CSV files
            if period_name:
                csv_files = list(self.data_dir.glob(f"*{period_name}*.csv"))
            else:
                csv_files = list(self.data_dir.glob("combined_historical_*.csv"))
            
            if not csv_files:
                logger.warning("No historical CSV files found")
                return pd.DataFrame()
            
            # Load the most recent file
            latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
            logger.info(f"Loading data from: {latest_file}")
            
            df = pd.read_csv(latest_file)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Apply filters
            if variables:
                df = df[df['variable_name'].isin(variables)]
                logger.info(f"Filtered to variables: {variables}")
            
            if start_date:
                df = df[df['timestamp'] >= start_date]
                logger.info(f"Filtered to start date: {start_date}")
            
            if end_date:
                df = df[df['timestamp'] <= end_date]
                logger.info(f"Filtered to end date: {end_date}")
            
            if data_sources:
                df = df[df['data_source'].isin(data_sources)]
                logger.info(f"Filtered to data sources: {data_sources}")
            
            logger.info(f"Loaded {len(df)} records with {df['variable_name'].nunique()} variables")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            return pd.DataFrame()
    
    def get_available_variables(self) -> List[str]:
        """Get list of available HAB variables"""
        try:
            df = self.load_historical_data()
            if not df.empty:
                variables = sorted(df['variable_name'].unique().tolist())
                logger.info(f"Available variables: {len(variables)}")
                return variables
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to get available variables: {e}")
            return []
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available data"""
        try:
            df = self.load_historical_data()
            
            if df.empty:
                return {"error": "No data available"}
            
            summary = {
                "total_records": len(df),
                "variables": {
                    "count": df['variable_name'].nunique(),
                    "list": sorted(df['variable_name'].unique().tolist())
                },
                "data_sources": {
                    "count": df['data_source'].nunique(),
                    "list": sorted(df['data_source'].unique().tolist())
                },
                "temporal_coverage": {
                    "start": df['timestamp'].min().isoformat(),
                    "end": df['timestamp'].max().isoformat(),
                    "duration_days": (df['timestamp'].max() - df['timestamp'].min()).days
                },
                "spatial_coverage": {
                    "lat_range": [df['latitude'].min(), df['latitude'].max()],
                    "lon_range": [df['longitude'].min(), df['longitude'].max()],
                    "unique_locations": len(df[['latitude', 'longitude']].drop_duplicates())
                },
                "data_quality": {
                    "good_quality": (df['quality_flag'] == 'good').sum(),
                    "total_records": len(df),
                    "quality_percentage": (df['quality_flag'] == 'good').mean() * 100
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate data summary: {e}")
            return {"error": str(e)}
    
    def get_variable_time_series(self, 
                                variable_name: str,
                                aggregation: str = 'mean') -> pd.DataFrame:
        """
        Get time series for a specific variable
        
        Args:
            variable_name: Name of the variable
            aggregation: Aggregation method ('mean', 'median', 'min', 'max')
            
        Returns:
            Time series DataFrame
        """
        try:
            df = self.load_historical_data(variables=[variable_name])
            
            if df.empty:
                logger.warning(f"No data found for variable: {variable_name}")
                return pd.DataFrame()
            
            # Group by timestamp and aggregate spatially
            if aggregation == 'mean':
                ts_data = df.groupby('timestamp')['value'].mean()
            elif aggregation == 'median':
                ts_data = df.groupby('timestamp')['value'].median()
            elif aggregation == 'min':
                ts_data = df.groupby('timestamp')['value'].min()
            elif aggregation == 'max':
                ts_data = df.groupby('timestamp')['value'].max()
            else:
                ts_data = df.groupby('timestamp')['value'].mean()
            
            ts_df = pd.DataFrame({
                'timestamp': ts_data.index,
                'value': ts_data.values,
                'variable_name': variable_name,
                'unit': df['unit'].iloc[0]
            })
            
            logger.info(f"Generated time series for {variable_name}: {len(ts_df)} points")
            return ts_df
            
        except Exception as e:
            logger.error(f"Failed to generate time series for {variable_name}: {e}")
            return pd.DataFrame()
    
    def get_spatial_data(self, 
                        variable_name: str,
                        date: Optional[str] = None) -> pd.DataFrame:
        """
        Get spatial data for a specific variable and date
        
        Args:
            variable_name: Name of the variable
            date: Specific date (YYYY-MM-DD), uses latest if None
            
        Returns:
            Spatial DataFrame
        """
        try:
            df = self.load_historical_data(variables=[variable_name])
            
            if df.empty:
                logger.warning(f"No data found for variable: {variable_name}")
                return pd.DataFrame()
            
            if date:
                # Filter to specific date
                target_date = pd.to_datetime(date).date()
                df['date'] = df['timestamp'].dt.date
                df = df[df['date'] == target_date]
            else:
                # Use latest available date
                latest_date = df['timestamp'].max().date()
                df['date'] = df['timestamp'].dt.date
                df = df[df['date'] == latest_date]
                logger.info(f"Using latest date: {latest_date}")
            
            if df.empty:
                logger.warning(f"No data found for {variable_name} on {date or 'latest date'}")
                return pd.DataFrame()
            
            spatial_df = df[['latitude', 'longitude', 'value', 'unit', 'timestamp']].copy()
            spatial_df['variable_name'] = variable_name
            
            logger.info(f"Generated spatial data for {variable_name}: {len(spatial_df)} points")
            return spatial_df
            
        except Exception as e:
            logger.error(f"Failed to generate spatial data for {variable_name}: {e}")
            return pd.DataFrame()
    
    def get_correlation_matrix(self, variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix between HAB variables
        
        Args:
            variables: List of variables to include (uses all if None)
            
        Returns:
            Correlation matrix DataFrame
        """
        try:
            df = self.load_historical_data(variables=variables)
            
            if df.empty:
                logger.warning("No data available for correlation analysis")
                return pd.DataFrame()
            
            # Pivot data to have variables as columns
            pivot_df = df.pivot_table(
                values='value',
                index=['timestamp', 'latitude', 'longitude'],
                columns='variable_name',
                aggfunc='mean'
            )
            
            # Calculate correlation matrix
            corr_matrix = pivot_df.corr()
            
            logger.info(f"Generated correlation matrix: {corr_matrix.shape}")
            return corr_matrix
            
        except Exception as e:
            logger.error(f"Failed to generate correlation matrix: {e}")
            return pd.DataFrame()
    
    def export_for_modeling(self, 
                          variables: Optional[List[str]] = None,
                          output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Export data in format suitable for machine learning
        
        Args:
            variables: Variables to include
            output_file: Output CSV file path
            
        Returns:
            Modeling-ready DataFrame
        """
        try:
            df = self.load_historical_data(variables=variables)
            
            if df.empty:
                logger.warning("No data available for modeling export")
                return pd.DataFrame()
            
            # Create modeling format: rows are observations, columns are variables
            modeling_df = df.pivot_table(
                values='value',
                index=['timestamp', 'latitude', 'longitude'],
                columns='variable_name',
                aggfunc='mean'
            ).reset_index()
            
            # Add temporal features
            modeling_df['year'] = modeling_df['timestamp'].dt.year
            modeling_df['month'] = modeling_df['timestamp'].dt.month
            modeling_df['day_of_year'] = modeling_df['timestamp'].dt.dayofyear
            modeling_df['season'] = (modeling_df['month'] % 12 + 3) // 3
            
            # Fill missing values with mean (basic imputation)
            numeric_columns = modeling_df.select_dtypes(include=[np.number]).columns
            modeling_df[numeric_columns] = modeling_df[numeric_columns].fillna(
                modeling_df[numeric_columns].mean()
            )
            
            if output_file:
                output_path = Path(output_file)
                modeling_df.to_csv(output_path, index=False)
                logger.info(f"Exported modeling data to: {output_path}")
            
            logger.info(f"Prepared modeling dataset: {modeling_df.shape}")
            return modeling_df
            
        except Exception as e:
            logger.error(f"Failed to export modeling data: {e}")
            return pd.DataFrame()
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """Validate data quality and identify issues"""
        try:
            df = self.load_historical_data()
            
            if df.empty:
                return {"error": "No data available for validation"}
            
            validation_results = {
                "total_records": len(df),
                "missing_values": {
                    "timestamp": df['timestamp'].isna().sum(),
                    "latitude": df['latitude'].isna().sum(),
                    "longitude": df['longitude'].isna().sum(),
                    "value": df['value'].isna().sum()
                },
                "duplicate_records": len(df) - len(df.drop_duplicates()),
                "outliers": {},
                "temporal_gaps": {},
                "spatial_coverage": {}
            }
            
            # Check for outliers (values beyond 3 standard deviations)
            for variable in df['variable_name'].unique():
                var_data = df[df['variable_name'] == variable]['value']
                mean_val = var_data.mean()
                std_val = var_data.std()
                outliers = var_data[(var_data < mean_val - 3*std_val) | (var_data > mean_val + 3*std_val)]
                validation_results["outliers"][variable] = len(outliers)
            
            # Check temporal gaps
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff()
            max_gap = time_diffs.max()
            validation_results["temporal_gaps"] = {
                "max_gap_days": max_gap.days if max_gap else 0,
                "gaps_over_week": (time_diffs > pd.Timedelta(days=7)).sum()
            }
            
            # Spatial coverage check
            validation_results["spatial_coverage"] = {
                "lat_range_degrees": df['latitude'].max() - df['latitude'].min(),
                "lon_range_degrees": df['longitude'].max() - df['longitude'].min(),
                "unique_locations": len(df[['latitude', 'longitude']].drop_duplicates())
            }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate data quality: {e}")
            return {"error": str(e)}