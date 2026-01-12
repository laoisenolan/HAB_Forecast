"""
HAB Event Data Integrator

This module integrates real HAB event data from historical records
into the model training pipeline to improve accuracy and provide
ground truth labels for training.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class HABEventIntegrator:
    """
    Integrates real HAB event data into the model training pipeline.
    
    This class processes historical HAB event records and converts them
    into training labels that can be used to improve model accuracy.
    """
    
    def __init__(self, config_path: str = "config/features.yaml"):
        """Initialize the HAB event integrator."""
        self.config_path = config_path
        self.config = self._load_config()
        self.hab_events = None
        self.processed_events = None
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError as e:
            # Try with different encodings if UTF-8 fails
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(self.config_path, 'r', encoding=encoding) as f:
                        return yaml.safe_load(f)
                except (UnicodeDecodeError, yaml.YAMLError):
                    continue
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return {}
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return {}
    
    def load_hab_events(self, csv_path: str) -> pd.DataFrame:
        """
        Load HAB event data from CSV file.
        
        Args:
            csv_path: Path to the CSV file containing HAB event data
            
        Returns:
            DataFrame with processed HAB event data
        """
        try:
            # Load the CSV file
            df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
            logger.info(f"Loaded {len(df)} HAB events from {csv_path}")
            
            # Clean and process the data
            df = self._clean_hab_data(df)
            self.hab_events = df
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load HAB events from {csv_path}: {e}")
            raise
    
    def _clean_hab_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize HAB event data.
        
        Args:
            df: Raw HAB event DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Parse dates
        df_clean['date_occurrence_parsed'] = pd.to_datetime(
            df_clean['date_occurrence_parsed'], errors='coerce'
        )
        df_clean['date_end_parsed'] = pd.to_datetime(
            df_clean['date_end_parsed'], errors='coerce'
        )
        
        # Clean cell count data
        df_clean['highest_cell_count_cells_per_ml'] = pd.to_numeric(
            df_clean['highest_cell_count_cells_per_ml'].str.replace(',', '').str.replace('cells/ml', '').str.replace('cells/day', ''),
            errors='coerce'
        )
        
        # Extract species information
        df_clean['primary_species'] = df_clean['plankton_component'].str.split(';').str[0]
        df_clean['species_count'] = df_clean['plankton_component'].str.count(';') + 1
        
        # Calculate event duration
        df_clean['duration_days'] = (
            df_clean['date_end_parsed'] - df_clean['date_occurrence_parsed']
        ).dt.days
        
        # Create severity categories based on cell count
        df_clean['severity_category'] = self._categorize_severity(
            df_clean['highest_cell_count_cells_per_ml']
        )
        
        # Extract geographic information
        df_clean['region'] = df_clean['nada_name'].fillna('Unknown')
        df_clean['prefecture'] = df_clean['prefecture_name'].fillna('Unknown')
        
        logger.info(f"Cleaned {len(df_clean)} HAB events")
        return df_clean
    
    def _categorize_severity(self, cell_counts: pd.Series) -> pd.Series:
        """
        Categorize HAB events by severity based on cell counts.
        
        Args:
            cell_counts: Series of cell count values
            
        Returns:
            Series of severity categories
        """
        def categorize(count):
            if pd.isna(count):
                return 'Unknown'
            elif count < 1000:
                return 'Low'
            elif count < 10000:
                return 'Moderate'
            elif count < 100000:
                return 'High'
            else:
                return 'Extreme'
        
        return cell_counts.apply(categorize)
    
    def create_training_labels(self, 
                             start_date: str = "2023-01-01",
                             end_date: str = "2023-12-31",
                             spatial_bounds: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create training labels from HAB events for the specified time period.
        
        Args:
            start_date: Start date for label generation (YYYY-MM-DD)
            end_date: End date for label generation (YYYY-MM-DD)
            spatial_bounds: Optional spatial bounds for filtering events
            
        Returns:
            DataFrame with training labels
        """
        if self.hab_events is None:
            raise ValueError("No HAB events loaded. Call load_hab_events() first.")
        
        # Filter events by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        mask = (
            (self.hab_events['date_occurrence_parsed'] >= start_dt) &
            (self.hab_events['date_occurrence_parsed'] <= end_dt)
        )
        
        filtered_events = self.hab_events[mask].copy()
        logger.info(f"Filtered to {len(filtered_events)} events in date range")
        
        # Create daily labels
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        labels = []
        
        for date in date_range:
            # Find events active on this date
            active_events = filtered_events[
                (filtered_events['date_occurrence_parsed'] <= date) &
                (filtered_events['date_end_parsed'] >= date)
            ]
            
            if len(active_events) > 0:
                # Multiple events possible, take the most severe
                # Handle NaN values in cell counts
                valid_cell_counts = active_events['highest_cell_count_cells_per_ml'].dropna()
                
                if len(valid_cell_counts) > 0:
                    max_severity = valid_cell_counts.max()
                    max_severity_idx = valid_cell_counts.idxmax()
                    max_severity_category = active_events.loc[max_severity_idx, 'severity_category']
                    primary_species = active_events.loc[max_severity_idx, 'primary_species']
                else:
                    # No valid cell counts, use default values
                    max_severity = 1000  # Default moderate level
                    max_severity_category = 'Moderate'
                    primary_species = active_events.iloc[0]['primary_species']
                
                # Convert to risk percentage
                risk_percentage = self._severity_to_risk_percentage(max_severity_category, max_severity)
                
                label = {
                    'date': date,
                    'hab_active': True,
                    'risk_percentage': risk_percentage,
                    'severity_category': max_severity_category,
                    'cell_count': max_severity,
                    'primary_species': primary_species,
                    'event_count': len(active_events)
                }
            else:
                # No HAB events on this date
                label = {
                    'date': date,
                    'hab_active': False,
                    'risk_percentage': 0.0,
                    'severity_category': 'None',
                    'cell_count': 0,
                    'primary_species': 'None',
                    'event_count': 0
                }
            
            labels.append(label)
        
        labels_df = pd.DataFrame(labels)
        logger.info(f"Created {len(labels_df)} daily labels")
        
        return labels_df
    
    def _severity_to_risk_percentage(self, severity: str, cell_count: float) -> float:
        """
        Convert severity category and cell count to risk percentage.
        
        Args:
            severity: Severity category (Low, Moderate, High, Extreme)
            cell_count: Cell count value
            
        Returns:
            Risk percentage (0-100)
        """
        if pd.isna(cell_count) or cell_count == 0:
            return 0.0
        
        # Base risk percentages
        base_risks = {
            'Low': 10,
            'Moderate': 30,
            'High': 60,
            'Extreme': 90,
            'Unknown': 20
        }
        
        base_risk = base_risks.get(severity, 20)
        
        # Adjust based on cell count (logarithmic scale)
        if cell_count > 0:
            log_count = np.log10(cell_count)
            # Normalize to 0-1 range for typical HAB cell counts (10^2 to 10^6)
            normalized_count = max(0, min(1, (log_count - 2) / 4))
            adjusted_risk = base_risk + (normalized_count * 20)
            return min(100, max(0, adjusted_risk))
        
        return base_risk
    
    def get_species_statistics(self) -> Dict:
        """
        Get statistics about HAB species in the dataset.
        
        Returns:
            Dictionary with species statistics
        """
        if self.hab_events is None:
            return {}
        
        species_stats = self.hab_events.groupby('primary_species').agg({
            'record_number': 'count',
            'highest_cell_count_cells_per_ml': ['mean', 'max', 'min'],
            'duration_days': 'mean',
            'severity_category': lambda x: x.value_counts().to_dict()
        }).round(2)
        
        return species_stats.to_dict()
    
    def get_geographic_statistics(self) -> Dict:
        """
        Get statistics about HAB events by geographic region.
        
        Returns:
            Dictionary with geographic statistics
        """
        if self.hab_events is None:
            return {}
        
        geo_stats = self.hab_events.groupby(['region', 'prefecture']).agg({
            'record_number': 'count',
            'highest_cell_count_cells_per_ml': 'mean',
            'duration_days': 'mean',
            'severity_category': lambda x: x.value_counts().to_dict()
        }).round(2)
        
        return geo_stats.to_dict()
    
    def get_temporal_statistics(self) -> Dict:
        """
        Get statistics about HAB events by time period.
        
        Returns:
            Dictionary with temporal statistics
        """
        if self.hab_events is None:
            return {}
        
        # Add month and season columns
        df_temp = self.hab_events.copy()
        df_temp['month'] = df_temp['date_occurrence_parsed'].dt.month
        df_temp['season'] = df_temp['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        monthly_stats = df_temp.groupby('month').agg({
            'record_number': 'count',
            'highest_cell_count_cells_per_ml': 'mean'
        }).round(2)
        
        seasonal_stats = df_temp.groupby('season').agg({
            'record_number': 'count',
            'highest_cell_count_cells_per_ml': 'mean'
        }).round(2)
        
        return {
            'monthly': monthly_stats.to_dict(),
            'seasonal': seasonal_stats.to_dict()
        }
    
    def save_processed_labels(self, labels_df: pd.DataFrame, output_path: str):
        """
        Save processed labels to file.
        
        Args:
            labels_df: DataFrame with training labels
            output_path: Path to save the labels
        """
        try:
            labels_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(labels_df)} labels to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save labels to {output_path}: {e}")
            raise
    
    def generate_summary_report(self) -> str:
        """
        Generate a summary report of the HAB event data.
        
        Returns:
            String containing the summary report
        """
        if self.hab_events is None:
            return "No HAB events loaded."
        
        report = []
        report.append("=" * 60)
        report.append("HAB EVENT DATA SUMMARY REPORT")
        report.append("=" * 60)
        
        # Basic statistics
        report.append(f"Total Events: {len(self.hab_events)}")
        report.append(f"Date Range: {self.hab_events['date_occurrence_parsed'].min()} to {self.hab_events['date_occurrence_parsed'].max()}")
        report.append(f"Unique Species: {self.hab_events['primary_species'].nunique()}")
        report.append(f"Unique Regions: {self.hab_events['region'].nunique()}")
        
        # Severity distribution
        severity_counts = self.hab_events['severity_category'].value_counts()
        report.append("\nSeverity Distribution:")
        for severity, count in severity_counts.items():
            report.append(f"  {severity}: {count} events")
        
        # Top species
        top_species = self.hab_events['primary_species'].value_counts().head(5)
        report.append("\nTop 5 Species:")
        for species, count in top_species.items():
            report.append(f"  {species}: {count} events")
        
        # Geographic distribution
        top_regions = self.hab_events['region'].value_counts().head(5)
        report.append("\nTop 5 Regions:")
        for region, count in top_regions.items():
            report.append(f"  {region}: {count} events")
        
        # Cell count statistics
        cell_stats = self.hab_events['highest_cell_count_cells_per_ml'].describe()
        report.append(f"\nCell Count Statistics:")
        report.append(f"  Mean: {cell_stats['mean']:.0f} cells/ml")
        report.append(f"  Max: {cell_stats['max']:.0f} cells/ml")
        report.append(f"  Min: {cell_stats['min']:.0f} cells/ml")
        
        report.append("=" * 60)
        
        return "\n".join(report) 