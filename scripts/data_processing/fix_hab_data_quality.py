#!/usr/bin/env python3
"""
Enhanced HAB Data Fixer
=======================

Advanced data fixer for HAB occurrence CSV data with comprehensive error handling,
logging, validation, and quality assessment capabilities.

Features:
- Comprehensive logging and error handling
- Data validation and quality assessment
- Progress tracking for large datasets
- Detailed issue detection and reporting
- Backup and recovery mechanisms
- Configurable settings and parameters

Author: Enhanced for HAB Forecast Project
"""

import pandas as pd
import numpy as np
import re
import logging
import os
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DataQualityMetrics:
    """Data quality metrics for assessment"""
    total_rows: int = 0
    valid_years: int = 0
    valid_start_dates: int = 0
    valid_end_dates: int = 0
    valid_species: int = 0
    valid_cell_counts: int = 0
    valid_locations: int = 0
    missing_data_percentage: float = 0.0
    duplicate_entries: int = 0
    date_range_years: Tuple[int, int] = (0, 0)
    species_count: int = 0
    location_count: int = 0


@dataclass
class FixingSummary:
    """Summary of fixing operations"""
    rows_processed: int = 0
    rows_fixed: int = 0
    issues_by_category: Dict[str, int] = None
    fixes_by_year: Dict[str, int] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.issues_by_category is None:
            self.issues_by_category = {}
        if self.fixes_by_year is None:
            self.fixes_by_year = {}


class EnhancedHABDataFixer:
    """
    Enhanced HAB data fixer with comprehensive error handling and validation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the enhanced data fixer
        
        Args:
            config: Configuration dictionary with fixer settings
        """
        self.config = config or self._get_default_config()
        self.setup_logging()
        
        # Initialize tracking variables
        self.fixed_rows = []
        self.issues_found = []
        self.quality_issues = []
        self.raw_lines = []
        self.processing_stats = FixingSummary()
        
        # Define expected column structure
        self.expected_columns = [
            'report_year', 'source_file', 'pref_event_id', 'start_date', 'end_date', 
            'duration_days', 'sea_area_jp', 'sea_area_en', 'prefecture_jp', 
            'prefecture_en', 'locality_jp', 'plankton', 'fishery_damage', 
            'max_cells_per_ml_num', 'max_area_km2'
        ]
        
        # Issue categories for tracking
        self.issue_categories = {
            'column_misalignment': 0,
            'date_format_error': 0,
            'missing_data': 0,
            'data_type_error': 0,
            'validation_error': 0,
            'encoding_error': 0,
            'duplicate_entry': 0
        }
        
        self.logger.info("Enhanced HAB Data Fixer initialized")
        self.logger.info(f"Configuration: {self.config}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'create_backup': True,
            'validate_data': True,
            'check_duplicates': True,
            'fix_encoding_issues': True,
            'progress_reporting': True,
            'detailed_logging': True,
            'output_format': 'csv',
            'date_format': '%Y-%m-%d',
            'cell_count_threshold': 1000000,  # Flag unusually high cell counts
            'duration_threshold': 365,  # Flag events longer than a year
        }
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_level = logging.DEBUG if self.config.get('detailed_logging') else logging.INFO
        
        # Create logs directory if it doesn't exist
        Path('logs').mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/hab_data_fixer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('HABDataFixer')
        self.logger.info("Logging system initialized")
    
    def create_backup(self, input_file: str) -> str:
        """
        Create backup of original file
        
        Args:
            input_file: Path to input file
            
        Returns:
            Path to backup file
        """
        if not self.config.get('create_backup'):
            return None
            
        try:
            backup_dir = Path('backups')
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"{Path(input_file).stem}_backup_{timestamp}.csv"
            
            shutil.copy2(input_file, backup_file)
            self.logger.info(f"Backup created: {backup_file}")
            
            return str(backup_file)
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return None
    
    def analyze_file_structure(self, filename: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of file structure and issues
        
        Args:
            filename: Path to CSV file
            
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info("Starting comprehensive file structure analysis")
        
        try:
            # Read file with encoding detection
            encoding = self._detect_encoding(filename)
            self.logger.info(f"Detected encoding: {encoding}")
            
            with open(filename, 'r', encoding=encoding) as f:
                lines = f.readlines()
            
            self.raw_lines = [line.strip() for line in lines if line.strip()]
            header = self.raw_lines[0].split(',')
            
            self.logger.info(f"File contains {len(self.raw_lines)} total lines")
            self.logger.info(f"Header has {len(header)} columns")
            self.logger.debug(f"Header: {header}")
            
            # Analyze structure by year
            year_analysis = self._analyze_by_year()
            
            # Detect specific issue patterns
            issue_patterns = self._detect_issue_patterns()
            
            # Calculate file statistics
            file_stats = {
                'total_lines': len(self.raw_lines),
                'header_columns': len(header),
                'expected_columns': len(self.expected_columns),
                'encoding': encoding,
                'year_analysis': year_analysis,
                'issue_patterns': issue_patterns
            }
            
            self.logger.info("File structure analysis completed")
            return file_stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing file structure: {e}")
            raise
    
    def _detect_encoding(self, filename: str) -> str:
        """
        Detect file encoding
        
        Args:
            filename: Path to file
            
        Returns:
            Detected encoding
        """
        try:
            import chardet
            
            with open(filename, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                return result['encoding']
                
        except ImportError:
            self.logger.warning("chardet not available, assuming utf-8")
            return 'utf-8'
        except Exception as e:
            self.logger.warning(f"Encoding detection failed: {e}, assuming utf-8")
            return 'utf-8'
    
    def _analyze_by_year(self) -> Dict[str, Any]:
        """
        Analyze data structure by year
        
        Returns:
            Year-wise analysis results
        """
        year_stats = {}
        
        for i, line in enumerate(self.raw_lines[1:], 1):
            try:
                cols = line.split(',')
                year = cols[0] if cols else 'Unknown'
                
                if year not in year_stats:
                    year_stats[year] = {
                        'count': 0,
                        'column_counts': {},
                        'issues': [],
                        'line_numbers': []
                    }
                
                year_stats[year]['count'] += 1
                year_stats[year]['line_numbers'].append(i + 1)  # +1 for header
                
                col_count = len(cols)
                if col_count not in year_stats[year]['column_counts']:
                    year_stats[year]['column_counts'][col_count] = 0
                year_stats[year]['column_counts'][col_count] += 1
                
                # Detect specific issues by year
                self._detect_year_specific_issues(year, cols, i + 1, year_stats[year]['issues'])
                
            except Exception as e:
                self.logger.warning(f"Error processing line {i + 1}: {e}")
                continue
        
        # Log year analysis
        for year in sorted(year_stats.keys()):
            data = year_stats[year]
            self.logger.info(f"Year {year}: {data['count']} rows")
            
            if data['issues']:
                unique_issues = set(data['issues'])
                self.logger.warning(f"  Issues in {year}: {unique_issues}")
        
        return year_stats
    
    def _detect_year_specific_issues(self, year: str, cols: List[str], line_num: int, issues: List[str]):
        """
        Detect specific issues for different year ranges
        
        Args:
            year: Year value
            cols: Column values
            line_num: Line number
            issues: List to append issues to
        """
        try:
            if year in ['2020', '2021']:
                # Check for column misalignment
                if len(cols) > 6 and cols[6] and '/' in cols[6]:
                    issues.append('dates_in_sea_area_cols')
                if len(cols) > 2 and not cols[3] and not cols[4]:  # Empty start/end dates
                    issues.append('empty_date_fields')
                if len(cols) > 10 and cols[10]:  # Species in wrong column
                    issues.append('species_misaligned')
                    
            elif year in ['2022', '2023']:
                # Check for missing fields
                if len(cols) > 1 and not cols[1]:  # Empty source_file
                    issues.append('empty_source_file')
                if len(cols) > 2 and not cols[2]:  # Empty pref_event_id
                    issues.append('empty_pref_event_id')
                if len(cols) > 11 and (not cols[11] or cols[11] in ['Unknown', 'None']):
                    issues.append('missing_species_data')
                    
            elif year.isdigit() and 2014 <= int(year) <= 2019:
                # Check for date format issues
                if len(cols) > 3 and cols[3]:
                    if not self._is_valid_date_format(cols[3]):
                        issues.append('invalid_date_format')
                        
        except Exception as e:
            self.logger.debug(f"Error detecting year-specific issues for line {line_num}: {e}")
    
    def _detect_issue_patterns(self) -> Dict[str, Any]:
        """
        Detect common issue patterns across the dataset
        
        Returns:
            Dictionary of detected patterns
        """
        patterns = {
            'misaligned_columns': 0,
            'empty_critical_fields': 0,
            'date_format_issues': 0,
            'encoding_issues': 0,
            'potential_duplicates': 0
        }
        
        for i, line in enumerate(self.raw_lines[1:], 1):
            try:
                cols = line.split(',')
                
                # Check for misaligned columns
                if len(cols) != len(self.expected_columns):
                    patterns['misaligned_columns'] += 1
                
                # Check for empty critical fields
                if len(cols) > 3 and (not cols[0] or not cols[3]):  # year or start_date
                    patterns['empty_critical_fields'] += 1
                
                # Check for date format issues
                if len(cols) > 3 and cols[3] and not self._is_valid_date_format(cols[3]):
                    patterns['date_format_issues'] += 1
                
                # Check for encoding issues
                if any('\ufffd' in col for col in cols):  # Replacement character
                    patterns['encoding_issues'] += 1
                    
            except Exception as e:
                self.logger.debug(f"Error detecting patterns in line {i + 1}: {e}")
        
        return patterns
    
    def _is_valid_date_format(self, date_str: str) -> bool:
        """
        Check if date string is in valid format
        
        Args:
            date_str: Date string to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not date_str or not date_str.strip():
            return False
        
        date_str = date_str.strip()
        
        # Check common valid formats
        valid_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{4}/\d{1,2}/\d{1,2}$',  # YYYY/M/D or YYYY/MM/DD
            r'^\d{1,2}/\d{1,2}$'  # M/D or MM/DD (needs year context)
        ]
        
        return any(re.match(pattern, date_str) for pattern in valid_patterns)
    
    def assess_data_quality(self, df: pd.DataFrame) -> DataQualityMetrics:
        """
        Assess data quality metrics
        
        Args:
            df: DataFrame to assess
            
        Returns:
            Data quality metrics
        """
        self.logger.info("Assessing data quality")
        
        try:
            metrics = DataQualityMetrics()
            
            metrics.total_rows = len(df)
            metrics.valid_years = df['report_year'].notna().sum()
            metrics.valid_start_dates = df['start_date'].notna().sum()
            metrics.valid_end_dates = df['end_date'].notna().sum()
            metrics.valid_species = df['plankton'].notna().sum()
            metrics.valid_cell_counts = df['max_cells_per_ml_num'].notna().sum()
            metrics.valid_locations = df['locality_jp'].notna().sum()
            
            # Calculate missing data percentage
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            metrics.missing_data_percentage = (missing_cells / total_cells) * 100
            
            # Check for duplicates
            metrics.duplicate_entries = df.duplicated().sum()
            
            # Date range
            valid_years = df['report_year'].dropna()
            if len(valid_years) > 0:
                metrics.date_range_years = (int(valid_years.min()), int(valid_years.max()))
            
            # Species and location counts
            metrics.species_count = df['plankton'].nunique()
            metrics.location_count = df['locality_jp'].nunique()
            
            self.logger.info(f"Data quality assessment completed: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {e}")
            return DataQualityMetrics()
    
    def fix_2020_2021_misaligned_data(self, cols: List[str], year: str, line_num: int) -> List[str]:
        """
        Fix 2020-2021 data with severe column misalignment
        
        Args:
            cols: Original column values
            year: Year value
            line_num: Line number for logging
            
        Returns:
            Fixed column values
        """
        fixed = [''] * 15
        
        try:
            self.logger.debug(f"Fixing 2020-2021 misalignment for line {line_num}")
            
            # Column 0: year (correct)
            fixed[0] = year
            
            # Column 1: source_file (usually correct)
            if len(cols) > 1:
                fixed[1] = cols[1]
            
            # Column 2: contains location info - move to locality_jp
            if len(cols) > 2 and cols[2]:
                fixed[10] = cols[2].strip()
            
            # Columns 6-7: contain dates in MM/DD format
            start_date = self._extract_date_from_misaligned(cols, 6, year)
            if start_date:
                fixed[3] = start_date
            
            end_date = self._extract_date_from_misaligned(cols, 7, year)
            if end_date:
                fixed[4] = end_date
            
            # Column 10: contains species name
            if len(cols) > 10 and cols[10]:
                fixed[11] = cols[10].strip()
            
            # Columns 11-12: contain split cell count data
            cell_count = self._reconstruct_cell_count(cols, [11, 12])
            if cell_count is not None:
                fixed[13] = str(cell_count)
            
            # Column 13: fishery damage info
            if len(cols) > 13:
                fixed[12] = cols[13]
            
            # Calculate duration
            fixed[5] = self._calculate_duration(fixed[3], fixed[4])
            
            self.issue_categories['column_misalignment'] += 1
            self.issues_found.append(f"Line {line_num}: Fixed {year} column misalignment")
            
        except Exception as e:
            self.logger.error(f"Error fixing 2020-2021 data at line {line_num}: {e}")
            self.issue_categories['data_type_error'] += 1
        
        return fixed
    
    def fix_2022_2023_data(self, cols: List[str], year: str, line_num: int) -> List[str]:
        """
        Fix 2022-2023 data (mostly structural, some missing fields)
        
        Args:
            cols: Original column values
            year: Year value
            line_num: Line number for logging
            
        Returns:
            Fixed column values
        """
        fixed = [''] * 15
        
        try:
            self.logger.debug(f"Fixing 2022-2023 data for line {line_num}")
            
            # Copy existing structure (mostly correct)
            for i in range(min(len(cols), 15)):
                fixed[i] = cols[i]
            
            # Ensure year is correct
            fixed[0] = year
            
            # Handle missing source_file
            if not fixed[1]:
                fixed[1] = f"missing_source_{year}.csv"
                self.issue_categories['missing_data'] += 1
            
            # Handle missing pref_event_id
            if not fixed[2]:
                fixed[2] = f"MISSING-{year}-{line_num}"
                self.issue_categories['missing_data'] += 1
            
            # Calculate duration if dates exist
            fixed[5] = self._calculate_duration(fixed[3], fixed[4])
            
            # Validate and fix species data
            if fixed[11] in ['Unknown', 'None', '']:
                fixed[11] = 'Unidentified'
                self.issue_categories['missing_data'] += 1
            
            self.issues_found.append(f"Line {line_num}: Fixed {year} missing fields")
            
        except Exception as e:
            self.logger.error(f"Error fixing 2022-2023 data at line {line_num}: {e}")
            self.issue_categories['data_type_error'] += 1
        
        return fixed
    
    def fix_2014_2019_data(self, cols: List[str], year: str, line_num: int) -> List[str]:
        """
        Fix 2014-2019 data (mainly date format issues)
        
        Args:
            cols: Original column values
            year: Year value
            line_num: Line number for logging
            
        Returns:
            Fixed column values
        """
        fixed = cols.copy()
        
        try:
            self.logger.debug(f"Fixing 2014-2019 data for line {line_num}")
            
            # Ensure we have 15 columns
            while len(fixed) < 15:
                fixed.append('')
            fixed = fixed[:15]
            
            # Fix start_date (column 3)
            if len(fixed) > 3:
                original_date = fixed[3]
                fixed[3] = self.standardize_date(fixed[3], year)
                if original_date != fixed[3] and original_date:
                    self.issue_categories['date_format_error'] += 1
            
            # Fix end_date (column 4)
            if len(fixed) > 4:
                original_date = fixed[4]
                fixed[4] = self.standardize_date(fixed[4], year)
                if original_date != fixed[4] and original_date:
                    self.issue_categories['date_format_error'] += 1
            
            # Calculate duration
            fixed[5] = self._calculate_duration(fixed[3], fixed[4])
            
            # Clean numeric fields
            if fixed[13]:
                original_value = fixed[13]
                fixed[13] = self.clean_numeric(fixed[13])
                if original_value != fixed[13]:
                    self.issue_categories['data_type_error'] += 1
            
            if fixed[14]:
                original_value = fixed[14]
                fixed[14] = self.clean_numeric(fixed[14])
                if original_value != fixed[14]:
                    self.issue_categories['data_type_error'] += 1
            
            self.issues_found.append(f"Line {line_num}: Fixed {year} date format and numeric fields")
            
        except Exception as e:
            self.logger.error(f"Error fixing 2014-2019 data at line {line_num}: {e}")
            self.issue_categories['data_type_error'] += 1
        
        return fixed
    
    def _extract_date_from_misaligned(self, cols: List[str], col_index: int, year: str) -> Optional[str]:
        """
        Extract and format date from misaligned column
        
        Args:
            cols: Column values
            col_index: Index of column containing date
            year: Year for context
            
        Returns:
            Formatted date string or None
        """
        try:
            if len(cols) > col_index and cols[col_index]:
                date_raw = cols[col_index].strip()
                
                if '/' in date_raw:
                    # Handle MM/DD format
                    parts = date_raw.split('/')
                    if len(parts) == 2:
                        month, day = parts
                        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    
                elif '-' in date_raw and len(date_raw) >= 8:
                    # Already in good format
                    return date_raw
            
        except Exception as e:
            self.logger.debug(f"Error extracting date from column {col_index}: {e}")
        
        return None
    
    def _reconstruct_cell_count(self, cols: List[str], col_indices: List[int]) -> Optional[float]:
        """
        Reconstruct cell count from split columns
        
        Args:
            cols: Column values
            col_indices: Indices of columns containing cell count parts
            
        Returns:
            Reconstructed cell count or None
        """
        try:
            cell_parts = []
            
            for idx in col_indices:
                if len(cols) > idx and cols[idx]:
                    part = cols[idx].replace('"', '').strip()
                    if part:
                        cell_parts.append(part)
            
            if cell_parts:
                # Join parts and extract numbers
                cell_str = ''.join(cell_parts)
                numbers = re.findall(r'\d+', cell_str.replace(',', ''))
                
                if numbers:
                    count = float(''.join(numbers))
                    
                    # Validate reasonable range
                    if count > self.config.get('cell_count_threshold', 1000000):
                        self.quality_issues.append(f"Unusually high cell count: {count}")
                    
                    return count
            
        except Exception as e:
            self.logger.debug(f"Error reconstructing cell count: {e}")
        
        return None
    
    def _calculate_duration(self, start_date: str, end_date: str) -> str:
        """
        Calculate duration between dates
        
        Args:
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Duration in days as string
        """
        try:
            if start_date and end_date:
                start = datetime.strptime(start_date, '%Y-%m-%d')
                end = datetime.strptime(end_date, '%Y-%m-%d')
                duration = (end - start).days + 1
                
                # Validate reasonable duration
                if duration > self.config.get('duration_threshold', 365):
                    self.quality_issues.append(f"Unusually long duration: {duration} days")
                elif duration < 1:
                    self.quality_issues.append(f"Invalid duration: {duration} days")
                    duration = 1
                
                return str(duration)
            elif start_date:
                return '1'  # Default to 1 day if only start date available
                
        except Exception as e:
            self.logger.debug(f"Error calculating duration: {e}")
        
        return ''
    
    def standardize_date(self, date_str: str, year: str) -> str:
        """
        Convert various date formats to YYYY-MM-DD with enhanced validation
        
        Args:
            date_str: Original date string
            year: Year for context
            
        Returns:
            Standardized date string
        """
        if not date_str or not date_str.strip():
            return ''
        
        date_str = date_str.strip()
        
        try:
            # Already correct format
            if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                # Validate the date
                datetime.strptime(date_str, '%Y-%m-%d')
                return date_str
            
            # MM/DD format
            if re.match(r'^\d{1,2}/\d{1,2}$', date_str):
                month, day = date_str.split('/')
                month = month.zfill(2)
                day = day.zfill(2)
                
                # Validate date
                result = f"{year}-{month}-{day}"
                datetime.strptime(result, '%Y-%m-%d')
                return result
            
            # YYYY/MM/DD format
            if re.match(r'^\d{4}/\d{1,2}/\d{1,2}$', date_str):
                y, m, d = date_str.split('/')
                result = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
                datetime.strptime(result, '%Y-%m-%d')
                return result
            
        except ValueError as e:
            self.logger.warning(f"Invalid date encountered: {date_str} - {e}")
            self.issue_categories['validation_error'] += 1
        except Exception as e:
            self.logger.debug(f"Error standardizing date {date_str}: {e}")
        
        return ''
    
    def clean_numeric(self, value: str) -> str:
        """
        Clean numeric fields with enhanced validation
        
        Args:
            value: Value to clean
            
        Returns:
            Cleaned numeric string
        """
        if not value:
            return ''
        
        try:
            # Remove quotes, commas, and extra whitespace
            clean_val = str(value).replace('"', '').replace(',', '').strip()
            
            # Extract number
            numbers = re.findall(r'\d+\.?\d*', clean_val)
            if numbers:
                result = numbers[0]
                
                # Validate number
                float(result)  # Will raise ValueError if invalid
                return result
            
        except ValueError as e:
            self.logger.debug(f"Invalid numeric value: {value} - {e}")
            self.issue_categories['validation_error'] += 1
        except Exception as e:
            self.logger.debug(f"Error cleaning numeric value {value}: {e}")
        
        return ''
    
    def process_all_data(self, filename: str) -> bool:
        """
        Process the entire file with comprehensive error handling
        
        Args:
            filename: Path to input file
            
        Returns:
            True if processing succeeded, False otherwise
        """
        self.logger.info("Starting comprehensive data processing")
        start_time = datetime.now()
        
        try:
            # Create backup
            backup_file = self.create_backup(filename)
            
            # Analyze file structure
            file_analysis = self.analyze_file_structure(filename)
            
            # Process each row with progress tracking
            total_rows = len(self.raw_lines) - 1  # Exclude header
            processed = 0
            
            for i, line in enumerate(self.raw_lines[1:], 1):
                try:
                    cols = line.split(',')
                    
                    if not cols or not cols[0]:
                        self.logger.warning(f"Skipping empty line {i + 1}")
                        continue
                    
                    year = cols[0]
                    line_num = i + 1  # +1 for header
                    
                    # Apply appropriate fixing strategy
                    if year in ['2020', '2021']:
                        fixed_row = self.fix_2020_2021_misaligned_data(cols, year, line_num)
                        self.processing_stats.fixes_by_year['2020-2021'] = \
                            self.processing_stats.fixes_by_year.get('2020-2021', 0) + 1
                    elif year in ['2022', '2023']:
                        fixed_row = self.fix_2022_2023_data(cols, year, line_num)
                        self.processing_stats.fixes_by_year['2022-2023'] = \
                            self.processing_stats.fixes_by_year.get('2022-2023', 0) + 1
                    else:  # 2014-2019 and others
                        fixed_row = self.fix_2014_2019_data(cols, year, line_num)
                        self.processing_stats.fixes_by_year['2014-2019'] = \
                            self.processing_stats.fixes_by_year.get('2014-2019', 0) + 1
                    
                    self.fixed_rows.append(fixed_row)
                    processed += 1
                    
                    # Progress reporting
                    if self.config.get('progress_reporting') and processed % 100 == 0:
                        progress = (processed / total_rows) * 100
                        self.logger.info(f"Processing progress: {progress:.1f}% ({processed}/{total_rows})")
                    
                except Exception as e:
                    self.logger.error(f"Error processing line {i + 1}: {e}")
                    
                    # Create fallback row
                    fallback = cols.copy() if 'cols' in locals() else [''] * 15
                    while len(fallback) < 15:
                        fallback.append('')
                    self.fixed_rows.append(fallback[:15])
                    
                    self.issue_categories['data_type_error'] += 1
            
            # Update processing statistics
            self.processing_stats.rows_processed = processed
            self.processing_stats.rows_fixed = len(self.fixed_rows)
            self.processing_stats.issues_by_category = self.issue_categories.copy()
            self.processing_stats.processing_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Data processing completed: {processed} rows processed")
            self.logger.info(f"Processing time: {self.processing_stats.processing_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Critical error during data processing: {e}")
            return False
    
    def create_cleaned_dataframe(self) -> pd.DataFrame:
        """
        Create final cleaned DataFrame with comprehensive validation
        
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Creating final cleaned DataFrame")
        
        try:
            # Create DataFrame
            df = pd.DataFrame(self.fixed_rows, columns=self.expected_columns)
            
            # Data type conversions with error handling
            df['report_year'] = pd.to_numeric(df['report_year'], errors='coerce')
            df['duration_days'] = pd.to_numeric(df['duration_days'], errors='coerce')
            df['max_cells_per_ml_num'] = pd.to_numeric(df['max_cells_per_ml_num'], errors='coerce')
            df['max_area_km2'] = pd.to_numeric(df['max_area_km2'], errors='coerce')
            
            # Parse dates with error handling
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
            df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Check for duplicates if enabled
            if self.config.get('check_duplicates'):
                initial_count = len(df)
                df = df.drop_duplicates()
                duplicates_removed = initial_count - len(df)
                if duplicates_removed > 0:
                    self.logger.warning(f"Removed {duplicates_removed} duplicate rows")
                    self.issue_categories['duplicate_entry'] = duplicates_removed
            
            # Assess data quality
            quality_metrics = self.assess_data_quality(df)
            
            # Log quality summary
            self.logger.info(f"Final DataFrame quality:")
            self.logger.info(f"  Total rows: {len(df)}")
            self.logger.info(f"  Valid years: {quality_metrics.valid_years}")
            self.logger.info(f"  Valid start dates: {quality_metrics.valid_start_dates}")
            self.logger.info(f"  Valid species: {quality_metrics.valid_species}")
            self.logger.info(f"  Valid cell counts: {quality_metrics.valid_cell_counts}")
            self.logger.info(f"  Missing data: {quality_metrics.missing_data_percentage:.2f}%")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating cleaned DataFrame: {e}")
            raise
    
    def save_results(self, df: pd.DataFrame, output_filename: str) -> Tuple[str, str]:
        """
        Save cleaned data and create comprehensive report
        
        Args:
            df: Cleaned DataFrame
            output_filename: Output file path
            
        Returns:
            Tuple of (data_file, report_file) paths
        """
        self.logger.info("Saving results and generating reports")
        
        try:
            # Save cleaned CSV
            df.to_csv(output_filename, index=False)
            self.logger.info(f"Cleaned data saved: {output_filename}")
            
            # Create comprehensive report
            report_file = output_filename.replace('.csv', '_COMPREHENSIVE_REPORT.txt')
            self._create_comprehensive_report(df, output_filename, report_file)
            
            # Create JSON summary for programmatic access
            json_file = output_filename.replace('.csv', '_SUMMARY.json')
            self._create_json_summary(df, json_file)
            
            return output_filename, report_file
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise
    
    def _create_comprehensive_report(self, df: pd.DataFrame, output_filename: str, report_file: str):
        """
        Create comprehensive text report
        
        Args:
            df: Cleaned DataFrame
            output_filename: Output data file
            report_file: Report file path
        """
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("COMPREHENSIVE HAB DATA CLEANING REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Generated: {datetime.now()}\n")
                f.write(f"Input file: master_occurrences_2014_2023.csv\n")
                f.write(f"Output file: {output_filename}\n")
                f.write(f"Processing time: {self.processing_stats.processing_time:.2f} seconds\n\n")
                
                # Processing summary
                f.write("PROCESSING SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total rows processed: {self.processing_stats.rows_processed}\n")
                f.write(f"Final clean rows: {len(df)}\n")
                f.write(f"Total issues found: {len(self.issues_found)}\n")
                f.write(f"Quality issues detected: {len(self.quality_issues)}\n\n")
                
                # Issues by category
                f.write("ISSUES BY CATEGORY\n")
                f.write("-" * 20 + "\n")
                for category, count in self.issue_categories.items():
                    if count > 0:
                        f.write(f"  {category.replace('_', ' ').title()}: {count}\n")
                f.write("\n")
                
                # Fixes by year
                f.write("FIXES BY YEAR RANGE\n")
                f.write("-" * 20 + "\n")
                for year_range, count in self.processing_stats.fixes_by_year.items():
                    f.write(f"  {year_range}: {count} rows\n")
                f.write("\n")
                
                # Data quality metrics
                quality_metrics = self.assess_data_quality(df)
                f.write("DATA QUALITY METRICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"  Total rows: {quality_metrics.total_rows}\n")
                f.write(f"  Valid years: {quality_metrics.valid_years}\n")
                f.write(f"  Valid start dates: {quality_metrics.valid_start_dates}\n")
                f.write(f"  Valid end dates: {quality_metrics.valid_end_dates}\n")
                f.write(f"  Valid species: {quality_metrics.valid_species}\n")
                f.write(f"  Valid cell counts: {quality_metrics.valid_cell_counts}\n")
                f.write(f"  Valid locations: {quality_metrics.valid_locations}\n")
                f.write(f"  Missing data percentage: {quality_metrics.missing_data_percentage:.2f}%\n")
                f.write(f"  Duplicate entries: {quality_metrics.duplicate_entries}\n")
                f.write(f"  Date range: {quality_metrics.date_range_years[0]}-{quality_metrics.date_range_years[1]}\n")
                f.write(f"  Unique species: {quality_metrics.species_count}\n")
                f.write(f"  Unique locations: {quality_metrics.location_count}\n\n")
                
                # Year-wise distribution
                f.write("YEAR-WISE EVENT DISTRIBUTION\n")
                f.write("-" * 30 + "\n")
                year_counts = df['report_year'].value_counts().sort_index()
                for year, count in year_counts.items():
                    if pd.notna(year):
                        f.write(f"  {int(year)}: {count} events\n")
                f.write("\n")
                
                # Top species
                f.write("TOP 15 SPECIES BY FREQUENCY\n")
                f.write("-" * 30 + "\n")
                species_counts = df['plankton'].value_counts().head(15)
                for species, count in species_counts.items():
                    if pd.notna(species) and species.strip():
                        f.write(f"  {species}: {count} events\n")
                f.write("\n")
                
                # Top locations
                f.write("TOP 15 LOCATIONS BY FREQUENCY\n")
                f.write("-" * 30 + "\n")
                location_counts = df['locality_jp'].value_counts().head(15)
                for location, count in location_counts.items():
                    if pd.notna(location) and location.strip():
                        f.write(f"  {location}: {count} events\n")
                f.write("\n")
                
                # Quality issues (if any)
                if self.quality_issues:
                    f.write("QUALITY ISSUES DETECTED\n")
                    f.write("-" * 25 + "\n")
                    for issue in self.quality_issues[:20]:  # Limit to first 20
                        f.write(f"  {issue}\n")
                    if len(self.quality_issues) > 20:
                        f.write(f"  ... and {len(self.quality_issues) - 20} more issues\n")
                    f.write("\n")
                
                # Configuration used
                f.write("CONFIGURATION USED\n")
                f.write("-" * 20 + "\n")
                for key, value in self.config.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
                
                f.write("=" * 60 + "\n")
                f.write("Report completed successfully!\n")
                f.write("Data is now ready for machine learning training.\n")
                f.write("=" * 60 + "\n")
            
            self.logger.info(f"Comprehensive report created: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive report: {e}")
    
    def _create_json_summary(self, df: pd.DataFrame, json_file: str):
        """
        Create JSON summary for programmatic access
        
        Args:
            df: Cleaned DataFrame
            json_file: JSON file path
        """
        try:
            quality_metrics = self.assess_data_quality(df)
            
            summary = {
                'processing_timestamp': datetime.now().isoformat(),
                'processing_stats': {
                    'rows_processed': self.processing_stats.rows_processed,
                    'rows_in_final_dataset': len(df),
                    'processing_time_seconds': self.processing_stats.processing_time,
                    'fixes_by_year': self.processing_stats.fixes_by_year,
                    'issues_by_category': self.issue_categories
                },
                'data_quality': {
                    'total_rows': quality_metrics.total_rows,
                    'valid_years': quality_metrics.valid_years,
                    'valid_start_dates': quality_metrics.valid_start_dates,
                    'valid_end_dates': quality_metrics.valid_end_dates,
                    'valid_species': quality_metrics.valid_species,
                    'valid_cell_counts': quality_metrics.valid_cell_counts,
                    'valid_locations': quality_metrics.valid_locations,
                    'missing_data_percentage': quality_metrics.missing_data_percentage,
                    'duplicate_entries': quality_metrics.duplicate_entries,
                    'date_range_years': quality_metrics.date_range_years,
                    'species_count': quality_metrics.species_count,
                    'location_count': quality_metrics.location_count
                },
                'year_distribution': df['report_year'].value_counts().sort_index().to_dict(),
                'top_species': df['plankton'].value_counts().head(10).to_dict(),
                'top_locations': df['locality_jp'].value_counts().head(10).to_dict(),
                'config_used': self.config
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"JSON summary created: {json_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating JSON summary: {e}")


def run_enhanced_fix(input_file: str = 'test_data/master_occurrences_2014_2023.csv',
                    config: Dict[str, Any] = None) -> bool:
    """
    Run the enhanced fixing process with comprehensive error handling
    
    Args:
        input_file: Path to input CSV file
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    print("🌊 ENHANCED HAB DATA FIXER 🌊")
    print("=" * 40)
    print("Advanced data cleaning with comprehensive validation and logging\n")
    
    try:
        # Initialize enhanced fixer
        fixer = EnhancedHABDataFixer(config)
        
        # Verify input file exists
        if not os.path.exists(input_file):
            fixer.logger.error(f"Input file not found: {input_file}")
            return False
        
        # Process all data
        fixer.logger.info(f"Starting processing of {input_file}")
        success = fixer.process_all_data(input_file)
        
        if not success:
            fixer.logger.error("Data processing failed")
            return False
        
        # Create clean DataFrame
        clean_df = fixer.create_cleaned_dataframe()
        
        # Save results
        output_file = input_file.replace('.csv', '_ENHANCED_CLEANED.csv')
        data_file, report_file = fixer.save_results(clean_df, output_file)
        
        print(f"\n🎉 SUCCESS!")
        print(f"📁 Clean data: {data_file}")
        print(f"📋 Comprehensive report: {report_file}")
        print(f"📊 JSON summary: {output_file.replace('.csv', '_SUMMARY.json')}")
        print(f"⏱️  Processing time: {fixer.processing_stats.processing_time:.2f} seconds")
        print(f"🔧 Total fixes applied: {sum(fixer.issue_categories.values())}")
        print(f"\n🎯 Next: Use the cleaned file for ML training!")
        
        return True
        
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Configuration for the enhanced fixer
    config = {
        'create_backup': True,
        'validate_data': True,
        'check_duplicates': True,
        'fix_encoding_issues': True,
        'progress_reporting': True,
        'detailed_logging': True,
        'cell_count_threshold': 1000000,
        'duration_threshold': 365
    }
    
    # Run the enhanced fix
    success = run_enhanced_fix(config=config)
    
    if success:
        print("\n✅ Your HAB data is now thoroughly cleaned and ready for ML!")
        print("📖 Check the comprehensive report for detailed analysis.")
    else:
        print("\n❌ Enhanced fixing failed. Check error logs for details.")
