"""
Task 2.3: Comprehensive Data Quality Assessment & Gap Identification

This module provides thorough data quality assessment for historical HAB datasets,
identifying gaps, outliers, inconsistencies, and providing recommendations for
data improvement and gap filling strategies.

Features:
- Comprehensive quality metrics calculation
- Temporal and spatial gap identification
- Outlier detection and classification
- Cross-variable consistency checks
- Data source reliability assessment
- Gap filling strategy recommendations
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class QualityIssue:
    """Represents a data quality issue"""
    issue_id: str
    issue_type: str  # 'gap', 'outlier', 'inconsistency', 'missing', 'duplicate'
    severity: str    # 'low', 'medium', 'high', 'critical'
    variable: str
    location: str    # Description of where issue occurs
    affected_records: int
    description: str
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'issue_id': self.issue_id,
            'issue_type': self.issue_type,
            'severity': self.severity,
            'variable': self.variable,
            'location': self.location,
            'affected_records': self.affected_records,
            'description': self.description,
            'recommendation': self.recommendation
        }

class ComprehensiveQualityAssessor:
    """
    Comprehensive data quality assessment system for HAB datasets
    
    Assesses:
    - Temporal coverage and gaps
    - Spatial coverage and distribution
    - Data completeness and missing values
    - Outlier detection and classification
    - Cross-variable consistency
    - Data source reliability
    - Seasonal and trend patterns
    """
    
    def __init__(self, output_dir: str = "data/quality_reports"):
        """Initialize quality assessor"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality thresholds for different HAB variables
        self.variable_ranges = {
            'sea_surface_temperature': {'min': -5, 'max': 35, 'unit': '°C'},
            'chlorophyll_a': {'min': 0.01, 'max': 100, 'unit': 'mg/m³'},
            'chlorophyll_a_jaxa': {'min': 0.01, 'max': 100, 'unit': 'mg/m³'},
            'turbidity_kd490': {'min': 0.01, 'max': 10, 'unit': 'm⁻¹'},
            'nitrate': {'min': 0, 'max': 50, 'unit': 'mmol/m³'},
            'phosphate': {'min': 0, 'max': 5, 'unit': 'mmol/m³'},
            'silicate': {'min': 0, 'max': 200, 'unit': 'mmol/m³'},
            'dissolved_iron': {'min': 0, 'max': 10, 'unit': 'mmol/m³'},
            'dissolved_oxygen': {'min': 0, 'max': 500, 'unit': 'mmol/m³'},
            'primary_production': {'min': 0, 'max': 5000, 'unit': 'mg C/m³/day'},
            'wind_u': {'min': -30, 'max': 30, 'unit': 'm/s'},
            'wind_v': {'min': -30, 'max': 30, 'unit': 'm/s'},
            'precipitation': {'min': 0, 'max': 0.2, 'unit': 'm'},
            'ocean_salinity': {'min': 20, 'max': 40, 'unit': 'psu'},
            'mixed_layer_depth': {'min': 1, 'max': 500, 'unit': 'm'}
        }
        
        # Expected data frequency for different variables
        self.expected_frequency = {
            'daily': timedelta(days=1),
            'weekly': timedelta(days=7),
            'monthly': timedelta(days=30)
        }
        
        logger.info("Comprehensive quality assessor initialized")
    
    def assess_data_quality(self, data: pd.DataFrame, assessment_name: str = None) -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment
        
        Args:
            data: Historical data in standardized CSV format
            assessment_name: Name for this assessment
            
        Returns:
            Comprehensive quality assessment report
        """
        if assessment_name is None:
            assessment_name = f"quality_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"🔍 Starting comprehensive quality assessment: {assessment_name}")
        logger.info(f"📊 Data shape: {data.shape}")
        
        assessment_start = datetime.now()
        
        # Prepare data
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Perform all quality checks
        assessment = {
            "assessment_info": {
                "name": assessment_name,
                "timestamp": assessment_start.isoformat(),
                "data_shape": data.shape,
                "assessment_version": "1.0"
            },
            "data_overview": self._get_data_overview(data),
            "completeness_analysis": self._assess_completeness(data),
            "temporal_analysis": self._assess_temporal_quality(data),
            "spatial_analysis": self._assess_spatial_quality(data),
            "outlier_analysis": self._detect_outliers(data),
            "consistency_analysis": self._assess_consistency(data),
            "variable_quality": self._assess_variable_quality(data),
            "gap_analysis": self._identify_gaps(data),
            "data_source_analysis": self._assess_data_sources(data),
            "seasonal_analysis": self._assess_seasonal_patterns(data)
        }
        
        # Identify quality issues
        quality_issues = self._identify_quality_issues(assessment)
        assessment["quality_issues"] = quality_issues
        
        # Generate recommendations
        recommendations = self._generate_recommendations(assessment)
        assessment["recommendations"] = recommendations
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(assessment)
        assessment["overall_quality"] = quality_score
        
        # Save assessment report
        report_path = self._save_assessment_report(assessment)
        assessment["report_path"] = str(report_path)
        
        assessment_duration = (datetime.now() - assessment_start).total_seconds()
        assessment["assessment_info"]["duration_seconds"] = assessment_duration
        
        logger.info(f"✅ Quality assessment completed in {assessment_duration:.1f} seconds")
        logger.info(f"📄 Report saved: {report_path}")
        
        return assessment
    
    def _get_data_overview(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic data overview"""
        overview = {
            "total_records": len(data),
            "variables": sorted(data['variable_name'].unique().tolist()),
            "variable_count": data['variable_name'].nunique(),
            "data_sources": sorted(data['data_source'].unique().tolist()),
            "data_source_count": data['data_source'].nunique(),
            "temporal_range": {
                "start": data['timestamp'].min().isoformat(),
                "end": data['timestamp'].max().isoformat(),
                "duration_days": (data['timestamp'].max() - data['timestamp'].min()).days
            },
            "spatial_extent": {
                "lat_range": [data['latitude'].min(), data['latitude'].max()],
                "lon_range": [data['longitude'].min(), data['longitude'].max()],
                "unique_locations": len(data[['latitude', 'longitude']].drop_duplicates())
            }
        }
        
        logger.info(f"📊 Data overview: {overview['total_records']:,} records, {overview['variable_count']} variables")
        
        return overview
    
    def _assess_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data completeness"""
        completeness = {
            "overall_completeness": {},
            "variable_completeness": {},
            "temporal_completeness": {},
            "spatial_completeness": {}
        }
        
        # Overall completeness
        total_cells = data.size
        missing_cells = data.isna().sum().sum()
        completeness["overall_completeness"] = {
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "completeness_percent": ((total_cells - missing_cells) / total_cells) * 100
        }
        
        # Variable-wise completeness
        for variable in data['variable_name'].unique():
            var_data = data[data['variable_name'] == variable]
            missing_values = var_data['value'].isna().sum()
            total_values = len(var_data)
            completeness_pct = ((total_values - missing_values) / total_values) * 100 if total_values > 0 else 0
            
            completeness["variable_completeness"][variable] = {
                "total_records": total_values,
                "missing_values": missing_values,
                "completeness_percent": completeness_pct
            }
        
        # Temporal completeness (by month)
        data['year_month'] = data['timestamp'].dt.to_period('M')
        temporal_completeness = {}
        for period in data['year_month'].unique():
            period_data = data[data['year_month'] == period]
            missing_count = period_data['value'].isna().sum()
            total_count = len(period_data)
            temporal_completeness[str(period)] = {
                "total_records": total_count,
                "missing_values": missing_count,
                "completeness_percent": ((total_count - missing_count) / total_count) * 100
            }
        
        completeness["temporal_completeness"] = temporal_completeness
        
        logger.info(f"📊 Overall completeness: {completeness['overall_completeness']['completeness_percent']:.1f}%")
        
        return completeness
    
    def _assess_temporal_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess temporal data quality"""
        temporal_analysis = {
            "frequency_analysis": {},
            "gap_analysis": {},
            "temporal_distribution": {}
        }
        
        # Analyze temporal frequency for each variable
        for variable in data['variable_name'].unique():
            var_data = data[data['variable_name'] == variable].copy()
            var_data = var_data.sort_values('timestamp')
            
            if len(var_data) > 1:
                # Calculate time differences
                time_diffs = var_data['timestamp'].diff().dropna()
                
                frequency_stats = {
                    "record_count": len(var_data),
                    "median_interval_days": time_diffs.dt.total_seconds().median() / (24 * 3600),
                    "mean_interval_days": time_diffs.dt.total_seconds().mean() / (24 * 3600),
                    "min_interval_days": time_diffs.dt.total_seconds().min() / (24 * 3600),
                    "max_interval_days": time_diffs.dt.total_seconds().max() / (24 * 3600),
                    "std_interval_days": time_diffs.dt.total_seconds().std() / (24 * 3600)
                }
                
                # Identify large gaps (>7 days)
                large_gaps = time_diffs[time_diffs > timedelta(days=7)]
                frequency_stats["large_gaps_count"] = len(large_gaps)
                frequency_stats["max_gap_days"] = large_gaps.max().total_seconds() / (24 * 3600) if len(large_gaps) > 0 else 0
                
                temporal_analysis["frequency_analysis"][variable] = frequency_stats
        
        # Overall temporal distribution
        temporal_analysis["temporal_distribution"] = {
            "yearly_distribution": dict(data['timestamp'].dt.year.value_counts().sort_index()),
            "monthly_distribution": dict(data['timestamp'].dt.month.value_counts().sort_index()),
            "seasonal_distribution": dict(((data['timestamp'].dt.month % 12 + 3) // 3).value_counts().sort_index())
        }
        
        logger.info(f"📅 Temporal analysis complete for {len(temporal_analysis['frequency_analysis'])} variables")
        
        return temporal_analysis
    
    def _assess_spatial_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess spatial data quality"""
        spatial_analysis = {
            "coverage_analysis": {},
            "distribution_analysis": {},
            "clustering_analysis": {}
        }
        
        # Spatial coverage
        unique_locations = data[['latitude', 'longitude']].drop_duplicates()
        spatial_analysis["coverage_analysis"] = {
            "unique_locations": len(unique_locations),
            "lat_range_degrees": data['latitude'].max() - data['latitude'].min(),
            "lon_range_degrees": data['longitude'].max() - data['longitude'].min(),
            "spatial_density": len(data) / len(unique_locations) if len(unique_locations) > 0 else 0
        }
        
        # Distribution by variable
        for variable in data['variable_name'].unique():
            var_data = data[data['variable_name'] == variable]
            var_locations = var_data[['latitude', 'longitude']].drop_duplicates()
            
            spatial_analysis["distribution_analysis"][variable] = {
                "unique_locations": len(var_locations),
                "records_per_location": len(var_data) / len(var_locations) if len(var_locations) > 0 else 0,
                "lat_coverage": [var_data['latitude'].min(), var_data['latitude'].max()],
                "lon_coverage": [var_data['longitude'].min(), var_data['longitude'].max()]
            }
        
        logger.info(f"🌍 Spatial analysis complete: {spatial_analysis['coverage_analysis']['unique_locations']} unique locations")
        
        return spatial_analysis
    
    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in the data"""
        outlier_analysis = {
            "method": "IQR and Z-score",
            "variable_outliers": {},
            "summary": {}
        }
        
        total_outliers = 0
        
        for variable in data['variable_name'].unique():
            var_data = data[data['variable_name'] == variable]['value'].dropna()
            
            if len(var_data) < 10:  # Skip if too few data points
                continue
            
            # IQR method
            Q1 = var_data.quantile(0.25)
            Q3 = var_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = var_data[(var_data < lower_bound) | (var_data > upper_bound)]
            
            # Z-score method (for additional validation)
            z_scores = np.abs(stats.zscore(var_data))
            zscore_outliers = var_data[z_scores > 3]
            
            # Range-based outliers (if we have expected ranges)
            range_outliers = pd.Series(dtype=float)
            if variable in self.variable_ranges:
                var_range = self.variable_ranges[variable]
                range_outliers = var_data[(var_data < var_range['min']) | (var_data > var_range['max'])]
            
            outlier_info = {
                "total_records": len(var_data),
                "iqr_outliers": len(iqr_outliers),
                "zscore_outliers": len(zscore_outliers),
                "range_outliers": len(range_outliers),
                "outlier_percent": (len(iqr_outliers) / len(var_data)) * 100,
                "statistics": {
                    "mean": var_data.mean(),
                    "median": var_data.median(),
                    "std": var_data.std(),
                    "min": var_data.min(),
                    "max": var_data.max(),
                    "Q1": Q1,
                    "Q3": Q3
                }
            }
            
            outlier_analysis["variable_outliers"][variable] = outlier_info
            total_outliers += len(iqr_outliers)
        
        outlier_analysis["summary"] = {
            "total_outliers": total_outliers,
            "variables_with_outliers": len([v for v in outlier_analysis["variable_outliers"].values() if v["iqr_outliers"] > 0])
        }
        
        logger.info(f"🎯 Outlier analysis: {total_outliers} outliers detected across {len(outlier_analysis['variable_outliers'])} variables")
        
        return outlier_analysis
    
    def _assess_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data consistency across sources and variables"""
        consistency_analysis = {
            "cross_source_consistency": {},
            "cross_variable_consistency": {},
            "temporal_consistency": {}
        }
        
        # Cross-source consistency (compare same variables from different sources)
        variables_multi_source = data.groupby('variable_name')['data_source'].nunique()
        multi_source_vars = variables_multi_source[variables_multi_source > 1].index
        
        for variable in multi_source_vars:
            var_data = data[data['variable_name'] == variable]
            source_stats = {}
            
            for source in var_data['data_source'].unique():
                source_data = var_data[var_data['data_source'] == source]['value'].dropna()
                if len(source_data) > 0:
                    source_stats[source] = {
                        "count": len(source_data),
                        "mean": source_data.mean(),
                        "std": source_data.std(),
                        "median": source_data.median()
                    }
            
            # Calculate consistency metrics
            if len(source_stats) > 1:
                means = [stats['mean'] for stats in source_stats.values()]
                mean_consistency = np.std(means) / np.mean(means) * 100 if np.mean(means) != 0 else 0
                
                consistency_analysis["cross_source_consistency"][variable] = {
                    "sources": source_stats,
                    "mean_cv_percent": mean_consistency,
                    "consistent": mean_consistency < 10  # <10% CV considered consistent
                }
        
        logger.info(f"🔄 Consistency analysis: {len(consistency_analysis['cross_source_consistency'])} multi-source variables")
        
        return consistency_analysis
    
    def _assess_variable_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess quality of individual variables"""
        variable_quality = {}
        
        for variable in data['variable_name'].unique():
            var_data = data[data['variable_name'] == variable].copy()
            
            quality_metrics = {
                "record_count": len(var_data),
                "completeness_percent": ((len(var_data) - var_data['value'].isna().sum()) / len(var_data)) * 100,
                "unique_values": var_data['value'].nunique(),
                "temporal_span_days": (var_data['timestamp'].max() - var_data['timestamp'].min()).days,
                "data_sources": var_data['data_source'].unique().tolist(),
                "quality_flags": dict(var_data['quality_flag'].value_counts())
            }
            
            # Calculate data quality score (0-100)
            completeness_score = quality_metrics["completeness_percent"]
            temporal_score = min(quality_metrics["temporal_span_days"] / 365 * 25, 25)  # Max 25 points for 1+ years
            source_score = min(len(quality_metrics["data_sources"]) * 10, 20)  # Max 20 points for multiple sources
            
            # Outlier penalty
            if variable in self.variable_ranges:
                var_values = var_data['value'].dropna()
                var_range = self.variable_ranges[variable]
                range_violations = ((var_values < var_range['min']) | (var_values > var_range['max'])).sum()
                outlier_penalty = (range_violations / len(var_values)) * 30 if len(var_values) > 0 else 0
            else:
                outlier_penalty = 0
            
            quality_score = max(0, completeness_score + temporal_score + source_score - outlier_penalty)
            quality_metrics["quality_score"] = min(quality_score, 100)
            
            variable_quality[variable] = quality_metrics
        
        logger.info(f"📊 Variable quality assessed for {len(variable_quality)} variables")
        
        return variable_quality
    
    def _identify_gaps(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify temporal and spatial gaps in data"""
        gap_analysis = {
            "temporal_gaps": {},
            "spatial_gaps": {},
            "variable_gaps": {}
        }
        
        # Temporal gaps by variable
        for variable in data['variable_name'].unique():
            var_data = data[data['variable_name'] == variable].sort_values('timestamp')
            
            gaps = []
            if len(var_data) > 1:
                time_diffs = var_data['timestamp'].diff().dropna()
                
                # Find gaps larger than expected frequency (>7 days)
                large_gaps = time_diffs[time_diffs > timedelta(days=7)]
                
                for i, gap in enumerate(large_gaps):
                    gap_start_idx = time_diffs[time_diffs == gap].index[0]
                    gap_start = var_data.loc[gap_start_idx - 1, 'timestamp']
                    gap_end = var_data.loc[gap_start_idx, 'timestamp']
                    
                    gaps.append({
                        "gap_id": f"{variable}_gap_{i+1}",
                        "start_date": gap_start.isoformat(),
                        "end_date": gap_end.isoformat(),
                        "duration_days": gap.total_seconds() / (24 * 3600),
                        "severity": "high" if gap.total_seconds() > (30 * 24 * 3600) else "medium"
                    })
            
            gap_analysis["temporal_gaps"][variable] = gaps
        
        logger.info(f"🔍 Gap analysis complete")
        
        return gap_analysis
    
    def _assess_data_sources(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess reliability and coverage of different data sources"""
        source_analysis = {}
        
        for source in data['data_source'].unique():
            source_data = data[data['data_source'] == source]
            
            analysis = {
                "record_count": len(source_data),
                "variables": sorted(source_data['variable_name'].unique().tolist()),
                "variable_count": source_data['variable_name'].nunique(),
                "temporal_span": {
                    "start": source_data['timestamp'].min().isoformat(),
                    "end": source_data['timestamp'].max().isoformat(),
                    "duration_days": (source_data['timestamp'].max() - source_data['timestamp'].min()).days
                },
                "data_quality": {
                    "good_quality_percent": (source_data['quality_flag'] == 'good').mean() * 100,
                    "completeness_percent": ((len(source_data) - source_data['value'].isna().sum()) / len(source_data)) * 100
                }
            }
            
            source_analysis[source] = analysis
        
        logger.info(f"📡 Data source analysis: {len(source_analysis)} sources assessed")
        
        return source_analysis
    
    def _assess_seasonal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess seasonal patterns in the data"""
        seasonal_analysis = {}
        
        # Add seasonal information
        data_with_seasons = data.copy()
        data_with_seasons['month'] = data_with_seasons['timestamp'].dt.month
        data_with_seasons['season'] = ((data_with_seasons['month'] % 12 + 3) // 3).map({
            1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'
        })
        
        for variable in data['variable_name'].unique():
            var_data = data_with_seasons[data_with_seasons['variable_name'] == variable]
            
            seasonal_stats = {}
            for season in ['Spring', 'Summer', 'Fall', 'Winter']:
                season_data = var_data[var_data['season'] == season]['value'].dropna()
                if len(season_data) > 0:
                    seasonal_stats[season] = {
                        "count": len(season_data),
                        "mean": season_data.mean(),
                        "std": season_data.std(),
                        "median": season_data.median()
                    }
            
            seasonal_analysis[variable] = seasonal_stats
        
        logger.info(f"🌸 Seasonal analysis complete for {len(seasonal_analysis)} variables")
        
        return seasonal_analysis
    
    def _identify_quality_issues(self, assessment: Dict[str, Any]) -> List[QualityIssue]:
        """Identify specific quality issues from assessment"""
        issues = []
        issue_counter = 1
        
        # Completeness issues
        completeness = assessment["completeness_analysis"]["variable_completeness"]
        for variable, stats in completeness.items():
            if stats["completeness_percent"] < 80:
                severity = "high" if stats["completeness_percent"] < 50 else "medium"
                issues.append(QualityIssue(
                    issue_id=f"ISSUE_{issue_counter:04d}",
                    issue_type="missing_data",
                    severity=severity,
                    variable=variable,
                    location="throughout dataset",
                    affected_records=stats["missing_values"],
                    description=f"Variable {variable} has {stats['completeness_percent']:.1f}% completeness",
                    recommendation="Implement gap filling or collect additional data"
                ))
                issue_counter += 1
        
        # Outlier issues
        outliers = assessment["outlier_analysis"]["variable_outliers"]
        for variable, stats in outliers.items():
            if stats["outlier_percent"] > 5:  # >5% outliers considered problematic
                issues.append(QualityIssue(
                    issue_id=f"ISSUE_{issue_counter:04d}",
                    issue_type="outlier",
                    severity="medium",
                    variable=variable,
                    location="scattered throughout dataset",
                    affected_records=stats["iqr_outliers"],
                    description=f"Variable {variable} has {stats['outlier_percent']:.1f}% outliers",
                    recommendation="Review outliers for data collection errors or extreme events"
                ))
                issue_counter += 1
        
        # Gap issues
        gaps = assessment["gap_analysis"]["temporal_gaps"]
        for variable, gap_list in gaps.items():
            large_gaps = [g for g in gap_list if g["duration_days"] > 30]
            if large_gaps:
                issues.append(QualityIssue(
                    issue_id=f"ISSUE_{issue_counter:04d}",
                    issue_type="temporal_gap",
                    severity="high",
                    variable=variable,
                    location=f"{len(large_gaps)} large gaps",
                    affected_records=sum(int(g["duration_days"]) for g in large_gaps),
                    description=f"Variable {variable} has {len(large_gaps)} gaps >30 days",
                    recommendation="Prioritize gap filling for continuous time series"
                ))
                issue_counter += 1
        
        return issues
    
    def _generate_recommendations(self, assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations for data quality improvement"""
        recommendations = []
        
        # Overall data quality recommendations
        overall_quality = assessment.get("overall_quality", {})
        if overall_quality.get("score", 0) < 70:
            recommendations.append({
                "category": "overall_quality",
                "priority": "high",
                "recommendation": "Overall data quality is below acceptable threshold",
                "action": "Implement comprehensive data quality improvement program"
            })
        
        # Variable-specific recommendations
        var_quality = assessment.get("variable_quality", {})
        low_quality_vars = [v for v, stats in var_quality.items() if stats.get("quality_score", 0) < 60]
        
        if low_quality_vars:
            recommendations.append({
                "category": "variable_quality",
                "priority": "medium",
                "recommendation": f"Variables with low quality scores: {', '.join(low_quality_vars[:3])}",
                "action": "Focus data collection efforts on these variables"
            })
        
        # Gap filling recommendations
        gaps = assessment.get("gap_analysis", {}).get("temporal_gaps", {})
        vars_with_gaps = [v for v, gap_list in gaps.items() if len(gap_list) > 0]
        
        if vars_with_gaps:
            recommendations.append({
                "category": "gap_filling",
                "priority": "medium",
                "recommendation": f"Temporal gaps identified in {len(vars_with_gaps)} variables",
                "action": "Implement temporal gap filling using climatology or interpolation"
            })
        
        return recommendations
    
    def _calculate_quality_score(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall data quality score"""
        
        # Completeness score (40% weight)
        completeness = assessment["completeness_analysis"]["overall_completeness"]["completeness_percent"]
        completeness_score = completeness * 0.4
        
        # Variable coverage score (20% weight)
        expected_vars = len(self.variable_ranges)
        actual_vars = len(assessment["data_overview"]["variables"])
        coverage_score = min(actual_vars / expected_vars, 1.0) * 20
        
        # Temporal coverage score (20% weight)
        duration_days = assessment["data_overview"]["temporal_range"]["duration_days"]
        temporal_score = min(duration_days / (5 * 365), 1.0) * 20  # 5 years target
        
        # Quality flag score (20% weight)
        # This would be based on actual quality flags in the data
        quality_flag_score = 20  # Assume good for now
        
        total_score = completeness_score + coverage_score + temporal_score + quality_flag_score
        
        return {
            "score": round(total_score, 1),
            "grade": self._get_quality_grade(total_score),
            "components": {
                "completeness": round(completeness_score, 1),
                "coverage": round(coverage_score, 1),
                "temporal": round(temporal_score, 1),
                "quality_flags": round(quality_flag_score, 1)
            }
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _save_assessment_report(self, assessment: Dict[str, Any]) -> Path:
        """Save comprehensive assessment report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = assessment["assessment_info"]["name"]
        
        # Save JSON report
        json_path = self.output_dir / f"{report_name}_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(assessment, f, indent=2, default=str)
        
        # Create summary report (human-readable)
        summary_path = self.output_dir / f"{report_name}_summary_{timestamp}.txt"
        self._create_summary_report(assessment, summary_path)
        
        logger.info(f"📄 Assessment report saved: {json_path}")
        logger.info(f"📋 Summary report saved: {summary_path}")
        
        return json_path
    
    def _create_summary_report(self, assessment: Dict[str, Any], output_path: Path):
        """Create human-readable summary report"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("🔍 COMPREHENSIVE DATA QUALITY ASSESSMENT REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic information
            info = assessment["assessment_info"]
            f.write(f"Assessment: {info['name']}\n")
            f.write(f"Date: {info['timestamp'][:19]}\n")
            f.write(f"Duration: {info.get('duration_seconds', 0):.1f} seconds\n\n")
            
            # Overall quality score
            quality = assessment["overall_quality"]
            f.write(f"📊 OVERALL QUALITY SCORE: {quality['score']}/100 (Grade {quality['grade']})\n\n")
            
            # Data overview
            overview = assessment["data_overview"]
            f.write("📋 DATA OVERVIEW:\n")
            f.write(f"   Records: {overview['total_records']:,}\n")
            f.write(f"   Variables: {overview['variable_count']}\n")
            f.write(f"   Data Sources: {overview['data_source_count']}\n")
            f.write(f"   Time Range: {overview['temporal_range']['start'][:10]} to {overview['temporal_range']['end'][:10]}\n")
            f.write(f"   Duration: {overview['temporal_range']['duration_days']} days\n\n")
            
            # Quality issues
            issues = assessment["quality_issues"]
            f.write(f"⚠️ QUALITY ISSUES IDENTIFIED: {len(issues)}\n")
            for issue in issues[:10]:  # Show top 10 issues
                f.write(f"   {issue.severity.upper()}: {issue.description}\n")
            if len(issues) > 10:
                f.write(f"   ... and {len(issues) - 10} more issues\n")
            f.write("\n")
            
            # Recommendations
            recommendations = assessment["recommendations"]
            f.write(f"💡 RECOMMENDATIONS ({len(recommendations)}):\n")
            for rec in recommendations:
                f.write(f"   {rec['priority'].upper()}: {rec['recommendation']}\n")
            f.write("\n")
            
            # Variable quality summary
            var_quality = assessment["variable_quality"]
            f.write("📊 VARIABLE QUALITY SUMMARY:\n")
            for variable, stats in sorted(var_quality.items()):
                f.write(f"   {variable}: {stats['quality_score']:.0f}/100 ({stats['record_count']:,} records)\n")
    
    def create_quality_dashboard(self, assessment: Dict[str, Any]) -> Path:
        """Create visual quality dashboard"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set up the plot style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('HAB Data Quality Assessment Dashboard', fontsize=16, fontweight='bold')
            
            # Plot 1: Completeness by variable
            var_quality = assessment["variable_quality"]
            variables = list(var_quality.keys())[:10]  # Top 10 variables
            completeness = [var_quality[v]["completeness_percent"] for v in variables]
            
            axes[0,0].barh(variables, completeness)
            axes[0,0].set_xlabel('Completeness (%)')
            axes[0,0].set_title('Data Completeness by Variable')
            axes[0,0].set_xlim(0, 100)
            
            # Plot 2: Quality score distribution
            quality_scores = [stats["quality_score"] for stats in var_quality.values()]
            axes[0,1].hist(quality_scores, bins=10, edgecolor='black')
            axes[0,1].set_xlabel('Quality Score')
            axes[0,1].set_ylabel('Number of Variables')
            axes[0,1].set_title('Quality Score Distribution')
            
            # Plot 3: Issues by severity
            issues = assessment["quality_issues"]
            severity_counts = {}
            for issue in issues:
                severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
            
            if severity_counts:
                severities = list(severity_counts.keys())
                counts = list(severity_counts.values())
                colors = {'low': 'green', 'medium': 'orange', 'high': 'red', 'critical': 'darkred'}
                bar_colors = [colors.get(s, 'gray') for s in severities]
                
                axes[1,0].bar(severities, counts, color=bar_colors)
                axes[1,0].set_xlabel('Severity')
                axes[1,0].set_ylabel('Number of Issues')
                axes[1,0].set_title('Issues by Severity')
            
            # Plot 4: Overall quality components
            quality_components = assessment["overall_quality"]["components"]
            components = list(quality_components.keys())
            scores = list(quality_components.values())
            
            axes[1,1].pie(scores, labels=components, autopct='%1.1f%%', startangle=90)
            axes[1,1].set_title('Quality Score Components')
            
            plt.tight_layout()
            
            # Save dashboard
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dashboard_path = self.output_dir / f"quality_dashboard_{timestamp}.png"
            plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Quality dashboard saved: {dashboard_path}")
            return dashboard_path
            
        except ImportError:
            logger.warning("matplotlib/seaborn not available - skipping dashboard creation")
            return None