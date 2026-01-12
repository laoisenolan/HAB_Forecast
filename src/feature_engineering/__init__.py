"""
Feature Engineering Module for HAB Forecasting System

This module handles advanced feature engineering, including comprehensive
spatial, temporal, species-specific, and interaction features for ML pipeline.
"""

# Basic feature engineering components (if they exist)
try:
    from .anomaly_calculator import AnomalyCalculator
except ImportError:
    AnomalyCalculator = None

try:
    from .lag_features import LagFeatureGenerator
except ImportError:
    LagFeatureGenerator = None

try:
    from .seasonal_features import SeasonalFeatureGenerator  
except ImportError:
    SeasonalFeatureGenerator = None

try:
    from .feature_selector import FeatureSelector
except ImportError:
    FeatureSelector = None

try:
    from .feature_scaler import FeatureScaler
except ImportError:
    FeatureScaler = None

# Advanced feature engineering components
try:
    from .comprehensive_feature_pipeline import ComprehensiveHABFeaturePipeline
except ImportError:
    ComprehensiveHABFeaturePipeline = None

try:
    from .advanced_species_features import AdvancedSpeciesFeatureEngineer
except ImportError:
    AdvancedSpeciesFeatureEngineer = None

try:
    from .advanced_spatial_features import AdvancedSpatialFeatureEngineer
except ImportError:
    AdvancedSpatialFeatureEngineer = None

try:
    from .advanced_temporal_features import AdvancedTemporalFeatureEngineer
except ImportError:
    AdvancedTemporalFeatureEngineer = None

try:
    from .advanced_interaction_features import AdvancedInteractionFeatureEngineer
except ImportError:
    AdvancedInteractionFeatureEngineer = None

try:
    from .advanced_turbidity_features import AdvancedTurbidityFeatureEngineer
except ImportError:
    AdvancedTurbidityFeatureEngineer = None

try:
    from .advanced_biogeochemical_features import AdvancedBiogeochemicalFeatureEngineer
except ImportError:
    AdvancedBiogeochemicalFeatureEngineer = None

__all__ = [
    # Basic components
    'AnomalyCalculator',
    'LagFeatureGenerator', 
    'SeasonalFeatureGenerator',
    'FeatureSelector',
    'FeatureScaler',
    # Advanced components
    'ComprehensiveHABFeaturePipeline',
    'AdvancedSpeciesFeatureEngineer',
    'AdvancedSpatialFeatureEngineer',
    'AdvancedTemporalFeatureEngineer',
    'AdvancedInteractionFeatureEngineer',
    'AdvancedTurbidityFeatureEngineer', 
    'AdvancedBiogeochemicalFeatureEngineer'
]