"""
HAB Forecasting System for Seto Inland Sea
Source code package

This package contains all the source code for the HAB forecasting system,
including data collection, processing, modeling, and evaluation modules.
"""

__version__ = "1.0.0"
__author__ = "HAB Forecasting Team"
__description__ = "Harmful Algal Bloom forecasting system for Seto Inland Sea"

# Import main modules
from . import data_collection
from . import data_processing
from . import feature_engineering
from . import model
from . import evaluation
from . import visualization 