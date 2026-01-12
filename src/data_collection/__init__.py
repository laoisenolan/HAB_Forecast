"""
Data Collection Module for HAB Forecasting System

This module handles data collection from various sources:
- Copernicus Marine (physics and ocean color data)
- ERDDAP (atmospheric data via PacIOOS)
- Google Earth Engine (climatology computation)
- NOAA CoastWatch (gap-filled fallback data)
"""

from .copernicus_downloader import CopernicusDownloader
from .erddap_downloader import ERDDAPDownloader
from .gee_climatology import GEEClimatology
from .data_orchestrator import DataCollectionOrchestrator
from .quality_control import DataQualityController

__all__ = [
    'CopernicusDownloader',
    'ERDDAPDownloader', 
    'GEEClimatology',
    'DataCollectionOrchestrator',
    'DataQualityController'
] 