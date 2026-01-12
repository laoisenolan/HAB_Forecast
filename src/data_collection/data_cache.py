"""
Simple Data Cache for HAB Forecasting System

This module provides a lightweight caching system for downloaded data,
optimized for 5-year datasets and local development workflows.
"""

import os
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, Union
import pickle
import xarray as xr
import pandas as pd

logger = logging.getLogger(__name__)

class DataCache:
    """
    Simple file-based cache for HAB forecasting data
    
    Features:
    - Local file storage (no external dependencies)
    - Automatic cache key generation from parameters
    - TTL (Time To Live) for cache expiration
    - Memory-efficient for 5-year datasets
    - Easy cache management and cleanup
    """
    
    def __init__(self, cache_dir: str = "data/cache", ttl_days: int = 30):
        """
        Initialize the data cache
        
        Args:
            cache_dir: Directory to store cached files
            ttl_days: Number of days before cache expires
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days
        
        # Create subdirectories
        (self.cache_dir / "datasets").mkdir(exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)
        
        logger.info(f"Data cache initialized at: {self.cache_dir}")
    
    def _generate_cache_key(self, 
                           data_source: str, 
                           variable: str, 
                           start_date: str, 
                           end_date: str,
                           **kwargs) -> str:
        """
        Generate a unique cache key from download parameters
        
        Args:
            data_source: Source of data (copernicus, gee, erddap)
            variable: Variable name
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters
            
        Returns:
            Unique cache key string
        """
        # Create parameter string
        params = {
            'data_source': data_source,
            'variable': variable,
            'start_date': start_date,
            'end_date': end_date,
            **kwargs
        }
        
        # Sort parameters for consistent keys
        param_str = json.dumps(params, sort_keys=True)
        
        # Generate hash
        cache_key = hashlib.md5(param_str.encode()).hexdigest()
        return cache_key
    
    def _get_cache_path(self, cache_key: str, data_type: str = "dataset") -> Path:
        """
        Get file path for cached data
        
        Args:
            cache_key: Cache key
            data_type: Type of cached data (dataset, metadata)
            
        Returns:
            Path to cached file
        """
        if data_type == "dataset":
            return self.cache_dir / "datasets" / f"{cache_key}.nc"
        elif data_type == "metadata":
            return self.cache_dir / "metadata" / f"{cache_key}.json"
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """
        Check if cached file is still valid (not expired)
        
        Args:
            cache_path: Path to cached file
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_path.exists():
            return False
        
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age.days < self.ttl_days
    
    def get(self, 
            data_source: str, 
            variable: str, 
            start_date: str, 
            end_date: str,
            **kwargs) -> Optional[Union[xr.Dataset, pd.DataFrame]]:
        """
        Get data from cache if available and valid
        
        Args:
            data_source: Source of data
            variable: Variable name
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters
            
        Returns:
            Cached data if available, None otherwise
        """
        cache_key = self._generate_cache_key(data_source, variable, start_date, end_date, **kwargs)
        cache_path = self._get_cache_path(cache_key, "dataset")
        
        if self._is_cache_valid(cache_path):
            try:
                logger.info(f"Loading from cache: {cache_key}")
                
                # Load dataset
                if cache_path.suffix == '.nc':
                    data = xr.open_dataset(cache_path)
                else:
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                
                # Load metadata
                metadata_path = self._get_cache_path(cache_key, "metadata")
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    logger.info(f"Cache hit: {metadata.get('size_mb', 0):.2f} MB")
                
                return data
                
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
                return None
        else:
            logger.info(f"Cache miss or expired: {cache_key}")
            return None
    
    def set(self, 
            data: Union[xr.Dataset, pd.DataFrame],
            data_source: str, 
            variable: str, 
            start_date: str, 
            end_date: str,
            **kwargs) -> bool:
        """
        Store data in cache
        
        Args:
            data: Data to cache
            data_source: Source of data
            variable: Variable name
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters
            
        Returns:
            True if successfully cached, False otherwise
        """
        cache_key = self._generate_cache_key(data_source, variable, start_date, end_date, **kwargs)
        
        try:
            # Save dataset
            cache_path = self._get_cache_path(cache_key, "dataset")
            
            if isinstance(data, xr.Dataset):
                # Save as NetCDF for xarray datasets
                data.to_netcdf(cache_path, engine='netcdf4')
                file_size = cache_path.stat().st_size / (1024 * 1024)  # MB
            else:
                # Save as pickle for other data types
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
                file_size = cache_path.stat().st_size / (1024 * 1024)  # MB
            
            # Save metadata
            metadata = {
                'data_source': data_source,
                'variable': variable,
                'start_date': start_date,
                'end_date': end_date,
                'cached_at': datetime.now().isoformat(),
                'size_mb': file_size,
                'ttl_days': self.ttl_days,
                'parameters': kwargs
            }
            
            metadata_path = self._get_cache_path(cache_key, "metadata")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Cached data: {cache_key} ({file_size:.2f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")
            return False
    
    def clear(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear expired or old cache files
        
        Args:
            older_than_days: Clear files older than this many days (None = use TTL)
            
        Returns:
            Number of files cleared
        """
        if older_than_days is None:
            older_than_days = self.ttl_days
        
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        cleared_count = 0
        
        # Clear dataset files
        for cache_file in (self.cache_dir / "datasets").glob("*.nc"):
            if datetime.fromtimestamp(cache_file.stat().st_mtime) < cutoff_time:
                cache_file.unlink()
                cleared_count += 1
        
        # Clear pickle files
        for cache_file in (self.cache_dir / "datasets").glob("*.pkl"):
            if datetime.fromtimestamp(cache_file.stat().st_mtime) < cutoff_time:
                cache_file.unlink()
                cleared_count += 1
        
        # Clear metadata files
        for metadata_file in (self.cache_dir / "metadata").glob("*.json"):
            if datetime.fromtimestamp(metadata_file.stat().st_mtime) < cutoff_time:
                metadata_file.unlink()
        
        logger.info(f"Cleared {cleared_count} expired cache files")
        return cleared_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        dataset_files = list((self.cache_dir / "datasets").glob("*"))
        metadata_files = list((self.cache_dir / "metadata").glob("*"))
        
        total_size = sum(f.stat().st_size for f in dataset_files if f.is_file())
        total_size_mb = total_size / (1024 * 1024)
        
        # Count valid vs expired files
        valid_count = 0
        expired_count = 0
        
        for file_path in dataset_files:
            if self._is_cache_valid(file_path):
                valid_count += 1
            else:
                expired_count += 1
        
        return {
            'total_files': len(dataset_files),
            'valid_files': valid_count,
            'expired_files': expired_count,
            'total_size_mb': total_size_mb,
            'cache_dir': str(self.cache_dir),
            'ttl_days': self.ttl_days
        }
    
    def list_cached_variables(self) -> Dict[str, list]:
        """
        List all cached variables by data source
        
        Returns:
            Dictionary mapping data sources to lists of cached variables
        """
        cached_variables = {}
        
        for metadata_file in (self.cache_dir / "metadata").glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                data_source = metadata.get('data_source', 'unknown')
                variable = metadata.get('variable', 'unknown')
                
                if data_source not in cached_variables:
                    cached_variables[data_source] = []
                
                if variable not in cached_variables[data_source]:
                    cached_variables[data_source].append(variable)
                    
            except Exception as e:
                logger.warning(f"Failed to read metadata file {metadata_file}: {e}")
        
        return cached_variables


# Convenience function for easy caching
def cached_download(download_func, 
                   data_source: str,
                   variable: str,
                   start_date: str,
                   end_date: str,
                   cache_dir: str = "data/cache",
                   **kwargs):
    """
    Convenience function to download data with caching
    
    Args:
        download_func: Function that downloads data
        data_source: Source of data
        variable: Variable name
        start_date: Start date
        end_date: End date
        cache_dir: Cache directory
        **kwargs: Additional parameters for download function
        
    Returns:
        Downloaded data (from cache or fresh download)
    """
    cache = DataCache(cache_dir)
    
    # Try to get from cache first
    cached_data = cache.get(data_source, variable, start_date, end_date, **kwargs)
    if cached_data is not None:
        return cached_data
    
    # Download fresh data
    logger.info(f"Downloading fresh data: {variable} from {start_date} to {end_date}")
    data = download_func(start_date=start_date, end_date=end_date, **kwargs)
    
    # Cache the result
    if data is not None:
        cache.set(data, data_source, variable, start_date, end_date, **kwargs)
    
    return data 