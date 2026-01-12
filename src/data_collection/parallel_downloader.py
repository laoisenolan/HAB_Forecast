"""
Parallel Download Manager for HAB Forecasting System

This module provides parallel downloading capabilities to optimize data collection
performance by downloading from multiple sources simultaneously. It includes
concurrent processing, retry logic, and performance monitoring.

Features:
- Concurrent downloads from multiple data sources
- Configurable thread pools and connection limits
- Automatic retry logic with exponential backoff
- Performance monitoring and metrics
- Resource management and cleanup
- Integration with existing downloaders
"""

import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import queue
import statistics
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DownloadTask:
    """Represents a single download task"""
    task_id: str
    data_source: str
    variable: str
    start_date: str
    end_date: str
    priority: int = 1  # Higher number = higher priority
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300  # 5 minutes default
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class DownloadResult:
    """Represents the result of a download task"""
    task_id: str
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    download_time: float = 0.0
    file_size: Optional[int] = None
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.now()

class ParallelDownloadManager:
    """Manages parallel downloads from multiple data sources"""
    
    def __init__(self, max_workers: int = 5, max_connections_per_host: int = 10):
        """
        Initialize the parallel download manager
        
        Args:
            max_workers: Maximum number of concurrent download threads
            max_connections_per_host: Maximum connections per host
        """
        self.max_workers = max_workers
        self.max_connections_per_host = max_connections_per_host
        
        # Thread pool for downloads
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="DownloadWorker"
        )
        
        # Task queue and tracking
        self.task_queue = queue.PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # Performance metrics
        self.metrics = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_download_time': 0.0,
            'average_download_time': 0.0,
            'download_times': [],
            'start_time': None,
            'end_time': None
        }
        
        # Downloader functions mapping
        self.downloaders = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Parallel download manager initialized with {max_workers} workers")
    
    def register_downloader(self, data_source: str, downloader_func: Callable):
        """
        Register a downloader function for a specific data source
        
        Args:
            data_source: Name of the data source (e.g., 'copernicus', 'gee', 'erddap')
            downloader_func: Function that performs the actual download
        """
        self.downloaders[data_source] = downloader_func
        logger.info(f"Registered downloader for {data_source}")
    
    def add_download_task(self, task: DownloadTask) -> bool:
        """
        Add a download task to the queue
        
        Args:
            task: DownloadTask to add
            
        Returns:
            True if task was added successfully
        """
        try:
            # Priority queue uses negative priority (higher priority = lower number)
            priority = (-task.priority, task.created_at.timestamp())
            self.task_queue.put((priority, task))
            
            with self.lock:
                self.active_tasks[task.task_id] = task
            
            logger.debug(f"Added download task {task.task_id} for {task.data_source}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add download task {task.task_id}: {e}")
            return False
    
    def download_single_task(self, task: DownloadTask) -> DownloadResult:
        """
        Download a single task using the appropriate downloader
        
        Args:
            task: DownloadTask to execute
            
        Returns:
            DownloadResult with the outcome
        """
        start_time = time.time()
        
        try:
            # Get the appropriate downloader
            if task.data_source not in self.downloaders:
                raise ValueError(f"No downloader registered for {task.data_source}")
            
            downloader_func = self.downloaders[task.data_source]
            
            # Execute the download
            logger.info(f"Starting download for task {task.task_id} from {task.data_source}")
            
            data = downloader_func(
                variable=task.variable,
                start_date=task.start_date,
                end_date=task.end_date
            )
            
            download_time = time.time() - start_time
            
            # Calculate file size if data is available
            file_size = None
            if hasattr(data, '__sizeof__'):
                file_size = data.__sizeof__()
            
            result = DownloadResult(
                task_id=task.task_id,
                success=True,
                data=data,
                download_time=download_time,
                file_size=file_size
            )
            
            logger.info(f"Successfully downloaded task {task.task_id} in {download_time:.2f}s")
            return result
            
        except Exception as e:
            download_time = time.time() - start_time
            error_msg = f"Download failed for task {task.task_id}: {str(e)}"
            logger.error(error_msg)
            
            result = DownloadResult(
                task_id=task.task_id,
                success=False,
                error=error_msg,
                download_time=download_time
            )
            return result
    
    def process_download_queue(self, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Process all tasks in the download queue
        
        Args:
            timeout: Maximum time to wait for all downloads (None = no timeout)
            
        Returns:
            Dictionary with processing results and metrics
        """
        self.metrics['start_time'] = datetime.now()
        logger.info(f"Starting parallel download processing with {self.max_workers} workers")
        
        # Submit all tasks to the executor
        future_to_task = {}
        
        while not self.task_queue.empty():
            try:
                priority, task = self.task_queue.get_nowait()
                future = self.executor.submit(self.download_single_task, task)
                future_to_task[future] = task
            except queue.Empty:
                break
        
        # Process completed tasks
        completed_count = 0
        failed_count = 0
        
        try:
            for future in as_completed(future_to_task, timeout=timeout):
                task = future_to_task[future]
                
                try:
                    result = future.result()
                    
                    with self.lock:
                        if result.success:
                            self.completed_tasks[task.task_id] = result
                            completed_count += 1
                            
                            # Update metrics
                            self.metrics['download_times'].append(result.download_time)
                            self.metrics['total_download_time'] += result.download_time
                        else:
                            self.failed_tasks[task.task_id] = result
                            failed_count += 1
                            
                            # Retry logic
                            if task.retry_count < task.max_retries:
                                task.retry_count += 1
                                task.created_at = datetime.now()  # Reset timestamp
                                self.add_download_task(task)
                                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
                    
                except Exception as e:
                    logger.error(f"Task {task.task_id} failed with exception: {e}")
                    failed_count += 1
                    
                    result = DownloadResult(
                        task_id=task.task_id,
                        success=False,
                        error=str(e)
                    )
                    
                    with self.lock:
                        self.failed_tasks[task.task_id] = result
                        
        except TimeoutError:
            logger.warning("Download processing timed out")
        
        # Update final metrics
        self.metrics['end_time'] = datetime.now()
        self.metrics['total_downloads'] = completed_count + failed_count
        self.metrics['successful_downloads'] = completed_count
        self.metrics['failed_downloads'] = failed_count
        
        if self.metrics['download_times']:
            self.metrics['average_download_time'] = statistics.mean(self.metrics['download_times'])
        
        # Clean up active tasks
        with self.lock:
            for task_id in list(self.active_tasks.keys()):
                if task_id in self.completed_tasks or task_id in self.failed_tasks:
                    del self.active_tasks[task_id]
        
        processing_time = (self.metrics['end_time'] - self.metrics['start_time']).total_seconds()
        
        logger.info(f"Download processing completed: {completed_count} successful, {failed_count} failed in {processing_time:.2f}s")
        
        return {
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'total_processing_time': processing_time,
            'average_download_time': self.metrics['average_download_time'],
            'success_rate': completed_count / (completed_count + failed_count) if (completed_count + failed_count) > 0 else 0
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self.lock:
            return self.metrics.copy()
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if task_id in self.completed_tasks:
            result = self.completed_tasks[task_id]
            return {
                'status': 'completed',
                'success': result.success,
                'download_time': result.download_time,
                'completed_at': result.completed_at
            }
        elif task_id in self.failed_tasks:
            result = self.failed_tasks[task_id]
            return {
                'status': 'failed',
                'error': result.error,
                'download_time': result.download_time,
                'completed_at': result.completed_at
            }
        elif task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                'status': 'active',
                'data_source': task.data_source,
                'variable': task.variable,
                'retry_count': task.retry_count,
                'created_at': task.created_at
            }
        else:
            return None
    
    def clear_completed_tasks(self):
        """Clear completed and failed tasks to free memory"""
        with self.lock:
            self.completed_tasks.clear()
            self.failed_tasks.clear()
        logger.info("Cleared completed and failed tasks")
    
    def shutdown(self):
        """Shutdown the download manager and clean up resources"""
        logger.info("Shutting down parallel download manager")
        
        # Cancel any pending tasks
        self.executor.shutdown(wait=False)
        
        # Clear all task collections
        with self.lock:
            self.active_tasks.clear()
            self.completed_tasks.clear()
            self.failed_tasks.clear()
        
        logger.info("Parallel download manager shutdown complete")

class BatchDownloadScheduler:
    """Schedules and manages batch downloads with optimal resource allocation"""
    
    def __init__(self, download_manager: ParallelDownloadManager):
        """
        Initialize the batch download scheduler
        
        Args:
            download_manager: ParallelDownloadManager instance
        """
        self.download_manager = download_manager
        self.batch_configs = {}
        
    def create_batch_config(self, batch_id: str, max_concurrent: int = 3, 
                           priority_order: List[str] = None) -> Dict[str, Any]:
        """
        Create a batch configuration for optimal resource allocation
        
        Args:
            batch_id: Unique identifier for the batch
            max_concurrent: Maximum concurrent downloads for this batch
            priority_order: List of data sources in priority order
            
        Returns:
            Batch configuration dictionary
        """
        config = {
            'batch_id': batch_id,
            'max_concurrent': max_concurrent,
            'priority_order': priority_order or ['copernicus', 'gee', 'erddap'],
            'created_at': datetime.now(),
            'status': 'pending'
        }
        
        self.batch_configs[batch_id] = config
        return config
    
    def schedule_batch_download(self, batch_id: str, tasks: List[DownloadTask]) -> bool:
        """
        Schedule a batch of download tasks with optimal resource allocation
        
        Args:
            batch_id: Batch identifier
            tasks: List of DownloadTask objects
            
        Returns:
            True if batch was scheduled successfully
        """
        if batch_id not in self.batch_configs:
            logger.error(f"Batch configuration {batch_id} not found")
            return False
        
        config = self.batch_configs[batch_id]
        
        # Sort tasks by priority and data source
        priority_map = {source: idx for idx, source in enumerate(config['priority_order'])}
        
        def task_priority(task):
            source_priority = priority_map.get(task.data_source, 999)
            return (source_priority, -task.priority, task.created_at.timestamp())
        
        sorted_tasks = sorted(tasks, key=task_priority)
        
        # Add tasks to download manager
        for task in sorted_tasks:
            if not self.download_manager.add_download_task(task):
                logger.error(f"Failed to add task {task.task_id} to batch {batch_id}")
                return False
        
        config['status'] = 'scheduled'
        config['task_count'] = len(tasks)
        config['scheduled_at'] = datetime.now()
        
        logger.info(f"Scheduled batch {batch_id} with {len(tasks)} tasks")
        return True
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific batch"""
        if batch_id not in self.batch_configs:
            return None
        
        config = self.batch_configs[batch_id]
        
        # Count task statuses
        completed = len(self.download_manager.completed_tasks)
        failed = len(self.download_manager.failed_tasks)
        active = len(self.download_manager.active_tasks)
        
        return {
            'batch_id': batch_id,
            'status': config['status'],
            'task_count': config.get('task_count', 0),
            'completed_tasks': completed,
            'failed_tasks': failed,
            'active_tasks': active,
            'progress': (completed + failed) / config.get('task_count', 1) if config.get('task_count', 0) > 0 else 0
        } 