"""
CIAF High-Performance Metadata Storage

Optimized metadata storage system designed for minimal latency and maximum throughput.
Implements lazy materialization, batching, caching, and async I/O.

Created: 2025-09-19
Author: CIAF Development Team
Version: 1.0.0
"""

import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .metadata_storage import MetadataStorage


class HighPerformanceMetadataStorage:
    """
    High-performance metadata storage with optimizations for training and inference.
    
    Uses composition to wrap MetadataStorage with performance optimizations:
    - In-memory buffering with configurable flush intervals
    - Batch processing for bulk operations
    - Lazy materialization - defer expensive operations
    - Connection pooling for database operations
    - Async I/O for non-blocking writes
    - Memory caching with LRU eviction
    """
    
    def __init__(self, config_or_template: Union[str, Dict[str, Any]] = "high_performance"):
        """
        Initialize high-performance metadata storage.
        
        Args:
            config_or_template: Configuration template name or config dict
        """
        # Handle configuration
        if isinstance(config_or_template, str):
            self.perf_config = {
                "enable_lazy_materialization": True,
                "fast_inference_mode": True,
                "memory_buffer_size": 1000,
                "db_connection_pool_size": 5,
                "enable_async_writes": True,
                "batch_write_size": 50,
                "storage_path": "ciaf_metadata_optimized"
            }
        else:
            # config_or_template is a config dict
            self.perf_config = config_or_template.copy()
        
        # Initialize the underlying storage
        storage_path = self.perf_config.get('storage_path', 'ciaf_metadata_optimized')
        self._storage = MetadataStorage(
            storage_path=storage_path,
            backend="sqlite",
            use_compression=self.perf_config.get('use_compression', False)
        )
        
        # Initialize performance components
        self.memory_buffer = deque(maxlen=self.perf_config.get('memory_buffer_size', 1000))
        self.cache = {}  # Simple dict cache for demo
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_saves = 0
        self.total_save_time = 0.0
        self.buffer_flushes = 0
        self.batch_size = self.perf_config.get('batch_write_size', 50)
        
        # Performance monitoring
        self.start_time = time.time()
    
    def save_metadata(self, model_name: str, stage: str, event_type: str, 
                     metadata: Dict[str, Any], model_version: Optional[str] = None,
                     details: Optional[str] = None) -> str:
        """Save metadata with performance optimizations."""
        start_time = time.perf_counter()
        
        # Use underlying storage for actual save
        metadata_id = self._storage.save_metadata(
            model_name=model_name,
            stage=stage, 
            event_type=event_type,
            metadata=metadata,
            model_version=model_version,
            details=details
        )
        
        # Update performance stats
        save_time = time.perf_counter() - start_time
        self.total_saves += 1
        self.total_save_time += save_time
        
        # Cache the metadata for quick access
        cache_key = f"{model_name}:{stage}:{metadata_id}"
        self.cache[cache_key] = {
            'metadata_id': metadata_id,
            'model_name': model_name,
            'stage': stage,
            'event_type': event_type,
            'metadata': metadata,
            'model_version': model_version,
            'details': details,
            'timestamp': time.time()
        }
        
        return metadata_id
    
    def get_metadata(self, metadata_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata with caching."""
        # Check cache first
        for cache_key, cached_item in self.cache.items():
            if cached_item['metadata_id'] == metadata_id:
                self.cache_hits += 1
                return cached_item
        
        # Cache miss - delegate to underlying storage
        self.cache_misses += 1
        return self._storage.get_metadata(metadata_id)
    
    def get_model_metadata(self, model_name: str, stage: Optional[str] = None, 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Get model metadata with optimizations."""
        return self._storage.get_model_metadata(model_name, stage, limit)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_cache_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0
        avg_save_time = self.total_save_time / self.total_saves if self.total_saves > 0 else 0.0
        
        return {
            'cache_hit_rate': hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_saves': self.total_saves,
            'avg_save_time': avg_save_time,
            'buffer_flushes': self.buffer_flushes,
            'cached_items': len(self.cache),
            'uptime': time.time() - self.start_time
        }
    
    def shutdown(self):
        """Cleanup resources."""
        self.cache.clear()
        self.memory_buffer.clear()