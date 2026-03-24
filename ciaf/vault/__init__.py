"""
CIAF Vault - Centralized Storage Backend

The vault module provides centralized storage for:
- Metadata (training, inference, compliance)
- Receipts and audit trails
- Database backends (SQLite, PostgreSQL)
- Compressed and optimized storage

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

from .metadata_storage import (
    MetadataStorage,
    get_metadata_storage,
    save_pipeline_metadata,
    get_pipeline_trace
)

from .metadata_config import (
    MetadataConfig,
    get_metadata_config,
    load_config_from_file,
    create_config_template,
    create_deferred_lcm_config,
    create_high_performance_config,
    create_compliance_first_config,
    create_balanced_config
)

from .metadata_integration import (
    MetadataCapture,
    ModelMetadataManager,
    ComplianceTracker,
    capture_metadata,
    create_model_manager,
    create_compliance_tracker,
    quick_log
)

from .metadata_storage_compressed import CompressedMetadataStorage
from .metadata_storage_optimized import HighPerformanceMetadataStorage

__all__ = [
    # Core storage
    "MetadataStorage",
    "get_metadata_storage",
    "save_pipeline_metadata",
    "get_pipeline_trace",

    # Configuration
    "MetadataConfig",
    "get_metadata_config",
    "load_config_from_file",
    "create_config_template",
    "create_deferred_lcm_config",
    "create_high_performance_config",
    "create_compliance_first_config",
    "create_balanced_config",

    # Integration
    "MetadataCapture",
    "ModelMetadataManager",
    "ComplianceTracker",
    "capture_metadata",
    "create_model_manager",
    "create_compliance_tracker",
    "quick_log",

    # Specialized storage
    "CompressedMetadataStorage",
    "HighPerformanceMetadataStorage",
]
