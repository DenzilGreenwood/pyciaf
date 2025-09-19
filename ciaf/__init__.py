"""
Cognitive Insight Audit Framework (CIAF)

A modular framework for creating verifiable AI training and inference pipelines
with lazy capsule materialization and cryptographic provenance tracking.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

from .anchoring import DatasetAnchor, LazyManager, LazyProvenanceManager
from .api.framework import CIAFFramework
from .core import CryptoUtils, BaseAnchorManager, AnchorManager, MerkleTree
from .inference import InferenceReceipt, ZKEConnections
from .metadata_config import (
    MetadataConfig,
    create_config_template,
    get_metadata_config,
    load_config_from_file,
)
from .metadata_integration import (
    ComplianceTracker,
    MetadataCapture,
    ModelMetadataManager,
    capture_metadata,
    create_compliance_tracker,
    create_model_manager,
    quick_log,
)

# Metadata storage and integration
from .metadata_storage import (
    MetadataStorage,
    get_metadata_storage,
    get_pipeline_trace,
    save_pipeline_metadata,
)
from .provenance import ModelAggregationAnchor, ProvenanceCapsule, TrainingSnapshot
from .simulation import MLFrameworkSimulator, MockLLM
from .wrappers import CIAFModelWrapper

# Optional modules - import with warnings if dependencies missing
try:
    from . import compliance
    from .compliance import AuditTrailGenerator, AuditTrail
    COMPLIANCE_AVAILABLE = True
except ImportError:
    COMPLIANCE_AVAILABLE = False

try:
    from . import explainability
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False

try:
    from . import uncertainty
    UNCERTAINTY_AVAILABLE = True
except ImportError:
    UNCERTAINTY_AVAILABLE = False

try:
    from . import preprocessing
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False

try:
    from . import metadata_tags
    METADATA_TAGS_AVAILABLE = True
except ImportError:
    METADATA_TAGS_AVAILABLE = False

__version__ = "1.0.0"
__all__ = [
    # Core components
    "CryptoUtils",
    "BaseAnchorManager",
    "AnchorManager",
    "KeyManager",  # Legacy alias
    "MerkleTree",
    "DatasetAnchor",
    "LazyManager",
    "LazyProvenanceManager",
    "ProvenanceCapsule",
    "TrainingSnapshot",
    "ModelAggregationAnchor",
    "MockLLM",
    "MLFrameworkSimulator",
    "InferenceReceipt",
    "ZKEConnections",
    "CIAFModelWrapper",
    "CIAFFramework",
    # Enhanced audit components
    "AuditTrailGenerator",
    "AuditTrail",
    # Metadata storage and management
    "MetadataStorage",
    "get_metadata_storage",
    "save_pipeline_metadata",
    "get_pipeline_trace",
    "MetadataConfig",
    "get_metadata_config",
    "load_config_from_file",
    "create_config_template",
    "MetadataCapture",
    "capture_metadata",
    "ModelMetadataManager",
    "ComplianceTracker",
    "create_model_manager",
    "create_compliance_tracker",
    "quick_log",
    # Feature availability flags
    "COMPLIANCE_AVAILABLE",
    "EXPLAINABILITY_AVAILABLE", 
    "UNCERTAINTY_AVAILABLE",
    "PREPROCESSING_AVAILABLE",
    "METADATA_TAGS_AVAILABLE",
]

# Create legacy alias for backward compatibility
from .core import AnchorManager
KeyManager = AnchorManager

# Export enhanced API methods
CIAFFramework.create_model_anchor = CIAFFramework.create_model_anchor
