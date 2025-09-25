"""
Cognitive Insight Audit Framework (CIAF)

A modular framework for creating verifiable AI training and inference pipelines
with lazy capsule materialization and cryptographic provenance tracking.

Created: 2025-09-09
Last Modified: 2025-01-19
Author: Denzil James Greenwood
Version: 1.1.0
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
# Simulation and wrappers
from .simulation import MLFrameworkSimulator, MockLLM
from .wrappers import CIAFModelWrapper

# Enhanced wrapper (with availability check)
try:
    from .wrappers import EnhancedCIAFModelWrapper, ENHANCED_WRAPPER_AVAILABLE
except ImportError:
    ENHANCED_WRAPPER_AVAILABLE = False
    EnhancedCIAFModelWrapper = None

# Deferred LCM components (high-performance processing)
try:
    from .deferred_lcm import (
        LightweightReceipt,
        ReceiptQueue, 
        DeferredLCMProcessor,
        ReceiptHasher
    )
    from .adaptive_lcm import (
        LCMMode,
        InferencePriority,
        AdaptiveLCMConfig,
        SystemMonitor,
        AdaptiveLCMWrapper
    )
    DEFERRED_LCM_AVAILABLE = True
except ImportError:
    DEFERRED_LCM_AVAILABLE = False
    LightweightReceipt = None
    ReceiptQueue = None
    DeferredLCMProcessor = None
    ReceiptHasher = None
    LCMMode = None
    InferencePriority = None
    AdaptiveLCMConfig = None
    SystemMonitor = None
    AdaptiveLCMWrapper = None

# Optional modules - import with warnings if dependencies missing
try:
    from . import compliance
    from .compliance import AuditTrailGenerator, AuditTrail
    # New enterprise compliance features
    try:
        from .compliance.human_oversight import HumanOversightEngine, OversightAlert, OversightReview
        from .compliance.web_dashboard import CIAFDashboard, create_dashboard
        from .compliance.robustness_testing import RobustnessTestSuite, TestResult, RobustnessReport
        ENTERPRISE_COMPLIANCE_AVAILABLE = True
    except ImportError:
        ENTERPRISE_COMPLIANCE_AVAILABLE = False
    COMPLIANCE_AVAILABLE = True
except ImportError:
    COMPLIANCE_AVAILABLE = False
    ENTERPRISE_COMPLIANCE_AVAILABLE = False

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

__version__ = "1.1.0"
__all__ = [
    # Core components
    "CryptoUtils",
    "BaseAnchorManager",
    "AnchorManager",
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
    "EnhancedCIAFModelWrapper",
    "CIAFFramework",
    # Deferred LCM components
    "LightweightReceipt",
    "ReceiptQueue",
    "DeferredLCMProcessor", 
    "ReceiptHasher",
    "LCMMode",
    "InferencePriority",
    "AdaptiveLCMConfig",
    "SystemMonitor",
    "AdaptiveLCMWrapper",
    # Enhanced audit components
    "AuditTrailGenerator",
    "AuditTrail",
    # Enterprise compliance features (if available)
    "HumanOversightEngine",
    "OversightAlert", 
    "OversightReview",
    "CIAFDashboard",
    "create_dashboard",
    "RobustnessTestSuite",
    "TestResult",
    "RobustnessReport",
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
    "ENTERPRISE_COMPLIANCE_AVAILABLE",
    "ENHANCED_WRAPPER_AVAILABLE",
    "DEFERRED_LCM_AVAILABLE",
    "EXPLAINABILITY_AVAILABLE", 
    "UNCERTAINTY_AVAILABLE",
    "PREPROCESSING_AVAILABLE",
    "METADATA_TAGS_AVAILABLE",
]

# Export enhanced API methods
CIAFFramework.create_model_anchor = CIAFFramework.create_model_anchor
