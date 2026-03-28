"""
Cognitive Insight Audit Framework (CIAF)

A modular framework for creating verifiable AI training and inference pipelines
with lazy capsule materialization and cryptographic provenance tracking.

Created: 2025-09-09
Last Modified: 2026-03-28
Author: Denzil James Greenwood
Version: 1.3.2
"""
# type: ignore  # Conditional imports set variables to None which mypy doesn't like

# Anchoring module removed - using LCM system instead
from .lcm import LCMDatasetAnchor, LCMDatasetManager
from .api.framework import CIAFFramework
from .core import CryptoUtils, MerkleTree
from .inference import InferenceReceipt, ZKEConnections
from .vault.metadata_config import (
    MetadataConfig,
    create_config_template,
    get_metadata_config,
    load_config_from_file,
)
from .vault.metadata_integration import (
    ComplianceTracker,
    MetadataCapture,
    ModelMetadataManager,
    capture_metadata,
    create_compliance_tracker,
    create_model_manager,
    quick_log,
)

# Metadata storage and integration
from .vault.metadata_storage import (
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
        ReceiptHasher,
    )
    from .adaptive_lcm import (
        LCMMode,
        InferencePriority,
        AdaptiveLCMConfig,
        SystemMonitor,
        AdaptiveLCMWrapper,
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

# Enhanced validation and determinism components
try:
    from .evidence_strength import (
        EvidenceStrength,
        EvidenceTracker,
        get_evidence_tracker,
    )
    from .determinism_metadata import (
        DeterminismMetadata,
        capture_determinism_metadata,
        set_reproducible_seeds,
    )
    from .enhanced_receipts import (
        TrainingReceipt,
        InferenceReceipt,
        ReceiptValidator,
        create_training_receipt,
        create_inference_receipt,
    )
    from .crypto_health import (
        crypto_health_check,
        generate_secure_salt,
        generate_unique_nonce,
    )

    ENHANCED_VALIDATION_AVAILABLE = True
except ImportError:
    ENHANCED_VALIDATION_AVAILABLE = False
    EvidenceStrength = None
    EvidenceTracker = None
    get_evidence_tracker = None
    DeterminismMetadata = None
    capture_determinism_metadata = None
    set_reproducible_seeds = None
    TrainingReceipt = None
    InferenceReceipt = None
    ReceiptValidator = None
    create_training_receipt = None
    create_inference_receipt = None
    crypto_health_check = None
    generate_secure_salt = None
    generate_unique_nonce = None

# Optional modules - import with warnings if dependencies missing
try:
    from . import compliance
    from .compliance import AuditTrailGenerator, AuditTrail

    # New enterprise compliance features
    try:
        from .compliance.human_oversight import (
            HumanOversightEngine,
            OversightAlert,
            OversightReview,
        )
        from .compliance.web_dashboard import CIAFDashboard, create_dashboard
        from .compliance.robustness_testing import (
            RobustnessTestSuite,
            TestResult,
            RobustnessReport,
        )

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

# Agentic Execution Boundaries (NEW in v1.3.0)
try:
    from . import agents
    from .agents import (
        Identity,
        PrincipalType,
        Resource,
        ActionRequest,
        Permission,
        RoleDefinition,
        ExecutionResult,
        ElevationGrant,
        ActionReceipt,
        IAMStore,
        PAMStore,
        PolicyEngine,
        EvidenceVault,
        ToolExecutor,
        same_tenant_only,
        same_environment_only,
        sensitivity_level_check,
    )

    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False
    Identity = None
    PrincipalType = None
    Resource = None
    ActionRequest = None
    Permission = None
    RoleDefinition = None
    ExecutionResult = None
    ElevationGrant = None
    ActionReceipt = None
    IAMStore = None
    PAMStore = None
    PolicyEngine = None
    EvidenceVault = None
    ToolExecutor = None
    same_tenant_only = None
    same_environment_only = None
    sensitivity_level_check = None

try:
    from . import web
    from .web import (
        WebAIEvent,
        EventType,
        PolicyDecision,
        DataClassification,
        ToolCategory,
        AIToolDetector,
        detect_ai_tool,
        ContentClassifier,
        classify_content,
        PolicyEngine,
        evaluate_policy,
        WebAIReceipt,
        generate_receipt,
        WebAIVaultAdapter,
    )

    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    WebAIEvent = None  # type: ignore[assignment,misc]
    EventType = None  # type: ignore[assignment,misc]
    PolicyDecision = None  # type: ignore[assignment,misc]
    DataClassification = None  # type: ignore[assignment,misc]
    ToolCategory = None  # type: ignore[assignment,misc]
    AIToolDetector = None  # type: ignore[assignment,misc]
    detect_ai_tool = None  # type: ignore[assignment]
    ContentClassifier = None  # type: ignore[assignment,misc]
    classify_content = None  # type: ignore[assignment]
    PolicyEngine = None  # type: ignore[assignment,misc]
    evaluate_policy = None  # type: ignore[assignment]
    WebAIReceipt = None  # type: ignore[assignment,misc]
    generate_receipt = None  # type: ignore[assignment]
    WebAIVaultAdapter = None  # type: ignore[assignment,misc]

__version__ = "1.3.2"
__all__ = [
    # Core components
    "CryptoUtils",
    "MerkleTree",
    # LCM System (replaces legacy anchoring)
    "LCMDatasetAnchor",
    "LCMDatasetManager",
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
    # Enhanced validation and determinism
    "EvidenceStrength",
    "EvidenceTracker",
    "get_evidence_tracker",
    "DeterminismMetadata",
    "capture_determinism_metadata",
    "set_reproducible_seeds",
    "TrainingReceipt",
    "InferenceReceipt",
    "ReceiptValidator",
    "create_training_receipt",
    "create_inference_receipt",
    "crypto_health_check",
    "generate_secure_salt",
    "generate_unique_nonce",
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
    # Agentic Execution Boundaries (v1.3.0)
    "Identity",
    "PrincipalType",
    "Resource",
    "ActionRequest",
    "Permission",
    "RoleDefinition",
    "ExecutionResult",
    "ElevationGrant",
    "ActionReceipt",
    "IAMStore",
    "PAMStore",
    "PolicyEngine",
    "EvidenceVault",
    "ToolExecutor",
    "same_tenant_only",
    "same_environment_only",
    "sensitivity_level_check",
    # Feature availability flags
    "COMPLIANCE_AVAILABLE",
    "ENTERPRISE_COMPLIANCE_AVAILABLE",
    "ENHANCED_WRAPPER_AVAILABLE",
    "DEFERRED_LCM_AVAILABLE",
    "ENHANCED_VALIDATION_AVAILABLE",
    "EXPLAINABILITY_AVAILABLE",
    "UNCERTAINTY_AVAILABLE",
    "PREPROCESSING_AVAILABLE",
    "METADATA_TAGS_AVAILABLE",
    "AGENTS_AVAILABLE",
]

# Export enhanced API methods
CIAFFramework.create_model_anchor = CIAFFramework.create_model_anchor
