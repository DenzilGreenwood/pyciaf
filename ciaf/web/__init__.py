"""
CIAF Web - Browser-Based AI Usage Governance

Organizational AI usage governance through browser-based capture, policy enforcement,
and cryptographically verifiable evidence generation.

Key Capabilities:
- AI tool usage detection (approved & shadow AI)
- Content sensitivity classification
- Policy enforcement (allow/warn/redact/block)
- Evidence-bearing governance event capture
- Cryptographic receipt generation
- Incident reconstruction

Core Principles:
- Not surveillance - governance
- Minimum necessary capture by default
- Privacy-preserving hashing
- Verifiable evidence, not just logs

Modes:
1. Discovery - Find AI usage patterns
2. Policy - Enforce organizational rules
3. Evidence - Generate cryptographic receipts
4. Provenance - Track AI-generated outputs

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

__version__ = "1.0.0"

# Core event models
from .events import (
    WebAIEvent,
    EventType,
    PolicyDecision,
    DataClassification,
    ToolCategory,
)

# Detectors
from .detectors import (
    AIToolDetector,
    detect_ai_tool,
    is_approved_tool,
)

# Classifiers
from .classifiers import (
    ContentClassifier,
    SensitivityScorer,
    classify_content,
    score_sensitivity,
)

# Policy engine
from .policy import (
    PolicyEngine,
    PolicyRule,
    PolicyResult,
    evaluate_policy,
)

# Receipt generation
from .receipts import (
    WebAIReceipt,
    generate_receipt,
    verify_receipt,
)

# Vault integration
from .vault_adapter import (
    WebAIVaultAdapter,
    store_event,
    retrieve_events,
    search_events,
)

# Redaction
from .redaction import (
    redact_content,
    redact_pii,
    hash_sensitive_fields,
)

__all__ = [
    # Version
    "__version__",

    # Event models
    "WebAIEvent",
    "EventType",
    "PolicyDecision",
    "DataClassification",
    "ToolCategory",

    # Detection
    "AIToolDetector",
    "detect_ai_tool",
    "is_approved_tool",

    # Classification
    "ContentClassifier",
    "SensitivityScorer",
    "classify_content",
    "score_sensitivity",

    # Policy
    "PolicyEngine",
    "PolicyRule",
    "PolicyResult",
    "evaluate_policy",

    # Receipts
    "WebAIReceipt",
    "generate_receipt",
    "verify_receipt",

    # Vault
    "WebAIVaultAdapter",
    "store_event",
    "retrieve_events",
    "search_events",

    # Redaction
    "redact_content",
    "redact_pii",
    "hash_sensitive_fields",
]
