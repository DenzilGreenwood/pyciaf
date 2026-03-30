# CIAF Complete Schema Documentation

**Version**: 2.0.0
**Date**: 2026-03-30
**Status**: Production-Ready with Agentic Events

---

## Table of Contents

1. [Overview](#overview)
2. [Core CIAF Schemas](#core-ciaf-schemas)
3. [Agentic Execution Schemas](#agentic-execution-schemas)
4. [Web AI Governance Schemas](#web-ai-governance-schemas)
5. [Database Schemas (PostgreSQL)](#database-schemas-postgresql)
6. [Watermarking Schemas](#watermarking-schemas)
7. [Compliance Schemas](#compliance-schemas)
8. [Schema Relationships](#schema-relationships)
9. [Complete Type Index](#complete-type-index)

---

## Overview

This document provides the complete schema reference for the CIAF (Cognitive Insight Audit Framework) codebase. All schemas follow CIAF's principles:

- **Cryptographic Integrity**: SHA-256 hashing, Ed25519 signatures
- **Immutability**: Frozen dataclasses where appropriate
- **Privacy-First**: Hash-based by default, optional content capture
- **Audit-Ready**: Complete provenance tracking with hash chains
- **Compliance-Native**: Built-in regulatory framework mappings

### Schema Categories

| Category | Purpose | Primary Location |
|----------|---------|------------------|
| **Core CIAF** | Provenance, anchoring, Merkle trees | `ciaf/core/` |
| **Agentic Execution** | Agent IAM/PAM, receipts, events | `ciaf/agents/` |
| **Web AI Governance** | Browser-based AI monitoring | `ciaf/web/` |
| **Watermarking** | AI output verification | `ciaf/watermarks/` |
| **Compliance** | Regulatory mappings | `ciaf/compliance/` |

---

## Core CIAF Schemas

### 1. Dataset Anchor Schema

**Purpose**: Cryptographic root for dataset provenance
**File**: `ciaf/core/anchoring.py`

```python
@dataclass
class DatasetAnchor:
    """Cryptographic anchor for a dataset."""

    dataset_id: str              # Unique dataset identifier
    master_anchor: bytes         # Root cryptographic anchor
    dataset_anchor: bytes        # Derived dataset-specific anchor
    dataset_metadata: Dict[str, Any]  # Dataset description
    merkle_root: Optional[str]   # Merkle tree root hash
    created_at: str              # ISO 8601 timestamp
    anchor_hash: str             # SHA-256 of anchor
```

**Key Fields**:
- `master_anchor`: High-entropy binary root (256-bit)
- `dataset_anchor`: HMAC-SHA256 derivation from master
- `merkle_root`: Cryptographic proof of dataset integrity

### 2. Provenance Capsule Schema

**Purpose**: Individual data item with provenance
**File**: `ciaf/provenance/capsules.py`

```python
@dataclass
class ProvenanceCapsule:
    """Provenance wrapper for a single data item."""

    capsule_id: str              # Unique capsule identifier
    dataset_id: str              # Parent dataset
    content_hash: str            # SHA-256 of content
    metadata_hash: str           # SHA-256 of metadata
    capsule_anchor: bytes        # Derived anchor for this capsule
    content: Optional[Any]       # Actual content (lazy materialization)
    metadata: Dict[str, Any]     # Capsule metadata
    created_at: str              # ISO 8601 timestamp
    merkle_proof: Optional[List[str]]  # Merkle inclusion proof
```

**Privacy Model**:
- Content stored as hash by default
- Actual content loaded on-demand (Lazy Capsule Materialization)
- Metadata can include sensitive classifications

### 3. Model Anchor Schema

**Purpose**: Immutable fingerprint of model configuration
**File**: `ciaf/core/anchoring.py`

```python
@dataclass
class ModelAnchor:
    """Cryptographic anchor for a model."""

    model_name: str              # Model identifier
    model_version: str           # Version string
    parameters_fingerprint: str  # SHA-256 of model parameters
    architecture_fingerprint: str  # SHA-256 of model architecture
    authorized_datasets: List[str]  # Allowed training datasets
    model_anchor: bytes          # Derived model anchor
    created_at: str              # ISO 8601 timestamp
    metadata: Dict[str, Any]     # Additional model info
```

**Immutability Guarantee**:
- Parameter changes → New fingerprint
- Architecture changes → New fingerprint
- Cannot train model on unauthorized datasets

### 4. Training Snapshot Schema

**Purpose**: Verifiable record of a model training run
**File**: `ciaf/provenance/snapshots.py`

```python
@dataclass
class TrainingSnapshot:
    """Snapshot of model training state."""

    snapshot_id: str             # Unique snapshot identifier
    model_name: str              # Model being trained
    model_version: str           # Model version
    epoch: Optional[int]         # Training epoch
    dataset_anchors: List[str]   # Datasets used
    parameters_fingerprint: str  # Parameter state
    merkle_root: str             # Dataset integrity proof
    metrics: Dict[str, float]    # Training metrics
    timestamp: str               # ISO 8601 timestamp
    snapshot_hash: str           # SHA-256 of snapshot
```

### 5. Inference Receipt Schema

**Purpose**: Cryptographic proof of inference
**File**: `ciaf/core/receipts.py`

```python
@dataclass
class InferenceReceipt:
    """Cryptographic receipt of model inference."""

    receipt_id: str              # Unique receipt identifier
    model_name: str              # Model used
    model_version: str           # Model version
    query_hash: str              # SHA-256 of input
    output_hash: str             # SHA-256 of output
    timestamp: str               # ISO 8601 timestamp
    training_snapshot_ref: Optional[str]  # Link to training
    signature: str               # Ed25519 or HMAC signature
    receipt_hash: str            # SHA-256 of receipt
    metadata: Dict[str, Any]     # Additional context
```

**Verification Chain**:
```
Query → Model (with fingerprint) → Output
   ↓         ↓                        ↓
Hash     Snapshot Ref             Hash
                ↓
         Training Data (Merkle root)
```

---

## Agentic Execution Schemas

### 6. Principal Types (Identity Layer)

**Purpose**: Types of entities that can perform actions
**File**: `ciaf/agents/core/types.py`

```python
class PrincipalType(str, Enum):
    """Types of principals."""

    AGENT = "agent"      # Autonomous AI agents
    HUMAN = "human"      # Human users
    SERVICE = "service"  # Service accounts
    SYSTEM = "system"    # System processes
```

### 7. Identity Schema

**Purpose**: Immutable identity for agents and users
**File**: `ciaf/agents/core/types.py`

```python
@dataclass(frozen=True)
class Identity:
    """Immutable identity for a principal."""

    # Core identity
    principal_id: str              # Unique identifier
    principal_type: PrincipalType  # Type of principal
    display_name: str              # Human-readable name

    # Authorization
    roles: Set[str]                # RBAC roles (frozen)
    attributes: Dict[str, Any]     # ABAC attributes (frozen)

    # Multi-tenancy
    tenant_id: Optional[str]       # Tenant isolation
    environment: Optional[str]     # env (dev/staging/prod)

    # Metadata
    created_at: str                # ISO 8601 timestamp

    def get_fingerprint(self) -> str:
        """Generate cryptographic fingerprint."""
```

**Immutability**: Uses `frozen=True` to prevent modification after creation

### 8. Resource Schema

**Purpose**: Resources that can be accessed
**File**: `ciaf/agents/core/types.py`

```python
@dataclass(frozen=True)
class Resource:
    """A resource that can be accessed."""

    resource_id: str              # Unique identifier
    resource_type: str            # Type (e.g., "patient_record")
    owner_tenant: Optional[str]   # Tenant ownership
    attributes: Dict[str, Any]    # Resource metadata
    sensitivity_level: str        # "standard", "sensitive", "critical"
```

### 9. Permission Schema

**Purpose**: Grant of an action on a resource type
**File**: `ciaf/agents/core/types.py`

```python
@dataclass
class Permission:
    """A permission granting an action."""

    action: str                   # Action verb
    resource_type: str            # Resource type
    condition: Optional[Callable[[Identity, Resource], bool]]  # ABAC
    description: str
    requires_elevation: bool      # PAM-gated

    def allows(self, identity: Identity, resource: Resource) -> bool:
        """Check if permission allows action."""
```

**Built-in Conditions**:
- `same_tenant_only(identity, resource) -> bool`
- `same_environment_only(identity, resource) -> bool`
- `sensitivity_level_check(max_level: str)`

### 10. Elevation Grant Schema (PAM)

**Purpose**: Just-in-time privilege escalation
**File**: `ciaf/agents/core/types.py`

```python
@dataclass
class ElevationGrant:
    """JIT privilege elevation grant."""

    grant_id: str                 # Unique grant identifier
    principal_id: str             # Who gets the grant
    elevated_role: str            # Role being granted

    # Scope & approval
    scope: Dict[str, Any]         # What's allowed
    approved_by: str              # Who approved
    ticket_reference: str         # JIRA/ServiceNow ticket
    purpose: str                  # Justification

    # Temporal bounds
    valid_from: str               # ISO 8601 start
    valid_until: str              # ISO 8601 expiry

    # Usage limits
    used_count: int               # Times used
    max_uses: Optional[int]       # Maximum allowed

    def is_valid(self, now: Optional[datetime] = None) -> bool:
        """Check if grant is currently valid."""
```

### 11. Action Request Schema

**Purpose**: Request to perform an action
**File**: `ciaf/agents/core/types.py`

```python
@dataclass
class ActionRequest:
    """Request to perform an action."""

    action: str                   # Action to perform
    resource: Resource            # Target resource
    params: Dict[str, Any]        # Action parameters
    justification: str            # Why needed
    requested_by: Optional[Identity]  # Who requested
    correlation_id: Optional[str]  # Request correlation
    timestamp: str                # ISO 8601 timestamp

    def get_params_hash(self) -> str:
        """Get SHA-256 of parameters."""
```

### 12. Execution Result Schema

**Purpose**: Result of action execution
**File**: `ciaf/agents/core/types.py`

```python
@dataclass
class ExecutionResult:
    """Result of action execution."""

    request: ActionRequest
    allowed: bool                 # Was action authorized?
    reason: str                   # Why allowed/denied
    executed: bool                # Was it actually run?
    result: Any                   # Execution output
    error: Optional[str]          # Error if failed
    elevation_grant_id: Optional[str]  # PAM grant used
    policy_obligations: list[str]  # Compliance requirements
    timestamp: str                # ISO 8601 timestamp
```

### 13. Action Receipt Schema

**Purpose**: Cryptographic proof of action execution
**File**: `ciaf/agents/core/types.py`

```python
@dataclass
class ActionReceipt:
    """Cryptographic receipt of action."""

    receipt_id: str               # Unique receipt ID
    timestamp: str                # ISO 8601 timestamp
    principal_id: str             # Who performed
    principal_type: PrincipalType  # Type of principal
    action: str                   # What action
    resource_id: str              # On what resource
    resource_type: str            # Resource type
    correlation_id: Optional[str]  # Request correlation

    # Authorization
    decision: bool                # Allowed/denied
    reason: str                   # Why
    elevation_grant_id: Optional[str]  # PAM grant
    approved_by: Optional[str]    # Who approved

    # Cryptography
    params_hash: str              # SHA-256 of parameters
    prior_receipt_hash: str       # Hash chain linkage
    signature: str                # HMAC-SHA256 signature
    policy_obligations: list[str]  # Compliance

    def get_receipt_hash(self) -> str:
        """Generate SHA-256 hash of receipt."""
```

**Hash Chain**:
```
Receipt 1 (hash: abc...)
    ↓ prior_receipt_hash
Receipt 2 (hash: def..., prior: abc...)
    ↓ prior_receipt_hash
Receipt 3 (hash: ghi..., prior: def...)
```

### 14. Agent Event Type Enumeration (NEW)

**Purpose**: Formal event types for agent actions
**File**: `ciaf/agents/events.py`

```python
class AgentEventType(str, Enum):
    """First-class agent governance event types."""

    # Data operations
    READ = "agent_read"
    WRITE = "agent_write"
    DELETE = "agent_delete"
    SEARCH = "agent_search"
    EXPORT = "agent_export"

    # External interactions
    API_CALL = "agent_api_call"
    HTTP_REQUEST = "agent_http_request"
    DATABASE_QUERY = "agent_database_query"
    FILE_ACCESS = "agent_file_access"

    # Autonomous behavior
    DECISION = "agent_decision"
    REASONING = "agent_reasoning"
    PLAN_GENERATION = "agent_plan"
    GOAL_UPDATE = "agent_goal_update"

    # Governance & control
    POLICY_CHECK = "agent_policy_check"
    ELEVATION_REQUEST = "agent_elevation_request"
    HUMAN_OVERRIDE = "agent_human_override"
    APPROVAL_REQUEST = "agent_approval_request"

    # Tool & function usage
    TOOL_CALL = "agent_tool_call"
    FUNCTION_EXECUTION = "agent_function_execution"

    # Inter-agent
    AGENT_MESSAGE = "agent_message"
    AGENT_DELEGATION = "agent_delegation"

    # System
    SESSION_START = "agent_session_start"
    SESSION_END = "agent_session_end"
    ERROR = "agent_error"
```

### 15. Agent Action Type Enumeration (NEW)

**Purpose**: Formal action types for governance
**File**: `ciaf/agents/events.py`

```python
class AgentActionType(str, Enum):
    """Formal action types for agent operations."""

    # Data access
    READ_RECORD = "read_record"
    WRITE_RECORD = "write_record"
    UPDATE_RECORD = "update_record"
    DELETE_RECORD = "delete_record"
    SEARCH_RECORDS = "search_records"
    EXPORT_DATA = "export_data"

    # Approvals & workflows
    APPROVE_PAYMENT = "approve_payment"
    APPROVE_CLAIM = "approve_claim"
    APPROVE_CHANGE = "approve_change"
    REJECT_REQUEST = "reject_request"

    # Infrastructure
    DEPLOY_CODE = "deploy_code"
    ROLLBACK_DEPLOY = "rollback_deploy"
    UPDATE_CONFIG = "update_config"
    RESTART_SERVICE = "restart_service"

    # Database operations
    DATABASE_QUERY = "database_query"
    DATABASE_WRITE = "database_write"
    DATABASE_BACKUP = "database_backup"

    # API operations
    EXTERNAL_API_CALL = "external_api_call"
    INTERNAL_API_CALL = "internal_api_call"

    # Sensitive data
    ACCESS_PHI = "access_phi"
    ACCESS_PII = "access_pii"
    ACCESS_SECRETS = "access_secrets"

    # Custom
    CUSTOM = "custom"
```

### 16. Agent Event Schema (NEW)

**Purpose**: First-class governance event for agent actions
**File**: `ciaf/agents/events.py`

```python
@dataclass
class AgentEvent:
    """First-class agent governance event."""

    # Primary identifiers
    event_id: str                 # Unique event ID
    event_type: AgentEventType    # Event type
    occurred_at: str              # ISO 8601 timestamp

    # Agent identity
    agent_id: str                 # Agent identifier
    agent_name: str               # Display name
    principal_type: str           # "agent", "human", etc.
    session_id: str               # Session ID

    # Organizational context
    org_id: Optional[str]         # Organization
    tenant_id: Optional[str]      # Tenant
    environment: Optional[str]    # dev/staging/prod

    # Action details
    action: str                   # Action performed
    resource_type: str            # Resource type
    resource_id: str              # Resource ID
    params: Dict[str, Any]        # Parameters
    justification: str            # Why
    correlation_id: Optional[str]  # Request correlation

    # Policy & authorization
    policy_decision: PolicyDecision
    policy_rule_id: Optional[str]
    policy_reason: Optional[str]
    elevation_grant_id: Optional[str]
    approved_by: Optional[str]

    # Classification
    sensitivity_level: Optional[SensitivityLevel]
    data_classification: Optional[str]

    # Execution
    executed: bool                # Was it run?
    success: bool                 # Did it succeed?
    error_message: Optional[str]  # Error if failed

    # Privacy hashes
    params_hash: str              # SHA-256 of params
    input_hash: str               # SHA-256 of input
    output_hash: str              # SHA-256 of output

    # Evidence
    signature: Optional[str]      # Ed25519 signature
    signature_algorithm: Optional[str]
    prior_event_hash: str         # Hash chain

    # Compliance
    compliance_frameworks: List[str]
    policy_obligations: List[str]

    # Metadata
    metadata: Dict[str, Any]
    tags: List[str]

    def get_event_hash(self) -> str:
        """Generate SHA-256 hash."""

    def requires_elevation(self) -> bool:
        """Check if elevated."""

    def is_sensitive(self) -> bool:
        """Check if sensitive."""

    def is_high_risk(self) -> bool:
        """Check if high-risk."""
```

---

## Web AI Governance Schemas

### 17. Web AI Event Type Enumeration

**Purpose**: Types of browser-based AI interactions
**File**: `ciaf/web/events.py`

```python
class EventType(str, Enum):
    """Types of AI interaction events."""

    PROMPT_SUBMIT = "prompt_submit"
    OUTPUT_RECEIVE = "output_receive"
    FILE_UPLOAD = "file_upload"
    FILE_DOWNLOAD = "file_download"
    PASTE_CONTENT = "paste_content"
    COPY_OUTPUT = "copy_output"
    PAGE_VISIT = "page_visit"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    POLICY_BLOCK = "policy_block"
    POLICY_WARN = "policy_warn"
    POLICY_REDACT = "policy_redact"
    SHADOW_AI_DETECT = "shadow_ai_detect"
    APPROVED_TOOL_USE = "approved_tool_use"
```

### 18. Policy Decision Enumeration

**Purpose**: Policy evaluation results
**File**: `ciaf/web/events.py`

```python
class PolicyDecision(str, Enum):
    """Policy evaluation results."""

    ALLOW = "allow"
    WARN = "warn"
    REDACT = "redact"
    BLOCK = "block"
    ESCALATE = "escalate"
    NOT_EVALUATED = "not_evaluated"
```

### 19. Data Classification Enumeration

**Purpose**: Content sensitivity levels
**File**: `ciaf/web/events.py`

```python
class DataClassification(str, Enum):
    """Content sensitivity classifications."""

    PUBLIC = "public"               # 0.0-0.3
    INTERNAL = "internal"           # 0.3-0.6
    CONFIDENTIAL = "confidential"   # 0.6-0.8
    RESTRICTED = "restricted"       # 0.8-0.9
    HIGHLY_RESTRICTED = "highly_restricted"  # 0.9-1.0
    UNKNOWN = "unknown"
```

### 20. Tool Category Enumeration

**Purpose**: Categories of AI tools
**File**: `ciaf/web/events.py`

```python
class ToolCategory(str, Enum):
    """Categories of AI tools."""

    LLM_CHAT = "llm_chat"              # ChatGPT, Claude
    CODE_ASSISTANT = "code_assistant"   # Copilot, Cursor
    IMAGE_GEN = "image_generation"      # Midjourney, DALL-E
    DOCUMENT_AI = "document_ai"
    TRANSLATION = "translation"
    SEARCH_AI = "search_ai"             # Perplexity
    PRODUCTIVITY = "productivity"       # Notion AI
    VOICE_AI = "voice_ai"
    VIDEO_AI = "video_ai"
    OTHER = "other"
```

### 21. Web AI Event Schema

**Purpose**: Browser-based AI usage event
**File**: `ciaf/web/events.py`

```python
@dataclass
class WebAIEvent:
    """Web AI usage governance event."""

    # Primary identifiers
    event_id: str                 # Unique event ID
    event_type: EventType         # Event type
    occurred_at: str              # ISO 8601 timestamp

    # Actor context
    org_id: str                   # Organization
    user_id: str                  # User
    session_id: str               # Session
    device_id: Optional[str]      # Device
    browser_id: Optional[str]     # Browser

    # Tool context
    tool_name: Optional[str]      # "ChatGPT", "Claude", etc.
    tool_domain: Optional[str]    # "chat.openai.com"
    tool_category: Optional[ToolCategory]
    tool_approved: Optional[bool]  # Is it approved?

    # Privacy-preserving hashes
    page_url_hash: Optional[str]   # SHA-256 of URL
    prompt_hash: Optional[str]     # SHA-256 of prompt
    output_hash: Optional[str]     # SHA-256 of output
    uploaded_file_hashes: List[str]

    # Classification & policy
    data_classification: Optional[DataClassification]
    sensitivity_score: Optional[float]  # 0.0-1.0
    policy_decision: Optional[PolicyDecision]
    policy_rule_id: Optional[str]
    policy_reason: Optional[str]

    # Evidence
    raw_content_ref: Optional[str]  # Reference if stored
    signature: Optional[str]        # Ed25519 signature
    signature_algorithm: Optional[str]
    witness_hash: Optional[str]     # Merkle inclusion

    # Metadata
    metadata: Dict[str, Any]
    tags: List[str]

    def is_shadow_ai(self) -> bool:
        """Check if shadow AI."""

    def is_high_risk(self, threshold: float = 0.7) -> bool:
        """Check if high-risk."""

    def was_blocked(self) -> bool:
        """Check if blocked."""

    def needs_review(self) -> bool:
        """Check if needs review."""
```

### 22. Web AI Receipt Schema

**Purpose**: Cryptographic proof of web AI usage
**File**: `ciaf/web/receipts.py`

```python
@dataclass
class WebAIReceipt:
    """Cryptographic receipt for web AI event."""

    receipt_id: str               # Unique receipt ID
    event_id: str                 # Associated event
    created_at: str               # ISO 8601 timestamp
    event_type: str               # Event type
    org_id: str                   # Organization

    # Cryptographic proof
    event_hash: str               # SHA-256 of event
    receipt_hash: str             # SHA-256 of receipt
    signature: str                # HMAC or Ed25519
    signature_algorithm: str      # Algorithm used

    # Chain of custody
    previous_receipt_id: Optional[str]
    chain_position: Optional[int]

    # Metadata
    metadata: Dict[str, Any]

    def verify_signature(self, public_key: str) -> bool:
        """Verify cryptographic signature."""

    def is_chained(self) -> bool:
        """Check if part of chain."""
```

---

## Database Schemas (PostgreSQL)

### 23. Core CIAF Tables

#### Table: `ciaf_metadata`
**Purpose**: Model and training metadata

```sql
CREATE TABLE ciaf_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metadata_id TEXT UNIQUE NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,
    stage TEXT NOT NULL,  -- 'training', 'inference', 'deployment'
    event_type TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata_hash TEXT,
    details TEXT,
    metadata_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_metadata_model_name ON ciaf_metadata(model_name);
CREATE INDEX idx_metadata_stage ON ciaf_metadata(stage);
CREATE INDEX idx_metadata_timestamp ON ciaf_metadata(timestamp);
CREATE INDEX idx_metadata_json ON ciaf_metadata USING GIN(metadata_json);
```

#### Table: `ciaf_audit_trail`
**Purpose**: Cryptographic audit events

```sql
CREATE TABLE ciaf_audit_trail (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id TEXT UNIQUE NOT NULL,
    parent_id TEXT,
    action TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id TEXT,
    details TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_audit_parent_id ON ciaf_audit_trail(parent_id);
CREATE INDEX idx_audit_timestamp ON ciaf_audit_trail(timestamp);
```

#### Table: `ciaf_compliance_events`
**Purpose**: Regulatory compliance tracking

```sql
CREATE TABLE ciaf_compliance_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id TEXT UNIQUE NOT NULL,
    framework TEXT NOT NULL,  -- 'EU_AI_ACT', 'GDPR', 'NIST_AI_RMF'
    requirement TEXT NOT NULL,
    status TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    details TEXT,
    evidence JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_compliance_framework ON ciaf_compliance_events(framework);
CREATE INDEX idx_compliance_timestamp ON ciaf_compliance_events(timestamp);
```

#### Table: `ciaf_inference_receipts`
**Purpose**: Inference receipts

```sql
CREATE TABLE ciaf_inference_receipts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    receipt_id TEXT UNIQUE NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,
    query_hash TEXT,
    output_hash TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    receipt_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_receipt_model_name ON ciaf_inference_receipts(model_name);
CREATE INDEX idx_receipt_timestamp ON ciaf_inference_receipts(timestamp);
```

#### Table: `ciaf_training_snapshots`
**Purpose**: Training checkpoints

```sql
CREATE TABLE ciaf_training_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id TEXT UNIQUE NOT NULL,
    model_name TEXT NOT NULL,
    epoch INTEGER,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metrics JSONB,
    snapshot_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_snapshot_model_name ON ciaf_training_snapshots(model_name);
CREATE INDEX idx_snapshot_timestamp ON ciaf_training_snapshots(timestamp);
```

#### Table: `ciaf_provenance_capsules`
**Purpose**: Data lineage

```sql
CREATE TABLE ciaf_provenance_capsules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    capsule_id TEXT UNIQUE NOT NULL,
    dataset_id TEXT NOT NULL,
    data_hash TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    capsule_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_capsule_dataset_id ON ciaf_provenance_capsules(dataset_id);
CREATE INDEX idx_capsule_timestamp ON ciaf_provenance_capsules(timestamp);
```

### 24. Agentic Event Tables (NEW)

#### Table: `agent_events`
**Purpose**: First-class agent governance events

```sql
CREATE TABLE agent_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id TEXT UNIQUE NOT NULL,
    event_type TEXT NOT NULL,  -- AgentEventType
    occurred_at TIMESTAMPTZ NOT NULL,

    -- Agent identity
    agent_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    principal_type TEXT NOT NULL,
    session_id TEXT NOT NULL,

    -- Organizational context
    org_id TEXT,
    tenant_id TEXT,
    environment TEXT,

    -- Action details
    action TEXT NOT NULL,  -- AgentActionType
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    justification TEXT,
    correlation_id TEXT,

    -- Policy & authorization
    policy_decision TEXT,
    policy_rule_id TEXT,
    policy_reason TEXT,
    elevation_grant_id TEXT,
    approved_by TEXT,

    -- Classification
    sensitivity_level TEXT,
    data_classification TEXT,

    -- Execution
    executed BOOLEAN DEFAULT FALSE,
    success BOOLEAN DEFAULT FALSE,
    error_message TEXT,

    -- Hashes
    params_hash TEXT,
    input_hash TEXT,
    output_hash TEXT,

    -- Evidence
    signature TEXT,
    signature_algorithm TEXT,
    prior_event_hash TEXT DEFAULT '0000000000000000000000000000000000000000000000000000000000000000',

    -- Full event data
    event_json JSONB NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_agent_event_agent_id ON agent_events(agent_id);
CREATE INDEX idx_agent_event_session_id ON agent_events(session_id);
CREATE INDEX idx_agent_event_org_id ON agent_events(org_id);
CREATE INDEX idx_agent_event_occurred_at ON agent_events(occurred_at);
CREATE INDEX idx_agent_event_action ON agent_events(action);
CREATE INDEX idx_agent_event_resource_type ON agent_events(resource_type);
CREATE INDEX idx_agent_event_policy_decision ON agent_events(policy_decision);
CREATE INDEX idx_agent_event_elevation ON agent_events(elevation_grant_id) WHERE elevation_grant_id IS NOT NULL;
CREATE INDEX idx_agent_event_high_risk ON agent_events(sensitivity_level) WHERE sensitivity_level IN ('restricted', 'highly_restricted');
CREATE INDEX idx_agent_event_json ON agent_events USING GIN(event_json);
```

### 25. Web AI Governance Tables (NEW)

#### Table: `web_ai_events`
**Purpose**: Browser-based AI usage events

```sql
CREATE TABLE web_ai_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id TEXT UNIQUE NOT NULL,
    event_type TEXT NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,

    -- User context
    org_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    device_id TEXT,
    browser_id TEXT,

    -- Tool context
    tool_name TEXT,
    tool_domain TEXT,
    tool_category TEXT,
    tool_approved BOOLEAN,

    -- Privacy-preserving hashes
    page_url_hash TEXT,
    prompt_hash TEXT,
    output_hash TEXT,

    -- Classification & policy
    data_classification TEXT,
    sensitivity_score NUMERIC(3, 2),  -- 0.00-1.00
    policy_decision TEXT,
    policy_rule_id TEXT,
    policy_reason TEXT,

    -- Evidence
    signature TEXT,
    witness_hash TEXT,

    -- Full event data
    event_json JSONB NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_web_ai_event_org_id ON web_ai_events(org_id);
CREATE INDEX idx_web_ai_event_user_id ON web_ai_events(user_id);
CREATE INDEX idx_web_ai_event_tool_name ON web_ai_events(tool_name);
CREATE INDEX idx_web_ai_event_occurred_at ON web_ai_events(occurred_at);
CREATE INDEX idx_web_ai_event_shadow_ai ON web_ai_events(tool_approved) WHERE tool_approved = false;
CREATE INDEX idx_web_ai_event_high_risk ON web_ai_events(sensitivity_score) WHERE sensitivity_score >= 0.7;
CREATE INDEX idx_web_ai_event_json ON web_ai_events USING GIN(event_json);
```

#### Table: `web_ai_receipts`
**Purpose**: Cryptographic receipts for web events

```sql
CREATE TABLE web_ai_receipts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    receipt_id TEXT UNIQUE NOT NULL,
    event_id TEXT REFERENCES web_ai_events(event_id),
    org_id TEXT NOT NULL,
    issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Cryptographic proof
    event_hash TEXT NOT NULL,
    receipt_hash TEXT NOT NULL,
    signature TEXT NOT NULL,
    signature_algorithm TEXT DEFAULT 'HMAC-SHA256',

    -- Chain of custody
    previous_receipt_id TEXT,
    chain_position INTEGER,

    receipt_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_web_ai_receipt_event_id ON web_ai_receipts(event_id);
CREATE INDEX idx_web_ai_receipt_org_id ON web_ai_receipts(org_id);
CREATE INDEX idx_web_ai_receipt_chain ON web_ai_receipts(previous_receipt_id);
```

---

## Watermarking Schemas

### 26. Artifact Evidence Schema

**Purpose**: Forensic provenance for AI outputs
**File**: `ciaf/watermarks/models.py`

```python
@dataclass
class ArtifactEvidence:
    """Forensic provenance for AI-generated content."""

    artifact_id: str              # Unique artifact ID
    artifact_type: str            # "text", "image", "video", "audio"
    created_at: str               # ISO 8601 timestamp

    # Watermark
    watermark_embedded: bool
    watermark_method: Optional[str]

    # Forensic fragments (DNA)
    forensic_fragments: List[ForensicFragment]

    # Hashes
    before_hash: str              # SHA-256 before watermark
    after_hash: str               # SHA-256 after watermark
    perceptual_hash: Optional[str]  # SimHash

    # Provenance
    model_name: str
    model_version: str
    training_snapshot_ref: Optional[str]

    # Verification
    verification_tier: int        # 1 (fast), 2 (standard), 3 (forensic)
    last_verified_at: Optional[str]
```

### 27. Forensic Fragment Schema

**Purpose**: DNA-level sampling for tamper detection
**File**: `ciaf/watermarks/models.py`

```python
@dataclass
class ForensicFragment:
    """DNA sample from artifact."""

    fragment_id: str              # Unique fragment ID
    fragment_type: str            # "text_span", "pixel_region", etc.
    position: int                 # Position in artifact
    length: int                   # Length of fragment
    content_hash: str             # SHA-256 of fragment
    embedding_hash: Optional[str]  # Hash of embedding
    created_at: str               # ISO 8601 timestamp
```

---

## Compliance Schemas

### 28. Compliance Framework Enumeration

**Purpose**: Supported regulatory frameworks
**File**: `ciaf/compliance/__init__.py`

```python
class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""

    EU_AI_ACT = "EU_AI_ACT"
    GDPR = "GDPR"
    HIPAA = "HIPAA"
    NIST_AI_RMF = "NIST_AI_RMF"
    SOX = "SOX"
    ISO_27001 = "ISO_27001"
    CCPA = "CCPA"
    PCI_DSS = "PCI_DSS"
```

### 29. Compliance Status Schema

**Purpose**: Compliance assessment result
**File**: `ciaf/compliance/validators.py`

```python
@dataclass
class ComplianceStatus:
    """Compliance assessment result."""

    framework: ComplianceFramework
    model_name: str
    assessed_at: str              # ISO 8601 timestamp

    # Overall status
    compliant: bool
    score: float                  # 0.0-100.0

    # Control breakdown
    controls_met: int
    controls_total: int
    controls_partial: int

    # Gaps
    gaps: List[ComplianceGap]

    # Evidence
    evidence_refs: List[str]
```

---

## Schema Relationships

### Entity Relationship Diagram

```
┌─────────────────────┐
│   Dataset Anchor    │
└──────────┬──────────┘
           │ 1:N
           ↓
┌─────────────────────┐      ┌─────────────────────┐
│ Provenance Capsule  │──────│  Merkle Tree Proof  │
└──────────┬──────────┘  N:1 └─────────────────────┘
           │ N:M
           ↓
┌─────────────────────┐
│   Model Anchor      │
└──────────┬──────────┘
           │ 1:N
           ↓
┌─────────────────────┐
│ Training Snapshot   │
└──────────┬──────────┘
           │ 1:N
           ↓
┌─────────────────────┐
│ Inference Receipt   │
└─────────────────────┘
```

### Agentic Flow

```
┌─────────────────────┐
│      Identity       │
└──────────┬──────────┘
           │ 1:N
           ↓
┌─────────────────────┐
│   ActionRequest     │
└──────────┬──────────┘
           │ 1:1
           ↓
┌─────────────────────┐      ┌─────────────────────┐
│  ExecutionResult    │──N:1─│  ElevationGrant     │
└──────────┬──────────┘      └─────────────────────┘
           │ 1:1
           ↓
┌─────────────────────┐      ┌─────────────────────┐
│   ActionReceipt     │──1:1─│    AgentEvent       │
└──────────┬──────────┘      └─────────────────────┘
           │ Hash Chain
           ↓
┌─────────────────────┐
│  Prior Receipt      │
└─────────────────────┘
```

### Web AI Flow

```
┌─────────────────────┐
│   Browser Event     │
└──────────┬──────────┘
           │ Detection
           ↓
┌─────────────────────┐      ┌─────────────────────┐
│    WebAIEvent       │──N:1─│  Policy Rule        │
└──────────┬──────────┘      └─────────────────────┘
           │ 1:1
           ↓
┌─────────────────────┐
│   WebAIReceipt      │
└──────────┬──────────┘
           │ Hash Chain
           ↓
┌─────────────────────┐
│  Prior Receipt      │
└─────────────────────┘
```

---

## Complete Type Index

### Enumerations

| Enum | File | Values |
|------|------|--------|
| `PrincipalType` | `ciaf/agents/core/types.py` | AGENT, HUMAN, SERVICE, SYSTEM |
| `AgentEventType` | `ciaf/agents/events.py` | READ, WRITE, API_CALL, DECISION, etc. (23 types) |
| `AgentActionType` | `ciaf/agents/events.py` | READ_RECORD, APPROVE_PAYMENT, etc. (20+ types) |
| `PolicyDecision` (Agents) | `ciaf/agents/events.py` | ALLOW, DENY, REQUIRE_ELEVATION, etc. |
| `SensitivityLevel` | `ciaf/agents/events.py` | PUBLIC, INTERNAL, CONFIDENTIAL, etc. |
| `EventType` (Web AI) | `ciaf/web/events.py` | PROMPT_SUBMIT, OUTPUT_RECEIVE, etc. (14 types) |
| `PolicyDecision` (Web) | `ciaf/web/events.py` | ALLOW, WARN, REDACT, BLOCK, etc. |
| `DataClassification` | `ciaf/web/events.py` | PUBLIC, INTERNAL, CONFIDENTIAL, etc. |
| `ToolCategory` | `ciaf/web/events.py` | LLM_CHAT, CODE_ASSISTANT, etc. (9 categories) |
| `ComplianceFramework` | `ciaf/compliance/__init__.py` | EU_AI_ACT, GDPR, HIPAA, etc. (8 frameworks) |

### Dataclasses

| Dataclass | Purpose | File |
|-----------|---------|------|
| `DatasetAnchor` | Dataset provenance root | `ciaf/core/anchoring.py` |
| `ProvenanceCapsule` | Individual data item | `ciaf/provenance/capsules.py` |
| `ModelAnchor` | Model fingerprint | `ciaf/core/anchoring.py` |
| `TrainingSnapshot` | Training checkpoint | `ciaf/provenance/snapshots.py` |
| `InferenceReceipt` | Inference proof | `ciaf/core/receipts.py` |
| `Identity` | Principal identity | `ciaf/agents/core/types.py` |
| `Resource` | Accessible resource | `ciaf/agents/core/types.py` |
| `Permission` | Action grant | `ciaf/agents/core/types.py` |
| `RoleDefinition` | RBAC role | `ciaf/agents/core/types.py` |
| `ActionRequest` | Action proposal | `ciaf/agents/core/types.py` |
| `ExecutionResult` | Action outcome | `ciaf/agents/core/types.py` |
| `ElevationGrant` | PAM grant | `ciaf/agents/core/types.py` |
| `ActionReceipt` | Action proof | `ciaf/agents/core/types.py` |
| `AgentEvent` | Agent governance event | `ciaf/agents/events.py` |
| `AgentEventBatch` | Batch of agent events | `ciaf/agents/events.py` |
| `WebAIEvent` | Web AI usage event | `ciaf/web/events.py` |
| `EventBatch` | Batch of web events | `ciaf/web/events.py` |
| `WebAIReceipt` | Web AI proof | `ciaf/web/receipts.py` |
| `ArtifactEvidence` | Watermark provenance | `ciaf/watermarks/models.py` |
| `ForensicFragment` | DNA sample | `ciaf/watermarks/models.py` |
| `ComplianceStatus` | Compliance result | `ciaf/compliance/validators.py` |

### Database Tables

| Table | Purpose | Row Count (Typical) |
|-------|---------|---------------------|
| `ciaf_metadata` | Metadata storage | 1,000s |
| `ciaf_audit_trail` | Audit events | 10,000s |
| `ciaf_compliance_events` | Compliance tracking | 100s |
| `ciaf_inference_receipts` | Inference proofs | 1,000,000s |
| `ciaf_training_snapshots` | Training checkpoints | 100s |
| `ciaf_provenance_capsules` | Data lineage | 1,000,000s |
| `agent_events` | Agent actions | 100,000s |
| `web_ai_events` | Web AI usage | 1,000,000s |
| `web_ai_receipts` | Web proofs | 1,000,000s |

---

## Implementation Notes

### Hash Algorithms

- **SHA-256**: Primary hashing (content, receipts, fingerprints)
- **HMAC-SHA256**: Anchor derivation, receipt signatures
- **Ed25519**: Digital signatures (receipts, evidence)
- **SimHash**: Perceptual similarity (watermarks)

### Timestamp Format

All timestamps use **ISO 8601**:
```python
"2026-03-30T14:32:45.123456+00:00"
```

Generated via:
```python
datetime.now(timezone.utc).isoformat()
```

### Hash Chain Pattern

```python
# Genesis (first receipt)
prior_receipt_hash = "0" * 64  # 64 zeros

# Subsequent receipts
current_hash = sha256_hash(receipt_data)
prior_receipt_hash = previous_receipt.get_receipt_hash()
```

### Privacy Model

**Default**: Hash-only storage
```python
content_hash = sha256_hash(content)
# Content not stored
```

**Optional**: Full content capture
```python
content_hash = sha256_hash(content)
raw_content_ref = store_encrypted(content)  # If configured
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-03-24 | Initial schema documentation |
| 2.0.0 | 2026-03-30 | Added agentic events, agent_events table, web_ai_events table |

---

**Document Version**: 2.0.0
**Last Updated**: 2026-03-30
**Maintainer**: CIAF Development Team
**License**: Business Source License 1.1 (BUSL-1.1)
