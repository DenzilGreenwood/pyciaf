"""
CIAF Web - Event Models

Core data structures for AI usage governance events.

WebAIEvent is the central model capturing:
- Who (user, org, device)
- What (tool, action, content)
- When (timestamp)
- Where (domain, URL)
- Why (classification, policy result)
- Evidence (hashes, signatures)

Design Principles:
- Hash sensitive content, don't store raw by default
- Minimum necessary capture
- Privacy-preserving by design
- Cryptographically verifiable

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
from enum import Enum
import uuid
from datetime import datetime, timezone


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


class PolicyDecision(str, Enum):
    """Policy evaluation results."""
    ALLOW = "allow"
    WARN = "warn"
    REDACT = "redact"
    BLOCK = "block"
    ESCALATE = "escalate"
    NOT_EVALUATED = "not_evaluated"


class DataClassification(str, Enum):
    """Content sensitivity classifications."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    HIGHLY_RESTRICTED = "highly_restricted"
    UNKNOWN = "unknown"


class ToolCategory(str, Enum):
    """Categories of AI tools."""
    LLM_CHAT = "llm_chat"  # ChatGPT, Claude, etc.
    CODE_ASSISTANT = "code_assistant"  # GitHub Copilot, Cursor
    IMAGE_GEN = "image_generation"  # Midjourney, DALL-E
    DOCUMENT_AI = "document_ai"  # Document analysis
    TRANSLATION = "translation"
    SEARCH_AI = "search_ai"  # Perplexity, Bing Chat
    PRODUCTIVITY = "productivity"  # Notion AI, etc.
    VOICE_AI = "voice_ai"
    VIDEO_AI = "video_ai"
    OTHER = "other"


@dataclass
class WebAIEvent:
    """
    Core event model for AI usage governance.

    Captures an AI interaction as an evidence-bearing governance event,
    not just a log entry.

    Minimum necessary capture - stores hashes by default,
    raw content only when explicitly configured.
    """
    # Primary identifiers
    event_id: str
    event_type: EventType
    occurred_at: str  # ISO 8601 timestamp

    # Actor context
    org_id: str
    user_id: str
    session_id: str
    device_id: Optional[str] = None
    browser_id: Optional[str] = None

    # Tool context
    tool_name: Optional[str] = None
    tool_domain: Optional[str] = None
    tool_category: Optional[ToolCategory] = None
    tool_approved: Optional[bool] = None

    # Content hashes (privacy-preserving)
    page_url_hash: Optional[str] = None
    prompt_hash: Optional[str] = None
    output_hash: Optional[str] = None
    uploaded_file_hashes: List[str] = field(default_factory=list)

    # Classification & policy
    data_classification: Optional[DataClassification] = None
    sensitivity_score: Optional[float] = None  # 0.0-1.0
    policy_decision: Optional[PolicyDecision] = None
    policy_rule_id: Optional[str] = None
    policy_reason: Optional[str] = None

    # Evidence
    raw_content_ref: Optional[str] = None  # Reference to stored content (if any)
    signature: Optional[str] = None  # Cryptographic signature
    signature_algorithm: Optional[str] = None
    witness_hash: Optional[str] = None  # Merkle tree inclusion

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        event_type: EventType,
        org_id: str,
        user_id: str,
        session_id: str,
        tool_name: Optional[str] = None,
        tool_domain: Optional[str] = None,
        **kwargs
    ) -> WebAIEvent:
        """
        Create a new WebAIEvent with auto-generated ID and timestamp.

        Args:
            event_type: Type of event
            org_id: Organization identifier
            user_id: User identifier
            session_id: Session identifier
            tool_name: AI tool name (optional)
            tool_domain: Tool domain (optional)
            **kwargs: Additional event attributes

        Returns:
            WebAIEvent instance
        """
        return cls(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            occurred_at=utc_now_iso(),
            org_id=org_id,
            user_id=user_id,
            session_id=session_id,
            tool_name=tool_name,
            tool_domain=tool_domain,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        result = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "occurred_at": self.occurred_at,
            "org_id": self.org_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }

        # Optional fields
        if self.device_id:
            result["device_id"] = self.device_id
        if self.browser_id:
            result["browser_id"] = self.browser_id
        if self.tool_name:
            result["tool_name"] = self.tool_name
        if self.tool_domain:
            result["tool_domain"] = self.tool_domain
        if self.tool_category:
            result["tool_category"] = self.tool_category.value
        if self.tool_approved is not None:
            result["tool_approved"] = self.tool_approved

        # Hashes
        if self.page_url_hash:
            result["page_url_hash"] = self.page_url_hash
        if self.prompt_hash:
            result["prompt_hash"] = self.prompt_hash
        if self.output_hash:
            result["output_hash"] = self.output_hash
        if self.uploaded_file_hashes:
            result["uploaded_file_hashes"] = self.uploaded_file_hashes

        # Classification
        if self.data_classification:
            result["data_classification"] = self.data_classification.value
        if self.sensitivity_score is not None:
            result["sensitivity_score"] = self.sensitivity_score
        if self.policy_decision:
            result["policy_decision"] = self.policy_decision.value
        if self.policy_rule_id:
            result["policy_rule_id"] = self.policy_rule_id
        if self.policy_reason:
            result["policy_reason"] = self.policy_reason

        # Evidence
        if self.raw_content_ref:
            result["raw_content_ref"] = self.raw_content_ref
        if self.signature:
            result["signature"] = self.signature
        if self.signature_algorithm:
            result["signature_algorithm"] = self.signature_algorithm
        if self.witness_hash:
            result["witness_hash"] = self.witness_hash

        # Metadata
        if self.metadata:
            result["metadata"] = self.metadata
        if self.tags:
            result["tags"] = self.tags

        return result

    def is_shadow_ai(self) -> bool:
        """Check if this event represents shadow AI usage."""
        return self.tool_approved is False

    def is_high_risk(self, threshold: float = 0.7) -> bool:
        """
        Check if event is high risk based on sensitivity score.

        Args:
            threshold: Sensitivity threshold (default 0.7)

        Returns:
            True if sensitivity score exceeds threshold
        """
        return self.sensitivity_score is not None and self.sensitivity_score >= threshold

    def was_blocked(self) -> bool:
        """Check if action was blocked by policy."""
        return self.policy_decision == PolicyDecision.BLOCK

    def needs_review(self) -> bool:
        """Check if event requires manual review."""
        return (
            self.policy_decision == PolicyDecision.ESCALATE or
            self.is_shadow_ai() or
            self.is_high_risk()
        )


@dataclass
class EventBatch:
    """
    Batch of events for efficient processing.

    Used for bulk operations like batch classification,
    policy evaluation, or vault storage.
    """
    batch_id: str
    events: List[WebAIEvent]
    created_at: str
    org_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, events: List[WebAIEvent], org_id: str) -> EventBatch:
        """Create a new event batch."""
        return cls(
            batch_id=str(uuid.uuid4()),
            events=events,
            created_at=utc_now_iso(),
            org_id=org_id,
        )

    def count(self) -> int:
        """Count events in batch."""
        return len(self.events)

    def filter_by_type(self, event_type: EventType) -> List[WebAIEvent]:
        """Filter events by type."""
        return [e for e in self.events if e.event_type == event_type]

    def filter_shadow_ai(self) -> List[WebAIEvent]:
        """Get shadow AI events."""
        return [e for e in self.events if e.is_shadow_ai()]

    def filter_high_risk(self, threshold: float = 0.7) -> List[WebAIEvent]:
        """Get high-risk events."""
        return [e for e in self.events if e.is_high_risk(threshold)]


def utc_now_iso() -> str:
    """Get current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "WebAIEvent",
    "EventBatch",
    "EventType",
    "PolicyDecision",
    "DataClassification",
    "ToolCategory",
    "utc_now_iso",
]
