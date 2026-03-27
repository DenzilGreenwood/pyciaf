"""
CIAF Web - Event Collectors

Browser-side collectors for capturing AI usage events.

Collection methods:
- Browser extension integration
- Proxy-based capture
- Local agent monitoring
- JavaScript injection

Collectors normalize events from various sources into WebAIEvent format.

Architecture:
1. Detect AI interaction (detectors.py)
2. Classify content (classifiers.py)
3. Evaluate policy (policy.py)
4. Generate event (collectors.py)
5. Create receipt (receipts.py)
6. Store evidence (vault_adapter.py)

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import hashlib

from .events import WebAIEvent, EventType
from .detectors import AIToolDetector, DetectionResult
from .classifiers import ContentClassifier, ClassificationResult
from .policy import PolicyEngine, PolicyResult, PolicyContext
from .receipts import ReceiptGenerator, WebAIReceipt
from .vault_adapter import WebAIVaultAdapter
from .redaction import ContentRedactor


@dataclass
class CollectionConfig:
    """
    Configuration for event collection.

    Controls what data is captured and how it's processed.
    """

    # Detection
    approved_tools: set[str] = None
    detect_shadow_ai: bool = True

    # Classification
    classify_content: bool = True
    min_content_length: int = 10  # Don't classify very short content

    # Policy
    enforce_policy: bool = True
    policy_rules: list = None

    # Capture
    capture_raw_content: bool = False  # By default, only hash
    hash_content: bool = True
    redact_sensitive: bool = True

    # Storage
    store_events: bool = True
    store_receipts: bool = True
    generate_receipts: bool = True

    # Privacy
    anonymize_user_id: bool = False

    def __post_init__(self):
        if self.approved_tools is None:
            self.approved_tools = set()


class EventCollector:
    """
    Collect and process AI usage events.

    Main entry point for browser extensions and agents to
    submit AI interaction events for governance processing.
    """

    def __init__(
        self,
        config: Optional[CollectionConfig] = None,
        vault_adapter: Optional[WebAIVaultAdapter] = None,
    ):
        """
        Initialize event collector.

        Args:
            config: Collection configuration
            vault_adapter: Vault storage adapter
        """
        self.config = config or CollectionConfig()
        self.vault_adapter = vault_adapter or WebAIVaultAdapter()

        # Initialize components
        self.detector = AIToolDetector(self.config.approved_tools)
        self.classifier = ContentClassifier()
        self.policy_engine = PolicyEngine(rules=self.config.policy_rules)
        self.receipt_generator = ReceiptGenerator(signing_enabled=True)
        self.redactor = ContentRedactor()

    def collect_prompt_submit(
        self,
        org_id: str,
        user_id: str,
        session_id: str,
        url: str,
        prompt: str,
        device_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CollectionResult:
        """
        Collect prompt submission event.

        Args:
            org_id: Organization identifier
            user_id: User identifier
            session_id: Session identifier
            url: Page URL where prompt was submitted
            prompt: Prompt text
            device_id: Device identifier
            metadata: Additional metadata

        Returns:
            CollectionResult with event, policy decision, and receipt
        """
        # Detect AI tool
        detection = self.detector.detect(url)

        # Classify content
        classification = None
        if (
            self.config.classify_content
            and len(prompt) >= self.config.min_content_length
        ):
            classification = self.classifier.classify(prompt)

        # Hash prompt
        prompt_hash = (
            hashlib.sha256(prompt.encode()).hexdigest()
            if self.config.hash_content
            else None
        )

        # Create event
        event = WebAIEvent.create(
            event_type=EventType.PROMPT_SUBMIT,
            org_id=org_id,
            user_id=user_id,
            session_id=session_id,
            device_id=device_id,
            tool_name=detection.tool_name if detection else None,
            tool_domain=detection.tool_domain if detection else None,
            tool_category=detection.tool_category if detection else None,
            tool_approved=detection.tool_approved if detection else None,
            page_url_hash=hashlib.sha256(url.encode()).hexdigest(),
            prompt_hash=prompt_hash,
            data_classification=(
                classification.classification if classification else None
            ),
            sensitivity_score=(
                classification.sensitivity_score if classification else None
            ),
            metadata=metadata or {},
        )

        # Evaluate policy
        policy_result = None
        if self.config.enforce_policy:
            policy_result = self.policy_engine.evaluate(event)
            event.policy_decision = policy_result.decision
            event.policy_rule_id = (
                policy_result.matched_rule.rule_id
                if policy_result.matched_rule
                else None
            )
            event.policy_reason = policy_result.reason

        # Redact if needed
        content_to_store = None
        if self.config.capture_raw_content:
            if (
                self.config.redact_sensitive
                and classification
                and classification.is_restricted()
            ):
                content_to_store = self.redactor.redact(prompt)
            else:
                content_to_store = prompt

        # Generate receipt
        receipt = None
        if self.config.generate_receipts:
            receipt = self.receipt_generator.generate(event, content=content_to_store)

        # Store
        if self.config.store_events:
            self.vault_adapter.store_event(event)

        if self.config.store_receipts and receipt:
            self.vault_adapter.store_receipt(receipt)

        return CollectionResult(
            event=event,
            detection=detection,
            classification=classification,
            policy_result=policy_result,
            receipt=receipt,
            action_allowed=(policy_result.is_allowed() if policy_result else True),
        )

    def collect_file_upload(
        self,
        org_id: str,
        user_id: str,
        session_id: str,
        url: str,
        file_name: str,
        file_size: int,
        file_content: Optional[str] = None,
        device_id: Optional[str] = None,
    ) -> CollectionResult:
        """
        Collect file upload event.

        Args:
            org_id: Organization identifier
            user_id: User identifier
            session_id: Session identifier
            url: Upload URL
            file_name: Name of uploaded file
            file_size: File size in bytes
            file_content: File content (optional)
            device_id: Device identifier

        Returns:
            CollectionResult
        """
        # Detect tool
        detection = self.detector.detect(url)

        # Classify content if provided
        classification = None
        if file_content and self.config.classify_content:
            classification = self.classifier.classify(file_content)

        # Hash file content
        file_hash = None
        if file_content and self.config.hash_content:
            file_hash = hashlib.sha256(file_content.encode()).hexdigest()

        # Create event
        event = WebAIEvent.create(
            event_type=EventType.FILE_UPLOAD,
            org_id=org_id,
            user_id=user_id,
            session_id=session_id,
            device_id=device_id,
            tool_name=detection.tool_name if detection else None,
            tool_domain=detection.tool_domain if detection else None,
            tool_approved=detection.tool_approved if detection else None,
            page_url_hash=hashlib.sha256(url.encode()).hexdigest(),
            uploaded_file_hashes=[file_hash] if file_hash else [],
            data_classification=(
                classification.classification if classification else None
            ),
            sensitivity_score=(
                classification.sensitivity_score if classification else None
            ),
            metadata={
                "file_name": file_name,
                "file_size": file_size,
            },
        )

        # Evaluate policy
        policy_result = None
        if self.config.enforce_policy:
            context = PolicyContext(content_size=file_size)
            policy_result = self.policy_engine.evaluate(event, context)
            event.policy_decision = policy_result.decision
            event.policy_rule_id = (
                policy_result.matched_rule.rule_id
                if policy_result.matched_rule
                else None
            )

        # Generate receipt
        receipt = None
        if self.config.generate_receipts:
            receipt = self.receipt_generator.generate(event)

        # Store
        if self.config.store_events:
            self.vault_adapter.store_event(event)

        if self.config.store_receipts and receipt:
            self.vault_adapter.store_receipt(receipt)

        return CollectionResult(
            event=event,
            detection=detection,
            classification=classification,
            policy_result=policy_result,
            receipt=receipt,
            action_allowed=(policy_result.is_allowed() if policy_result else True),
        )

    def collect_output_receive(
        self,
        org_id: str,
        user_id: str,
        session_id: str,
        url: str,
        output: str,
        device_id: Optional[str] = None,
    ) -> CollectionResult:
        """
        Collect AI output reception event.

        Args:
            org_id: Organization identifier
            user_id: User identifier
            session_id: Session identifier
            url: Page URL
            output: AI-generated output
            device_id: Device identifier

        Returns:
            CollectionResult
        """
        # Similar to prompt_submit but for outputs
        detection = self.detector.detect(url)

        # Hash output
        output_hash = (
            hashlib.sha256(output.encode()).hexdigest()
            if self.config.hash_content
            else None
        )

        # Create event
        event = WebAIEvent.create(
            event_type=EventType.OUTPUT_RECEIVE,
            org_id=org_id,
            user_id=user_id,
            session_id=session_id,
            device_id=device_id,
            tool_name=detection.tool_name if detection else None,
            tool_domain=detection.tool_domain if detection else None,
            tool_approved=detection.tool_approved if detection else None,
            page_url_hash=hashlib.sha256(url.encode()).hexdigest(),
            output_hash=output_hash,
        )

        # Store
        if self.config.store_events:
            self.vault_adapter.store_event(event)

        # Generate receipt
        receipt = None
        if self.config.generate_receipts:
            receipt = self.receipt_generator.generate(event)
            if self.config.store_receipts:
                self.vault_adapter.store_receipt(receipt)

        return CollectionResult(
            event=event,
            detection=detection,
            receipt=receipt,
            action_allowed=True,
        )


@dataclass
class CollectionResult:
    """Result of event collection."""

    event: WebAIEvent
    detection: Optional[DetectionResult] = None
    classification: Optional[ClassificationResult] = None
    policy_result: Optional[PolicyResult] = None
    receipt: Optional[WebAIReceipt] = None
    action_allowed: bool = True

    def is_shadow_ai(self) -> bool:
        """Check if shadow AI was detected."""
        return self.event.is_shadow_ai()

    def is_high_risk(self) -> bool:
        """Check if high-risk content was detected."""
        return self.event.is_high_risk()

    def was_blocked(self) -> bool:
        """Check if action was blocked."""
        return not self.action_allowed

    def needs_review(self) -> bool:
        """Check if event needs manual review."""
        return self.event.needs_review()


# Convenience function for simple collection


def collect_ai_event(
    event_type: EventType,
    org_id: str,
    user_id: str,
    session_id: str,
    url: str,
    content: Optional[str] = None,
    approved_tools: Optional[set[str]] = None,
    **kwargs,
) -> CollectionResult:
    """
    Simple event collection function.

    Args:
        event_type: Type of event
        org_id: Organization ID
        user_id: User ID
        session_id: Session ID
        url: Page URL
        content: Content (prompt/output)
        approved_tools: Set of approved tool names
        **kwargs: Additional event attributes

    Returns:
        CollectionResult
    """
    config = CollectionConfig(approved_tools=approved_tools or set())
    collector = EventCollector(config=config)

    if event_type == EventType.PROMPT_SUBMIT and content:
        return collector.collect_prompt_submit(
            org_id, user_id, session_id, url, content, **kwargs
        )
    elif event_type == EventType.OUTPUT_RECEIVE and content:
        return collector.collect_output_receive(
            org_id, user_id, session_id, url, content, **kwargs
        )
    else:
        # Generic event
        detection = collector.detector.detect(url)
        event = WebAIEvent.create(
            event_type=event_type,
            org_id=org_id,
            user_id=user_id,
            session_id=session_id,
            tool_name=detection.tool_name if detection else None,
            tool_domain=detection.tool_domain if detection else None,
            tool_approved=detection.tool_approved if detection else None,
            **kwargs,
        )

        if config.store_events:
            collector.vault_adapter.store_event(event)

        return CollectionResult(event=event, detection=detection)


__all__ = [
    "EventCollector",
    "CollectionConfig",
    "CollectionResult",
    "collect_ai_event",
]
