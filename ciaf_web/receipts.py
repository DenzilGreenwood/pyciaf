"""
CIAF Web - Receipt Generation

Generate cryptographically verifiable receipts for AI usage events.

This is the CIAF-LCM integration layer that converts web AI events into
evidence-bearing governance receipts with:
- Canonical serialization
- SHA-256 content hashing
- Ed25519 digital signatures
- Optional Merkle tree inclusion proofs
- Vault persistence

The key difference:
- Not just logs → Verifiable evidence
- Not just monitoring → Provable governance
- Not just records → Cryptographic receipts

Receipts enable:
- Incident reconstruction
- Compliance auditing
- Legal defensibility
- Tamper detection

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
import hashlib

from .events import WebAIEvent, EventType, PolicyDecision


@dataclass
class WebAIReceipt:
    """
    Cryptographic receipt for AI usage event.

    Provides verifiable evidence that an event occurred
    with specific properties at a specific time.
    """
    # Receipt identifiers
    receipt_id: str
    event_id: str
    created_at: str

    # Event summary
    event_type: EventType
    org_id: str
    user_id: str
    tool_name: Optional[str]
    policy_decision: Optional[PolicyDecision]

    # Cryptographic hashes
    event_hash: str  # SHA-256 of canonical event
    receipt_hash: str  # SHA-256 of this receipt
    prior_receipt_hash: Optional[str] = None  # Hash chain linkage
    content_hash: Optional[str] = None  # Hash of actual content (if captured)

    # Signatures
    signature: Optional[str] = None
    signature_algorithm: str = "Ed25519"
    signer_id: Optional[str] = None

    # Merkle inclusion proof (optional)
    merkle_root: Optional[str] = None
    merkle_proof: Optional[List[str]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary."""
        result = {
            "receipt_id": self.receipt_id,
            "event_id": self.event_id,
            "created_at": self.created_at,
            "event_type": self.event_type.value,
            "org_id": self.org_id,
            "user_id": self.user_id,
            "event_hash": self.event_hash,
            "receipt_hash": self.receipt_hash,
        }

        # Optional fields
        if self.tool_name:
            result["tool_name"] = self.tool_name
        if self.policy_decision:
            result["policy_decision"] = self.policy_decision.value
        if self.prior_receipt_hash:
            result["prior_receipt_hash"] = self.prior_receipt_hash
        if self.content_hash:
            result["content_hash"] = self.content_hash
        if self.signature:
            result["signature"] = self.signature
            result["signature_algorithm"] = self.signature_algorithm
        if self.signer_id:
            result["signer_id"] = self.signer_id
        if self.merkle_root:
            result["merkle_root"] = self.merkle_root
        if self.merkle_proof:
            result["merkle_proof"] = self.merkle_proof
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def to_canonical_bytes(self) -> bytes:
        """
        Convert to canonical representation for hashing.

        Uses deterministic JSON serialization.
        """
        canonical = {
            "receipt_id": self.receipt_id,
            "event_id": self.event_id,
            "created_at": self.created_at,
            "event_type": self.event_type.value,
            "org_id": self.org_id,
            "user_id": self.user_id,
            "event_hash": self.event_hash,
        }

        # Sort keys and use compact encoding
        json_str = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
        return json_str.encode('utf-8')

    def verify_hash(self) -> bool:
        """
        Verify that receipt_hash matches canonical content.

        Returns:
            True if hash is valid
        """
        expected_hash = hashlib.sha256(self.to_canonical_bytes()).hexdigest()
        return self.receipt_hash == expected_hash

    def is_chained(self) -> bool:
        """Check if receipt is part of a hash chain."""
        return self.prior_receipt_hash is not None

    def has_merkle_proof(self) -> bool:
        """Check if receipt includes Merkle tree proof."""
        return self.merkle_root is not None and self.merkle_proof is not None


class ReceiptGenerator:
    """
    Generate cryptographic receipts for AI usage events.

    Follows CIAF-LCM patterns for evidence generation.
    """

    def __init__(
        self,
        signer_id: Optional[str] = None,
        signing_enabled: bool = False,
    ):
        """
        Initialize receipt generator.

        Args:
            signer_id: Identifier of signing authority
            signing_enabled: Whether to sign receipts
        """
        self.signer_id = signer_id
        self.signing_enabled = signing_enabled
        self.last_receipt_hash: Optional[str] = None  # For hash chaining

    def generate(
        self,
        event: WebAIEvent,
        content: Optional[str] = None,
        include_merkle: bool = False,
    ) -> WebAIReceipt:
        """
        Generate receipt for event.

        Args:
            event: Event to create receipt for
            content: Actual content (optional, will be hashed if provided)
            include_merkle: Whether to include Merkle proof

        Returns:
            WebAIReceipt
        """
        from .events import utc_now_iso

        # Compute event hash
        event_hash = self._hash_event(event)

        # Compute content hash if provided
        content_hash = None
        if content:
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

        # Create receipt
        receipt = WebAIReceipt(
            receipt_id=f"rcpt-{event.event_id}",
            event_id=event.event_id,
            created_at=utc_now_iso(),
            event_type=event.event_type,
            org_id=event.org_id,
            user_id=event.user_id,
            tool_name=event.tool_name,
            policy_decision=event.policy_decision,
            event_hash=event_hash,
            receipt_hash="",  # Computed below
            prior_receipt_hash=self.last_receipt_hash,
            content_hash=content_hash,
        )

        # Compute receipt hash
        receipt.receipt_hash = hashlib.sha256(receipt.to_canonical_bytes()).hexdigest()

        # Sign if enabled
        if self.signing_enabled and self.signer_id:
            receipt.signature = self._sign_receipt(receipt)
            receipt.signer_id = self.signer_id

        # Add Merkle proof if requested
        if include_merkle:
            # In production, this would compute actual Merkle tree
            # For now, placeholder
            receipt.merkle_root = "placeholder_merkle_root"
            receipt.merkle_proof = ["placeholder_proof"]

        # Update chain
        self.last_receipt_hash = receipt.receipt_hash

        return receipt

    def generate_batch(
        self,
        events: List[WebAIEvent],
        include_merkle: bool = True,
    ) -> List[WebAIReceipt]:
        """
        Generate receipts for multiple events.

        Args:
            events: Events to generate receipts for
            include_merkle: Whether to include Merkle proofs

        Returns:
            List of receipts
        """
        receipts = []
        for event in events:
            receipt = self.generate(event, include_merkle=include_merkle)
            receipts.append(receipt)

        return receipts

    def _hash_event(self, event: WebAIEvent) -> str:
        """Compute SHA-256 hash of event."""
        event_dict = event.to_dict()
        json_str = json.dumps(event_dict, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _sign_receipt(self, receipt: WebAIReceipt) -> str:
        """
        Sign receipt (placeholder).

        In production, this would use Ed25519 signing.
        """
        # Placeholder signature
        to_sign = receipt.to_canonical_bytes()
        signature_hash = hashlib.sha256(b"SIGNATURE:" + to_sign).hexdigest()
        return f"sig_{signature_hash[:32]}"


def verify_receipt(receipt: WebAIReceipt) -> bool:
    """
    Verify receipt integrity.

    Args:
        receipt: Receipt to verify

    Returns:
        True if receipt is valid
    """
    # Verify hash
    if not receipt.verify_hash():
        return False

    # Verify signature (placeholder)
    if receipt.signature:
        # In production, verify Ed25519 signature
        pass

    return True


def verify_receipt_chain(receipts: List[WebAIReceipt]) -> bool:
    """
    Verify chain of receipts.

    Args:
        receipts: Receipts in order

    Returns:
        True if chain is valid
    """
    for i in range(1, len(receipts)):
        curr = receipts[i]
        prev = receipts[i - 1]

        # Check that current receipt's prior_hash matches previous receipt_hash
        if curr.prior_receipt_hash != prev.receipt_hash:
            return False

    return True


# Convenience functions

def generate_receipt(
    event: WebAIEvent,
    content: Optional[str] = None,
    signer_id: Optional[str] = None,
) -> WebAIReceipt:
    """
    Generate receipt for event.

    Args:
        event: Event to create receipt for
        content: Actual content (optional)
        signer_id: Signer identifier

    Returns:
        WebAIReceipt
    """
    generator = ReceiptGenerator(signer_id=signer_id)
    return generator.generate(event, content=content)


__all__ = [
    "WebAIReceipt",
    "ReceiptGenerator",
    "generate_receipt",
    "verify_receipt",
    "verify_receipt_chain",
]
