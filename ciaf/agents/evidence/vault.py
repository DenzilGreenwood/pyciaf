"""
Evidence Vault implementation for cryptographic receipts.

Generates tamper-evident receipts with HMAC signatures and hash chaining,
following CIAF's audit trail patterns.

Created: 2026-03-28
Author: Denzil James Greenwood
Version: 1.0.0
"""

import hashlib
import hmac
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from ..core.interfaces import EvidenceRecorder
from ..core.types import ActionReceipt, ExecutionResult, PrincipalType


class EvidenceVault(EvidenceRecorder):
    """
    Cryptographic evidence vault for agent action receipts.

    Implements CIAF's hash-chaining pattern for tamper-evident audit trails.
    Each receipt is cryptographically signed and chained to the previous receipt.
    """

    def __init__(self, signing_secret: str = ""):
        """
        Initialize the evidence vault.

        Args:
            signing_secret: Secret key for HMAC-SHA256 signatures.
                          If empty, a random key will be generated.
        """
        if not signing_secret:
            signing_secret = uuid.uuid4().hex

        self._signing_secret = signing_secret.encode()
        self._receipts: List[ActionReceipt] = []
        self._receipts_by_principal: dict[str, List[ActionReceipt]] = {}
        self._last_receipt_hash = "0" * 64  # Genesis hash

    def record_action(self, result: ExecutionResult) -> ActionReceipt:
        """
        Record an action execution and generate a cryptographic receipt.

        Args:
            result: The execution result to record

        Returns:
            Signed and chained ActionReceipt
        """
        receipt_id = f"rcpt-{uuid.uuid4().hex[:16]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        # Get identity info
        principal_id = (
            result.request.requested_by.principal_id
            if result.request.requested_by
            else "unknown"
        )
        principal_type = (
            result.request.requested_by.principal_type
            if result.request.requested_by
            else PrincipalType.SYSTEM  # Use SYSTEM for unknown principals
        )

        # Create receipt
        receipt = ActionReceipt(
            receipt_id=receipt_id,
            timestamp=timestamp,
            principal_id=principal_id,
            principal_type=principal_type,
            action=result.request.action,
            resource_id=result.request.resource.resource_id,
            resource_type=result.request.resource.resource_type,
            correlation_id=result.request.correlation_id,
            decision=result.allowed,
            reason=result.reason,
            elevation_grant_id=result.elevation_grant_id,
            approved_by=(
                result.request.requested_by.principal_id
                if result.elevation_grant_id and result.request.requested_by
                else None
            ),
            params_hash=result.request.get_params_hash(),
            policy_obligations=result.policy_obligations,
            prior_receipt_hash=self._last_receipt_hash,
            signature="",  # Will be computed below
        )

        # Generate signature (HMAC-SHA256)
        receipt_hash = receipt.get_receipt_hash()
        signature_input = f"{receipt_hash}:{self._last_receipt_hash}".encode()
        signature = hmac.new(
            self._signing_secret, signature_input, hashlib.sha256
        ).hexdigest()

        # Update receipt with signature
        receipt.signature = f"hmac-sha256:{signature}"

        # Store receipt
        self._receipts.append(receipt)

        # Index by principal
        if principal_id not in self._receipts_by_principal:
            self._receipts_by_principal[principal_id] = []
        self._receipts_by_principal[principal_id].append(receipt)

        # Update chain
        self._last_receipt_hash = receipt_hash

        return receipt

    def verify_receipt(self, receipt: ActionReceipt) -> bool:
        """
        Verify a receipt's cryptographic integrity.

        Args:
            receipt: The receipt to verify

        Returns:
            True if receipt is cryptographically valid, False otherwise
        """
        # Recompute receipt hash
        expected_hash = receipt.get_receipt_hash()

        # Verify signature
        if not receipt.signature.startswith("hmac-sha256:"):
            return False

        signature_hex = receipt.signature.replace("hmac-sha256:", "")
        signature_input = f"{expected_hash}:{receipt.prior_receipt_hash}".encode()

        expected_signature = hmac.new(
            self._signing_secret, signature_input, hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(signature_hex, expected_signature)

    def get_receipts_by_principal(self, principal_id: str) -> List[ActionReceipt]:
        """
        Retrieve all receipts for a given principal.

        Args:
            principal_id: The principal identifier

        Returns:
            List of receipts for the principal
        """
        return self._receipts_by_principal.get(principal_id, [])

    def get_all_receipts(self) -> List[ActionReceipt]:
        """Get all receipts in chronological order."""
        return self._receipts.copy()

    def get_denied_receipts(self) -> List[ActionReceipt]:
        """Get all receipts where the decision was denied."""
        return [r for r in self._receipts if not r.decision]

    def get_receipts_by_action(self, action: str) -> List[ActionReceipt]:
        """Get all receipts for a specific action."""
        return [r for r in self._receipts if r.action == action]

    def get_receipts_by_correlation(self, correlation_id: str) -> List[ActionReceipt]:
        """Get all receipts with a specific correlation ID."""
        return [r for r in self._receipts if r.correlation_id == correlation_id]

    def verify_chain(self) -> bool:
        """
        Verify the complete receipt chain integrity.

        Checks that all receipts are correctly chained and signed.

        Returns:
            True if entire chain is valid, False if any receipt fails
        """
        if not self._receipts:
            return True

        expected_prior_hash = "0" * 64  # Genesis hash

        for receipt in self._receipts:
            # Verify chain link
            if receipt.prior_receipt_hash != expected_prior_hash:
                return False

            # Verify receipt signature
            if not self.verify_receipt(receipt):
                return False

            # Update expected prior hash for next iteration
            expected_prior_hash = receipt.get_receipt_hash()

        return True

    def get_receipt_by_id(self, receipt_id: str) -> Optional[ActionReceipt]:
        """
        Find a receipt by its ID.

        Args:
            receipt_id: The receipt identifier

        Returns:
            ActionReceipt if found, None otherwise
        """
        for receipt in self._receipts:
            if receipt.receipt_id == receipt_id:
                return receipt
        return None

    def export_receipts(self) -> List[dict]:
        """
        Export all receipts as dictionaries for serialization.

        Returns:
            List of receipt dictionaries
        """
        return [
            {
                "receipt_id": r.receipt_id,
                "timestamp": r.timestamp,
                "principal_id": r.principal_id,
                "principal_type": r.principal_type.value,
                "action": r.action,
                "resource_id": r.resource_id,
                "resource_type": r.resource_type,
                "correlation_id": r.correlation_id,
                "decision": r.decision,
                "reason": r.reason,
                "elevation_grant_id": r.elevation_grant_id,
                "approved_by": r.approved_by,
                "params_hash": r.params_hash,
                "policy_obligations": r.policy_obligations,
                "prior_receipt_hash": r.prior_receipt_hash,
                "signature": r.signature,
            }
            for r in self._receipts
        ]

    def clear(self) -> None:
        """Clear all receipts (useful for testing)."""
        self._receipts.clear()
        self._receipts_by_principal.clear()
        self._last_receipt_hash = "0" * 64
