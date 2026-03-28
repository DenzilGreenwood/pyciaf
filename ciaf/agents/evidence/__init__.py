"""
Evidence vault for cryptographic receipts of agent actions.

Provides tamper-evident audit trails with hash-chained receipts
integrated with CIAF's core cryptographic primitives.

Created: 2026-03-28
Author: Denzil James Greenwood
Version: 1.0.0
"""

from .vault import EvidenceVault

__all__ = ["EvidenceVault"]
