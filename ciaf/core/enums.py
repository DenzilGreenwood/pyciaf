"""
Enums for the Cognitive Insight Audit Framework.

This module centralizes all enum definitions for record types, algorithms,
and other categorical constants used throughout the CIAF core.

Created: 2025-09-26
Author: Denzil James Greenwood
Version: 1.0.0
"""

from enum import Enum


class RecordType(str, Enum):
    """Types of audit records."""
    DATASET = "dataset"
    MODEL = "model"
    INFERENCE = "inference"
    ANCHOR = "anchor"
    MONITORING = "monitoring"
    COMPLIANCE = "compliance"


class HashAlgorithm(str, Enum):
    """Supported hash algorithms for algorithm agility."""
    SHA256 = "sha256"
    SHA3_256 = "sha3-256"
    BLAKE3 = "blake3"


class SignatureAlgorithm(str, Enum):
    """Supported signature algorithms."""
    ED25519 = "ed25519"  # Production default
    MOCK = "mock"        # Legacy/testing only ## DEPRECATED ## DO NOT USE