"""
CIAF LCM Policy Framework

Defines the canonical policies for hashing, domains, Merkle trees, and commitments
used throughout the CIAF Lazy Capsule Materialization system.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


class DomainType(Enum):
    """CIAF domain types for anchoring."""
    DATASET = "CIAF|dataset"
    DATASET_FAMILY = "CIAF|dataset|family"
    DATASET_SPLIT = "CIAF|dataset|split"
    MODEL = "CIAF|model" 
    TRAIN = "CIAF|train"
    DEPLOYMENT = "CIAF|deployment"
    INFERENCE = "CIAF|inference"


class CommitmentType(Enum):
    """Commitment types for privacy protection."""
    SALTED = "salted"
    HMAC_SHA256 = "HMAC-SHA256"
    PLAINTEXT = "plaintext"  # For non-sensitive data


@dataclass
class MerklePolicy:
    """Merkle tree construction policy."""
    fanout: int = 2
    padding: str = "duplicate_last"
    leaf_encoding: str = "raw32"


@dataclass  
class LCMPolicy:
    """
    Comprehensive CIAF LCM policy defining all cryptographic and structural policies.
    """
    
    # Core policy
    hash_algorithm: str = "SHA-256"
    canonicalization: str = "json(sorted,utf-8)"
    domains: List[DomainType] = None
    merkle: MerklePolicy = None
    commitments: CommitmentType = CommitmentType.SALTED
    
    # Schema versions
    anchor_schema_version: str = "1.0"
    merkle_policy_version: str = "1.0"
    
    def __post_init__(self):
        """Initialize default values."""
        if self.domains is None:
            self.domains = [
                DomainType.DATASET_FAMILY,
                DomainType.DATASET_SPLIT,
                DomainType.MODEL,
                DomainType.TRAIN,
                DomainType.DEPLOYMENT,
                DomainType.INFERENCE
            ]
        if self.merkle is None:
            self.merkle = MerklePolicy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "hash": self.hash_algorithm,
            "canon": self.canonicalization,
            "domains": [d.value for d in self.domains],
            "merkle": {
                "fanout": self.merkle.fanout,
                "padding": self.merkle.padding,
                "leaf_encoding": self.merkle.leaf_encoding
            },
            "commitments": self.commitments.value,
            "anchor_schema_version": self.anchor_schema_version,
            "merkle_policy_version": self.merkle_policy_version
        }
    
    def canonical_json(self) -> str:
        """Get canonical JSON representation."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))
    
    def policy_digest(self) -> str:
        """Get digest of the policy itself."""
        from ..core import sha256_hash
        return sha256_hash(self.canonical_json().encode('utf-8'))
    
    def format_policy_line(self) -> str:
        """Format policy line for pretty printing."""
        return (f"policy: hash={self.hash_algorithm} | canon={self.canonicalization} | "
                f"domains={[d.value.split('|')[1] for d in self.domains]}")
    
    def format_detailed_policy(self) -> str:
        """Format detailed policy for logging."""
        return (f"hash={self.hash_algorithm} | canon={self.canonicalization}\n"
                f"merkle: fanout={self.merkle.fanout}, padding={self.merkle.padding}, leaf_encoding={self.merkle.leaf_encoding}\n"
                f"commitments: default={self.commitments.value}")
    
    @classmethod
    def default(cls) -> "LCMPolicy":
        """Get default CIAF LCM policy."""
        return cls()


# Global default policy instance
DEFAULT_LCM_POLICY = LCMPolicy.default()


def get_default_policy() -> LCMPolicy:
    """Get the default LCM policy."""
    return DEFAULT_LCM_POLICY


def set_default_policy(policy: LCMPolicy) -> None:
    """Set the global default LCM policy."""
    global DEFAULT_LCM_POLICY
    DEFAULT_LCM_POLICY = policy
