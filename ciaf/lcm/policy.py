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
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

from ..core import secure_random_bytes, sha256_hash

if TYPE_CHECKING:
    from ..core.interfaces import Signer, RNG, Merkle, AnchorDeriver, AnchorStore


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
    Now includes protocol implementations for dependency injection.
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
    
    # Protocol implementations (optional, for dependency injection)
    rng: Optional["RNG"] = None
    anchor_deriver: Optional["AnchorDeriver"] = None
    anchor_store: Optional["AnchorStore"] = None
    signer: Optional["Signer"] = None
    merkle_factory: Optional[Any] = None  # Callable that returns Merkle instances
    
    def __post_init__(self):
        """Initialize default values and protocols."""
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
        
        # Initialize default protocols if not provided
        if not any([self.rng, self.anchor_deriver, self.anchor_store, self.signer, self.merkle_factory]):
            self._init_default_protocols()
    
    def _init_default_protocols(self):
        """Initialize default protocol implementations."""
        # Import here to avoid circular imports
        from .protocol_implementations import create_default_protocols
        defaults = create_default_protocols()
        
        if self.rng is None:
            self.rng = defaults['rng']
        if self.anchor_deriver is None:
            self.anchor_deriver = defaults['anchor_deriver']
        if self.anchor_store is None:
            self.anchor_store = defaults['anchor_store']
        if self.signer is None:
            self.signer = defaults['signer']
        if self.merkle_factory is None:
            self.merkle_factory = defaults['merkle_factory']
    
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
        return canonical_json(self.to_dict())
    
    def policy_digest(self) -> str:
        """Get digest of the policy itself."""
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


def create_commitment(
    data: Any, 
    commitment_type: CommitmentType, 
    anchor: bytes = None,
    rng: Optional["RNG"] = None
) -> str:
    """
    Create commitment for data according to the specified type.
    
    Args:
        data: Data to create commitment for
        commitment_type: Type of commitment to create
        anchor: Optional anchor bytes for HMAC commitments
        rng: Optional RNG implementation (uses default if None)
        
    Returns:
        Commitment string
    """
    if commitment_type == CommitmentType.PLAINTEXT:
        return str(data)
    elif commitment_type == CommitmentType.SALTED:
        # Use RNG protocol if provided, fallback to direct import
        if rng is not None:
            salt = rng.random_bytes(16)
        else:
            salt = secure_random_bytes(16)
        data_str = json.dumps(data, sort_keys=True) if not isinstance(data, str) else data
        return sha256_hash((salt + data_str.encode('utf-8')))[:16] + "..."
    elif commitment_type == CommitmentType.HMAC_SHA256:
        # HMAC-based commitment using provided anchor
        if anchor is None:
            raise ValueError("HMAC commitment requires anchor bytes")
        import hmac
        data_str = json.dumps(data, sort_keys=True) if not isinstance(data, str) else data
        return hmac.new(anchor, data_str.encode('utf-8'), 'sha256').hexdigest()[:16] + "..."
    else:
        raise ValueError(f"Unknown commitment type: {commitment_type}")


def canonical_json(data: Any) -> str:
    """
    Create canonical JSON representation for consistent hashing.
    
    Args:
        data: Data to serialize
        
    Returns:
        Canonical JSON string with consistent formatting
    """
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


def canonical_hash(data: Any) -> str:
    """
    Create canonical hash of data via JSON serialization.
    
    Args:
        data: Data to hash
        
    Returns:
        SHA-256 hash of canonical JSON representation
    """
    return sha256_hash(canonical_json(data).encode('utf-8'))
