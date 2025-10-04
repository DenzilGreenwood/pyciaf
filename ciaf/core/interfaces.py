"""
Interfaces (typing.Protocol) for the Cognitive Insight Audit Framework.

This module defines contracts for swappable components like signers, RNG sources,
Merkle trees, anchor derivers, and anchor stores. Using Protocol allows for
clean dependency injection and testing.

Created: 2025-09-26
Author: Denzil James Greenwood
Version: 1.0.0
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable, Optional, List, Tuple, Dict, Any


@runtime_checkable
class Signer(Protocol):
    """Protocol for digital signature implementations."""
    key_id: str
    
    def sign(self, data: bytes) -> str:
        """Sign data and return signature string."""
        ...
    
    def verify(self, data: bytes, signature: str) -> bool:
        """Verify signature against data."""
        ...


@runtime_checkable
class RNG(Protocol):
    """Protocol for random number generation sources."""
    
    def random_bytes(self, n: int) -> bytes:
        """Generate n cryptographically secure random bytes."""
        ...


@runtime_checkable
class Merkle(Protocol):
    """Protocol for Merkle tree implementations."""
    
    def add_leaf(self, leaf_hash: str) -> str:
        """Add leaf and return new root."""
        ...
    
    def get_root(self) -> str:
        """Get the current Merkle root hash."""
        ...
    
    def get_proof(self, leaf_hash: str) -> List[Tuple[str, str]]:
        """Get inclusion proof for a leaf as list of (hash, position) tuples."""
        ...
    
    def get_merkle_path(self, leaf_hash: str) -> List[Tuple[str, str]]:
        """Alias for get_proof to maintain backward compatibility."""
        ...
    
    def verify_proof(self, leaf_hash: str, proof: List[Tuple[str, str]], root: str) -> bool:
        """Verify Merkle inclusion proof."""
        ...


@runtime_checkable
class AnchorDeriver(Protocol):
    """Protocol for hierarchical anchor derivation."""
    
    def derive_master_anchor(self, password: str, salt: bytes) -> bytes:
        """Derive master anchor from password and salt."""
        ...
    
    def derive_dataset_anchor(self, master_anchor: bytes, dataset_hash: str) -> bytes:
        """Derive dataset anchor from master anchor and dataset hash."""
        ...
    
    def derive_model_anchor(self, master_anchor: bytes, model_hash: str) -> bytes:
        """Derive model anchor from master anchor and model hash."""
        ...
    
    def derive_capsule_anchor(self, dataset_anchor: bytes, capsule_id: str) -> bytes:
        """Derive capsule anchor from dataset anchor and capsule ID."""
        ...


@runtime_checkable
class AnchorStore(Protocol):
    """Protocol for anchor storage implementations."""
    
    def append_anchor(self, anchor: Dict[str, Any]) -> None:
        """Append a new anchor to the store (WORM semantics)."""
        ...
    
    def get_latest_anchor(self) -> Optional[Dict[str, Any]]:
        """Get the most recent anchor from the store."""
        ...