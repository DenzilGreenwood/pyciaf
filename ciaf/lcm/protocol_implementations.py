"""
Concrete implementations of core protocols for the LCM system.

This module provides concrete implementations of the Protocol interfaces
that wrap the existing core functionality, enabling dependency injection
and better architecture in the LCM system.

Created: 2025-09-26
Author: Denzil James Greenwood
Version: 1.0.0
"""

from typing import List, Tuple, Dict, Any, Optional
from ..core import (
    secure_random_bytes,
    MerkleTree,
    derive_master_anchor as _derive_master_anchor,
    derive_dataset_anchor as _derive_dataset_anchor,
    derive_model_anchor as _derive_model_anchor,
    derive_capsule_anchor as _derive_capsule_anchor,
    Ed25519Signer
)
from ..core.interfaces import Signer, RNG, Merkle, AnchorDeriver, AnchorStore


class DefaultRNG(RNG):
    """Default RNG implementation using core secure_random_bytes."""
    
    def random_bytes(self, n: int) -> bytes:
        """Generate n cryptographically secure random bytes."""
        return secure_random_bytes(n)


class DefaultMerkle(Merkle):
    """Default Merkle implementation wrapping core MerkleTree."""
    
    def __init__(self, leaves: List[str] = None):
        """Initialize with optional leaves."""
        self._leaves = leaves or []
        self._tree = None
        if self._leaves:
            self._tree = MerkleTree(self._leaves)
    
    def add_leaf(self, leaf_hash: str) -> None:
        """Add a leaf to the tree."""
        self._leaves.append(leaf_hash)
        self._tree = MerkleTree(self._leaves)
    
    def get_root(self) -> str:
        """Get the current Merkle root hash."""
        if not self._leaves:
            return "empty_root"  # Return a default for empty trees
        if self._tree is None:
            self._tree = MerkleTree(self._leaves)
        return self._tree.get_root()
    
    def get_proof(self, leaf_hash: str) -> List[Tuple[str, str]]:
        """Get inclusion proof for a leaf as list of (hash, position) tuples."""
        if not self._leaves or self._tree is None:
            return []
        return self._tree.get_proof(leaf_hash)
    
    @staticmethod
    def verify_proof(leaf_hash: str, root_hash: str, proof: List[Tuple[str, str]]) -> bool:
        """Verify a Merkle inclusion proof."""
        return MerkleTree.verify_proof(leaf_hash, root_hash, proof)


class DefaultAnchorDeriver(AnchorDeriver):
    """Default anchor derivation implementation using core functions."""
    
    def derive_master_anchor(self, password: str, salt: bytes) -> bytes:
        """Derive master anchor from password and salt."""
        return _derive_master_anchor(password, salt)
    
    def derive_dataset_anchor(self, master_anchor: bytes, dataset_hash: str) -> bytes:
        """Derive dataset anchor from master anchor and dataset hash."""
        return _derive_dataset_anchor(master_anchor, dataset_hash)
    
    def derive_model_anchor(self, master_anchor: bytes, model_hash: str) -> bytes:
        """Derive model anchor from master anchor and model hash."""
        return _derive_model_anchor(master_anchor, model_hash)
    
    def derive_capsule_anchor(self, dataset_anchor: bytes, capsule_id: str) -> bytes:
        """Derive capsule anchor from dataset anchor and capsule ID."""
        return _derive_capsule_anchor(dataset_anchor, capsule_id)


class InMemoryAnchorStore(AnchorStore):
    """In-memory anchor store implementation."""
    
    def __init__(self):
        """Initialize empty anchor store."""
        self._anchors: List[Dict[str, Any]] = []
    
    def append_anchor(self, anchor: Dict[str, Any]) -> None:
        """Append a new anchor to the store (WORM semantics)."""
        # Create a copy to prevent external modification
        anchor_copy = dict(anchor)
        if 'timestamp' not in anchor_copy:
            from datetime import datetime
            anchor_copy['timestamp'] = datetime.now().isoformat()
        self._anchors.append(anchor_copy)
    
    def get_latest_anchor(self) -> Optional[Dict[str, Any]]:
        """Get the most recent anchor from the store."""
        return self._anchors[-1] if self._anchors else None
    
    def get_all_anchors(self) -> List[Dict[str, Any]]:
        """Get all anchors (for debugging/testing)."""
        return self._anchors.copy()
    
    def count(self) -> int:
        """Get number of stored anchors."""
        return len(self._anchors)


class DefaultSigner(Signer):
    """Default signer implementation wrapping Ed25519Signer."""
    
    def __init__(self, key_id: str = "lcm_default_key"):
        """Initialize with key ID."""
        self.key_id = key_id
        self._signer = Ed25519Signer(key_id)
    
    def sign(self, data: bytes) -> str:
        """Sign data and return signature string."""
        return self._signer.sign(data)
    
    def verify(self, data: bytes, signature: str) -> bool:
        """Verify signature against data."""
        return self._signer.verify(data, signature)


def create_default_protocols() -> Dict[str, Any]:
    """
    Create default protocol implementations for LCM system.
    
    Returns:
        Dictionary containing default protocol implementations
    """
    return {
        'rng': DefaultRNG(),
        'merkle_factory': lambda leaves=None: DefaultMerkle(leaves),
        'anchor_deriver': DefaultAnchorDeriver(),
        'anchor_store': InMemoryAnchorStore(),
        'signer': DefaultSigner()
    }