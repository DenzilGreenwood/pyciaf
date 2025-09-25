"""
Core cryptographic and foundational components for CIAF.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

from .crypto import (
    SALT_LENGTH,
    CryptoUtils,
    decrypt_aes_gcm,
    encrypt_aes_gcm,
    hmac_sha256,
    secure_random_bytes,
    sha256_hash,
    derive_anchor_from_master,
    derive_master_anchor,
    derive_dataset_anchor,
    derive_model_anchor,
    derive_capsule_anchor,
    to_hex,
    from_hex,
    # Legacy aliases
    derive_master_key,
    derive_dataset_key,
    derive_capsule_key,
)
try:
    from .keys import AnchorManager
    BaseAnchorManager = AnchorManager  # Create alias
except ImportError:
    # Handle circular import gracefully
    BaseAnchorManager = None
    AnchorManager = None
from .merkle import MerkleTree

# Create legacy alias for backward compatibility
KeyManager = AnchorManager

__all__ = [
    # Crypto utilities
    "encrypt_aes_gcm",
    "decrypt_aes_gcm",
    "sha256_hash",
    "hmac_sha256",
    "secure_random_bytes",
    "SALT_LENGTH",
    "CryptoUtils",
    # Anchor derivation functions
    "derive_anchor_from_master",
    "derive_master_anchor",
    "derive_dataset_anchor", 
    "derive_model_anchor",
    "derive_capsule_anchor",
    "to_hex",
    "from_hex",
    # Legacy aliases
    "derive_master_key",
    "derive_dataset_key", 
    "derive_capsule_key",
    # Anchor management (conditional export)
    "BaseAnchorManager",
    "AnchorManager",
    # Backwards compatibility (legacy key terminology)
    "KeyManager",
    # Merkle tree
    "MerkleTree",
]
