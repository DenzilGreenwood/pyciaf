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
)
from .base_anchor import (
    BaseAnchorManager,
    derive_anchor,
    derive_capsule_anchor,
    derive_dataset_anchor,
    derive_master_anchor,
    derive_model_anchor,
    from_hex,
    to_hex,
    # Backwards compatibility aliases
    derive_key,
    derive_master_key,
    derive_dataset_key,
    derive_capsule_key,
    AnchorManager,
)
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
    # Anchor derivation (new terminology)
    "derive_anchor",
    "derive_master_anchor",
    "derive_dataset_anchor",
    "derive_capsule_anchor",
    "derive_model_anchor",
    "BaseAnchorManager",
    "AnchorManager",
    "to_hex",
    "from_hex",
    # Backwards compatibility (legacy key terminology)
    "derive_key",
    "derive_master_key",
    "derive_dataset_key",
    "derive_capsule_key",
    "KeyManager",
    # Merkle tree
    "MerkleTree",
]
