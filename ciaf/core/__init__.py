"""
Core cryptographic and foundational components for CIAF.

Created: 2025-09-09
Last Modified: 2025-09-26
Author: Denzil James Greenwood
Version: 1.2.1
"""

from .constants import (
    ANCHOR_SCHEMA_VERSION,
    MERKLE_POLICY_VERSION,
    PBKDF2_ITERATIONS,
    KDF_DKLEN,
    SALT_LENGTH,  # single source of truth
    DEFAULT_HASH_FUNCTION,
    DEFAULT_SIGNATURE_ALGORITHM,
    DEFAULT_PUBKEY_ID,
)

from .enums import RecordType, HashAlgorithm, SignatureAlgorithm
from .interfaces import Signer, RNG, Merkle, AnchorDeriver, AnchorStore

from .crypto import (
    CryptoUtils,
    decrypt_aes_gcm,
    encrypt_aes_gcm,
    hmac_sha256,
    secure_random_bytes,
    sha256_hash,
    blake3_hash,
    sha3_256_hash,
    compute_hash,
    derive_anchor_from_master,
    derive_master_anchor,
    derive_dataset_anchor,
    derive_model_anchor,
    derive_capsule_anchor,
    to_hex,
    from_hex,
    make_aad,
)

from .signers import Ed25519Signer, Ed25519Verifier, ProductionSigner
from .canonicalization import create_production_signer
from .merkle import MerkleTree

# Legacy anchor managers removed - using LCM system instead
BaseAnchorManager = None
AnchorManager = None

__all__ = [
    # Constants
    "ANCHOR_SCHEMA_VERSION",
    "MERKLE_POLICY_VERSION",
    "PBKDF2_ITERATIONS",
    "KDF_DKLEN",
    "SALT_LENGTH",
    "DEFAULT_HASH_FUNCTION",
    "DEFAULT_SIGNATURE_ALGORITHM",
    "DEFAULT_PUBKEY_ID",
    # Enums
    "RecordType",
    "HashAlgorithm",
    "SignatureAlgorithm",
    # Interfaces
    "Signer",
    "RNG",
    "Merkle",
    "AnchorDeriver",
    "AnchorStore",
    # Crypto
    "encrypt_aes_gcm",
    "decrypt_aes_gcm",
    "sha256_hash",
    "blake3_hash",
    "sha3_256_hash",
    "compute_hash",
    "hmac_sha256",
    "secure_random_bytes",
    "CryptoUtils",
    # Signers
    "Ed25519Signer",
    "Ed25519Verifier",
    "ProductionSigner",
    "create_production_signer",
    # Anchors
    "derive_anchor_from_master",
    "derive_master_anchor",
    "derive_dataset_anchor",
    "derive_model_anchor",
    "derive_capsule_anchor",
    "to_hex",
    "from_hex",
    "make_aad",
    # Legacy managers (removed - using LCM system)
    # "BaseAnchorManager", 
    # "AnchorManager",
    # Merkle
    "MerkleTree",
]
