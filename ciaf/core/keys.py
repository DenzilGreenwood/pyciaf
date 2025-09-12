"""
Key derivation and management for the Cognitive Insight Audit Framework.

This module provides secure key derivation using PBKDF2HMAC and key management
utilities for the lazy capsule materialization system.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import hashlib
import hmac
import binascii
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .crypto import SALT_LENGTH, hmac_sha256


def derive_key(salt: bytes, password: bytes, length: int = 32) -> bytes:
    """
    Derives a cryptographic key using PBKDF2HMAC.

    Args:
        salt: A unique salt for key derivation.
        password: The base password/secret for derivation.
        length: Desired length of the derived key in bytes (default: 32).

    Returns:
        The derived key as bytes.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        iterations=100000,  # Industry standard iterations
        backend=default_backend(),
    )
    return kdf.derive(password)


def derive_master_key(passphrase: str, salt: bytes, length: int = 32) -> bytes:
    """
    Derives a master key from a passphrase using PBKDF2HMAC.
    This is the root key for lazy capsule materialization.

    Args:
        passphrase: The model name or secret passphrase.
        salt: A unique salt for key derivation.
        length: Desired length of the derived key in bytes (default: 32).

    Returns:
        The derived master key as bytes.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        iterations=100000,  # Industry standard iterations
        backend=default_backend(),
    )
    return kdf.derive(passphrase.encode("utf-8"))


def derive_dataset_key(master_key: bytes, dataset_hash: str) -> bytes:
    """
    Derive a dataset key from master key and dataset hash.

    Args:
        master_key: The master key bytes.
        dataset_hash: Hash of the dataset metadata.

    Returns:
        Dataset key as bytes.
    """
    # Return raw HMAC bytes; hex only at serialization time
    return hmac.new(master_key, dataset_hash.encode("utf-8"), hashlib.sha256).digest()


def derive_capsule_key(dataset_key: bytes, capsule_id: str) -> bytes:
    """
    Derive a capsule key from dataset key and capsule ID.

    Args:
        dataset_key: The dataset key bytes.
        capsule_id: Unique identifier for the capsule.

    Returns:
        Capsule key as bytes.
    """
    return hmac.new(dataset_key, capsule_id.encode("utf-8"), hashlib.sha256).digest()


def to_hex(b: bytes) -> str:
    """Convert bytes to hex string for storage/display."""
    return binascii.hexlify(b).decode()


def from_hex(s: str) -> bytes:
    """Convert hex string back to bytes."""
    return binascii.unhexlify(s)


class AnchorManager:
    """
    Utility class for key derivation and management operations.
    """

    @staticmethod
    def derive_key_pbkdf2(
        password: str, salt: bytes, key_length: int, iterations: int = 100000
    ) -> bytes:
        """Derive a key using PBKDF2."""
        # Note: derive_key function expects (salt, password, length)
        return derive_key(salt, password.encode("utf-8"), key_length)

    @staticmethod
    def derive_master_key(password: str, salt: bytes) -> bytes:
        """Derive master key from password and salt (bytes)."""
        return derive_master_key(password, salt)

    @staticmethod
    def derive_dataset_key(master_key: bytes, dataset_id: str) -> bytes:
        """Derive dataset key from master key and dataset ID."""
        return derive_dataset_key(master_key, dataset_id)

    @staticmethod
    def derive_capsule_key(dataset_key: bytes, capsule_id: str) -> bytes:
        """Derive capsule key from dataset key and capsule ID."""
        return derive_capsule_key(dataset_key, capsule_id)
