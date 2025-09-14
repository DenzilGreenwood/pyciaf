"""
Base Anchor Derivation and Management for the Cognitive Insight Audit Framework.

This module provides secure anchor derivation using PBKDF2HMAC and anchor management
utilities for the lazy capsule materialization system. An anchor represents a
cryptographic root for a specific scope (dataset, model, etc.) that can derive
child anchors in a hierarchical manner.

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


def derive_anchor(salt: bytes, password: bytes, length: int = 32) -> bytes:
    """
    Derives a cryptographic anchor using PBKDF2HMAC.

    Args:
        salt: A unique salt for anchor derivation.
        password: The base password/secret for derivation.
        length: Desired length of the derived anchor in bytes (default: 32).

    Returns:
        The derived anchor as bytes.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        iterations=100000,  # Industry standard iterations
        backend=default_backend(),
    )
    return kdf.derive(password)


def derive_master_anchor(passphrase: str, salt: bytes, length: int = 32) -> bytes:
    """
    Derives a master anchor from a passphrase using PBKDF2HMAC.
    This is the root anchor for lazy capsule materialization.

    Args:
        passphrase: The model name or secret passphrase.
        salt: A unique salt for anchor derivation.
        length: Desired length of the derived anchor in bytes (default: 32).

    Returns:
        The derived master anchor as bytes.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        iterations=100000,  # Industry standard iterations
        backend=default_backend(),
    )
    return kdf.derive(passphrase.encode("utf-8"))


def derive_dataset_anchor(master_anchor: bytes, dataset_hash: str) -> bytes:
    """
    Derive a dataset anchor from master anchor and dataset hash.

    Args:
        master_anchor: The master anchor bytes.
        dataset_hash: Hash of the dataset metadata.

    Returns:
        Dataset anchor as bytes.
    """
    # Return raw HMAC bytes; hex only at serialization time
    return hmac.new(master_anchor, dataset_hash.encode("utf-8"), hashlib.sha256).digest()


def derive_capsule_anchor(dataset_anchor: bytes, capsule_id: str) -> bytes:
    """
    Derive a capsule anchor from dataset anchor and capsule ID.

    Args:
        dataset_anchor: The dataset anchor bytes.
        capsule_id: Unique identifier for the capsule.

    Returns:
        Capsule anchor as bytes.
    """
    return hmac.new(dataset_anchor, capsule_id.encode("utf-8"), hashlib.sha256).digest()


def derive_model_anchor(master_anchor: bytes, model_hash: str) -> bytes:
    """
    Derive a model anchor from master anchor and model hash.

    Args:
        master_anchor: The master anchor bytes.
        model_hash: Hash of the model metadata.

    Returns:
        Model anchor as bytes.
    """
    return hmac.new(master_anchor, model_hash.encode("utf-8"), hashlib.sha256).digest()


def to_hex(b: bytes) -> str:
    """Convert bytes to hex string for storage/display."""
    return binascii.hexlify(b).decode()


def from_hex(s: str) -> bytes:
    """Convert hex string back to bytes."""
    return binascii.unhexlify(s)


class BaseAnchorManager:
    """
    Utility class for anchor derivation and management operations.
    
    This class provides a centralized interface for deriving various types
    of anchors in the CIAF hierarchical anchor system.
    """

    @staticmethod
    def derive_anchor_pbkdf2(
        password: str, salt: bytes, anchor_length: int, iterations: int = 100000
    ) -> bytes:
        """Derive an anchor using PBKDF2."""
        # Note: derive_anchor function expects (salt, password, length)
        return derive_anchor(salt, password.encode("utf-8"), anchor_length)

    @staticmethod
    def derive_master_anchor(password: str, salt: bytes) -> bytes:
        """Derive master anchor from password and salt (bytes)."""
        return derive_master_anchor(password, salt)

    @staticmethod
    def derive_dataset_anchor(master_anchor: bytes, dataset_id: str) -> bytes:
        """Derive dataset anchor from master anchor and dataset ID."""
        return derive_dataset_anchor(master_anchor, dataset_id)

    @staticmethod
    def derive_capsule_anchor(dataset_anchor: bytes, capsule_id: str) -> bytes:
        """Derive capsule anchor from dataset anchor and capsule ID."""
        return derive_capsule_anchor(dataset_anchor, capsule_id)

    @staticmethod
    def derive_model_anchor(master_anchor: bytes, model_id: str) -> bytes:
        """Derive model anchor from master anchor and model ID."""
        return derive_model_anchor(master_anchor, model_id)


# Backwards compatibility aliases for existing code
derive_key = derive_anchor
derive_master_key = derive_master_anchor
derive_dataset_key = derive_dataset_anchor
derive_capsule_key = derive_capsule_anchor
AnchorManager = BaseAnchorManager
