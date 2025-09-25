"""
Core cryptographic utilities for the Cognitive Insight Audit Framework.

This module provides essential cryptographic functions including AES-GCM encryption/decryption,
SHA256 hashing, and secure random number generation.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import hashlib
import hmac
import os

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# --- Configuration ---
SALT_LENGTH = 16


def encrypt_aes_gcm(key: bytes, plaintext: bytes, aad: bytes | None = None) -> tuple[bytes, bytes, bytes]:
    """
    Encrypts plaintext using AES-256 GCM.

    Args:
        key: 32-byte AES key.
        plaintext: Data to encrypt.
        aad: Optional additional authenticated data.

    Returns:
        A tuple containing (ciphertext, nonce, tag).
    """
    nonce = os.urandom(12)  # 96-bit nonce recommended for GCM
    cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
    encryptor = cipher.encryptor()
    if aad:
        encryptor.authenticate_additional_data(aad)
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return ciphertext, nonce, encryptor.tag


def decrypt_aes_gcm(key: bytes, ciphertext: bytes, nonce: bytes, tag: bytes, aad: bytes | None = None) -> bytes:
    """
    Decrypts ciphertext using AES-256 GCM.

    Args:
        key: 32-byte AES key.
        ciphertext: Encrypted data.
        nonce: Nonce used during encryption.
        tag: Authentication tag.
        aad: Optional additional authenticated data.

    Returns:
        Decrypted plaintext as bytes.

    Raises:
        InvalidTag: If the authentication tag is invalid (data tampered).
    """
    cipher = Cipher(
        algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend()
    )
    decryptor = cipher.decryptor()
    if aad:
        decryptor.authenticate_additional_data(aad)
    return decryptor.update(ciphertext) + decryptor.finalize()


def sha256_hash(data: bytes) -> str:
    """
    Generates a SHA256 hash of the given bytes data.

    Args:
        data: The bytes data to hash.

    Returns:
        The hexadecimal representation of the SHA256 hash.
    """
    return hashlib.sha256(data).hexdigest()


def hmac_sha256(key: bytes, data: bytes) -> str:
    """
    Generates an HMAC-SHA256 hash for dataset-level anchor derivation.
    Used in lazy capsule materialization for deriving dataset anchors and capsule anchors.

    Args:
        key: The HMAC key (e.g., master anchor or dataset anchor).
        data: The data to authenticate (e.g., dataset hash or capsule ID).

    Returns:
        The hexadecimal representation of the HMAC-SHA256.
    """
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def secure_random_bytes(length: int) -> bytes:
    """
    Generate cryptographically secure random bytes.

    Args:
        length: Number of bytes to generate.

    Returns:
        Random bytes.
    """
    return os.urandom(length)


def derive_anchor_from_master(master_password: str, dataset_id: str) -> str:
    """
    Derive a dataset anchor from master password and dataset ID.
    
    Args:
        master_password: Master password for derivation
        dataset_id: Dataset identifier
        
    Returns:
        Hex-encoded anchor string
    """
    # Use the master password as HMAC key, dataset_id as data
    key_bytes = master_password.encode('utf-8')
    data_bytes = dataset_id.encode('utf-8')
    return hmac_sha256(key_bytes, data_bytes)


def derive_master_anchor(password: str, salt: bytes) -> bytes:
    """
    Derive a master anchor from password and salt using PBKDF2.
    
    Args:
        password: Master password string
        salt: Salt bytes for derivation
        
    Returns:
        Derived master anchor as bytes
    """
    import hashlib
    password_bytes = password.encode('utf-8')
    return hashlib.pbkdf2_hmac('sha256', password_bytes, salt, 100000, dklen=32)


def derive_dataset_anchor(master_anchor: bytes, dataset_hash: str) -> bytes:
    """
    Derive a dataset anchor from master anchor and dataset hash.
    
    Args:
        master_anchor: Master anchor bytes
        dataset_hash: Dataset hash string
        
    Returns:
        Derived dataset anchor as bytes
    """
    data_bytes = dataset_hash.encode('utf-8')
    derived = hmac_sha256_bytes(master_anchor, data_bytes)
    return bytes.fromhex(derived)


def derive_model_anchor(master_anchor: bytes, model_hash: str) -> bytes:
    """
    Derive a model anchor from master anchor and model hash.
    
    Args:
        master_anchor: Master anchor bytes
        model_hash: Model hash string
        
    Returns:
        Derived model anchor as bytes
    """
    data_bytes = model_hash.encode('utf-8')
    derived = hmac_sha256_bytes(master_anchor, data_bytes)
    return bytes.fromhex(derived)


def derive_capsule_anchor(dataset_anchor: bytes, capsule_id: str) -> bytes:
    """
    Derive a capsule anchor from dataset anchor and capsule ID.
    
    Args:
        dataset_anchor: Dataset anchor bytes
        capsule_id: Capsule identifier string
        
    Returns:
        Derived capsule anchor as bytes
    """
    data_bytes = capsule_id.encode('utf-8')
    derived = hmac_sha256_bytes(dataset_anchor, data_bytes)
    return bytes.fromhex(derived)


def hmac_sha256_bytes(key: bytes, data: bytes) -> str:
    """
    Compute HMAC-SHA256 and return as hex string.
    
    Args:
        key: HMAC key as bytes
        data: Data to authenticate as bytes
        
    Returns:
        Hex-encoded HMAC digest
    """
    import hmac
    import hashlib
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def to_hex(data: bytes) -> str:
    """Convert bytes to hex string."""
    return data.hex()


def from_hex(hex_str: str) -> bytes:
    """Convert hex string to bytes."""
    return bytes.fromhex(hex_str)


# Legacy aliases for backwards compatibility
derive_master_key = derive_master_anchor
derive_dataset_key = derive_dataset_anchor
derive_capsule_key = derive_capsule_anchor


def generate_master_password() -> str:
    """
    Generate a secure master password.
    
    Returns:
        A secure random password string
    """
    import secrets
    import string
    
    # Generate a 32-character password with letters and digits
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(32))


class CryptoUtils:
    """
    Utility class providing cryptographic operations for CIAF.
    """

    @staticmethod
    def sha256_hash(data: bytes) -> str:
        """Compute SHA256 hash of data."""
        return sha256_hash(data)

    @staticmethod
    def hmac_sha256(key: bytes, data: bytes) -> str:
        """Compute HMAC-SHA256 of data."""
        return hmac_sha256(key, data)

    @staticmethod
    def encrypt_aes_gcm(key: bytes, plaintext: bytes, aad: bytes | None = None) -> tuple[bytes, bytes, bytes]:
        """Encrypt data using AES-GCM."""
        return encrypt_aes_gcm(key, plaintext, aad)

    @staticmethod
    def decrypt_aes_gcm(
        key: bytes, ciphertext: bytes, nonce: bytes, tag: bytes, aad: bytes | None = None
    ) -> bytes:
        """Decrypt data using AES-GCM."""
        return decrypt_aes_gcm(key, ciphertext, nonce, tag, aad)

    @staticmethod
    def secure_random_bytes(length: int) -> bytes:
        """Generate secure random bytes."""
        return secure_random_bytes(length)
