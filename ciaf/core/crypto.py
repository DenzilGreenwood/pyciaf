"""
Core cryptographic utilities for the Cognitive Insight Audit Framework.

Created: 2025-09-09
Last Modified: 2025-09-26
Author: Denzil James Greenwood
Version: 1.1.0
"""

import hashlib
import hmac
import os
from typing import Optional

# Optional blake3
try:
    import blake3
    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from .constants import SALT_LENGTH, PBKDF2_ITERATIONS, KDF_DKLEN


def make_aad(dataset_anchor_hex: str, capsule_id: str, policy_id: str) -> bytes:
    """
    Create Additional Authenticated Data (AAD) for AEAD encryption.
    
    Binds encryption context to dataset anchor, capsule, and policy.
    
    Args:
        dataset_anchor_hex: Hex-encoded dataset anchor
        capsule_id: Capsule identifier
        policy_id: Policy identifier
        
    Returns:
        AAD bytes for use with AES-GCM encryption
    """
    return f"{dataset_anchor_hex}|{capsule_id}|{policy_id}".encode("utf-8")


def encrypt_aes_gcm(key: bytes, plaintext: bytes, aad: Optional[bytes] = None) -> tuple[bytes, bytes, bytes]:
    nonce = os.urandom(12)
    cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
    enc = cipher.encryptor()
    if aad:
        enc.authenticate_additional_data(aad)
    ciphertext = enc.update(plaintext) + enc.finalize()
    return ciphertext, nonce, enc.tag


def decrypt_aes_gcm(key: bytes, ciphertext: bytes, nonce: bytes, tag: bytes, aad: Optional[bytes] = None) -> bytes:
    cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend())
    dec = cipher.decryptor()
    if aad:
        dec.authenticate_additional_data(aad)
    return dec.update(ciphertext) + dec.finalize()


def sha256_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blake3_hash(data: bytes) -> str:
    if not BLAKE3_AVAILABLE:
        raise ImportError("blake3 library is required (pip install blake3)")
    from blake3 import blake3 as Blake3Hasher
    h = Blake3Hasher()
    h.update(data)
    return h.hexdigest()


def sha3_256_hash(data: bytes) -> str:
    return hashlib.sha3_256(data).hexdigest()


def compute_hash(data: bytes, algorithm: str = "sha256") -> str:
    alg = algorithm.lower()
    if alg == "sha256":
        return sha256_hash(data)
    if alg == "sha3-256":
        return sha3_256_hash(data)
    if alg == "blake3":
        return blake3_hash(data)
    supported = ["sha256", "sha3-256"]
    if BLAKE3_AVAILABLE:
        supported.append("blake3")
    raise ValueError(f"Unsupported hash algorithm: {algorithm}. Supported: {supported}")


def hmac_sha256(key: bytes, data: bytes) -> str:
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def secure_random_bytes(length: int) -> bytes:
    return os.urandom(length)


def derive_anchor_from_master(master_password: str, dataset_id: str) -> bytes:
    key_bytes = master_password.encode("utf-8")
    data_bytes = dataset_id.encode("utf-8")
    return bytes.fromhex(hmac_sha256(key_bytes, data_bytes))


def derive_master_anchor(password: str, salt: bytes) -> bytes:
    password_bytes = password.encode("utf-8")
    return hashlib.pbkdf2_hmac("sha256", password_bytes, salt, PBKDF2_ITERATIONS, dklen=KDF_DKLEN)


def _derive_hmac_bytes(key: bytes, data: bytes) -> bytes:
    return bytes.fromhex(hmac_sha256(key, data))


def derive_dataset_anchor(master_anchor: bytes, dataset_hash: str) -> bytes:
    return _derive_hmac_bytes(master_anchor, dataset_hash.encode("utf-8"))


def derive_model_anchor(master_anchor: bytes, model_hash: str) -> bytes:
    return _derive_hmac_bytes(master_anchor, model_hash.encode("utf-8"))


def derive_capsule_anchor(dataset_anchor: bytes, capsule_id: str) -> bytes:
    return _derive_hmac_bytes(dataset_anchor, capsule_id.encode("utf-8"))


def to_hex(data: bytes) -> str:
    return data.hex()


def from_hex(hex_str: str) -> bytes:
    return bytes.fromhex(hex_str)


def generate_master_password(length: int = 32) -> str:
    import secrets, string
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


class CryptoUtils:
    @staticmethod
    def sha256_hash(data: bytes) -> str: return sha256_hash(data)
    @staticmethod
    def blake3_hash(data: bytes) -> str: return blake3_hash(data)
    @staticmethod
    def sha3_256_hash(data: bytes) -> str: return sha3_256_hash(data)
    @staticmethod
    def compute_hash(data: bytes, algorithm: str = "sha256") -> str: return compute_hash(data, algorithm)
    @staticmethod
    def hmac_sha256(key: bytes, data: bytes) -> str: return hmac_sha256(key, data)
    @staticmethod
    def encrypt_aes_gcm(key: bytes, plaintext: bytes, aad: Optional[bytes] = None) -> tuple[bytes, bytes, bytes]:
        return encrypt_aes_gcm(key, plaintext, aad)
    @staticmethod
    def decrypt_aes_gcm(key: bytes, ciphertext: bytes, nonce: bytes, tag: bytes, aad: Optional[bytes] = None) -> bytes:
        return decrypt_aes_gcm(key, ciphertext, nonce, tag, aad)
    @staticmethod
    def secure_random_bytes(length: int) -> bytes: return secure_random_bytes(length)