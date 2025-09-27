"""
Production cryptographic signers for the Cognitive Insight Audit Framework.

This module provides production-ready digital signature implementations
using industry-standard algorithms like Ed25519.

Created: 2025-09-26
Author: Denzil James Greenwood
Version: 1.0.0
"""

import base64
import hashlib
from typing import Optional, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

from .interfaces import Signer


class Ed25519Signer:
    """Production Ed25519 digital signer implementing the Signer protocol."""
    
    def __init__(self, key_id: str, private_key: Optional[Ed25519PrivateKey] = None):
        """
        Initialize Ed25519 signer.
        
        Args:
            key_id: Identifier for this key
            private_key: Ed25519 private key. If None, generates a new key.
        """
        self.key_id = key_id
        self._private_key = private_key or Ed25519PrivateKey.generate()
        self._public_key = self._private_key.public_key()
    
    def sign(self, data: bytes) -> str:
        """
        Sign data with Ed25519 and return base64-encoded signature.
        
        Args:
            data: Data to sign
            
        Returns:
            Base64-encoded signature string
        """
        signature_bytes = self._private_key.sign(data)
        return base64.b64encode(signature_bytes).decode('ascii')
    
    def verify(self, data: bytes, signature: str) -> bool:
        """
        Verify Ed25519 signature against data.
        
        Args:
            data: Original data that was signed
            signature: Base64-encoded signature string
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            signature_bytes = base64.b64decode(signature.encode('ascii'))
            self._public_key.verify(signature_bytes, data)
            return True
        except (InvalidSignature, ValueError, TypeError):
            return False
    
    def get_public_key_pem(self) -> str:
        """
        Get the public key in PEM format.
        
        Returns:
            PEM-encoded public key string
        """
        pem_bytes = self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem_bytes.decode('ascii')
    
    def get_private_key_pem(self) -> str:
        """
        Get the private key in PEM format (handle with care!).
        
        Returns:
            PEM-encoded private key string
        """
        pem_bytes = self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        return pem_bytes.decode('ascii')
    
    def get_public_key_fingerprint(self) -> str:
        """
        Get a SHA256 fingerprint of the public key.
        
        Returns:
            Hex-encoded SHA256 hash of the public key
        """
        public_key_bytes = self._public_key.public_bytes_raw()
        return hashlib.sha256(public_key_bytes).hexdigest()
    
    @classmethod
    def from_private_key_pem(cls, key_id: str, pem_data: str) -> 'Ed25519Signer':
        """
        Create signer from PEM-encoded private key.
        
        Args:
            key_id: Identifier for this key
            pem_data: PEM-encoded private key string
            
        Returns:
            Ed25519Signer instance
        """
        private_key = serialization.load_pem_private_key(
            pem_data.encode('ascii'),
            password=None
        )
        if not isinstance(private_key, Ed25519PrivateKey):
            raise ValueError("PEM data does not contain an Ed25519 private key")
        
        return cls(key_id, private_key)
    
    @classmethod
    def generate_key_pair(cls, key_id: str) -> Tuple['Ed25519Signer', str]:
        """
        Generate a new Ed25519 key pair.
        
        Args:
            key_id: Identifier for this key
            
        Returns:
            Tuple of (Ed25519Signer instance, public key PEM string)
        """
        signer = cls(key_id)
        public_key_pem = signer.get_public_key_pem()
        return signer, public_key_pem


class Ed25519Verifier:
    """Ed25519 signature verifier for cases where you only have the public key."""
    
    def __init__(self, key_id: str, public_key_pem: str):
        """
        Initialize verifier with public key.
        
        Args:
            key_id: Identifier for this key
            public_key_pem: PEM-encoded public key string
        """
        self.key_id = key_id
        self._public_key = serialization.load_pem_public_key(public_key_pem.encode('ascii'))
        if not isinstance(self._public_key, Ed25519PublicKey):
            raise ValueError("PEM data does not contain an Ed25519 public key")
    
    def verify(self, data: bytes, signature: str) -> bool:
        """
        Verify Ed25519 signature against data.
        
        Args:
            data: Original data that was signed
            signature: Base64-encoded signature string
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            signature_bytes = base64.b64decode(signature.encode('ascii'))
            self._public_key.verify(signature_bytes, data)
            return True
        except (InvalidSignature, ValueError, TypeError):
            return False
    
    def get_public_key_fingerprint(self) -> str:
        """
        Get a SHA256 fingerprint of the public key.
        
        Returns:
            Hex-encoded SHA256 hash of the public key
        """
        public_key_bytes = self._public_key.public_bytes_raw()
        return hashlib.sha256(public_key_bytes).hexdigest()


# For backward compatibility, create an alias
ProductionSigner = Ed25519Signer