"""
Enhanced key management surface for CIAF production deployments.

Provides secure key generation, storage, rotation, and lifecycle management
for cryptographic operations with multiple backend support.

Created: 2025-09-26
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
import os

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .crypto import secure_random_bytes
from .determinism import canonical_timestamp
from .enums import SignatureAlgorithm
from .signers import Ed25519Signer


class KeyStatus(str, Enum):
    """Key lifecycle status."""
    ACTIVE = "active"
    RETIRED = "retired"
    REVOKED = "revoked"
    PENDING = "pending"


class KeyType(str, Enum):
    """Types of cryptographic keys."""
    SIGNING = "signing"
    ENCRYPTION = "encryption"
    MASTER = "master"


@dataclass
class KeyMetadata:
    """Metadata for a cryptographic key."""
    key_id: str
    key_type: KeyType
    algorithm: str
    status: KeyStatus
    created_at: str
    expires_at: Optional[str] = None
    retired_at: Optional[str] = None
    purpose: str = ""
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
    
    def is_active(self) -> bool:
        """Check if key is currently active."""
        if self.status != KeyStatus.ACTIVE:
            return False
        
        if self.expires_at:
            expires = datetime.fromisoformat(self.expires_at.replace('Z', '+00:00'))
            if datetime.now(timezone.utc) > expires:
                return False
        
        return True
    
    def days_until_expiry(self) -> Optional[int]:
        """Get days until key expires."""
        if not self.expires_at:
            return None
        
        expires = datetime.fromisoformat(self.expires_at.replace('Z', '+00:00'))
        delta = expires - datetime.now(timezone.utc)
        return max(0, delta.days)


@dataclass
class KeyBundle:
    """Bundle containing key material and metadata."""
    metadata: KeyMetadata
    private_key_pem: Optional[str] = None
    public_key_pem: Optional[str] = None
    key_material: Optional[bytes] = None
    
    def get_signer(self) -> Optional[Ed25519Signer]:
        """Get signer instance if this is a signing key."""
        if (self.metadata.key_type == KeyType.SIGNING and 
            self.metadata.algorithm == SignatureAlgorithm.ED25519 and
            self.private_key_pem):
            return Ed25519Signer.from_private_key_pem(
                self.metadata.key_id, 
                self.private_key_pem
            )
        return None


class KeyStore(ABC):
    """Abstract base class for key storage backends."""
    
    @abstractmethod
    def store_key(self, key_bundle: KeyBundle) -> bool:
        """Store a key bundle."""
        ...
    
    @abstractmethod
    def retrieve_key(self, key_id: str) -> Optional[KeyBundle]:
        """Retrieve a key bundle by ID."""
        ...
    
    @abstractmethod
    def list_keys(self, key_type: Optional[KeyType] = None, 
                  status: Optional[KeyStatus] = None) -> List[KeyMetadata]:
        """List keys with optional filtering."""
        ...
    
    @abstractmethod
    def update_key_status(self, key_id: str, status: KeyStatus) -> bool:
        """Update key status."""
        ...
    
    @abstractmethod
    def delete_key(self, key_id: str) -> bool:
        """Delete a key (if supported)."""
        ...


class FileSystemKeyStore(KeyStore):
    """File system-based key store for development and testing."""
    
    def __init__(self, key_dir: str = "keys"):
        """
        Initialize file system key store.
        
        Args:
            key_dir: Directory to store keys
        """
        self.key_dir = Path(key_dir)
        self.key_dir.mkdir(parents=True, exist_ok=True)
        
        # Secure directory permissions (Unix-like systems)
        try:
            os.chmod(self.key_dir, 0o700)
        except (OSError, AttributeError):
            # Windows or permission error - continue anyway
            pass
    
    def _key_file_path(self, key_id: str) -> Path:
        """Get file path for a key."""
        return self.key_dir / f"{key_id}.json"
    
    def store_key(self, key_bundle: KeyBundle) -> bool:
        """Store key bundle to file system."""
        try:
            key_file = self._key_file_path(key_bundle.metadata.key_id)
            
            # Serialize key bundle
            bundle_dict = {
                'metadata': asdict(key_bundle.metadata),
                'private_key_pem': key_bundle.private_key_pem,
                'public_key_pem': key_bundle.public_key_pem,
                'key_material': key_bundle.key_material.hex() if key_bundle.key_material else None
            }
            
            # Write to file with secure permissions
            with open(key_file, 'w') as f:
                json.dump(bundle_dict, f, indent=2)
            
            # Secure file permissions
            try:
                os.chmod(key_file, 0o600)
            except (OSError, AttributeError):
                pass
            
            return True
        
        except Exception:
            return False
    
    def retrieve_key(self, key_id: str) -> Optional[KeyBundle]:
        """Retrieve key bundle from file system."""
        try:
            key_file = self._key_file_path(key_id)
            if not key_file.exists():
                return None
            
            with open(key_file, 'r') as f:
                bundle_dict = json.load(f)
            
            # Reconstruct key bundle
            metadata = KeyMetadata(**bundle_dict['metadata'])
            key_material = None
            if bundle_dict['key_material']:
                key_material = bytes.fromhex(bundle_dict['key_material'])
            
            return KeyBundle(
                metadata=metadata,
                private_key_pem=bundle_dict['private_key_pem'],
                public_key_pem=bundle_dict['public_key_pem'],
                key_material=key_material
            )
        
        except Exception:
            return None
    
    def list_keys(self, key_type: Optional[KeyType] = None, 
                  status: Optional[KeyStatus] = None) -> List[KeyMetadata]:
        """List keys in the store."""
        keys = []
        
        for key_file in self.key_dir.glob("*.json"):
            try:
                with open(key_file, 'r') as f:
                    bundle_dict = json.load(f)
                
                metadata = KeyMetadata(**bundle_dict['metadata'])
                
                # Apply filters
                if key_type and metadata.key_type != key_type:
                    continue
                if status and metadata.status != status:
                    continue
                
                keys.append(metadata)
            
            except Exception:
                continue
        
        return keys
    
    def update_key_status(self, key_id: str, status: KeyStatus) -> bool:
        """Update key status."""
        try:
            key_bundle = self.retrieve_key(key_id)
            if not key_bundle:
                return False
            
            key_bundle.metadata.status = status
            if status in [KeyStatus.RETIRED, KeyStatus.REVOKED]:
                key_bundle.metadata.retired_at = canonical_timestamp()
            
            return self.store_key(key_bundle)
        
        except Exception:
            return False
    
    def delete_key(self, key_id: str) -> bool:
        """Delete key file."""
        try:
            key_file = self._key_file_path(key_id)
            if key_file.exists():
                key_file.unlink()
                return True
            return False
        
        except Exception:
            return False


class KeyManager:
    """
    Central key management system for CIAF.
    
    Provides high-level key lifecycle management including generation,
    storage, rotation, and access control.
    """
    
    def __init__(self, key_store: KeyStore, default_key_validity_days: int = 365):
        """
        Initialize key manager.
        
        Args:
            key_store: Backend key storage
            default_key_validity_days: Default key validity period
        """
        self.key_store = key_store
        self.default_key_validity_days = default_key_validity_days
    
    def generate_signing_key(self, key_id: str, purpose: str = "", 
                           validity_days: Optional[int] = None,
                           tags: Optional[Dict[str, str]] = None) -> KeyBundle:
        """
        Generate new Ed25519 signing key.
        
        Args:
            key_id: Unique identifier for the key
            purpose: Description of key purpose
            validity_days: Key validity period (defaults to configured value)
            tags: Optional key tags
            
        Returns:
            Generated key bundle
        """
        # Generate Ed25519 key pair
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        
        # Get PEM representations
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('ascii')
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('ascii')
        
        # Create metadata
        now = datetime.now(timezone.utc)
        validity_days = validity_days or self.default_key_validity_days
        expires_at = (now + timedelta(days=validity_days)).isoformat().replace('+00:00', 'Z')
        
        metadata = KeyMetadata(
            key_id=key_id,
            key_type=KeyType.SIGNING,
            algorithm=SignatureAlgorithm.ED25519,
            status=KeyStatus.ACTIVE,
            created_at=canonical_timestamp(now),
            expires_at=expires_at,
            purpose=purpose,
            tags=tags or {}
        )
        
        # Create key bundle
        key_bundle = KeyBundle(
            metadata=metadata,
            private_key_pem=private_pem,
            public_key_pem=public_pem
        )
        
        # Store the key
        if not self.key_store.store_key(key_bundle):
            raise RuntimeError(f"Failed to store key {key_id}")
        
        return key_bundle
    
    def generate_master_key(self, key_id: str, purpose: str = "",
                          key_length: int = 32,
                          tags: Optional[Dict[str, str]] = None) -> KeyBundle:
        """
        Generate symmetric master key.
        
        Args:
            key_id: Unique identifier for the key
            purpose: Description of key purpose
            key_length: Key length in bytes
            tags: Optional key tags
            
        Returns:
            Generated key bundle
        """
        # Generate random key material
        key_material = secure_random_bytes(key_length)
        
        # Create metadata
        metadata = KeyMetadata(
            key_id=key_id,
            key_type=KeyType.MASTER,
            algorithm="AES-256",
            status=KeyStatus.ACTIVE,
            created_at=canonical_timestamp(),
            purpose=purpose,
            tags=tags or {}
        )
        
        # Create key bundle
        key_bundle = KeyBundle(
            metadata=metadata,
            key_material=key_material
        )
        
        # Store the key
        if not self.key_store.store_key(key_bundle):
            raise RuntimeError(f"Failed to store key {key_id}")
        
        return key_bundle
    
    def get_active_signing_key(self, purpose: str = "") -> Optional[KeyBundle]:
        """
        Get an active signing key, optionally filtered by purpose.
        
        Args:
            purpose: Optional purpose filter
            
        Returns:
            Active signing key bundle or None
        """
        signing_keys = self.key_store.list_keys(
            key_type=KeyType.SIGNING,
            status=KeyStatus.ACTIVE
        )
        
        # Filter by purpose if specified
        if purpose:
            signing_keys = [k for k in signing_keys if k.purpose == purpose]
        
        # Find keys that are actually active (not expired)
        for key_meta in signing_keys:
            if key_meta.is_active():
                return self.key_store.retrieve_key(key_meta.key_id)
        
        return None
    
    def rotate_key(self, old_key_id: str, new_key_id: str) -> Optional[KeyBundle]:
        """
        Rotate a key by retiring the old one and generating a new one.
        
        Args:
            old_key_id: ID of key to retire
            new_key_id: ID for new key
            
        Returns:
            New key bundle or None if rotation failed
        """
        # Get old key to copy its properties
        old_key = self.key_store.retrieve_key(old_key_id)
        if not old_key:
            return None
        
        # Retire old key
        if not self.key_store.update_key_status(old_key_id, KeyStatus.RETIRED):
            return None
        
        # Generate new key with same properties
        if old_key.metadata.key_type == KeyType.SIGNING:
            return self.generate_signing_key(
                new_key_id,
                old_key.metadata.purpose,
                tags=old_key.metadata.tags
            )
        elif old_key.metadata.key_type == KeyType.MASTER:
            key_length = len(old_key.key_material) if old_key.key_material else 32
            return self.generate_master_key(
                new_key_id,
                old_key.metadata.purpose,
                key_length,
                old_key.metadata.tags
            )
        
        return None
    
    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke a key (mark as unusable).
        
        Args:
            key_id: ID of key to revoke
            
        Returns:
            True if revocation successful
        """
        return self.key_store.update_key_status(key_id, KeyStatus.REVOKED)
    
    def get_expiring_keys(self, days_ahead: int = 30) -> List[KeyMetadata]:
        """
        Get keys that will expire within specified days.
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of keys expiring soon
        """
        all_keys = self.key_store.list_keys(status=KeyStatus.ACTIVE)
        expiring = []
        
        for key_meta in all_keys:
            days_until_expiry = key_meta.days_until_expiry()
            if days_until_expiry is not None and days_until_expiry <= days_ahead:
                expiring.append(key_meta)
        
        return expiring
    
    def cleanup_retired_keys(self, retention_days: int = 90) -> List[str]:
        """
        Clean up retired keys older than retention period.
        
        Args:
            retention_days: Days to retain retired keys
            
        Returns:
            List of deleted key IDs
        """
        retired_keys = self.key_store.list_keys(status=KeyStatus.RETIRED)
        deleted = []
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        
        for key_meta in retired_keys:
            if key_meta.retired_at:
                retired_date = datetime.fromisoformat(key_meta.retired_at.replace('Z', '+00:00'))
                if retired_date < cutoff_date:
                    if self.key_store.delete_key(key_meta.key_id):
                        deleted.append(key_meta.key_id)
        
        return deleted
    
    def export_public_keys(self) -> Dict[str, str]:
        """
        Export all public keys for sharing/verification.
        
        Returns:
            Dictionary mapping key IDs to public key PEMs
        """
        public_keys = {}
        
        signing_keys = self.key_store.list_keys(key_type=KeyType.SIGNING)
        for key_meta in signing_keys:
            if key_meta.status in [KeyStatus.ACTIVE, KeyStatus.RETIRED]:
                key_bundle = self.key_store.retrieve_key(key_meta.key_id)
                if key_bundle and key_bundle.public_key_pem:
                    public_keys[key_meta.key_id] = key_bundle.public_key_pem
        
        return public_keys


# Factory functions

def create_filesystem_key_manager(key_dir: str = "keys", 
                                key_validity_days: int = 365) -> KeyManager:
    """Create key manager with file system backend."""
    key_store = FileSystemKeyStore(key_dir)
    return KeyManager(key_store, key_validity_days)


def create_default_ciaf_key_manager() -> KeyManager:
    """Create default CIAF key manager with standard configuration."""
    return create_filesystem_key_manager("ciaf_keys", 365)


# Convenience functions for common operations

def generate_ciaf_signing_key(key_id: str, purpose: str = "ciaf_signing") -> Ed25519Signer:
    """
    Generate and store a CIAF signing key, return signer.
    
    Args:
        key_id: Unique key identifier
        purpose: Key purpose description
        
    Returns:
        Ed25519Signer instance
    """
    key_manager = create_default_ciaf_key_manager()
    key_bundle = key_manager.generate_signing_key(key_id, purpose)
    signer = key_bundle.get_signer()
    if not signer:
        raise RuntimeError("Failed to create signer from generated key")
    return signer


def get_ciaf_signer(key_id: Optional[str] = None, purpose: str = "ciaf_signing") -> Optional[Ed25519Signer]:
    """
    Get an active CIAF signer.
    
    Args:
        key_id: Specific key ID to retrieve, or None for any active key
        purpose: Key purpose filter
        
    Returns:
        Ed25519Signer instance or None
    """
    key_manager = create_default_ciaf_key_manager()
    
    if key_id:
        key_bundle = key_manager.key_store.retrieve_key(key_id)
    else:
        key_bundle = key_manager.get_active_signing_key(purpose)
    
    if key_bundle:
        return key_bundle.get_signer()
    return None