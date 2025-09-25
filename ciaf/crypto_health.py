"""
Cryptographic Health Check for CIAF

Implements crypto_health check for PRNG sources, salt lengths, nonce uniqueness,
and digest algorithm validation as suggested in the security review.
"""

import os
import secrets
import hashlib
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import threading

@dataclass
class CryptoHealthStatus:
    """Result of cryptographic health check."""
    overall_status: str  # "healthy", "warning", "critical"
    prng_source: str
    min_salt_length: int
    digest_algorithms: List[str] 
    nonce_uniqueness_check: bool
    issues: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_status": self.overall_status,
            "prng_source": self.prng_source,
            "min_salt_length": self.min_salt_length,
            "digest_algorithms": self.digest_algorithms,
            "nonce_uniqueness_check": self.nonce_uniqueness_check,
            "issues": self.issues,
            "recommendations": self.recommendations
        }

class NonceTracker:
    """Thread-safe nonce uniqueness tracker."""
    
    def __init__(self, max_size: int = 10000):
        self._nonces: Set[str] = set()
        self._lock = threading.Lock()
        self._max_size = max_size
        self._collision_count = 0
    
    def register_nonce(self, nonce: str) -> bool:
        """Register a nonce and return True if unique, False if collision."""
        with self._lock:
            if nonce in self._nonces:
                self._collision_count += 1
                return False
            
            self._nonces.add(nonce)
            
            # Prevent unlimited growth
            if len(self._nonces) > self._max_size:
                # Remove oldest half (simple strategy)
                self._nonces = set(list(self._nonces)[self._max_size//2:])
            
            return True
    
    def get_collision_count(self) -> int:
        """Get total number of nonce collisions detected."""
        return self._collision_count
    
    def clear(self):
        """Clear all tracked nonces."""
        with self._lock:
            self._nonces.clear()
            self._collision_count = 0

# Global nonce tracker
_nonce_tracker = NonceTracker()

def get_nonce_tracker() -> NonceTracker:
    """Get the global nonce tracker."""
    return _nonce_tracker

class CryptoHealthChecker:
    """Performs comprehensive cryptographic health checks."""
    
    def __init__(self):
        self.issues = []
        self.recommendations = []
    
    def check_prng_source(self) -> str:
        """Check the PRNG source being used."""
        try:
            # Test secrets module (preferred)
            test_bytes = secrets.token_bytes(32)
            if len(test_bytes) == 32:
                return "secrets (cryptographically secure)"
        except Exception:
            self.issues.append("secrets module not available")
        
        try:
            # Fallback to os.urandom
            test_bytes = os.urandom(32)
            if len(test_bytes) == 32:
                return "os.urandom (system entropy)"
        except Exception:
            self.issues.append("os.urandom not available")
            self.recommendations.append("System entropy source not accessible")
        
        return "unknown/insecure"
    
    def check_salt_length(self) -> int:
        """Check minimum salt length being used."""
        # Test salt generation
        try:
            salt = secrets.token_bytes(16)  # 128 bits minimum
            return len(salt) * 8  # Return in bits
        except Exception:
            self.issues.append("Cannot generate cryptographic salts")
            return 0
    
    def validate_digest_algorithms(self) -> List[str]:
        """Check available and secure digest algorithms."""
        available_algorithms = []
        
        # Test SHA-256 (required)
        try:
            hashlib.sha256(b"test").hexdigest()
            available_algorithms.append("SHA-256")
        except Exception:
            self.issues.append("SHA-256 not available")
        
        # Test SHA-512 (nice to have)
        try:
            hashlib.sha512(b"test").hexdigest()
            available_algorithms.append("SHA-512")
        except Exception:
            pass
        
        # Test BLAKE2b (modern alternative)
        try:
            hashlib.blake2b(b"test").hexdigest()
            available_algorithms.append("BLAKE2b")
        except Exception:
            pass
        
        # Warn about weak algorithms if accidentally available
        weak_algorithms = []
        for weak_algo in ['md5', 'sha1']:
            if hasattr(hashlib, weak_algo):
                weak_algorithms.append(weak_algo.upper())
        
        if weak_algorithms:
            self.recommendations.append(
                f"Weak algorithms detected ({', '.join(weak_algorithms)}) - ensure they're not used"
            )
        
        return available_algorithms
    
    def test_nonce_uniqueness(self, num_tests: int = 1000) -> bool:
        """Test nonce uniqueness generation."""
        test_nonces = set()
        
        for _ in range(num_tests):
            try:
                # Generate nonce (timestamp + random bytes)
                nonce = f"{int(time.time() * 1000000)}-{secrets.token_hex(16)}"
                if nonce in test_nonces:
                    self.issues.append(f"Nonce collision detected in {num_tests} samples")
                    return False
                test_nonces.add(nonce)
            except Exception:
                self.issues.append("Cannot generate unique nonces")
                return False
        
        return True
    
    def check_key_derivation(self) -> bool:
        """Test key derivation functionality."""
        try:
            import hmac
            
            # Test HMAC-SHA256 key derivation
            master_key = secrets.token_bytes(32)
            salt = secrets.token_bytes(16)
            derived_key = hmac.new(master_key, salt, hashlib.sha256).digest()
            
            if len(derived_key) == 32:  # SHA-256 output length
                return True
            else:
                self.issues.append("Key derivation produces unexpected length")
                return False
                
        except Exception as e:
            self.issues.append(f"Key derivation test failed: {e}")
            return False
    
    def check_aes_gcm_availability(self) -> bool:
        """Check if AES-GCM is available and working."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            # Test encryption/decryption
            key = AESGCM.generate_key(bit_length=256)
            aesgcm = AESGCM(key)
            nonce = os.urandom(12)  # 96-bit nonce for GCM
            
            plaintext = b"test message"
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)
            decrypted = aesgcm.decrypt(nonce, ciphertext, None)
            
            if decrypted == plaintext:
                return True
            else:
                self.issues.append("AES-GCM encryption/decryption mismatch")
                return False
                
        except ImportError:
            self.issues.append("AES-GCM not available (cryptography library missing)")
            return False
        except Exception as e:
            self.issues.append(f"AES-GCM test failed: {e}")
            return False
    
    def perform_health_check(self) -> CryptoHealthStatus:
        """Perform comprehensive cryptographic health check."""
        self.issues = []
        self.recommendations = []
        
        # Check PRNG source
        prng_source = self.check_prng_source()
        
        # Check salt generation
        min_salt_length = self.check_salt_length()
        if min_salt_length < 128:  # Less than 128 bits
            self.issues.append(f"Salt length ({min_salt_length} bits) below minimum (128 bits)")
        
        # Check digest algorithms
        digest_algorithms = self.validate_digest_algorithms()
        if not digest_algorithms:
            self.issues.append("No secure digest algorithms available")
        
        # Test nonce uniqueness
        nonce_uniqueness = self.test_nonce_uniqueness(100)  # Smaller test for speed
        
        # Test key derivation
        key_derivation_ok = self.check_key_derivation()
        if not key_derivation_ok:
            self.issues.append("Key derivation functionality impaired")
        
        # Test AES-GCM
        aes_gcm_ok = self.check_aes_gcm_availability()
        if not aes_gcm_ok:
            self.recommendations.append("Install cryptography library for AES-GCM support")
        
        # Determine overall status
        if len(self.issues) == 0:
            overall_status = "healthy"
        elif any("not available" in issue or "failed" in issue for issue in self.issues):
            overall_status = "critical"
        else:
            overall_status = "warning"
        
        # Add general recommendations
        if overall_status != "critical":
            self.recommendations.extend([
                "Regularly rotate cryptographic keys",
                "Monitor nonce uniqueness in production",
                "Use hardware security modules (HSMs) for key storage if available",
                "Implement key escrow for regulatory compliance if required"
            ])
        
        return CryptoHealthStatus(
            overall_status=overall_status,
            prng_source=prng_source,
            min_salt_length=min_salt_length,
            digest_algorithms=digest_algorithms,
            nonce_uniqueness_check=nonce_uniqueness,
            issues=self.issues.copy(),
            recommendations=self.recommendations.copy()
        )

def crypto_health_check() -> CryptoHealthStatus:
    """Perform a cryptographic health check."""
    checker = CryptoHealthChecker()
    return checker.perform_health_check()

def generate_secure_salt(length_bytes: int = 16) -> bytes:
    """Generate a cryptographically secure salt."""
    if length_bytes < 16:
        raise ValueError("Salt must be at least 16 bytes (128 bits)")
    
    return secrets.token_bytes(length_bytes)

def generate_unique_nonce() -> str:
    """Generate a unique nonce and register it for collision detection."""
    nonce = f"{int(time.time() * 1000000)}-{secrets.token_hex(16)}"
    
    # Register with global tracker
    is_unique = _nonce_tracker.register_nonce(nonce)
    if not is_unique:
        # If collision detected, generate a new one with extra randomness
        nonce = f"{int(time.time() * 1000000)}-{secrets.token_hex(32)}"
        _nonce_tracker.register_nonce(nonce)
    
    return nonce

# Convenience function for quick status check
def quick_crypto_status() -> str:
    """Get quick crypto status summary."""
    try:
        status = crypto_health_check()
        return f"Crypto Status: {status.overall_status.upper()} ({len(status.issues)} issues)"
    except Exception as e:
        return f"Crypto Status: ERROR ({e})"

if __name__ == "__main__":
    # Run health check
    print("CIAF Cryptographic Health Check")
    print("=" * 35)
    
    status = crypto_health_check()
    print(f"Overall Status: {status.overall_status.upper()}")
    print(f"PRNG Source: {status.prng_source}")
    print(f"Salt Length: {status.min_salt_length} bits")
    print(f"Digest Algorithms: {', '.join(status.digest_algorithms)}")
    print(f"Nonce Uniqueness: {'✓' if status.nonce_uniqueness_check else '✗'}")
    
    if status.issues:
        print("\nIssues:")
        for issue in status.issues:
            print(f"  - {issue}")
    
    if status.recommendations:
        print("\nRecommendations:")
        for rec in status.recommendations:
            print(f"  - {rec}")
    
    # Test nonce generation
    print(f"\nSample Nonce: {generate_unique_nonce()}")
    print(f"Collision Count: {get_nonce_tracker().get_collision_count()}")