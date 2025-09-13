# CIAF Core Components

The core package provides the foundational cryptographic and utility components that underpin all CIAF operations. It implements the essential building blocks for secure anchor derivation, encryption, hashing, and Merkle tree operations.

## Overview

The core package contains the cryptographic primitives and foundational utilities that ensure CIAF's security and integrity:

- **Cryptographic Functions** — AES-GCM encryption, SHA-256 hashing, HMAC operations
- **Anchor Derivation** — Hierarchical anchor/key derivation system
- **Merkle Trees** — Tamper-evident data structures for provenance
- **Utility Functions** — Secure random generation, hex encoding/decoding
- **Constants** — Standard cryptographic parameters

## Components

### Cryptographic Utilities (`crypto.py`)

Core cryptographic functions providing industry-standard security primitives.

**Key Features:**
- **AES-256-GCM** — Authenticated encryption with optional Additional Authenticated Data (AAD)
- **SHA-256** — Cryptographic hashing for integrity verification
- **HMAC-SHA256** — Message authentication codes for anchor derivation
- **Secure Random** — Cryptographically secure random number generation
- **CryptoUtils Class** — Convenient wrapper for all cryptographic operations

**Usage Examples:**

```python
from ciaf.core import (
    encrypt_aes_gcm, decrypt_aes_gcm, 
    sha256_hash, hmac_sha256, 
    secure_random_bytes, CryptoUtils
)

# AES-GCM Encryption
key = secure_random_bytes(32)  # 256-bit key
plaintext = b"sensitive data"
aad = b"additional authenticated data"

ciphertext, nonce, tag = encrypt_aes_gcm(key, plaintext, aad)
decrypted = decrypt_aes_gcm(key, ciphertext, nonce, tag, aad)

# SHA-256 Hashing
data = b"data to hash"
hash_value = sha256_hash(data)
print(f"SHA-256: {hash_value}")

# HMAC-SHA256
secret_key = secure_random_bytes(32)
message = b"message to authenticate"
mac = hmac_sha256(secret_key, message)

# Using CryptoUtils class
crypto = CryptoUtils()
encrypted_data = crypto.encrypt(plaintext, key, aad)
decrypted_data = crypto.decrypt(encrypted_data, key, aad)
```

### Base Anchor Management (`base_anchor.py`)

Hierarchical anchor derivation system providing the cryptographic foundation for CIAF's lazy materialization.

**Key Features:**
- **Master Anchor Derivation** — Root anchor from password and salt
- **Dataset Anchor Derivation** — Dataset-specific anchors from master anchor
- **Capsule Anchor Derivation** — Item-specific anchors for lazy materialization
- **Model Anchor Derivation** — Model-specific anchors for parameter fingerprinting
- **Backwards Compatibility** — Support for legacy "key" terminology
- **Hex Encoding/Decoding** — Convenient string representation of binary anchors

**Usage Examples:**

```python
from ciaf.core import (
    derive_master_anchor, derive_dataset_anchor, 
    derive_capsule_anchor, derive_model_anchor,
    BaseAnchorManager, to_hex, from_hex
)

# Master anchor derivation
master_password = "secure_master_password"
salt = secure_random_bytes(16)
master_anchor = derive_master_anchor(master_password, salt)

# Dataset anchor derivation
dataset_metadata = {"dataset_id": "medical_data", "version": "v1.0"}
dataset_anchor = derive_dataset_anchor(master_anchor, dataset_metadata)

# Capsule anchor derivation (for lazy materialization)
item_metadata = {"item_id": "patient_001", "type": "xray"}
capsule_anchor = derive_capsule_anchor(dataset_anchor, item_metadata)

# Model anchor derivation
model_metadata = {"model_name": "classifier", "version": "v1.0"}
model_anchor = derive_model_anchor(master_anchor, model_metadata)

# Hex representation for storage/display
anchor_hex = to_hex(dataset_anchor)
anchor_bytes = from_hex(anchor_hex)

# Using BaseAnchorManager
anchor_manager = BaseAnchorManager()
anchors = anchor_manager.derive_anchor_hierarchy(
    master_password="password",
    salt=salt,
    dataset_metadata=dataset_metadata,
    item_metadatas=[item_metadata]
)
```

### Merkle Tree Implementation (`merkle.py`)

Tamper-evident data structure for efficient cryptographic proofs and data integrity.

**Key Features:**
- **Binary Tree Structure** — Efficient tree construction with configurable fanout
- **Canonical Concatenation** — Deterministic tree construction
- **Proof Generation** — Cryptographic proofs of inclusion/exclusion
- **Root Hash Computation** — Tamper-evident root hash for data sets
- **Efficient Verification** — O(log n) proof verification

**Usage Examples:**

```python
from ciaf.core import MerkleTree, sha256_hash

# Create Merkle tree from data items
data_items = [
    b"patient_001_data",
    b"patient_002_data", 
    b"patient_003_data",
    b"patient_004_data"
]

# Hash data items
hashed_items = [sha256_hash(item) for item in data_items]

# Build Merkle tree
merkle_tree = MerkleTree(hashed_items)

# Get root hash (tamper-evident)
root_hash = merkle_tree.get_root()
print(f"Merkle root: {root_hash}")

# Generate proof of inclusion
item_index = 1
proof = merkle_tree.get_proof(item_index)
print(f"Proof for item {item_index}: {proof}")

# Verify proof
item_hash = hashed_items[item_index]
is_valid = merkle_tree.verify_proof(item_hash, item_index, proof, root_hash)
print(f"Proof valid: {is_valid}")

# Tree statistics
print(f"Tree height: {merkle_tree.height}")
print(f"Leaf count: {merkle_tree.leaf_count}")
```

### Legacy Key Management (`keys.py`)

Backwards compatibility support for legacy "key" terminology while transitioning to "anchor" terminology.

**Key Features:**
- **Legacy API Support** — Maintains compatibility with older CIAF versions
- **Transparent Migration** — Gradual transition from "keys" to "anchors"
- **Alias Functions** — All key functions aliased to anchor functions
- **Deprecation Warnings** — Helpful warnings for legacy usage

**Usage Examples:**

```python
# Legacy usage (still supported)
from ciaf.core import derive_master_key, derive_dataset_key, derive_capsule_key

# These work identically to anchor functions
master_key = derive_master_key("password", salt)
dataset_key = derive_dataset_key(master_key, dataset_metadata)
capsule_key = derive_capsule_key(dataset_key, item_metadata)

# Modern usage (recommended)
from ciaf.core import derive_master_anchor, derive_dataset_anchor, derive_capsule_anchor

master_anchor = derive_master_anchor("password", salt)
dataset_anchor = derive_dataset_anchor(master_anchor, dataset_metadata) 
capsule_anchor = derive_capsule_anchor(dataset_anchor, item_metadata)

# Both approaches yield identical results
assert master_key == master_anchor
assert dataset_key == dataset_anchor
assert capsule_key == capsule_anchor
```

### Constants (`constants.py`)

Standard cryptographic parameters and configuration values.

**Key Constants:**
- **SALT_LENGTH** — Standard salt length for cryptographic operations (16 bytes)
- **AES_KEY_LENGTH** — AES key length (32 bytes for AES-256)
- **NONCE_LENGTH** — GCM nonce length (12 bytes recommended)
- **HASH_LENGTH** — SHA-256 output length (32 bytes)

**Usage Examples:**

```python
from ciaf.core.constants import SALT_LENGTH, AES_KEY_LENGTH

# Generate salt of standard length
salt = secure_random_bytes(SALT_LENGTH)

# Generate AES key of correct length
aes_key = secure_random_bytes(AES_KEY_LENGTH)

# Validate lengths
assert len(salt) == SALT_LENGTH
assert len(aes_key) == AES_KEY_LENGTH
```

## Cryptographic Architecture

### Anchor Derivation Hierarchy

```
Master Password + Salt → Master Anchor (HMAC-SHA256)
        ↓
Master Anchor + Dataset Metadata → Dataset Anchor (HMAC-SHA256)
        ↓
Dataset Anchor + Item Metadata → Capsule Anchor (HMAC-SHA256)
        ↓
Master Anchor + Model Metadata → Model Anchor (HMAC-SHA256)
```

**Properties:**
- **Deterministic** — Same inputs always produce same outputs
- **One-Way** — Cannot derive parent from child anchors
- **Collision Resistant** — Based on SHA-256 cryptographic security
- **Hierarchical** — Parent anchors control access to child anchors

### AES-GCM Encryption Scheme

```
Key (32 bytes) + Plaintext + AAD → Ciphertext + Nonce (12 bytes) + Tag (16 bytes)
```

**Security Properties:**
- **Confidentiality** — AES-256 encryption
- **Integrity** — GCM authentication tag
- **Authenticity** — Additional Authenticated Data (AAD) support
- **Nonce-based** — Unique nonce for each encryption operation

### Merkle Tree Construction

```
Leaf Nodes: H(data₁), H(data₂), H(data₃), H(data₄)
Internal Nodes: H(H(data₁) || H(data₂)), H(H(data₃) || H(data₄))
Root: H(H(H(data₁) || H(data₂)) || H(H(data₃) || H(data₄)))
```

**Properties:**
- **Tamper Detection** — Any change alters the root hash
- **Efficient Proofs** — O(log n) proof size and verification time
- **Canonical Construction** — Deterministic tree building
- **Scalable** — Supports arbitrary data set sizes

## Security Considerations

### Cryptographic Standards

**Algorithms Used:**
- **AES-256-GCM** — NIST-approved authenticated encryption
- **SHA-256** — NIST-approved cryptographic hash function
- **HMAC-SHA256** — RFC 2104 compliant message authentication
- **PBKDF2** — RFC 2898 compliant key derivation (when needed)

**Random Number Generation:**
- Uses operating system's cryptographically secure random source
- Suitable for cryptographic operations (keys, nonces, salts)
- Entropy sourced from `/dev/urandom` (Unix) or `CryptGenRandom` (Windows)

### Key Management

**Best Practices:**
```python
# Use strong passwords for master anchor derivation
master_password = "cryptographically_strong_password_123!@#"

# Generate unique salts for each dataset
salt = secure_random_bytes(SALT_LENGTH)

# Store anchors securely (consider using hardware security modules)
# Never log or store master passwords in plaintext
```

### Memory Security

**Considerations:**
- Sensitive data (keys, passwords) should be zeroed after use
- Consider using `mlock()` to prevent swapping for highly sensitive operations
- Be aware of garbage collection in Python may leave copies in memory

## Performance Characteristics

### Cryptographic Operations

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| SHA-256 Hash | O(n) | ~1µs per KB |
| HMAC-SHA256 | O(n) | ~1µs per KB |
| AES-GCM Encrypt | O(n) | ~10µs per KB |
| AES-GCM Decrypt | O(n) | ~10µs per KB |
| Anchor Derivation | O(1) | ~10µs |

### Merkle Tree Operations

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| Tree Construction | O(n log n) | ~1ms per 1000 items |
| Root Computation | O(n) | ~500µs per 1000 items |
| Proof Generation | O(log n) | ~10µs per proof |
| Proof Verification | O(log n) | ~5µs per proof |

## Integration Patterns

### With CIAF Framework

```python
from ciaf.api import CIAFFramework
from ciaf.core import derive_master_anchor, secure_random_bytes

# Framework uses core components internally
framework = CIAFFramework("MyProject")

# Core components power the framework's operations
salt = secure_random_bytes(16)
master_anchor = derive_master_anchor("password", salt)

# Framework abstracts core complexity
dataset_anchor = framework.create_dataset_anchor(
    dataset_id="my_data",
    dataset_metadata={"version": "v1.0"},
    master_password="password"
)
```

### With Anchoring System

```python
from ciaf.anchoring import DatasetAnchor
from ciaf.core import derive_dataset_anchor, secure_random_bytes

# Anchoring system builds on core cryptographic functions
salt = secure_random_bytes(16)
anchor = DatasetAnchor(
    dataset_id="dataset",
    master_password="password",
    salt=salt
)

# Core functions provide the cryptographic foundation
print(f"Dataset anchor: {anchor.dataset_anchor_hex}")
```

### With Provenance System

```python
from ciaf.provenance import ProvenanceCapsule, TrainingSnapshot
from ciaf.core import MerkleTree, sha256_hash

# Provenance system uses Merkle trees for integrity
capsule_hashes = [sha256_hash(capsule.content) for capsule in capsules]
merkle_tree = MerkleTree(capsule_hashes)

training_snapshot = TrainingSnapshot(
    model_id="model_v1",
    training_data_root=merkle_tree.get_root(),
    timestamp=datetime.now().isoformat()
)
```

## Error Handling

### Cryptographic Errors

```python
from cryptography.exceptions import InvalidTag
from ciaf.core import encrypt_aes_gcm, decrypt_aes_gcm

try:
    # Encryption/decryption operations
    ciphertext, nonce, tag = encrypt_aes_gcm(key, plaintext)
    decrypted = decrypt_aes_gcm(key, ciphertext, nonce, tag)
    
except InvalidTag:
    print("Authentication failed - data may be tampered")
except ValueError as e:
    print(f"Invalid parameters: {e}")
except Exception as e:
    print(f"Cryptographic error: {e}")
```

### Anchor Derivation Errors

```python
from ciaf.core import derive_master_anchor, derive_dataset_anchor

try:
    master_anchor = derive_master_anchor("password", salt)
    dataset_anchor = derive_dataset_anchor(master_anchor, metadata)
    
except ValueError as e:
    print(f"Invalid anchor parameters: {e}")
except TypeError as e:
    print(f"Wrong parameter types: {e}")
```

## Testing and Validation

### Unit Testing

```python
import unittest
from ciaf.core import sha256_hash, encrypt_aes_gcm, MerkleTree

class TestCoreComponents(unittest.TestCase):
    
    def test_sha256_deterministic(self):
        """Test SHA-256 produces deterministic results."""
        data = b"test data"
        hash1 = sha256_hash(data)
        hash2 = sha256_hash(data)
        self.assertEqual(hash1, hash2)
    
    def test_aes_gcm_roundtrip(self):
        """Test AES-GCM encryption/decryption roundtrip."""
        key = secure_random_bytes(32)
        plaintext = b"sensitive data"
        
        ciphertext, nonce, tag = encrypt_aes_gcm(key, plaintext)
        decrypted = decrypt_aes_gcm(key, ciphertext, nonce, tag)
        
        self.assertEqual(plaintext, decrypted)
    
    def test_merkle_tree_integrity(self):
        """Test Merkle tree root changes with data modification."""
        data1 = [b"item1", b"item2", b"item3"]
        data2 = [b"item1", b"item2", b"modified"]
        
        tree1 = MerkleTree([sha256_hash(d) for d in data1])
        tree2 = MerkleTree([sha256_hash(d) for d in data2])
        
        self.assertNotEqual(tree1.get_root(), tree2.get_root())
```

### Integration Testing

```python
def test_anchor_hierarchy():
    """Test complete anchor derivation hierarchy."""
    from ciaf.core import (
        derive_master_anchor, derive_dataset_anchor, 
        derive_capsule_anchor, secure_random_bytes
    )
    
    # Test data
    password = "test_password"
    salt = secure_random_bytes(16)
    dataset_metadata = {"dataset_id": "test", "version": "v1.0"}
    item_metadata = {"item_id": "item_001"}
    
    # Derive anchor hierarchy
    master_anchor = derive_master_anchor(password, salt)
    dataset_anchor = derive_dataset_anchor(master_anchor, dataset_metadata)
    capsule_anchor = derive_capsule_anchor(dataset_anchor, item_metadata)
    
    # Verify deterministic behavior
    master_anchor2 = derive_master_anchor(password, salt)
    assert master_anchor == master_anchor2
    
    # Verify independence (different salt = different anchors)
    salt2 = secure_random_bytes(16)
    master_anchor3 = derive_master_anchor(password, salt2)
    assert master_anchor != master_anchor3
```

## Contributing

When extending the core components:

1. **Maintain Cryptographic Standards** — Use only NIST-approved algorithms
2. **Add Comprehensive Tests** — Include unit tests and integration tests
3. **Document Security Properties** — Clearly document all cryptographic guarantees
4. **Backwards Compatibility** — Support legacy interfaces during transitions
5. **Performance Benchmarks** — Include performance tests for cryptographic operations

## Dependencies

The core package depends on:
- **cryptography** — Industry-standard cryptographic library
- **hashlib** — Python standard library hashing
- **hmac** — Python standard library HMAC implementation
- **os** — Operating system interface for secure random numbers

---

*For usage examples and integration patterns, see the [examples folder](../examples/) and other CIAF components.*