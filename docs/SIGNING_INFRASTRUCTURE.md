# CIAF-LCM Cryptographic Standard Implementation Guide

## Version: 2.0.0 (Production-Ready Signing Infrastructure)
## Last Updated: March 30, 2026

---

## Executive Summary

This document defines the **mandatory cryptographic standard** for CIAF-LCM and provides implementation guidance for the shared signing infrastructure. This standard supersedes earlier proof-metadata approaches with a production-ready, KMS/HSM-integrated signing system.

---

## 1. CIAF-LCM Cryptographic Standard

### 1.1 Core Requirements

**SHA-256** is the standard content hash for all signed CIAF-LCM objects.

**Ed25519** is the standard signature algorithm for externally verifiable receipts.

**KMS, HSM, CloudHSM, and equivalent external key managers** are approved key custody and signing backends.

### 1.2 Mandatory Fields for Externally Verifiable Receipts

Every signed receipt object **MUST** include:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `signature_algorithm` | string (const: "Ed25519") | ✅ | Signature algorithm |
| `key_id` | string | ✅ | Signing key identifier |
| `canonicalization_version` | string | ✅ | Canonicalization profile |
| `key_backend` | enum | ✅ | Key custody backend |
| `payload_hash` | string (64 hex chars) | ✅ | SHA-256 of payload |
| `signature_value` | string | ✅ | Encoded signature |
| `signed_at` | RFC3339 datetime | ✅ | Signing timestamp |

**Recommended fields:**
- `signing_service`: Logical service performing signing
- `public_key_ref`: Reference to verification key (e.g., jwks:// URL)
- `verification_method`: Verification procedure identifier

---

## 2. Schema Architecture

### 2.1 Shared Schema Components

CIAF-LCM uses **three reusable shared schemas** that all other schemas reference:

#### **common/signature-metadata.json**
Describes how an object was signed (algorithm, key_id, backend, etc.)

#### **common/signature-envelope.json**
Contains the actual signature payload (hash, signature_value, metadata)

#### **common/cryptographic-policy.json**
Defines the standard cryptographic policy profile

### 2.2 Schema Folder Structure

```
ciaf/schemas/
├── common/
│   ├── signature-metadata.json
│   ├── signature-envelope.json
│   └── cryptographic-policy.json
├── receipts/
│   ├── runtime-receipt.json
│   └── runtime-receipt-example.json
├── merkle/
│   ├── merkle-batch.json
│   └── merkle-batch-example.json
├── [existing schemas remain at root]
│   ├── anchor.schema.json
│   ├── capsule.schema.json
│   └── ...
```

---

## 3. Using the Shared Signing Schema

### 3.1 Reference Pattern

Any CIAF schema requiring cryptographic signatures should include:

```json
{
  "type": "object",
  "properties": {
    "signature": {
      "$ref": "https://cognitiveinsight.ai/schemas/common/signature-envelope.json"
    }
  }
}
```

This automatically provides:
- `payload_hash` (SHA-256 of canonicalized content)
- `hash_algorithm` (const: "SHA-256")
- `signature_value` (Ed25519 signature)
- `signature_encoding` (base64, base64url, or hex)
- `signed_at` (RFC3339 timestamp)
- `metadata` (full signature metadata with key_backend, key_id, etc.)

### 3.2 Complete Example

See [runtime-receipt-example.json](d:/Github/UsefulStuf/Resume/base/pyciaf/ciaf/schemas/receipts/runtime-receipt-example.json) for a fully populated example including:
- Event data (receipt_id, type, actor, model, policy_decision)
- Cryptographic signature envelope
- Complete signature metadata with KMS backend reference

---

## 4. Key Backend Requirements

### 4.1 Approved Backends

| Backend | Description | Use Case |
|---------|-------------|----------|
| `kms` | AWS KMS, Azure Key Vault, GCP KMS | Cloud-native production |
| `hsm` | On-premises Hardware Security Module | Enterprise data centers |
| `cloudhsm` | AWS CloudHSM, Azure Dedicated HSM | Hybrid/multi-cloud |
| `external_kms` | Third-party KMS (e.g., HashiCorp Vault) | Custom integrations |
| `local` | Local key storage (dev/test only) | ⚠️ **NOT for production** |

### 4.2 Key ID Format

Key IDs should follow the pattern:

```
<backend>:<key-path-or-alias>

Examples:
- aws-kms:alias/ciaf-receipts-prod-2026-03
- cloudhsm:ciaf-merkle-signer-prod
- hsm:partition-1/signing-key-prod
- azure-keyvault:https://ciaf-vault.vault.azure.net/keys/signing-2026-03
```

### 4.3 Public Key Reference Format

Public keys should be referenced using verifiable URIs:

```
jwks://cognitiveinsight.ai/keys/ciaf-prod-2026-01
https://cognitiveinsight.ai/.well-known/jwks.json
x509://path/to/cert.pem
```

---

## 5. Canonicalization Standard

### 5.1 Canonicalization Version

Default: `"RFC8785-like/1.0"`

This follows JSON Canonicalization Scheme (JCS) with:
- Keys sorted alphabetically
- No whitespace
- Compact separators (`,` and `:`)
- Unicode preserved (no escaping)

### 5.2 Implementation

**Python:**
```python
import json

def canonicalize(obj):
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False
    ).encode('utf-8')

# Compute payload hash
import hashlib
canonical_bytes = canonicalize(receipt_data)
payload_hash = hashlib.sha256(canonical_bytes).hexdigest()
```

**JavaScript:**
```javascript
function canonicalize(obj) {
    return JSON.stringify(obj, Object.keys(obj).sort());
}

// Compute payload hash
const crypto = require('crypto');
const canonical = canonicalize(receiptData);
const payloadHash = crypto.createHash('sha256')
    .update(canonical)
    .digest('hex');
```

---

## 6. Signing Workflow

### 6.1 Complete Signing Process

```
1. Prepare receipt/event data
   ↓
2. Canonicalize using RFC8785-like/1.0
   ↓
3. Compute SHA-256 payload_hash
   ↓
4. Send payload_hash to KMS/HSM for signing
   ↓
5. Receive Ed25519 signature_value
   ↓
6. Build signature envelope with:
   - payload_hash
   - hash_algorithm: "SHA-256"
   - signature_value (base64 encoded)
   - signature_encoding: "base64"
   - signed_at (current timestamp)
   - metadata:
     * signature_algorithm: "Ed25519"
     * key_id: <from KMS>
     * canonicalization_version: "RFC8785-like/1.0"
     * key_backend: <kms|hsm|cloudhsm>
     * signing_service: <service name>
     * public_key_ref: <jwks:// URL>
     * verification_method: "ciaf-verify/v1"
   ↓
7. Attach signature envelope to receipt/event
   ↓
8. Store in vault or transmit
```

### 6.2 Python Implementation Example

```python
from nacl.signing import SigningKey
from nacl.encoding import Base64Encoder
import json
import hashlib
from datetime import datetime, timezone

def sign_receipt(receipt_data, signing_key, key_id, key_backend):
    """
    Sign a CIAF receipt using Ed25519.
    
    Args:
        receipt_data: Dictionary containing receipt data
        signing_key: SigningKey instance (or KMS client)
        key_id: Key identifier string
        key_backend: Backend type (kms, hsm, etc.)
    
    Returns:
        Complete receipt with signature envelope
    """
    # 1. Canonicalize
    canonical = json.dumps(
        receipt_data,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False
    ).encode('utf-8')
    
    # 2. Compute hash
    payload_hash = hashlib.sha256(canonical).hexdigest()
    
    # 3. Sign (using local key for example; use KMS API in production)
    signature_bytes = signing_key.sign(canonical).signature
    signature_value = Base64Encoder.encode(signature_bytes).decode('utf-8')
    
    # 4. Build signature envelope
    signature_envelope = {
        "payload_hash": payload_hash,
        "hash_algorithm": "SHA-256",
        "signature_value": signature_value,
        "signature_encoding": "base64",
        "signed_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "signature_algorithm": "Ed25519",
            "key_id": key_id,
            "canonicalization_version": "RFC8785-like/1.0",
            "key_backend": key_backend,
            "signing_service": "ciaf-vault-signer",
            "public_key_ref": f"jwks://cognitiveinsight.ai/keys/{key_id}",
            "verification_method": "ciaf-verify/v1"
        }
    }
    
    # 5. Attach to receipt
    receipt_data['signature'] = signature_envelope
    
    return receipt_data
```

---

## 7. Verification Workflow

### 7.1 Complete Verification Process

```
1. Extract signature envelope from receipt
   ↓
2. Extract original receipt data (without signature)
   ↓
3. Canonicalize using same canonicalization_version
   ↓
4. Compute SHA-256 hash
   ↓
5. Verify computed hash matches payload_hash
   ↓
6. Retrieve public key using public_key_ref or key_id
   ↓
7. Verify signature_value using Ed25519
   ↓
8. Verify signed_at is within acceptable time window
   ↓
9. Return verification result
```

### 7.2 Python Verification Example

```python
from nacl.signing import VerifyKey
from nacl.encoding import Base64Encoder
import json
import hashlib

def verify_receipt(receipt_with_signature, public_key):
    """
    Verify a signed CIAF receipt.
    
    Args:
        receipt_with_signature: Complete receipt dict with signature
        public_key: VerifyKey instance
    
    Returns:
        True if valid, False otherwise
    """
    # 1. Extract signature
    signature_envelope = receipt_with_signature.pop('signature')
    
    # 2. Canonicalize original data
    canonical = json.dumps(
        receipt_with_signature,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False
    ).encode('utf-8')
    
    # 3. Verify hash
    computed_hash = hashlib.sha256(canonical).hexdigest()
    if computed_hash != signature_envelope['payload_hash']:
        return False
    
    # 4. Verify signature
    signature_bytes = Base64Encoder.decode(
        signature_envelope['signature_value'].encode('utf-8')
    )
    
    try:
        public_key.verify(canonical, signature_bytes)
        return True
    except Exception:
        return False
```

---

## 8. Merkle Batch Signing

Merkle batches follow the same signing standard but operate on Merkle roots instead of individual events.

### 8.1 Merkle Batch Schema

See [merkle-batch.json](d:/Github/UsefulStuf/Resume/base/pyciaf/ciaf/schemas/merkle/merkle-batch.json)

### 8.2 Batch Creation Process

```
1. Collect receipts until threshold (e.g., 1000 for inference)
   ↓
2. Compute leaf hashes: SHA-256(canonical(receipt))
   ↓
3. Build Merkle tree using SHA-256
   ↓
4. Extract root_hash
   ↓
5. Create batch object with root_hash, leaf_count, timestamps
   ↓
6. Sign batch object using signature-envelope.json
   ↓
7. Store signed batch
```

### 8.3 Example

See [merkle-batch-example.json](d:/Github/UsefulStuf/Resume/base/pyciaf/ciaf/schemas/merkle/merkle-batch-example.json)

---

## 9. Migration from proof-metadata.schema.json

### 9.1 Key Differences

| Old (proof-metadata) | New (signature-envelope) |
|---------------------|-------------------------|
| Flat structure | Nested envelope structure |
| `receipt_hash` field | `payload_hash` field |
| Optional `key_backend` | **Mandatory** `key_backend` |
| `signature` as direct field | `signature_value` in envelope |
| No encoding specification | Explicit `signature_encoding` |

### 9.2 Migration Steps

1. Replace `$ref: "proof-metadata.schema.json"` with `$ref: "common/signature-envelope.json"`
2. Update code to nest signature fields under `signature` object
3. Rename `receipt_hash` → `payload_hash`
4. Make `key_backend` mandatory
5. Add `signature_encoding` field (default: "base64")
6. Update verification code to extract signature from envelope

---

## 10. Testing and Validation

### 10.1 Schema Validation

Validate all receipts against JSON Schema:

```bash
# Using ajv-cli
ajv validate -s ciaf/schemas/receipts/runtime-receipt.json \
             -r ciaf/schemas/common/signature-envelope.json \
             -r ciaf/schemas/common/signature-metadata.json \
             -d receipt-instance.json
```

### 10.2 Signature Validation Test Cases

- [ ] Valid signature verifies successfully
- [ ] Modified content fails verification
- [ ] Tampered signature_value fails verification
- [ ] Mismatched payload_hash fails verification
- [ ] Expired key_id is detected
- [ ] Unknown key_backend is rejected
- [ ] Invalid canonicalization_version is rejected

---

## 11. Production Deployment Checklist

- [ ] KMS/HSM backend configured and tested
- [ ] Key rotation policy implemented
- [ ] Public keys published at `jwks://` URL
- [ ] Canonicalization version documented
- [ ] All receipts include mandatory fields
- [ ] Verification service deployed
- [ ] Audit trail configured for key usage
- [ ] Monitoring for signature failures enabled
- [ ] Backup key custody procedures documented

---

## 12. Compliance Mapping

| Standard Field | GDPR | EU AI Act | SOC 2 | ISO 27001 |
|----------------|------|-----------|-------|-----------|
| signature_algorithm | Art. 32 | Art. 17 | CC6.1 | A.10.1.1 |
| key_backend | Art. 25 | Art. 14 | CC6.3 | A.9.4.1 |
| signed_at | Art. 30 | Art. 12 | CC7.2 | A.12.4.1 |
| payload_hash | Art. 32 | Art. 17 | CC6.7 | A.10.1.2 |

---

## 13. References

- **RFC 8785:** JSON Canonicalization Scheme (JCS)
- **RFC 8032:** Edwards-Curve Digital Signature Algorithm (EdDSA)
- **FIPS 180-4:** SHA-256 Specification
- **NIST SP 800-57:** Key Management Recommendations

---

**Document Owner:** CIAF Architecture Team  
**Implementation Status:** Production Ready  
**Next Review:** June 30, 2026
