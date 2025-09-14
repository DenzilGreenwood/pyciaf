# CIAF Receipts Schema

## Overview

CIAF receipts are cryptographically verifiable artifacts that capture the complete provenance of AI operations. This document describes the structure and verification process for CIAF receipts.

## Receipt Types

### Training Receipt

Captures the complete state of a model training operation.

```json
{
  "receipt_id": "training_<timestamp>_<hash>",
  "receipt_type": "training",
  "timestamp": "2025-09-12T10:30:00Z",
  "ciaf_version": "0.1.0",
  
  "dataset": {
    "dataset_id": "customer_reviews_v1",
    "anchor_id": "a1b2c3d4e5f6...",
    "items_count": 10000,
    "merkle_root": "f7e8d9c0b1a2...",
    "leaves": [
      "hash_of_item_1",
      "hash_of_item_2",
      "..."
    ]
  },
  
  "model": {
    "model_name": "sentiment_classifier",
    "version": "v1.0",
    "parameters": {
      "learning_rate": 0.001,
      "batch_size": 32,
      "epochs": 10
    },
    "parameter_fingerprint": "x7y8z9a0b1c2...",
    "architecture": {
      "type": "neural_network",
      "layers": [128, 64, 3],
      "activation": "relu"
    },
    "architecture_fingerprint": "m3n4o5p6q7r8..."
  },
  
  "training": {
    "session_id": "train_001",
    "started_at": "2025-09-12T10:00:00Z",
    "completed_at": "2025-09-12T10:30:00Z",
    "snapshot_id": "snap_a1b2c3d4...",
    "integrity_verified": true
  },
  
  "audit_chain": [
    {
      "event_id": "train_start_001",
      "event_type": "training_started",
      "timestamp": "2025-09-12T10:00:00Z",
      "previous_hash": "0000000000000000...",
      "hash": "calculated_hash_1",
      "details": {
        "user_id": "data_scientist_alice",
        "environment": "production"
      }
    },
    {
      "event_id": "train_complete_001",
      "event_type": "training_completed",
      "timestamp": "2025-09-12T10:30:00Z",
      "previous_hash": "calculated_hash_1",
      "hash": "calculated_hash_2",
      "details": {
        "final_loss": 0.023,
        "accuracy": 0.945
      }
    }
  ],
  
  "verification": {
    "merkle_verification": "passed",
    "anchor_verification": "passed",
    "audit_chain_verification": "passed",
    "overall_status": "verified"
  }
}
```

### Inference Receipt

Captures a single inference operation with full provenance.

```json
{
  "receipt_id": "inference_<timestamp>_<hash>",
  "receipt_type": "inference",
  "timestamp": "2025-09-12T11:00:00Z",
  "ciaf_version": "0.1.0",
  
  "inference": {
    "inference_id": "infer_001",
    "model_name": "sentiment_classifier",
    "model_version": "v1.0",
    "training_snapshot": "snap_a1b2c3d4...",
    "input_hash": "hash_of_input_data",
    "output_hash": "hash_of_model_output",
    "confidence": 0.95
  },
  
  "traceability": {
    "training_receipt": "training_<timestamp>_<hash>",
    "dataset_anchor": "a1b2c3d4e5f6...",
    "model_fingerprint": "x7y8z9a0b1c2..."
  },
  
  "audit_event": {
    "event_id": "inference_001",
    "event_type": "inference_performed",
    "timestamp": "2025-09-12T11:00:00Z",
    "previous_hash": "calculated_hash_2",
    "hash": "calculated_hash_3",
    "details": {
      "user_id": "api_user",
      "session_id": "sess_123"
    }
  },
  
  "verification": {
    "training_linkage": "verified",
    "input_integrity": "verified",
    "output_integrity": "verified",
    "audit_linkage": "verified",
    "overall_status": "verified"
  }
}
```

## Verification Process

### Manual Verification

Receipts can be verified independently using the verifier tool:

```bash
# Verify complete receipt
python tools/verify_receipt.py training_receipt.json

# Verify specific components
python tools/verify_receipt.py --verify-merkle dataset_merkle.json
python tools/verify_receipt.py --verify-audit-chain audit_chain.json
```

### Programmatic Verification

```python
from tools.verify_receipt import CIAFVerifier
import json

# Load receipt
with open('training_receipt.json', 'r') as f:
    receipt = json.load(f)

# Verify receipt
verifier = CIAFVerifier()
is_valid = verifier.verify_receipt(receipt)

print(f"Receipt valid: {is_valid}")
```

## Verification Components

### 1. Merkle Root Verification

Verifies that the Merkle root correctly represents the dataset:

```python
leaves = receipt['dataset']['leaves']
expected_root = receipt['dataset']['merkle_root']

# Rebuild Merkle tree from leaves
is_valid = CIAFVerifier.verify_merkle_root(leaves, expected_root)
```

### 2. Parameter Fingerprint Verification

Verifies model parameter fingerprints:

```python
parameters = receipt['model']['parameters']
expected_fingerprint = receipt['model']['parameter_fingerprint']

# Recalculate fingerprint
is_valid = CIAFVerifier.verify_parameter_fingerprint(parameters, expected_fingerprint)
```

### 3. Audit Chain Verification

Verifies hash-chained audit events:

```python
audit_records = receipt['audit_chain']

# Verify chain integrity
is_valid = CIAFVerifier.verify_audit_chain(audit_records)
```

## Receipt Properties

### Immutability

- **Content Hash**: Each receipt has a content-based hash
- **Tampering Detection**: Any modification breaks verification
- **Cryptographic Integrity**: Uses SHA-256 throughout

### Completeness

- **Full Provenance**: From raw data to model output
- **Audit Trail**: Complete chain of events
- **Verification Data**: All info needed for independent verification

### Verifiability

- **Standalone**: Receipts contain all verification data
- **Tool Independence**: Can be verified without CIAF
- **Mathematical Soundness**: Based on well-established cryptographic primitives

## Common Verification Failures

### Merkle Root Mismatch

```
❌ Dataset Merkle root: Invalid
   Expected: f7e8d9c0b1a2...
   Calculated: a2b1c0d9e8f7...
```

**Causes:**
- Dataset modified after receipt generation
- Incorrect leaf ordering
- Hash function mismatch

### Audit Chain Break

```
❌ Audit chain: Invalid
   Chain break at record 2: previous_hash mismatch
```

**Causes:**
- Event removed or modified
- Events inserted out of order
- Hash calculation error

### Parameter Fingerprint Mismatch

```
❌ Model parameters: Invalid
   Expected: x7y8z9a0b1c2...
   Calculated: c2b1a0z9y8x7...
```

**Causes:**
- Model parameters changed
- JSON serialization differences
- Hash calculation differences

## Best Practices

### Receipt Storage

1. **Immutable Storage**: Store receipts in write-once systems
2. **Backup Strategy**: Maintain multiple copies
3. **Access Control**: Limit modification permissions
4. **Retention Policy**: Keep receipts for compliance periods

### Verification Schedule

1. **Regular Verification**: Scheduled integrity checks
2. **Compliance Audits**: Pre-audit verification
3. **Incident Response**: Immediate verification after suspicious activity

### Integration Patterns

1. **API Integration**: Verify receipts before serving models
2. **CI/CD Pipeline**: Include verification in deployment
3. **Monitoring**: Alert on verification failures

## Compliance Integration

Receipts support various compliance frameworks:

### EU AI Act
- **Risk Documentation**: Receipts provide risk assessment audit trail
- **Quality Management**: Training receipts support QMS requirements
- **Record Keeping**: Immutable records for regulatory review

### NIST AI RMF
- **System Inventory**: Model and dataset tracking
- **Risk Assessment**: Complete provenance for risk evaluation
- **Monitoring**: Continuous verification capabilities

### GDPR/HIPAA
- **Data Lineage**: Complete data processing history
- **Consent Tracking**: Audit trail of consent decisions
- **Breach Detection**: Tampering detection capabilities