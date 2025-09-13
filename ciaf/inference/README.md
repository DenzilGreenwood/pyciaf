# CIAF Inference Management

The inference package provides comprehensive capabilities for managing AI model inference with cryptographic receipts, audit chains, and privacy-preserving evidence collection.

## Overview

The inference system creates a complete audit trail for AI model predictions while maintaining privacy and enabling regulatory compliance:

- **Inference Receipts** — Cryptographic receipts for every model prediction
- **Zero-Knowledge Evidence (ZKE) Chains** — Privacy-preserving audit chains
- **Audit Integration** — Seamless integration with CIAF's audit framework
- **Privacy Protection** — Commitment schemes for sensitive input/output data
- **Chain-of-Custody** — Immutable chain linking training to inference
- **Batch Processing** — Efficient handling of high-volume inference workloads

## Components

### InferenceReceipt (`receipts.py`)

Cryptographic receipts that provide tamper-evident records of AI model inferences.

**Key Features:**
- **Cryptographic Integrity** — SHA-256 hashing for tamper detection
- **Complete Metadata** — Model, input, output, timestamp, user tracking
- **Privacy Commitments** — Hash commitments for sensitive data protection
- **Chain Linking** — Links to previous receipts for audit continuity
- **Compliance Ready** — Designed for regulatory audit requirements

**Usage Example:**
```python
from ciaf.inference import InferenceReceipt
from datetime import datetime

# Create inference receipt
receipt = InferenceReceipt(
    receipt_id="inf_20250912_001",
    model_id="pneumonia_classifier_v1",
    model_version="1.0.0",
    query="Is there pneumonia in this chest X-ray?",
    ai_output="No pneumonia detected (confidence: 95%)",
    confidence_score=0.95,
    timestamp=datetime.now().isoformat(),
    user_id="physician_alice",
    session_id="clinical_session_123"
)

# Access receipt properties
print(f"Receipt ID: {receipt.receipt_id}")
print(f"Receipt hash: {receipt.receipt_hash}")
print(f"Model used: {receipt.model_id}")
print(f"Confidence: {receipt.confidence_score}")

# Verify receipt integrity
is_valid = receipt.verify_integrity()
print(f"Receipt integrity: {is_valid}")

# Export receipt for audit
receipt_data = receipt.to_dict()
```

**Receipt Structure:**
```json
{
  "receipt_id": "inf_20250912_001",
  "model_id": "pneumonia_classifier_v1",
  "model_version": "1.0.0",
  "query": "Is there pneumonia in this chest X-ray?",
  "ai_output": "No pneumonia detected (confidence: 95%)",
  "confidence_score": 0.95,
  "timestamp": "2025-09-12T14:30:00.000Z",
  "user_id": "physician_alice",
  "session_id": "clinical_session_123",
  "receipt_hash": "sha256:abc123...",
  "metadata": {
    "model_anchor_ref": "model_anchor_hash",
    "training_snapshot_ref": "training_snapshot_hash",
    "deployment_info": "production_v1"
  }
}
```

### ZKEChain (`receipts.py`)

Zero-Knowledge Evidence Chain for privacy-preserving audit trails.

**Key Features:**
- **Privacy-Preserving** — Maintains audit trail without exposing sensitive data
- **Chain Integrity** — Cryptographic linking between inference receipts
- **Zero-Knowledge Proofs** — Prove properties without revealing data
- **Efficient Verification** — Fast verification of chain integrity
- **Regulatory Compliance** — Supports GDPR, HIPAA privacy requirements

**Usage Example:**
```python
from ciaf.inference import ZKEChain

# Create ZKE chain for a model
zke_chain = ZKEChain(
    model_id="pneumonia_classifier_v1",
    chain_id="medical_inferences_2025"
)

# Add inference receipts to chain
receipt1 = InferenceReceipt(...)
receipt2 = InferenceReceipt(...)

zke_chain.add_receipt(receipt1)
zke_chain.add_receipt(receipt2)

# Verify chain integrity
is_chain_valid = zke_chain.verify_chain_integrity()
print(f"Chain integrity: {is_chain_valid}")

# Get chain statistics
chain_stats = zke_chain.get_chain_summary()
print(f"Total receipts: {chain_stats['total_receipts']}")
print(f"Chain started: {chain_stats['start_time']}")
print(f"Last receipt: {chain_stats['last_receipt_time']}")

# Generate zero-knowledge proof
proof = zke_chain.generate_zk_proof(
    property="inference_count_greater_than",
    threshold=100,
    time_window="2025-09-01 to 2025-09-12"
)

# Verify proof without exposing sensitive data
proof_valid = zke_chain.verify_zk_proof(proof)
print(f"ZK proof valid: {proof_valid}")
```

**Chain Structure:**
```json
{
  "chain_id": "medical_inferences_2025",
  "model_id": "pneumonia_classifier_v1",
  "receipts": [
    {
      "receipt_hash": "sha256:abc123...",
      "prev_hash": null,
      "chain_index": 0
    },
    {
      "receipt_hash": "sha256:def456...", 
      "prev_hash": "sha256:abc123...",
      "chain_index": 1
    }
  ],
  "chain_root": "merkle_root_hash",
  "created_at": "2025-09-01T00:00:00.000Z",
  "last_updated": "2025-09-12T14:30:00.000Z"
}
```

## Advanced Features

### Privacy-Preserving Receipts

For sensitive healthcare or financial applications:

```python
# Create receipt with privacy commitments
from ciaf.core import sha256_hash, secure_random_bytes

# Generate privacy-preserving commitments
query_salt = secure_random_bytes(16)
output_salt = secure_random_bytes(16)

# Create commitments (hash + salt)
query_commitment = sha256_hash(query.encode() + query_salt)
output_commitment = sha256_hash(ai_output.encode() + output_salt)

# Create privacy-preserving receipt
privacy_receipt = InferenceReceipt(
    receipt_id="private_inf_001",
    model_id="sensitive_model_v1",
    query_commitment=query_commitment,  # Instead of raw query
    output_commitment=output_commitment,  # Instead of raw output
    confidence_score=0.95,
    timestamp=datetime.now().isoformat(),
    user_id="authorized_user",
    privacy_mode=True
)

# Original data can be revealed later with salts for audit
def reveal_query(commitment, query, salt):
    return sha256_hash(query.encode() + salt) == commitment
```

### Batch Inference Processing

For high-volume inference workloads:

```python
# Process batch of inferences
batch_receipts = []
model_id = "production_classifier_v2"

# Process multiple inferences
for i, (query, output, confidence) in enumerate(inference_batch):
    receipt = InferenceReceipt(
        receipt_id=f"batch_001_{i:04d}",
        model_id=model_id,
        query=query,
        ai_output=output,
        confidence_score=confidence,
        timestamp=datetime.now().isoformat(),
        user_id="batch_processor",
        batch_id="batch_001"
    )
    batch_receipts.append(receipt)

# Add all receipts to chain efficiently
zke_chain.add_batch_receipts(batch_receipts)

# Generate batch summary
batch_summary = zke_chain.generate_batch_summary("batch_001")
print(f"Batch processed: {batch_summary['receipt_count']} inferences")
```

### Audit Trail Integration

Integration with CIAF's audit framework:

```python
from ciaf.compliance import AuditTrailGenerator
from ciaf.inference import InferenceReceipt, ZKEChain

# Create integrated audit system
audit_generator = AuditTrailGenerator("ai_system")
zke_chain = ZKEChain("model_v1", "audit_chain")

# Create inference with full audit integration
def create_audited_inference(query, ai_output, user_id):
    # Create inference receipt
    receipt = InferenceReceipt(
        receipt_id=f"inf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        model_id="audited_model_v1",
        query=query,
        ai_output=ai_output,
        user_id=user_id,
        timestamp=datetime.now().isoformat()
    )
    
    # Add to ZKE chain
    zke_chain.add_receipt(receipt)
    
    # Log to audit trail
    audit_generator.log_inference_event(
        receipt_id=receipt.receipt_id,
        user_id=user_id,
        model_id=receipt.model_id,
        compliance_frameworks=["EU_AI_ACT", "HIPAA"]
    )
    
    return receipt

# Example usage
receipt = create_audited_inference(
    query="Patient symptoms: fever, cough, chest pain",
    ai_output="Recommendation: chest X-ray and blood work",
    user_id="physician_bob"
)
```

## Integration Patterns

### With CIAF Framework

```python
from ciaf.api import CIAFFramework
from ciaf.inference import InferenceReceipt

# Framework automatically manages inference receipts
framework = CIAFFramework("MyMedicalAI")

# Train model with framework
training_snapshot = framework.train_model(...)

# Perform inference with automatic receipt generation
inference_receipt = framework.perform_inference_with_audit(
    model_name="diagnostic_model",
    query="Patient chest X-ray analysis",
    ai_output="No abnormalities detected",
    training_snapshot=training_snapshot,
    user_id="radiologist_alice"
)

# Access inference chain
zke_chain = framework.inference_chains["diagnostic_model"]
chain_summary = zke_chain.get_chain_summary()
```

### With LCM System

```python
from ciaf.lcm import LCMInferenceManager, LCMInferenceReceipt
from ciaf.inference import ZKEChain

# LCM provides enhanced inference management
lcm_inference = LCMInferenceManager()

# Create LCM inference receipt
lcm_receipt = lcm_inference.create_inference_receipt(
    receipt_id="lcm_inf_001",
    model_anchor_ref="model_anchor_hash",
    deployment_anchor_ref="deployment_anchor_hash",
    query="Medical query",
    ai_output="AI response"
)

# Convert to standard inference receipt for chain
standard_receipt = InferenceReceipt.from_lcm_receipt(lcm_receipt)
zke_chain.add_receipt(standard_receipt)
```

### With Compliance Engine

```python
from ciaf.compliance import ComplianceValidator, TransparencyReportGenerator
from ciaf.inference import ZKEChain

# Validate inference compliance
validator = ComplianceValidator("inference_system")
zke_chain = ZKEChain("model_v1", "compliance_chain")

# Generate compliance report from inference chain
compliance_report = validator.validate_inference_compliance(
    chain=zke_chain,
    frameworks=["EU_AI_ACT", "HIPAA"],
    time_period="2025-09-01 to 2025-09-12"
)

# Generate transparency report
transparency_generator = TransparencyReportGenerator("model_v1")
transparency_report = transparency_generator.generate_inference_transparency_report(
    zke_chain=zke_chain,
    include_privacy_analysis=True
)
```

## Performance Considerations

### Memory Management

```python
# Efficient memory usage for large chains
class EfficientZKEChain(ZKEChain):
    def __init__(self, model_id, chain_id, max_memory_receipts=1000):
        super().__init__(model_id, chain_id)
        self.max_memory_receipts = max_memory_receipts
        self.receipt_cache = {}
    
    def add_receipt(self, receipt):
        # Add to chain
        super().add_receipt(receipt)
        
        # Manage memory usage
        if len(self.receipt_cache) > self.max_memory_receipts:
            # Archive older receipts to disk
            self._archive_old_receipts()
    
    def _archive_old_receipts(self):
        # Move old receipts to persistent storage
        pass
```

### Batch Optimization

```python
# Optimize for high-throughput inference
def optimize_batch_processing(model_id, inference_batch):
    # Create chain with optimized settings
    zke_chain = ZKEChain(
        model_id=model_id,
        chain_id=f"batch_{datetime.now().strftime('%Y%m%d')}",
        batch_size=1000,  # Process in batches of 1000
        compression=True   # Enable receipt compression
    )
    
    # Process in chunks for memory efficiency
    for chunk in chunks(inference_batch, 1000):
        receipts = [create_receipt(inf) for inf in chunk]
        zke_chain.add_batch_receipts(receipts)
        
        # Periodic integrity checks
        if zke_chain.receipt_count % 10000 == 0:
            assert zke_chain.verify_chain_integrity()
    
    return zke_chain
```

## Security Features

### Cryptographic Integrity

```python
# Verify receipt hasn't been tampered with
def verify_receipt_integrity(receipt):
    # Recompute hash from receipt data
    computed_hash = receipt._compute_receipt_hash()
    stored_hash = receipt.receipt_hash
    
    if computed_hash != stored_hash:
        raise SecurityError("Receipt integrity violation detected")
    
    return True

# Verify entire chain integrity
def verify_full_chain_integrity(zke_chain):
    for i, receipt in enumerate(zke_chain.receipts):
        # Verify individual receipt
        verify_receipt_integrity(receipt)
        
        # Verify chain linkage
        if i > 0:
            prev_receipt = zke_chain.receipts[i-1]
            if receipt.prev_hash != prev_receipt.receipt_hash:
                raise SecurityError(f"Chain break at receipt {i}")
    
    return True
```

### Access Control

```python
# Implement role-based access to inference receipts
class SecureInferenceReceipt(InferenceReceipt):
    def __init__(self, *args, access_level="public", **kwargs):
        super().__init__(*args, **kwargs)
        self.access_level = access_level
    
    def can_access(self, user_role, user_id):
        if self.access_level == "public":
            return True
        elif self.access_level == "user_only":
            return user_id == self.user_id
        elif self.access_level == "admin_only":
            return user_role in ["admin", "auditor"]
        else:
            return False
    
    def get_filtered_data(self, user_role, user_id):
        if not self.can_access(user_role, user_id):
            return self._get_redacted_receipt()
        return self.to_dict()
```

## Best Practices

### 1. Receipt ID Generation

```python
import uuid
from datetime import datetime

# Generate unique, traceable receipt IDs
def generate_receipt_id(model_id, user_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{model_id}_{timestamp}_{user_id}_{unique_id}"
```

### 2. Error Handling

```python
try:
    # Create inference receipt
    receipt = InferenceReceipt(...)
    
    # Add to chain
    zke_chain.add_receipt(receipt)
    
    # Verify integrity
    assert receipt.verify_integrity()
    assert zke_chain.verify_chain_integrity()
    
except ValueError as e:
    logger.error(f"Invalid receipt data: {e}")
except SecurityError as e:
    logger.critical(f"Security violation: {e}")
    # Trigger security incident response
except Exception as e:
    logger.error(f"Unexpected error: {e}")
```

### 3. Privacy Protection

```python
# For sensitive applications
def create_privacy_preserving_receipt(query, output, user_id):
    # Use commitments instead of raw data
    query_hash = sha256_hash(query.encode())
    output_hash = sha256_hash(output.encode())
    
    return InferenceReceipt(
        receipt_id=generate_receipt_id("private_model", user_id),
        model_id="sensitive_model",
        query_commitment=query_hash,
        output_commitment=output_hash,
        user_id=user_id,
        privacy_mode=True
    )
```

## Contributing

When extending the inference package:

1. **Maintain Cryptographic Integrity** — Ensure all receipts are tamper-evident
2. **Privacy by Design** — Support privacy-preserving options by default
3. **Performance Optimization** — Consider high-volume inference scenarios
4. **Compliance Integration** — Ensure compatibility with regulatory requirements
5. **Comprehensive Testing** — Include security and integrity tests

## Dependencies

The inference package depends on:
- `ciaf.core` — Cryptographic utilities for hashing and integrity
- `datetime` — Timestamp generation for receipts
- `json` — Serialization for receipt data
- `typing` — Type hints for better code clarity

---

*For integration examples and advanced patterns, see the [examples folder](../examples/) and [API documentation](../api/).*