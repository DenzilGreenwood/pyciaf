# CIAF API Package

High-level API components providing a unified interface for the Cognitive Insight Audit Framework.

## Overview

The API package serves as the primary entry point for users of CIAF, providing a simplified, high-level interface that orchestrates the underlying cryptographic, anchoring, and compliance components. It abstracts away complexity while maintaining full audit trail capabilities.

## Components

### CIAFFramework (`framework.py`)

The main framework class that provides the primary API for all CIAF operations. It implements the complete audit flow from dataset anchoring through model training to inference receipts.

**Key Features:**
- **Dataset Anchor Management** — Create and manage cryptographic dataset anchors
- **Model Anchor Creation** — Immutable model parameter and architecture fingerprinting
- **Training Orchestration** — Complete training session management with audit trails
- **Inference Management** — Receipt generation and zero-knowledge chain management
- **Lazy Materialization** — On-demand capsule materialization for large datasets
- **Compliance Integration** — Built-in audit trail generation for regulatory requirements

## Usage Examples

### Basic Framework Setup

```python
from ciaf.api import CIAFFramework

# Initialize the framework
framework = CIAFFramework("MyAI_Project")
```

### Dataset Anchor Creation

```python
# Create a dataset anchor for your data
anchor = framework.create_dataset_anchor(
    dataset_id="healthcare_data",
    dataset_metadata={
        "source": "hospital_system",
        "type": "medical_records",
        "compliance": ["HIPAA", "GDPR"]
    },
    master_password="secure_password_123"
)

# The framework automatically creates a lazy manager
lazy_manager = framework.lazy_managers["healthcare_data"]
```

### Model Anchor Creation

```python
# Create immutable model anchor with parameter fingerprinting
model_anchor = framework.create_model_anchor(
    model_name="diagnostic_model",
    model_parameters={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "adam"
    },
    model_architecture={
        "type": "transformer",
        "layers": 12,
        "hidden_size": 768,
        "attention_heads": 12
    },
    authorized_datasets=["healthcare_data"],
    master_password="secure_model_password"
)

print(f"Model fingerprint: {model_anchor['parameters_fingerprint']}")
print(f"Architecture fingerprint: {model_anchor['architecture_fingerprint']}")
```

### Provenance Capsule Creation

```python
# Create provenance capsules for your training data
data_items = [
    {"content": "patient_record_1", "metadata": {"id": "p001", "consent": True}},
    {"content": "patient_record_2", "metadata": {"id": "p002", "consent": True}},
    {"content": "patient_record_3", "metadata": {"id": "p003", "consent": True}}
]

capsules = framework.create_provenance_capsules("healthcare_data", data_items)
print(f"Created {len(capsules)} provenance capsules")
```

### Model Training with Audit Trail

```python
# Train model with complete audit trail
training_snapshot = framework.train_model(
    model_name="diagnostic_model",
    capsules=capsules,
    maa=model_anchor,  # Model Aggregation Anchor
    training_params={
        "epochs": 100,
        "learning_rate": 0.001,
        "validation_split": 0.2
    },
    model_version="v1.0"
)

# Validate training integrity
is_valid = framework.validate_training_integrity(training_snapshot)
print(f"Training integrity verified: {is_valid}")
```

### Enhanced Training with Audit

```python
# Complete training with user tracking
training_snapshot = framework.train_model_with_audit(
    model_name="diagnostic_model",
    capsules=capsules,
    training_params={
        "epochs": 100,
        "learning_rate": 0.001,
        "early_stopping": True
    },
    model_version="1.0.0",
    user_id="data_scientist_alice"
)
```

### Inference with Audit Chain

```python
# Perform inference with complete audit chain
inference_receipt = framework.perform_inference_with_audit(
    model_name="diagnostic_model",
    query="Patient presents with chest pain and shortness of breath",
    ai_output="Recommend chest X-ray and ECG (confidence: 0.87)",
    training_snapshot=training_snapshot,
    user_id="physician_bob"
)

print(f"Inference receipt ID: {inference_receipt.receipt_id}")
```

### Model Aggregation Anchor (MAA)

```python
# Create Model Aggregation Anchor for dataset authorization
maa = framework.create_model_aggregation_anchor(
    model_name="diagnostic_model",
    authorized_datasets=["healthcare_data", "validation_data"]
)

# Use MAA in training to enforce dataset authorization
training_snapshot = framework.train_model(
    model_name="diagnostic_model",
    capsules=capsules,
    maa=maa,
    training_params={"epochs": 50},
    model_version="v1.0"
)
```

### Complete Audit Trail Retrieval

```python
# Get comprehensive audit trail for a model
audit_trail = framework.get_complete_audit_trail("diagnostic_model")

print("Audit Trail Summary:")
print(f"- Total datasets: {audit_trail['verification']['total_datasets']}")
print(f"- Model anchor with fingerprints: 1")
print(f"- Audit records: {audit_trail['verification']['total_audit_records']}")
print(f"- Inference receipts: {audit_trail['inference_chain']['total_receipts']}")
```

### Performance Metrics

```python
# Monitor lazy materialization performance
metrics = framework.get_performance_metrics("healthcare_data")
print(f"Materialization rate: {metrics['materialization_rate']:.2%}")
print(f"Total items: {metrics['total_items']}")
print(f"Materialized capsules: {metrics['materialized_capsules']}")
```

## Advanced Features

### ML Framework Simulation

The framework includes built-in simulation capabilities for testing and development:

```python
# Access ML framework simulator
simulator = framework.ml_simulators.get("diagnostic_model")
if simulator:
    # Run simulated training
    sim_snapshot = simulator.train_model(
        training_data_capsules=capsules,
        maa=model_anchor,
        training_params={"epochs": 10, "batch_size": 16},
        model_version="sim_v1.0"
    )
```

### Zero-Knowledge Evidence Chains

The framework automatically manages zero-knowledge evidence chains for privacy-preserving audit trails:

```python
# Access inference chain for a model
zke_chain = framework.inference_chains.get("diagnostic_model")
if zke_chain:
    # Get chain summary
    chain_summary = zke_chain.get_chain_summary()
    print(f"Chain length: {chain_summary['total_receipts']}")
```

### Audit Trail Generation

Automatic audit trail generation for compliance:

```python
# Access audit generator for a model
audit_generator = framework.audit_generators.get("diagnostic_model")
if audit_generator:
    # Generate compliance report
    compliance_report = audit_generator.generate_compliance_report(
        framework_name="EU AI Act",
        validation_period_days=30
    )
```

## Integration with Other CIAF Components

The API framework seamlessly integrates with all CIAF subsystems:

### Anchoring System
- Automatically creates and manages `DatasetAnchor` instances
- Provides `LazyManager` for efficient capsule materialization
- Handles hierarchical anchor derivation

### Core Cryptography
- Uses `CryptoUtils` for all cryptographic operations
- Implements secure anchor derivation with `BaseAnchorManager`
- Provides Merkle tree operations for provenance

### Provenance Tracking
- Creates `ProvenanceCapsule` instances for data lineage
- Generates `TrainingSnapshot` for model state capture
- Manages `ModelAggregationAnchor` for dataset authorization

### Inference Management
- Generates `InferenceReceipt` for each prediction
- Maintains `ZKEChain` for privacy-preserving audit trails
- Supports batch inference processing

### Compliance Engine
- Integrates with `AuditTrailGenerator` for regulatory reporting
- Supports multiple compliance frameworks (EU AI Act, NIST, GDPR/HIPAA)
- Provides evidence collection for audit purposes

## Error Handling and Validation

The framework includes comprehensive error handling:

```python
try:
    # Framework operations
    anchor = framework.create_dataset_anchor(...)
    training_snapshot = framework.train_model(...)
    
    # Validate integrity
    assert framework.validate_training_integrity(training_snapshot)
    
except Exception as e:
    print(f"Framework error: {e}")
    # Handle errors appropriately
```

## Performance Considerations

### Memory Management
- Lazy materialization minimizes memory usage
- Capsules are created on-demand, not stored in memory
- Automatic cleanup of unused components

### Scalability
- Supports large datasets through lazy evaluation
- Efficient Merkle tree operations (O(log n) proofs)
- Batch processing for inference receipts

### Caching
- Automatic caching of frequently accessed anchors
- Performance metrics tracking for optimization
- Configurable cache policies

## Security Features

### Cryptographic Security
- All anchors use HMAC-SHA256 with secure salts
- AES-256-GCM encryption for sensitive data (optional)
- Secure random number generation

### Access Control
- Dataset authorization through Model Aggregation Anchors
- User tracking for all operations
- Audit logging for security events

### Data Protection
- Privacy-preserving commitments for sensitive data
- Minimal data exposure through lazy materialization
- Compliance with data protection regulations

## Best Practices

### 1. Password Management
```python
# Use strong, unique passwords for each component
dataset_password = "strong_dataset_password_123"
model_password = "strong_model_password_456"
```

### 2. Metadata Organization
```python
# Include comprehensive metadata for audit trails
metadata = {
    "source": "data_source",
    "version": "v1.0",
    "compliance": ["GDPR", "HIPAA"],
    "owner": "data_team",
    "created": "2025-09-12T10:00:00Z"
}
```

### 3. Error Handling
```python
# Always validate operations
try:
    snapshot = framework.train_model(...)
    assert framework.validate_training_integrity(snapshot)
except AssertionError:
    print("Training integrity check failed")
except Exception as e:
    print(f"Training error: {e}")
```

### 4. Performance Monitoring
```python
# Monitor performance for large datasets
metrics = framework.get_performance_metrics(dataset_id)
if metrics['materialization_rate'] < 0.8:
    print("Consider optimizing lazy materialization")
```

## Contributing

When extending the API framework:

1. Maintain backward compatibility with existing methods
2. Add comprehensive error handling and validation
3. Include performance monitoring for new operations
4. Update documentation and examples
5. Ensure all new features support audit trail generation

## Dependencies

The API framework depends on:
- `ciaf.anchoring` — Dataset anchoring and lazy management
- `ciaf.core` — Cryptographic utilities and base classes
- `ciaf.provenance` — Provenance tracking and snapshots
- `ciaf.inference` — Inference receipts and chains
- `ciaf.compliance` — Audit trail generation
- `ciaf.simulation` — ML framework simulation

---

*For detailed implementation examples, see the [examples folder](../examples/) and the main [CIAF documentation](../../README.md).*