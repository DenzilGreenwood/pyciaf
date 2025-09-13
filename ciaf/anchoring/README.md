# CIAF Anchoring System

The anchoring system provides the cryptographic foundation for lazy capsule materialization in CIAF. It implements a hierarchical anchor derivation system that enables on-demand data access while maintaining cryptographic consistency and audit integrity.

## Overview

The anchoring system creates a secure hierarchy for data access:
```
Master Password → Master Anchor → Dataset Anchor → Capsule Anchors (on-demand)
```

This design allows for:
- **Lazy Materialization** — Data items are materialized only when needed
- **Cryptographic Consistency** — All anchors are deterministically derived
- **Audit Integrity** — Complete provenance tracking without storing full data
- **Security** — Hierarchical access control with cryptographic separation

## Components

### DatasetAnchor (`dataset_anchor.py`)

The core anchor class that manages the cryptographic hierarchy for a dataset.

**Key Features:**
- **Hierarchical Anchor Derivation** — Master → Dataset → Capsule anchor chain
- **Backwards Compatibility** — Supports legacy "key" terminology while using modern "anchor" terminology
- **Lazy Capsule Management** — Creates capsule anchors on-demand
- **Metadata Integration** — Comprehensive dataset metadata management
- **Audit Trail Support** — Built-in provenance tracking

**Usage Example:**
```python
from ciaf.anchoring import DatasetAnchor

# Create dataset anchor
anchor = DatasetAnchor(
    dataset_id="medical_dataset",
    metadata={
        "source": "hospital_system",
        "type": "radiology_images",
        "compliance": ["HIPAA"],
        "version": "v1.0"
    },
    master_password="secure_master_password",
    salt=b"optional_salt_bytes"
)

# Access anchor properties
print(f"Dataset ID: {anchor.dataset_id}")
print(f"Master anchor: {anchor.master_anchor_hex}")
print(f"Dataset anchor: {anchor.dataset_anchor_hex}")

# Add data items (creates capsule anchors on-demand)
data_items = [
    {"content": "patient_001.dcm", "metadata": {"patient_id": "P001"}},
    {"content": "patient_002.dcm", "metadata": {"patient_id": "P002"}}
]
anchor.add_data_items(data_items)

# Get capsule anchor for specific item
capsule_anchor = anchor.get_capsule_anchor("item_0")
print(f"Capsule anchor: {capsule_anchor}")
```

### LazyManager (`simple_lazy_manager.py`)

A simple lazy manager that provides basic on-demand capsule materialization.

**Key Features:**
- **Simple Interface** — Easy-to-use lazy materialization
- **Memory Efficient** — Stores only metadata, not full content
- **On-Demand Access** — Materializes capsules when requested
- **Audit Tracking** — Tracks access patterns for compliance

**Usage Example:**
```python
from ciaf.anchoring import LazyManager

# Create lazy manager with dataset anchor
lazy_manager = LazyManager(dataset_anchor)

# Access statistics
print(f"Total items: {lazy_manager.total_items}")
print(f"Materialized: {lazy_manager.materialized_count}")

# Materialize specific capsule
capsule = lazy_manager.materialize_capsule("item_0")
print(f"Materialized capsule: {capsule}")

# Get performance metrics
metrics = lazy_manager.get_metrics()
print(f"Materialization rate: {metrics['materialization_rate']:.2%}")
```

### LazyProvenanceManager (`lazy_manager.py`)

Advanced lazy manager with comprehensive provenance tracking and audit capabilities.

**Key Features:**
- **Provenance Tracking** — Complete audit trail for all operations
- **Advanced Metrics** — Detailed performance and access analytics
- **Compliance Integration** — Built-in support for regulatory requirements
- **Batch Operations** — Efficient bulk materialization
- **Access Control** — User-based access tracking

**Usage Example:**
```python
from ciaf.anchoring import LazyProvenanceManager

# Create advanced lazy manager
provenance_manager = LazyProvenanceManager(
    dataset_anchor=dataset_anchor,
    enable_audit=True,
    compliance_frameworks=["HIPAA", "GDPR"]
)

# Materialize with user tracking
capsule = provenance_manager.materialize_capsule_with_audit(
    item_id="item_0",
    user_id="researcher_alice",
    purpose="model_training"
)

# Get comprehensive audit trail
audit_trail = provenance_manager.get_audit_trail()
print(f"Access events: {len(audit_trail['access_events'])}")

# Batch materialization
item_ids = ["item_0", "item_1", "item_2"]
capsules = provenance_manager.materialize_batch(
    item_ids=item_ids,
    user_id="researcher_alice"
)
```

### TrueLazyManager (`true_lazy_manager.py`)

Most advanced lazy manager with lazy references and minimal memory footprint.

**Key Features:**
- **Lazy References** — Ultra-lightweight capsule references
- **Minimal Memory** — Stores only essential anchor data
- **JIT Materialization** — Just-in-time capsule creation
- **Reference Counting** — Track capsule usage patterns
- **Garbage Collection** — Automatic cleanup of unused capsules

**Usage Example:**
```python
from ciaf.anchoring import TrueLazyManager, LazyReference

# Create true lazy manager
true_lazy = TrueLazyManager(dataset_anchor)

# Get lazy reference (no materialization yet)
lazy_ref = true_lazy.get_lazy_reference("item_0")
print(f"Reference type: {type(lazy_ref)}")  # LazyReference

# Materialize when actually needed
if lazy_ref.is_valid():
    actual_capsule = lazy_ref.materialize()
    print(f"Materialized: {actual_capsule}")

# Bulk reference creation
refs = true_lazy.create_lazy_references(["item_0", "item_1", "item_2"])

# Selective materialization
materialized = []
for ref in refs:
    if ref.should_materialize():  # Custom logic
        materialized.append(ref.materialize())
```

## Hierarchical Anchor System

### Master Anchor Derivation
```python
# Master anchor derived from password and salt
master_password = "secure_password"
salt = secure_random_bytes(SALT_LENGTH)
master_anchor = derive_master_anchor(master_password, salt)
```

### Dataset Anchor Derivation
```python
# Dataset anchor derived from master anchor and dataset metadata
dataset_metadata = {"dataset_id": "med_data", "version": "v1.0"}
dataset_anchor = derive_dataset_anchor(master_anchor, dataset_metadata)
```

### Capsule Anchor Derivation
```python
# Capsule anchors derived on-demand for each data item
item_metadata = {"item_id": "item_0", "type": "medical_image"}
capsule_anchor = derive_capsule_anchor(dataset_anchor, item_metadata)
```

## Integration Patterns

### With CIAF Framework
```python
from ciaf.api import CIAFFramework
from ciaf.anchoring import DatasetAnchor

# Framework automatically creates and manages anchors
framework = CIAFFramework("MyProject")
anchor = framework.create_dataset_anchor(
    dataset_id="training_data",
    dataset_metadata={"source": "internal"},
    master_password="framework_password"
)

# Access through framework's lazy managers
lazy_manager = framework.lazy_managers["training_data"]
```

### With Provenance System
```python
from ciaf.provenance import ProvenanceCapsule
from ciaf.anchoring import DatasetAnchor

# Create capsules using anchor system
anchor = DatasetAnchor(...)
data_items = [{"content": "data", "metadata": {"id": "001"}}]
anchor.add_data_items(data_items)

# Create provenance capsules
capsules = []
for i in range(len(data_items)):
    capsule_anchor = anchor.get_capsule_anchor(f"item_{i}")
    capsule = ProvenanceCapsule(
        content=data_items[i]["content"],
        metadata=data_items[i]["metadata"],
        anchor=capsule_anchor
    )
    capsules.append(capsule)
```

### With LCM System
```python
from ciaf.lcm import LCMDatasetManager
from ciaf.anchoring import DatasetAnchor

# LCM system uses anchoring for dataset management
lcm_manager = LCMDatasetManager()
dataset_anchor = lcm_manager.create_dataset_anchor(
    dataset_id="lcm_dataset",
    metadata={"name": "training_set"},
    master_password="lcm_password"
)

# Anchor system provides the cryptographic foundation
print(f"LCM dataset anchor: {dataset_anchor.dataset_anchor_hex}")
```

## Performance Characteristics

### Memory Usage
- **Dataset Anchor**: ~1KB per dataset (stores only anchors and metadata)
- **Lazy Manager**: ~100 bytes per item (item metadata only)
- **True Lazy Manager**: ~50 bytes per item (minimal reference data)

### Computational Complexity
- **Anchor Derivation**: O(1) per anchor (HMAC-SHA256)
- **Capsule Materialization**: O(1) per capsule
- **Batch Operations**: O(n) for n items

### Storage Efficiency
- **Traditional Approach**: Stores full data in memory
- **Lazy Approach**: Stores only anchors and metadata (99%+ reduction)
- **True Lazy**: Stores only references (99.9%+ reduction)

## Security Features

### Cryptographic Properties
- **HMAC-SHA256**: Message authentication for anchor derivation
- **Deterministic**: Same inputs always produce same anchors
- **One-Way**: Cannot derive parent anchors from child anchors
- **Collision Resistant**: SHA-256 provides cryptographic security

### Access Control
- **Hierarchical**: Master anchor controls all dataset access
- **Granular**: Individual capsule anchors for fine-grained control
- **Audit Trail**: Complete logging of all access operations
- **User Tracking**: Support for user-based access control

### Data Protection
- **Minimal Exposure**: Only requested data is materialized
- **Lazy Loading**: Data accessed only when explicitly requested
- **Secure Cleanup**: Automatic cleanup of materialized data
- **Privacy Preserving**: Supports privacy-preserving commitments

## Best Practices

### 1. Password Management
```python
# Use strong, unique passwords for each dataset
import secrets
master_password = secrets.token_urlsafe(32)

# Store passwords securely (not in code)
# Consider using environment variables or secure vaults
```

### 2. Salt Management
```python
# Let the system generate secure salts
anchor = DatasetAnchor(
    dataset_id="my_dataset",
    master_password=master_password,
    # salt=None  # System generates secure salt automatically
)

# Or provide custom salt for reproducibility
custom_salt = hashlib.sha256(b"dataset_specific_info").digest()
anchor = DatasetAnchor(..., salt=custom_salt)
```

### 3. Metadata Organization
```python
# Include comprehensive metadata for audit trails
metadata = {
    "dataset_id": "medical_images_v1",
    "source": "hospital_database",
    "version": "1.0.0",
    "owner": "medical_ai_team",
    "created": datetime.now().isoformat(),
    "compliance": ["HIPAA", "GDPR"],
    "description": "Chest X-ray images for pneumonia detection"
}
```

### 4. Lazy Manager Selection
```python
# Choose appropriate lazy manager based on needs:

# Simple use cases
lazy_manager = LazyManager(anchor)

# Compliance and audit requirements
provenance_manager = LazyProvenanceManager(anchor, enable_audit=True)

# Memory-constrained environments
true_lazy = TrueLazyManager(anchor)
```

### 5. Performance Monitoring
```python
# Monitor materialization patterns
metrics = lazy_manager.get_metrics()
if metrics['materialization_rate'] > 0.8:
    print("Consider caching frequently accessed items")

# Track access patterns
access_patterns = provenance_manager.get_access_patterns()
print(f"Most accessed items: {access_patterns['top_items']}")
```

## Error Handling

### Common Error Scenarios
```python
try:
    # Anchor creation
    anchor = DatasetAnchor(
        dataset_id="test_dataset",
        master_password="password",
        metadata={"version": "v1.0"}
    )
    
    # Capsule materialization
    capsule = lazy_manager.materialize_capsule("item_0")
    
except ValueError as e:
    print(f"Invalid parameters: {e}")
except KeyError as e:
    print(f"Item not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Validation
```python
# Validate anchor integrity
if not anchor.validate_anchor_chain():
    raise ValueError("Anchor chain validation failed")

# Validate capsule anchors
for item_id in anchor.get_item_ids():
    capsule_anchor = anchor.get_capsule_anchor(item_id)
    if not capsule_anchor:
        print(f"Invalid capsule anchor for {item_id}")
```

## Contributing

When extending the anchoring system:

1. **Maintain Compatibility**: Support both "anchor" and legacy "key" terminology
2. **Add Tests**: Include comprehensive tests for new functionality
3. **Document Security**: Clearly document cryptographic properties
4. **Performance**: Consider memory and computational efficiency
5. **Audit Support**: Ensure all operations support audit trail generation

## Dependencies

The anchoring system depends on:
- `ciaf.core` — Cryptographic utilities and anchor derivation functions
- `typing` — Type hints for better code clarity
- `datetime` — Timestamp generation for audit trails
- `json` — Metadata serialization

---

*For implementation examples, see the [examples folder](../examples/) and integration with the [API framework](../api/).*