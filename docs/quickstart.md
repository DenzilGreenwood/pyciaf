# CIAF Quickstart Guide

Get up and running with CIAF in 5 minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/DenzilGreenwood/pyciaf.git
cd pyciaf

# Install in development mode
pip install -e .

# Or install from PyPI (when available)
pip install pyciaf
```

## Basic Usage

### 1. Initialize Framework

```python
from ciaf import CIAFFramework

# Create a new CIAF project
framework = CIAFFramework("my_ai_project")
```

### 2. Create Dataset Anchor

```python
# Create cryptographic anchor for your dataset
anchor = framework.create_dataset_anchor(
    dataset_id="customer_reviews",
    dataset_metadata={
        "source": "e-commerce_platform", 
        "type": "text_reviews",
        "size": 10000
    },
    master_password="secure_password_123"
)

print(f"Dataset anchor: {anchor['anchor_id']}")
```

### 3. Create Provenance Capsules

```python
# Your training data
data_items = [
    {"content": "Great product!", "metadata": {"label": "positive", "id": "rev001"}},
    {"content": "Poor quality", "metadata": {"label": "negative", "id": "rev002"}},
    {"content": "Amazing service", "metadata": {"label": "positive", "id": "rev003"}},
]

# Create verifiable capsules
capsules = framework.create_provenance_capsules("customer_reviews", data_items)
print(f"Created {len(capsules)} provenance capsules")
```

### 4. Create Model Anchor

```python
# Define your model
model_anchor = framework.create_model_anchor(
    model_name="sentiment_classifier",
    model_parameters={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    },
    model_architecture={
        "type": "neural_network",
        "layers": [128, 64, 3],
        "activation": "relu"
    },
    authorized_datasets=["customer_reviews"],
    master_password="secure_model_password"
)

print(f"Model anchor created: {model_anchor['model_name']}")
```

### 5. Train with Verifiable Snapshot

```python
# Train your model with full provenance tracking
snapshot = framework.train_model(
    model_name="sentiment_classifier",
    capsules=capsules,
    maa=model_anchor,  # Model Aggregation Anchor
    training_params={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    },
    model_version="v1.0"
)

print(f"Training snapshot: {snapshot['snapshot_id']}")
```

### 6. Validate Integrity

```python
# Verify training integrity
is_valid = framework.validate_training_integrity(snapshot)
print(f"Training integrity: {'✅ Valid' if is_valid else '❌ Invalid'}")
```

## Complete Example

Run the complete quickstart example:

```bash
python examples/quickstart.py
```

Expected output:
```
🚀 CIAF Quickstart Example
==================================================
✅ CIAF Framework initialized

📊 Step 1: Creating Dataset Anchor
-----------------------------------
✅ Dataset anchor created: a1b2c3d4e5f6...

📦 Step 2: Creating Provenance Capsules
-----------------------------------------
✅ Created 2 provenance capsules

🤖 Step 3: Creating Model Anchor
--------------------------------
✅ Model anchor created: sentiment_classifier
   Parameters fingerprint: f7e8d9c0b1a2...

🏋️ Step 4: Training Model
--------------------------
✅ Training snapshot created: snap_a1b2c3d4...

🔐 Step 5: Validating Training Integrity
------------------------------------------
✅ Training integrity validated: True

🎉 SUCCESS: CIAF quickstart completed successfully!
```

## Next Steps

### Explore More Examples

```bash
# Basic LCM features
python ciaf/examples/basic_example.py

# Advanced audit trails
python ciaf/examples/advanced_example.py

# Risk assessment patterns
python ciaf/examples/risk_audit_example.py
```

### Generate Compliance Reports

```bash
# Setup metadata storage
python -m ciaf.cli setup my_project --backend sqlite

# Generate EU AI Act compliance report
python -m ciaf.cli compliance eu_ai_act my_model --format html
```

### Verify Receipts

```bash
# Create sample receipt
python tools/verify_receipt.py --create-sample > sample_receipt.json

# Verify the receipt
python tools/verify_receipt.py sample_receipt.json
```

## Common Patterns

### Working with Large Datasets

```python
# Use lazy managers for efficient memory usage
lazy_manager = framework.lazy_managers["large_dataset"]

# Materialize only what you need
for item_id in ["item_001", "item_042", "item_999"]:
    capsule = lazy_manager.materialize_capsule(item_id)
    # Process capsule...
```

### Healthcare/HIPAA Patterns

```python
from ciaf import ModelMetadataManager
from ciaf.compliance import ComplianceFramework

# Enable PHI protection
manager = ModelMetadataManager("healthcare_ai", "1.0.0")
manager.enable_phi_protection()
manager.set_compliance_frameworks([ComplianceFramework.HIPAA])
```

### Model Versioning

```python
# Version 1
model_v1 = framework.create_model_anchor(
    model_name="classifier",
    model_parameters={"lr": 0.01},
    # ... other params
    model_version="v1.0"
)

# Version 2 with different parameters
model_v2 = framework.create_model_anchor(
    model_name="classifier",
    model_parameters={"lr": 0.001},  # Changed learning rate
    # ... other params
    model_version="v2.0"
)

# Fingerprints will be different, enabling change detection
```

## Troubleshooting

### Import Errors

If you get import errors:

```bash
# Make sure you're in the right directory
cd pyciaf

# Install in development mode
pip install -e .

# Check installation
python -c "import ciaf; print('CIAF imported successfully')"
```

### Memory Issues with Large Datasets

Use lazy materialization:

```python
# Instead of loading everything
# capsules = framework.create_provenance_capsules(dataset_id, all_items)

# Use lazy pattern
lazy_manager = framework.get_lazy_manager(dataset_id)
for item_id in item_ids:
    capsule = lazy_manager.materialize_capsule(item_id)
    # Process one at a time
```

### Compliance Questions

Check the compliance mapping documentation:

```bash
# View compliance status
python -m ciaf.cli compliance --list-frameworks

# Generate detailed report
python -m ciaf.cli compliance eu_ai_act my_model --verbose
```

## Getting Help

- **Documentation**: [Full Documentation](index.md)
- **Examples**: Check the `examples/` directory
- **Issues**: [GitHub Issues](https://github.com/DenzilGreenwood/pyciaf/issues)
- **Security**: [Security Policy](../ciaf/SECURITY.md)