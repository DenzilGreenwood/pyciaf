# CIAF – Cognitive Insight Audit Framework

**Version:** 1.0.0

A Python framework for verifiable AI training and inference with cryptographic provenance, selective ("lazy") capsule materialization, and compliance-ready audit receipts.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Code Style: Black](https://img.shields.io/badge/code%20style-Black-000000.svg)
[![Security Policy](https://img.shields.io/badge/Security-Policy-informational.svg)](ciaf/SECURITY.md)

---

## Overview

CIAF (Cognitive Insight Audit Framework) addresses AI transparency, auditability, and compliance in production. It provides cryptographically verifiable provenance tracking, **Lazy Capsule Materialization (LCM)**, and audit artifacts designed to map to major regulatory frameworks.

### Key Features

- **Cryptographic Provenance Tracking** — End-to-end verifiable data lineage with Merkle trees and hash chains.  
- **Lazy Capsule Materialization (LCM)** — On-demand proof capsule materialization to minimize storage and exposure.  
- **Compliance Mapping** — Artifacts designed to map to EU AI Act, NIST AI RMF, GDPR/HIPAA, SOX, ISO/IEC 27001 (see `docs/compliance/`).  
- **Security-First Design** — Optional AES-256-GCM, secure anchor derivation, tamper-evident audit trails.  
- **Risk Assessment Patterns** — Bias/fairness checks and uncertainty-quantification scaffolding.  
- **Transparency & Explainability** — Hooks for decision transparency and receipt generation.  
- **Healthcare Patterns** — PHI minimization and consent-tracking patterns (final compliance depends on deployment).  
- **Performance Monitoring** — Basic metrics for LCM operations.

---

## Installation

### Option A: From source
```bash
git clone https://github.com/DenzilGreenwood/pyciaf.git
cd pyciaf
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -U pip build
pip install -e .
```

### Option B: Directly from GitHub
```bash
pip install "git+https://github.com/DenzilGreenwood/pyciaf.git#egg=ciaf"
```

### Option C: PyPI (when published)
```bash
pip install pyciaf
```

---

## Quick Start

```python
from ciaf import CIAFFramework, ModelMetadataManager

framework = CIAFFramework("MyAI_Project")

# 1) Create a dataset anchor (cryptographic root for dataset operations)
anchor = framework.create_dataset_anchor(
    dataset_id="healthcare_data",
    dataset_metadata={"source": "hospital_system", "type": "medical_records"},
    master_password="secure_password_123"
)

# 2) Create provenance capsules for your data
data_items = [
    {"content": "patient_record_1", "metadata": {"id": "p001", "consent": True}},
    {"content": "patient_record_2", "metadata": {"id": "p002", "consent": True}},
]
capsules = framework.create_provenance_capsules("healthcare_data", data_items)

# 3) Create a model anchor (immutable parameter/architecture fingerprints + dataset authorization)
model_anchor = framework.create_model_anchor(
    model_name="diagnostic_model",
    model_parameters={"epochs": 100, "lr": 0.001},
    model_architecture={"type": "bert_classifier", "hidden": 768},
    authorized_datasets=["healthcare_data"],
    master_password="secure_model_password"
)

# 4) Produce a verifiable training snapshot
snapshot = framework.train_model(
    model_name="diagnostic_model",
    capsules=capsules,
    maa=model_anchor,
    training_params={"epochs": 100, "lr": 0.001},
    model_version="v1.0"
)

# 5) Validate integrity
assert framework.validate_training_integrity(snapshot)
print("Training integrity verified.")
```

---

## Architecture

```scss
CIAF Framework
├─ Core Components
│  ├─ Cryptographic Utilities (AES-256-GCM, SHA-256, HMAC)
│  ├─ Anchor Management (hierarchical anchor derivation)
│  └─ Merkle Tree Implementation
├─ Anchoring System
│  ├─ Dataset Anchors (Master → Dataset → Capsule)
│  └─ Lazy Managers (selective materialization)
├─ Provenance Tracking
│  ├─ Provenance Capsules (content + metadata)
│  └─ Training Snapshots (verifiable model states)
├─ Compliance Engine
│  ├─ Regulatory Mapping (EU AI Act, NIST, GDPR/HIPAA, etc.)
│  ├─ Validators (automated checks, where implemented)
│  └─ Audit Trails (append-only/WORM)
├─ Risk Assessment
│  ├─ Bias & Fairness patterns
│  ├─ Uncertainty-quantification scaffolding
│  └─ Security-assessment hooks
├─ Metadata Management
│  ├─ Storage backends (JSON, SQLite, Pickle)
│  ├─ Configuration templates
│  └─ Integration utilities
└─ Utilities
   ├─ CLI Tools
   ├─ Model Wrappers
   └─ ML Framework Simulators
```

---

## Compliance Support

**Compliance Mapping:** CIAF's audit artifacts are designed to map to control intents across EU AI Act, NIST AI RMF, GDPR/HIPAA, SOX, ISO/IEC 27001. Coverage varies by control and typically requires organizational process overlays. See `docs/compliance/` for current status and gaps. **This is not legal advice.**

---

## Advanced Features

### Lazy Capsule Materialization (LCM)

Materialize only what you need, when you need it—while preserving cryptographic verifiability.

```python
# Create dataset anchor with a lazy manager
anchor = framework.create_dataset_anchor(
    dataset_id="large_dataset",
    dataset_metadata={"size": "1TB", "type": "image_data"},
    master_password="secure_anchor_password"
)

# Access the dataset's lazy manager
lazy_manager = framework.lazy_managers["large_dataset"]

# Materialize a capsule on demand
capsule = lazy_manager.materialize_capsule("item_001")
```

### Enhanced Model Anchor System

Immutable parameter/architecture fingerprints and dataset authorization.

```python
model_anchor = framework.create_model_anchor(
    model_name="sentiment_classifier",
    model_parameters={"learning_rate": 2e-5, "batch_size": 16, "num_epochs": 3, "model_type": "bert_classifier"},
    model_architecture={"base_model": "bert-base-uncased", "num_labels": 3, "hidden_size": 768},
    authorized_datasets=["training_data_v1", "validation_data_v1"],
    master_password="secure_model_password"
)
print("Model fingerprint:", model_anchor["parameters_fingerprint"])
print("Architecture fingerprint:", model_anchor["architecture_fingerprint"])
```

### Complete Audit Flow Integration

```python
# 1) Train with complete audit
training_snapshot = framework.train_model_with_audit(
    model_name="sentiment_classifier",
    capsules=training_capsules,
    training_params=training_params,
    model_version="1.0.0",
    user_id="data_scientist_alice"
)

# 2) Perform inference with audit chain
receipt = framework.perform_inference_with_audit(
    model_name="sentiment_classifier",
    query="This product is amazing!",
    ai_output="positive (confidence: 0.95)",
    training_snapshot=training_snapshot,
    user_id="api_user"
)

# 3) Retrieve complete audit trail
audit_trail = framework.get_complete_audit_trail("sentiment_classifier")
print("Datasets:", audit_trail["verification"]["total_datasets"])
print("Audit records:", audit_trail["verification"]["total_audit_records"])
print("Inference receipts:", audit_trail["inference_chain"]["total_receipts"])
```

---

## CLI Tools

```bash
# Setup metadata storage
python -m ciaf.cli setup my_project --backend sqlite --template production

# Generate a compliance report
python -m ciaf.cli compliance eu_ai_act my_model_id --format html --output compliance_report.html
```

---

## Integration Examples

### Scikit-learn

```python
from ciaf import CIAFModelWrapper
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
wrapped = CIAFModelWrapper(model, "fraud_detection_v1")
wrapped.fit(X_train, y_train)
preds = wrapped.predict(X_test)
```

### TensorFlow / PyTorch (simulated)

```python
from ciaf.simulation import MLFrameworkSimulator

sim = MLFrameworkSimulator("neural_network")
training_snapshot = sim.train_model(
    training_data_capsules=capsules,
    maa=model_anchor,
    training_params={"epochs": 50, "batch_size": 32},
    model_version="v2.0"
)
```

---

## Performance & Metrics

```python
metrics = framework.get_performance_metrics("my_dataset")
print("Materialization rate:", f"{metrics['materialization_rate']:.2%}")
print("Total items:", metrics["total_items"])
print("Materialized capsules:", metrics["materialized_capsules"])
```

---

## Security

> **See our [Security Policy](ciaf/SECURITY.md) for reporting vulnerabilities, supported versions, and secure deployment guidance.**

### Cryptographic Security

- **AES-256-GCM** (optional) for authenticated encryption (supports AAD).
- **SHA-256** for integrity hashing.
- **HMAC-SHA-256** for anchor derivation and message authentication.
- **Merkle Trees** with canonical concatenation for tamper-evident sets.

### Anchor Management

- Hierarchical anchor derivation (Master → Dataset → Capsule).
- Cryptographically secure randomness and high-entropy binary anchors.
- Canonicalized operations for derivations & Merkle policies.
- Backwards compatibility for legacy key-based terminology.

### Access Controls (patterns)

- Role-based access patterns.
- Audit-logging hooks.
- Session-management scaffolding.

---

## Healthcare & HIPAA Patterns

```python
from ciaf import ModelMetadataManager
from ciaf.compliance import ComplianceFramework

manager = ModelMetadataManager("healthcare_ai", "1.0.0")
manager.enable_phi_protection()
manager.set_compliance_frameworks([ComplianceFramework.HIPAA])
manager.capture_metadata({
    "patient_id": "XXXXX",  # handled with PHI patterns
    "diagnosis": "diabetes",
    "consent_status": "active"
})
```

> **Note:** CIAF provides patterns for PHI minimization and consent tracking. Final compliance depends on your deployment architecture, governance, and policies.

---

## Contributing

We welcome contributions!

1. **Code Style** — Black
2. **Testing** — Add tests; ensure all pass
3. **Docs** — Update documentation for any API changes
4. **Security** — Follow secure coding practices; report issues via `SECURITY.md`

### Development Setup

```bash
git clone https://github.com/DenzilGreenwood/pyciaf.git
cd pyciaf
pip install -e .
```

---

## Support & Community

- **Documentation**: [https://ciaf.readthedocs.io](https://ciaf.readthedocs.io)
- **Issues**: [https://github.com/DenzilGreenwood/pyciaf/issues](https://github.com/DenzilGreenwood/pyciaf/issues)
- **Discussions**: [https://github.com/DenzilGreenwood/pyciaf/discussions](https://github.com/DenzilGreenwood/pyciaf/discussions)
- **Security**: See [ciaf/SECURITY.md](ciaf/SECURITY.md)

---

## Status & Roadmap

### Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| **Core Framework** | ✅ Working | Anchoring + LCM |
| **Cryptographic Primitives** | ✅ Working | SHA-256, HMAC, AES-GCM |
| **Merkle Trees** | ✅ Working | Deterministic proofs |
| **Dataset Anchoring** | ✅ Working | Hierarchical derivation |
| **Model Anchoring** | ✅ Working | Param/arch fingerprints |
| **Audit Trails** | ✅ Working | Hash-chained events |
| **Lazy Materialization** | ✅ Working | On-demand capsules |
| **Basic CLI** | 🧪 Prototype | Setup & compliance |
| **Compliance Mapping** | 🧪 Prototype | EU AI Act, NIST |
| **Receipt Verification** | ✅ Working | Independent verifier |
| **Healthcare Patterns** | 🧪 Prototype | PHI scaffolding |

### Near-term Roadmap (Q4 2024)

| Feature | Priority | Target |
|---------|----------|--------|
| **API Stabilization** | 🔴 High | Finalize public APIs |
| **Documentation** | 🔴 High | Complete API reference |
| **Test Coverage** | 🔴 High | >90% |
| **Performance Optimization** | 🟡 Medium | LCM efficiency |
| **CLI Enhancement** | 🟡 Medium | Full-featured CLI |

### Medium-term Roadmap (2025)

| Feature | Priority | Target |
|---------|----------|--------|
| **GDPR/HIPAA Compliance** | 🔴 High | Production-ready patterns |
| **Advanced Analytics** | 🟡 Medium | Bias/fairness metrics |
| **Integration Libraries** | 🟡 Medium | TF/PyTorch wrappers |
| **Web Dashboard** | 🟢 Low | Audit visualization |
| **Enterprise Features** | 📋 Planned | SSO, RBAC, deployment |

### Research Areas

- **Zero-Knowledge Proofs** (ZK-SNARKs) for privacy-preserving verification
- **Distributed ledger** anchoring for audit immutability
- **Homomorphic encryption** for computation on encrypted data
- **Formal verification** of cryptographic correctness

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **cryptography** library and the broader Python security ecosystem
- **Regulatory frameworks**: EU AI Act, NIST AI RMF, GDPR/HIPAA, ISO/IEC 27001, SOX (for mapping inspiration)

> **Personal note:** This project is a work in progress and reflects a commitment to secure, verifiable, and compliant AI systems. Feedback and contributions are highly appreciated!
> 
> *— Denzil James Greenwood*