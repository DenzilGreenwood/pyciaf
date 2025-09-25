# CIAF – Cognitive Insight Audit Framework

**Version:** 1.1.0 (Production Ready)

> **🎉 PRODUCTION RELEASE**  
> CIAF v1.1.0 is now production-ready with comprehensive enterprise features! All mock implementations have been replaced with realistic, enterprise-grade functionality. This release includes enhanced compliance, performance optimization, and full test coverage.

A Python framework for verifiable AI training and inference with cryptographic provenance, selective ("lazy") capsule materialization, and compliance mapping.

![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)
![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Code Style: Black](https://img.shields.io/badge/code%20style-Black-000000.svg)
[![Security Policy](https://img.shields.io/badge/Security-Policy-informational.svg)](ciaf/SECURITY.md)

---

## Overview

CIAF (Cognitive Insight Audit Framework) addresses AI transparency, auditability, and compliance in production. It provides cryptographically verifiable provenance tracking, **Lazy Capsule Materialization (LCM)**, and audit artifacts designed to map to major regulatory frameworks.

### Key Features

- **Cryptographic Provenance Tracking** — End-to-end verifiable data lineage with Merkle trees and hash connections.  
- **Lazy Capsule Materialization (LCM)** — On-demand proof capsule materialization to minimize storage and exposure.  
- **Compliance Mapping** — Artifacts designed to map to EU AI Act, NIST AI RMF, GDPR/HIPAA, SOX, ISO/IEC 27001 (see `docs/compliance/`).  
- **Security-First Design** — Optional AES-256-GCM, secure anchor derivation, tamper-evident audit trails.  
- **Risk Assessment Patterns** — Bias/fairness checks and uncertainty-quantification scaffolding.  
- **Transparency & Explainability** — Hooks for decision transparency and receipt generation.  
- **Healthcare Patterns** — PHI minimization and consent-tracking patterns (final compliance depends on deployment).  
- **Performance Monitoring** — Basic metrics for LCM operations.
- **Metadata Traceability** — Complete inference-to-model lineage tracking with single receipt lookup.

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

## Project Structure

CIAF v1.1.0 follows a clean, professional project structure:

```
PYPI/                           # Root project directory
├── ciaf/                       # Main CIAF package
│   ├── core/                   # Core functionality
│   ├── api/                    # High-level API
│   ├── lcm/                    # Lifecycle Management
│   ├── compliance/             # Regulatory compliance
│   ├── wrappers/               # Model wrappers
│   └── ...                     # Additional modules
├── examples/                   # Usage examples and demos
├── tests/                      # Comprehensive test suite
├── docs/                       # Complete documentation
├── tools/                      # Development utilities
└── PROJECT_STRUCTURE.md        # Detailed structure guide
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for complete details.

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
├─ Inference Management
│  ├─ Inference Receipts (verifiable prediction records)
│  ├─ ZKE Connections (privacy-preserving audit connections)
│  └─ Metadata Reveal (complete lineage tracing)
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

# 2) Perform inference with audit connections
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
print("Inference receipts:", audit_trail["inference_connections"]["total_receipts"])
```

---

## Tools & Verification

CIAF includes a comprehensive suite of tools for demonstration, verification, and audit compliance located in the `tools/` directory.

### 🔧 Verification Tools

#### Independent Receipt Verification
```bash
# Verify any CIAF receipt with detailed cryptographic validation
cd tools/
python verify_receipt.py path/to/receipt.json
```

The verification tool provides detailed output including:
- **Dataset Merkle root validation** with expected vs calculated hashes
- **Model parameter fingerprints** with complete parameter display
- **Model architecture verification** with full architecture specs
- **Audit connection integrity** with hash chain validation for each event

#### Enhanced Verification Features
- ✅ **Complete hash transparency** - Shows expected vs calculated values for all cryptographic operations
- ✅ **Parameter visibility** - Displays full model configuration and architecture
- ✅ **Audit chain details** - Individual event validation with hash linking verification
- ✅ **Error diagnostics** - Clear indication of validation failures with specific details
- ✅ **Compliance ready** - Output suitable for regulatory audits and forensic investigation

### 🚀 Demo & Benchmarking Tools

#### Deferred LCM Performance Demo
```bash
cd tools/
python deferred_lcm_benchmark.py
```

This benchmark demonstrates:
- **Performance comparison** between standard CIAF, high-performance deferred LCM, and adaptive LCM
- **Real-world fraud detection** scenario with 1000+ predictions
- **Adaptive mode switching** based on system load and processing requirements
- **Comprehensive metrics** including throughput (samples/sec) and latency analysis

#### Receipt Verification Workflow Demo
```bash
cd tools/
python demo_receipt_verification.py
```

Complete workflow demonstration:
1. **Extracts receipts** from deferred LCM audit batches
2. **Converts to verifiable format** compatible with independent verification
3. **Runs verification** using the enhanced verification tool
4. **Shows detailed results** with full audit trail information

#### Receipt Extraction Tool
```bash
cd tools/
python extract_receipt_for_verification.py
```

Converts deferred LCM audit batches into standalone CIAF receipts for independent verification:
- **Merkle tree construction** from training data samples
- **Model fingerprint generation** for parameters and architecture
- **Audit chain creation** with proper hash linking
- **Deferred LCM metadata** preservation for compliance tracking

### 📊 Demo Features

The tools demonstrate:

#### **Enhanced Model Wrapper** (`enhanced_model_wrapper.py`)
- **Deferred LCM integration** with background audit materialization
- **Adaptive mode switching** between immediate and deferred processing
- **Performance optimization** while maintaining full compliance
- **Receipt generation** with lightweight audit creation

#### **Performance Benchmarking**
Example output from deferred LCM benchmark:
```
Performance Comparison Results:
=====================================
Standard CIAF: 0.0006s avg (1723 samples/sec)
High-performance: 0.0016s avg (625 samples/sec)  
Adaptive LCM: 0.0029s avg (548 samples/sec)

Audit Trail Generation: 50 receipts created
Verification: All receipts independently verified ✅
```

#### **Verification Transparency**
Example verification output:
```
🔍 Verifying CIAF Receipt...
========================================
📊 Dataset Merkle root: ✅ Valid
   📋 Dataset ID: deferred_lcm_demo_dataset
   🌿 Leaf count: 4
   🔍 Expected root: 3d1081642ad6c5e2f327f8f288dafaba...
   🧮 Calculated root: 3d1081642ad6c5e2f327f8f288dafaba...
🤖 Model parameters: ✅ Valid
   📝 Model name: Enhanced_CIAF_Demo_Model
   🔧 Parameters: {'model_type': 'RandomForestClassifier'...}
   🔍 Expected fingerprint: 94c603d9c0c024cf124ecd9dc136107b...
   🧮 Calculated fingerprint: 94c603d9c0c024cf124ecd9dc136107b...
📋 Audit connections: ✅ Valid
   🔗 Event count: 2
   📄 Event 1: training_started (✅)
      🆔 Event ID: training_start
      ⏰ Timestamp: 2025-09-19T10:00:00Z
      🔍 Expected hash: 3ae6ac3adf1cf3579d9c99fb4c1d52bf...
      🧮 Calculated hash: 3ae6ac3adf1cf3579d9c99fb4c1d52bf...
========================================
🎯 Overall Receipt: ✅ VALID
```

### 🎯 Usage Instructions

1. **Run the benchmark** to see deferred LCM performance improvements:
   ```bash
   cd tools/
   python deferred_lcm_benchmark.py
   ```

2. **Verify generated receipts** using the independent verification tool:
   ```bash
   python verify_receipt.py ../extracted_ciaf_receipt_for_verification.json
   ```

3. **Complete workflow demo** from generation to verification:
   ```bash
   python demo_receipt_verification.py
   ```

4. **Extract custom receipts** from any audit batch:
   ```bash
   python extract_receipt_for_verification.py
   ```

### 📁 Tools Directory Structure
```
tools/
├── verify_receipt.py              # Independent receipt verification
├── deferred_lcm_benchmark.py      # Performance demonstration  
├── enhanced_model_wrapper.py      # Enhanced CIAF wrapper
├── demo_receipt_verification.py   # Complete workflow demo
├── extract_receipt_for_verification.py  # Receipt extraction
├── verification_enhancement_summary.py  # Feature summary
└── examples/                      # Additional examples
    ├── quickstart.py
    ├── lcm_integration_demo.py
    └── credit_model_demo.py
```

These tools provide everything needed to:
- **Understand CIAF capabilities** through working demonstrations
- **Verify audit integrity** with independent cryptographic validation
- **Benchmark performance** across different LCM configurations
- **Generate compliance reports** suitable for regulatory review
- **Debug verification issues** with detailed diagnostic output

---

## CLI Tools

```bash
# Setup metadata storage
python -m ciaf.cli setup my_project --backend sqlite --template production

# Generate a compliance report
python -m ciaf.cli compliance eu_ai_act my_model_id --format html --output compliance_report.html

# Trace metadata lineage from inference receipt
python -m ciaf.examples.metadata_reveal
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

### Metadata Lineage Tracing

```python
from ciaf.examples.metadata_reveal import MetadataReveal

# Trace complete lineage from single inference receipt
revealer = MetadataReveal()
trail = revealer.reveal_metadata_trail("r_a1b2c3d4")

# Verify integrity and generate compliance report
integrity_ok = revealer.verify_trail_integrity(trail)
report = revealer.export_trail_report(trail, "compliance_report.json")
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
| **Audit Trails** | ✅ Working | Hash-connected events |
| **Lazy Materialization** | ✅ Working | On-demand capsules |
| **Inference Connections** | ✅ Working | ZKE connections system |
| **Metadata Traceability** | ✅ Working | Complete lineage tracking |
| **Basic CLI** | 🧪 Prototype | Setup & compliance |
| **Compliance Mapping** | 🧪 Prototype | EU AI Act, NIST |
| **Receipt Verification** | ✅ Working | Independent verifier |
| **Healthcare Patterns** | 🧪 Prototype | PHI scaffolding |

### Near-term Roadmap 

| Feature | Priority | Target |
|---------|----------|--------|
| **API Stabilization** | 🔴 High | Finalize public APIs |
| **Documentation** | 🔴 High | Complete API reference |
| **Test Coverage** | 🔴 High | >90% |
| **Performance Optimization** | 🟡 Medium | LCM efficiency |
| **CLI Enhancement** | 🟡 Medium | Full-featured CLI |

### Medium-term Roadmap 

| Feature | Priority | Target |
|---------|----------|--------|
| **GDPR/HIPAA Compliance** | 🔴 High | Production-ready patterns |
| **Advanced Analytics** | 🟡 Medium | Bias/fairness metrics |
| **Integration Libraries** | 🟡 Medium | TF/PyTorch wrappers |
| **Web Dashboard** | 🟢 Low | Audit visualization |
| **Enterprise Features** | 📋 Planned | SSO, RBAC, deployment |

### Research Areas

- **Zero-Knowledge Proofs** (ZK-SNARKs) for privacy-preserving verification
- **Immutable Audit Ledgers** for tamper-evident audit storage 
- **Homomorphic encryption** for computation on encrypted data
- **Formal verification** of cryptographic correctness

---

## License

This project is licensed under a Proprietary License by CognitiveInsight.AI — see [LICENSE](LICENSE) for details.

**Key restrictions:**
- Non-commercial research and evaluation use only
- No redistribution or commercial use without written consent
- Contact 📧 founder@cognitiveinsight.ai for commercial licensing

---

## Acknowledgments

- **cryptography** library and the broader Python security ecosystem
- **Regulatory frameworks**: EU AI Act, NIST AI RMF, GDPR/HIPAA, ISO/IEC 27001, SOX (for mapping inspiration)

> **Personal note:** This project is a work in progress and reflects a commitment to secure, verifiable, and compliant AI systems. The framework is updated periodically as needed to maintain relevance with evolving regulatory requirements and technological advances. Feedback is highly appreciated!
> 
> *— Denzil James Greenwood*