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

- **Cryptographic Provenance Tracking** — End-to-end verifiable data lineage with Merkle trees and hash chains  
- **Lazy Capsule Materialization (LCM)** — On-demand proof capsule materialization to minimize storage and exposure  
- **Compliance Mapping** — Artifacts designed to map to EU AI Act, NIST AI RMF, GDPR/HIPAA, SOX, ISO/IEC 27001 (see `docs/compliance/`)  
- **Security-First Design** — AES-256-GCM (optional), secure anchor derivation, tamper-evident audit trails  
- **Risk Assessment Patterns** — Bias/fairness checks and uncertainty quantification scaffolding  
- **Transparency & Explainability** — Hooks for decision transparency and receipt generation  
- **Healthcare Patterns** — PHI minimization and consent tracking patterns (final compliance depends on deployment)  
- **Performance Monitoring** — Basic metrics for LCM operations

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

## Option B: Directly from GitHub
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

# Dataset anchor (cryptographic root for dataset operations)
anchor = framework.create_dataset_anchor(
    dataset_id="healthcare_data",
    dataset_metadata={"source": "hospital_system", "type": "medical_records"},
    master_password="secure_password_123"
)

# Create provenance capsules
data_items = [
    {"content": "patient_record_1", "metadata": {"id": "p001", "consent": True}},
    {"content": "patient_record_2", "metadata": {"id": "p002", "consent": True}},
]
capsules = framework.create_provenance_capsules("healthcare_data", data_items)

# Model anchor (immutable parameter/architecture fingerprints + dataset authorization)
model_anchor = framework.create_model_anchor(
    model_name="diagnostic_model",
    model_parameters={"epochs": 100, "lr": 0.001},
    model_architecture={"type": "bert_classifier", "hidden": 768},
    authorized_datasets=["healthcare_data"],
    master_password="secure_model_password"
)

# Verifiable training snapshot
snapshot = framework.train_model(
    model_name="diagnostic_model",
    capsules=capsules,
    maa=model_anchor,
    training_params={"epochs": 100, "lr": 0.001},
    model_version="v1.0"
)

# Integrity check
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
│  ├─ Uncertainty quantification scaffolding
│  └─ Security assessment hooks
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

# Materialize a capsule on-demand
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
ciaf-setup-metadata my_project --backend sqlite --template production

# Generate compliance report
ciaf-compliance-report eu_ai_act my_model_id --format html --output compliance_report.html
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

- **AES-256-GCM**: optional authenticated encryption (with AAD)
- **SHA-256**: hashing for integrity verification
- **HMAC-SHA-256**: anchor derivation and message authentication
- **Merkle Trees**: canonical binary concatenation for tamper-evident sets

### Anchor Management

- Hierarchical anchor derivation (Master → Dataset → Capsule)
- Cryptographically secure randomness
- High-entropy binary anchors
- Canonicalized operations for derivations & Merkle policies
- Backwards compatibility for legacy key-based terminology

### Access Controls (patterns)

- Role-based access patterns
- Audit logging hooks
- Session management scaffolding

---

## Healthcare & HIPAA Patterns

```python
from ciaf import ModelMetadataManager
from ciaf.compliance import ComplianceFramework

manager = ModelMetadataManager("healthcare_ai", "1.0.0")
manager.enable_phi_protection()
manager.set_compliance_frameworks([ComplianceFramework.HIPAA])

manager.capture_metadata({
    "patient_id": "XXXXX",     # handled with PHI patterns
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
- **Security**: [Security Policy](ciaf/SECURITY.md)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **cryptography** library and broader Python security ecosystem
- **Regulatory frameworks**: EU AI Act, NIST AI RMF, GDPR/HIPAA, ISO/IEC 27001, SOX (for mapping inspiration)

> **Personal note:** This project is a work in progress and reflects a commitment to secure, verifiable, and compliant AI systems. Feedback and contributions are highly appreciated!
> 
> *— Denzil James Greenwood*    master_password="secure_password_123"

CIAF Framework

├── Core Components)

│   ├── Cryptographic Utilities (AES-256-GCM, SHA256, HMAC)

│   ├── Anchor Management (Hierarchical anchor derivation)```python

│   └── Merkle Tree Implementation

├── Anchoring Systemfrom ciaf import CIAFFramework, ModelMetadataManager# Create provenance capsules for your data

│   ├── Dataset Anchors (Master anchor → Dataset anchor → Capsule anchors)

│   └── Lazy Managers (Efficient capsule materialization)data_items = [

├── Provenance Tracking

│   ├── Provenance Capsules (Encrypted data with metadata)framework = CIAFFramework("MyAI_Project")    {"content": "patient_record_1", "metadata": {"id": "p001", "consent": True}},

│   └── Training Snapshots (Verifiable model states)

├── Compliance Engine    {"content": "patient_record_2", "metadata": {"id": "p002", "consent": True}}

│   ├── Regulatory Mapping (EU AI Act, NIST, GDPR, HIPAA, etc.)

│   ├── Validators (Automated compliance checking)# Dataset anchor (cryptographic root for dataset operations)]

│   └── Audit Trails (Immutable event logging)

├── Risk Assessmentanchor = framework.create_dataset_anchor(

│   ├── Bias Detection & Fairness Validation

│   ├── Uncertainty Quantification    dataset_id="healthcare_data",capsules = framework.create_provenance_capsules("healthcare_data", data_items)

│   └── Security Assessment

├── Metadata Management    dataset_metadata={"source": "hospital_system", "type": "medical_records"},

│   ├── Storage Backends (JSON, SQLite, Pickle)

│   ├── Configuration Templates    master_password="secure_password_123"# Create Model Aggregation Anchor for training authorization

│   └── Integration Utilities

└── Utilities)maa = framework.create_model_aggregation_anchor(

    ├── CLI Tools

    ├── Model Wrappers    model_name="diagnostic_model", 

    └── ML Framework Simulators

```# Create provenance capsules    authorized_datasets=["healthcare_data"]



## Compliance Supportdata_items = [)



**Compliance Mapping:** CIAF maps audit artifacts to controls in EU AI Act, NIST AI RMF, GDPR/HIPAA, SOX, ISO/IEC 27001. Coverage varies by control; see `docs/compliance/` for current status and gaps.    {"content": "patient_record_1", "metadata": {"id": "p001", "consent": True}},



### EU AI Act    {"content": "patient_record_2", "metadata": {"id": "p002", "consent": True}},# Train your model with verifiable provenance

- Risk Management System patterns

- Quality Management System templates]snapshot = framework.train_model(

- Data Governance & Bias Monitoring tools

- Technical Documentation generationcapsules = framework.create_provenance_capsules("healthcare_data", data_items)    model_name="diagnostic_model",

- Record Keeping & Audit Trails

    capsules=capsules,

### NIST AI Risk Management Framework

- AI Risk Management Strategy templates# Model anchor (immutable parameter/architecture fingerprints + dataset authorization)    maa=maa,

- AI System Inventory & Mapping

- Impact Assessment toolsmodel_anchor = framework.create_model_anchor(    training_params={"epochs": 100, "lr": 0.001},

- Continuous Monitoring patterns

    model_name="diagnostic_model",    model_version="v1.0"

### Data Protection (GDPR, HIPAA, CCPA)

- Data Subject Rights Management patterns    model_parameters={"epochs": 100, "lr": 0.001},)

- Consent Tracking & Validation

- Data Minimization & Purpose Limitation    model_architecture={"type": "bert_classifier", "hidden": 768},

- Breach Detection & Notification

    authorized_datasets=["healthcare_data"],# Validate training integrity

### Financial & Security (SOX, PCI DSS, ISO 27001)

- Internal Controls Over Financial Reporting    master_password="secure_model_password"is_valid = framework.validate_training_integrity(snapshot)

- Documentation & Retention

- Information Security Management)print(f"Training integrity verified: {is_valid}")

- Access Controls & Monitoring

```

## Advanced Features

# Verifiable training snapshot

### Lazy Capsule Materialization (LCM)

snapshot = framework.train_model(## 🏗️ Architecture

CIAF's LCM system allows efficient handling of large datasets:

    model_name="diagnostic_model",

```python

# Create dataset anchor with lazy manager    capsules=capsules,CIAF follows a modular architecture with clear separation of concerns:

# The anchor provides the cryptographic foundation for secure lazy evaluation

anchor = framework.create_dataset_anchor(    maa=model_anchor,

    dataset_id="large_dataset",

    dataset_metadata={"size": "1TB", "type": "image_data"},    training_params={"epochs": 100, "lr": 0.001},```

    master_password="secure_anchor_password"

)    model_version="v1.0"📦 CIAF Framework



# Capsules are created on-demand, not stored in memory)├── 🔑 Core Components

# Each capsule is derived from the dataset anchor on materialization

lazy_manager = framework.lazy_managers["large_dataset"]│   ├── Cryptographic Utilities (AES-256-GCM, SHA256, HMAC)



# Materialize only when needed - anchor provides cryptographic verification# Integrity check│   ├── Anchor Management (Hierarchical anchor derivation)

capsule = lazy_manager.materialize_capsule("item_001")

```assert framework.validate_training_integrity(snapshot)│   └── Merkle Tree Implementation



### Enhanced Model Anchor Systemprint("Training integrity verified.")├── ⚓ Anchoring System



**NEW**: Comprehensive model tracking with immutable parameter fingerprinting:```│   ├── Dataset Anchors (Master anchor → Dataset anchor → Capsule anchors)



```python│   └── Lazy Managers (Efficient capsule materialization)

from ciaf import CIAFFramework

## Architecture├── 📦 Provenance Tracking

framework = CIAFFramework("MyAI_Project")

│   ├── Provenance Capsules (Encrypted data with metadata)

# Create model anchor with parameter hashing (ENHANCED FEATURE)

model_anchor = framework.create_model_anchor(CIAF follows a modular architecture with clear separation of concerns:│   └── Training Snapshots (Verifiable model states)

    model_name="diagnostic_model",

    model_parameters={├── 🛡️ Compliance Engine

        "learning_rate": 2e-5,

        "batch_size": 16,```│   ├── Regulatory Mapping (EU AI Act, NIST, GDPR, HIPAA, etc.)

        "num_epochs": 3,

        "model_type": "bert_classifier"CIAF Framework│   ├── Validators (Automated compliance checking)

    },

    model_architecture={├── Core Components│   └── Audit Trails (Immutable event logging)

        "base_model": "bert-base-uncased",

        "num_labels": 3,│   ├── Cryptographic Utilities (AES-256-GCM, SHA256, HMAC)├── 🎯 Risk Assessment

        "hidden_size": 768

    },│   ├── Anchor Management (Hierarchical anchor derivation)│   ├── Bias Detection & Fairness Validation

    authorized_datasets=["training_data_v1", "validation_data_v1"],

    master_password="secure_model_password"│   └── Merkle Tree Implementation│   ├── Uncertainty Quantification

)

├── Anchoring System│   └── Security Assessment

print(f"Model fingerprint: {model_anchor['parameters_fingerprint']}")

print(f"Architecture fingerprint: {model_anchor['architecture_fingerprint']}")│   ├── Dataset Anchors (Master anchor → Dataset anchor → Capsule anchors)├── 📊 Metadata Management

```

│   └── Lazy Managers (Efficient capsule materialization)│   ├── Storage Backends (JSON, SQLite, Pickle)

### Complete Audit Flow Integration

├── Provenance Tracking│   ├── Configuration Templates

**NEW**: End-to-end audit trail from dataset to inference:

│   ├── Provenance Capsules (Encrypted data with metadata)│   └── Integration Utilities

```python

# Step 1: Train with complete audit│   └── Training Snapshots (Verifiable model states)└── 🔧 Utilities

training_snapshot = framework.train_model_with_audit(

    model_name="diagnostic_model",├── Compliance Engine    ├── CLI Tools

    capsules=training_capsules,

    training_params=training_params,│   ├── Regulatory Mapping (EU AI Act, NIST, GDPR, HIPAA, etc.)    ├── Model Wrappers

    model_version="1.0.0",

    user_id="data_scientist_alice"│   ├── Validators (Automated compliance checking)    └── ML Framework Simulators

)

│   └── Audit Trails (Immutable event logging)```

# Step 2: Perform inference with audit chain

receipt = framework.perform_inference_with_audit(├── Risk Assessment

    model_name="diagnostic_model",

    query="This product is amazing!",│   ├── Bias Detection & Fairness Validation## 📋 Compliance Support

    ai_output="positive (confidence: 0.95)",

    training_snapshot=training_snapshot,│   ├── Uncertainty Quantification

    user_id="api_user"

)│   └── Security AssessmentCIAF provides built-in support for major regulatory frameworks:



# Step 3: Get complete audit trail├── Metadata Management

audit_trail = framework.get_complete_audit_trail("diagnostic_model")

│   ├── Storage Backends (JSON, SQLite, Pickle)### 🇪🇺 EU AI Act

print(f"Complete audit includes:")

print(f"- {audit_trail['verification']['total_datasets']} dataset anchors")│   ├── Configuration Templates- ✅ Risk Management System

print(f"- 1 model anchor with parameter fingerprints")

print(f"- {audit_trail['verification']['total_audit_records']} audit records")│   └── Integration Utilities- ✅ Quality Management System  

print(f"- {audit_trail['inference_chain']['total_receipts']} inference receipts")

```└── Utilities- ✅ Data Governance & Bias Monitoring



### Compliance Validation    ├── CLI Tools- ✅ Technical Documentation



Automated compliance checking across multiple frameworks:    ├── Model Wrappers- ✅ Record Keeping & Audit Trails



```python    └── ML Framework Simulators

from ciaf.compliance import ComplianceValidator, ComplianceFramework

```### 🏛️ NIST AI Risk Management Framework

validator = ComplianceValidator("diagnostic_model")

- ✅ AI Risk Management Strategy

# Validate against multiple frameworks

frameworks = [## Compliance Support- ✅ AI System Inventory & Mapping

    ComplianceFramework.EU_AI_ACT,

    ComplianceFramework.NIST_AI_RMF,- ✅ Impact Assessment

    ComplianceFramework.GDPR

]**Compliance Mapping:** CIAF maps audit artifacts to controls in EU AI Act, NIST AI RMF, GDPR/HIPAA, SOX, ISO/IEC 27001. Coverage varies by control; see `docs/compliance/` for current status and gaps.- ✅ Continuous Monitoring



results = validator.validate_multiple_frameworks(

    frameworks, 

    audit_generator, ### EU AI Act### 🛡️ Data Protection (GDPR, HIPAA, CCPA)

    validation_period_days=30

)- Risk Management System patterns- ✅ Data Subject Rights Management



# Get comprehensive compliance report- Quality Management System templates- ✅ Consent Tracking & Validation

summary = validator.get_validation_summary()

print(f"Compliance rate: {summary['pass_rate']:.1f}%")- Data Governance & Bias Monitoring tools- ✅ Data Minimization & Purpose Limitation

```

- Technical Documentation generation- ✅ Breach Detection & Notification

### Risk Assessment & Bias Detection

- Record Keeping & Audit Trails

Comprehensive fairness and bias validation:

### 🏦 Financial & Security (SOX, PCI DSS, ISO 27001)

```python

from ciaf.compliance import BiasValidator, FairnessValidator### NIST AI Risk Management Framework- ✅ Internal Controls Over Financial Reporting



bias_validator = BiasValidator()- AI Risk Management Strategy templates- ✅ Documentation & Retention

fairness_validator = FairnessValidator()

- AI System Inventory & Mapping- ✅ Information Security Management

# Detect bias in model predictions

bias_results = bias_validator.validate_predictions(- Impact Assessment tools- ✅ Access Controls & Monitoring

    predictions=model_predictions,

    protected_attributes={"gender": gender_data, "age": age_data}- Continuous Monitoring patterns

)

## 🔧 Advanced Features

# Calculate fairness metrics

fairness_metrics = fairness_validator.calculate_fairness_metrics(### Data Protection (GDPR, HIPAA, CCPA)

    predictions=model_predictions,

    protected_attributes=protected_attributes,- Data Subject Rights Management patterns### Lazy Capsule Materialization

    ground_truth=labels

)- Consent Tracking & Validation

```

- Data Minimization & Purpose LimitationCIAF's lazy evaluation system allows efficient handling of large datasets:

## CLI Tools

- Breach Detection & Notification

```bash

ciaf-setup-metadata my_project --backend sqlite --template production```python

ciaf-compliance-report eu_ai_act my_model_id --format html --output compliance_report.html

```### Financial & Security (SOX, PCI DSS, ISO 27001)# Create dataset anchor with lazy manager



## Documentation Structure- Internal Controls Over Financial Reporting# The anchor provides the cryptographic foundation for secure lazy evaluation



- **Core Concepts**: Understand CIAF's cryptographic foundations- Documentation & Retentionanchor = framework.create_dataset_anchor(

- **Compliance Guides**: Step-by-step compliance implementation

- **API Reference**: Comprehensive API documentation- Information Security Management    dataset_id="large_dataset",

- **Integration Examples**: Real-world use cases and patterns

- **Security Best Practices**: Guidelines for secure deployment- Access Controls & Monitoring    dataset_metadata={"size": "1TB", "type": "image_data"},

- See our [Security Policy](ciaf/SECURITY.md) for reporting, supported versions, and secure deployment guidance

    master_password="secure_anchor_password"

## Security Features

## Advanced Features)

See our [Security Policy](ciaf/SECURITY.md) for comprehensive security information, vulnerability reporting, and security best practices.



### Cryptographic Security

- **AES-256-GCM**: Industry-standard authenticated encryption with optional AAD support### Lazy Capsule Materialization (LCM)# Capsules are created on-demand, not stored in memory

- **SHA256**: Cryptographic hashing for integrity verification  

- **HMAC-SHA256**: Binary message authentication for anchor derivation# Each capsule is derived from the dataset anchor on materialization

- **Merkle Trees**: Canonical binary concatenation for tamper-evident data structures

CIAF's LCM system allows efficient handling of large datasets:lazy_manager = framework.lazy_managers["large_dataset"]

### Anchor Management

- **Hierarchical Anchor Derivation**: Master anchor → Dataset anchor → Capsule anchor hierarchy with binary HMAC anchors

- **Secure Random Generation**: Cryptographically secure randomness

- **Binary Anchor Security**: True binary anchors for maximum entropy and cryptographic strength```python# Materialize only when needed - anchor provides cryptographic verification

- **Canonical Operations**: Industry-standard anchor derivation and Merkle tree implementations

- **Legacy Compatibility**: Maintains backwards compatibility with previous key-based terminology# Create dataset anchor with lazy managercapsule = lazy_manager.materialize_capsule("item_001")



### Access Controls# The anchor provides the cryptographic foundation for secure lazy evaluation```

- **Role-Based Access**: Granular permission management

- **Audit Logging**: Comprehensive access trackinganchor = framework.create_dataset_anchor(

- **Session Management**: Secure session handling

    dataset_id="large_dataset",### ✨ Enhanced Model Anchor System

## Healthcare & HIPAA Compliance

    dataset_metadata={"size": "1TB", "type": "image_data"},

CIAF provides patterns for healthcare applications:

    master_password="secure_anchor_password"**NEW**: Comprehensive model tracking with immutable parameter fingerprinting:

```python

from ciaf import ModelMetadataManager)

from ciaf.compliance import ComplianceFramework

```python

# Healthcare-specific setup

manager = ModelMetadataManager("healthcare_ai", "1.0.0")# Capsules are created on-demand, not stored in memoryfrom ciaf import CIAFFramework

manager.enable_phi_protection()

manager.set_compliance_frameworks([ComplianceFramework.HIPAA])# Each capsule is derived from the dataset anchor on materialization



# Automatic PHI detection and protection patternslazy_manager = framework.lazy_managers["large_dataset"]framework = CIAFFramework("MyAI_Project")

manager.capture_metadata({

    "patient_id": "XXXXX",  # Automatically detected and protected

    "diagnosis": "diabetes",

    "consent_status": "active"# Materialize only when needed - anchor provides cryptographic verification# Create model anchor with parameter hashing (ENHANCED FEATURE)

})

```capsule = lazy_manager.materialize_capsule("item_001")model_anchor = framework.create_model_anchor(



**Note:** Patterns for PHI minimization and consent tracking are provided; final compliance depends on your deployment and data governance.```    model_name="sentiment_classifier",



## Integration Examples    model_parameters={



### Scikit-learn Integration### Enhanced Model Anchor System        "learning_rate": 2e-5,

```python

from ciaf import CIAFModelWrapper        "batch_size": 16,

from sklearn.ensemble import RandomForestClassifier

**NEW**: Comprehensive model tracking with immutable parameter fingerprinting:        "num_epochs": 3,

# Wrap your model for automatic provenance tracking

model = RandomForestClassifier()        "model_type": "bert_classifier"

wrapped_model = CIAFModelWrapper(model, "diagnostic_model_v1")

```python    },

# Training and predictions are automatically tracked

wrapped_model.fit(X_train, y_train)from ciaf import CIAFFramework    model_architecture={

predictions = wrapped_model.predict(X_test)

```        "base_model": "bert-base-uncased",



### TensorFlow/PyTorch Integrationframework = CIAFFramework("MyAI_Project")        "num_labels": 3,

```python

from ciaf.simulation import MLFrameworkSimulator        "hidden_size": 768



# Simulate ML framework interactions# Create model anchor with parameter hashing (ENHANCED FEATURE)    },

simulator = MLFrameworkSimulator("neural_network")

training_snapshot = simulator.train_model(model_anchor = framework.create_model_anchor(    authorized_datasets=["training_data_v1", "validation_data_v1"],

    training_data_capsules=capsules,

    maa=model_anchor,    model_name="diagnostic_model",    master_password="secure_model_password"

    training_params={"epochs": 50, "batch_size": 32},

    model_version="v2.0"    model_parameters={)

)

```        "learning_rate": 2e-5,



## Performance & Metrics        "batch_size": 16,print(f"Model fingerprint: {model_anchor['parameters_fingerprint']}")



CIAF provides comprehensive performance monitoring for LCM operations:        "num_epochs": 3,print(f"Architecture fingerprint: {model_anchor['architecture_fingerprint']}")



```python        "model_type": "bert_classifier"```

# Get performance metrics for lazy operations

metrics = framework.get_performance_metrics("my_dataset")    },

print(f"Materialization rate: {metrics['materialization_rate']:.2%}")

print(f"Total items: {metrics['total_items']}")    model_architecture={### 🔄 Complete Audit Flow Integration

print(f"Materialized capsules: {metrics['materialized_capsules']}")

```        "base_model": "bert-base-uncased",



## Contributing        "num_labels": 3,**NEW**: End-to-end audit trail from dataset to inference:



We welcome contributions to CIAF! Please see our contributing guidelines:        "hidden_size": 768



1. **Code Style**: We use Black for code formatting    },```python

2. **Testing**: Ensure all tests pass and add tests for new features

3. **Documentation**: Update documentation for any API changes    authorized_datasets=["training_data_v1", "validation_data_v1"],# Step 1: Train with complete audit

4. **Security**: Follow secure coding practices and report security issues responsibly

    master_password="secure_model_password"training_snapshot = framework.train_model_with_audit(

### Development Setup

)    model_name="sentiment_classifier",

```bash

git clone https://github.com/DenzilGreenwood/pyciaf.git    capsules=training_capsules,

cd pyciaf

pip install -e .print(f"Model fingerprint: {model_anchor['parameters_fingerprint']}")    training_params=training_params,

```

print(f"Architecture fingerprint: {model_anchor['architecture_fingerprint']}")    model_version="1.0.0",

## License

```    user_id="data_scientist_alice"

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

)

## Support & Community

### Complete Audit Flow Integration

- **Documentation**: [https://ciaf.readthedocs.io](https://ciaf.readthedocs.io)

- **Issues**: [GitHub Issues](https://github.com/DenzilGreenwood/pyciaf/issues)# Step 2: Perform inference with audit chain

- **Discussions**: [GitHub Discussions](https://github.com/DenzilGreenwood/pyciaf/discussions)

- **Security**: [Security Policy](ciaf/SECURITY.md)**NEW**: End-to-end audit trail from dataset to inference:receipt = framework.perform_inference_with_audit(



## Acknowledgments    model_name="sentiment_classifier",



CIAF is built on the shoulders of giants. We acknowledge the following projects and standards:```python    query="This product is amazing!",



- **Cryptography**: Built on the excellent `cryptography` library# Step 1: Train with complete audit    ai_output="positive (confidence: 0.95)",

- **Regulatory Frameworks**: Implements guidelines from EU AI Act, NIST AI RMF, and others

training_snapshot = framework.train_model_with_audit(    training_snapshot=training_snapshot,

---

    model_name="diagnostic_model",    user_id="api_user"

**Personal Note**: This project is a work in progress and reflects my passion for secure and compliant AI systems. Feedback and contributions are highly appreciated!

    capsules=training_capsules,)

*- Denzil James Greenwood*
    training_params=training_params,

    model_version="1.0.0",# Step 3: Get complete audit trail

    user_id="data_scientist_alice"audit_trail = framework.get_complete_audit_trail("sentiment_classifier")

)

print(f"Complete audit includes:")

# Step 2: Perform inference with audit chainprint(f"- {audit_trail['verification']['total_datasets']} dataset anchors")

receipt = framework.perform_inference_with_audit(print(f"- 1 model anchor with parameter fingerprints")

    model_name="diagnostic_model",print(f"- {audit_trail['verification']['total_audit_records']} audit records")

    query="This product is amazing!",print(f"- {audit_trail['inference_chain']['total_receipts']} inference receipts")

    ai_output="positive (confidence: 0.95)",```

    training_snapshot=training_snapshot,

    user_id="api_user"### Compliance Validation

)

Automated compliance checking across multiple frameworks:

# Step 3: Get complete audit trail

audit_trail = framework.get_complete_audit_trail("diagnostic_model")```python

from ciaf.compliance import ComplianceValidator, ComplianceFramework

print(f"Complete audit includes:")

print(f"- {audit_trail['verification']['total_datasets']} dataset anchors")validator = ComplianceValidator("my_model")

print(f"- 1 model anchor with parameter fingerprints")

print(f"- {audit_trail['verification']['total_audit_records']} audit records")# Validate against multiple frameworks

print(f"- {audit_trail['inference_chain']['total_receipts']} inference receipts")frameworks = [

```    ComplianceFramework.EU_AI_ACT,

    ComplianceFramework.NIST_AI_RMF,

### Compliance Validation    ComplianceFramework.GDPR

]

Automated compliance checking across multiple frameworks:

results = validator.validate_multiple_frameworks(

```python    frameworks, 

from ciaf.compliance import ComplianceValidator, ComplianceFramework    audit_generator, 

    validation_period_days=30

validator = ComplianceValidator("diagnostic_model"))



# Validate against multiple frameworks# Get comprehensive compliance report

frameworks = [summary = validator.get_validation_summary()

    ComplianceFramework.EU_AI_ACT,print(f"Compliance rate: {summary['pass_rate']:.1f}%")

    ComplianceFramework.NIST_AI_RMF,```

    ComplianceFramework.GDPR

]### Risk Assessment & Bias Detection



results = validator.validate_multiple_frameworks(Comprehensive fairness and bias validation:

    frameworks, 

    audit_generator, ```python

    validation_period_days=30from ciaf.compliance import BiasValidator, FairnessValidator

)

bias_validator = BiasValidator()

# Get comprehensive compliance reportfairness_validator = FairnessValidator()

summary = validator.get_validation_summary()

print(f"Compliance rate: {summary['pass_rate']:.1f}%")# Detect bias in model predictions

```bias_results = bias_validator.validate_predictions(

    predictions=model_predictions,

### Risk Assessment & Bias Detection    protected_attributes={"gender": gender_data, "age": age_data}

)

Comprehensive fairness and bias validation:

# Calculate fairness metrics

```pythonfairness_metrics = fairness_validator.calculate_fairness_metrics(

from ciaf.compliance import BiasValidator, FairnessValidator    predictions=model_predictions,

    protected_attributes=protected_attributes,

bias_validator = BiasValidator()    ground_truth=labels

fairness_validator = FairnessValidator())

```

# Detect bias in model predictions

bias_results = bias_validator.validate_predictions(## 🛠️ CLI Tools

    predictions=model_predictions,

    protected_attributes={"gender": gender_data, "age": age_data}CIAF includes command-line tools for common operations:

)

### Setup Metadata Storage

# Calculate fairness metrics```bash

fairness_metrics = fairness_validator.calculate_fairness_metrics(ciaf-setup-metadata my_project --backend sqlite --template production

    predictions=model_predictions,```

    protected_attributes=protected_attributes,

    ground_truth=labels### Generate Compliance Reports

)```bash

```ciaf-compliance-report eu_ai_act my_model_id --format html --output compliance_report.html

```

## CLI Tools

## 📚 Documentation Structure

```bash

ciaf-setup-metadata my_project --backend sqlite --template production- **Core Concepts**: Understand CIAF's cryptographic foundations

ciaf-compliance-report eu_ai_act my_model_id --format html --output compliance_report.html- **Compliance Guides**: Step-by-step compliance implementation

```- **API Reference**: Comprehensive API documentation

- **Integration Examples**: Real-world use cases and patterns

## Documentation Structure- **Security Best Practices**: Guidelines for secure deployment

- **[ciaf/SECURITY.md](ciaf/SECURITY.md)**: Comprehensive security policy and vulnerability reporting

- **Core Concepts**: Understand CIAF's cryptographic foundations

- **Compliance Guides**: Step-by-step compliance implementation## 🔒 Security Features

- **API Reference**: Comprehensive API documentation

- **Integration Examples**: Real-world use cases and patterns> **🛡️ For comprehensive security information, vulnerability reporting, and security best practices, see our [ciaf/SECURITY.md](ciaf/SECURITY.md) file.**

- **Security Best Practices**: Guidelines for secure deployment

- See our [Security Policy](ciaf/SECURITY.md) for reporting, supported versions, and secure deployment guidance### Cryptographic Security

- **AES-256-GCM**: Industry-standard authenticated encryption with optional AAD support

## Security Features- **SHA256**: Cryptographic hashing for integrity verification  

- **HMAC-SHA256**: Binary message authentication for anchor derivation

See our [Security Policy](ciaf/SECURITY.md) for comprehensive security information, vulnerability reporting, and security best practices.- **Merkle Trees**: Canonical binary concatenation for tamper-evident data structures



### Cryptographic Security### Anchor Management

- **AES-256-GCM**: Industry-standard authenticated encryption with optional AAD support- **Hierarchical Anchor Derivation**: Master anchor → Dataset anchor → Capsule anchor hierarchy with binary HMAC anchors

- **SHA256**: Cryptographic hashing for integrity verification  - **Secure Random Generation**: Cryptographically secure randomness

- **HMAC-SHA256**: Binary message authentication for anchor derivation- **Binary Anchor Security**: True binary anchors for maximum entropy and cryptographic strength

- **Merkle Trees**: Canonical binary concatenation for tamper-evident data structures- **Canonical Operations**: Industry-standard anchor derivation and Merkle tree implementations

- **Legacy Compatibility**: Maintains backwards compatibility with previous key-based terminology

### Anchor Management

- **Hierarchical Anchor Derivation**: Master anchor → Dataset anchor → Capsule anchor hierarchy with binary HMAC anchors### Access Controls

- **Secure Random Generation**: Cryptographically secure randomness- **Role-Based Access**: Granular permission management

- **Binary Anchor Security**: True binary anchors for maximum entropy and cryptographic strength- **Audit Logging**: Comprehensive access tracking

- **Canonical Operations**: Industry-standard anchor derivation and Merkle tree implementations- **Session Management**: Secure session handling

- **Legacy Compatibility**: Maintains backwards compatibility with previous key-based terminology

## 🏥 Healthcare & HIPAA Compliance

### Access Controls

- **Role-Based Access**: Granular permission managementCIAF provides specialized support for healthcare applications:

- **Audit Logging**: Comprehensive access tracking

- **Session Management**: Secure session handling```python

from ciaf import ModelMetadataManager

## Healthcare & HIPAA Compliancefrom ciaf.compliance import ComplianceFramework



CIAF provides patterns for healthcare applications:# Healthcare-specific setup

manager = ModelMetadataManager("healthcare_ai", "1.0.0")

```pythonmanager.enable_phi_protection()

from ciaf import ModelMetadataManagermanager.set_compliance_frameworks([ComplianceFramework.HIPAA])

from ciaf.compliance import ComplianceFramework

# Automatic PHI detection and protection

# Healthcare-specific setupmanager.capture_metadata({

manager = ModelMetadataManager("healthcare_ai", "1.0.0")    "patient_id": "XXXXX",  # Automatically detected and protected

manager.enable_phi_protection()    "diagnosis": "diabetes",

manager.set_compliance_frameworks([ComplianceFramework.HIPAA])    "consent_status": "active"

})

# Automatic PHI detection and protection patterns```

manager.capture_metadata({

    "patient_id": "XXXXX",  # Automatically detected and protected## 🌐 Integration Examples

    "diagnosis": "diabetes",

    "consent_status": "active"### Scikit-learn Integration

})```python

```from ciaf import CIAFModelWrapper

from sklearn.ensemble import RandomForestClassifier

**Note:** Patterns for PHI minimization and consent tracking are provided; final compliance depends on your deployment and data governance.

# Wrap your model for automatic provenance tracking

## Integration Examplesmodel = RandomForestClassifier()

wrapped_model = CIAFModelWrapper(model, "fraud_detection_v1")

### Scikit-learn Integration

```python# Training and predictions are automatically tracked

from ciaf import CIAFModelWrapperwrapped_model.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifierpredictions = wrapped_model.predict(X_test)

```

# Wrap your model for automatic provenance tracking

model = RandomForestClassifier()### TensorFlow/PyTorch Integration

wrapped_model = CIAFModelWrapper(model, "diagnostic_model_v1")```python

from ciaf.simulation import MLFrameworkSimulator

# Training and predictions are automatically tracked

wrapped_model.fit(X_train, y_train)# Simulate ML framework interactions

predictions = wrapped_model.predict(X_test)simulator = MLFrameworkSimulator("neural_network")

```training_snapshot = simulator.train_model(

    training_data_capsules=capsules,

### TensorFlow/PyTorch Integration    maa=model_anchor,

```python    training_params={"epochs": 50, "batch_size": 32},

from ciaf.simulation import MLFrameworkSimulator    model_version="v2.0"

)

# Simulate ML framework interactions```

simulator = MLFrameworkSimulator("neural_network")

training_snapshot = simulator.train_model(## 📊 Performance & Metrics

    training_data_capsules=capsules,

    maa=model_anchor,CIAF provides comprehensive performance monitoring:

    training_params={"epochs": 50, "batch_size": 32},

    model_version="v2.0"```python

)# Get performance metrics for lazy operations

```metrics = framework.get_performance_metrics("my_dataset")

print(f"Materialization rate: {metrics['materialization_rate']:.2%}")

## Performance & Metricsprint(f"Total items: {metrics['total_items']}")

print(f"Materialized capsules: {metrics['materialized_capsules']}")

CIAF provides comprehensive performance monitoring for LCM operations:```



```python## 🤝 Contributing

# Get performance metrics for lazy operations

metrics = framework.get_performance_metrics("my_dataset")We welcome contributions to CIAF! Please see our contributing guidelines:

print(f"Materialization rate: {metrics['materialization_rate']:.2%}")

print(f"Total items: {metrics['total_items']}")1. **Code Style**: We use Black for code formatting

print(f"Materialized capsules: {metrics['materialized_capsules']}")2. **Testing**: Ensure all tests pass and add tests for new features

```3. **Documentation**: Update documentation for any API changes

4. **Security**: Follow secure coding practices and report security issues responsibly

## Contributing

### Development Setup

We welcome contributions to CIAF! Please see our contributing guidelines:

```bash

1. **Code Style**: We use Black for code formattinggit clone https://github.com/DenzilGreenwood/pyciaf.git

2. **Testing**: Ensure all tests pass and add tests for new featurescd pyciaf

3. **Documentation**: Update documentation for any API changespip install -e .

4. **Security**: Follow secure coding practices and report security issues responsibly```



### Development Setup## 🆘 Support & Community



```bash- **Documentation**: [https://ciaf.readthedocs.io](https://ciaf.readthedocs.io)

git clone https://github.com/DenzilGreenwood/pyciaf.git- **Issues**: [GitHub Issues](https://github.com/DenzilGreenwood/pyciaf/issues)

cd pyciaf- **Discussions**: [GitHub Discussions](https://github.com/DenzilGreenwood/pyciaf/discussions)

pip install -e .- **Security**: [Security Policy](ciaf/SECURITY.md)

```

## 🙏 Acknowledgments

## License

CIAF is built on the shoulders of giants. We acknowledge the following projects and standards:

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

- **Cryptography**: Built on the excellent `cryptography` library

## Support & Community- **Regulatory Frameworks**: Implements guidelines from EU AI Act, NIST AI RMF, and others



- **Documentation**: [https://ciaf.readthedocs.io](https://ciaf.readthedocs.io)---

- **Issues**: [GitHub Issues](https://github.com/DenzilGreenwood/pyciaf/issues)

- **Discussions**: [GitHub Discussions](https://github.com/DenzilGreenwood/pyciaf/discussions)**Personal Note**: This project is a work in progress and reflects my passion for secure and compliant AI systems. Feedback and contributions are highly appreciated!

- **Security**: [Security Policy](ciaf/SECURITY.md)

*- Denzil James Greenwood*

## Acknowledgments



CIAF is built on the shoulders of giants. We acknowledge the following projects and standards:

- **Cryptography**: Built on the excellent `cryptography` library
- **Regulatory Frameworks**: Implements guidelines from EU AI Act, NIST AI RMF, and others

---

**Personal Note**: This project is a work in progress and reflects my passion for secure and compliant AI systems. Feedback and contributions are highly appreciated!

*- Denzil James Greenwood*