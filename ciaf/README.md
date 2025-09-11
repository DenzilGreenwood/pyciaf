# 🔒 CIAF - Cognitive Insight AI Framework

**Version 1.0.0**

A comprehensive Python framework for creating verifiable AI training and inference pipelines with cryptographic provenance tracking, regulatory compliance, and lazy capsule materialization.

[![License: MIT (Non-Commercial)](https://img.shields.io/badge/License-MIT%20(Non--Commercial)-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security Policy](https://img.shields.io/badge/security-policy-blue.svg)](SECURITY.md)

## 🌟 Overview

CIAF (Cognitive Insight AI Framework) is a modular framework designed to address the critical challenges of AI transparency, auditability, and compliance in production environments. It provides cryptographically verifiable provenance tracking, lazy data materialization, and comprehensive compliance validation across multiple regulatory frameworks.

### 🎯 Key Features

- **🔐 Cryptographic Provenance Tracking**: End-to-end verifiable data lineage using Merkle trees and hash chains
- **⚡ Lazy Capsule Materialization**: Efficient data handling with on-demand materialization
- **📋 Multi-Framework Compliance**: Support for EU AI Act, NIST AI RMF, GDPR, HIPAA, SOX, ISO 27001, and more
- **🛡️ Security-First Design**: AES-256-GCM encryption, secure anchor derivation, and tamper-evident audit trails
- **📊 Risk Assessment**: Comprehensive bias detection, fairness validation, and uncertainty quantification
- **🔍 Transparency & Explainability**: Algorithmic transparency metrics and decision explanations
- **🏥 HIPAA Compliance**: Built-in support for PHI protection and healthcare data governance
- **📈 Performance Monitoring**: Real-time metrics and performance tracking

## 🚀 Quick Start

### Installation

```bash
pip install ciaf
```

### Basic Usage

```python
from ciaf import CIAFFramework, ModelMetadataManager

# Initialize the framework
framework = CIAFFramework("MyAI_Project")

# Create a dataset anchor for lazy materialization
# The anchor provides cryptographic root for dataset-specific operations
anchor = framework.create_dataset_anchor(
    dataset_id="healthcare_data",
    dataset_metadata={"source": "hospital_system", "type": "medical_records"},
    master_password="secure_password_123"
)

# Create provenance capsules for your data
data_items = [
    {"content": "patient_record_1", "metadata": {"id": "p001", "consent": True}},
    {"content": "patient_record_2", "metadata": {"id": "p002", "consent": True}}
]

capsules = framework.create_provenance_capsules("healthcare_data", data_items)

# Create Model Aggregation Anchor for training authorization
maa = framework.create_model_aggregation_anchor(
    model_name="diagnostic_model", 
    authorized_datasets=["healthcare_data"]
)

# Train your model with verifiable provenance
snapshot = framework.train_model(
    model_name="diagnostic_model",
    capsules=capsules,
    maa=maa,
    training_params={"epochs": 100, "lr": 0.001},
    model_version="v1.0"
)

# Validate training integrity
is_valid = framework.validate_training_integrity(snapshot)
print(f"Training integrity verified: {is_valid}")
```

## 🏗️ Architecture

CIAF follows a modular architecture with clear separation of concerns:

```
📦 CIAF Framework
├── 🔑 Core Components
│   ├── Cryptographic Utilities (AES-256-GCM, SHA256, HMAC)
│   ├── Anchor Management (Hierarchical anchor derivation)
│   └── Merkle Tree Implementation
├── ⚓ Anchoring System
│   ├── Dataset Anchors (Master anchor → Dataset anchor → Capsule anchors)
│   └── Lazy Managers (Efficient capsule materialization)
├── 📦 Provenance Tracking
│   ├── Provenance Capsules (Encrypted data with metadata)
│   └── Training Snapshots (Verifiable model states)
├── 🛡️ Compliance Engine
│   ├── Regulatory Mapping (EU AI Act, NIST, GDPR, HIPAA, etc.)
│   ├── Validators (Automated compliance checking)
│   └── Audit Trails (Immutable event logging)
├── 🎯 Risk Assessment
│   ├── Bias Detection & Fairness Validation
│   ├── Uncertainty Quantification
│   └── Security Assessment
├── 📊 Metadata Management
│   ├── Storage Backends (JSON, SQLite, Pickle)
│   ├── Configuration Templates
│   └── Integration Utilities
└── 🔧 Utilities
    ├── CLI Tools
    ├── Model Wrappers
    └── ML Framework Simulators
```

## 📋 Compliance Support

CIAF provides built-in support for major regulatory frameworks:

### 🇪🇺 EU AI Act
- ✅ Risk Management System
- ✅ Quality Management System  
- ✅ Data Governance & Bias Monitoring
- ✅ Technical Documentation
- ✅ Record Keeping & Audit Trails

### 🏛️ NIST AI Risk Management Framework
- ✅ AI Risk Management Strategy
- ✅ AI System Inventory & Mapping
- ✅ Impact Assessment
- ✅ Continuous Monitoring

### 🛡️ Data Protection (GDPR, HIPAA, CCPA)
- ✅ Data Subject Rights Management
- ✅ Consent Tracking & Validation
- ✅ Data Minimization & Purpose Limitation
- ✅ Breach Detection & Notification

### 🏦 Financial & Security (SOX, PCI DSS, ISO 27001)
- ✅ Internal Controls Over Financial Reporting
- ✅ Documentation & Retention
- ✅ Information Security Management
- ✅ Access Controls & Monitoring

## 🔧 Advanced Features

### Lazy Capsule Materialization

CIAF's lazy evaluation system allows efficient handling of large datasets:

```python
# Create dataset anchor with lazy manager
# The anchor provides the cryptographic foundation for secure lazy evaluation
anchor = framework.create_dataset_anchor(
    dataset_id="large_dataset",
    dataset_metadata={"size": "1TB", "type": "image_data"},
    master_password="secure_anchor_password"
)

# Capsules are created on-demand, not stored in memory
# Each capsule is derived from the dataset anchor on materialization
lazy_manager = framework.lazy_managers["large_dataset"]

# Materialize only when needed - anchor provides cryptographic verification
capsule = lazy_manager.materialize_capsule("item_001")
```

### ✨ Enhanced Model Anchor System

**NEW**: Comprehensive model tracking with immutable parameter fingerprinting:

```python
from ciaf import CIAFFramework

framework = CIAFFramework("MyAI_Project")

# Create model anchor with parameter hashing (ENHANCED FEATURE)
model_anchor = framework.create_model_anchor(
    model_name="sentiment_classifier",
    model_parameters={
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 3,
        "model_type": "bert_classifier"
    },
    model_architecture={
        "base_model": "bert-base-uncased",
        "num_labels": 3,
        "hidden_size": 768
    },
    authorized_datasets=["training_data_v1", "validation_data_v1"],
    master_password="secure_model_password"
)

print(f"Model fingerprint: {model_anchor['parameters_fingerprint']}")
print(f"Architecture fingerprint: {model_anchor['architecture_fingerprint']}")
```

### 🔄 Complete Audit Flow Integration

**NEW**: End-to-end audit trail from dataset to inference:

```python
# Step 1: Train with complete audit
training_snapshot = framework.train_model_with_audit(
    model_name="sentiment_classifier",
    capsules=training_capsules,
    training_params=training_params,
    model_version="1.0.0",
    user_id="data_scientist_alice"
)

# Step 2: Perform inference with audit chain
receipt = framework.perform_inference_with_audit(
    model_name="sentiment_classifier",
    query="This product is amazing!",
    ai_output="positive (confidence: 0.95)",
    training_snapshot=training_snapshot,
    user_id="api_user"
)

# Step 3: Get complete audit trail
audit_trail = framework.get_complete_audit_trail("sentiment_classifier")

print(f"Complete audit includes:")
print(f"- {audit_trail['verification']['total_datasets']} dataset anchors")
print(f"- 1 model anchor with parameter fingerprints")
print(f"- {audit_trail['verification']['total_audit_records']} audit records")
print(f"- {audit_trail['inference_chain']['total_receipts']} inference receipts")
```

### Compliance Validation

Automated compliance checking across multiple frameworks:

```python
from ciaf.compliance import ComplianceValidator, ComplianceFramework

validator = ComplianceValidator("my_model")

# Validate against multiple frameworks
frameworks = [
    ComplianceFramework.EU_AI_ACT,
    ComplianceFramework.NIST_AI_RMF,
    ComplianceFramework.GDPR
]

results = validator.validate_multiple_frameworks(
    frameworks, 
    audit_generator, 
    validation_period_days=30
)

# Get comprehensive compliance report
summary = validator.get_validation_summary()
print(f"Compliance rate: {summary['pass_rate']:.1f}%")
```

### Risk Assessment & Bias Detection

Comprehensive fairness and bias validation:

```python
from ciaf.compliance import BiasValidator, FairnessValidator

bias_validator = BiasValidator()
fairness_validator = FairnessValidator()

# Detect bias in model predictions
bias_results = bias_validator.validate_predictions(
    predictions=model_predictions,
    protected_attributes={"gender": gender_data, "age": age_data}
)

# Calculate fairness metrics
fairness_metrics = fairness_validator.calculate_fairness_metrics(
    predictions=model_predictions,
    protected_attributes=protected_attributes,
    ground_truth=labels
)
```

## 🛠️ CLI Tools

CIAF includes command-line tools for common operations:

### Setup Metadata Storage
```bash
ciaf-setup-metadata my_project --backend sqlite --template production
```

### Generate Compliance Reports
```bash
ciaf-compliance-report eu_ai_act my_model_id --format html --output compliance_report.html
```

## 📚 Documentation Structure

- **Core Concepts**: Understand CIAF's cryptographic foundations
- **Compliance Guides**: Step-by-step compliance implementation
- **API Reference**: Comprehensive API documentation
- **Integration Examples**: Real-world use cases and patterns
- **Security Best Practices**: Guidelines for secure deployment
- **[SECURITY.md](SECURITY.md)**: Comprehensive security policy and vulnerability reporting

## 🔒 Security Features

> **🛡️ For comprehensive security information, vulnerability reporting, and security best practices, see our [SECURITY.md](SECURITY.md) file.**

### Cryptographic Security
- **AES-256-GCM**: Industry-standard authenticated encryption with optional AAD support
- **SHA256**: Cryptographic hashing for integrity verification  
- **HMAC-SHA256**: Binary message authentication for anchor derivation
- **Merkle Trees**: Canonical binary concatenation for tamper-evident data structures

### Anchor Management
- **Hierarchical Anchor Derivation**: Master anchor → Dataset anchor → Capsule anchor hierarchy with binary HMAC anchors
- **Secure Random Generation**: Cryptographically secure randomness
- **Binary Anchor Security**: True binary anchors for maximum entropy and cryptographic strength
- **Canonical Operations**: Industry-standard anchor derivation and Merkle tree implementations
- **Legacy Compatibility**: Maintains backwards compatibility with previous key-based terminology

### Access Controls
- **Role-Based Access**: Granular permission management
- **Audit Logging**: Comprehensive access tracking
- **Session Management**: Secure session handling

## 🏥 Healthcare & HIPAA Compliance

CIAF provides specialized support for healthcare applications:

```python
from ciaf import ModelMetadataManager
from ciaf.compliance import ComplianceFramework

# Healthcare-specific setup
manager = ModelMetadataManager("healthcare_ai", "1.0.0")
manager.enable_phi_protection()
manager.set_compliance_frameworks([ComplianceFramework.HIPAA])

# Automatic PHI detection and protection
manager.capture_metadata({
    "patient_id": "XXXXX",  # Automatically detected and protected
    "diagnosis": "diabetes",
    "consent_status": "active"
})
```

## 🌐 Integration Examples

### Scikit-learn Integration
```python
from ciaf import CIAFModelWrapper
from sklearn.ensemble import RandomForestClassifier

# Wrap your model for automatic provenance tracking
model = RandomForestClassifier()
wrapped_model = CIAFModelWrapper(model, "fraud_detection_v1")

# Training and predictions are automatically tracked
wrapped_model.fit(X_train, y_train)
predictions = wrapped_model.predict(X_test)
```

### TensorFlow/PyTorch Integration
```python
from ciaf.simulation import MLFrameworkSimulator

# Simulate ML framework interactions
simulator = MLFrameworkSimulator("neural_network")
training_snapshot = simulator.train_model(
    training_data_capsules=capsules,
    maa=model_anchor,
    training_params={"epochs": 50, "batch_size": 32},
    model_version="v2.0"
)
```

## 📊 Performance & Metrics

CIAF provides comprehensive performance monitoring:

```python
# Get performance metrics for lazy operations
metrics = framework.get_performance_metrics("my_dataset")
print(f"Materialization rate: {metrics['materialization_rate']:.2%}")
print(f"Total items: {metrics['total_items']}")
print(f"Materialized capsules: {metrics['materialized_capsules']}")
```

## 🤝 Contributing

We welcome contributions to CIAF! Please see our contributing guidelines:

1. **Code Style**: We use Black for code formatting
2. **Testing**: Ensure all tests pass and add tests for new features
3. **Documentation**: Update documentation for any API changes
4. **Security**: Follow secure coding practices and report security issues responsibly

### Development Setup

```bash
git clone https://github.com/your-org/ciaf.git
cd ciaf
pip install -e ".[dev]"
pre-commit install
```

## 📝 License

### MIT License (Non-Commercial Use)
**Copyright © 2025 MyImaginaryFriends.ai and CognitiveInsight.AI**

This project is licensed under a **Non-Commercial MIT License**. 

**✅ Permitted Uses:**
- Personal use and learning
- Educational purposes
- Non-commercial research
- Academic projects

**❌ Restricted Uses:**
- Commercial products or services
- Paid consulting or monetized solutions
- Proprietary platform integration
- Redistribution or resale

**Commercial License Required For:**
- Integration into SaaS products
- Paid consulting services  
- Enterprise or commercial deployments
- Any monetization of the software

**📧 Commercial License Contact:** legal@MyImaginaryFriends.ai

For complete license terms, see the [LICENSE](LICENSE) file.

**Jurisdiction:** This License is governed by the laws of the State of Oklahoma, United States.

## 🆘 Support & Community

- **Documentation**: [https://ciaf.readthedocs.io](https://ciaf.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/cognitiveinsights/ciaf/issues)
- **Discussions**: [GitHub Discussions](https://github.com/cognitiveinsights/ciaf/discussions)
- **Security**: [Security Policy](SECURITY.md)

## 🙏 Acknowledgments

CIAF is built on the shoulders of giants. We acknowledge the following projects and standards:

- **Cryptography**: Built on the excellent `cryptography` library
- **Regulatory Frameworks**: Implements guidelines from EU AI Act, NIST AI RMF, and others

## Personal Note 

- **Personal Note**: This project is a work in progress and reflects my passion for secure and compliant AI systems. Feedback and contributions are highly appreciated! 
- - Claude 4


