# CIAF Documentation

Welcome to the Cognitive Insight Audit Framework (CIAF) v1.1.0 documentation.

## Quick Navigation

### Getting Started
- [Quickstart Guide](quickstart.md) - Get started with CIAF in 5 minutes
- [Core Concepts](concepts.md) - Understand the anchor-based architecture
- [Project Structure](PROJECT_STRUCTURE.md) - Codebase organization

### User Guides  
- [Deployment Guide](DEPLOYABLE_MODEL_DEMO_GUIDE.md) - Production deployment patterns
- [Deferred LCM](DEFERRED_LCM_README.md) - High-performance inference architecture
- [Model Building](MODEL_BUILDING_GUIDE_V1_1_0.md) - Complete model development guide

### Reference
- [Receipts Schema](receipts.md) - Audit receipt structure and verification
- [Compliance Mapping](compliance-mapping.md) - Regulatory framework support
- [Coding Standards](CODING_STANDARDS.md) - Development guidelines

### Advanced Topics
- [Performance Optimization](PERFORMANCE_OPTIMIZATION_SUMMARY.md) - Performance tuning
- [Enterprise Features](ENTERPRISE_FEATURES_IMPLEMENTATION_SUMMARY.md) - Enterprise integration

## Overview

CIAF provides cryptographically verifiable provenance tracking for AI systems with complete audit trails and regulatory compliance support.

### Core Features

**[SUCCESS] Anchor-Based Architecture**: Hierarchical cryptographic anchors for data lineage  
**[SUCCESS] Lazy Capsule Materialization**: Efficient on-demand proof generation  
**[SUCCESS] Model Anchoring**: Immutable parameter and architecture fingerprinting  
**[SUCCESS] Audit Trails**: Hash-connected event logging with tamper detection  
**[SUCCESS] Compliance Ready**: Pre-built mappings for EU AI Act, NIST AI RMF, GDPR, HIPAA

## Architecture Overview

```
Master Password --> Dataset Anchor --> Capsule Anchors --> Merkle Tree
       |                                                        |
       v                                                        v
Model Parameters --> Parameter Fingerprint              Training Snapshot
       |                     |                                |
       v                     v                                v
Architecture --> Architecture Fingerprint --> Inference Receipt --> Audit Connections
```

### Anchor Hierarchy

1. **Master Password**: High-entropy root of trust
2. **Dataset Anchor**: Derived using HMAC-SHA256 from master password
3. **Capsule Anchors**: Item-specific anchors derived from dataset anchor
4. **Model Anchors**: Parameter and architecture fingerprints

## Getting Started

### Installation

```bash
# Install from wheel (recommended)
pip install ciaf-1.1.0-py3-none-any.whl

# Or install in development mode
git clone https://github.com/DenzilGreenwood/pyciaf.git
cd pyciaf
pip install -e .
```

### Quick Example

```python
from ciaf import CIAFFramework

# Initialize framework
framework = CIAFFramework("my_ai_project")

# Create dataset anchor with master password
anchor = framework.create_dataset_anchor(
    dataset_id="training_data",
    dataset_metadata={"source": "production", "size": 1000},
    master_password="secure_master_password"
)

# Create training data capsules
training_data = [
    {"content": "sample data 1", "metadata": {"id": "001", "label": "positive"}},
    {"content": "sample data 2", "metadata": {"id": "002", "label": "negative"}}
]

capsules = framework.create_provenance_capsules("training_data", training_data)

print(f"[INFO] Created {len(capsules)} provenance capsules")
print(f"[INFO] Dataset anchor: {anchor.dataset_id}")
```

## Core Capabilities

### [SUCCESS] Cryptographic Security
- AES-256-GCM authenticated encryption
- SHA-256 hashing for integrity verification
- HMAC-SHA-256 for anchor derivation  
- Cryptographically secure random generation

### [SUCCESS] Anchor System
- Hierarchical anchor derivation from master passwords
- Deterministic capsule generation with lazy materialization
- Verifiable anchor chains with mathematical soundness

### [SUCCESS] Compliance Framework  
- **EU AI Act**: Risk management and quality management system patterns
- **NIST AI RMF**: System inventory and risk assessment capabilities
- **GDPR**: Data lineage and consent tracking with minimization
- **HIPAA**: PHI protection patterns and technical safeguards
- **SOX/ISO 27001**: Internal controls and security management

### [SUCCESS] Production Features
- **High Performance**: Deferred LCM reduces inference overhead by 90%
- **Scalable**: Background processing for high-volume inference
- **Pickle Preservation**: Complete LCM metadata survives serialization
- **Enterprise Ready**: Role-based access and compliance reporting

## What's New in v1.1.0

### [SUCCESS] Major Architecture Update
- **Anchor-Based Cryptography**: Migrated from legacy "key" terminology to modern "anchor" architecture
- **Type Safety**: Improved bytes/string handling in cryptographic operations  
- **Clean API**: Removed legacy compatibility functions for consistency
- **Test Coverage**: 32/32 tests passing with zero warnings

### [SUCCESS] Performance Improvements  
- **Deferred LCM**: Up to 90% reduction in inference overhead
- **Optimized Storage**: 63% improvement in storage performance
- **Memory Efficiency**: Lazy materialization reduces memory usage

### [SUCCESS] Enhanced Compliance
- **Updated Mappings**: Latest regulatory framework support
- **Automated Reporting**: Streamlined compliance documentation
- **Audit Trail Integrity**: Enhanced tamper detection capabilities

## Support and Community

- **Issues**: [GitHub Issues](https://github.com/DenzilGreenwood/pyciaf/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DenzilGreenwood/pyciaf/discussions)  
- **Security**: [Security Policy](../ciaf/SECURITY.md)
- **Contributing**: See [Coding Standards](CODING_STANDARDS.md)

## Quick Links

| Documentation | Description |
|---------------|-------------|
| [Quickstart](quickstart.md) | 5-minute getting started guide |
| [Concepts](concepts.md) | Core architectural concepts |
| [Receipts](receipts.md) | Audit receipt formats and verification |
| [Compliance](compliance-mapping.md) | Regulatory framework mappings |
| [Performance](DEFERRED_LCM_README.md) | High-performance inference patterns |

## License

CIAF is released under the terms specified in the [LICENSE](../LICENSE) file.