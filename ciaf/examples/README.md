# CIAF LCM Examples

This folder contains comprehensive examples demonstrating the CIAF Lifecycle Management (LCM) system capabilities, from basic usage to enterprise-grade audit and compliance scenarios.

## 📋 Available Examples

### 🟢 `basic_example.py` - Getting Started
**Recommended first step for new users**

Demonstrates core CIAF LCM concepts:
- Dataset family creation with train/val/test splits
- Model anchor creation with architecture and environment tracking
- Training session initialization
- Basic integrity verification with Merkle trees

```bash
python basic_example.py
```

**Key Learning Points:**
- Dataset family vs individual dataset approach
- Model anchor fingerprinting (params, architecture, environment)
- Split-based training organization
- Cryptographic integrity verification

---


### 🔴 `advanced_example.py` - Enterprise Audit & Compliance
**For enterprise environments requiring full compliance**

Demonstrates complete enterprise capabilities:
- Comprehensive data governance with compliance frameworks
- Production-grade model architecture with detailed metadata
- Multi-stage deployment pipeline with infrastructure tracking
- Real-time inference monitoring
- Complete compliance reporting (PCI-DSS, GDPR, SOX)
- Enterprise audit report generation

```bash
python advanced_example.py
```

**Key Learning Points:**
- Enterprise data governance practices
- Compliance framework integration
- Multi-region deployment strategies
- Comprehensive audit report generation
- Regulatory compliance verification

---

### 🎯 `showcase_example.py` - Production Demo
**Showcases all production enhancements and fixes**

Features the exact enhanced output format with all improvements:
- Clean Unicode output (no replacement characters)
- Fixed typos and improved formatting
- Reference links in inference receipts
- Capsule signature preparation
- Enhanced timestamping with notarization options
- Extensible RNG source tracking
- Centralized policy constants
- Professional audit-ready JSON output

```bash
python showcase_example.py
```

**Key Learning Points:**
- Production-ready output formatting
- Audit-ready JSON structure with reference links
- Signature and timestamping preparation
- Domain tagging for verifier clarity
- Professional compliance output

---

### 🏁 `quick_lcm_test_final.py` - Reference Implementation
**The working reference that demonstrates all fixes**

This is the working implementation that showcases all the improvements we made during development. It serves as the reference for the exact output format and demonstrates all the "tiny fixes" and hardening features.

```bash
python quick_lcm_test_final.py
```

---

## 🚀 Quick Start Guide

1. **New to CIAF?** Start with `basic_example.py`
2. **Need enterprise compliance?** Try `advanced_example.py`
3. **Want to see all enhancements?** Check `showcase_example.py`

## 🛠️ Running Examples

All examples are self-contained and can be run directly:

```bash
# From the ciaf repository root
cd ciaf/examples

# Run any example
python basic_example.py
python advanced_example.py
python showcase_example.py
```

## 📊 Example Progression

```
Basic Example (Dataset + Model + Training)
    ↓
Advanced Example (+ Enterprise Compliance)
    ↓
Showcase Example (+ All Production Enhancements)
```

## 🎯 What Each Example Teaches

### Core Concepts (All Examples)
- **Dataset Families**: Unified management of train/val/test splits
- **Model Anchoring**: Immutable fingerprinting of parameters and architecture
- **Training Sessions**: Trackable training with checkpoint management
- **Merkle Verification**: Cryptographic integrity across all components

### Enterprise Features (Advanced+)
- **Compliance Integration**: PCI-DSS, GDPR, SOX compliance tracking
- **Multi-Region Deployment**: Geographic distribution with disaster recovery
- **Comprehensive Reporting**: Regulatory audit report generation
- **Infrastructure Tracking**: Container orchestration and service mesh metadata

### Production Enhancements (Showcase+)
- **Clean Output**: Professional formatting without artifacts
- **Reference Links**: Complete audit traversal capabilities
- **Signature Preparation**: Bundle integrity via cryptographic signatures
- **Timestamp Integration**: Multi-authority notarization support
- **Extensible Tracking**: Multi-framework RNG and environment support

## 🔗 Integration with Main CIAF Framework

These LCM examples complement the main CIAF framework examples in the parent repository. While the main CIAF framework focuses on the foundational anchoring and provenance capabilities, the LCM examples demonstrate the complete lifecycle management system built on top of those foundations.

## 📚 Additional Resources

- **CIAF Main Documentation**: `../README.md`
- **Security Policy**: `../SECURITY.md`
- **License Information**: `../LICENSE` 
- **Changelog**: `../CHANGELOG.md`
- **LCM Implementation Guide**: `../../CIAF_LCM_IMPLEMENTATION_COMPLETE.md`

---

**🎉 Ready to build verifiable AI systems with enterprise-grade audit trails!**
