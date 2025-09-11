# Changelog

All notable changes to the Cognitive Insight AI Framework (CIAF) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-09-11

### Fixed
- Fixed critical bug in `showcase_example.py` where `DatasetFamilyMetadata` was called with incorrect `dataset_id` parameter (should be `name`)
- Fixed `MerkleTree` initialization error in `showcase_example.py` - constructor now properly accepts leaves list parameter
- Added creation and modification dates to main framework files

### Changed
- **BREAKING**: Updated license from standard MIT to MIT License (Non-Commercial Use)
- Updated copyright ownership to MyImaginaryFriends.ai and CognitiveInsight.AI
- Updated all project URLs and contact information in `pyproject.toml`
- Changed license classifier from OSI-approved to "Other/Proprietary License" in `pyproject.toml`
- Removed emojis from `showcase_example.py` output for more professional console display
- Enhanced `showcase_example.py` with comprehensive productionization features:
  - Added deployment pipeline with predeploy/deploy/test_eval domains
  - Enhanced inference pipeline with chaining and batch root capabilities
  - Added complete root computation including training session roots and release roots
  - Updated policy domains to include all lifecycle phases
  - Added finalized cryptographic signatures and TSA timestamps
  - Enhanced JSON capsule output with production-ready schema v1.1

### Added  
- Creation and modification date headers to core modules:
  - `ciaf/__init__.py`
  - `ciaf/api/framework.py` 
  - `ciaf/core/crypto.py`
- Commercial license contact information (`legal@myimaginaryfriends.ai`)
- New license section in README.md with clear commercial/non-commercial usage guidelines
- Enhanced production features in showcase example:
  - Deployment anchors and intent-actual binding
  - Test evaluation metrics tracking
  - Source control tracking with git commit information
  - Enhanced root computation with complete audit trail

### Documentation
- Enhanced file-level documentation with proper authorship and versioning
- Updated README.md with new license information and usage restrictions
- Updated license badge in README.md to reflect non-commercial status
- Added clear commercial licensing path and contact information

## [1.0.0] - 2025-09-09

### Added
- Initial public release of CIAF (Cognitive Insight AI Framework)
- Core cryptographic utilities with AES-256-GCM encryption and SHA256 hashing
- Hierarchical anchor management system for secure data provenance
- Lazy capsule materialization system for efficient data handling
- Comprehensive compliance engine supporting multiple regulatory frameworks:
  - EU AI Act compliance validation
  - NIST AI Risk Management Framework
  - GDPR data protection compliance
  - HIPAA healthcare data compliance  
  - SOX financial reporting compliance
  - ISO 27001 information security compliance
- Advanced risk assessment capabilities including bias detection and fairness validation
- Uncertainty quantification with multiple methods (Monte Carlo, Bootstrap, etc.)
- Explainability module with SHAP and LIME integration
- Metadata tagging system for AI-generated content
- Preprocessing utilities for text and numerical data
- Model wrapper for drop-in integration with existing ML models
- Command-line tools for metadata setup and compliance reporting
- Comprehensive audit trail and documentation generation
- Transparency reporting and stakeholder impact assessment
- Cybersecurity compliance assessment tools
- Visualization engine for compliance and risk metrics

### Security
- Industry-standard cryptographic implementations
- Secure random number generation
- Binary anchor derivation with HMAC-SHA256
- Tamper-evident audit trails using Merkle trees
- Authenticated encryption with additional data (AAD) support

### Compliance
- Built-in support for major regulatory frameworks
- Automated compliance validation and reporting
- Risk assessment and bias detection capabilities
- Comprehensive documentation generation
- Audit trail maintenance with cryptographic verification

### Documentation
- Complete API documentation
- Integration examples for popular ML frameworks
- Security best practices guide
- Compliance implementation guides
- CLI usage documentation

[1.0.0]: https://github.com/your-org/ciaf/releases/tag/v1.0.0
