# CIAF Test Suite

Comprehensive test suite for the CIAF (Cognitive Insight Audit Framework) codebase.

## Test Coverage

### Core Modules

- **test_core_comprehensive.py** - Core cryptographic components
  - SHA-256 hashing and HMAC
  - Ed25519 signatures (RFC 8032)
  - Merkle trees and inclusion proofs
  - WORM storage
  - Key derivation and anchor generation

- **test_agents.py** - Agentic execution framework
  - Agent event types (READ, WRITE, API_CALL, DECISION, etc.)
  - Identity and access management (IAM)
  - Privilege access management (PAM)
  - Policy engine and authorization
  - Action requests and receipts
  - Elevation grants

- **test_watermarks_comprehensive.py** - Watermarking and forensic provenance
  - Artifact types (text, image, PDF, video, audio)
  - Watermark types (visible, metadata, embedded, hybrid)
  - Dual-state hashing (before/after watermark)
  - Forensic DNA fragments
  - Artifact verification and tamper detection
  - Perceptual hashing

- **test_lcm_comprehensive.py** - Lazy Capsule Materialization (LCM)
  - Dataset anchoring and split management
  - Model anchoring and versioning
  - Training session tracking
  - Deployment management (pre-deployment → deployment)
  - Inference receipts
  - Merkle root computation
  - Capsule headers (8 stages: A-H)

- **test_compliance_comprehensive.py** - Regulatory compliance
  - Framework mapping (EU AI Act, GDPR, HIPAA, NIST AI RMF, SOX)
  - Risk assessment (unacceptable, high, limited, minimal)
  - Bias validation and fairness metrics
  - Audit trails and transparency reports
  - Human oversight requirements
  - Cybersecurity controls
  - Pre-ingestion validation
  - Robustness testing

- **test_web_comprehensive.py** - Web AI governance
  - Web AI event tracking (prompt submit, output receive, file upload)
  - Policy engine for web domains
  - Content classification (text, code, JSON)
  - AI-generated content detection
  - Telemetry and usage monitoring
  - PII redaction and GDPR compliance
  - Receipt generation and chaining

- **test_vault_comprehensive.py** - Vault storage and management
  - Backend implementations (PostgreSQL, in-memory)
  - Event storage (agent events, web AI events, watermark evidence)
  - Query capabilities (time range, risk level, agent ID)
  - Hash chain validation
  - Bulk operations
  - Data retention and GDPR erasure

- **test_integration_comprehensive.py** - Cross-module integration workflows
  - Agent action → policy → vault workflow
  - Watermark lifecycle (creation → storage → verification)
  - Complete ML pipeline (dataset → training → deployment → inference)
  - Web AI governance with policy enforcement
  - Healthcare AI compliance pipeline
  - Cross-module integrations

### Existing Tests (Legacy)

- **test_anchors.py** - Anchor derivation tests
- **test_merkle.py** - Merkle tree tests
- **test_fragment_verification.py** - Fragment verification
- **test_signature_envelope.py** - Signature envelope pattern
- **test_pdf_visual_watermarking.py** - PDF watermarking
- **test_perceptual_hashing.py** - Perceptual hash functions
- **test_performance_benchmarks.py** - Performance benchmarks
- **test_integration_workflow.py** - Legacy integration tests

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Module Tests

```bash
# Core cryptographic tests
pytest tests/test_core_comprehensive.py -v

# Agent framework tests
pytest tests/test_agents.py -v

# Watermarking tests
pytest tests/test_watermarks_comprehensive.py -v

# LCM tests
pytest tests/test_lcm_comprehensive.py -v

# Compliance tests
pytest tests/test_compliance_comprehensive.py -v

# Web AI tests
pytest tests/test_web_comprehensive.py -v

# Vault tests
pytest tests/test_vault_comprehensive.py -v

# Integration tests
pytest tests/test_integration_comprehensive.py -v
```

### Run Tests with Coverage

```bash
pytest tests/ --cov=ciaf --cov-report=html
```

### Run Tests by Category

```bash
# Cryptographic tests
pytest tests/ -k "sha256 or merkle or signature" -v

# Governance tests
pytest tests/ -k "agent or policy or compliance" -v

# Provenance tests
pytest tests/ -k "watermark or forensic or evidence" -v
```

## Test Structure

Each test file follows a consistent structure:

1. **Import Section** - Import relevant modules with graceful fallback
2. **Test Classes** - Grouped by functionality
   - `TestXXXTypes` - Enum/constant tests
   - `TestXXXModel` - Data model tests
   - `TestXXXOperations` - Operation/function tests
   - `TestXXXWorkflows` - Real-world scenario tests
   - `TestXXXIntegration` - Cross-module tests
3. **Test Methods** - Individual test cases with descriptive names

### Test Naming Convention

- `test_<feature>` - Basic feature test
- `test_<feature>_<scenario>` - Specific scenario
- `test_<feature>_<edge_case>` - Edge case
- `test_detect_<problem>` - Negative test (detecting failures)

## Test Data

Tests use:
- **Fixtures** - Reusable test data
- **Mock Data** - Generated on-the-fly
- **Real Examples** - Actual use cases from documentation

## Dependencies

```bash
pip install pytest pytest-cov
```

## Continuous Integration

Tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest tests/ -v --cov=ciaf
```

## Test Coverage Goals

- **Core Modules**: >90% coverage
- **Agent Framework**: >85% coverage
- **Watermarking**: >85% coverage
- **LCM**: >80% coverage
- **Compliance**: >75% coverage
- **Web AI**: >80% coverage
- **Vault**: >85% coverage

## Writing New Tests

When adding new features, follow this pattern:

```python
"""
Module Tests

Description of what's being tested.

Created: YYYY-MM-DD
Version: X.X.X
"""

import pytest

class TestNewFeature:
    """Test new feature functionality."""

    def test_basic_operation(self):
        """Test basic operation works."""
        # Arrange
        input_data = setup_test_data()

        # Act
        result = perform_operation(input_data)

        # Assert
        assert result.success is True

    def test_edge_case(self):
        """Test edge case handling."""
        # Test implementation
        pass

    def test_error_handling(self):
        """Test error is properly handled."""
        with pytest.raises(ValueError):
            invalid_operation()
```

## Troubleshooting

### Import Errors

If tests fail with import errors, ensure CIAF is installed:

```bash
pip install -e .
```

### Database Tests

PostgreSQL tests require a test database:

```bash
export CIAF_TEST_DB_URL="postgresql://user:pass@localhost/ciaf_test"
pytest tests/test_vault_comprehensive.py -v
```

### Skipped Tests

Some tests are skipped if optional dependencies are missing:

```python
@pytest.mark.skipif(not MODULE_AVAILABLE, reason="Module not available")
```

Install optional dependencies:

```bash
pip install -e ".[all]"
```

## Contributing

When contributing tests:

1. Follow existing test structure
2. Use descriptive test names
3. Add docstrings to test classes and methods
4. Group related tests in classes
5. Test both success and failure paths
6. Include real-world scenario tests

## Test Statistics

- **Total Test Files**: 18+
- **Total Test Cases**: 500+
- **Coverage Target**: >85%
- **Test Execution Time**: <5 minutes

## License

Tests are part of the CIAF project and follow the same license.
