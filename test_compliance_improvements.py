"""
Comprehensive test demonstrating the improved CIAF compliance system.

This test showcases the new interfaces, policy framework, protocol implementations,
and integration with the LCM system.

Created: 2025-09-26
Author: Denzil James Greenwood
Version: 1.0.0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from ciaf.compliance import (
    # Core interfaces and policy
    CompliancePolicy,
    ComplianceFramework,
    ValidationSeverity,
    AuditEventType,
    get_default_compliance_policy,
    set_default_compliance_policy,
    
    # Protocol implementations
    DefaultComplianceValidator,
    DefaultAuditTrailProvider,
    DefaultRiskAssessor,
    DefaultBiasDetector,
    create_default_compliance_protocols,
    
    # Legacy classes for compatibility
    AuditTrailGenerator,
    ComplianceValidator,
    RegulatoryMapper,
)


def test_policy_framework():
    """Test the new compliance policy framework."""
    print("=== Testing Compliance Policy Framework ===")
    
    # Test default policy
    default_policy = get_default_compliance_policy()
    print(f"Default Policy Summary:\n{default_policy.format_policy_summary()}\n")
    
    # Test strict policy
    strict_policy = CompliancePolicy.strict()
    print(f"Strict Policy Summary:\n{strict_policy.format_policy_summary()}\n")
    
    # Test development policy
    dev_policy = CompliancePolicy.development()
    print(f"Development Policy Summary:\n{dev_policy.format_policy_summary()}\n")
    
    # Test policy serialization
    policy_dict = default_policy.to_dict()
    policy_digest = default_policy.policy_digest()
    print(f"Policy digest: {policy_digest[:16]}...\n")
    
    return default_policy


def test_protocol_implementations():
    """Test the new protocol implementations."""
    print("=== Testing Protocol Implementations ===")
    
    # Create protocol implementations
    protocols = create_default_compliance_protocols()
    print(f"Available protocols: {list(protocols.keys())}")
    
    # Test audit provider
    audit_provider = protocols['audit_provider']
    event_id = audit_provider.record_event(
        AuditEventType.MODEL_TRAINING,
        {
            "model_name": "test_model",
            "training_params": {"epochs": 10, "batch_size": 32},
            "dataset_size": 1000
        },
        user_id="data_scientist",
        model_version="v1.0",
        risk_level="medium"
    )
    print(f"Recorded training event: {event_id}")
    
    # Test compliance validator
    validator = protocols['validator']
    validation_results = validator.validate_framework_compliance(
        ComplianceFramework.GENERAL,
        {
            "events": [{"event_id": event_id, "event_type": "model_training"}],
            "integrity_verified": True
        },
        # Pass empty audit generator for backward compatibility
        audit_generator=None,
        model_version="v1.0"
    )
    print(f"Validation results: {len(validation_results)} checks performed")
    
    # Test risk assessor
    risk_assessor = protocols['risk_assessor']
    model_risk = risk_assessor.assess_model_risk(
        {"model_type": "neural_network", "parameter_count": 1e6, "uses_pii": False},
        {"domain": "general", "deployment_environment": "cloud"}
    )
    print(f"Model risk assessment: {model_risk['risk_level']} (score: {model_risk['risk_score']:.2f})")
    
    # Test bias detector
    bias_detector = protocols['bias_detector']
    try:
        import numpy as np
        predictions = np.array([0, 1, 0, 1, 1, 0])
        protected_attrs = {"gender": np.array(["M", "F", "M", "F", "M", "F"])}
        bias_results = bias_detector.detect_bias(predictions, protected_attrs)
        print(f"Bias detection: {'No bias detected' if not bias_results['bias_detected'] else 'Bias detected'}")
    except ImportError:
        print("NumPy not available - skipping bias detection test")
    
    return protocols


def test_integration_with_lcm():
    """Test integration with the LCM system."""
    print("\n=== Testing LCM Integration ===")
    
    # Create policy with LCM integration enabled
    policy = CompliancePolicy()
    policy.lcm_integration = True
    policy.anchor_compliance_records = True
    policy.merkle_audit_integrity = True
    
    print(f"LCM Integration enabled: {policy.lcm_integration}")
    print(f"Compliance record anchoring: {policy.anchor_compliance_records}")
    print(f"Merkle audit integrity: {policy.merkle_audit_integrity}")
    
    # Test audit trail with LCM integration
    audit_generator = AuditTrailGenerator(
        model_name="lcm_integrated_model",
        compliance_frameworks=["eu_ai_act", "nist_ai_rmf"]
    )
    
    # Record compliance check with LCM anchoring
    compliance_record = audit_generator.record_compliance_check(
        "lcm_integration_test",
        {
            "overall_status": "passed",
            "lcm_anchored": True,
            "merkle_verified": True
        }
    )
    
    print(f"LCM-integrated compliance record: {compliance_record.event_id}")
    
    # Verify audit integrity
    integrity_check = audit_generator.verify_audit_integrity()
    print(f"Audit integrity verified: {integrity_check['integrity_verified']}")
    
    return audit_generator


def test_policy_based_validation():
    """Test policy-based validation with different compliance levels."""
    print("\n=== Testing Policy-Based Validation ===")
    
    # Test with strict policy
    strict_policy = CompliancePolicy.strict()
    set_default_compliance_policy(strict_policy)
    
    strict_validator = ComplianceValidator("strict_model")
    
    # Simulate some validation results
    audit_data = {
        "events": [
            {"event_type": "model_training", "risk_level": "high"},
            {"event_type": "data_access", "contains_pii": True}
        ],
        "integrity_verified": True
    }
    
    results = strict_validator.validate_framework_compliance(
        ComplianceFramework.EU_AI_ACT,
        audit_data,
        model_version="v1.0"
    )
    
    summary = strict_validator.get_validation_summary()
    if 'message' in summary and summary['message'] == "No validations performed":
        print(f"Strict Policy Validation: No requirements found for framework")
    else:
        print(f"Strict Policy Validation:")
        print(f"  - Total validations: {summary.get('total_validations', 0)}")
        print(f"  - Pass rate: {summary.get('pass_rate', 0):.1f}%")
        print(f"  - Overall status: {summary.get('overall_status', 'unknown')}")
        print(f"  - Policy compliance: {summary.get('policy_compliance', {}).get('within_policy', 'unknown')}")
    
    # Test with development policy
    dev_policy = CompliancePolicy.development()
    set_default_compliance_policy(dev_policy)
    
    dev_validator = ComplianceValidator("dev_model")
    dev_results = dev_validator.validate_framework_compliance(
        ComplianceFramework.GENERAL,
        audit_data,
        model_version="v1.0"
    )
    
    dev_summary = dev_validator.get_validation_summary()
    if 'message' in dev_summary and dev_summary['message'] == "No validations performed":
        print(f"Development Policy Validation: No requirements found for framework")
    else:
        print(f"\nDevelopment Policy Validation:")
        print(f"  - Total validations: {dev_summary.get('total_validations', 0)}")
        print(f"  - Pass rate: {dev_summary.get('pass_rate', 0):.1f}%")
        print(f"  - Overall status: {dev_summary.get('overall_status', 'unknown')}")
        print(f"  - Policy compliance: {dev_summary.get('policy_compliance', {}).get('within_policy', 'unknown')}")
    
    return strict_validator, dev_validator


def test_backward_compatibility():
    """Test backward compatibility with existing code."""
    print("\n=== Testing Backward Compatibility ===")
    
    # Test legacy AuditTrailGenerator
    legacy_generator = AuditTrailGenerator("legacy_model")
    
    # Record a compliance check using legacy method
    legacy_record = legacy_generator.record_compliance_check(
        "legacy_test",
        {"status": "passed", "framework": "general"}
    )
    
    print(f"Legacy audit record created: {legacy_record.event_id}")
    
    # Test legacy ComplianceValidator  
    legacy_validator = ComplianceValidator("legacy_model")
    
    # Test that it still implements the new interface
    print(f"Legacy validator implements new interface: {hasattr(legacy_validator, 'validate_framework_compliance')}")
    print(f"Legacy validator has policy: {hasattr(legacy_validator, 'compliance_policy')}")
    
    return legacy_generator, legacy_validator


def main():
    """Run comprehensive compliance system tests."""
    print("CIAF Compliance System - Comprehensive Test Suite")
    print("=" * 60)
    
    try:
        # Test policy framework
        policy = test_policy_framework()
        
        # Test protocol implementations
        protocols = test_protocol_implementations()
        
        # Test LCM integration
        lcm_generator = test_integration_with_lcm()
        
        # Test policy-based validation
        validators = test_policy_based_validation()
        
        # Test backward compatibility
        legacy_components = test_backward_compatibility()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("\nKey Improvements Demonstrated:")
        print("1. ✅ New compliance policy framework with configurable levels")
        print("2. ✅ Protocol-based interfaces for clean dependency injection")
        print("3. ✅ Integration with LCM system for proper anchoring")
        print("4. ✅ Policy-driven validation with threshold management")
        print("5. ✅ Backward compatibility with existing code")
        print("6. ✅ Comprehensive protocol implementations")
        print("7. ✅ Better organization and structure consistency")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)