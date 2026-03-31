"""
Compliance Acceptance Tests for CIAF

These tests validate the non-bypassable invariants and regulatory compliance
requirements for EU AI Act, GDPR, NIST AI RMF, ISO/IEC 42001, HIPAA, and SOX.

Each test both asserts invariants and emits an auditable receipt.

Created: 2025-09-23
Author: Denzil James Greenwood
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone

import sys
import os

# Add the parent directory to the path so we can import ciaf modules
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

try:
    from ciaf.api.framework import CIAFFramework, ComplianceError
    from ciaf.core.canonicalization import Policy, Signer
    from ciaf.extensions.compliance import ConsentPurpose, OversightAction
    COMPLIANCE_EXTENSIONS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    COMPLIANCE_EXTENSIONS_AVAILABLE = False
    # Create mock classes for type checking
    CIAFFramework = None
    ComplianceError = Exception
    Policy = None
    Signer = None
    ConsentPurpose = None
    OversightAction = None

@pytest.mark.skipif(not COMPLIANCE_EXTENSIONS_AVAILABLE, reason="Compliance extensions not available")
class TestEUAIActArticle14HumanOversight:
    """Test EU AI Act Article 14 - Human Oversight requirements."""

    def setup_method(self):
        """Setup test environment."""
        self.policy = Policy(
            policy_id="eu_ai_act_test",
            schema_version="1.0.0",
            domain_labels=["high_risk", "healthcare", "ai_governance"],
        )
        self.signer = Signer("test_key")
        self.framework = CIAFFramework("CIAF_Test", self.policy, self.signer)

    def test_oversight_required_positive(self):
        """
        Positive test: Inference with oversight checkpoint should succeed.
        """
        # Create oversight checkpoint
        checkpoint = (
            self.framework.compliance.oversight_manager.create_oversight_checkpoint(
                decision_context={"test": "high_risk_inference"},
                risk_level="high",
                automated_decision=True,
            )
        )

        # Complete oversight
        success = self.framework.compliance.oversight_manager.complete_oversight(
            checkpoint.checkpoint_id,
            actor_id="human_reviewer_001",
            action=OversightAction.APPROVED,
            reason="Reviewed and approved for regulatory compliance",
            review_time_seconds=45.2,
        )

        assert success, "Oversight completion should succeed"
        assert checkpoint.is_complete(), "Checkpoint should be complete"

        # Create inference metadata with oversight
        inference_meta = {
            "model_id": "test_model_001",
            "inference_id": "test_inference_001",
            "input_hash": "sha256:abcd1234" + "0" * 56,
            "output_hash": "sha256:efgh5678" + "0" * 56,
            "oversight_required": True,
            "oversight_checkpoint": {
                "status": "approved",
                "oversight_actor_id": "human_reviewer_001",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": "Reviewed and approved",
            },
        }

        # Should succeed with proper oversight
        receipt = self.framework.commit_inference(inference_meta)

        assert receipt is not None, "Receipt should be generated"
        assert receipt.record_type.value == "inference", "Should be inference record"
        print(
            f"✓ Oversight test passed - Receipt: {receipt.get_receipt_hash()[:16]}..."
        )

    def test_oversight_required_negative(self):
        """
        Negative test: Inference with oversight_required=true but no checkpoint should fail.
        """
        inference_meta = {
            "model_id": "test_model_002",
            "inference_id": "test_inference_002",
            "input_hash": "sha256:abcd1234" + "0" * 56,
            "output_hash": "sha256:efgh5678" + "0" * 56,
            "oversight_required": True,
            # Missing oversight_checkpoint
        }

        with pytest.raises(
            ComplianceError, match="Human oversight required but not completed"
        ):
            self.framework.commit_inference(inference_meta)

        print(
            "✓ Negative oversight test passed - Correctly blocked inference without oversight"
        )

    def test_oversight_not_required(self):
        """
        Test: Inference with oversight_required=false should succeed without checkpoint.
        """
        inference_meta = {
            "model_id": "test_model_003",
            "inference_id": "test_inference_003",
            "input_hash": "sha256:abcd1234" + "0" * 56,
            "output_hash": "sha256:efgh5678" + "0" * 56,
            "oversight_required": False,
        }

        receipt = self.framework.commit_inference(inference_meta)

        assert (
            receipt is not None
        ), "Receipt should be generated for non-oversight inference"
        print(
            f"✓ Non-oversight inference test passed - Receipt: {receipt.get_receipt_hash()[:16]}..."
        )


@pytest.mark.skipif(not COMPLIANCE_EXTENSIONS_AVAILABLE, reason="Compliance extensions not available")
class TestEUAIActArticle15Robustness:
    """Test EU AI Act Article 15 - Robustness and Cybersecurity requirements."""

    def setup_method(self):
        """Setup test environment."""
        self.policy = Policy(
            policy_id="eu_ai_act_robustness_test",
            schema_version="1.0.0",
            domain_labels=["high_risk", "critical", "ai_governance"],
        )
        self.signer = Signer("robustness_test_key")
        self.framework = CIAFFramework("CIAF_Robustness_Test", self.policy, self.signer)

    def test_robustness_required_positive(self):
        """
        Positive test: Model with robustness tests should succeed.
        """
        # Create robustness test
        robustness_test = (
            self.framework.compliance.robustness_manager.create_adversarial_test(
                epsilon=0.03, attack_method="pgd", accuracy_threshold=0.9
            )
        )

        assert robustness_test.result == "passed", "Robustness test should pass"

        # Create security proof
        security_proof = (
            self.framework.compliance.robustness_manager.create_security_proof(
                security_property="integrity",
                proof_method="formal_verification",
                evidence={
                    "verification_tool": "test_tool",
                    "properties_verified": ["integrity"],
                },
            )
        )

        assert security_proof.verification_result, "Security proof should verify"

        # Create model metadata with robustness attestations
        model_meta = {
            "model_id": "robust_model_001",
            "model_hash": "sha256:model123" + "0" * 56,
            "parameters_hash": "sha256:params12" + "0" * 56,
            "compliance_extensions": {
                "robustness_tests": [
                    {
                        "test_id": robustness_test.test_id,
                        "test_type": robustness_test.test_type,
                        "result": robustness_test.result,
                        "metrics": robustness_test.metrics,
                        "test_parameters": robustness_test.test_parameters,
                    }
                ],
                "security_proofs": [
                    {
                        "proof_id": security_proof.proof_id,
                        "security_property": security_proof.security_property,
                        "verification_result": security_proof.verification_result,
                        "proof_method": security_proof.proof_method,
                        "evidence": security_proof.evidence,
                    }
                ],
            },
        }

        receipt = self.framework.commit_model_checkpoint(model_meta)

        assert receipt is not None, "Receipt should be generated"
        assert receipt.record_type.value == "model", "Should be model record"
        print(
            f"✓ Robustness test passed - Receipt: {receipt.get_receipt_hash()[:16]}..."
        )

    def test_robustness_required_negative(self):
        """
        Negative test: High-risk model without robustness tests should fail.
        """
        model_meta = {
            "model_id": "unsafe_model_001",
            "model_hash": "sha256:model456" + "0" * 56,
            "parameters_hash": "sha256:params45" + "0" * 56,
            # Missing robustness tests for high-risk domain
        }

        with pytest.raises(
            ValueError, match="EU AI Act Article 15 requires robustness testing"
        ):
            self.framework.commit_model_checkpoint(model_meta)

        print(
            "✓ Negative robustness test passed - Correctly blocked model without robustness testing"
        )


@pytest.mark.skipif(not COMPLIANCE_EXTENSIONS_AVAILABLE, reason="Compliance extensions not available")
class TestGDPRCompliance:
    """Test GDPR compliance requirements."""

    def setup_method(self):
        """Setup test environment."""
        self.policy = Policy(
            policy_id="gdpr_test",
            schema_version="1.0.0",
            domain_labels=["gdpr", "personal_data", "eu"],
        )
        self.signer = Signer("gdpr_test_key")
        self.framework = CIAFFramework("CIAF_GDPR_Test", self.policy, self.signer)

    def test_consent_required_positive(self):
        """
        Positive test: Dataset with personal data and consent should succeed.
        """
        # Create consent receipt
        consent = self.framework.compliance.gdpr_manager.create_consent_receipt(
            actor_id="data_subject_001",
            purpose=ConsentPurpose.TRAINING,
            data_categories=["personal", "biometric"],
            legal_basis="consent",
        )

        assert consent.status.value == "given", "Consent should be given"

        # Create dataset metadata with consent
        dataset_meta = {
            "dataset_id": "personal_dataset_001",
            "dataset_hash": "sha256:dataset12" + "0" * 56,
            "data_categories": ["personal", "biometric"],
            "compliance_extensions": {
                "consent_receipts": [
                    {
                        "consent_id": consent.consent_id,
                        "actor_id": consent.actor_id,
                        "purpose": consent.purpose.value,
                        "status": consent.status.value,
                        "data_categories": consent.data_categories,
                        "legal_basis": consent.legal_basis,
                        "consent_hash": consent.consent_hash,
                    }
                ]
            },
        }

        receipt = self.framework.commit_dataset_record(dataset_meta)

        assert receipt is not None, "Receipt should be generated"
        assert receipt.record_type.value == "dataset", "Should be dataset record"
        print(
            f"✓ GDPR consent test passed - Receipt: {receipt.get_receipt_hash()[:16]}..."
        )

    def test_consent_required_negative(self):
        """
        Negative test: Personal data without consent should fail.
        """
        dataset_meta = {
            "dataset_id": "personal_dataset_002",
            "dataset_hash": "sha256:dataset34" + "0" * 56,
            "data_categories": ["personal", "sensitive"],
            # Missing consent receipts
        }

        with pytest.raises(
            ValueError, match="GDPR consent required for personal data processing"
        ):
            self.framework.commit_dataset_record(dataset_meta)

        print(
            "✓ Negative GDPR test passed - Correctly blocked personal data without consent"
        )

    def test_erasure_request(self):
        """
        Test GDPR Article 17 right to erasure.
        """
        # First create a consent and dataset
        consent = self.framework.compliance.gdpr_manager.create_consent_receipt(
            actor_id="erasure_subject_001",
            purpose=ConsentPurpose.ANALYTICS,
            data_categories=["personal"],
        )

        # Process erasure request
        erasure_result = self.framework.compliance.gdpr_manager.process_erasure_request(
            actor_id="erasure_subject_001", data_categories=["personal"]
        )

        assert (
            erasure_result["actor_id"] == "erasure_subject_001"
        ), "Erasure should be for correct actor"
        assert (
            len(erasure_result["actions_taken"]) > 0
        ), "Erasure actions should be taken"

        # Verify consent is withdrawn
        updated_consent = self.framework.compliance.gdpr_manager.consent_receipts[
            consent.consent_id
        ]
        assert (
            updated_consent.status.value == "withdrawn"
        ), "Consent should be withdrawn after erasure"

        print(
            f"✓ GDPR erasure test passed - Erasure ID: {erasure_result['erasure_id']}"
        )


@pytest.mark.skipif(not COMPLIANCE_EXTENSIONS_AVAILABLE, reason="Compliance extensions not available")
class TestNISTAIRMFMonitoring:
    """Test NIST AI RMF continuous monitoring requirements."""

    def setup_method(self):
        """Setup test environment."""
        self.policy = Policy(
            policy_id="nist_ai_rmf_test",
            schema_version="1.0.0",
            domain_labels=["nist_ai_rmf", "continuous_monitoring", "us"],
        )
        self.signer = Signer("nist_test_key")
        self.framework = CIAFFramework("CIAF_NIST_Test", self.policy, self.signer)

    def test_continuous_monitoring(self):
        """
        Test continuous monitoring event creation and anchoring.
        """
        # Create monitoring event
        drift_event = (
            self.framework.compliance.monitoring_manager.create_drift_monitoring_event(
                kl_divergence=0.08,  # Below threshold
                psi_score=0.15,  # Below threshold
                model_version="1.0.0",
            )
        )

        assert not drift_event.has_alerts(), "Should not have alerts for low drift"

        # Create monitoring event with alerts
        alert_event = (
            self.framework.compliance.monitoring_manager.create_drift_monitoring_event(
                kl_divergence=0.12,  # Above threshold
                psi_score=0.25,  # Above threshold
                model_version="1.0.0",
            )
        )

        assert alert_event.has_alerts(), "Should have alerts for high drift"
        assert len(alert_event.alerts) == 2, "Should have 2 alerts"

        # Create automated monitoring capsule
        monitoring_capsule = self.framework.compliance.monitoring_manager.create_automated_monitoring_capsule(
            hours_interval=24
        )

        assert monitoring_capsule["monitoring_type"] == "automated_compliance_check"
        assert len(monitoring_capsule["events"]) > 0, "Should contain monitoring events"

        print(
            f"✓ NIST monitoring test passed - Capsule: {monitoring_capsule['capsule_id']}"
        )


@pytest.mark.skipif(not COMPLIANCE_EXTENSIONS_AVAILABLE, reason="Compliance extensions not available")
class TestISO42001CorrectiveActions:
    """Test ISO/IEC 42001 corrective action requirements."""

    def setup_method(self):
        """Setup test environment."""
        self.policy = Policy(
            policy_id="iso_42001_test",
            schema_version="1.0.0",
            domain_labels=["iso_42001", "corrective_actions", "continuous_improvement"],
        )
        self.signer = Signer("iso_test_key")
        self.framework = CIAFFramework("CIAF_ISO_Test", self.policy, self.signer)

    def test_corrective_action_loop(self):
        """
        Test create issue → corrective action → completion cycle.
        """
        from ciaf.extensions.compliance import RemediationAction

        issue_id = "bias_detected_001"

        # Create corrective action
        action = self.framework.compliance.corrective_action_manager.create_corrective_action(
            issue_id=issue_id,
            remediation_action=RemediationAction.REBALANCE_DATASET,
            actor_id="ml_engineer_001",
            description="Rebalance training dataset to address demographic bias",
            success_metrics={"demographic_parity": 0.9, "equalized_odds": 0.85},
            verification_method="bias_audit",
        )

        assert not action.is_completed(), "Action should not be completed initially"

        # Complete corrective action
        completion_success = self.framework.compliance.corrective_action_manager.complete_corrective_action(
            action.action_id,
            completion_metrics={"demographic_parity": 0.92, "equalized_odds": 0.87},
        )

        assert completion_success, "Action completion should succeed"
        assert action.is_completed(), "Action should be completed"

        # Verify issue tracking
        tracked_actions = (
            self.framework.compliance.corrective_action_manager.issue_tracking.get(
                issue_id, []
            )
        )
        assert action.action_id in tracked_actions, "Action should be tracked for issue"

        print(f"✓ ISO 42001 corrective action test passed - Action: {action.action_id}")


@pytest.mark.skipif(not COMPLIANCE_EXTENSIONS_AVAILABLE, reason="Compliance extensions not available")
class TestHIPAASOXAccessLogging:
    """Test HIPAA/SOX access control and audit logging requirements."""

    def setup_method(self):
        """Setup test environment."""
        self.policy = Policy(
            policy_id="hipaa_sox_test",
            schema_version="1.0.0",
            domain_labels=["hipaa", "sox", "access_control", "audit_logging"],
        )
        self.signer = Signer("access_test_key")
        self.framework = CIAFFramework("CIAF_Access_Test", self.policy, self.signer)

    def test_capsule_verification_logging(self):
        """
        Test that capsule verification creates audit logs.
        """
        # First create a capsule to verify
        test_capsule_id = "test_capsule_001"

        # Log capsule verification access
        access_event = (
            self.framework.compliance.access_control_manager.log_capsule_verification(
                actor_id="auditor_001",
                capsule_id=test_capsule_id,
                access_granted=True,
                ip_address="192.168.1.100",
                session_id="session_12345",
            )
        )

        assert access_event.access_granted, "Access should be granted"
        assert access_event.actor_id == "auditor_001", "Actor should be logged"
        assert access_event.resource_id == test_capsule_id, "Resource should be logged"

        # Get access log summary
        access_summary = (
            self.framework.compliance.access_control_manager.get_access_log_summary(
                days=1
            )
        )

        assert access_summary["total_access_events"] > 0, "Should have access events"
        assert (
            "capsule_verification" in access_summary["access_types"]
        ), "Should track capsule verification"

        print(
            f"✓ HIPAA/SOX access logging test passed - Events: {access_summary['total_access_events']}"
        )

    def test_access_denied_logging(self):
        """
        Test that denied access is properly logged.
        """
        # Log denied access
        denied_event = (
            self.framework.compliance.access_control_manager.log_capsule_verification(
                actor_id="unauthorized_user",
                capsule_id="restricted_capsule_001",
                access_granted=False,
                ip_address="10.0.0.50",
            )
        )

        assert not denied_event.access_granted, "Access should be denied"

        # Verify denied access is tracked
        access_summary = (
            self.framework.compliance.access_control_manager.get_access_log_summary(
                days=1
            )
        )
        assert access_summary["denied_access_count"] > 0, "Should track denied access"

        print(
            f"✓ Access denial logging test passed - Denied count: {access_summary['denied_access_count']}"
        )


@pytest.mark.skipif(not COMPLIANCE_EXTENSIONS_AVAILABLE, reason="Compliance extensions not available")
class TestComplianceIntegration:
    """Integration tests for complete compliance workflow."""

    def setup_method(self):
        """Setup test environment."""
        self.policy = Policy(
            policy_id="integration_test",
            schema_version="1.0.0",
            domain_labels=["high_risk", "healthcare", "gdpr", "nist_ai_rmf"],
        )
        self.signer = Signer("integration_test_key")
        self.framework = CIAFFramework(
            "CIAF_Integration_Test", self.policy, self.signer
        )

    def test_end_to_end_compliance_workflow(self):
        """
        Test complete workflow: dataset → model → inference with full compliance.
        """
        # Step 1: Create dataset with GDPR consent
        consent = self.framework.compliance.gdpr_manager.create_consent_receipt(
            actor_id="patient_001",
            purpose=ConsentPurpose.RESEARCH,
            data_categories=["health", "biometric"],
        )

        dataset_meta = {
            "dataset_id": "healthcare_dataset_001",
            "dataset_hash": "sha256:health123" + "0" * 56,
            "data_categories": ["health", "biometric"],
            "compliance_extensions": {
                "consent_receipts": [
                    dict(
                        consent_id=consent.consent_id,
                        actor_id=consent.actor_id,
                        purpose=consent.purpose.value,
                        status=consent.status.value,
                        data_categories=consent.data_categories,
                        legal_basis=consent.legal_basis,
                        consent_hash=consent.consent_hash,
                    )
                ]
            },
        }

        dataset_receipt = self.framework.commit_dataset_record(dataset_meta)

        # Step 2: Create model with robustness testing
        robustness_test = (
            self.framework.compliance.robustness_manager.create_adversarial_test(
                epsilon=0.01, attack_method="fgsm"
            )
        )

        model_meta = {
            "model_id": "healthcare_model_001",
            "model_hash": "sha256:model789" + "0" * 56,
            "parameters_hash": "sha256:params78" + "0" * 56,
            "compliance_extensions": {
                "robustness_tests": [
                    dict(
                        test_id=robustness_test.test_id,
                        test_type=robustness_test.test_type,
                        result=robustness_test.result,
                        metrics=robustness_test.metrics,
                        test_parameters=robustness_test.test_parameters,
                    )
                ]
            },
        }

        model_receipt = self.framework.commit_model_checkpoint(model_meta)

        # Step 3: Create inference with human oversight
        checkpoint = (
            self.framework.compliance.oversight_manager.create_oversight_checkpoint(
                decision_context={"patient_diagnosis": True}, risk_level="high"
            )
        )

        self.framework.compliance.oversight_manager.complete_oversight(
            checkpoint.checkpoint_id,
            actor_id="doctor_001",
            action=OversightAction.APPROVED,
            reason="Medical professional approval",
        )

        inference_meta = {
            "model_id": "healthcare_model_001",
            "inference_id": "diagnosis_001",
            "input_hash": "sha256:input123" + "0" * 56,
            "output_hash": "sha256:output12" + "0" * 56,
            "oversight_required": True,
            "oversight_checkpoint": dict(
                status="approved",
                oversight_actor_id="doctor_001",
                timestamp=datetime.now(timezone.utc).isoformat(),
                reason="Medical professional approval",
            ),
            "compliance_extensions": {
                "consent_receipts": [
                    dict(
                        consent_id=consent.consent_id,
                        actor_id=consent.actor_id,
                        purpose="inference",
                        status=consent.status.value,
                    )
                ]
            },
        }

        inference_receipt = self.framework.commit_inference(inference_meta)

        # Step 4: Generate proof capsule
        proof_capsule = self.framework.materialize_proof_capsule("diagnosis_001")

        # Step 5: Generate compliance report
        compliance_report = self.framework.compliance.generate_compliance_report()

        # Assertions
        assert dataset_receipt is not None, "Dataset receipt should be generated"
        assert model_receipt is not None, "Model receipt should be generated"
        assert inference_receipt is not None, "Inference receipt should be generated"
        assert proof_capsule is not None, "Proof capsule should be generated"
        assert compliance_report is not None, "Compliance report should be generated"

        # Verify compliance report contents
        assert (
            compliance_report["gdpr_summary"]["active_consents"] > 0
        ), "Should have active consents"
        assert (
            compliance_report["robustness_summary"]["total_tests"] > 0
        ), "Should have robustness tests"
        assert (
            compliance_report["oversight_summary"]["completed_checkpoints"] > 0
        ), "Should have completed oversight"

        print("✓ End-to-end compliance workflow test passed")
        print(f"  Dataset receipt: {dataset_receipt.get_receipt_hash()[:16]}...")
        print(f"  Model receipt: {model_receipt.get_receipt_hash()[:16]}...")
        print(f"  Inference receipt: {inference_receipt.get_receipt_hash()[:16]}...")
        print(
            f"  Proof capsule hash: {proof_capsule['verification']['capsule_hash'][:16]}..."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
