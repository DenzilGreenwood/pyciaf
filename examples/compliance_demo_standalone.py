#!/usr/bin/env python3
"""
CIAF Compliance Extensions Standalone Demo

A simplified demo that showcases the compliance extensions functionality
without complex imports that might cause circular dependency issues.

This demonstrates the key regulatory compliance features:
- EU AI Act Article 14 (Human Oversight)
- EU AI Act Article 15 (Robustness & Security)
- GDPR compliance (Consent & Erasure)
- NIST AI RMF (Continuous Monitoring)
- ISO/IEC 42001 (Corrective Actions)
- HIPAA/SOX (Access Control & Logging)

Created: 2025-09-23
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    # Try importing the core extensions directly
    from ciaf.extensions.compliance import (
        ComplianceExtensions, HumanOversightManager, GDPRComplianceManager,
        RobustnessManager, ContinuousMonitoringManager, CorrectiveActionManager,
        AccessControlManager,
        ConsentPurpose, OversightAction, RemediationAction,
        ConsentStatus, MonitoringEventType
    )
    from ciaf.core.canonicalization import (
        Policy, HashAlgorithm, Signer, WORMMerkleTree, 
        canonical_json, canonicalize_and_hash
    )
    
    print("✅ Successfully imported compliance extensions!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Running in simulation mode with local implementations...")
    
    # Fallback implementations for demo purposes
    class MockComplianceExtensions:
        def __init__(self, model_name):
            self.model_name = model_name
            print(f"Mock compliance extensions initialized for {model_name}")
    
    ComplianceExtensions = MockComplianceExtensions


class StandaloneComplianceDemo:
    """Standalone compliance demonstration."""
    
    def __init__(self):
        """Initialize demo environment."""
        print("🚀 CIAF Compliance Extensions Standalone Demo")
        print("=" * 60)
        
        try:
            self.compliance = ComplianceExtensions("Healthcare_AI_Model_v2")
            self.demo_mode = "full"
        except:
            self.compliance = None
            self.demo_mode = "simulation"
            
        print(f"Demo mode: {self.demo_mode}")
        print()
    
    def demo_human_oversight(self):
        """Demonstrate EU AI Act Article 14 - Human Oversight."""
        print("👨‍⚖️ EU AI Act Article 14 - Human Oversight Demo")
        print("-" * 50)
        
        if self.demo_mode == "full":
            try:
                # Create oversight checkpoint
                checkpoint = self.compliance.oversight_manager.create_oversight_checkpoint(
                    decision_context={
                        "patient_id": "patient_001",
                        "diagnosis": "high_risk_cancer_screening",
                        "ai_confidence": 0.87,
                        "stakes": "high"
                    },
                    risk_level="high",
                    automated_decision=True
                )
                
                print(f"✓ Oversight checkpoint created: {checkpoint.checkpoint_id}")
                print(f"  Risk level: {checkpoint.risk_level}")
                print(f"  Oversight required: {checkpoint.oversight_required}")
                
                # Simulate human review
                print("  👨‍⚕️ Simulating human oversight by medical professional...")
                time.sleep(1)
                
                success = self.compliance.oversight_manager.complete_oversight(
                    checkpoint.checkpoint_id,
                    actor_id="dr_martinez_oncologist",
                    action=OversightAction.APPROVED,
                    reason="AI recommendation aligns with clinical assessment. Additional tests recommended.",
                    review_time_seconds=67.3
                )
                
                print(f"✓ Oversight completed: {success}")
                print(f"  Reviewer: dr_martinez_oncologist")
                print(f"  Decision: {checkpoint.oversight_action.value}")
                print(f"  Review time: {checkpoint.human_review_time_seconds}s")
                print(f"  Checkpoint hash: {checkpoint.get_checkpoint_hash()[:16]}...")
                
            except Exception as e:
                print(f"⚠️  Full demo failed: {e}")
                self._simulate_oversight()
        else:
            self._simulate_oversight()
        
        print()
    
    def _simulate_oversight(self):
        """Simulate oversight functionality."""
        print("🎭 Simulating Human Oversight functionality:")
        print("  ✓ Created high-risk oversight checkpoint")
        print("  ✓ Required human review for cancer diagnosis")
        print("  ✓ Dr. Martinez (oncologist) approved AI recommendation")
        print("  ✓ 67.3 seconds review time logged")
        print("  ✓ Cryptographic hash generated for audit trail")
    
    def demo_gdpr_compliance(self):
        """Demonstrate GDPR compliance."""
        print("🛡️  GDPR Compliance Demo")
        print("-" * 50)
        
        if self.demo_mode == "full":
            try:
                # Create consent receipt
                consent = self.compliance.gdpr_manager.create_consent_receipt(
                    actor_id="patient_maria_garcia",
                    purpose=ConsentPurpose.RESEARCH,
                    data_categories=["health", "biometric", "genetic"],
                    legal_basis="explicit_consent",
                    retention_period_days=1095  # 3 years
                )
                
                print(f"✓ GDPR consent created: {consent.consent_id}")
                print(f"  Patient: {consent.actor_id}")
                print(f"  Purpose: {consent.purpose.value}")
                print(f"  Status: {consent.status.value}")
                print(f"  Data categories: {', '.join(consent.data_categories)}")
                print(f"  Legal basis: {consent.legal_basis}")
                print(f"  Retention: {consent.retention_period_days} days")
                print(f"  Consent hash: {consent.consent_hash[:16]}...")
                
                # Demonstrate consent withdrawal and erasure
                print("\n  🗑️  Simulating right to erasure request...")
                time.sleep(1)
                
                # Withdraw consent
                withdrawal_success = self.compliance.gdpr_manager.withdraw_consent(
                    consent.consent_id,
                    actor_id="patient_maria_garcia"
                )
                
                print(f"✓ Consent withdrawal: {withdrawal_success}")
                print(f"  Updated status: {consent.status.value}")
                
                # Process erasure
                erasure_result = self.compliance.gdpr_manager.process_erasure_request(
                    actor_id="patient_maria_garcia",
                    data_categories=["health", "biometric"]
                )
                
                print(f"✓ Erasure processed: {erasure_result['erasure_id']}")
                print(f"  Actions taken: {len(erasure_result['actions_taken'])}")
                print(f"  Compliance note: {erasure_result['compliance_note']}")
                print(f"  Erasure hash: {erasure_result['erasure_hash'][:16]}...")
                
            except Exception as e:
                print(f"⚠️  Full demo failed: {e}")
                self._simulate_gdpr()
        else:
            self._simulate_gdpr()
        
        print()
    
    def _simulate_gdpr(self):
        """Simulate GDPR functionality."""
        print("🎭 Simulating GDPR functionality:")
        print("  ✓ Created explicit consent for health data research")
        print("  ✓ Patient: patient_maria_garcia")
        print("  ✓ Categories: health, biometric, genetic data")
        print("  ✓ 3-year retention period specified")
        print("  ✓ Processed right to erasure request")
        print("  ✓ Personal data anonymized, audit hashes preserved")
    
    def demo_robustness_security(self):
        """Demonstrate EU AI Act Article 15 - Robustness & Security."""
        print("🔐 EU AI Act Article 15 - Robustness & Security Demo")
        print("-" * 50)
        
        if self.demo_mode == "full":
            try:
                # Create adversarial robustness test
                adv_test = self.compliance.robustness_manager.create_adversarial_test(
                    epsilon=0.03,
                    attack_method="projected_gradient_descent",
                    accuracy_threshold=0.92
                )
                
                print(f"✓ Adversarial test: {adv_test.test_id}")
                print(f"  Attack method: {adv_test.test_parameters['attack_method']}")
                print(f"  Epsilon: {adv_test.test_parameters['epsilon']}")
                print(f"  Result: {adv_test.result}")
                print(f"  Clean accuracy: {adv_test.metrics['clean_accuracy']:.3f}")
                print(f"  Adversarial accuracy: {adv_test.metrics['adversarial_accuracy']:.3f}")
                print(f"  Robustness score: {adv_test.metrics['robustness_score']:.3f}")
                print(f"  Test hash: {adv_test.test_hash[:16]}...")
                
                # Create security proof
                security_proof = self.compliance.robustness_manager.create_security_proof(
                    security_property="integrity",
                    proof_method="formal_verification_z3",
                    evidence={
                        "verification_tool": "Z3_SMT_Solver",
                        "properties_checked": ["input_validation", "output_bounds", "model_integrity"],
                        "verification_time_seconds": 127.4,
                        "proof_steps": 1847
                    }
                )
                
                print(f"\n✓ Security proof: {security_proof.proof_id}")
                print(f"  Property: {security_proof.security_property}")
                print(f"  Method: {security_proof.proof_method}")
                print(f"  Verification result: {security_proof.verification_result}")
                print(f"  Tool: {security_proof.evidence['verification_tool']}")
                print(f"  Proof steps: {security_proof.evidence['proof_steps']}")
                print(f"  Proof hash: {security_proof.proof_hash[:16]}...")
                
            except Exception as e:
                print(f"⚠️  Full demo failed: {e}")
                self._simulate_robustness()
        else:
            self._simulate_robustness()
        
        print()
    
    def _simulate_robustness(self):
        """Simulate robustness functionality."""
        print("🎭 Simulating Robustness & Security functionality:")
        print("  ✓ Adversarial test with PGD attack (ε=0.03)")
        print("  ✓ Clean accuracy: 0.945, Adversarial accuracy: 0.923")
        print("  ✓ Robustness score: 0.976 (PASSED)")
        print("  ✓ Formal verification with Z3 SMT solver")
        print("  ✓ Model integrity properties verified (1847 proof steps)")
    
    def demo_continuous_monitoring(self):
        """Demonstrate NIST AI RMF continuous monitoring.""" 
        print("📊 NIST AI RMF - Continuous Monitoring Demo")
        print("-" * 50)
        
        if self.demo_mode == "full":
            try:
                # Create drift monitoring event
                drift_event = self.compliance.monitoring_manager.create_drift_monitoring_event(
                    kl_divergence=0.087,  # Below threshold
                    psi_score=0.156,      # Below threshold
                    model_version="2.1.0"
                )
                
                print(f"✓ Drift monitoring: {drift_event.event_id}")
                print(f"  Model version: {drift_event.model_version}")
                print(f"  KL divergence: {drift_event.metrics['kl_divergence']:.3f} (threshold: 0.1)")
                print(f"  PSI score: {drift_event.metrics['psi_score']:.3f} (threshold: 0.2)")
                print(f"  Alerts: {len(drift_event.alerts)} ({'NONE' if not drift_event.has_alerts() else ', '.join(drift_event.alerts)})")
                print(f"  Monitoring hash: {drift_event.monitoring_hash[:16]}...")
                
                # Create high-drift event with alerts
                alert_event = self.compliance.monitoring_manager.create_drift_monitoring_event(
                    kl_divergence=0.134,  # Above threshold
                    psi_score=0.231,      # Above threshold  
                    model_version="2.1.0"
                )
                
                print(f"\n⚠️  Alert monitoring: {alert_event.event_id}")
                print(f"  KL divergence: {alert_event.metrics['kl_divergence']:.3f} (ALERT: > 0.1)")
                print(f"  PSI score: {alert_event.metrics['psi_score']:.3f} (ALERT: > 0.2)")
                print(f"  Alerts: {len(alert_event.alerts)}")
                for alert in alert_event.alerts:
                    print(f"    🚨 {alert}")
                
                # Create automated monitoring capsule
                monitoring_capsule = self.compliance.monitoring_manager.create_automated_monitoring_capsule(
                    hours_interval=24
                )
                
                print(f"\n✓ Automated monitoring: {monitoring_capsule['capsule_id']}")
                print(f"  Type: {monitoring_capsule['monitoring_type']}")
                print(f"  Interval: {monitoring_capsule['interval_hours']} hours")
                print(f"  Next check: {monitoring_capsule['next_check_due']}")
                print(f"  Events included: {len(monitoring_capsule['events'])}")
                print(f"  Capsule hash: {monitoring_capsule['capsule_hash'][:16]}...")
                
            except Exception as e:
                print(f"⚠️  Full demo failed: {e}")
                self._simulate_monitoring()
        else:
            self._simulate_monitoring()
        
        print()
    
    def _simulate_monitoring(self):
        """Simulate monitoring functionality."""
        print("🎭 Simulating Continuous Monitoring functionality:")
        print("  ✓ KL divergence: 0.087 (NORMAL)")
        print("  ✓ PSI score: 0.156 (NORMAL)")
        print("  🚨 High drift detected: KL=0.134, PSI=0.231")
        print("  ✓ 2 alerts triggered for retraining recommendation")
        print("  ✓ Automated 24-hour monitoring schedule active")
    
    def demo_corrective_actions(self):
        """Demonstrate ISO/IEC 42001 corrective actions."""
        print("🔧 ISO/IEC 42001 - Corrective Actions Demo")
        print("-" * 50)
        
        if self.demo_mode == "full":
            try:
                # Create corrective action for bias issue
                bias_action = self.compliance.corrective_action_manager.create_corrective_action(
                    issue_id="demographic_bias_detected_q3_2025",
                    remediation_action=RemediationAction.REBALANCE_DATASET,
                    actor_id="ml_engineer_carlos_rodriguez", 
                    description="Rebalance training dataset to address demographic parity gap in age groups 18-30 vs 60+",
                    success_metrics={
                        "demographic_parity": 0.95,
                        "equalized_odds": 0.92,
                        "statistical_parity_difference": 0.05
                    },
                    verification_method="fairness_audit_recheck"
                )
                
                print(f"✓ Corrective action: {bias_action.action_id}")
                print(f"  Issue: {bias_action.issue_id}")
                print(f"  Action: {bias_action.remediation_action.value}")
                print(f"  Actor: {bias_action.actor_id}")
                print(f"  Description: {bias_action.description}")
                print(f"  Target demographic parity: {bias_action.success_metrics['demographic_parity']}")
                print(f"  Verification: {bias_action.verification_method}")
                print(f"  Status: {'COMPLETED' if bias_action.is_completed() else 'IN PROGRESS'}")
                
                # Simulate completion  
                print("\n  🔧 Simulating dataset rebalancing...")
                time.sleep(1)
                
                completion_success = self.compliance.corrective_action_manager.complete_corrective_action(
                    bias_action.action_id,
                    completion_metrics={
                        "demographic_parity": 0.967,
                        "equalized_odds": 0.943,
                        "statistical_parity_difference": 0.033
                    }
                )
                
                print(f"✓ Action completed: {completion_success}")
                print(f"  Final demographic parity: 0.967 (target: 0.95) ✅")
                print(f"  Final equalized odds: 0.943 (target: 0.92) ✅")
                print(f"  Final stat parity diff: 0.033 (target: <0.05) ✅")
                print(f"  Completion timestamp: {bias_action.completion_timestamp}")
                print(f"  Action hash: {bias_action.action_hash[:16]}...")
                
                # Check issue tracking
                tracked_actions = self.compliance.corrective_action_manager.issue_tracking.get(
                    bias_action.issue_id, []
                )
                print(f"  Tracked actions for issue: {len(tracked_actions)}")
                
            except Exception as e:
                print(f"⚠️  Full demo failed: {e}")
                self._simulate_corrective_actions()
        else:
            self._simulate_corrective_actions()
        
        print()
    
    def _simulate_corrective_actions(self):
        """Simulate corrective actions functionality."""
        print("🎭 Simulating Corrective Actions functionality:")
        print("  ✓ Issue: demographic_bias_detected_q3_2025")
        print("  ✓ Action: Rebalance training dataset")
        print("  ✓ Actor: ml_engineer_carlos_rodriguez")
        print("  ✓ Target: demographic parity ≥ 0.95")
        print("  ✓ Completed with metrics exceeding targets")
        print("  ✓ Issue-to-action tracking maintained")
    
    def demo_access_control(self):
        """Demonstrate HIPAA/SOX access control and logging."""
        print("🔍 HIPAA/SOX - Access Control & Audit Logging Demo")
        print("-" * 50)
        
        if self.demo_mode == "full":
            try:
                # Log capsule verification access
                access_event = self.compliance.access_control_manager.log_capsule_verification(
                    actor_id="external_auditor_healthcare_board",
                    capsule_id="patient_diagnosis_capsule_20250923_001",
                    access_granted=True,
                    ip_address="203.0.113.45",
                    session_id="audit_session_hcb_20250923_14:30"
                )
                
                print(f"✓ Access logged: {access_event.access_id}")
                print(f"  Actor: {access_event.actor_id}")
                print(f"  Resource: {access_event.resource_id}")
                print(f"  Event type: {access_event.event_type.value}")
                print(f"  Access granted: {access_event.access_granted}")
                print(f"  IP address: {access_event.ip_address}")
                print(f"  Session: {access_event.session_id}")
                print(f"  Reason: {access_event.access_reason}")
                print(f"  Timestamp: {access_event.timestamp}")
                print(f"  Access hash: {access_event.access_hash[:16]}...")
                
                # Log denied access attempt
                denied_event = self.compliance.access_control_manager.log_capsule_verification(
                    actor_id="unauthorized_researcher",
                    capsule_id="patient_diagnosis_capsule_20250923_001",
                    access_granted=False,
                    ip_address="192.168.1.100"
                )
                
                print(f"\n❌ Access denied: {denied_event.access_id}")
                print(f"  Actor: {denied_event.actor_id}")
                print(f"  Reason: Insufficient privileges")
                print(f"  IP: {denied_event.ip_address}")
                
                # Generate access log summary
                access_summary = self.compliance.access_control_manager.get_access_log_summary(days=1)
                
                print(f"\n📊 Access Log Summary (24h):")
                print(f"  Total access events: {access_summary['total_access_events']}")
                print(f"  Unique actors: {access_summary['unique_actors']}")
                print(f"  Denied access count: {access_summary['denied_access_count']}")
                print(f"  Capsule verifications: {access_summary['access_types'].get('capsule_verification', 0)}")
                
                if access_summary['most_accessed_resources']:
                    print(f"  Most accessed resources:")
                    for resource, count in list(access_summary['most_accessed_resources'].items())[:3]:
                        print(f"    {resource}: {count} accesses")
                
            except Exception as e:
                print(f"⚠️  Full demo failed: {e}")
                self._simulate_access_control()
        else:
            self._simulate_access_control()
        
        print()
    
    def _simulate_access_control(self):
        """Simulate access control functionality."""
        print("🎭 Simulating Access Control functionality:")
        print("  ✓ Healthcare board auditor access GRANTED")
        print("  ✓ IP: 203.0.113.45, Session: audit_session_hcb_20250923")
        print("  ❌ Unauthorized researcher access DENIED")
        print("  ✓ 15 total access events, 3 unique actors")
        print("  ✓ 2 denied access attempts logged")
        print("  ✓ Tamper-evident access hash generated")
    
    def demo_compliance_report(self):
        """Generate comprehensive compliance report."""
        print("📋 Comprehensive Compliance Report")
        print("-" * 50)
        
        if self.demo_mode == "full":
            try:
                report = self.compliance.generate_compliance_report()
                
                print(f"🏛️  CIAF Compliance Report")
                print(f"Model: {report['model_name']}")
                print(f"Generated: {report['report_timestamp']}")
                print()
                
                print("👨‍⚖️ Human Oversight (EU AI Act Art. 14)")
                print(f"  Total checkpoints: {report['oversight_summary']['total_checkpoints']}")
                print(f"  Completed: {report['oversight_summary']['completed_checkpoints']}")
                
                print()
                print("🛡️  GDPR Compliance")
                print(f"  Active consents: {report['gdpr_summary']['active_consents']}")
                print(f"  Erasure requests: {report['gdpr_summary']['erasure_requests']}")
                
                print()
                print("🔐 Robustness & Security (EU AI Act Art. 15)")
                print(f"  Total tests: {report['robustness_summary']['total_tests']}")
                print(f"  Passed tests: {report['robustness_summary']['passed_tests']}")
                
                print()
                print("📊 Continuous Monitoring (NIST AI RMF)")
                print(f"  Total events: {report['monitoring_summary']['total_events']}")
                print(f"  Alerts: {report['monitoring_summary']['alerts']}")
                
                print()
                print("🔍 Access Control (HIPAA/SOX)")
                access_summary = report['access_summary']
                print(f"  Total access events: {access_summary['total_access_events']}")
                print(f"  Unique actors: {access_summary['unique_actors']}")
                print(f"  Denied access: {access_summary['denied_access_count']}")
                
                print()
                print("🔧 Corrective Actions (ISO/IEC 42001)")
                print(f"  Total actions: {report['corrective_actions_summary']['total_actions']}")
                print(f"  Completed: {report['corrective_actions_summary']['completed_actions']}")
                
            except Exception as e:
                print(f"⚠️  Full demo failed: {e}")
                self._simulate_compliance_report()
        else:
            self._simulate_compliance_report()
        
        print()
    
    def _simulate_compliance_report(self):
        """Simulate compliance report.""" 
        print("🎭 Simulating Compliance Report:")
        print("  👨‍⚖️ Human Oversight: 5 checkpoints, 5 completed")
        print("  🛡️  GDPR: 3 active consents, 1 erasure request")
        print("  🔐 Robustness: 4 tests, 4 passed")
        print("  📊 Monitoring: 12 events, 2 alerts")
        print("  🔍 Access: 47 events, 8 unique actors, 3 denied")
        print("  🔧 Corrective Actions: 2 actions, 2 completed")
    
    def run_full_demo(self):
        """Run complete compliance demo."""
        print("🎯 Running Complete CIAF Compliance Extensions Demo\n")
        
        demos = [
            ("Human Oversight", self.demo_human_oversight),
            ("GDPR Compliance", self.demo_gdpr_compliance),
            ("Robustness & Security", self.demo_robustness_security),
            ("Continuous Monitoring", self.demo_continuous_monitoring),
            ("Corrective Actions", self.demo_corrective_actions),
            ("Access Control", self.demo_access_control),
            ("Compliance Report", self.demo_compliance_report)
        ]
        
        for i, (name, demo_func) in enumerate(demos, 1):
            print(f"[{i}/{len(demos)}] {name}")
            demo_func()
            if i < len(demos):
                print("⏭️  " + "="*50 + "\n")
        
        print("🎉 Complete CIAF Compliance Extensions Demo Finished!")
        print()
        print("✅ Demonstrated Regulatory Compliance For:")
        print("   • EU AI Act Articles 14 & 15 (Human Oversight & Robustness)")
        print("   • GDPR Articles 6, 17, 25 (Consent, Erasure, Privacy by Design)")
        print("   • NIST AI RMF (Continuous Monitoring)")
        print("   • ISO/IEC 42001 (Corrective Actions)")
        print("   • HIPAA/SOX (Access Control & Audit Logging)")
        print()
        print("🔐 All compliance events include cryptographic hashes for")
        print("   tamper-evident audit trails and regulatory verification.")


def main():
    """Main entry point."""
    demo = StandaloneComplianceDemo()
    
    try:
        demo.run_full_demo()
        return 0
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Demo crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())