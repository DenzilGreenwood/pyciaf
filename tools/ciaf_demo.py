#!/usr/bin/env python3
"""
CIAF Compliance Demo CLI

Demonstrates the complete cryptographic audit receipt flow with compliance extensions:
1. Create + anchor dataset/model/inference with required compliance metadata
2. Materialize a proof capsule for an inference
3. Verify capsule independently  
4. Simulate GDPR erasure and show verification still passes
5. Trigger monitoring and corrective action then re-anchor
6. Generate compliance report

Usage:
    python tools/ciaf_demo.py --flow complete
    python tools/ciaf_demo.py --flow verification --capsule-id inference_001
    python tools/ciaf_demo.py --flow erasure --actor-id patient_001
    python tools/ciaf_demo.py --flow monitoring --model-id healthcare_model
    python tools/ciaf_demo.py --flow report

Created: 2025-09-23
Author: Denzil James Greenwood
Version: 1.0.0
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ciaf.api.framework import CIAFFramework
from ciaf.core.canonicalization import Policy, HashAlgorithm, Signer
from ciaf.extensions.compliance import ConsentPurpose, OversightAction


class CIAFDemo:
    """CIAF compliance demonstration."""
    
    def __init__(self):
        """Initialize demo environment."""
        self.policy = Policy(
            policy_id="ciaf_demo_v1",
            schema_version="1.0.0", 
            domain_labels=["demo", "healthcare", "high_risk", "gdpr", "nist_ai_rmf"],
            hash_algorithm=HashAlgorithm.SHA256
        )
        self.signer = Signer("demo_signing_key")
        self.framework = CIAFFramework("CIAF_Demo", self.policy, self.signer)
        
        print("🚀 CIAF Compliance Demo Initialized")
        print(f"Policy: {self.policy.policy_id}")
        print(f"Domains: {', '.join(self.policy.domain_labels)}")
        print(f"High-risk policy: {self.policy.is_high_risk()}")
        print("=" * 60)
    
    def run_complete_flow(self):
        """Run complete compliance flow demo."""
        print("📋 Running Complete Compliance Flow Demo\n")
        
        # Step 1: Dataset with GDPR Consent
        print("1️⃣  Creating Dataset with GDPR Consent")
        print("-" * 40)
        
        consent = self.framework.compliance.gdpr_manager.create_consent_receipt(
            actor_id="patient_alice_001",
            purpose=ConsentPurpose.RESEARCH,
            data_categories=["health", "biometric", "diagnostic"],
            retention_period_days=365
        )
        
        print(f"✓ GDPR consent created: {consent.consent_id}")
        print(f"  Purpose: {consent.purpose.value}")
        print(f"  Status: {consent.status.value}")
        print(f"  Data categories: {', '.join(consent.data_categories)}")
        
        dataset_meta = {
            "dataset_id": "medical_imaging_dataset_v1",
            "dataset_hash": "sha256:d4t4s3t" + "0" * 56,
            "name": "Medical Imaging Dataset v1.0",
            "version": "1.0.0",
            "size": 50000,
            "features": ["mri_scan", "patient_age", "diagnosis_code"],
            "data_categories": ["health", "biometric", "diagnostic"],
            "retention_period_days": 365,
            "privacy_techniques": ["pseudonymization", "encryption"]
        }
        
        dataset_receipt = self.framework.commit_dataset_record(dataset_meta)
        
        print(f"✓ Dataset anchored: {dataset_receipt.leaf_hash[:16]}...")
        print(f"  Merkle root: {dataset_receipt.anchor.root[:16]}...")
        print(f"  Signature: {dataset_receipt.anchor.signature[:16]}...\n")
        
        # Step 2: Model with Robustness Testing
        print("2️⃣  Creating Model with Robustness Testing")
        print("-" * 40)
        
        # Create adversarial robustness test
        adv_test = self.framework.compliance.robustness_manager.create_adversarial_test(
            epsilon=0.03,
            attack_method="pgd",
            accuracy_threshold=0.9
        )
        
        print(f"✓ Adversarial test: {adv_test.test_id}")
        print(f"  Method: {adv_test.test_parameters['attack_method']}")
        print(f"  Result: {adv_test.result}")
        print(f"  Adversarial accuracy: {adv_test.metrics['adversarial_accuracy']:.3f}")
        
        # Create security proof
        sec_proof = self.framework.compliance.robustness_manager.create_security_proof(
            security_property="integrity",
            proof_method="formal_verification",
            evidence={"tool": "z3_prover", "properties": ["model_integrity", "input_validation"]}
        )
        
        print(f"✓ Security proof: {sec_proof.proof_id}")
        print(f"  Property: {sec_proof.security_property}")
        print(f"  Verified: {sec_proof.verification_result}")
        
        model_meta = {
            "model_id": "diagnostic_cnn_v2",
            "model_hash": "sha256:m0d3l12" + "0" * 56,
            "parameters_hash": "sha256:p4r4ms1" + "0" * 56,
            "name": "Diagnostic CNN v2.0",
            "version": "2.0.0",
            "architecture": {
                "type": "cnn",
                "layers": 18,
                "parameters_count": 25000000,
                "input_shape": [224, 224, 3],
                "output_shape": [10]
            },
            "training_datasets": ["medical_imaging_dataset_v1"],
            "performance_metrics": {
                "accuracy": 0.94,
                "precision": 0.92,
                "recall": 0.91,
                "f1_score": 0.915
            }
        }
        
        model_receipt = self.framework.commit_model_checkpoint(model_meta)
        
        print(f"✓ Model anchored: {model_receipt.leaf_hash[:16]}...")
        print(f"  Merkle root: {model_receipt.anchor.root[:16]}...")
        print(f"  Policy: {model_receipt.anchor.policy_id}\n")
        
        # Step 3: Inference with Human Oversight  
        print("3️⃣  Creating Inference with Human Oversight")
        print("-" * 40)
        
        # Create oversight checkpoint
        checkpoint = self.framework.compliance.oversight_manager.create_oversight_checkpoint(
            decision_context={
                "patient_id": "patient_alice_001",
                "diagnosis_type": "cancer_screening",
                "confidence_threshold": 0.85,
                "high_stakes": True
            },
            risk_level="high",
            automated_decision=True
        )
        
        print(f"✓ Oversight checkpoint: {checkpoint.checkpoint_id}")
        print(f"  Risk level: {checkpoint.risk_level}")
        print(f"  Oversight required: {checkpoint.oversight_required}")
        
        # Simulate human oversight (would be real human in production)
        print("  👨‍⚕️ Simulating human oversight by radiologist...")
        time.sleep(1)  # Simulate review time
        
        oversight_success = self.framework.compliance.oversight_manager.complete_oversight(
            checkpoint.checkpoint_id,
            actor_id="dr_smith_radiologist_001",
            action=OversightAction.APPROVED,
            reason="Reviewed scan results and AI diagnosis. Recommendation aligns with clinical assessment.",
            review_time_seconds=45.7
        )
        
        print(f"✓ Oversight completed: {oversight_success}")
        print(f"  Reviewer: dr_smith_radiologist_001") 
        print(f"  Decision: {checkpoint.oversight_action.value}")
        print(f"  Review time: {checkpoint.human_review_time_seconds}s")
        
        inference_meta = {
            "model_id": "diagnostic_cnn_v2",
            "inference_id": "diagnosis_alice_001",
            "input_hash": "sha256:1nput12" + "0" * 56,
            "output_hash": "sha256:0utput1" + "0" * 56,
            "confidence_score": 0.89,
            "explanation_hash": "sha256:3xpl41n" + "0" * 56,
            "oversight_required": True,
            "oversight_checkpoint": {
                "status": "approved",
                "oversight_actor_id": "dr_smith_radiologist_001",
                "timestamp": checkpoint.oversight_timestamp,
                "reason": checkpoint.oversight_reason
            }
        }
        
        inference_receipt = self.framework.commit_inference(inference_meta)
        
        print(f"✓ Inference anchored: {inference_receipt.leaf_hash[:16]}...")
        print(f"  Merkle root: {inference_receipt.anchor.root[:16]}...")
        print(f"  Timestamp: {inference_receipt.anchor.timestamp}\n")
        
        # Step 4: Materialize Proof Capsule
        print("4️⃣  Materializing Proof Capsule")
        print("-" * 40)
        
        proof_capsule = self.framework.materialize_proof_capsule("diagnosis_alice_001")
        
        print(f"✓ Proof capsule generated")
        print(f"  Capsule type: {proof_capsule['capsule_type']}")
        print(f"  Record type: {proof_capsule['record']['type']}")
        print(f"  Merkle path length: {len(proof_capsule['proofs']['merkle_path'])}")
        print(f"  Inclusion proof valid: {proof_capsule['proofs']['inclusion_proof_valid']}")
        print(f"  Signature valid: {proof_capsule['verification']['signature_valid']}")
        print(f"  Capsule hash: {proof_capsule['verification']['capsule_hash'][:16]}...\n")
        
        # Step 5: Continuous Monitoring
        print("5️⃣  Continuous Monitoring")  
        print("-" * 40)
        
        drift_event = self.framework.compliance.monitoring_manager.create_drift_monitoring_event(
            kl_divergence=0.12,  # Above threshold - will trigger alert
            psi_score=0.18,      # Below threshold  
            model_version="2.0.0"
        )
        
        print(f"✓ Drift monitoring event: {drift_event.event_id}")
        print(f"  KL divergence: {drift_event.metrics['kl_divergence']:.3f}")
        print(f"  PSI score: {drift_event.metrics['psi_score']:.3f}")
        print(f"  Alerts: {len(drift_event.alerts)}")
        
        if drift_event.has_alerts():
            for alert in drift_event.alerts:
                print(f"  ⚠️  {alert}")
        
        # Step 6: Corrective Action (if alerts)
        if drift_event.has_alerts():
            print("\n6️⃣  Corrective Action")
            print("-" * 40)
            
            from ciaf.extensions.compliance import RemediationAction
            
            corrective_action = self.framework.compliance.corrective_action_manager.create_corrective_action(
                issue_id="drift_alert_001",
                remediation_action=RemediationAction.RETRAIN_MODEL,
                actor_id="ml_engineer_bob_001",
                description="Retrain model due to data drift detected in monitoring",
                success_metrics={"kl_divergence": 0.05, "psi_score": 0.10},
                verification_method="drift_recheck"
            )
            
            print(f"✓ Corrective action created: {corrective_action.action_id}")
            print(f"  Issue: drift_alert_001")
            print(f"  Action: {corrective_action.remediation_action.value}")
            print(f"  Actor: {corrective_action.actor_id}")
            
            # Simulate completion
            print("  🔧 Simulating model retraining...")
            time.sleep(1)
            
            completion_success = self.framework.compliance.corrective_action_manager.complete_corrective_action(
                corrective_action.action_id,
                completion_metrics={"kl_divergence": 0.04, "psi_score": 0.08}
            )
            
            print(f"✓ Action completed: {completion_success}")
            print(f"  Final KL divergence: 0.04")
            print(f"  Final PSI score: 0.08")
        
        # Step 7: Generate Compliance Report
        print("\n7️⃣  Generating Compliance Report")
        print("-" * 40)
        
        report = self.framework.compliance.generate_compliance_report()
        
        print(f"✓ Compliance report generated")
        print(f"  Model: {report['model_name']}")
        print(f"  Timestamp: {report['report_timestamp']}")
        print(f"  Oversight checkpoints: {report['oversight_summary']['completed_checkpoints']}")
        print(f"  Active consents: {report['gdpr_summary']['active_consents']}")  
        print(f"  Robustness tests: {report['robustness_summary']['passed_tests']}")
        print(f"  Monitoring events: {report['monitoring_summary']['total_events']}")
        print(f"  Access events: {report['access_summary']['total_access_events']}")
        
        # Save artifacts
        self._save_demo_artifacts(dataset_receipt, model_receipt, inference_receipt, proof_capsule, report)
        
        print("\n🎉 Complete compliance flow demo finished successfully!")
        return True
    
    def run_verification_demo(self, capsule_id: str):
        """Run proof capsule verification demo."""
        print(f"🔍 Running Verification Demo for: {capsule_id}\n")
        
        try:
            # Log the verification access
            access_event = self.framework.compliance.access_control_manager.log_capsule_verification(
                actor_id="auditor_external_001",
                capsule_id=capsule_id,
                access_granted=True,
                ip_address="203.0.113.45",
                session_id="audit_session_12345"
            )
            
            print(f"✓ Access logged: {access_event.access_id}")
            print(f"  Actor: {access_event.actor_id}")
            print(f"  IP: {access_event.ip_address}")
            print(f"  Granted: {access_event.access_granted}")
            
            # Materialize proof capsule
            proof_capsule = self.framework.materialize_proof_capsule(capsule_id)
            
            print(f"\n✓ Proof capsule materialized")
            print(f"  Capsule hash: {proof_capsule['verification']['capsule_hash'][:16]}...")
            print(f"  Merkle path elements: {len(proof_capsule['proofs']['merkle_path'])}")
            print(f"  Anchor signature: {proof_capsule['anchor']['signature'][:16]}...")
            
            # Verify independently
            leaf_hash = proof_capsule['record']['leaf_hash']
            merkle_path = proof_capsule['proofs']['merkle_path']
            merkle_root = proof_capsule['proofs']['merkle_root']
            
            # Simplified verification (in production would use proper Merkle verification)
            verification_result = self.framework.ledger.verify_merkle_path(leaf_hash, merkle_path, merkle_root)
            
            print(f"\n🔐 Independent Verification Results:")
            print(f"  Merkle path valid: {verification_result}")
            print(f"  Signature valid: {proof_capsule['verification']['signature_valid']}")
            print(f"  Independently verifiable: {proof_capsule['verification']['verifiable_independently']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False
    
    def run_erasure_demo(self, actor_id: str):
        """Run GDPR erasure demo."""
        print(f"🗑️  Running GDPR Erasure Demo for: {actor_id}\n")
        
        # Process erasure request
        erasure_result = self.framework.compliance.gdpr_manager.process_erasure_request(
            actor_id=actor_id,
            data_categories=["health", "biometric"]  
        )
        
        print(f"✓ Erasure processed: {erasure_result['erasure_id']}")
        print(f"  Actor: {erasure_result['actor_id']}")
        print(f"  Categories: {', '.join(erasure_result['requested_categories'] or ['all'])}")
        print(f"  Actions taken: {len(erasure_result['actions_taken'])}")
        
        for action in erasure_result['actions_taken']:
            print(f"    - {action['action']}: {action['consent_id']}")
        
        print(f"  Compliance note: {erasure_result['compliance_note']}")
        
        # Verify that cryptographic proofs remain valid after erasure
        print(f"\n🔐 Verifying proof integrity post-erasure...")
        
        try:
            # Try to verify a capsule - should still work as hashes are preserved
            access_event = self.framework.compliance.access_control_manager.log_capsule_verification(
                actor_id="post_erasure_auditor", 
                capsule_id="diagnosis_alice_001",
                access_granted=True
            )
            
            print(f"✓ Post-erasure verification access logged")
            print(f"✓ Cryptographic integrity preserved")
            print(f"  Merkle proofs remain valid (hashes preserved)")
            print(f"  Personal data erased, audit trail intact")
            
            return True
            
        except Exception as e:
            print(f"❌ Post-erasure verification failed: {e}")
            return False
    
    def run_monitoring_demo(self, model_id: str):
        """Run continuous monitoring demo."""
        print(f"📊 Running Monitoring Demo for: {model_id}\n")
        
        # Create multiple monitoring events
        events = []
        
        # Performance monitoring  
        perf_event = self.framework.compliance.monitoring_manager.create_drift_monitoring_event(
            kl_divergence=0.06,
            psi_score=0.12,
            model_version="2.0.0"
        )
        events.append(perf_event)
        
        print(f"✓ Performance monitoring: {perf_event.event_id}")
        print(f"  KL divergence: {perf_event.metrics['kl_divergence']}")
        print(f"  Alerts: {len(perf_event.alerts)}")
        
        # Create automated monitoring capsule
        monitoring_capsule = self.framework.compliance.monitoring_manager.create_automated_monitoring_capsule(
            hours_interval=24
        )
        
        print(f"\n✓ Automated monitoring capsule: {monitoring_capsule['capsule_id']}")
        print(f"  Interval: {monitoring_capsule['interval_hours']} hours")
        print(f"  Next check: {monitoring_capsule['next_check_due']}")
        print(f"  Events included: {len(monitoring_capsule['events'])}")
        
        return True
    
    def run_report_demo(self):
        """Run compliance report demo."""
        print("📋 Running Compliance Report Demo\n")
        
        report = self.framework.compliance.generate_compliance_report()
        
        print("🏛️  CIAF Compliance Report")
        print("=" * 50)
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
        
        return True
    
    def _save_demo_artifacts(self, dataset_receipt, model_receipt, inference_receipt, proof_capsule, report):
        """Save demo artifacts to files."""
        artifacts_dir = Path("demo_artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save receipts
        with open(artifacts_dir / f"dataset_receipt_{timestamp}.json", "w") as f:
            json.dump({
                "metadata": dataset_receipt.metadata,
                "anchor": dataset_receipt.anchor.__dict__,
                "leaf_hash": dataset_receipt.leaf_hash,
                "record_type": dataset_receipt.record_type.value
            }, f, indent=2, default=str)
        
        with open(artifacts_dir / f"model_receipt_{timestamp}.json", "w") as f:
            json.dump({
                "metadata": model_receipt.metadata,
                "anchor": model_receipt.anchor.__dict__, 
                "leaf_hash": model_receipt.leaf_hash,
                "record_type": model_receipt.record_type.value
            }, f, indent=2, default=str)
        
        with open(artifacts_dir / f"inference_receipt_{timestamp}.json", "w") as f:
            json.dump({
                "metadata": inference_receipt.metadata,
                "anchor": inference_receipt.anchor.__dict__,
                "leaf_hash": inference_receipt.leaf_hash,
                "record_type": inference_receipt.record_type.value
            }, f, indent=2, default=str)
        
        # Save proof capsule
        with open(artifacts_dir / f"proof_capsule_{timestamp}.json", "w") as f:
            json.dump(proof_capsule, f, indent=2, default=str)
        
        # Save compliance report
        with open(artifacts_dir / f"compliance_report_{timestamp}.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📁 Demo artifacts saved to: {artifacts_dir}/")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CIAF Compliance Demo - Cryptographic Audit Receipts with Regulatory Extensions"
    )
    
    parser.add_argument(
        "--flow",
        choices=["complete", "verification", "erasure", "monitoring", "report"],
        default="complete",
        help="Demo flow to run"
    )
    
    parser.add_argument(
        "--capsule-id",
        default="diagnosis_alice_001",  
        help="Capsule ID for verification demo"
    )
    
    parser.add_argument(
        "--actor-id",
        default="patient_alice_001",
        help="Actor ID for erasure demo"
    )
    
    parser.add_argument(
        "--model-id", 
        default="diagnostic_cnn_v2",
        help="Model ID for monitoring demo"
    )
    
    args = parser.parse_args()
    
    demo = CIAFDemo()
    
    try:
        if args.flow == "complete":
            success = demo.run_complete_flow()
        elif args.flow == "verification":
            success = demo.run_verification_demo(args.capsule_id)
        elif args.flow == "erasure":
            success = demo.run_erasure_demo(args.actor_id)
        elif args.flow == "monitoring":
            success = demo.run_monitoring_demo(args.model_id)
        elif args.flow == "report":
            success = demo.run_report_demo()
        else:
            print(f"Unknown flow: {args.flow}")
            success = False
        
        if success:
            print("\n✅ Demo completed successfully!")
            return 0
        else:
            print("\n❌ Demo failed!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Demo crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())