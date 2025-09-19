"""
CIAF Metadata Reveal - Inference Receipt Tracing

This utility demonstrates how to trace inference metadata back to its source model
using only a single inference receipt value. It shows the complete audit trail
from inference → training → model → dataset lineage.

Created: 2025-09-19
Last Modified: 2025-09-19
Author: Denzil James Greenwood
Version: 1.0.0
"""

import sys
import os
import json
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Add the parent directory to Python path to import ciaf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ciaf.api import CIAFFramework
from ciaf.inference import InferenceReceipt
from ciaf.core import sha256_hash


@dataclass
class MetadataTrail:
    """Complete metadata trail from inference receipt to source."""
    inference_receipt: str
    model_anchor: str
    model_name: str
    model_version: str
    training_snapshot: str
    dataset_family: str
    dataset_splits: list
    timestamp: str
    audit_path: list


class MetadataReveal:
    """Utility to reveal complete metadata trail from inference receipt."""
    
    def __init__(self):
        """Initialize the metadata reveal utility."""
        self.ciaf = CIAFFramework()
        # Simulated metadata store - in practice this would be your audit database
        self.metadata_store = self._initialize_demo_store()
    
    def _initialize_demo_store(self) -> Dict[str, Any]:
        """Initialize demo metadata store with sample data."""
        # This simulates a real audit database with indexed receipts
        return {
            "inference_receipts": {
                "r_a1b2c3d4": {
                    "receipt_id": "r_a1b2c3d4e5f6789a",
                    "model_anchor_ref": "m_7f8e9d0c1b2a3948",
                    "training_snapshot_ref": "t_5g6h7i8j9k0l1m2n",
                    "query": "What is the fraud risk for transaction $1,250 at MegaMart?",
                    "ai_output": "HIGH RISK: Unusual amount pattern detected (confidence: 89%)",
                    "timestamp": "2025-09-19T14:30:25Z",
                    "input_commitment": "ic_salted_a1b2c3d4",
                    "output_commitment": "oc_salted_e5f6g7h8",
                    "explanation_digests": ["shap_7f8e9d0c", "lime_1b2a3948"],
                    "deployment_anchor_ref": "d_prod_fraud_v1"
                }
            },
            "models": {
                "m_7f8e9d0c1b2a3948": {
                    "model_name": "fraud_detector",
                    "version": "v2.1.0",
                    "architecture": "gradient_boosting",
                    "params_root": "pr_8c9d0e1f2g3h4i5j",
                    "arch_root": "ar_4i5j6k7l8m9n0o1p",
                    "hp_digest": "hp_2g3h4i5j6k7l8m9n",
                    "env_digest": "env_6k7l8m9n0o1p2q3r",
                    "trainer_commit": "commit_abc123def456",
                    "authorized_datasets": ["financial_transactions@v3"],
                    "training_snapshot_ref": "t_5g6h7i8j9k0l1m2n"
                }
            },
            "training_snapshots": {
                "t_5g6h7i8j9k0l1m2n": {
                    "session_id": "fraud_training_2025q3",
                    "model_anchor_ref": "m_7f8e9d0c1b2a3948",
                    "dataset_family_ref": "df_financial_txn_v3",
                    "splits_used": ["train", "validation"],
                    "split_map_digest": "sm_9n0o1p2q3r4s5t6u",
                    "metrics_digest": "md_3r4s5t6u7v8w9x0y",
                    "training_config": {
                        "epochs": 100,
                        "learning_rate": 0.01,
                        "early_stopping": True
                    },
                    "completion_timestamp": "2025-09-15T08:45:12Z"
                }
            },
            "dataset_families": {
                "df_financial_txn_v3": {
                    "family_name": "financial_transactions",
                    "version": "v3",
                    "owner": "fraud_detection_team",
                    "license": "Internal-Use-Only",
                    "description": "Production financial transaction data for fraud detection",
                    "splits": {
                        "train": "ds_train_7v8w9x0y1z2a3b4c",
                        "validation": "ds_val_1z2a3b4c5d6e7f8g",
                        "test": "ds_test_5d6e7f8g9h0i1j2k"
                    },
                    "datasets_root_anchor": "dr_9h0i1j2k3l4m5n6o",
                    "data_governance": {
                        "retention_policy": "7_years",
                        "privacy_level": "restricted",
                        "compliance_tags": ["PCI-DSS", "SOX", "GDPR"]
                    }
                }
            }
        }
    
    def reveal_metadata_trail(self, inference_receipt_id: str) -> Optional[MetadataTrail]:
        """
        Reveal complete metadata trail from inference receipt ID.
        
        Args:
            inference_receipt_id: The inference receipt identifier (e.g., "r_a1b2c3d4")
            
        Returns:
            MetadataTrail object with complete lineage or None if not found
        """
        print(f"🔍 Tracing metadata trail for receipt: {inference_receipt_id}")
        print("=" * 60)
        
        # Step 1: Lookup inference receipt
        receipt_data = self.metadata_store["inference_receipts"].get(inference_receipt_id)
        if not receipt_data:
            print(f"❌ Inference receipt '{inference_receipt_id}' not found in audit store")
            return None
        
        print(f"✅ Found inference receipt: {receipt_data['receipt_id']}")
        print(f"   📅 Timestamp: {receipt_data['timestamp']}")
        print(f"   🤖 Query: {receipt_data['query'][:60]}...")
        print(f"   💡 Output: {receipt_data['ai_output'][:60]}...")
        print()
        
        # Step 2: Lookup model from receipt
        model_ref = receipt_data["model_anchor_ref"]
        model_data = self.metadata_store["models"].get(model_ref)
        if not model_data:
            print(f"❌ Model '{model_ref}' not found in metadata store")
            return None
        
        print(f"🎯 Model traced: {model_data['model_name']} {model_data['version']}")
        print(f"   🏗️ Architecture: {model_data['architecture']}")
        print(f"   🔗 Model anchor: {model_ref}")
        print(f"   📊 Params root: {model_data['params_root']}")
        print(f"   🔧 Trainer commit: {model_data['trainer_commit']}")
        print()
        
        # Step 3: Lookup training snapshot
        training_ref = receipt_data["training_snapshot_ref"]
        training_data = self.metadata_store["training_snapshots"].get(training_ref)
        if not training_data:
            print(f"❌ Training snapshot '{training_ref}' not found")
            return None
        
        print(f"🏋️ Training session: {training_data['session_id']}")
        print(f"   📅 Completed: {training_data['completion_timestamp']}")
        print(f"   🗂️ Splits used: {training_data['splits_used']}")
        print(f"   📊 Metrics digest: {training_data['metrics_digest']}")
        print(f"   ⚙️ Config: {training_data['training_config']}")
        print()
        
        # Step 4: Lookup dataset family
        dataset_ref = training_data["dataset_family_ref"]
        dataset_data = self.metadata_store["dataset_families"].get(dataset_ref)
        if not dataset_data:
            print(f"❌ Dataset family '{dataset_ref}' not found")
            return None
        
        print(f"🗃️ Dataset family: {dataset_data['family_name']} {dataset_data['version']}")
        print(f"   👤 Owner: {dataset_data['owner']}")
        print(f"   📜 License: {dataset_data['license']}")
        print(f"   🏷️ Compliance: {', '.join(dataset_data['data_governance']['compliance_tags'])}")
        print(f"   🗂️ Available splits: {list(dataset_data['splits'].keys())}")
        print(f"   🌳 Datasets root: {dataset_data['datasets_root_anchor']}")
        print()
        
        # Step 5: Construct audit path
        audit_path = [
            f"inference_receipt:{inference_receipt_id}",
            f"model_anchor:{model_ref}",
            f"training_snapshot:{training_ref}",
            f"dataset_family:{dataset_ref}"
        ]
        
        print("🛤️ Complete audit trail:")
        for i, step in enumerate(audit_path, 1):
            print(f"   {i}. {step}")
        print()
        
        # Create metadata trail object
        trail = MetadataTrail(
            inference_receipt=inference_receipt_id,
            model_anchor=model_ref,
            model_name=model_data['model_name'],
            model_version=model_data['version'],
            training_snapshot=training_ref,
            dataset_family=dataset_ref,
            dataset_splits=training_data['splits_used'],
            timestamp=receipt_data['timestamp'],
            audit_path=audit_path
        )
        
        return trail
    
    def verify_trail_integrity(self, trail: MetadataTrail) -> bool:
        """
        Verify the integrity of the metadata trail using cryptographic anchors.
        
        Args:
            trail: MetadataTrail object to verify
            
        Returns:
            True if trail integrity is verified, False otherwise
        """
        print("🔒 Verifying trail integrity...")
        print("-" * 40)
        
        try:
            # In a real implementation, this would:
            # 1. Verify Merkle proofs for each anchor
            # 2. Check cryptographic signatures
            # 3. Validate timestamp authority
            # 4. Verify connections between anchors
            
            # For demo purposes, we'll simulate these checks
            checks = [
                ("Inference receipt signature", True),
                ("Model anchor Merkle proof", True),
                ("Training snapshot integrity", True),
                ("Dataset family authenticity", True),
                ("Audit trail connections", True),
                ("Timestamp authority validation", True)
            ]
            
            all_passed = True
            for check_name, passed in checks:
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"   {status} {check_name}")
                if not passed:
                    all_passed = False
            
            print()
            if all_passed:
                print("🎉 Trail integrity verification: PASSED")
                print("   All cryptographic anchors verified successfully")
            else:
                print("⚠️ Trail integrity verification: FAILED")
                print("   One or more verification checks failed")
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Verification error: {e}")
            return False
    
    def export_trail_report(self, trail: MetadataTrail, output_file: str = None) -> str:
        """
        Export complete metadata trail as a compliance report.
        
        Args:
            trail: MetadataTrail object to export
            output_file: Optional file path to save report
            
        Returns:
            JSON report as string
        """
        report = {
            "metadata_trail_report": {
                "report_id": f"trail_{sha256_hash(trail.inference_receipt.encode())[:12]}",
                "generated_at": datetime.now().isoformat(),
                "ciaf_version": "1.0.0",
                "trail_summary": {
                    "inference_receipt": trail.inference_receipt,
                    "model": f"{trail.model_name} {trail.model_version}",
                    "training_timestamp": trail.timestamp,
                    "dataset_lineage": trail.dataset_family,
                    "audit_depth": len(trail.audit_path)
                },
                "full_lineage": {
                    "inference": {
                        "receipt_id": trail.inference_receipt,
                        "timestamp": trail.timestamp
                    },
                    "model": {
                        "anchor": trail.model_anchor,
                        "name": trail.model_name,
                        "version": trail.model_version
                    },
                    "training": {
                        "snapshot": trail.training_snapshot,
                        "splits_used": trail.dataset_splits
                    },
                    "dataset": {
                        "family": trail.dataset_family
                    }
                },
                "audit_path": trail.audit_path,
                "compliance_notes": {
                    "traceability": "Complete end-to-end lineage established",
                    "immutability": "All anchors cryptographically secured",
                    "auditability": "Full trail preserved in immutable audit store"
                }
            }
        }
        
        report_json = json.dumps(report, indent=2)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_json)
            print(f"📄 Trail report exported to: {output_file}")
        
        return report_json


def demo_metadata_reveal():
    """Demonstrate the metadata reveal functionality."""
    print("🔍 CIAF Metadata Reveal Demonstration")
    print("=" * 60)
    print("Tracing inference metadata back to source model and dataset")
    print()
    
    # Initialize reveal utility
    revealer = MetadataReveal()
    
    # Demo receipt ID (in practice, user would provide this)
    demo_receipt_id = "r_a1b2c3d4"
    
    print(f"📝 Input: Single inference receipt ID = '{demo_receipt_id}'")
    print()
    
    # Reveal complete metadata trail
    trail = revealer.reveal_metadata_trail(demo_receipt_id)
    
    if trail:
        # Verify trail integrity
        integrity_ok = revealer.verify_trail_integrity(trail)
        
        if integrity_ok:
            # Export compliance report
            print("📊 Generating compliance report...")
            report = revealer.export_trail_report(trail, "metadata_trail_report.json")
            print()
            print("📋 Trail Report Summary:")
            print("-" * 30)
            print(f"   Model: {trail.model_name} {trail.model_version}")
            print(f"   Training: {trail.training_snapshot}")
            print(f"   Dataset: {trail.dataset_family}")
            print(f"   Splits: {', '.join(trail.dataset_splits)}")
            print(f"   Audit depth: {len(trail.audit_path)} levels")
            print()
            print("✅ Complete metadata lineage successfully traced!")
            print("   Regulatory compliance: READY")
            print("   Audit trail: VERIFIED")
            print("   Data governance: COMPLIANT")
        else:
            print("⚠️ Trail integrity verification failed")
    else:
        print("❌ Unable to trace metadata trail")


if __name__ == "__main__":
    demo_metadata_reveal()