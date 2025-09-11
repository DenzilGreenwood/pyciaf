"""
CIAF LCM RNG Audit Example - Demonstrating Reproducible Splits

This example shows how the enhanced dataset managers now support:
1. RNG seed + source capture for reproducible splits
2. Split assignment commitments for tamper-evident membership proof
3. Stratification parameter tracking
4. Enhanced split map digests that include assignment proofs

Usage: python rng_audit_example.py

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import sys
import os

# Add the parent directory to Python path to import ciaf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ciaf.lcm.dataset_family_manager import LCMDatasetFamilyManager, DatasetFamilyMetadata, compute_split_assignment_digest
from ciaf.lcm.dataset_manager import LCMDatasetManager, DatasetMetadata
from ciaf.lcm.training_manager import DatasetSplit
from ciaf.lcm.policy import get_default_policy
from ciaf.core.crypto import sha256_hash


def demonstrate_rng_reproducibility():
    """Demonstrate RNG seed capture and assignment digest generation."""
    print("🎯 RNG Reproducibility & Audit Trail Demo")
    print("=" * 50)
    
    policy = get_default_policy()
    
    # === DATASET FAMILY MANAGER DEMO ===
    print("\n📊 Dataset Family Manager - Enhanced Split Creation")
    print("-" * 55)
    
    family_manager = LCMDatasetFamilyManager(policy)
    
    # Create family with stratification config
    family_metadata = DatasetFamilyMetadata(
        name="fraud_detection_dataset",
        version="v2.1",
        owner="ML Engineering Team",
        license="Proprietary",
        description="Credit card fraud detection with stratified sampling",
        compliance_frameworks=["PCI-DSS"]
    )
    
    # Split configs with stratification
    split_configs = {
        DatasetSplit.TRAIN: {
            "ratio": 0.7, 
            "selection": "stratified", 
            "stratify_by": ["fraud_label", "merchant_category"]
        },
        DatasetSplit.VALIDATION: {
            "ratio": 0.15, 
            "selection": "stratified", 
            "stratify_by": ["fraud_label"]
        },
        DatasetSplit.TEST: {
            "ratio": 0.15, 
            "selection": "random"
        }
    }
    
    family_anchor = family_manager.create_dataset_family(
        dataset_id="fraud_detection",
        family_metadata=family_metadata,
        split_configs=split_configs
    )
    
    print(f"\n🔍 Examining Split Metadata:")
    splits = family_manager.get_all_splits("fraud_detection")
    
    for split_type, split_anchor in splits.items():
        meta = split_anchor.split_metadata
        print(f"\n   🗂️ {split_type.value.upper()} Split:")
        print(f"      🎲 RNG Seed: {meta.rng_seed}")
        print(f"      📚 RNG Source: {meta.rng_source}")
        print(f"      📊 Stratify By: {meta.stratify_by}")
        print(f"      🔐 Assignment Digest: {meta.split_assignment_digest[:16]}...")
        print(f"      📈 Sample Count: {meta.sample_count}")
    
    # Compute enhanced split map digest
    enhanced_digest = family_manager.compute_split_map_digest("fraud_detection")
    print(f"\n🌳 Enhanced Split Map Digest: {enhanced_digest[:16]}...")
    
    # === DATASET MANAGER DEMO ===
    print(f"\n📊 Dataset Manager - Individual Split Anchors")
    print("-" * 50)
    
    dataset_manager = LCMDatasetManager(policy)
    
    # Create dataset metadata with RNG info
    dataset_metadata = DatasetMetadata(
        name="transaction_data",
        owner="Data Engineering",
        license="Internal",
        schema_digest="schema_v1.2.3",
        sampling_rules={"method": "stratified", "balance": True},
        version="2.1.0",
        content_root="mock_content_root",
        stratify_by=["transaction_type", "risk_score_bin"]
    )
    
    # Create splits
    splits = dataset_manager.create_dataset_splits(
        dataset_id="transactions",
        metadata=dataset_metadata,
        master_password="audit_demo_password"
    )
    
    print(f"\n🔍 Examining Individual Split Anchors:")
    for split_type, anchor in splits.items():
        meta = anchor.metadata
        print(f"\n   🗂️ {split_type.value.upper()} Anchor:")
        print(f"      🎲 RNG Seed: {meta.rng_seed}")
        print(f"      📚 RNG Source: {meta.rng_source}")
        print(f"      📊 Stratify By: {meta.stratify_by}")
        print(f"      🔐 Assignment Digest: {meta.split_assignment_digest[:16]}...")
        print(f"      🆔 Anchor ID: {anchor.anchor_id}")
    
    # Compute enhanced split map digest for dataset manager
    enhanced_digest_dm = dataset_manager.compute_enhanced_split_map_digest("transactions")
    print(f"\n🌳 Enhanced Split Map Digest (DM): {enhanced_digest_dm[:16]}...")
    
    # === REPRODUCIBILITY DEMONSTRATION ===
    print(f"\n🔄 Reproducibility Verification")
    print("-" * 35)
    
    # Simulate the same split assignment twice
    record_ids = ["record_001", "record_002", "record_003", "record_004", "record_005"]
    
    digest1 = compute_split_assignment_digest(record_ids)
    digest2 = compute_split_assignment_digest(record_ids)
    
    print(f"   📋 Record IDs: {record_ids}")
    print(f"   🔐 Digest #1: {digest1[:16]}...")
    print(f"   🔐 Digest #2: {digest2[:16]}...")
    print(f"   ✅ Reproducible: {digest1 == digest2}")
    
    # Demonstrate privacy-preserving variant with salt
    salt = b"audit_salt_2025"
    digest_private = compute_split_assignment_digest(record_ids, salt=salt)
    print(f"   🔒 Private Digest: {digest_private[:16]}... (with salt)")
    
    # === AUDIT SUMMARY ===
    print(f"\n📋 Audit Capabilities Summary")
    print("-" * 32)
    print("   ✅ RNG Seed Capture: Each split records the seed used")
    print("   ✅ RNG Source Tracking: Library/framework identified")  
    print("   ✅ Assignment Commitments: Tamper-evident membership proof")
    print("   ✅ Stratification Rules: Parameters captured for reproduction")
    print("   ✅ Enhanced Digests: Split maps include assignment proofs")
    print("   ✅ Privacy Preservation: Optional salting for sensitive data")
    
    print(f"\n🎯 Audit Trail Benefits:")
    print("   🔍 Auditors can reproduce exact same splits with seed+source")
    print("   🛡️ Assignment digests prove data membership without revealing data")
    print("   📊 Stratification parameters eliminate reproduction divergence")
    print("   🌳 Enhanced digests detect any membership tampering")
    
    print(f"\n🎉 RNG Audit example completed successfully!")


if __name__ == "__main__":
    demonstrate_rng_reproducibility()
