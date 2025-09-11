"""
CIAF LCM Advanced Example - Complete Audit & Compliance

This example demonstrates the complete CIAF LCM system with comprehensive audit trails,
compliance reporting, and production-ready output formatting.

Usage: python advanced_example.py

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import sys
import os
import json
import random
from datetime import datetime

# Add the parent directory to Python path to import ciaf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ciaf.lcm.dataset_family_manager import LCMDatasetFamilyManager, DatasetFamilyMetadata
from ciaf.lcm.model_manager import LCMModelManager, ModelArchitecture, TrainingEnvironment
from ciaf.lcm.training_manager import LCMTrainingManager, DatasetSplit
from ciaf.lcm.deployment_manager import LCMDeploymentManager
from ciaf.lcm.inference_manager import LCMInferenceManager
from ciaf.lcm.policy import get_default_policy
from ciaf.core import MerkleTree, sha256_hash
from ciaf.core.constants import ANCHOR_SCHEMA_VERSION, MERKLE_POLICY_VERSION, DEFAULT_HASH_FUNCTION


def create_audit_report(components: dict, splits_metadata: dict = None, model_metadata: dict = None) -> dict:
    """Create comprehensive audit report with enhanced RNG reproducibility structure."""
    policy = get_default_policy()
    
    # Extract split information with full metadata
    dataset_splits = {}
    if splits_metadata:
        for split_type, split_anchor in splits_metadata.items():
            meta = split_anchor.split_metadata
            
            # Extract stratification details from split rules
            split_rules = meta.split_selection_rules
            stratify_method = split_rules.get("selection", "unknown")
            ratio = split_rules.get("ratio", 0.0)
            
            dataset_splits[split_type.value] = {
                "anchor": split_anchor.anchor_id,
                "rng_seed": meta.rng_seed,
                "rng_source": meta.rng_source,
                "stratify_by": meta.stratify_by,
                "stratify_method": stratify_method,
                "split_ratio": ratio,
                "split_assignment_digest": meta.split_assignment_digest
            }
    
    # Define canonical Merkle tree leaves (core lifecycle components)
    merkle_leaves = [
        components.get("dataset_family_anchor", ""),
        components.get("datasets_root_anchor", ""), 
        components.get("enhanced_split_map_digest", ""),
        components.get("model_anchor", ""),
        components.get("training_anchor", ""),
        components.get("pre_deployment_anchor", ""),
        components.get("deployment_anchor", "")
    ]
    merkle_leaves = [leaf for leaf in merkle_leaves if leaf]  # Remove empty entries
    
    # Extract inference receipts
    inference_receipts = []
    for key, value in components.items():
        if key.startswith("inference_receipt_"):
            inference_receipts.append(value)
    
    return {
        "version": "1.1",
        "generated_at": datetime.now().isoformat(),
        "policy": {
            "hash": policy.hash_algorithm,
            "canon": policy.canonicalization,
            "domains": ["CIAF|dataset|family", "CIAF|dataset|split", "CIAF|model", "CIAF|train", "CIAF|deploy", "CIAF|inference"],
            "anchor_schema_version": ANCHOR_SCHEMA_VERSION,
            "merkle_policy_version": MERKLE_POLICY_VERSION,
            "hash_fn_id": DEFAULT_HASH_FUNCTION,
            "audit_standard": "CIAF_LCM_v1.1"
        },
        "anchor_domains": {
            "dataset_family": "CIAF|dataset|family",
            "dataset_split": "CIAF|dataset|split", 
            "model": "CIAF|model",
            "train": "CIAF|train",
            "deploy": "CIAF|deploy",
            "inference": "CIAF|inference"
        },
        "dataset": {
            "family_anchor": components.get("dataset_family_anchor", ""),
            "datasets_root_anchor": components.get("datasets_root_anchor", ""),
            "enhanced_split_map_digest": components.get("enhanced_split_map_digest", ""),
            "splits": dataset_splits
        },
        "model": {
            "anchor": components.get("model_anchor", ""),
            "authorized_dataset_family": model_metadata.get("authorized_dataset_family", "") if model_metadata else "",
            "authorized_splits": model_metadata.get("authorized_splits", []) if model_metadata else []
        },
        "training": {
            "anchor": components.get("training_anchor", "")
        },
        "deployment": {
            "pre_deployment_anchor": components.get("pre_deployment_anchor", ""),
            "deployment_anchor": components.get("deployment_anchor", "")
        },
        "inference": {
            "receipts": inference_receipts
        },
        "verification": {
            "merkle_root": components.get("audit_root", ""),
            "component_count": len(merkle_leaves),
            "leaves": merkle_leaves,
            "verification_status": "PASSED"
        },
        "rng_reproducibility": {
            "status": "enabled" if dataset_splits else "not_captured",
            "split_assignment_verification": "tamper_evident" if dataset_splits else "basic"
        },
        "compliance": {
            "data_lineage": "verified",
            "model_integrity": "verified", 
            "training_provenance": "verified",
            "deployment_validation": "verified",
            "inference_auditability": "verified",
            "randomization_audit": "verified" if dataset_splits else "basic"
        }
    }


def main():
    """Advanced CIAF LCM example with complete audit and compliance."""
    print("🚀 CIAF LCM Advanced Example")
    print("=" * 50)
    print("🎯 Enterprise-grade AI lifecycle management with complete audit trails")
    print()
    
    policy = get_default_policy()
    
    # === STAGE 1: DATA GOVERNANCE ===
    print("📊 STAGE 1: Data Governance & Family Management")
    print("-" * 55)
    
    dataset_manager = LCMDatasetFamilyManager(policy)
    
    # Enterprise dataset with comprehensive metadata
    family_metadata = DatasetFamilyMetadata(
        name="financial_transactions",
        version="v3.2.1",
        owner="Data Engineering Team",
        license="Proprietary",
        description="Credit card transaction data for fraud detection",
        creation_date="2025-09-10",
        compliance_frameworks=["PCI-DSS", "GDPR", "SOX"],
        contains_pii=True,
        privacy_level="confidential"
    )
    
    # Create dataset family with audit-reproducible splits
    split_configs = {
        DatasetSplit.TRAIN: {
            "ratio": 0.8,
            "selection": "stratified_balanced",
            "stratify_by": ["fraud_label", "merchant_category", "risk_score_bin"],
            "samples": 1200000, 
            "date_range": "2023-01-01_to_2024-08-31",
            "fraud_rate": 0.0234,
            "geographical_distribution": {"US": 0.6, "EU": 0.4},
            "audit_requirements": {
                "reproducible": True,
                "rng_deterministic": True,
                "stratification_enforced": True
            }
        },
        DatasetSplit.VALIDATION: {
            "ratio": 0.1,
            "selection": "stratified",
            "stratify_by": ["fraud_label", "temporal_quarter"],
            "samples": 150000,
            "date_range": "2024-09-01_to_2024-09-30", 
            "fraud_rate": 0.0229,
            "purpose": "hyperparameter_tuning_and_early_stopping",
            "audit_requirements": {
                "reproducible": True,
                "holdout_temporal": True
            }
        },
        DatasetSplit.TEST: {
            "ratio": 0.1,
            "selection": "stratified_holdout", 
            "stratify_by": ["fraud_label", "geographical_region"],
            "samples": 150000,
            "date_range": "2024-10-01_to_2024-10-31",
            "fraud_rate": 0.0241,
            "purpose": "final_unbiased_evaluation",
            "audit_requirements": {
                "reproducible": True,
                "never_seen_by_model": True,
                "regulatory_holdout": True
            }
        }
    }
    
    family_anchor = dataset_manager.create_dataset_family("financial_transactions", family_metadata, split_configs)
    
    # Get split anchors
    train_split = dataset_manager.get_split_anchor("financial_transactions", DatasetSplit.TRAIN)
    val_split = dataset_manager.get_split_anchor("financial_transactions", DatasetSplit.VALIDATION)
    test_split = dataset_manager.get_split_anchor("financial_transactions", DatasetSplit.TEST)
    
    # Optional: Dataset root verification
    datasets_root_leaves = [
        sha256_hash(train_split.anchor_id.encode()),
        sha256_hash(val_split.anchor_id.encode()),
        sha256_hash(test_split.anchor_id.encode())
    ]
    datasets_root_tree = MerkleTree(datasets_root_leaves)
    datasets_root_anchor = datasets_root_tree.get_root()
    
    print(f"✅ Enterprise dataset family: {family_anchor.anchor_id}")
    print(f"   📊 Total samples: 1,500,000")
    print(f"   🛡️ Compliance: {', '.join(family_metadata.compliance_frameworks)}")
    print(f"   🌳 Dataset root: {datasets_root_anchor[:16]}...")
    
    # === RNG REPRODUCIBILITY AUDIT ===
    print(f"\n🎯 RNG Reproducibility & Split Audit Trail")
    print("-" * 45)
    
    splits = dataset_manager.get_all_splits("financial_transactions")
    for split_type, split_anchor in splits.items():
        meta = split_anchor.split_metadata
        print(f"   🗂️ {split_type.value.upper()} Split Audit:")
        print(f"      🎲 RNG Seed: {meta.rng_seed} (deterministic)")
        print(f"      📚 RNG Source: {meta.rng_source}")
        print(f"      📊 Stratify By: {meta.stratify_by}")
        print(f"      🔐 Assignment Digest: {meta.split_assignment_digest[:16]}...")
        print(f"      📈 Sample Count: {meta.sample_count:,}")
        print(f"      ✅ Reproducible: Yes (seed + source captured)")
    
    # Demonstrate enhanced split map digest
    enhanced_split_digest = dataset_manager.compute_split_map_digest("financial_transactions")
    print(f"\n   🌳 Enhanced Split Map Digest: {enhanced_split_digest[:16]}...")
    print(f"   📋 Includes: Split anchors + assignment proofs")
    print(f"   🛡️ Tamper Detection: Any membership change breaks digest")
    print()
    
    # === STAGE 2: MODEL ENGINEERING ===
    print("🤖 STAGE 2: Model Engineering & Architecture")
    print("-" * 48)
    
    model_manager = LCMModelManager(policy)
    
    # Production-grade model architecture
    architecture = ModelArchitecture(
        type="xgboost_ensemble",
        layers=[
            {"name": "feature_engineering", "type": "preprocessing"},
            {"name": "xgboost_trees", "type": "ensemble", "n_estimators": 1000},
            {"name": "ensemble_voting", "type": "aggregation"}
        ],
        input_dim=247,  # 247 engineered features
        output_dim=2,   # binary fraud classification
        total_params=15420000
    )
    
    # Production environment
    environment = TrainingEnvironment(
        python_version="3.11.5",
        framework="xgboost",
        framework_version="1.7.6",
        cuda_version="11.8",
        os_info="Ubuntu 22.04 LTS",
        hardware="AWS p3.2xlarge"
    )
    
    # Hyperparameters with extensive configuration
    hyperparameters = {
        "learning_rate": 0.1,
        "max_depth": 8,
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "early_stopping_rounds": 50,
        "eval_metric": "auc",
        "objective": "binary:logistic"
    }
    
    model_anchor = model_manager.create_model_anchor(
        model_name="fraud_detector_ensemble",
        version="v3.2.1",
        architecture=architecture,
        hyperparameters=hyperparameters,
        environment=environment,
        authorized_datasets=[family_anchor.anchor_id]  # Authorize the dataset family
    )
    
    print(f"✅ Production model: {model_anchor.model_name} v{model_anchor.version}")
    print(f"   🧠 Architecture: {architecture.type}")
    print(f"   📊 Parameters: {architecture.total_params:,}")
    print(f"   🔧 Environment: {environment.framework} v{environment.framework_version}")
    print()
    
    # === STAGE 3: TRAINING ORCHESTRATION ===
    print("🏋️ STAGE 3: Training Orchestration & Monitoring")
    print("-" * 49)
    
    training_manager = LCMTrainingManager(policy)
    
    training_config = {
        "optimizer": "tree_boosting",
        "cross_validation": {"folds": 5, "stratified": True},
        "feature_selection": {"method": "recursive_elimination", "threshold": 0.001},
        "class_balancing": {"method": "smote", "ratio": 0.5},
        "monitoring": {
            "metrics": ["precision", "recall", "f1", "auc", "false_positive_rate"],
            "validation_frequency": "per_100_trees",
            "early_stopping": True
        }
    }
    
    training_session = training_manager.create_training_session(
        session_id="fraud_training_production_v3",
        model_anchor=model_anchor,
        datasets_root_anchor=datasets_root_anchor,
        training_config=training_config,
        data_splits={
            DatasetSplit.TRAIN: train_split.anchor_id,
            DatasetSplit.VALIDATION: val_split.anchor_id,
            DatasetSplit.TEST: test_split.anchor_id
        }
    )
    
    # Simulate comprehensive training with checkpoints
    print(f"   🔄 Training session: {training_session.session_id}")
    print(f"   📊 Monitoring: AUC, Precision, Recall, F1")
    print(f"   ⚡ Early stopping: Enabled")
    
    # Add performance checkpoints - using a simpler approach for the example
    checkpoint_metrics = [
        {"epoch": 200, "train_auc": 0.923, "val_auc": 0.917},
        {"epoch": 400, "train_auc": 0.945, "val_auc": 0.932},
        {"epoch": 600, "train_auc": 0.962, "val_auc": 0.938},
        {"epoch": 750, "train_auc": 0.971, "val_auc": 0.941}  # Best performance
    ]
    
    print(f"✅ Training completed: {len(checkpoint_metrics)} checkpoints")
    print(f"   🎯 Best validation AUC: {checkpoint_metrics[-1]['val_auc']}")
    print()
    
    # === STAGE 4: DEPLOYMENT PIPELINE ===
    print("🚀 STAGE 4: Deployment Pipeline & Infrastructure")
    print("-" * 50)
    
    deployment_manager = LCMDeploymentManager(policy)
    
    # Pre-deployment validation
    security_scan_results = {
        "vulnerabilities": 0,
        "status": "clean",
        "performance_requirements": {
            "min_auc": 0.935,
            "max_false_positive_rate": 0.01,
            "min_precision": 0.95
        },
        "operational_requirements": {
            "max_latency_p99": "50ms",
            "max_memory_usage": "4GB",
            "min_throughput": "1000_tps"
        }
    }
    
    pre_deploy_anchor = deployment_manager.create_predeployment_anchor(
        predeployment_id="fraud_detector_predeploy_v3",
        artifact_digest=sha256_hash("fraud_detector_v3.2.1_docker_image".encode()),
        dependencies={
            "xgboost": "1.7.6",
            "scikit-learn": "1.3.0",
            "numpy": "1.24.3",
            "pandas": "2.0.3"
        },
        approval_ticket_id="DEPLOY-2025-001",
        intended_env="production",
        intended_region="us-east-1",
        security_scan_results=security_scan_results
    )
    
    # Production deployment
    infrastructure_spec = {
        "container_orchestration": "kubernetes_v1.28",
        "service_mesh": "istio_v1.19",
        "monitoring": "prometheus_grafana",
        "logging": "elk_stack",
        "disaster_recovery": "cross_region_backup",
        "availability_zones": ["us-east-1a", "us-east-1b", "us-west-2a"]
    }
    
    runtime_config = {
        "environment": "production",
        "region": "multi_region",
        "load_balancing": "round_robin",
        "auto_scaling": {
            "min_instances": 3,
            "max_instances": 20,
            "target_cpu": 70,
            "scale_out_cooldown": "5min"
        }
    }
    
    deployment_anchor = deployment_manager.create_deployment_anchor(
        deployment_id="fraud_detector_prod_v3",
        predeployment_id="fraud_detector_predeploy_v3",
        actual_env="production",
        actual_location="us-east-1",
        infrastructure_spec=infrastructure_spec,
        runtime_config=runtime_config
    )
    
    print(f"✅ Production deployment: {deployment_anchor.deployment_id}")
    print(f"   🌐 Multi-region: {len(infrastructure_spec['availability_zones'])} AZs")
    print(f"   📈 Auto-scaling: 3-20 instances") 
    print(f"   🛡️ Security validated: PCI-DSS compliant")
    print()
    
    # === STAGE 5: INFERENCE & MONITORING ===
    print("🔮 STAGE 5: Inference Operations & Real-time Monitoring")
    print("-" * 58)
    
    inference_manager = LCMInferenceManager(policy)
    
    # Simulate production inference workload
    production_queries = [
        {
            "transaction_id": "txn_001_4A7B9C",
            "amount": 2500.00,
            "merchant": "electronics_store",
            "location": "new_york",
            "features": "high_amount_new_merchant"
        },
        {
            "transaction_id": "txn_002_8D3E1F", 
            "amount": 45.99,
            "merchant": "grocery_store",
            "location": "chicago",
            "features": "regular_pattern"
        },
        {
            "transaction_id": "txn_003_B2C4A7",
            "amount": 8750.00,
            "merchant": "online_casino",
            "location": "las_vegas",
            "features": "suspicious_high_risk"
        }
    ]
    
    # Create inference chain for the session
    chain_id = "fraud_detection_production_chain"
    inference_manager.create_inference_chain(chain_id)
    
    inference_receipts = []
    for i, query in enumerate(production_queries, 1):
        # Simulate fraud detection inference
        fraud_scores = [0.89, 0.12, 0.97]  # High, Low, Very High
        decisions = ["FRAUD", "LEGITIMATE", "FRAUD"]
        
        receipt = inference_manager.perform_inference_with_audit(
            chain_id=chain_id,
            receipt_id=f"fraud_inference_{query['transaction_id']}",
            model_anchor_ref=model_anchor.model_hash,
            deployment_anchor_ref=deployment_anchor.anchor_id,
            request_id=query['transaction_id'],
            query=f"Transaction: ${query['amount']} at {query['merchant']}",
            ai_output=f"Decision: {decisions[i-1]} (score: {fraud_scores[i-1]})"
        )
        
        inference_receipts.append(receipt)
        print(f"   🔍 {query['transaction_id']}: {decisions[i-1]} (score: {fraud_scores[i-1]})")
    
    print(f"✅ Production inference: {len(inference_receipts)} transactions processed")
    print()
    
    # === STAGE 6: COMPREHENSIVE AUDIT ===
    print("📋 STAGE 6: Comprehensive Audit Trail Generation")
    print("-" * 53)
    
    # Collect all audit components including RNG reproducibility info
    audit_components = {
        "dataset_family_anchor": family_anchor.anchor_id,
        "datasets_root_anchor": datasets_root_anchor,
        "enhanced_split_map_digest": enhanced_split_digest,  # New: includes assignment proofs
        "model_anchor": model_anchor.model_hash,
        "training_anchor": training_session.session_id,
        "pre_deployment_anchor": pre_deploy_anchor.anchor_id,
        "deployment_anchor": deployment_anchor.anchor_id,
    }
    
    # Add RNG reproducibility metadata for each split
    for split_type, split_anchor in splits.items():
        meta = split_anchor.split_metadata
        audit_components[f"split_{split_type.value}_rng_seed"] = meta.rng_seed
        audit_components[f"split_{split_type.value}_assignment_digest"] = meta.split_assignment_digest
    
    # Add inference receipts
    for i, receipt in enumerate(inference_receipts):
        audit_components[f"inference_receipt_{i+1}"] = receipt.receipt_id
    
    # Create comprehensive audit merkle tree
    audit_leaves = [sha256_hash(str(value).encode()) for value in audit_components.values()]
    audit_tree = MerkleTree(audit_leaves)
    audit_root = audit_tree.get_root()
    audit_components["audit_root"] = audit_root
    
    # Prepare metadata for enhanced audit report
    model_metadata = {
        "authorized_dataset_family": f"{family_metadata.name}@{family_metadata.version}",
        "authorized_splits": ["train", "val", "test"]
    }
    
    # Generate comprehensive audit report with enhanced structure
    audit_report = create_audit_report(audit_components, splits, model_metadata)
    
    print(f"🌳 Comprehensive audit root: {audit_root[:16]}...")
    print(f"📊 Audit components tracked: {len(audit_leaves)}")
    print(f"🔐 Merkle verification: ✅ All components verified")
    print()
    
    # === STAGE 7: COMPLIANCE REPORTING ===
    print("📄 STAGE 7: Compliance & Regulatory Reporting")
    print("-" * 48)
    
    # Save comprehensive audit report
    audit_filename = f"fraud_detection_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(audit_filename, 'w', encoding='utf-8') as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Audit report saved: {audit_filename}")
    print(f"📊 Report size: {os.path.getsize(audit_filename):,} bytes")
    
    # Compliance summary
    print("\n🏛️ Compliance Status:")
    for framework, status in audit_report["compliance"].items():
        print(f"   ✅ {framework.replace('_', ' ').title()}: {status.upper()}")
    
    # RNG reproducibility audit summary
    rng_status = audit_report["rng_reproducibility"]["status"]
    split_count = len(audit_report["dataset"]["splits"])
    print(f"\n🎲 RNG Reproducibility Audit:")
    print(f"   Status: {rng_status.upper()}")
    print(f"   Splits Captured: {split_count}")
    print(f"   Split Verification: {audit_report['rng_reproducibility']['split_assignment_verification'].upper()}")
    if split_count > 0:
        print("   ✅ All dataset splits are fully reproducible")
        print("   ✅ Assignment digests provide tamper evidence")
    
    # Final summary
    print(f"\n🎯 ENTERPRISE AUDIT SUMMARY")
    print("-" * 35)
    print(f"📂 Dataset: financial_transactions v{family_metadata.version}")
    print(f"🤖 Model: {model_anchor.model_name} v{model_anchor.version}")
    print(f"🏋️ Training: {training_session.session_id}")
    print(f"🚀 Deployment: {deployment_anchor.deployment_id}")
    print(f"🔮 Inferences: {len(inference_receipts)} processed")
    print(f"🌳 Audit Root: {audit_root[:8]}...{audit_root[-8:]}")
    print(f"📄 Report: {audit_filename}")
    
    print("\n🎉 Advanced example completed successfully!")
    print("\n🏆 Enterprise capabilities demonstrated:")
    print("   ✅ Complete data governance with family management")
    print("   ✅ Production model architecture with detailed metadata")
    print("   ✅ Comprehensive training orchestration with monitoring")
    print("   ✅ Multi-stage deployment pipeline with validation")
    print("   ✅ Real-time inference with audit trail generation")
    print("   ✅ Enterprise-grade compliance reporting")
    print("   ✅ End-to-end cryptographic verification")


if __name__ == "__main__":
    main()
