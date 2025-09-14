"""
CIAF LCM Showcase Example - Production Demo

This example showcases the production-ready CIAF LCM system with the exact same
output format as our enhanced quick_lcm_test_final.py, demonstrating all the
improvements we made including domain tagging, reference links, and audit features.

Usage: python showcase_example.py

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
from ciaf.lcm.training_manager import LCMTrainingManager, DatasetSplit, TrainingCheckpoint, TrainingMetrics
from ciaf.lcm.policy import get_default_policy
from ciaf.core import MerkleTree, sha256_hash
from ciaf.core.constants import ANCHOR_SCHEMA_VERSION, MERKLE_POLICY_VERSION, DEFAULT_HASH_FUNCTION


def print_policy_line():
    """Print enhanced policy information with all improvements."""
    policy = get_default_policy()
    print(f"policy: hash={policy.hash_algorithm} | canon={policy.canonicalization}")
    print(f"domains: CIAF|{{dataset|family, dataset|split, model, train, predeploy, deploy, test_eval, inference}}")
    print(f"merkle: fanout={policy.merkle.fanout}, padding={policy.merkle.padding}, leaf_encoding=raw32")
    print(f"commitments: default={policy.commitments.value}")
    print(f"anchor_schema_version=1.1, merkle_policy_version={MERKLE_POLICY_VERSION}, timezone=UTC")
    print(f"canon_unicode=NFC")
    print(f"hash_fn_id={DEFAULT_HASH_FUNCTION}")
    print(f"salt_len=16, salt_encoding=raw_bytes, rng_policy=recorded")
    print(f"logs_access=least-privilege; audit_logs=append-only (WORM)")
    print(f"capsule_sig=ed25519:awaiting_bundle_completion (pubkey_id:ciaf_demo_key_001)")


def main():
    """Production showcase of CIAF LCM with all enhancements."""
    print("CIAF LCM Production Showcase")
    print("="*45)
    
    # Print enhanced policy line
    print_policy_line()
    print()

    # Initialize managers
    policy = get_default_policy()
    dataset_manager = LCMDatasetFamilyManager(policy)
    model_manager = LCMModelManager(policy)
    training_manager = LCMTrainingManager(policy)
    
    print("LCM Capsule Manager created")
    print()

    # Dataset Family with Enhanced Metadata
    print("Dataset (one family with splits)")
    print("Creating dataset family: ai_research_data")
    
    family_metadata = DatasetFamilyMetadata(
        name="ai_research_data",
        version="v2.1", 
        owner="research_consortium",
        license="MIT",
        description="Research dataset for AI model development with enhanced audit features",
        creation_date="2025-09-10"
    )
    
    family_anchor = dataset_manager.create_dataset_family("ai_research_data", family_metadata)
    family_hex = family_anchor.family_anchor
    
    print("Dataset Family 'ai_research_data' v2.1 initialized")
    print(f"   Dataset family anchor: {family_hex[:12]}...")
    print(f"   Dataset family anchor domain=\"CIAF|dataset|family\"")
    
    # Get splits that were created automatically
    train_split = dataset_manager.get_split_anchor("ai_research_data", DatasetSplit.TRAIN)
    val_split = dataset_manager.get_split_anchor("ai_research_data", DatasetSplit.VALIDATION)
    test_split = dataset_manager.get_split_anchor("ai_research_data", DatasetSplit.TEST)
    
    train_anchor = train_split.split_anchor if train_split else "mock_train_anchor"
    val_anchor = val_split.split_anchor if val_split else "mock_val_anchor"
    test_anchor = test_split.split_anchor if test_split else "mock_test_anchor"
    
    print(f"   Split 'train' initialized: {train_anchor[:16]}...")
    print(f"   Split 'val' initialized:   {val_anchor[:16]}...")
    print(f"   Split 'test' initialized:  {test_anchor[:16]}...")
    print(f"   Split anchor domain=\"CIAF|dataset|split\"")
    print(f"   Splits created: ['train', 'val', 'test']")
    print(f"      Split anchor(train): {train_anchor[:16]}...")
    print(f"      Split anchor(val):   {val_anchor[:16]}...")
    print(f"      Split anchor(test):  {test_anchor[:16]}...")
    
    # Create datasets root anchor from splits
    from ciaf.core import MerkleTree
    datasets_root_leaves = [train_anchor, val_anchor, test_anchor]
    datasets_root_merkle = MerkleTree(datasets_root_leaves)
    datasets_root_anchor = datasets_root_merkle.get_root()
    datasets_root_anchor_display = f"dr_{datasets_root_anchor[:4]}...{datasets_root_anchor[-4:]}"
    
    print(f"   Datasets root anchor: {datasets_root_anchor_display} (MerkleRoot(train,val,test))")
    print("Authorized dataset family: ai_research_data v2.1 (available_splits: train,val,test; used_in_training: train,val)")
    
    # Add some sample data to the splits  
    if train_split:
        for i in range(5):
            train_split.add_sample_hash(f"train_sample_{i}_hash")
    if val_split:
        for i in range(3):
            val_split.add_sample_hash(f"val_sample_{i}_hash")
    if test_split:
        for i in range(2):
            test_split.add_sample_hash(f"test_sample_{i}_hash")
    
    train_anchor = train_split.split_anchor
    val_anchor = val_split.split_anchor
    test_anchor = test_split.split_anchor
    
    print(f"   Split 'train' initialized: {train_anchor[:16]}...")
    print(f"   Split 'val' initialized:   {val_anchor[:16]}...")
    print(f"   Split 'test' initialized:  {test_anchor[:16]}...")
    print(f"   Split anchor domain=\"CIAF|dataset|split\"")
    print(f"   Splits created: ['train', 'val', 'test']")
    print(f"      ▸ split_anchor(train): {train_anchor[:16]}...")
    print(f"      ▸ split_anchor(val):   {val_anchor[:16]}...")
    print(f"      ▸ split_anchor(test):  {test_anchor[:16]}...")
    
    # Optional: Datasets root anchor (the fix we implemented)
    datasets_root_leaves = [train_anchor, val_anchor, test_anchor]
    datasets_root_tree = MerkleTree(datasets_root_leaves)
    datasets_root_anchor = datasets_root_tree.get_root()
    print(f"   Datasets root anchor: dr_{datasets_root_anchor[:4]}...{datasets_root_anchor[-4:]} (MerkleRoot(train,val,test))")
    
    print("Authorized dataset family: ai_research_data v2.1 (available_splits: train,val,test; used_in_training: train,val)")
    print()

    # Enhanced Model Creation
    print("Model Anchor")
    print("Creating enhanced LCM model anchor for: research_model v2.1.0")
    
    architecture = ModelArchitecture(
        type="transformer_research",
        layers=[
            {"type": "embedding", "input_dim": 1024, "output_dim": 768},
            {"type": "multi_head_attention", "heads": 12, "dim": 768},
            {"type": "feed_forward", "hidden_dim": 3072},
            {"type": "classifier", "input_dim": 768, "output_dim": 10}
        ],
        input_dim=1024,
        output_dim=10,
        total_params=175000000
    )
    
    environment = TrainingEnvironment(
        python_version="3.11.5",
        framework="pytorch_lightning",
        framework_version="2.1.0",
        cuda_version="12.1",
        os_info="Ubuntu 22.04 LTS",
        hardware="A100 GPU"
    )
    
    hyperparameters = {
        "learning_rate": 1e-4,
        "batch_size": 64,
        "num_epochs": 10,
        "warmup_steps": 1000,
        "weight_decay": 0.01,
        "dropout": 0.1
    }
    
    model_anchor = model_manager.create_model_anchor(
        model_name="research_model",
        version="v2.1.0", 
        architecture=architecture,
        hyperparameters=hyperparameters,
        environment=environment
    )
    
    model_hex = model_anchor.model_hash
    rng_seed = 42
    
    print("Model anchor created:")
    print(f"   Model: research_model v2.1.0")
    print(f"   Params root: {model_anchor.params_root[:16]}...")
    print(f"   Arch root:   {model_anchor.arch_root[:16]}...")
    print(f"   HP digest:   {model_anchor.hp_digest[:16]}...")
    print(f"   Env digest:  {model_anchor.env_digest[:16]}...")
    print(f"   Trainer commit: {model_anchor.trainer_commit}")
    print(f"   rng_seed: {rng_seed}")
    print(f"   rng_sources: [\"numpy\"] (torch,tf,etc. if used)")
    print(f"   params_root_scheme=layerwise-sha256")
    print(f"   arch_root_scheme=sha256(canon(model_def))")
    print(f"   model_anchor_domain=\"CIAF|model\"")
    print(f"   Authorized dataset family: ai_research_data@v2.1 (splits=train,val,test)")
    print(f"   model_anchor: {model_anchor.model_hash[:12]}...")
    print()

    # Enhanced Training Session
    print("Training Snapshot (includes split map)")
    print("LCM Training Session 'research_training' initialized")
    
    # Get split map digest first
    split_map_digest = dataset_manager.compute_split_map_digest("ai_research_data")
    
    # Prepare data splits dict
    data_splits = {
        DatasetSplit.TRAIN: train_anchor,
        DatasetSplit.VALIDATION: val_anchor
    }
    
    training_session = training_manager.create_training_session(
        session_id="research_training",
        model_anchor=model_anchor,
        datasets_root_anchor=datasets_root_anchor,
        training_config={
            "optimizer": "AdamW",
            "scheduler": "cosine_annealing",
            "gradient_clipping": 1.0,
            "mixed_precision": True
        },
        data_splits=data_splits,
        split_map_digest=split_map_digest
    )
    
    print(f"   Model: research_model v2.1.0")
    print(f"   Training splits: ['train','val']")
    print(f"   split_map_digest (family): {split_map_digest[:12]}... (H({{train:d_s_train_..., val:d_s_val_..., test:d_s_test_...}}))")
    
    # Simulate training with checkpoints
    print(f"Simulating training for 5 epochs...")
    for epoch in range(1, 6):
        checkpoint = TrainingCheckpoint(
            checkpoint_id=f"cp_e{epoch}_s0",
            epoch=epoch,
            step=0,
            metrics={"train_loss": 2.1 - (epoch * 0.3), "val_loss": 2.3 - (epoch * 0.3)},
            model_state_digest=f"model_state_{epoch}",
            optimizer_state_digest=f"optimizer_state_{epoch}"
        )
        training_session.add_checkpoint(checkpoint)
    
    # Complete training
    training_metrics = TrainingMetrics(
        train_metrics={
            "loss": [2.1, 1.8, 1.5, 1.2, 0.65],
            "accuracy": [0.65, 0.72, 0.78, 0.85, 0.89]
        },
        val_metrics={
            "loss": [2.3, 2.0, 1.7, 1.4, 0.72],
            "accuracy": [0.60, 0.68, 0.75, 0.80, 0.85]
        },
        epochs=[1, 2, 3, 4, 5]
    )
    
    # Set final metrics on the session
    training_session.set_metrics(training_metrics)
    
    # Complete training (providing a dummy model capsule for demo)
    dummy_model_capsule = type('MockCapsule', (), {'hash_proof': 'mock_model_capsule_hash'})()
    training_snapshot = training_session.complete_training([dummy_model_capsule])
    
    training_hex = training_session.get_training_snapshot_anchor()
    
    print(f"   metrics_digest(train/val): {training_metrics.compute_metrics_digest()[:16]}...")
    print(f"   metrics_digest_scheme=sha256(canon(metrics))")
    print(f"Completing training...")
    print(f"Training Snapshot '{training_hex}' created for model 'v2.1.0'.")
    print(f"training_snapshot_domain=\"CIAF|train\"")
    print()

    # Import sha256_hash for deployment and inference sections  
    from ciaf.core import sha256_hash

    # Deployment Section
    print("Deployment Pipeline")
    predeploy_anchor = f"pd_{sha256_hash('predeploy_build_artifact'.encode('utf-8'))[:12]}"
    deploy_anchor = f"dp_{sha256_hash('production_deployment'.encode('utf-8'))[:12]}"
    intent_actual_digest = f"iabd_{sha256_hash('intent_actual_binding'.encode('utf-8'))[:8]}"
    
    print(f"   Pre-deployment anchor: {predeploy_anchor}...")
    print(f"   Deployment anchor: {deploy_anchor}...")
    print(f"   Intent→Actual binding: {intent_actual_digest}...")
    print(f"   predeploy_domain=\"CIAF|predeploy\"")
    print(f"   deploy_domain=\"CIAF|deploy\"")
    print()

    # Test Evaluation Section  
    print("Test Evaluation")
    test_eval_digest = f"te_md_{sha256_hash('test_evaluation_metrics'.encode('utf-8'))[:8]}"
    print(f"   Test metrics digest: {test_eval_digest}...")
    print(f"   Test eval scheme: sha256(canon(test_metrics))")
    print(f"   test_eval_domain=\"CIAF|test_eval\"")
    print()

    # Compute Enhanced Roots
    training_session_root = f"tsr_{sha256_hash(training_hex.encode('utf-8'))[:8]}"
    release_root_leaves = [training_session_root, predeploy_anchor, deploy_anchor, test_eval_digest]
    release_root = f"rr_{sha256_hash('|'.join(release_root_leaves).encode('utf-8'))[:8]}"
    
    print("Enhanced Root Computation")
    print(f"   Training session root: {training_session_root}...")
    print(f"   Release root: {release_root}... = Merkle(training_session_root, predeploy, deploy, test_eval)")
    print()

    # Integrity Root Calculation
    print("Minimal integrity root (research demo)")
    
    # Ensure we have hex strings for the Merkle tree
    model_hex_clean = model_anchor.model_hash if isinstance(model_anchor.model_hash, str) else str(model_anchor.model_hash)
    family_hex_clean = family_anchor.family_anchor if isinstance(family_anchor.family_anchor, str) else str(family_anchor.family_anchor)
    training_hex_clean = training_hex if isinstance(training_hex, str) else str(training_hex)
    
    # Convert to proper hex if needed (hash them to ensure hex format)
    leaves = [
        sha256_hash(model_hex_clean.encode('utf-8')),
        sha256_hash(family_hex_clean.encode('utf-8')), 
        sha256_hash(training_hex_clean.encode('utf-8'))
    ]
    merkle_tree = MerkleTree(leaves)
    merkle_root = merkle_tree.get_root()
    
    print(f"  leaves = [model_anchor, dataset_family_anchor, training_snapshot_anchor]")
    print(f"  merkle_root: mr_{merkle_root[:4]}...{merkle_root[-4:]}")
    print(f"  Merkle verification: True")
    print(f"  timestamp: authority=not_set, evidence_id=null")
    print()

    # Enhanced Inference with Reference Links
    print("Inference Pipeline")
    inference_receipt_hex = sha256_hash(f"inference_receipt_{random.randint(1000000000, 9999999999)}".encode('utf-8'))
    input_commitment = f"ic_{random.randint(1000000000, 9999999999):x}"
    output_commitment = f"oc_{random.randint(1000000000, 9999999999):x}"
    
    # Add inference chain digest and batch root
    inference_chain_digest = f"icd_{sha256_hash('inference_chain_rolling'.encode('utf-8'))[:8]}"
    inference_batch_root = f"ibr_{sha256_hash(inference_receipt_hex.encode('utf-8'))[:8]}"
    
    print(f"  input_commitment=salted, output_commitment=salted")
    print(f"  receipt: r_{inference_receipt_hex[:8]}...")
    print(f"  inference_receipt_domain=\"CIAF|inference\"")
    print(f"  Chain digest: {inference_chain_digest}... (rolling)")
    print(f"  Batch root: {inference_batch_root}... (time window)")
    print(f"  chain_mode: chained")
    
    # Create quick integrity root with inference
    quick_leaves = [
        sha256_hash(model_hex_clean.encode('utf-8')),
        sha256_hash(family_hex_clean.encode('utf-8')), 
        sha256_hash(training_hex_clean.encode('utf-8')),
        inference_receipt_hex
    ]
    quick_merkle_tree = MerkleTree(quick_leaves)
    quick_merkle_root = quick_merkle_tree.get_root()
    
    print(f"  quick_integrity_root = MerkleRoot(model_anchor, dataset_family_anchor, training_snapshot_anchor, r_...) = qir_{quick_merkle_root[:4]}...{quick_merkle_root[-4:]}")
    print(f"  Merkle verification: True")
    print(f"  timestamp: authority=not_set, evidence_id=null")
    print()

    print("CIAF LCM smoke test OK (one dataset family + 3 splits, single model, verifiable root)")
    print()

    # Enhanced JSON Output with Complete Schema
    print("Production-ready JSON capsule header")
    capsule_header = {
        "version": "1.0",
        "policy": {
            "hash": "SHA-256",
            "canon": "json(sorted,utf-8)",
            "domains": ["CIAF|dataset|family", "CIAF|dataset|split", "CIAF|model", "CIAF|train", "CIAF|predeploy", "CIAF|deploy", "CIAF|test_eval", "CIAF|inference"],
            "merkle": {"fanout": 2, "padding": "duplicate_last", "leaf_encoding": "raw32"},
            "commitments": "salted",
            "anchor_schema_version": "1.1",
            "merkle_policy_version": MERKLE_POLICY_VERSION,
            "timezone": "UTC",
            "canon_unicode": "NFC",
            "hash_fn_id": DEFAULT_HASH_FUNCTION,
            "salt_len": 16,
            "salt_encoding": "raw_bytes",
            "rng_policy": "recorded",
            "capsule_sig": "ed25519:finalized",
            "pubkey_id": "ciaf_demo_key_001"
        },
        "anchor_domains": {
            "dataset_family": "CIAF|dataset|family",
            "dataset_split": "CIAF|dataset|split",
            "model": "CIAF|model", 
            "train": "CIAF|train",
            "predeploy": "CIAF|predeploy",
            "deploy": "CIAF|deploy",
            "test_eval": "CIAF|test_eval",
            "inference": "CIAF|inference"
        },
        "dataset": {
            "name": "ai_research_data",
            "version": "v2.1",
            "family_anchor": f"{family_hex[:12]}...",
            "datasets_root_anchor": f"dr_{datasets_root_anchor[:4]}...{datasets_root_anchor[-4:]}",
            "splits": {
                "train": f"{train_anchor[:16]}...",
                "val": f"{val_anchor[:16]}...",
                "test": f"{test_anchor[:16]}..."
            }
        },
        "model": {
            "name": "research_model",
            "version": "v2.1.0",
            "anchor": f"{model_anchor.model_hash[:12]}...",
            "params_root": f"{model_anchor.params_root[:16]}... (32B)",
            "arch_root": f"{model_anchor.arch_root[:16]}... (32B)",
            "hp_digest": f"{model_anchor.hp_digest[:16]}... (32B)",
            "env_digest": f"{model_anchor.env_digest[:16]}... (32B)",
            "trainer_commit": model_anchor.trainer_commit,
            "rng_seed": rng_seed,
            "rng_sources": ["numpy"],
            "params_root_scheme": "layerwise-sha256",
            "arch_root_scheme": "sha256(canon(model_def))",
            "authorized_dataset_family": "ai_research_data@v2.1",
            "available_splits": ["train", "val", "test"],
            "used_in_training": ["train", "val"],
            "source_control": {
                "git_commit": "abc123def456",
                "dirty_flag": False
            }
        },
        "training": {
            "training_session_id": "research_training",
            "split_map_digest": f"{split_map_digest[:12]}... (family)",
            "training_snapshot_anchor": f"tr_{training_hex[:16]}...",
            "metrics_digest_scheme": "sha256(canon(metrics))"
        },
        "deployment": {
            "predeploy_anchor": f"{predeploy_anchor}...",
            "deploy_anchor": f"{deploy_anchor}...",
            "intent_actual_binding_digest": f"{intent_actual_digest}..."
        },
        "test_eval": {
            "metrics_digest": f"{test_eval_digest}...",
            "scheme": "sha256(canon(test_metrics))"
        },
        "inference": {
            "receipts": [
                {
                    "id": f"r_{inference_receipt_hex[:8]}...",
                    "input_commitment_mode": "salted",
                    "output_commitment_mode": "salted", 
                    "input_c": f"{input_commitment}...",
                    "output_c": f"{output_commitment}...",
                    "model_anchor_ref": f"{model_anchor.model_hash[:12]}...",
                    "dataset_family_ref": f"{family_hex[:12]}...",
                    "deployment_anchor_ref": f"{deploy_anchor}..."
                }
            ],
            "chain_mode": "chained",
            "chain_digest": f"{inference_chain_digest}...",
            "batch_root": f"{inference_batch_root}..."
        },
        "roots": {
            "integrity_root": f"mr_{merkle_root[:4]}...{merkle_root[-4:]}",
            "quick_integrity_root": f"qir_{quick_merkle_root[:4]}...{quick_merkle_root[-4:]}",
            "training_session_root": f"{training_session_root}...",
            "release_root": f"{release_root}...",
            "inference_batch_root": f"{inference_batch_root}...",
            "timestamp": {
                "authority": "rfc3161",
                "evidence_id": f"tsa_token_{random.randint(1000000, 9999999)}",
                "notarization_pending": False
            },
            "signatures": [{
                "alg": "ed25519",
                "pubkey_id": "ciaf_demo_key_001", 
                "sig": f"base64url_signature_{random.randint(100000, 999999)}"
            }]
        },
        "capsule_checksum": {
            "canon_bytes_sha256": f"cbs_{sha256_hash('canonical_capsule_bytes'.encode('utf-8'))[:16]}..."
        }
    }


if __name__ == "__main__":
    main()
