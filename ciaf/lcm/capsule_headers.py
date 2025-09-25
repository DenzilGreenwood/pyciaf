"""
CIAF LCM Capsule Headers

Implements compact JSON "capsule headers" that provide comprehensive CIAF LCM state
including anchors, policy, and Merkle roots in a standardized JSON format.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from .policy import LCMPolicy, get_default_policy, DomainType, CommitmentType, MerklePolicy
from .dataset_manager import LCMDatasetAnchor, LCMDatasetManager
from .model_manager import LCMModelAnchor, LCMModelManager
from .training_manager import LCMTrainingSession, LCMTrainingManager
from .deployment_manager import LCMPreDeploymentAnchor, LCMDeploymentAnchor, LCMDeploymentManager
from .inference_manager import LCMInferenceReceipt, LCMInferenceManager
from .root_manager import TestEvaluationAnchor, LCMRootManager


@dataclass
class CapsuleHeader:
    """
    Compact JSON capsule header containing comprehensive CIAF LCM state.
    """
    # Core metadata
    capsule_version: str
    generated_at: str
    policy: Dict[str, Any]
    
    # Canonical stages
    stage_a_dataset: Optional[Dict[str, Any]] = None
    stage_b_model: Optional[Dict[str, Any]] = None
    stage_c_training: Optional[Dict[str, Any]] = None
    stage_d_predeployment: Optional[Dict[str, Any]] = None
    stage_e_deployment: Optional[Dict[str, Any]] = None
    stage_f_test_evaluation: Optional[Dict[str, Any]] = None
    stage_g_inference: Optional[Dict[str, Any]] = None
    stage_h_roots: Optional[Dict[str, Any]] = None
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to pretty-printed JSON."""
        return json.dumps(asdict(self), indent=indent, sort_keys=True)
    
    def to_compact_json(self) -> str:
        """Convert to compact JSON."""
        return json.dumps(asdict(self), separators=(',', ':'), sort_keys=True)


class LCMCapsuleManager:
    """
    Manager for creating and managing CIAF LCM capsule headers.
    """
    
    def __init__(self, policy: LCMPolicy = None):
        """Initialize capsule manager."""
        self.policy = policy or get_default_policy()
        self.dataset_manager = LCMDatasetManager(self.policy)
        self.model_manager = LCMModelManager(self.policy)
        self.training_manager = LCMTrainingManager(self.policy)
        self.deployment_manager = LCMDeploymentManager(self.policy)
        self.inference_manager = LCMInferenceManager(self.policy)
        self.root_manager = LCMRootManager(self.policy)
    
    def create_capsule_header(
        self,
        dataset_anchor: Optional[LCMDatasetAnchor] = None,
        model_anchor: Optional[LCMModelAnchor] = None,
        training_session: Optional[LCMTrainingSession] = None,
        predeployment_anchor: Optional[LCMPreDeploymentAnchor] = None,
        deployment_anchor: Optional[LCMDeploymentAnchor] = None,
        test_evaluation_anchor: Optional[TestEvaluationAnchor] = None,
        inference_receipt: Optional[LCMInferenceReceipt] = None,
        training_session_root: Optional[str] = None,
        release_root: Optional[str] = None,
        inference_batch_root: Optional[str] = None
    ) -> CapsuleHeader:
        """
        Create comprehensive capsule header from LCM components.
        
        Args:
            dataset_anchor: Dataset anchor
            model_anchor: Model anchor
            training_session: Training session
            predeployment_anchor: Pre-deployment anchor
            deployment_anchor: Deployment anchor
            test_evaluation_anchor: Test evaluation anchor
            inference_receipt: Inference receipt
            training_session_root: Training session root hash
            release_root: Release root hash
            inference_batch_root: Inference batch root hash
            
        Returns:
            CapsuleHeader instance
        """
        print("📦 Creating CIAF LCM capsule header...")
        
        # Policy information
        policy_dict = {
            "hash_algorithm": self.policy.hash_algorithm,
            "canonicalization": self.policy.canonicalization,
            "domain_types": [dt.value for dt in DomainType],
            "commitment_types": [ct.value for ct in CommitmentType],
            "merkle_policy": {
                "fanout": self.policy.merkle_policy.fanout,
                "padding_strategy": self.policy.merkle_policy.padding_strategy
            }
        }
        
        # Stage A: Dataset anchoring
        stage_a = None
        if dataset_anchor:
            stage_a = {
                "stage": "A",
                "description": "Dataset Anchoring",
                "split_type": dataset_anchor.split_type,
                "dataset_digest": dataset_anchor.dataset_hash[:8] + "...",
                "sample_count": dataset_anchor.metadata.sample_count if dataset_anchor.metadata else "unknown",
                "anchor_id": dataset_anchor.anchor_id
            }
        
        # Stage B: Model anchoring
        stage_b = None
        if model_anchor:
            stage_b = {
                "stage": "B",
                "description": "Model Anchoring",
                "params_root": model_anchor.params_root[:8] + "...",
                "arch_root": model_anchor.arch_root[:8] + "...",
                "hp_digest": model_anchor.hp_digest[:8] + "...",
                "env_digest": model_anchor.env_digest[:8] + "...",
                "trainer_commit": model_anchor.trainer_commit[:8] + "...",
                "anchor_id": model_anchor.anchor_id
            }
        
        # Stage C: Training
        stage_c = None
        if training_session:
            stage_c = {
                "stage": "C",
                "description": "Training Session",
                "session_id": training_session.session_id,
                "checkpoint_count": len(training_session.checkpoints),
                "final_loss": training_session.metrics.final_loss if training_session.metrics else "unknown",
                "training_snapshot": training_session.training_snapshot.merkle_root_hash[:8] + "..." if training_session.training_snapshot else "not_completed"
            }
        
        # Stage D: Pre-deployment
        stage_d = None
        if predeployment_anchor:
            stage_d = {
                "stage": "D",
                "description": "Pre-deployment",
                "predeployment_id": predeployment_anchor.predeployment_id,
                "intent_digest": predeployment_anchor.intent_digest[:8] + "...",
                "sbom_digest": predeployment_anchor.sbom_digest[:8] + "...",
                "artifacts_count": len(predeployment_anchor.artifacts),
                "anchor_id": predeployment_anchor.anchor_id
            }
        
        # Stage E: Deployment
        stage_e = None
        if deployment_anchor:
            stage_e = {
                "stage": "E",
                "description": "Deployment",
                "deployment_id": deployment_anchor.deployment_id,
                "actual_digest": deployment_anchor.actual_digest[:8] + "...",
                "environment": deployment_anchor.actual_environment,
                "location": deployment_anchor.actual_location,
                "anchor_id": deployment_anchor.anchor_id
            }
        
        # Stage F: Test evaluation
        stage_f = None
        if test_evaluation_anchor:
            stage_f = {
                "stage": "F",
                "description": "Test Evaluation",
                "test_id": test_evaluation_anchor.test_id,
                "evaluation_type": test_evaluation_anchor.evaluation_type,
                "test_metrics_digest": test_evaluation_anchor.test_metrics_digest[:8] + "...",
                "metrics": test_evaluation_anchor.metrics,
                "anchor_id": test_evaluation_anchor.anchor_id
            }
        
        # Stage G: Inference
        stage_g = None
        if inference_receipt:
            stage_g = {
                "stage": "G",
                "description": "Inference Receipt",
                "inference_id": inference_receipt.inference_id,
                "inference_type": inference_receipt.inference_type,
                "receipt_hash": inference_receipt.receipt_hash[:8] + "...",
                "prev_receipt_hash": inference_receipt.prev_receipt_hash[:8] + "..." if inference_receipt.prev_receipt_hash else "genesis"
            }
        
        # Stage H: Merkle roots
        stage_h = None
        if any([training_session_root, release_root, inference_batch_root]):
            stage_h = {
                "stage": "H",
                "description": "Merkle Roots & Publication",
                "training_session_root": training_session_root[:8] + "..." if training_session_root else None,
                "release_root": release_root[:8] + "..." if release_root else None,
                "inference_batch_root": inference_batch_root[:8] + "..." if inference_batch_root else None,
                "timestamp_authority": "not_set",
                "evidence_id": "null"
            }
        
        # Create capsule header
        capsule = CapsuleHeader(
            capsule_version="1.0.0",
            generated_at=datetime.now().isoformat(),
            policy=policy_dict,
            stage_a_dataset=stage_a,
            stage_b_model=stage_b,
            stage_c_training=stage_c,
            stage_d_predeployment=stage_d,
            stage_e_deployment=stage_e,
            stage_f_test_evaluation=stage_f,
            stage_g_inference=stage_g,
            stage_h_roots=stage_h
        )
        
        print("✅ Capsule header created successfully")
        return capsule
    
    def create_comprehensive_capsule(
        self,
        dataset_path: str = "mock_dataset.csv",
        model_params: Dict[str, Any] = None,
        training_config: Dict[str, Any] = None
    ) -> CapsuleHeader:
        """
        Create comprehensive capsule header by simulating full LCM flow.
        
        Args:
            dataset_path: Path to dataset
            model_params: Model parameters
            training_config: Training configuration
            
        Returns:
            CapsuleHeader with all stages populated
        """
        print("🚀 Creating comprehensive CIAF LCM capsule header...")
        
        # Default parameters
        if model_params is None:
            model_params = {"layer_1": {"weights": [[0.1, 0.2]], "bias": [0.1]}}
        
        if training_config is None:
            training_config = {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10,
                "optimizer": "adam"
            }
        
        # Stage A: Dataset anchoring
        print("\n📊 Stage A: Dataset Anchoring")
        dataset_anchor = self.dataset_manager.create_dataset_anchor(
            dataset_id="ds_001",
            dataset_path=dataset_path,
            split_type="train"
        )
        
        # Stage B: Model anchoring
        print("\n🤖 Stage B: Model Anchoring")
        model_anchor = self.model_manager.create_model_anchor(
            model_id="model_001",
            model_params=model_params
        )
        
        # Stage C: Training
        print("\n🎯 Stage C: Training Session")
        training_session = self.training_manager.create_and_run_training_session(
            session_id="train_001",
            model_anchor=model_anchor,
            datasets_root_anchor=dataset_anchor.dataset_hash,
            training_config=training_config
        )
        
        # Stage D: Pre-deployment
        print("\n📋 Stage D: Pre-deployment")
        predeployment_anchor = self.deployment_manager.create_predeployment_anchor(
            predeployment_id="predeploy_001",
            model_anchor=model_anchor
        )
        
        # Stage E: Deployment
        print("\n🚀 Stage E: Deployment")
        deployment_anchor = self.deployment_manager.create_deployment_anchor(
            deployment_id="deploy_001",
            predeployment_anchor=predeployment_anchor
        )
        
        # Stage F: Test evaluation
        print("\n🧪 Stage F: Test Evaluation")
        test_evaluation_anchor = self.root_manager.run_test_evaluation(
            test_id="test_001",
            test_dataset_ref=dataset_anchor.dataset_hash[:16]
        )
        
        # Stage G: Inference
        print("\n🔮 Stage G: Inference Receipt")
        inference_receipt = self.inference_manager.create_inference_receipt(
            inference_id="inf_001",
            model_anchor=model_anchor,
            deployment_anchor=deployment_anchor
        )
        
        # Stage H: Merkle roots
        print("\n🌳 Stage H: Merkle Roots")
        training_session_root = self.root_manager.compute_training_session_root(training_session)
        release_root = self.root_manager.compute_release_root(
            training_session_root, predeployment_anchor, deployment_anchor, test_evaluation_anchor
        )
        inference_batch_root = self.root_manager.compute_inference_batch_root(
            "batch_001", [inference_receipt.receipt_hash]
        )
        
        # Create comprehensive capsule header
        capsule = self.create_capsule_header(
            dataset_anchor=dataset_anchor,
            model_anchor=model_anchor,
            training_session=training_session,
            predeployment_anchor=predeployment_anchor,
            deployment_anchor=deployment_anchor,
            test_evaluation_anchor=test_evaluation_anchor,
            inference_receipt=inference_receipt,
            training_session_root=training_session_root,
            release_root=release_root,
            inference_batch_root=inference_batch_root
        )
        
        print("\n✅ Comprehensive CIAF LCM capsule header created!")
        return capsule
    
    def print_capsule_summary(self, capsule: CapsuleHeader):
        """Print formatted summary of capsule header."""
        print("\n" + "="*60)
        print("📦 CIAF LCM CAPSULE HEADER SUMMARY")
        print("="*60)
        
        print(f"📅 Generated: {capsule.generated_at}")
        print(f"🔧 Version: {capsule.capsule_version}")
        
        # Policy summary
        print(f"\n📋 Policy:")
        print(f"   Hash Algorithm: {capsule.policy['hash_algorithm']}")
        print(f"   Canonicalization: {capsule.policy['canonicalization']}")
        print(f"   Merkle Fanout: {capsule.policy['merkle_policy']['fanout']}")
        
        # Stages summary
        stages = [
            ("A", "Dataset", capsule.stage_a_dataset),
            ("B", "Model", capsule.stage_b_model),
            ("C", "Training", capsule.stage_c_training),
            ("D", "Pre-deploy", capsule.stage_d_predeployment),
            ("E", "Deployment", capsule.stage_e_deployment),
            ("F", "Test Eval", capsule.stage_f_test_evaluation),
            ("G", "Inference", capsule.stage_g_inference),
            ("H", "Roots", capsule.stage_h_roots)
        ]
        
        print(f"\n🎯 Stages:")
        for stage_id, stage_name, stage_data in stages:
            status = "✅" if stage_data else "⏸️"
            print(f"   Stage {stage_id} ({stage_name}): {status}")
        
        print("="*60)
