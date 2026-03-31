"""
CIAF LCM (Lazy Capsule Materialization) Module Tests

Comprehensive test suite for the CIAF LCM framework:
- Capsule headers and lifecycle stages
- Dataset anchoring and split management
- Model anchoring and versioning
- Training session tracking
- Deployment management
- Inference receipts
- Policy configuration
- Merkle trees and roots

Created: 2026-03-31
Version: 1.0.0
"""

import pytest
from datetime import datetime
import hashlib

from ciaf.lcm import (
    CapsuleHeader,
    LCMCapsuleManager,
    LCMPolicy,
    DomainType,
    CommitmentType,
    MerklePolicy,
    get_default_policy,
)
from ciaf.lcm.dataset_manager import (
    LCMDatasetManager,
    LCMDatasetAnchor,
    DatasetSplit,
)
from ciaf.lcm.model_manager import LCMModelManager, LCMModelAnchor
from ciaf.lcm.training_manager import (
    LCMTrainingManager,
    LCMTrainingSession,
)
from ciaf.lcm.deployment_manager import (
    LCMDeploymentManager,
    LCMPreDeploymentAnchor,
    LCMDeploymentAnchor,
)
from ciaf.lcm.inference_manager import LCMInferenceManager, LCMInferenceReceipt
from ciaf.lcm.root_manager import LCMRootManager, TestEvaluationAnchor


class TestLCMPolicy:
    """Test LCM policy configuration."""

    def test_default_policy(self):
        """Test default LCM policy creation."""
        policy = get_default_policy()

        assert policy.hash_algorithm == "SHA-256"
        assert policy.canonicalization == "json(sorted,utf-8)"
        assert policy.merkle.fanout == 2
        assert policy.commitments == CommitmentType.SALTED

    def test_custom_policy(self):
        """Test creating custom LCM policy."""
        policy = LCMPolicy(
            hash_algorithm="SHA3-256",
            canonicalization="json(sorted,utf-8)",
            merkle=MerklePolicy(fanout=4, padding="duplicate_last"),
            commitments=CommitmentType.HMAC_SHA256,
        )

        assert policy.hash_algorithm == "SHA3-256"
        assert policy.merkle.fanout == 4
        assert policy.commitments == CommitmentType.HMAC_SHA256

    def test_domain_types(self):
        """Test domain type enumeration."""
        assert DomainType.DATASET.value == "CIAF|dataset"
        assert DomainType.MODEL.value == "CIAF|model"
        assert DomainType.TRAIN.value == "CIAF|train"
        assert DomainType.DEPLOYMENT.value == "CIAF|deployment"
        assert DomainType.INFERENCE.value == "CIAF|inference"

    def test_commitment_types(self):
        """Test commitment type enumeration."""
        assert CommitmentType.SALTED.value == "salted"
        assert CommitmentType.HMAC_SHA256.value == "HMAC-SHA256"
        assert CommitmentType.PLAINTEXT.value == "plaintext"

    def test_policy_to_dict(self):
        """Test converting policy to dictionary."""
        policy = LCMPolicy()
        policy_dict = policy.to_dict()

        assert "hash" in policy_dict
        assert "canon" in policy_dict
        assert "merkle" in policy_dict
        assert policy_dict["hash"] == "SHA-256"


class TestDatasetManager:
    """Test LCM dataset manager."""

    def test_create_dataset_anchor(self):
        """Test creating a dataset anchor."""
        manager = LCMDatasetManager()

        anchor = manager.create_dataset_anchor(
            dataset_id="ds_001",
            dataset_path="test_dataset.csv",
            split_type="train",
        )

        assert anchor.dataset_id == "ds_001"
        assert anchor.split_type == "train"
        assert len(anchor.dataset_hash) == 64  # SHA-256 hex

    def test_dataset_splits(self):
        """Test dataset split enumeration."""
        assert DatasetSplit.TRAIN.value == "train"
        assert DatasetSplit.VALIDATION.value == "val"
        assert DatasetSplit.TEST.value == "test"
        assert DatasetSplit.FULL.value == "full"

    def test_dataset_family(self):
        """Test managing dataset families with train/val/test splits."""
        manager = LCMDatasetManager()

        # Create train split
        train_anchor = manager.create_dataset_anchor(
            dataset_id="ds_001",
            dataset_path="train.csv",
            split_type="train",
        )

        # Create validation split
        val_anchor = manager.create_dataset_anchor(
            dataset_id="ds_001",
            dataset_path="val.csv",
            split_type="val",
        )

        # Create test split
        test_anchor = manager.create_dataset_anchor(
            dataset_id="ds_001",
            dataset_path="test.csv",
            split_type="test",
        )

        assert train_anchor.split_type == "train"
        assert val_anchor.split_type == "val"
        assert test_anchor.split_type == "test"


class TestModelManager:
    """Test LCM model manager."""

    def test_create_model_anchor(self):
        """Test creating a model anchor."""
        manager = LCMModelManager()

        model_params = {
            "layer_1": {"weights": [[0.1, 0.2], [0.3, 0.4]], "bias": [0.1, 0.2]},
            "layer_2": {"weights": [[0.5, 0.6]], "bias": [0.3]},
        }

        anchor = manager.create_model_anchor(
            model_id="model_001",
            model_params=model_params,
        )

        assert anchor.model_id == "model_001"
        assert len(anchor.params_root) == 64  # SHA-256 hex
        assert len(anchor.arch_root) == 64
        assert len(anchor.hp_digest) == 64

    def test_model_versioning(self):
        """Test model versioning with different parameter sets."""
        manager = LCMModelManager()

        params_v1 = {"layer_1": {"weights": [[0.1, 0.2]]}}
        params_v2 = {"layer_1": {"weights": [[0.3, 0.4]]}}

        anchor_v1 = manager.create_model_anchor("model_001", params_v1)
        anchor_v2 = manager.create_model_anchor("model_001", params_v2)

        # Different params should produce different hashes
        assert anchor_v1.params_root != anchor_v2.params_root


class TestTrainingManager:
    """Test LCM training manager."""

    def test_create_training_session(self):
        """Test creating a training session."""
        manager = LCMTrainingManager()

        model_manager = LCMModelManager()
        model_anchor = model_manager.create_model_anchor(
            "model_001",
            {"layer_1": {"weights": [[0.1, 0.2]]}},
        )

        dataset_hash = hashlib.sha256(b"dataset_content").hexdigest()

        session = manager.create_and_run_training_session(
            session_id="train_001",
            model_anchor=model_anchor,
            datasets_root_anchor=dataset_hash,
            training_config={
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10,
            },
        )

        assert session.session_id == "train_001"
        assert session.model_ref == model_anchor.anchor_id
        assert len(session.checkpoints) > 0

    def test_training_checkpoints(self):
        """Test training checkpoints tracking."""
        manager = LCMTrainingManager()

        model_manager = LCMModelManager()
        model_anchor = model_manager.create_model_anchor(
            "model_001",
            {"layer_1": {"weights": [[0.1, 0.2]]}},
        )

        dataset_hash = hashlib.sha256(b"dataset").hexdigest()

        session = manager.create_and_run_training_session(
            session_id="train_002",
            model_anchor=model_anchor,
            datasets_root_anchor=dataset_hash,
            training_config={"epochs": 5},
        )

        # Should have checkpoints for each epoch
        assert len(session.checkpoints) >= 1


class TestDeploymentManager:
    """Test LCM deployment manager."""

    def test_create_predeployment_anchor(self):
        """Test creating a pre-deployment anchor."""
        manager = LCMDeploymentManager()

        model_manager = LCMModelManager()
        model_anchor = model_manager.create_model_anchor(
            "model_001",
            {"layer_1": {"weights": [[0.1, 0.2]]}},
        )

        predeploy = manager.create_predeployment_anchor(
            predeployment_id="predeploy_001",
            model_anchor=model_anchor,
        )

        assert predeploy.predeployment_id == "predeploy_001"
        assert len(predeploy.intent_digest) == 64
        assert len(predeploy.sbom_digest) == 64

    def test_create_deployment_anchor(self):
        """Test creating a deployment anchor."""
        manager = LCMDeploymentManager()

        model_manager = LCMModelManager()
        model_anchor = model_manager.create_model_anchor(
            "model_001",
            {"layer_1": {"weights": [[0.1, 0.2]]}},
        )

        predeploy = manager.create_predeployment_anchor(
            "predeploy_001",
            model_anchor,
        )

        deployment = manager.create_deployment_anchor(
            deployment_id="deploy_001",
            predeployment_anchor=predeploy,
        )

        assert deployment.deployment_id == "deploy_001"
        assert deployment.predeployment_ref == predeploy.anchor_id
        assert len(deployment.actual_digest) == 64

    def test_deployment_environments(self):
        """Test tracking deployment environments."""
        manager = LCMDeploymentManager()

        model_manager = LCMModelManager()
        model_anchor = model_manager.create_model_anchor(
            "model_001",
            {"layer_1": {"weights": [[0.1]]}},
        )

        predeploy = manager.create_predeployment_anchor("predeploy", model_anchor)

        # Deploy to staging
        staging_deploy = manager.create_deployment_anchor(
            deployment_id="deploy_staging",
            predeployment_anchor=predeploy,
        )

        # Deploy to production
        prod_deploy = manager.create_deployment_anchor(
            deployment_id="deploy_production",
            predeployment_anchor=predeploy,
        )

        assert staging_deploy.deployment_id != prod_deploy.deployment_id


class TestInferenceManager:
    """Test LCM inference manager."""

    def test_create_inference_receipt(self):
        """Test creating an inference receipt."""
        manager = LCMInferenceManager()

        model_manager = LCMModelManager()
        model_anchor = model_manager.create_model_anchor(
            "model_001",
            {"layer_1": {"weights": [[0.1]]}},
        )

        deploy_manager = LCMDeploymentManager()
        predeploy = deploy_manager.create_predeployment_anchor(
            "predeploy_001",
            model_anchor,
        )
        deployment = deploy_manager.create_deployment_anchor(
            "deploy_001",
            predeploy,
        )

        receipt = manager.create_inference_receipt(
            inference_id="inf_001",
            model_anchor=model_anchor,
            deployment_anchor=deployment,
        )

        assert receipt.inference_id == "inf_001"
        assert len(receipt.receipt_hash) == 64

    def test_inference_chaining(self):
        """Test inference receipt chaining."""
        manager = LCMInferenceManager()

        model_manager = LCMModelManager()
        model_anchor = model_manager.create_model_anchor("model", {"l": [[0.1]]})

        deploy_manager = LCMDeploymentManager()
        predeploy = deploy_manager.create_predeployment_anchor("pre", model_anchor)
        deployment = deploy_manager.create_deployment_anchor("deploy", predeploy)

        # First inference
        receipt1 = manager.create_inference_receipt(
            "inf_001",
            model_anchor,
            deployment,
        )

        # Second inference (chained to first)
        receipt2 = manager.create_inference_receipt(
            "inf_002",
            model_anchor,
            deployment,
            prev_receipt_hash=receipt1.receipt_hash,
        )

        assert receipt2.prev_receipt_hash == receipt1.receipt_hash


class TestRootManager:
    """Test LCM root manager for Merkle roots."""

    def test_compute_training_session_root(self):
        """Test computing training session Merkle root."""
        root_manager = LCMRootManager()

        training_manager = LCMTrainingManager()
        model_manager = LCMModelManager()

        model_anchor = model_manager.create_model_anchor("model", {"l": [[0.1]]})
        dataset_hash = hashlib.sha256(b"dataset").hexdigest()

        session = training_manager.create_and_run_training_session(
            "train_001",
            model_anchor,
            dataset_hash,
            {"epochs": 3},
        )

        root_hash = root_manager.compute_training_session_root(session)

        assert len(root_hash) == 64  # SHA-256 hex

    def test_compute_release_root(self):
        """Test computing release Merkle root."""
        root_manager = LCMRootManager()
        training_manager = LCMTrainingManager()
        model_manager = LCMModelManager()
        deploy_manager = LCMDeploymentManager()

        # Create training session
        model_anchor = model_manager.create_model_anchor("model", {"l": [[0.1]]})
        dataset_hash = hashlib.sha256(b"dataset").hexdigest()
        session = training_manager.create_and_run_training_session(
            "train",
            model_anchor,
            dataset_hash,
            {},
        )

        # Create deployment
        predeploy = deploy_manager.create_predeployment_anchor("pre", model_anchor)
        deployment = deploy_manager.create_deployment_anchor("deploy", predeploy)

        # Create test evaluation
        test_eval = root_manager.run_test_evaluation(
            "test_001",
            dataset_hash[:16],
        )

        training_root = root_manager.compute_training_session_root(session)

        release_root = root_manager.compute_release_root(
            training_root,
            predeploy,
            deployment,
            test_eval,
        )

        assert len(release_root) == 64

    def test_compute_inference_batch_root(self):
        """Test computing inference batch Merkle root."""
        root_manager = LCMRootManager()

        receipt_hashes = [
            hashlib.sha256(f"receipt_{i}".encode()).hexdigest() for i in range(5)
        ]

        batch_root = root_manager.compute_inference_batch_root(
            "batch_001",
            receipt_hashes,
        )

        assert len(batch_root) == 64


class TestCapsuleHeader:
    """Test CapsuleHeader data model."""

    def test_create_capsule_header(self):
        """Test creating a capsule header."""
        capsule = CapsuleHeader(
            capsule_version="1.0.0",
            generated_at=datetime.now().isoformat(),
            policy={"hash_algorithm": "SHA-256"},
            stage_a_dataset={"stage": "A", "description": "Dataset"},
        )

        assert capsule.capsule_version == "1.0.0"
        assert capsule.policy["hash_algorithm"] == "SHA-256"
        assert capsule.stage_a_dataset["stage"] == "A"

    def test_capsule_to_json(self):
        """Test converting capsule to JSON."""
        capsule = CapsuleHeader(
            capsule_version="1.0.0",
            generated_at="2026-03-31T10:00:00Z",
            policy={"hash": "SHA-256"},
        )

        json_str = capsule.to_json()
        assert "capsule_version" in json_str
        assert "1.0.0" in json_str

    def test_capsule_to_compact_json(self):
        """Test converting capsule to compact JSON."""
        capsule = CapsuleHeader(
            capsule_version="1.0.0",
            generated_at="2026-03-31T10:00:00Z",
            policy={},
        )

        compact = capsule.to_compact_json()
        assert "\n" not in compact  # No newlines in compact format


class TestLCMCapsuleManager:
    """Test LCM capsule manager."""

    def test_create_capsule_manager(self):
        """Test creating a capsule manager."""
        manager = LCMCapsuleManager()

        assert manager.policy is not None
        assert manager.dataset_manager is not None
        assert manager.model_manager is not None

    def test_create_simple_capsule_header(self):
        """Test creating a simple capsule header."""
        manager = LCMCapsuleManager()

        # Create dataset anchor
        dataset_anchor = manager.dataset_manager.create_dataset_anchor(
            "ds_001",
            "test.csv",
            "train",
        )

        # Create model anchor
        model_anchor = manager.model_manager.create_model_anchor(
            "model_001",
            {"layer": {"weights": [[0.1]]}},
        )

        # Create capsule header
        capsule = manager.create_capsule_header(
            dataset_anchor=dataset_anchor,
            model_anchor=model_anchor,
        )

        assert capsule.stage_a_dataset is not None
        assert capsule.stage_b_model is not None
        assert capsule.stage_a_dataset["stage"] == "A"
        assert capsule.stage_b_model["stage"] == "B"

    def test_create_comprehensive_capsule(self):
        """Test creating comprehensive capsule with all stages."""
        manager = LCMCapsuleManager()

        capsule = manager.create_comprehensive_capsule(
            dataset_path="mock_dataset.csv",
            model_params={"layer": {"weights": [[0.1]]}},
            training_config={"epochs": 2},
        )

        # Verify all stages are present
        assert capsule.stage_a_dataset is not None
        assert capsule.stage_b_model is not None
        assert capsule.stage_c_training is not None
        assert capsule.stage_d_predeployment is not None
        assert capsule.stage_e_deployment is not None
        assert capsule.stage_f_test_evaluation is not None
        assert capsule.stage_g_inference is not None
        assert capsule.stage_h_roots is not None


class TestLCMWorkflowScenarios:
    """Test real-world LCM workflow scenarios."""

    def test_complete_ml_lifecycle(self):
        """Test complete ML lifecycle from dataset to inference."""
        policy = LCMPolicy()

        # Stage A: Dataset Anchoring
        dataset_manager = LCMDatasetManager(policy)
        train_anchor = dataset_manager.create_dataset_anchor(
            "ds_001",
            "train.csv",
            "train",
        )
        val_anchor = dataset_manager.create_dataset_anchor(
            "ds_001",
            "val.csv",
            "val",
        )

        # Stage B: Model Anchoring
        model_manager = LCMModelManager(policy)
        model_anchor = model_manager.create_model_anchor(
            "model_001",
            {"layer_1": {"weights": [[0.1, 0.2]], "bias": [0.1]}},
        )

        # Stage C: Training
        training_manager = LCMTrainingManager(policy)
        session = training_manager.create_and_run_training_session(
            "train_001",
            model_anchor,
            train_anchor.dataset_hash,
            {"epochs": 5, "learning_rate": 0.001},
        )

        # Stage D & E: Deployment
        deploy_manager = LCMDeploymentManager(policy)
        predeploy = deploy_manager.create_predeployment_anchor(
            "predeploy_001",
            model_anchor,
        )
        deployment = deploy_manager.create_deployment_anchor(
            "deploy_001",
            predeploy,
        )

        # Stage F: Test Evaluation
        root_manager = LCMRootManager(policy)
        test_eval = root_manager.run_test_evaluation(
            "test_001",
            val_anchor.dataset_hash[:16],
        )

        # Stage G: Inference
        inference_manager = LCMInferenceManager(policy)
        receipt = inference_manager.create_inference_receipt(
            "inf_001",
            model_anchor,
            deployment,
        )

        # Stage H: Compute roots
        training_root = root_manager.compute_training_session_root(session)
        release_root = root_manager.compute_release_root(
            training_root,
            predeploy,
            deployment,
            test_eval,
        )

        # Verify all stages completed
        assert train_anchor is not None
        assert model_anchor is not None
        assert session is not None
        assert deployment is not None
        assert test_eval is not None
        assert receipt is not None
        assert len(release_root) == 64

    def test_model_versioning_workflow(self):
        """Test versioning multiple models."""
        model_manager = LCMModelManager()

        # Version 1.0
        model_v1 = model_manager.create_model_anchor(
            "model_classifier",
            {"layer_1": {"weights": [[0.1, 0.2]]}},
        )

        # Version 2.0 (improved)
        model_v2 = model_manager.create_model_anchor(
            "model_classifier",
            {"layer_1": {"weights": [[0.3, 0.4]], "bias": [0.1]}},
        )

        # Different model versions have different hashes
        assert model_v1.params_root != model_v2.params_root

    def test_multi_environment_deployment(self):
        """Test deploying to multiple environments."""
        model_manager = LCMModelManager()
        deploy_manager = LCMDeploymentManager()

        model = model_manager.create_model_anchor("model", {"l": [[0.1]]})
        predeploy = deploy_manager.create_predeployment_anchor("pre", model)

        # Deploy to dev
        dev_deploy = deploy_manager.create_deployment_anchor("deploy_dev", predeploy)

        # Deploy to staging
        staging_deploy = deploy_manager.create_deployment_anchor(
            "deploy_staging",
            predeploy,
        )

        # Deploy to production
        prod_deploy = deploy_manager.create_deployment_anchor("deploy_prod", predeploy)

        assert dev_deploy.deployment_id != staging_deploy.deployment_id
        assert staging_deploy.deployment_id != prod_deploy.deployment_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
