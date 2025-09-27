"""
CIAF Lazy Capsule Materialization (LCM) System

Complete end-to-end lazy capsule materialization for AI models with proper anchoring,
dataset families with splits, deployment stages, and audit trails.

Created: 2025-09-09
Last Modified: 2025-09-25
Author: Denzil James Greenwood
Version: 1.1.0
"""

from .policy import LCMPolicy, CommitmentType, DomainType, MerklePolicy, get_default_policy, create_commitment, canonical_json, canonical_hash
from .protocol_implementations import (
    DefaultRNG, DefaultMerkle, DefaultAnchorDeriver, InMemoryAnchorStore, DefaultSigner,
    create_default_protocols
)
from .dataset_family_manager import LCMDatasetFamilyManager, LCMDatasetFamilyAnchor, LCMDatasetSplitAnchor, DatasetFamilyMetadata, DatasetSplit
from .dataset_manager import LCMDatasetManager, LCMDatasetAnchor, DatasetMetadata, create_dataset_metadata_from_dataframe
from .model_manager import LCMModelManager, LCMModelAnchor
from .training_manager import LCMTrainingManager, LCMTrainingSession
from .deployment_manager import LCMDeploymentManager, LCMPreDeploymentAnchor, LCMDeploymentAnchor
from .inference_manager import LCMInferenceManager, LCMInferenceReceipt
from .root_manager import LCMRootManager, TestEvaluationAnchor
from .capsule_headers import CapsuleHeader, LCMCapsuleManager

__all__ = [
    "LCMPolicy",
    "CommitmentType", 
    "DomainType",
    "MerklePolicy",
    "get_default_policy",
    "create_commitment",
    "canonical_json",
    "canonical_hash",
    # Protocol implementations
    "DefaultRNG",
    "DefaultMerkle", 
    "DefaultAnchorDeriver",
    "InMemoryAnchorStore",
    "DefaultSigner",
    "create_default_protocols",
    "LCMDatasetFamilyManager",
    "LCMDatasetFamilyAnchor",
    "LCMDatasetSplitAnchor", 
    "DatasetFamilyMetadata",
    "DatasetSplit",
    "LCMDatasetManager",
    "LCMDatasetAnchor",
    "DatasetMetadata",
    "create_dataset_metadata_from_dataframe",
    "LCMModelManager",
    "LCMModelAnchor",
    "LCMTrainingManager",
    "LCMTrainingSession",
    "LCMDeploymentManager",
    "LCMPreDeploymentAnchor",
    "LCMDeploymentAnchor",
    "LCMInferenceManager",
    "LCMInferenceReceipt",
    "LCMRootManager",
    "TestEvaluationAnchor",
    "CapsuleHeader",
    "LCMCapsuleManager"
]
