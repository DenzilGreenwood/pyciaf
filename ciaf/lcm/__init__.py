"""
CIAF Lazy Capsule Materialization (LCM) System

Complete end-to-end lazy capsule materialization for AI models with proper anchoring,
dataset families with splits, deployment stages, and audit trails.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

from .policy import LCMPolicy, CommitmentType, DomainType, MerklePolicy, get_default_policy
from .dataset_family_manager import LCMDatasetFamilyManager, LCMDatasetFamilyAnchor, LCMDatasetSplitAnchor, DatasetFamilyMetadata, DatasetSplit
from .dataset_manager import LCMDatasetManager, LCMDatasetAnchor, DatasetMetadata
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
    "LCMDatasetFamilyManager",
    "LCMDatasetFamilyAnchor",
    "LCMDatasetSplitAnchor", 
    "DatasetFamilyMetadata",
    "DatasetSplit",
    "LCMDatasetManager",
    "LCMDatasetAnchor",
    "DatasetMetadata",
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
