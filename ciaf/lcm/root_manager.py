"""
CIAF LCM Root Manager

Manages the computation of various Merkle roots for training sessions, releases, 
and inference batches in the CIAF LCM system.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..core import sha256_hash, MerkleTree
from .policy import LCMPolicy, get_default_policy
from .training_manager import LCMTrainingSession
from .deployment_manager import LCMPreDeploymentAnchor, LCMDeploymentAnchor


@dataclass
class TestEvaluationAnchor:
    """Test evaluation anchor for pre/post-deployment testing."""
    test_id: str
    test_metrics_digest: str
    test_dataset_ref: str
    evaluation_type: str  # "pre_deploy" or "post_deploy"
    metrics: Dict[str, float]
    timestamp: str = None
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        
        # Compute test metrics digest if not provided
        if not self.test_metrics_digest:
            metrics_data = {
                "metrics": self.metrics,
                "test_dataset_ref": self.test_dataset_ref,
                "evaluation_type": self.evaluation_type
            }
            canonical_json = json.dumps(metrics_data, sort_keys=True, separators=(',', ':'))
            self.test_metrics_digest = sha256_hash(canonical_json.encode('utf-8'))
    
    @property
    def anchor_id(self) -> str:
        """Get anchor ID."""
        return f"te_{self.test_metrics_digest[:8]}..."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "test_metrics_digest": self.test_metrics_digest[:8] + "...",
            "test_dataset_ref": self.test_dataset_ref,
            "evaluation_type": self.evaluation_type,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "anchor": self.anchor_id
        }


class LCMRootManager:
    """
    Enhanced root manager for computing various Merkle roots in the LCM system.
    """
    
    def __init__(self, policy: LCMPolicy = None):
        """Initialize LCM root manager."""
        self.policy = policy or get_default_policy()
        self.test_evaluations: Dict[str, TestEvaluationAnchor] = {}
        self.computed_roots: Dict[str, str] = {}
    
    def create_test_evaluation_anchor(
        self,
        test_id: str,
        test_dataset_ref: str,
        evaluation_type: str,
        metrics: Dict[str, float]
    ) -> TestEvaluationAnchor:
        """
        Create test evaluation anchor.
        
        Args:
            test_id: Unique test identifier
            test_dataset_ref: Reference to test dataset
            evaluation_type: "pre_deploy" or "post_deploy"
            metrics: Test metrics (accuracy, precision, etc.)
            
        Returns:
            TestEvaluationAnchor instance
        """
        print(f"🧪 Creating test evaluation anchor: {test_id} ({evaluation_type})")
        
        anchor = TestEvaluationAnchor(
            test_id=test_id,
            test_metrics_digest="",  # Will be computed in __post_init__
            test_dataset_ref=test_dataset_ref,
            evaluation_type=evaluation_type,
            metrics=metrics
        )
        
        self.test_evaluations[test_id] = anchor
        
        print(f"✅ Test evaluation anchor created: {anchor.anchor_id}")
        print(f"   📊 Metrics: {metrics}")
        
        return anchor
    
    def compute_training_session_root(
        self,
        training_session: LCMTrainingSession
    ) -> str:
        """
        Compute training session root.
        
        Training Session Root = MerkleRoot([model_anchor, datasets_root_anchor, training_snapshot_anchor])
        
        Args:
            training_session: Training session
            
        Returns:
            Training session root hash
        """
        if not training_session.training_snapshot:
            raise ValueError("Training session not completed yet")
        
        anchor_hashes = [
            training_session.model_anchor.model_hash,
            training_session.datasets_root_anchor,
            training_session.training_snapshot.merkle_root_hash
        ]
        
        merkle_tree = MerkleTree(anchor_hashes)
        root = merkle_tree.get_root()
        
        root_key = f"training_session_{training_session.session_id}"
        self.computed_roots[root_key] = root
        
        return root
    
    def compute_release_root(
        self,
        training_session_root: str,
        predeployment_anchor: LCMPreDeploymentAnchor,
        deployment_anchor: LCMDeploymentAnchor,
        test_evaluation_anchor: TestEvaluationAnchor
    ) -> str:
        """
        Compute release root.
        
        Release Root = MerkleRoot([training_session_root, predeployment_anchor, deployment_anchor, test_evaluation_anchor])
        
        Args:
            training_session_root: Training session root hash
            predeployment_anchor: Pre-deployment anchor
            deployment_anchor: Deployment anchor
            test_evaluation_anchor: Test evaluation anchor
            
        Returns:
            Release root hash
        """
        anchor_hashes = [
            training_session_root,
            predeployment_anchor.predeployment_hash,
            deployment_anchor.deployment_hash,
            test_evaluation_anchor.test_metrics_digest
        ]
        
        merkle_tree = MerkleTree(anchor_hashes)
        root = merkle_tree.get_root()
        
        release_key = f"release_{predeployment_anchor.predeployment_id}_{deployment_anchor.deployment_id}"
        self.computed_roots[release_key] = root
        
        return root
    
    def compute_inference_batch_root(
        self,
        window_id: str,
        receipt_digests: List[str]
    ) -> str:
        """
        Compute inference batch root for a time window.
        
        Args:
            window_id: Time window identifier
            receipt_digests: List of inference receipt digests
            
        Returns:
            Inference batch root hash
        """
        if not receipt_digests:
            return "empty_batch"
        
        merkle_tree = MerkleTree(receipt_digests)
        root = merkle_tree.get_root()
        
        batch_key = f"inference_batch_{window_id}"
        self.computed_roots[batch_key] = root
        
        return root
    
    def get_test_evaluation_anchor(self, test_id: str) -> Optional[TestEvaluationAnchor]:
        """Get test evaluation anchor by ID."""
        return self.test_evaluations.get(test_id)
    
    def get_computed_root(self, root_key: str) -> Optional[str]:
        """Get computed root by key."""
        return self.computed_roots.get(root_key)
    
    def simulate_test_evaluation(
        self,
        test_id: str,
        test_dataset_ref: str,
        evaluation_type: str = "pre_deploy"
    ) -> TestEvaluationAnchor:
        """
        Simulate test evaluation with mock metrics.
        
        Args:
            test_id: Test identifier
            test_dataset_ref: Reference to test dataset
            evaluation_type: Type of evaluation
            
        Returns:
            TestEvaluationAnchor instance
        """
        # Simulate test metrics
        import random
        random.seed(42)  # For reproducible results
        
        mock_metrics = {
            "accuracy": round(0.85 + random.random() * 0.10, 3),
            "precision": round(0.82 + random.random() * 0.12, 3),
            "recall": round(0.78 + random.random() * 0.15, 3),
            "f1_score": round(0.80 + random.random() * 0.12, 3)
        }
        
        return self.create_test_evaluation_anchor(
            test_id=test_id,
            test_dataset_ref=test_dataset_ref,
            evaluation_type=evaluation_type,
            metrics=mock_metrics
        )
    
    def format_roots_summary(
        self,
        training_session_root: str,
        release_root: str,
        inference_batch_root: str,
        timestamp_authority: str = "not_set",
        evidence_id: str = "null"
    ) -> str:
        """Format roots summary for pretty printing."""
        lines = [
            f"  training_session_root: {training_session_root[:8]}...{training_session_root[-4:]}",
            f"  release_root:          R = MerkleRoot([training_session_root, predeployment_anchor,",
            f"                                        deployment_anchor, test_evaluation_anchor]) = R_{release_root[:8]}...{release_root[-4:]}",
            f"  inference_batch_root:  {inference_batch_root[:8]}...{inference_batch_root[-8:]}",
            f"  ⏱️ timestamp: authority={timestamp_authority}, evidence_id={evidence_id}"
        ]
        return "\n".join(lines)
    
    def format_test_evaluation_summary(self, test_id: str) -> str:
        """Format test evaluation summary for pretty printing."""
        anchor = self.get_test_evaluation_anchor(test_id)
        if not anchor:
            return f"Test evaluation {test_id} not found"
        
        # Format metrics for display
        metrics_str = f"acc={anchor.metrics.get('accuracy', 0):.3f}"
        if 'precision' in anchor.metrics:
            metrics_str += f",prec={anchor.metrics['precision']:.3f}"
        if 'recall' in anchor.metrics:
            metrics_str += f",rec={anchor.metrics['recall']:.3f}"
        
        lines = [
            f"  test_metrics_digest: {metrics_str} ⇒ {anchor.test_metrics_digest[:8]}...{anchor.test_metrics_digest[-4:]}",
            f"  ✅ test_evaluation_anchor: {anchor.anchor_id}"
        ]
        return "\n".join(lines)
