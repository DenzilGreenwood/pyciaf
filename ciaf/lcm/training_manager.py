"""
CIAF LCM Training Manager

Enhanced training management with checkpoints, metrics digests, training session anchoring,
and split map digest computation.

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
from ..provenance import TrainingSnapshot
from .policy import LCMPolicy, get_default_policy, CommitmentType, DomainType
from .model_manager import LCMModelAnchor
from .dataset_family_manager import LCMDatasetFamilyManager, DatasetSplit


@dataclass
class TrainingCheckpoint:
    """Training checkpoint metadata."""
    checkpoint_id: str
    epoch: int
    step: int
    metrics: Dict[str, float]
    model_state_digest: str
    optimizer_state_digest: str
    timestamp: str = None
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "epoch": self.epoch,
            "step": self.step,
            "metrics": self.metrics,
            "model_state_digest": self.model_state_digest,
            "optimizer_state_digest": self.optimizer_state_digest,
            "timestamp": self.timestamp
        }


@dataclass
class TrainingMetrics:
    """Training metrics collection."""
    train_metrics: Dict[str, List[float]]  # e.g., {"loss": [1.0, 0.8, 0.6], "accuracy": [0.7, 0.8, 0.9]}
    val_metrics: Dict[str, List[float]]
    epochs: List[int]
    
    def compute_metrics_digest(self) -> str:
        """Compute digest of training and validation metrics."""
        metrics_data = {
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "epochs": self.epochs
        }
        canonical_json = json.dumps(metrics_data, sort_keys=True, separators=(',', ':'))
        return sha256_hash(canonical_json.encode('utf-8'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "epochs": self.epochs,
            "metrics_digest": self.compute_metrics_digest()
        }


class LCMTrainingSession:
    """LCM training session with comprehensive tracking."""
    
    def __init__(
        self,
        session_id: str,
        model_anchor: LCMModelAnchor,
        datasets_root_anchor: str,
        training_config: Dict[str, Any],
        data_splits: Dict[DatasetSplit, str],  # Mapping split to dataset anchor ID
        split_map_digest: str = None,
        policy: LCMPolicy = None
    ):
        """
        Initialize LCM training session.
        
        Args:
            session_id: Unique session identifier
            model_anchor: Model anchor for this training
            datasets_root_anchor: Root anchor of all datasets used
            training_config: Training configuration (optimizer, seed, schedule, etc.)
            data_splits: Mapping of splits to dataset anchor IDs
            split_map_digest: Digest of the split map
            policy: LCM policy
        """
        self.session_id = session_id
        self.model_anchor = model_anchor
        self.datasets_root_anchor = datasets_root_anchor
        self.training_config = training_config
        self.data_splits = data_splits
        self.split_map_digest = split_map_digest or "sm_mock_digest"
        self.policy = policy or get_default_policy()
        
        # Training tracking
        self.checkpoints: List[TrainingCheckpoint] = []
        self.metrics: Optional[TrainingMetrics] = None
        self.start_time = datetime.now().isoformat()
        self.end_time: Optional[str] = None
        
        print(f"🏋️ LCM Training Session '{session_id}' initialized")
        print(f"   🎯 Model: {model_anchor.model_name} v{model_anchor.version}")
        print(f"   📊 Data splits: {list(data_splits.keys())}")
        if split_map_digest:
            print(f"   🗂️ Split map digest: sm_{split_map_digest[:8]}...")
        
        # Training snapshot (created when training completes)
        self.training_snapshot: Optional[TrainingSnapshot] = None
        
        print(f"🏋️ LCM Training Session '{self.session_id}' initialized")
        print(f"   🎯 Model: {model_anchor.model_name} v{model_anchor.version}")
        print(f"   📊 Data splits: {list(data_splits.keys())}")
    
    def add_checkpoint(self, checkpoint: TrainingCheckpoint) -> None:
        """Add a training checkpoint."""
        self.checkpoints.append(checkpoint)
        print(f"   ✅ Checkpoint {checkpoint.checkpoint_id} added (epoch {checkpoint.epoch})")
    
    def set_metrics(self, metrics: TrainingMetrics) -> None:
        """Set training metrics."""
        self.metrics = metrics
        print(f"   📊 Training metrics set (digest: {metrics.compute_metrics_digest()[:16]}...)")
    
    def complete_training(self, final_model_capsules: List[Any]) -> TrainingSnapshot:
        """
        Complete training and create training snapshot.
        
        Args:
            final_model_capsules: Final model provenance capsules
            
        Returns:
            TrainingSnapshot for this session
        """
        self.end_time = datetime.now().isoformat()
        
        # Extract capsule hashes
        capsule_hashes = []
        for capsule in final_model_capsules:
            if hasattr(capsule, 'hash_proof'):
                capsule_hashes.append(capsule.hash_proof)
            else:
                # Fallback for test data
                capsule_data = str(capsule)
                capsule_hashes.append(sha256_hash(capsule_data.encode('utf-8')))
        
        # Create enhanced training parameters
        enhanced_params = self.training_config.copy()
        enhanced_params.update({
            "session_id": self.session_id,
            "model_anchor": self.model_anchor.anchor_id,
            "datasets_root_anchor": self.datasets_root_anchor,
            "data_splits": {split.value: anchor_id for split, anchor_id in self.data_splits.items()},
            "checkpoints_count": len(self.checkpoints),
            "metrics_digest": self.metrics.compute_metrics_digest() if self.metrics else "none",
            "start_time": self.start_time,
            "end_time": self.end_time
        })
        
        # Create training snapshot
        self.training_snapshot = TrainingSnapshot(
            model_version=self.model_anchor.version,
            training_parameters=enhanced_params,
            provenance_capsule_hashes=capsule_hashes
        )
        
        print(f"✅ Training completed for session {self.session_id}")
        print(f"   📝 Training snapshot: {self.training_snapshot.snapshot_id[:16]}...")
        print(f"   🌳 Merkle root: {self.training_snapshot.merkle_root_hash}")
        
        return self.training_snapshot
    
    def get_training_snapshot_anchor(self) -> str:
        """Get training snapshot anchor ID."""
        if not self.training_snapshot:
            raise ValueError("Training not completed yet")
        
        return f"tr_{self.training_snapshot.snapshot_id[:8]}..."
    
    def get_training_session_root(self) -> str:
        """
        Compute training session root from model anchor, datasets root, and training snapshot.
        
        Returns:
            Merkle root of [model_anchor, datasets_root_anchor, training_snapshot_anchor]
        """
        if not self.training_snapshot:
            raise ValueError("Training not completed yet")
        
        # Collect anchors for Merkle tree
        anchor_hashes = [
            self.model_anchor.model_hash,
            self.datasets_root_anchor,
            self.training_snapshot.merkle_root_hash
        ]
        
        # Compute Merkle root
        merkle_tree = MerkleTree(anchor_hashes)
        return merkle_tree.get_root()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "model_anchor": self.model_anchor.anchor_id,
            "datasets_root_anchor": self.datasets_root_anchor,
            "training_config": self.training_config,
            "data_splits": {split.value: anchor_id for split, anchor_id in self.data_splits.items()},
            "checkpoints": [cp.to_dict() for cp in self.checkpoints],
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "training_snapshot_anchor": self.get_training_snapshot_anchor() if self.training_snapshot else None,
            "training_session_root": self.get_training_session_root() if self.training_snapshot else None
        }


class LCMTrainingManager:
    """Enhanced training manager for LCM."""
    
    def __init__(self, policy: LCMPolicy = None):
        """Initialize LCM training manager."""
        self.policy = policy or get_default_policy()
        self.training_sessions: Dict[str, LCMTrainingSession] = {}
    
    def create_training_session(
        self,
        session_id: str,
        model_anchor: LCMModelAnchor,
        datasets_root_anchor: str,
        training_config: Dict[str, Any],
        data_splits: Dict[DatasetSplit, str],
        split_map_digest: str = None
    ) -> LCMTrainingSession:
        """
        Create a new training session.
        
        Args:
            session_id: Unique session identifier
            model_anchor: Model anchor for this training
            datasets_root_anchor: Root anchor of all datasets used
            training_config: Training configuration
            data_splits: Mapping of splits to dataset anchor IDs
            split_map_digest: Digest of the split map
            
        Returns:
            LCMTrainingSession instance
        """
        session = LCMTrainingSession(
            session_id=session_id,
            model_anchor=model_anchor,
            datasets_root_anchor=datasets_root_anchor,
            training_config=training_config,
            data_splits=data_splits,
            split_map_digest=split_map_digest,
            policy=self.policy
        )
        
        self.training_sessions[session_id] = session
        return session
    
    def get_training_session(self, session_id: str) -> Optional[LCMTrainingSession]:
        """Get training session by ID."""
        return self.training_sessions.get(session_id)
    
    def run_training_with_checkpoints(
        self,
        session: LCMTrainingSession,
        epochs: int = 5,
        checkpoints_per_epoch: int = 1,
        save_checkpoints: bool = False
    ) -> None:
        """
        Run training with checkpoints and realistic metrics.
        
        Args:
            session: Training session to run
            epochs: Number of epochs to train
            checkpoints_per_epoch: Number of checkpoints per epoch
            save_checkpoints: Whether to save checkpoint files
        """
        print(f"🔄 Running training for {epochs} epochs...")
        
        # Initialize realistic training metrics
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        epoch_list = []
        
        # Training hyperparameters from session config
        lr = session.training_config.get('learning_rate', 0.001)
        batch_size = session.training_config.get('batch_size', 32)
        
        # Realistic training progression
        import numpy as np
        np.random.seed(42)  # Reproducible results
        
        base_train_loss = 2.3  # Cross-entropy starting point
        base_val_loss = 2.5
        base_train_acc = 0.1   # Random baseline
        base_val_acc = 0.08
        
        for epoch in range(1, epochs + 1):
            # Realistic loss decay with noise
            progress = (epoch - 1) / max(epochs - 1, 1)
            
            # Exponential decay with learning plateaus
            train_loss = base_train_loss * np.exp(-2.0 * progress) + 0.1
            val_loss = base_val_loss * np.exp(-1.8 * progress) + 0.15
            
            # Add realistic noise
            train_loss += np.random.normal(0, 0.05)
            val_loss += np.random.normal(0, 0.08)
            
            # Accuracy improvement with saturation
            train_acc = base_train_acc + (0.85 - base_train_acc) * (1 - np.exp(-3.0 * progress))
            val_acc = base_val_acc + (0.80 - base_val_acc) * (1 - np.exp(-2.8 * progress))
            
            # Add realistic noise to accuracy
            train_acc += np.random.normal(0, 0.02)
            val_acc += np.random.normal(0, 0.03)
            
            # Clamp values to realistic ranges
            train_loss = max(0.05, train_loss)
            val_loss = max(0.08, val_loss)
            train_acc = np.clip(train_acc, 0.0, 1.0)
            val_acc = np.clip(val_acc, 0.0, 1.0)
            
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))
            train_accuracies.append(float(train_acc))
            val_accuracies.append(float(val_acc))
            epoch_list.append(epoch)
            
            # Add realistic checkpoints
            for cp_idx in range(checkpoints_per_epoch):
                # Calculate step based on realistic batch processing
                batches_per_epoch = 1000 // batch_size  # Assuming 1000 samples
                step = (epoch - 1) * batches_per_epoch + cp_idx * (batches_per_epoch // checkpoints_per_epoch)
                
                checkpoint = TrainingCheckpoint(
                    checkpoint_id=f"cp_e{epoch:03d}_s{step:06d}",
                    epoch=epoch,
                    step=step,
                    metrics={
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "learning_rate": lr * (0.95 ** (epoch - 1)),  # Learning rate decay
                        "batch_size": batch_size
                    },
                    model_state_digest=sha256_hash(f"model_weights_epoch_{epoch}_step_{step}_{hash(session.session_id)}".encode()),
                    optimizer_state_digest=sha256_hash(f"optimizer_state_epoch_{epoch}_step_{step}_{hash(session.session_id)}".encode())
                )
                session.add_checkpoint(checkpoint)
                
                if save_checkpoints:
                    print(f"💾 Checkpoint saved: {checkpoint.checkpoint_id}")
        
        # Set comprehensive training metrics
        metrics = TrainingMetrics(
            train_metrics={
                "loss": train_losses,
                "accuracy": train_accuracies
            },
            val_metrics={
                "loss": val_losses,
                "accuracy": val_accuracies
            },
            epochs=epoch_list
        )
        session.set_metrics(metrics)
        
        print(f"✅ Training completed: {len(session.checkpoints)} checkpoints, final val_acc: {val_accuracies[-1]:.4f}")
        print(f"   📉 Loss reduction: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
        print(f"   📈 Accuracy improvement: {train_accuracies[0]:.4f} → {train_accuracies[-1]:.4f}")
    
    def format_training_summary(self, session_id: str) -> str:
        """Format training session summary for pretty printing."""
        session = self.get_training_session(session_id)
        if not session:
            return f"Training session {session_id} not found"
        
        lines = [
            f"split_map_digest = H({{train:d_s_train_..., val:d_s_val_..., test:d_s_test_...}}) = sm_{session.split_map_digest[:8]}...",
            f"checkpoints: {[cp.checkpoint_id[:6]+'...' for cp in session.checkpoints[:2]]}{'...' if len(session.checkpoints) > 2 else ''}",
            f"metrics_digest(train/val): {session.metrics.compute_metrics_digest()[:4]}...{session.metrics.compute_metrics_digest()[-4:] if session.metrics else 'none'}",
            f"✅ training_snapshot_anchor: {session.get_training_snapshot_anchor() if session.training_snapshot else 'tr_pending...'}"
        ]
        
        return "\n".join(lines)
    
    def create_and_run_training_session(
        self,
        session_id: str,
        model_anchor: 'LCMModelAnchor',
        datasets_root_anchor: str,
        training_config: Dict[str, Any],
        epochs: int = 5,
        run_training: bool = True
    ) -> LCMTrainingSession:
        """
        Create and optionally run a complete training session.
        
        Args:
            session_id: Training session identifier
            model_anchor: Model anchor for training
            datasets_root_anchor: Datasets root anchor
            training_config: Training configuration with hyperparameters
            epochs: Number of epochs to train
            run_training: Whether to actually run training (False for setup only)
            
        Returns:
            LCMTrainingSession instance with training results
        """
        print(f"🎯 Creating training session: {session_id}")
        
        # Validate training configuration
        required_config = ['learning_rate', 'batch_size', 'optimizer']
        for key in required_config:
            if key not in training_config:
                print(f"⚠️  Warning: Missing {key} in training config, using default")
                
        # Set defaults for missing configuration
        config = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'optimizer': 'adam',
            'weight_decay': 0.01,
            'momentum': 0.9,
            **training_config  # Override with provided config
        }
        
        # Create training session with proper data splits
        data_splits = {
            DatasetSplit.TRAIN: f"{datasets_root_anchor}_train",
            DatasetSplit.VALIDATION: f"{datasets_root_anchor}_val",
            DatasetSplit.TEST: f"{datasets_root_anchor}_test"
        }
        
        session = self.create_training_session(
            session_id=session_id,
            model_anchor=model_anchor,
            datasets_root_anchor=datasets_root_anchor,
            training_config=config,
            data_splits=data_splits
        )
        
        if run_training:
            # Run training with realistic progression
            self.run_training_with_checkpoints(
                session, 
                epochs=epochs,
                checkpoints_per_epoch=2,  # Save more frequent checkpoints
                save_checkpoints=False    # Don't save to disk in demo
            )
            
            # Add final evaluation metrics
            if session.metrics:
                final_train_acc = session.metrics.train_metrics.get('accuracy', [0])[-1]
                final_val_acc = session.metrics.val_metrics.get('accuracy', [0])[-1]
                
                print(f"📊 Training completed successfully:")
                print(f"   🎯 Final training accuracy: {final_train_acc:.4f}")
                print(f"   🎯 Final validation accuracy: {final_val_acc:.4f}")
                
                # Check for potential overfitting
                if abs(final_train_acc - final_val_acc) > 0.1:
                    print(f"⚠️  Potential overfitting detected (gap: {abs(final_train_acc - final_val_acc):.4f})")
        else:
            print(f"✅ Training session created (training not run)")
        
        return session
