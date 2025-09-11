"""
CIAF LCM Basic Example - Getting Started

This example demonstrates the basic usage of the CIAF LCM (Lifecycle Management) system,
showing how to create a simple dataset family with splits and a model with basic training.

Usage: python basic_example.py

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import sys
import os

# Add the parent directory to Python path to import ciaf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ciaf.lcm.dataset_family_manager import LCMDatasetFamilyManager, DatasetFamilyMetadata
from ciaf.lcm.model_manager import LCMModelManager, ModelArchitecture, TrainingEnvironment
from ciaf.lcm.training_manager import LCMTrainingManager, DatasetSplit
from ciaf.lcm.policy import get_default_policy
from ciaf.core import MerkleTree


def main():
    """Basic CIAF LCM usage example."""
    print("🚀 CIAF LCM Basic Example")
    print("=" * 50)
    
    # Initialize policy
    policy = get_default_policy()
    print(f"📋 Policy: {policy.hash_algorithm} | {policy.canonicalization}")
    print()
    
    # Step 1: Create Dataset Family
    print("🗃️ Step 1: Creating Dataset Family")
    print("-" * 30)
    
    dataset_manager = LCMDatasetFamilyManager(policy)
    
    # Create family metadata
    family_metadata = DatasetFamilyMetadata(
        name="customer_reviews",
        version="v1",
        owner="Data Team",
        license="CC-BY-4.0",
        description="Customer product reviews for sentiment analysis",
        creation_date="2025-09-10"
    )

    # Define split configs
    split_configs = {
        DatasetSplit.TRAIN: {"ratio": 0.8, "samples": 8000, "purpose": "model training"},
        DatasetSplit.VALIDATION: {"ratio": 0.1, "samples": 1000, "purpose": "hyperparameter tuning"},
        DatasetSplit.TEST: {"ratio": 0.1, "samples": 1000, "purpose": "final evaluation"}
    }

    # Create dataset family with splits
    family_anchor = dataset_manager.create_dataset_family("customer_reviews", family_metadata, split_configs)
    print(f"✅ Dataset family created: {family_anchor.anchor_id}")

    # Get split anchors
    train_split = dataset_manager.get_split_anchor("customer_reviews", DatasetSplit.TRAIN)
    val_split = dataset_manager.get_split_anchor("customer_reviews", DatasetSplit.VALIDATION)
    test_split = dataset_manager.get_split_anchor("customer_reviews", DatasetSplit.TEST)

    print(f"✅ Created 3 splits: train, validation, test")
    print()
    
    # Step 2: Create Model
    print("🤖 Step 2: Creating Model")
    print("-" * 25)
    
    model_manager = LCMModelManager(policy)
    
    # Define model architecture
    architecture = ModelArchitecture(
        type="transformer",
        layers=[{"name": "embedding"}, {"name": "transformer_block"}, {"name": "classifier"}],
        input_dim=512,  # max sequence length
        output_dim=3,   # 3 sentiment classes
        total_params=110000000
    )
    
    # Define training environment
    environment = TrainingEnvironment(
        framework="pytorch",
        framework_version="2.0.1",
        cuda_version="11.8",
        python_version="3.11.5",
        os_info="Ubuntu 22.04"
    )
    
    # Model hyperparameters
    hyperparameters = {
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 3,
        "warmup_steps": 500,
        "weight_decay": 0.01
    }
    
    # Create model anchor
    model_anchor = model_manager.create_model_anchor(
        model_name="sentiment_classifier",
        version="v1.0.0",
        architecture=architecture,
        hyperparameters=hyperparameters,
        environment=environment
    )
    
    print(f"✅ Model created: {model_anchor.model_name} v{model_anchor.version}")
    print()
    
    # Step 3: Training Session
    print("🏋️ Step 3: Training Session") 
    print("-" * 28)
    
    training_manager = LCMTrainingManager(policy)
    
    # Start training session
    training_session = training_manager.create_training_session(
        session_id="sentiment_training_001",
        model_anchor=model_anchor,
        datasets_root_anchor=family_anchor.anchor_id,
        training_config={
            "optimizer": "AdamW",
            "loss": "CrossEntropyLoss",
            "metrics": ["accuracy", "f1_score"]
        },
        data_splits={
            DatasetSplit.TRAIN: train_split.anchor_id,
            DatasetSplit.VALIDATION: val_split.anchor_id,
            DatasetSplit.TEST: test_split.anchor_id
        }
    )
    
    print(f"✅ Training session started: {training_session.session_id}")
    print()
    
    # Step 4: Verify Integrity
    print("🔐 Step 4: Integrity Verification")
    print("-" * 34)
    
    # Create integrity root from all components
    from ciaf.core.crypto import sha256_hash
    integrity_leaves = [
        sha256_hash(family_anchor.family_anchor.encode()),  # Dataset family anchor
        sha256_hash(model_anchor.model_hash.encode()),      # Model anchor
        sha256_hash(training_session.session_id.encode())   # Training session ID
    ]
    integrity_tree = MerkleTree(integrity_leaves)
    integrity_root = integrity_tree.get_root()
    
    print(f"🌳 Integrity root: {integrity_root[:16]}...")
    leaf_hash = sha256_hash(family_anchor.family_anchor.encode())
    print(f"✅ Merkle verification: {integrity_tree.verify_proof_cached(leaf_hash)}")
    print()
    
    print("🎉 Basic example completed successfully!")
    print("\n💡 Next steps:")
    print("   - Run intermediate_example.py for inference and deployment")
    print("   - Run advanced_example.py for complete audit trails")


if __name__ == "__main__":
    main()
