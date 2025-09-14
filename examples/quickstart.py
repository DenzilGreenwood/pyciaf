#!/usr/bin/env python3
"""
CIAF Quickstart Example

A minimal, runnable example demonstrating the core CIAF functionality
including dataset anchors, provenance capsules, model anchors, and training integrity.

This example runs end-to-end and demonstrates:
- Dataset anchor creation
- Provenance capsule generation
- Model anchor with parameter fingerprinting
- Training snapshot with integrity validation

Usage:
    python examples/quickstart.py

Created: 2025-09-12
Author: Denzil James Greenwood
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path to import ciaf
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ciaf import CIAFFramework
except ImportError as e:
    print(f"❌ Error importing CIAF: {e}")
    print("💡 Make sure you have installed the package: pip install -e .")
    sys.exit(1)


def main():
    """Run the CIAF quickstart example."""
    print("🚀 CIAF Quickstart Example")
    print("=" * 50)
    
    try:
        # Initialize the CIAF framework
        framework = CIAFFramework("demo_project")
        print("✅ CIAF Framework initialized")
        
        # Step 1: Create a dataset anchor
        print("\n📊 Step 1: Creating Dataset Anchor")
        print("-" * 35)
        
        anchor = framework.create_dataset_anchor(
            dataset_id="demo_ds",
            dataset_metadata={"source": "synthetic", "type": "text"},
            master_password="change_me"
        )
        print(f"✅ Dataset anchor created: {anchor.dataset_id}")
        print(f"   Dataset hash: {anchor.dataset_hash[:16]}...")
        
        # Step 2: Create provenance capsules
        print("\n📦 Step 2: Creating Provenance Capsules")
        print("-" * 41)
        
        data_items = [
            {"content": "item1", "metadata": {"id": "1", "label": "positive"}},
            {"content": "item2", "metadata": {"id": "2", "label": "negative"}},
        ]
        
        capsules = framework.create_provenance_capsules("demo_ds", data_items)
        print(f"✅ Created {len(capsules)} provenance capsules")
        
        # Step 3: Create model anchor
        print("\n🤖 Step 3: Creating Model Anchor")
        print("-" * 32)
        
        model_anchor = framework.create_model_anchor(
            model_name="demo_model",
            model_parameters={"epochs": 3, "lr": 0.01},
            model_architecture={"type": "logreg"},
            authorized_datasets=["demo_ds"],
            master_password="change_me"
        )
        print(f"✅ Model anchor created: {model_anchor['model_name']}")
        print(f"   Parameters fingerprint: {model_anchor.get('parameters_fingerprint', 'N/A')[:16]}...")
        
        # Step 4: Train model with snapshot
        print("\n🏋️ Step 4: Training Model")
        print("-" * 26)
        
        snapshot = framework.train_model(
            model_name="demo_model",
            capsules=capsules,
            maa=model_anchor,
            training_params={"epochs": 3, "lr": 0.01},
            model_version="v0"
        )
        print(f"✅ Training snapshot created: {snapshot.snapshot_id[:16]}...")
        
        # Step 5: Validate training integrity
        print("\n🔐 Step 5: Validating Training Integrity")
        print("-" * 42)
        
        is_valid = framework.validate_training_integrity(snapshot)
        print(f"✅ Training integrity validated: {is_valid}")
        
        if is_valid:
            print("\n🎉 SUCCESS: CIAF quickstart completed successfully!")
            print("\n💡 Next steps:")
            print("   - Explore examples/basic_example.py for more features")
            print("   - Check examples/advanced_example.py for full audit trails")
            print("   - Run compliance reports with: python -m ciaf.cli compliance")
        else:
            print("\n❌ ERROR: Training integrity validation failed")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("💡 This might be because some CIAF modules are not fully implemented yet.")
        print("   Check the basic_example.py for a working demonstration.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)