#!/usr/bin/env python3
"""
Test the LCM system after removing redundant anchoring modules.
"""

import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_lcm_system():
    """Test LCM dataset anchor functionality."""
    from ciaf.lcm.dataset_manager import LCMDatasetAnchor, DatasetMetadata, DatasetSplit
    from ciaf.core.crypto import sha256_hash
    
    # Create test metadata with correct parameters
    metadata = DatasetMetadata(
        name="test_dataset",
        owner="test_user",
        license="MIT",
        schema_digest="test_schema_123",
        sampling_rules={"method": "random", "seed": 42},
        version="1.0.0",
        content_root="test_root_hash",
        description="Test dataset for LCM",
        feature_names=["feature1", "feature2", "feature3"],
        num_samples=100,
        num_features=3,
        feature_types={
            "feature1": "numerical",
            "feature2": "categorical", 
            "feature3": "numerical"
        }
    )
    
    # Create LCM dataset anchor
    anchor = LCMDatasetAnchor(
        dataset_id="test_001",
        split=DatasetSplit.TRAIN,
        metadata=metadata,
        master_password="test_password_123"
    )
    
    # Verify anchor creation
    assert anchor.dataset_id == "test_001"
    assert anchor.split == DatasetSplit.TRAIN
    assert anchor.metadata.name == "test_dataset"
    
    # Add some sample hashes
    for i in range(5):
        sample_hash = sha256_hash(f"sample_{i}".encode())
        anchor.add_sample_hash(sample_hash)
    
    # Test JSON serialization
    json_data = anchor.to_json()
    assert isinstance(json_data, dict)
    assert "dataset_id" in json_data
    assert "split" in json_data


def test_no_legacy_imports():
    """Test that legacy modules are properly removed."""
    import pytest
    
    # These should fail now
    with pytest.raises(ImportError):
        from ciaf.core.base_anchor import DatasetAnchor
    
    with pytest.raises(ImportError):
        from ciaf.anchoring import DatasetAnchor
    
    with pytest.raises(ImportError):
        from ciaf.core.keys import AnchorManager


def main():
    """Run all tests."""
    print("=" * 60)
    print("LCM System Consolidation Test")
    print("=" * 60)
    
    tests = [
        ("LCM System Functionality", test_lcm_system),
        ("Legacy Module Removal", test_no_legacy_imports)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name}: PASS")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAIL - {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\nSUCCESS: LCM consolidation complete!")
        print("- Redundant anchoring modules removed")
        print("- LCM system is the single source of truth")
        print("- No backward compatibility maintained (as requested)")
    else:
        print(f"\nFAILED: {len(tests) - passed} test(s) failed")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)