"""
Simple test to verify CIAF framework functionality.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

from ciaf import CIAFFramework

def test_basic_functionality():
    """Test basic CIAF functionality."""
    print("Testing CIAF Framework...")
    
    # Initialize framework
    framework = CIAFFramework("Test_Framework")
    print("✓ Framework initialized")
    
    # Create a simple dataset anchor
    print("Testing dataset anchor creation...")
    dataset_anchor = framework.create_dataset_anchor(
        dataset_id="test_dataset",
        dataset_metadata={"source": "test", "type": "demo"},
        master_password="test_password"
    )
    print("✓ Dataset anchor created")
    
    # Create some test data
    test_data = [
        {"content": "sample_data_1", "metadata": {"id": "1"}},
        {"content": "sample_data_2", "metadata": {"id": "2"}}
    ]
    
    # Create provenance capsules
    print("Testing provenance capsule creation...")
    capsules = framework.create_provenance_capsules("test_dataset", test_data)
    print(f"✓ Created {len(capsules)} provenance capsules")
    
    # Test model anchor creation
    print("Testing model anchor creation...")
    model_anchor = framework.create_model_anchor(
        model_name="test_model",
        model_parameters={"learning_rate": 0.001, "epochs": 10},
        model_architecture={"type": "MLP", "layers": [100, 50, 10]},
        authorized_datasets=["test_dataset"],
        master_password="model_password"
    )
    print("✓ Model anchor created")
    
    # Test performance metrics
    print("Testing performance metrics...")
    metrics = framework.get_performance_metrics("test_dataset")
    print(f"✓ Performance metrics retrieved: {len(metrics)} categories")
    
    print("✓ All tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
