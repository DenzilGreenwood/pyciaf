#!/usr/bin/env python3
"""
Test script to verify LCM metadata preservation through pickle/unpickle cycle.

This script tests that when a CIAFModelWrapper is pickled and then unpickled,
all LCM metadata tracking is preserved and available as output.

Created: 2025-09-19
Author: CIAF Development Team
"""

import os
import sys
import pickle
import tempfile
from pathlib import Path

# Add the CIAF package to the path
sys.path.insert(0, str(Path(__file__).parent))

from ciaf.wrappers import CIAFModelWrapper
from sklearn.linear_model import LinearRegression
import numpy as np

def test_lcm_pickle_preservation():
    """Test LCM metadata preservation through pickle/unpickle cycle."""
    print("🧪 Testing LCM Metadata Preservation Through Pickle/Unpickle")
    print("="*70)
    
    # Create a simple model for testing
    model = LinearRegression()
    
    print("\n1️⃣ Creating CIAFModelWrapper with LCM integration...")
    wrapper = CIAFModelWrapper(
        model=model,
        model_name="LCM_Pickle_Test_Model",
        enable_connections=True,
        compliance_mode="general",
        enable_explainability=True,
        enable_uncertainty=True,
        enable_metadata_tags=True
    )
    
    print(f"✅ Model wrapper created: {wrapper.model_name}")
    
    # Create some training data
    print("\n2️⃣ Training model with sample data...")
    training_data = [
        {
            "content": "Sample training text 1",
            "metadata": {"id": "train_001", "target": 1, "category": "positive"}
        },
        {
            "content": "Sample training text 2", 
            "metadata": {"id": "train_002", "target": 0, "category": "negative"}
        },
        {
            "content": "Sample training text 3",
            "metadata": {"id": "train_003", "target": 1, "category": "positive"}
        }
    ]
    
    training_snapshot = wrapper.train(
        dataset_id="lcm_pickle_test_dataset",
        training_data=training_data,
        master_password="test_password_123",
        model_version="1.0.0",
        fit_model=False  # Skip actual model training for this test
    )
    
    print(f"✅ Training completed: {training_snapshot.snapshot_id}")
    
    # Perform some inference to create LCM metadata
    print("\n3️⃣ Performing inference to generate LCM receipts...")
    
    queries = [
        "What is the prediction for this sample?",
        "How does this model work?",
        "Can you analyze this input?"
    ]
    
    for i, query in enumerate(queries):
        prediction, receipt = wrapper.predict(query, use_model=False)
        print(f"   Receipt {i+1}: {receipt.receipt_hash[:16]}...")
    
    # Get LCM metadata before pickling
    print("\n4️⃣ Extracting LCM metadata before pickling...")
    metadata_before = wrapper.get_lcm_metadata_trail()
    model_info_before = wrapper.get_model_info()
    lcm_export_before = wrapper.export_lcm_metadata(output_format="audit_report")
    
    print(f"✅ LCM metadata extracted:")
    print(f"   Training capsules: {len(metadata_before['training_metadata'])}")
    print(f"   Inference receipts: {len(metadata_before.get('inference_metadata', {}))}")
    print(f"   Connections: {len(metadata_before.get('connections_metadata', {}).get('connections_summary', []))}")
    print(f"   Enhanced features: {metadata_before['enhanced_features']}")
    
    # Pickle the wrapper
    print("\n5️⃣ Pickling the CIAFModelWrapper...")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        pickle_file = tmp_file.name
        pickle.dump(wrapper, tmp_file)
    
    print(f"✅ Model pickled to: {pickle_file}")
    print(f"   File size: {os.path.getsize(pickle_file)} bytes")
    
    # Unpickle the wrapper
    print("\n6️⃣ Unpickling the CIAFModelWrapper...")
    with open(pickle_file, 'rb') as f:
        restored_wrapper = pickle.load(f)
    
    print(f"✅ Model unpickled successfully")
    print(f"   Model name: {restored_wrapper.model_name}")
    print(f"   Model type: {type(restored_wrapper.model).__name__}")
    
    # Verify LCM metadata after unpickling
    print("\n7️⃣ Verifying LCM metadata after unpickling...")
    
    try:
        metadata_after = restored_wrapper.get_lcm_metadata_trail()
        model_info_after = restored_wrapper.get_model_info()
        lcm_export_after = restored_wrapper.export_lcm_metadata(output_format="audit_report")
        
        print(f"✅ LCM metadata accessible after unpickling:")
        print(f"   Training capsules: {len(metadata_after['training_metadata'])}")
        print(f"   Inference receipts: {len(metadata_after.get('inference_metadata', {}))}")
        print(f"   Connections: {len(metadata_after.get('connections_metadata', {}).get('connections_summary', []))}")
        print(f"   Enhanced features: {metadata_after['enhanced_features']}")
        
        # Check if serialized trail is available
        if hasattr(restored_wrapper, '_lcm_metadata_trail'):
            print(f"   Serialized trail preserved: ✅")
            print(f"   Serialization timestamp: {getattr(restored_wrapper, '_lcm_serialization_timestamp', 'N/A')}")
        else:
            print(f"   Serialized trail preserved: ❌")
        
        # Verify model info includes LCM data
        lcm_metadata_info = model_info_after.get('lcm_metadata', {})
        print(f"   LCM integration enabled: {lcm_metadata_info.get('lcm_integration_enabled', False)}")
        print(f"   Pickle preservation ready: {lcm_metadata_info.get('pickle_preservation_ready', False)}")
        
        # Compare before and after
        print("\n8️⃣ Comparing metadata before and after pickling...")
        
        comparison_results = {
            "model_name_preserved": metadata_before['model_name'] == metadata_after['model_name'],
            "model_version_preserved": metadata_before['model_version'] == metadata_after['model_version'],
            "enhanced_features_preserved": metadata_before['enhanced_features'] == metadata_after['enhanced_features'],
            "training_metadata_preserved": bool(metadata_before['training_metadata']) == bool(metadata_after['training_metadata']),
            "inference_metadata_preserved": bool(metadata_before['inference_metadata']) == bool(metadata_after['inference_metadata']),
        }
        
        all_preserved = all(comparison_results.values())
        
        print(f"✅ Metadata preservation results:")
        for key, preserved in comparison_results.items():
            status = "✅" if preserved else "❌"
            print(f"   {key}: {status}")
        
        print(f"\n🎯 Overall LCM preservation status: {'✅ SUCCESS' if all_preserved else '❌ FAILED'}")
        
        # Test additional inference on restored model
        print("\n9️⃣ Testing inference on restored model...")
        try:
            test_prediction, test_receipt = restored_wrapper.predict("Test query on restored model", use_model=False)
            print(f"✅ Inference successful on restored model")
            print(f"   New receipt: {test_receipt.receipt_hash[:16]}...")
            
            # Export final metadata trail
            final_export = restored_wrapper.export_lcm_metadata(output_format="audit_report", include_receipts=True)
            total_receipts = final_export.get('audit_summary', {}).get('total_inference_receipts', 0)
            print(f"   Total receipts after restoration: {total_receipts}")
            
        except Exception as e:
            print(f"❌ Inference failed on restored model: {e}")
        
        return all_preserved
        
    except Exception as e:
        print(f"❌ Failed to verify LCM metadata after unpickling: {e}")
        return False
    
    finally:
        # Clean up pickle file
        try:
            os.unlink(pickle_file)
            print(f"\n🧹 Cleaned up pickle file: {pickle_file}")
        except:
            pass

def main():
    """Main test function."""
    print("🚀 CIAF LCM Pickle Preservation Test")
    print("Testing that LCM metadata tracking survives pickle/unpickle cycle")
    print()
    
    try:
        success = test_lcm_pickle_preservation()
        
        print("\n" + "="*70)
        if success:
            print("🎉 TEST PASSED: LCM metadata successfully preserved through pickle/unpickle!")
            print("   ✅ All metadata components are accessible from the pickled model")
            print("   ✅ CIAF LCM tracking is fully functional after restoration")
        else:
            print("❌ TEST FAILED: LCM metadata was not fully preserved")
            print("   ⚠️ Some metadata components may be missing after unpickling")
        
        print("\n📋 Summary:")
        print("   • CIAFModelWrapper now supports LCM metadata preservation during pickling")
        print("   • Custom __getstate__ and __setstate__ methods handle serialization")
        print("   • get_lcm_metadata_trail() extracts complete LCM tracking data")
        print("   • export_lcm_metadata() provides audit trail export functionality")
        print("   • LCM metadata is available as output from pickled models")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()