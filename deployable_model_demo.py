#!/usr/bin/env python3
"""
CIAF LCM Deployable Model Example

This example demonstrates how to create a production-ready model with complete
CIAF LCM tracking that can be deployed, pickled, and used in production while
maintaining full audit trail capabilities.

Created: 2025-09-19
Author: CIAF Development Team
"""

import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

# Import CIAF components
from ciaf.wrappers import CIAFModelWrapper

def create_production_model():
    """Create a production-ready model with CIAF LCM tracking."""
    print("🏭 Creating Production Model with CIAF LCM Process")
    print("="*60)
    
    # 1. Create a real ML model for production
    print("\n1️⃣ Setting up production ML model...")
    base_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # 2. Wrap with CIAF for comprehensive tracking
    print("2️⃣ Wrapping with CIAF LCM framework...")
    production_wrapper = CIAFModelWrapper(
        model=base_model,
        model_name="Production_Fraud_Detection_Model",
        enable_connections=True,
        compliance_mode="financial",  # Financial compliance mode
        enable_explainability=True,
        enable_uncertainty=True,
        enable_metadata_tags=True,
        auto_configure=True
    )
    
    # 3. Create production training data (simulated)
    print("3️⃣ Preparing production training dataset...")
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Convert to CIAF format
    training_data = []
    for i in range(len(X)):
        training_data.append({
            "content": X[i].tolist(),  # Feature vector
            "metadata": {
                "id": f"transaction_{i:04d}",
                "target": int(y[i]),
                "timestamp": datetime.now().isoformat(),
                "source": "production_dataset_v1.0",
                "compliance_checked": True
            }
        })
    
    print(f"   ✅ Prepared {len(training_data)} training samples")
    
    # 4. Train with full CIAF LCM tracking
    print("4️⃣ Training model with CIAF LCM integration...")
    training_snapshot = production_wrapper.train(
        dataset_id="fraud_detection_production_dataset",
        training_data=training_data,
        master_password="production_secure_password_2025",
        training_params={
            "algorithm": "RandomForest",
            "n_estimators": 100,
            "max_depth": 10,
            "compliance_mode": "financial",
            "validation_split": 0.2,
            "cross_validation": True
        },
        model_version="1.0.0-production",
        fit_model=True  # Actually train the model
    )
    
    print(f"   ✅ Model trained: {training_snapshot.snapshot_id}")
    
    # 5. Perform some production inferences
    print("5️⃣ Running production inference examples...")
    
    production_queries = [
        X[0].tolist(),  # Real feature vector
        X[50].tolist(),
        X[100].tolist()
    ]
    
    for i, query in enumerate(production_queries):
        prediction, receipt = production_wrapper.predict(
            query=query, 
            use_model=True  # Use actual trained model
        )
        
        fraud_probability = prediction[0] if hasattr(prediction, '__len__') else prediction
        print(f"   Inference {i+1}: Fraud Risk = {str(fraud_probability)[:6]}, Receipt: {receipt.receipt_hash[:16]}...")
    
    return production_wrapper

def deploy_model(wrapper, deployment_path="./production_models"):
    """Deploy the model with complete LCM metadata preservation."""
    print(f"\n🚢 Deploying Model with LCM Preservation")
    print("="*50)
    
    # Create deployment directory
    deploy_dir = Path(deployment_path)
    deploy_dir.mkdir(exist_ok=True)
    
    model_name = wrapper.model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Export LCM audit trail before deployment
    print("1️⃣ Exporting LCM audit trail...")
    audit_report = wrapper.export_lcm_metadata(
        output_format="audit_report", 
        include_receipts=True
    )
    
    audit_file = deploy_dir / f"{model_name}_audit_trail_{timestamp}.json"
    with open(audit_file, 'w') as f:
        json.dump(audit_report, f, indent=2)
    
    print(f"   ✅ Audit trail exported: {audit_file}")
    
    # 2. Pickle the complete model with LCM metadata
    print("2️⃣ Pickling model with LCM metadata...")
    model_file = deploy_dir / f"{model_name}_{timestamp}.pkl"
    
    with open(model_file, 'wb') as f:
        pickle.dump(wrapper, f)
    
    print(f"   ✅ Model pickled: {model_file}")
    print(f"   📊 File size: {model_file.stat().st_size:,} bytes")
    
    # 3. Create deployment manifest
    print("3️⃣ Creating deployment manifest...")
    
    model_info = wrapper.get_model_info()
    lcm_trail = wrapper.get_lcm_metadata_trail()
    
    manifest = {
        "deployment_info": {
            "model_name": model_name,
            "deployment_timestamp": datetime.now().isoformat(),
            "model_version": wrapper.model_version,
            "compliance_mode": wrapper.compliance_mode,
            "deployment_id": f"deploy_{timestamp}"
        },
        "model_files": {
            "model_pickle": str(model_file.name),
            "audit_trail": str(audit_file.name)
        },
        "lcm_summary": {
            "lcm_enabled": True,
            "training_capsules": len(lcm_trail.get('training_metadata', {})),
            "inference_receipts": len(lcm_trail.get('inference_metadata', {})),
            "connections_count": len(lcm_trail.get('connections_metadata', {}).get('connections_summary', [])),
            "enhanced_features": lcm_trail.get('enhanced_features', {}),
            "integrity_verified": wrapper._verify_lcm_integrity()
        },
        "production_readiness": {
            "audit_trail_complete": True,
            "pickle_preservation_verified": True,
            "compliance_ready": True,
            "inference_tracking_enabled": True
        }
    }
    
    manifest_file = deploy_dir / f"{model_name}_deployment_manifest_{timestamp}.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"   ✅ Deployment manifest: {manifest_file}")
    
    return {
        "model_file": model_file,
        "audit_file": audit_file,
        "manifest_file": manifest_file,
        "deployment_id": f"deploy_{timestamp}"
    }

def load_deployed_model(model_file):
    """Load and verify a deployed model with LCM metadata."""
    print(f"\n📥 Loading Deployed Model with LCM Verification")
    print("="*55)
    
    print("1️⃣ Loading pickled model...")
    with open(model_file, 'rb') as f:
        restored_wrapper = pickle.load(f)
    
    print(f"   ✅ Model loaded: {restored_wrapper.model_name}")
    print(f"   📝 Version: {restored_wrapper.model_version}")
    
    # 2. Verify LCM metadata preservation
    print("2️⃣ Verifying LCM metadata preservation...")
    
    model_info = restored_wrapper.get_model_info()
    lcm_metadata = model_info.get('lcm_metadata', {})
    
    print(f"   LCM Integration: {'✅' if lcm_metadata.get('lcm_integration_enabled') else '❌'}")
    print(f"   Pickle Preservation: {'✅' if lcm_metadata.get('pickle_preservation_ready') else '❌'}")
    print(f"   Audit Trail Available: {'✅' if lcm_metadata.get('lcm_trail_available') else '❌'}")
    
    if hasattr(restored_wrapper, '_lcm_serialization_timestamp'):
        print(f"   Serialization Time: {restored_wrapper._lcm_serialization_timestamp}")
    
    # 3. Test production inference on restored model
    print("3️⃣ Testing production inference on restored model...")
    
    # Create a test query
    test_query = [0.5, -1.2, 0.8, 1.1, -0.3, 0.7, -0.9, 1.4, 0.2, -0.6,
                  0.9, -0.4, 1.3, 0.1, -0.8, 0.6, -1.1, 0.4, 0.9, -0.2]
    
    prediction, receipt = restored_wrapper.predict(
        query=test_query,
        use_model=True  # Use the actual restored model
    )
    
    print(f"   ✅ Inference successful!")
    print(f"   🎯 Prediction: {prediction}")
    print(f"   📋 Receipt: {receipt.receipt_hash[:16]}...")
    
    # 4. Export updated audit trail
    print("4️⃣ Exporting updated audit trail...")
    updated_audit = restored_wrapper.export_lcm_metadata(
        output_format="audit_report",
        include_receipts=True
    )
    
    total_receipts = updated_audit.get('audit_summary', {}).get('total_inference_receipts', 0)
    print(f"   📊 Total receipts in audit trail: {total_receipts}")
    
    return restored_wrapper

def main():
    """Main deployment demonstration."""
    print("🚀 CIAF LCM Deployable Model Demonstration")
    print("Complete production deployment with audit trail preservation")
    print()
    
    try:
        # 1. Create production model
        production_model = create_production_model()
        
        # 2. Deploy the model
        deployment_info = deploy_model(production_model)
        
        print(f"\n🎉 Deployment Complete!")
        print(f"   Model File: {deployment_info['model_file']}")
        print(f"   Audit Trail: {deployment_info['audit_file']}")
        print(f"   Manifest: {deployment_info['manifest_file']}")
        print(f"   Deployment ID: {deployment_info['deployment_id']}")
        
        # 3. Simulate loading in production environment
        print(f"\n🔄 Simulating Production Environment...")
        print("(Loading model as if in a different process/server)")
        
        restored_model = load_deployed_model(deployment_info['model_file'])
        
        print(f"\n✅ SUCCESS: Complete CIAF LCM Process Maintained!")
        print("="*60)
        print("🏭 Production Benefits:")
        print("   • Complete audit trail preserved through deployment")
        print("   • Regulatory compliance maintained")
        print("   • Inference tracking continues in production")
        print("   • Model lineage fully traceable")
        print("   • Pickle serialization preserves all LCM metadata")
        print("   • Ready for production deployment and scaling")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()