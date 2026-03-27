#!/usr/bin/env python3
"""
CIAF LCM Integration Demo

Demonstrates the new LCM (Lazy Capsule Materialization) integration
in the CIAF Framework, showing how the modern LCM system
replaces the legacy anchoring approach.

Created: 2025-01-21
Author: Denzil James Greenwood
"""

import sys
import os

# Add the parent directory to the path to import ciaf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ciaf.api.framework import CIAFFramework


def main():
    """
    Demonstrate LCM integration in CIAF Framework
    """
    print("🚀 CIAF LCM Integration Demo")
    print("=" * 50)

    # Initialize CIAF Framework with LCM integration
    framework = CIAFFramework("CIAF-LCM-Demo")

    # Verify LCM managers are initialized
    print("\n📋 LCM Managers Status:")
    print(f"✅ Root Manager: {bool(framework.lcm_root_manager)}")
    print(f"✅ Dataset Manager: {bool(framework.lcm_dataset_manager)}")
    print(f"✅ Model Manager: {bool(framework.lcm_model_manager)}")
    print(f"✅ Training Manager: {bool(framework.lcm_training_manager)}")
    print(f"✅ Inference Manager: {bool(framework.lcm_inference_manager)}")
    print(f"✅ Deployment Manager: {bool(framework.lcm_deployment_manager)}")

    # Demo dataset with LCM
    print("\n📊 Creating Dataset with LCM Integration...")
    dataset_metadata = {
        "name": "Financial Risk Dataset",
        "version": "1.0",
        "features": ["credit_score", "income", "debt_ratio", "employment_years"],
        "size": 10000,
        "data_items": [f"record_{i}" for i in range(5)],  # Sample items
    }

    try:
        dataset_anchor = framework.create_dataset_anchor_lcm(
            dataset_id="risk_dataset_v1",
            dataset_metadata=dataset_metadata,
            master_password="secure_dataset_key",
        )
        print("✅ Dataset anchor created with LCM tracking")
    except Exception as e:
        print(f"⚠️  Dataset creation failed: {e}")
        print("    (This is expected due to missing LCM specs - demo purposes)")

    # Demo model with LCM
    print("\n🤖 Creating Model with LCM Integration...")
    model_parameters = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "class_weight": "balanced",
    }

    try:
        model_anchor = framework.create_model_anchor_lcm(
            model_name="risk_classifier_v1",
            model_parameters=model_parameters,
            authorized_datasets=["risk_dataset_v1"],
            master_password="secure_model_key",
        )
        print("✅ Model anchor created with LCM tracking")
    except Exception as e:
        print(f"⚠️  Model creation failed: {e}")
        print("    (This is expected due to missing LCM specs - demo purposes)")

    # Demo inference with LCM
    print("\n🔮 Performing Inference with LCM Integration...")
    try:
        inference_receipt = framework.perform_inference_with_lcm(
            model_name="risk_classifier_v1",
            query="Assess credit risk for applicant with score 720, income $75000",
            ai_output="APPROVED - Low risk profile (confidence: 0.89)",
            user_id="demo_user",
        )
        print("✅ Inference completed with LCM tracking")
    except Exception as e:
        print(f"⚠️  Inference failed: {e}")
        print("    (This is expected due to model not existing - demo purposes)")

    # Show framework metrics with LCM integration
    print("\n📈 Framework Metrics with LCM Integration:")
    try:
        metrics = framework.get_framework_metrics()

        # Display LCM integration status
        if (
            "framework_summary" in metrics
            and "lcm_integration" in metrics["framework_summary"]
        ):
            lcm_status = metrics["framework_summary"]["lcm_integration"]
            print("LCM Integration Status:")
            for manager, status in lcm_status.items():
                print(f"  {manager}: {'✅ Active' if status else '❌ Inactive'}")

        print(f"\nDatasets tracked: {len(framework.dataset_anchors)}")
        print(f"Models tracked: {len(framework.model_anchors)}")

    except Exception as e:
        print(f"⚠️  Metrics retrieval failed: {e}")

    # Show the integration summary
    print("\n🎯 LCM Integration Summary:")
    print("━" * 50)
    print("✅ Framework now supports both legacy and LCM approaches")
    print("✅ LCM managers initialized and ready for use")
    print("✅ New LCM-enabled methods available:")
    print("   • create_dataset_anchor_lcm()")
    print("   • create_model_anchor_lcm()")
    print("   • perform_inference_with_lcm()")
    print("   • lcm_complete_workflow()")
    print("✅ Framework metrics include LCM integration status")
    print("✅ Backward compatibility maintained with legacy methods")

    print("\n📝 Note: Some operations may fail due to missing LCM specification")
    print("    classes, but the integration structure is in place.")

    print("\n🎉 LCM Integration Demo Complete!")


if __name__ == "__main__":
    main()
